import os
import sys
import json
import glob
import math
import random
import argparse
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from scipy.signal import find_peaks
from tqdm import tqdm
from accelerate import Accelerator
from transformers import Dinov2Model, AutoImageProcessor
from decord import VideoReader, cpu
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import VideoDataset

@torch.no_grad()
def extract_batch_features(model, processor, images, device):
    inputs = processor(images, return_tensors="pt").to(device)
    return model(**inputs).last_hidden_state

def get_r_frames(video_path, model, processor, samples_per_sec, device='cuda', infer_batch_size=64):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    video_len_sec = int(total_frames / fps)
    
    num_segments = math.ceil(video_len_sec / 60)
    target_frame_indices = np.linspace(0, total_frames - 1, video_len_sec * samples_per_sec, dtype=int)
    segment_boundaries = np.linspace(0, total_frames - 1, num_segments + 1, dtype=int)

    frame_chunks = []
    for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        mask = (target_frame_indices >= start) & (target_frame_indices <= end if i == len(segment_boundaries) - 2 else target_frame_indices < end)
        frame_chunks.append(target_frame_indices[mask])

    cuts, result_frames = [0], []

    for i, chunk_indices in enumerate(frame_chunks):
        if len(chunk_indices) == 0: continue

        video_data = vr.get_batch(chunk_indices).asnumpy()
        chunk_features_list = []
        
        for j in range(0, len(video_data), infer_batch_size):
            batch_pil = [Image.fromarray(img) for img in video_data[j : j + infer_batch_size]]
            chunk_features_list.append(extract_batch_features(model, processor, batch_pil, device).mean(dim=1))

        if not chunk_features_list: continue

        chunk_features = torch.cat(chunk_features_list, dim=0)
        
        if chunk_features.shape[0] > 1:
            diffs = 1 - F.cosine_similarity(chunk_features[:-1], chunk_features[1:], dim=1)
            peaks, _ = find_peaks(diffs.detach().cpu().numpy(), prominence=0.1)
            current_peaks = chunk_indices[peaks].tolist()
            temp_cuts = [cuts[-1]] + current_peaks
            cuts.extend(current_peaks)
        else:
            temp_cuts = [cuts[-1]]

        if i == len(frame_chunks) - 1:
            cuts.append(total_frames - 1)
            temp_cuts.append(total_frames - 1)

        mid_points = [(start + end) // 2 for start, end in zip(temp_cuts[:-1], temp_cuts[1:])]
        if mid_points:
            sample_frames_np = np.array(target_frame_indices)
            nearest_indices = [np.abs(sample_frames_np - m).argmin() for m in mid_points]
            result_frames.extend(sample_frames_np[nearest_indices].tolist())

    return {"r_frame_idx": result_frames, "boundaries": cuts}

def merge_results(output_file):
    merged_results = []
    temp_files = glob.glob(f"{output_file}.rank_*")
    
    for temp_file in temp_files:
        with open(temp_file, 'r') as f:
            merged_results.extend(json.load(f))
        os.remove(temp_file)

    seen, deduplicated_results = set(), []
    for entry in merged_results:
        key = (entry['video_path'], entry['question'])
        if key not in seen:
            seen.add(key)
            deduplicated_results.append(entry)

    with open(output_file, 'w') as f:
        json.dump(deduplicated_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mlvu', 'longvideobench', 'videomme'], required=True)
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--sample_per_sec', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--infer_batch_size', type=int, default=64)
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=60))
    
    accelerator = Accelerator(device_placement=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = VideoDataset(args.json_file, args.video_dir, dataset_type=args.dataset)
    device = torch.device(f"cuda:{accelerator.process_index % torch.cuda.device_count()}")
    data_loader = accelerator.prepare(DataLoader(dataset, shuffle=False))

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()

    start_time = time.time()
    results = []
    cache = {}
    for batch in tqdm(data_loader):
        video_path, question = batch["video_path"][0], batch["question"][0]
        videoid, query_type = batch["videoid"][0], batch["query_type"][0]

        if query_type == 'global':
            results.append({"video_path": video_path, "videoid": videoid, "question": question, "query_type": query_type})
            continue
        
        if videoid in cache:
            metadata = cache[videoid]
        else:
            metadata = get_r_frames(video_path, model, processor, args.sample_per_sec, device, args.infer_batch_size)
            cache[videoid] = metadata
        
        results.append({
            "video_path": video_path, "videoid": videoid, "question": question, "query_type": query_type,
            "r_frame_idx": metadata["r_frame_idx"], "boundaries": metadata["boundaries"]
        })

    with open(f"{args.output_file}.rank_{accelerator.process_index}", 'w') as f:
        json.dump(results, f, indent=2)

    if dist.is_initialized():
        dist.barrier()

    if accelerator.is_main_process:
        merge_results(args.output_file)
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()