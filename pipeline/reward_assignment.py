import sys
import argparse
import json
import base64
import asyncio
import os
import httpx
from pathlib import Path
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from tqdm.asyncio import tqdm
from decord import VideoReader, cpu
from PIL import Image
from openai import AsyncOpenAI
from json_repair import repair_json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import API_URL, API_KEY, MODEL_NAME, REWARD_ASSIGNMENT_PROMPT, VideoDataset

MAX_RETRIES = 5
IMAGE_FORMAT = "PNG"

def encode_image_base64(image_array):
    buffer = BytesIO()
    Image.fromarray(image_array).save(buffer, format=IMAGE_FORMAT)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_frames_task(video_path, frame_indices):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    avg_fps, max_frame = vr.get_avg_fps(), len(vr) - 1
    valid_indices = [min(idx, max_frame) for idx in frame_indices]
    
    frames_array = vr.get_batch(valid_indices).asnumpy()
    results = {
        idx: {'encoded_frame': encode_image_base64(frames_array[i]), 'timestamp': round(idx / avg_fps, 2)}
        for i, idx in enumerate(frame_indices)
    }
    return results, round(len(vr) / avg_fps, 2)

async def fetch_reward_sem(semaphore, client, question, frame_b64, duration, timestamp):
    async with semaphore:
        prompt = REWARD_ASSIGNMENT_PROMPT.replace("<<<question>>>", question)\
                                         .replace("<<<duration>>>", str(duration))\
                                         .replace("<<<timestamp>>>", str(timestamp))
        messages = [{
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}}, 
                {"type": "text", "text": prompt}
            ]
        }]
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7)
                content = response.choices[0].message.content
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = json.loads(repair_json(content))
                
                if isinstance(data, dict) and 'reward' in data:
                    return float(data['reward'])
                print(f"[Warn] T={timestamp}s failed (Attempt {attempt+1}): Invalid JSON structure")
            except Exception as e:
                print(f"[Warn] T={timestamp}s failed (Attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
        return 0.0

async def process_item(item, process_pool, semaphore, client):
    video_path, frame_indices = item['video_path'], item.get('r_frame_idx', [])
    loop = asyncio.get_running_loop()
    
    frames_map, duration = await loop.run_in_executor(process_pool, extract_frames_task, video_path, frame_indices)
    
    tasks = [
        fetch_reward_sem(semaphore, client, item['question'], frames_map[idx]['encoded_frame'], duration, frames_map[idx]['timestamp'])
        if idx in frames_map else asyncio.sleep(0, result=0.0)
        for idx in frame_indices
    ]
    
    item['reward'] = await asyncio.gather(*tasks)
    return item

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--dataset', required=True, choices=['mlvu', 'longvideobench', 'videomme'])
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=200)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}...")
    dataset = VideoDataset(args.data_file, args.video_dir, args.dataset)
    full_data = [dataset[i] for i in range(len(dataset))]
    output_path = Path(args.output_file)
    
    completed = []
    if output_path.exists():
        try:
            completed = json.loads(output_path.read_text())
            print(f"Loaded {len(completed)} existing results.")
        except: pass

    processed_keys = {(item.get('videoid'), item.get('question')) for item in completed}
    todo = [item for item in full_data if (item.get('videoid'), item.get('question')) not in processed_keys]

    print(f"Total: {len(full_data)} | Done: {len(processed_keys)} | Todo: {len(todo)}")
    if not todo: return

    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0)) as http_client:
        client = AsyncOpenAI(base_url=API_URL, api_key=API_KEY, http_client=http_client)
        semaphore = asyncio.Semaphore(args.concurrency)
        
        print(f"Starting with {args.num_workers} workers & {args.concurrency} concurrent reqs.")
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            results = completed
            total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
            
            for i in range(0, len(todo), args.batch_size):
                chunk = todo[i : i + args.batch_size]
                batch_idx = i // args.batch_size + 1
                print(f"Processing Batch {batch_idx}/{total_batches} ({len(chunk)} items)...")

                tasks = [process_item(item, pool, semaphore, client) for item in chunk]
                for future in tqdm(asyncio.as_completed(tasks), total=len(chunk), desc=f"Batch {batch_idx}"):
                    results.append(await future)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)

    print(f"Done. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main_async())