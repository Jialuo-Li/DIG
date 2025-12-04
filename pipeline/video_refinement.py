import json
import argparse
import numpy as np
import decord
from collections import defaultdict

def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged

def normalize_scores(scores):
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val + 1e-8) * 100

def video_refinement(rewards, boundaries, wlen=2):
    scores = normalize_scores(np.array(rewards, dtype=float))
    non_zero_indices = np.where(scores > 0)[0]

    if len(non_zero_indices) == 0:
        return [[boundaries[0], boundaries[-1]]]
    
    if len(non_zero_indices) == 1:
        idx = non_zero_indices[0]
        return [[boundaries[max(0, idx - wlen)], boundaries[min(len(scores) - 1, idx + wlen)]]]

    active_scores = scores[scores > 0]
    
    s1 = active_scores.copy()
    while True:
        prev_zeros = np.sum(s1 == 0)
        s1 = np.maximum(s1 - np.mean(s1), 0)
        if np.sum(s1 == 0) == prev_zeros:
            break

    s2 = active_scores.copy()
    s2 = np.maximum(s2 - 0.5 * np.max(s2), 0) if len(s2) > 0 else np.array([])

    final_scores = s1 if np.sum(s1 > 0) <= np.sum(s2 > 0) else s2
    
    segments = []
    valid_indices = np.where(final_scores > 0)[0]
    for i in valid_indices:
        orig_idx = non_zero_indices[i]
        segments.append([
            boundaries[max(0, orig_idx - wlen)], 
            boundaries[min(len(scores) - 1, orig_idx + wlen)]
        ])

    return merge_intervals(segments) or [[boundaries[0], boundaries[-1]]]

def select_k_indices(intervals, k):
    if not intervals or k <= 0:
        return []

    total_length = sum(end - start + 1 for start, end in intervals)
    sample_points = np.linspace(0, total_length - 1, k, dtype=int)
    frame_indices = []

    for sample in sample_points:
        current_pos = sample
        for start, end in intervals:
            seg_len = end - start + 1
            if current_pos < seg_len:
                frame_indices.append(int(start + current_pos))
                break
            current_pos -= seg_len

    return frame_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='refined_frames.json')
    parser.add_argument('--k', type=int, default=8)
    args = parser.parse_args()

    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    indexed_results = defaultdict(dict)

    for item in data:
        videoid = item.get("videoid")
        question = item.get("question")

        if item.get('query_type') == 'local':
            segs = video_refinement(item['reward'], item['boundaries'], wlen=2)
            frames = select_k_indices(segs, args.k)
        else:
            vr = decord.VideoReader(item.get('video_path', ''), num_threads=2)
            frames = np.linspace(0, len(vr) - 1, args.k, dtype=int).tolist()

        indexed_results[videoid][question] = frames

    print(f"Saving to {args.output_file}...")
    with open(args.output_file, 'w', encoding="utf-8") as f:
        json.dump(indexed_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()