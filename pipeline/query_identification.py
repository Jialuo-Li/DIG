import sys
import os
import argparse
import json
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from json_repair import repair_json
from tqdm.asyncio import tqdm
from utils import API_URL, API_KEY, MODEL_NAME, QUERY_IDENTIFICATION_PROMPT, VideoDataset

MAX_RETRIES = 5

def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            decoded = json.loads(repair_json(text))
            if isinstance(decoded, list):
                for item in decoded:
                    if isinstance(item, dict) and 'isGlobal' in item:
                        return item
            return decoded
        except Exception:
            return None

async def classify_query_sem(semaphore, client, question: str) -> Optional[str]:
    async with semaphore:
        prompt = QUERY_IDENTIFICATION_PROMPT.replace("<<<question>>>", question)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                )
                text = response.choices[0].message.content
                result = parse_json_response(text)

                if result and 'isGlobal' in result:
                    val = result['isGlobal']
                    is_global = False
                    if isinstance(val, str):
                        if val.lower() == 'true': is_global = True
                    elif isinstance(val, bool):
                        is_global = val
                    return 'global' if is_global else 'local'
            except Exception as e:
                print(f"[Warn] Response failed (Attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
        return None

async def process_item(item, semaphore, client):
    query_type = await classify_query_sem(semaphore, client, item['question'])
    item['query_type'] = query_type
    return item

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['mlvu', 'longvideobench', 'videomme'])
    parser.add_argument('--output_file', type=str, default='query_classification_results.json')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=20)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset.upper()}...")
    dataset = VideoDataset(args.data_file, args.video_dir, args.dataset)
    full_data = [dataset[i] for i in range(len(dataset))]
    output_path = Path(args.output_file)

    completed = []
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            completed = json.load(f)
        print(f"Loaded {len(completed)} existing results.")

    processed_keys = {(item.get('videoid'), item.get('question')) for item in completed}
    todo = [item for item in full_data if (item.get('videoid'), item.get('question')) not in processed_keys]

    print(f"Total: {len(full_data)} | Done: {len(processed_keys)} | Todo: {len(todo)}")
    if not todo:
        print("All items processed.")
        return

    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0)) as http_client:
        client = AsyncOpenAI(base_url=API_URL, api_key=API_KEY, http_client=http_client)
        semaphore = asyncio.Semaphore(args.concurrency)
        
        print(f"Starting processing with concurrency limit: {args.concurrency}")
        results = completed
        
        tasks = [process_item(item, semaphore, client) for item in todo]
        
        for i, future in tqdm(enumerate(asyncio.as_completed(tasks)), total=len(tasks), desc="Processing"):
            processed_item = await future
            results.append(processed_item)
            
            if (i + 1) % args.batch_size == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {output_path}")
    print(f"Total processed: {len(results)}")

if __name__ == "__main__":
    asyncio.run(main_async())