import asyncio
import json
import os
import rtpt
import time
import torch
import pandas as pd
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.file_utils import get_file_paths
from util.policy import POLICY_DEFAULT

image_dir = "/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/test_10000/data_336"
output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/perf_benchmark_sglang/results_7B_laion2B-en_test_10000_336_01_17_01"

def save_json_annotations(annotations: list[str], output_dir: str, file_names: list[str] = None):
    """
    Processes a list of JSON strings and saves each valid entry as a .json file.
    Invalid JSON strings are stored as .txt files.

    Parameters:
    - json_list (list): List of JSON strings.
    - output_dir (str): Directory to save the .json and .txt files.

    Returns:
    - List of file names where annotations are not valid JSON.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Ensure there is exactly one name per file
    if file_names:
        assert len(annotations) == len(file_names)
    else:
        file_names = [str(idx) for idx in range(len(annotations))]

    invalid_json = []

    for json_str, file_name in zip(annotations, file_names):
        try:
            # Attempt to parse the JSON string
            data = json.loads(json_str)
            
            # Generate a file name
            file_path = os.path.join(output_dir, f"{file_name}.json")
            
            # Write the valid JSON data to a file
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)
        except json.JSONDecodeError:
            # Handle invalid JSON: Save as .txt file
            invalid_json.append(file_name)

            # Generate a file name
            file_path = os.path.join(output_dir, f"{file_name}.txt")
            
            # Write the invalid JSON string to a .txt file
            with open(file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(json_str)

    return invalid_json

def main():
    if os.path.exists(output_dir):
        raise Exception("The output directory already exists! Change or delete it to avoid losing data.")

    server = LlavaGuardServer()

    image_paths, image_names = get_file_paths(image_dir, file_extension='.jpg')
    inputs = [
        {
            'image': path,
            'image_name': name,
            'prompt': POLICY_DEFAULT,
        } for path, name in zip(image_paths, image_names)
    ]

    # server.setUpClass(model="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV")
    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=2,
        is_requests_wrapper=True
    )
    batch_size = 1000

    results = {
        'batch_id': [],
        'batch_size': [],
        'inference time [s]': [],
        'save annotations time [s]': [],
        'total time [s]': [],
        'invalid_json': [],
    }

    # https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    tic_inference = [torch.cuda.Event(enable_timing=True) for _ in range(len(inputs)//batch_size + 1)]
    toc_inference = [torch.cuda.Event(enable_timing=True) for _ in range(len(inputs)//batch_size + 1)]
    tic_save_annotation = [torch.cuda.Event(enable_timing=True) for _ in range(len(inputs)//batch_size + 1)]
    toc_save_annotation = [torch.cuda.Event(enable_timing=True) for _ in range(len(inputs)//batch_size + 1)]

    rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(inputs)//batch_size + 1)
    rt.start()
    print(f"Total {len(inputs)} annotations will be generated in {len(inputs)//batch_size + 1} batches")
    for i in range(0, len(inputs), batch_size):
        tic = time.perf_counter()

        idx_batch = i // batch_size
        print(f"Running batch {idx_batch + 1}/{len(inputs)//batch_size + 1}")

        results['batch_id'] = [*results['batch_id'], idx_batch]
        results['batch_size'] = [*results['batch_size'], min(batch_size, len(inputs) - i)]

        tic_inference[idx_batch].record()
        annotations = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))
        toc_inference[idx_batch].record()

        print(f"Saving annotations of batch {idx_batch + 1}.")
        tic_save_annotation[idx_batch].record()
        invalid_json = save_json_annotations(annotations, f"{output_dir}/image_annotations/batch_{idx_batch}", [input_['image_name'] for input_ in inputs[i:i+batch_size]])
        toc_save_annotation[idx_batch].record()

        results['invalid_json'] = [*results['invalid_json'], invalid_json]

        toc = time.perf_counter()
        results['total time [s]'] = [*results['total time [s]'], toc-tic]
        rt.step()

    # server.tearDownClass()

    torch.cuda.synchronize()
    results['inference time [s]'] = [start.elapsed_time(end) / 1000 for start, end in zip(tic_inference, toc_inference)]
    results['save annotations time [s]'] = [start.elapsed_time(end) / 1000 for start, end in zip(tic_save_annotation, toc_save_annotation)]

    df = pd.DataFrame(results)
    print(df)

    results_path = f"{output_dir}/results.csv"
    df.to_csv(results_path, encoding='utf-8', index=False, header=True)

if __name__ == '__main__':
    main()
