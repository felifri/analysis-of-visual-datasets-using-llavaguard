import asyncio
import json
import os
import rtpt
import time
import torch
import pandas as pd
from llavaguard_on_sglang.sglang_gpt_server import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.file_utils import get_file_paths
from util.policy import POLICY_DEFAULT
from util.download_images import download_images

# image_dir = "/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/data"
# image_dir = "/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/data/000004"
image_dir = "/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/test_10000/data_336"
output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/perf_benchmark_sglang/results_7B_laion2B-en_test_10000_336_25_03_13_02"

def download_test_images():
    image_urls = [
        "https://fastly.picsum.photos/id/662/1300/876.jpg?hmac=MtaSJNSKD6c3MIvW5T98S_KrE8bRbXKUpMSCJsMilW0",
        "https://fastly.picsum.photos/id/293/1300/876.jpg?hmac=Hq-z2-F6TnxA1sGIwvNzYeshjRBnCZ8eiSbSEU59Vwo",
        "https://fastly.picsum.photos/id/876/1300/876.jpg?hmac=WpaYlQYR68X8Z6asB1VAH3M4ji79FCWfZxi33nQ052g",
        "https://fastly.picsum.photos/id/938/1300/876.jpg?hmac=e2sNTtrCaKdtb1PWLPmcfx8NkzQIOjqAxZ3CiwfdgFg",
        "https://fastly.picsum.photos/id/157/1300/876.jpg?hmac=D-aRGmHtntB6uJ-X77dt5zFrJSL8l2mIA9drErCmdU8",
        "https://fastly.picsum.photos/id/48/1300/876.jpg?hmac=8p2dbt69Y6lc3202Zn13q6TOBu5OsyJAlzZi9KKur6w",
        "https://fastly.picsum.photos/id/127/1300/876.jpg?hmac=vfhFxKFvzPxqYnbX_ve84wDWp5uHZ_oV0qAluZETIm8",
        "https://fastly.picsum.photos/id/1050/1300/876.jpg?hmac=IzFCsxh0rOusbXDWzEhCpbud2rst36RBmI6S_5Hl4j8",
        "https://fastly.picsum.photos/id/1062/1300/876.jpg?hmac=9PTif3A5Ds6k2_Asm6NIGTckEfxr9YmDqAanmNaPzUc",
        "https://fastly.picsum.photos/id/605/1300/876.jpg?hmac=z9yYR6AJk05nfUp7PWrE8iZmiIzSQQcH76bNhKCQIz8",
    ]

    filename_handler = lambda url: url.split('/id/')[1].split('/')[0] + '.jpg'

    return download_images(
        image_urls=image_urls, 
        download_folder="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/perf_benchmark_sglang/test_images",
        filename_handler=filename_handler
    )

def main():
    if os.path.exists(output_dir):
        raise Exception("The output directory already exists! Change or delete it to avoid losing data.")

    server = LlavaGuardServer()

    # image_paths = download_test_images()
    image_paths, image_names = get_file_paths(image_dir, file_extension='.jpg')
    inputs = [
        {
            'image': path,
            'image_name': name,
            'prompt': POLICY_DEFAULT,
        } for path, name in zip(image_paths, image_names)
    ]

    server.setUpClass(
        # model="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV",
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        port=10003
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
        # This experiment tests an early stop, generating only the category and safety rating.
        annotations_early_stop = asyncio.run(server.request_async(
            inputs=[{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]],
            args={
                'max_tokens': 24,
            }
        ))
        toc_inference[idx_batch].record()

        print(f"Saving annotations of batch {idx_batch + 1}.")
        tic_save_annotation[idx_batch].record()

        # Convert the early stop annotations to valid JSON strings
        annotations = []

        for early_stop_annotation in annotations_early_stop:
            annotation = {}

            if "Safe" in early_stop_annotation:
                annotation['rating'] = "Safe"
            elif "Unsafe" in early_stop_annotation:
                annotation['rating'] = "Unsafe"
            else:
                annotations.append(early_stop_annotation)
                continue

            if "O1: Hate, Humiliation, Harassment" in early_stop_annotation:
                annotation['category'] = "O1: Hate, Humiliation, Harassment"
            elif "O2: Violence, Harm, or Cruelty" in early_stop_annotation:
                annotation['category'] = "O2: Violence, Harm, or Cruelty"
            elif "O3: Sexual Content" in early_stop_annotation:
                annotation['category'] = "O3: Sexual Content"
            elif "O4: Nudity Content" in early_stop_annotation:
                annotation['category'] = "O4: Nudity Content"
            elif "O5: Criminal Planning" in early_stop_annotation:
                annotation['category'] = "O5: Criminal Planning"
            elif "O6: Weapons or Substance Abuse" in early_stop_annotation:
                annotation['category'] = "O6: Weapons or Substance Abuse"
            elif "O7: Self-Harm" in early_stop_annotation:
                annotation['category'] = "O7: Self-Harm"
            elif "O8: Animal Cruelty" in early_stop_annotation:
                annotation['category'] = "O8: Animal Cruelty"
            elif "O9: Disasters or Emergencies" in early_stop_annotation:
                annotation['category'] = "O9: Disasters or Emergencies"
            elif "NA: None applying" in early_stop_annotation:
                annotation['category'] = "NA: None applying"
            else:
                annotations.append(early_stop_annotation)
                continue

            annotations.append(json.dumps(annotation))

        invalid_json = save_json_annotations(annotations, f"{output_dir}/image_annotations/batch_{idx_batch}", [input_['image_name'] for input_ in inputs[i:i+batch_size]])
        toc_save_annotation[idx_batch].record()

        results['invalid_json'] = [*results['invalid_json'], invalid_json]

        toc = time.perf_counter()
        results['total time [s]'] = [*results['total time [s]'], toc-tic]
        rt.step()

    server.tearDownClass()

    torch.cuda.synchronize()
    results['inference time [s]'] = [start.elapsed_time(end) / 1000 for start, end in zip(tic_inference, toc_inference)]
    results['save annotations time [s]'] = [start.elapsed_time(end) / 1000 for start, end in zip(tic_save_annotation, toc_save_annotation)]

    df = pd.DataFrame(results)
    print(df)

    results_path = f"{output_dir}/results.csv"
    df.to_csv(results_path, encoding='utf-8', index=False, header=True)

if __name__ == '__main__':
    main()
