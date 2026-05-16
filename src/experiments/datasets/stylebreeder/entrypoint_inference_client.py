import asyncio
import os
import rtpt
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.file_utils import get_file_paths
from util.annotation_utils import save_json_annotations
from util.policy import POLICY_DEFAULT

base_image_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME/datasets/stylebreeder___stylebreeder/default/0.0.0/images"
base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/dataset_stylebreeder/results/7B_25_01_23_01"

def main():
    server = LlavaGuardServer()

    # Will not actually start a server, but instead forward the requests to an already running instance of sglang.router
    # This provides almost linear data parallelisation up to 4 GPUs
    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        port=10000,
        is_requests_wrapper=True
    )
    batch_size = 1000

    # Stats dir may already be present, but that is acceptable, because we name the files themselves
    output_dir_stats = os.path.join(base_output_dir, "stats")
    os.makedirs(output_dir_stats, exist_ok=True)

    # Assign chunks to each worker
    chunk_ids_to_process = range(0, 200)

    rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(chunk_ids_to_process))
    rt.start()

    for idx_chunk in chunk_ids_to_process:
        idx_chunk = str(idx_chunk).zfill(3)

        image_dir = os.path.join(base_image_dir, idx_chunk)
        output_dir_annotations = os.path.join(base_output_dir, "annotations", idx_chunk)

        # Create annotation dir and raise an error if they already exist to prevent data loss
        os.makedirs(output_dir_annotations)

        image_paths, image_names = get_file_paths(image_dir, file_extension='.jpeg')
        inputs = [
            {
                'image': path,
                'image_name': name,
                'prompt': POLICY_DEFAULT,
            } for path, name in zip(image_paths, image_names)
        ]
        
        print(f"Processing chunk {idx_chunk}/{len(chunk_ids_to_process)}: {len(inputs)} annotations will be generated in {len(inputs)//batch_size + 1} batches.")
        for i in range(0, len(inputs), batch_size):
            idx_batch = i // batch_size
            # print(f"Running batch {idx_batch + 1}/{len(inputs)//batch_size + 1}.")

            annotations = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))

            # print(f"Saving annotations of batch {idx_batch + 1}.")
            invalid_json = save_json_annotations(annotations, output_dir_annotations, [input_['image_name'] for input_ in inputs[i:i+batch_size]])

            if invalid_json:
                print(f"Invalid JSON annotations found in batch {idx_batch + 1}: {invalid_json}")
            
            rt.step()


if __name__ == '__main__':
    main()
