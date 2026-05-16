import asyncio
import os
import rtpt
from datasets import load_dataset, IterableDataset, Image
import math
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.policy import POLICY_DEFAULT

base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/celeba/results/7B_25_03_06_01"

def main():
    print(os.environ['HF_HOME'])

    if "finngu" in os.environ['HF_HOME']:
        print("HF configured locally")
    else:
        print("HF uses shared dir")

    dataset = load_dataset("flwrlabs/celeba")

    print("Successfully loaded all arrow files.")

    splits = ["train", "valid", "test"]
    chunk_size = 10000
    batch_size = 1000

    server = LlavaGuardServer()

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=2,
        port=10000,
        is_requests_wrapper=True
    )

    for split in splits:
        len_split = len(dataset[split])
        num_chunks = math.ceil(len_split / chunk_size)
        chunk_ids_to_process = range(num_chunks) 
        
        iterable_ds = load_dataset("flwrlabs/celeba", split=split).to_iterable_dataset()
        # print(iterable_ds.features['image'])
        # iterable_ds = iterable_ds.cast_column("image", Image(decode=False))

        if not isinstance(iterable_ds, IterableDataset):
            raise ValueError("Dataset must be an IterableDataset for better performance.")

        iterable_ds = iterable_ds.map(
            function=lambda x, idx: {'image': x['image'], 'image_name': f"{str(idx).zfill(int(math.log10(len_split)) + 1)}"}, 
            with_indices=True
        )

        # TODO: Safeguard against len(chunk_ids_to_process) == 0
        rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(chunk_ids_to_process))
        rt.start()

        for chunk_idx in chunk_ids_to_process:
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            # len_chunk = min(chunk_size, len_split - chunk_idx * chunk_size)
            chunk_idx_str = str(chunk_idx).zfill(int(math.log10(num_chunks)) + 1)

            try:
                print("Taking chunk...")
                chunk = iterable_ds.skip(chunk_idx * chunk_size).take(chunk_size)
                print("Took chunk successfully.")
            except IndexError:
                print(f"\tChunk {chunk_idx} is out of range. Skipping.")
                continue

            # start_idx_chunk = chunk_idx * chunk_size
            # chunk = dataset[split].select(range(start_idx_chunk, start_idx_chunk + len_chunk))

            # Create annotation dir and raise an error if they already exist to prevent data loss
            output_dir_annotations = os.path.join(base_output_dir, "annotations", split, chunk_idx_str)
            os.makedirs(output_dir_annotations)

            # chunk = chunk.cast_column("image", Image(decode=False))
            # chunk = chunk.with_transform(decode_images)


            # num_batches = math.ceil(len(chunk) / batch_size)
            # print(f"Splitting chunk into batches of size {batch_size}.")
            for batch_idx, batch in enumerate(chunk.iter(batch_size)):
            # for batch_idx in range(num_batches):
                print(f"\tProcessing batch {batch_idx}")
                # start_idx_batch = batch_idx * batch_size
                # batch = chunk.select(range(start_idx_batch, start_idx_batch + batch_size))

                # batch = batch.map(
                #     function=lambda x, idx: {'image': x['image'], 'image_name': str(start_idx_chunk + start_idx_batch + idx).zfill(int(math.log10(len_split)) + 1)}, 
                #     with_indices=True,
                #     remove_columns=['label']
                # )
                
                annotations = asyncio.run(server.request_async([{"image": img, "prompt": POLICY_DEFAULT} for img in batch['image']]))
                invalid_json = save_json_annotations(annotations, output_dir_annotations, batch['image_name'])

                print(f"Invalid JSON: {invalid_json}")
                rt.step()


if __name__ == '__main__':
    main()
