import asyncio
import os
import warnings
import rtpt
import torch
from datasets import load_dataset, IterableDataset, Image
import math
import pandas as pd
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.policy import POLICY_DEFAULT

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/imagenet/results/7B_25_01_31_01"

# warnings.simplefilter("ignore", UserWarning)  # Suppress EXIF warnings globally

def main():
    print(os.environ['HF_HOME'])

    if "finngu" in os.environ['HF_HOME']:
        print("HF configured locally")
    else:
        print("HF uses shared dir")
        print("Aborting.")
        exit()

    dataset = load_dataset("ILSVRC/imagenet-1k")

    print("Successfully loaded all arrow files.")

    splits = ["train", "validation", "test"]
    chunk_size = 10000
    batch_size = 1000

    server = LlavaGuardServer()

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=4,
        port=10010,
        is_requests_wrapper=True
    )

    for split in splits:
        len_split = len(dataset[split])
        num_chunks = math.ceil(len_split / chunk_size)
        chunk_ids_to_process = range(num_chunks // 2 + 1, num_chunks // 4 * 3) # range(num_chunks // 4 * 3, num_chunks) 
        # iterable_ds = load_dataset("ILSVRC/imagenet-1k", split=split).to_iterable_dataset()
        # print(iterable_ds.features['image'])
        # iterable_ds = iterable_ds.cast_column("image", Image(decode=False))

        # if not isinstance(iterable_ds, IterableDataset):
        #     raise ValueError("Dataset must be an IterableDataset for better performance.")

        # iterable_ds = iterable_ds.map(
        #     function=lambda x, idx: {'image': x['image'], 'image_name': f"{str(idx).zfill(int(math.log10(len_split)) + 1)}.jpg"}, 
        #     with_indices=True,
        #     remove_columns=['label']
        # )

        # Stats dir may already be present, but that is acceptable, because we name the files themselves
        output_dir_stats = os.path.join(base_output_dir, "stats", split)
        os.makedirs(output_dir_stats, exist_ok=True)

        # TODO: Safeguard against len(chunk_ids_to_process) == 0
        rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(chunk_ids_to_process))
        rt.start()

        for chunk_idx in chunk_ids_to_process:
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            len_chunk = min(chunk_size, len_split - chunk_idx * chunk_size)
            chunk_idx_str = str(chunk_idx).zfill(int(math.log10(num_chunks)) + 1)

            # try:
            #     print("Taking chunk...")
            #     chunk = iterable_ds.skip(chunk_idx * chunk_size).take(chunk_size)
            #     print("Took chunk successfully.")
            # except IndexError:
            #     print(f"\tChunk {chunk_idx} is out of range. Skipping.")
            #     continue

            start_idx_chunk = chunk_idx * chunk_size
            chunk = dataset[split].select(range(start_idx_chunk, start_idx_chunk + len_chunk))

            # for example in chunk.skip(3333).take(5):
            #     print(example['image'])

            # Create annotation dir and raise an error if they already exist to prevent data loss
            output_dir_annotations = os.path.join(base_output_dir, "annotations", split, chunk_idx_str)
            os.makedirs(output_dir_annotations)

            results = {
                'chunk_size': len_chunk,
                'total time [s]': 0,
                'invalid_json': [],
            }

            # https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)

            tic.record()

            # Processing chunk 65/129
            # /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang/lib/python3.10/site-packages/PIL/TiffImagePlugin.py:900: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.
            #   warnings.warn(str(msg))
            # /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang/lib/python3.10/site-packages/PIL/TiffImagePlugin.py:900: UserWarning: Truncated File Read
            #   warnings.warn(str(msg))

            # def decode_images(batch):
            #     batch["image"] = [decode_image(raw_data) for raw_data in batch["image"]]
            #     return batch

            # chunk = chunk.cast_column("image", Image(decode=False))
            # chunk = chunk.with_transform(decode_images)

            # for idx in range(10000):
            #     print(f"{idx}: {chunk[idx]['image']}")


            num_batches = math.ceil(len(chunk) / batch_size)
            # print(f"Splitting chunk into batches of size {batch_size}.")
            # for batch_idx, batch in enumerate(chunk.iter(batch_size)):
            for batch_idx in range(num_batches):
                print(f"\tProcessing batch {batch_idx}")
                start_idx_batch = batch_idx * batch_size
                batch = chunk.select(range(start_idx_batch, start_idx_batch + batch_size))

                batch = batch.map(
                    function=lambda x, idx: {'image': x['image'], 'image_name': str(start_idx_chunk + start_idx_batch + idx).zfill(int(math.log10(len_split)) + 1)}, 
                    with_indices=True,
                    remove_columns=['label']
                )
                
                annotations = asyncio.run(server.request_async([{"image": img, "prompt": POLICY_DEFAULT} for img in batch['image']]))
                invalid_json = save_json_annotations(annotations, output_dir_annotations, batch['image_name'])

                results['invalid_json'] = [*results['invalid_json'], invalid_json]
                rt.step()
            
            toc.record()

            try:
                torch.cuda.synchronize()
                results['total time [s]'] = tic.elapsed_time(toc) / 1000

                stats_path = os.path.join(output_dir_stats, f"{chunk_idx_str}.csv")
                pd.DataFrame(results).to_csv(stats_path, encoding='utf-8', index=False, header=True)
            except RuntimeError:
                print("Error during stat calculation. Skipping stats file.")


if __name__ == '__main__':
    main()
