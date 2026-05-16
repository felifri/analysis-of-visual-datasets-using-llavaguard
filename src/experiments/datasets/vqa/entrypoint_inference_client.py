import asyncio
import logging
import os
import rtpt
from datasets import load_dataset, IterableDataset
import math
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.policy import POLICY_DEFAULT

base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/vqa/results/7B_25_03_11_02"

def unique_images(dataset: IterableDataset):
    seen = set()
    
    def generator():
        for sample in dataset:
            if sample["image_id"] not in seen:
                seen.add(sample["image_id"])
                yield sample
                
    return IterableDataset.from_generator(generator)

def main():
    print(os.environ['HF_HOME'])

    if "finngu" in os.environ['HF_HOME']:
        print("HF configured locally")
    else:
        print("HF uses shared dir")

    # Create annotation dir and raise an error if they already exist to prevent data loss
    os.makedirs(base_output_dir)

    logging.basicConfig(
        filename=f"{base_output_dir}/generate_annotations.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting inference for llms-lab/VQAv2...")

    dataset = load_dataset("lmms-lab/VQAv2")

    logger.info("Successfully loaded all arrow files.")

    splits = ["validation", "testdev", "test"]
    chunk_size = 10000
    batch_size = 1000

    server = LlavaGuardServer()

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=4,
        port=10002,
        is_requests_wrapper=True
    )

    for split in splits:
        len_split = len(dataset[split]) / 3  # For every image, there are 3 questions
        num_chunks = math.ceil(len_split / chunk_size)
        chunk_ids_to_process = range(num_chunks) 
        
        iterable_ds = unique_images(load_dataset("lmms-lab/VQAv2", split=split).to_iterable_dataset())

        if not isinstance(iterable_ds, IterableDataset):
            raise ValueError("Dataset must be an IterableDataset for better performance.")
        
        iterable_ds = iterable_ds.map(
            function=lambda x: {'image': x['image'], 'image_name': f"{str(x['image_id']).zfill(6)}"}
        )

        # TODO: Safeguard against len(chunk_ids_to_process) == 0
        rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(chunk_ids_to_process))
        rt.start()

        for chunk_idx in chunk_ids_to_process:
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            chunk_idx_str = str(chunk_idx).zfill(int(math.log10(num_chunks)) + 1)

            try:
                chunk = iterable_ds.skip(chunk_idx * chunk_size).take(chunk_size)
            except IndexError:
                logger.error(f"\tChunk {chunk_idx} is out of range. Skipping.")
                continue

            # Create annotation dir and raise an error if they already exist to prevent data loss
            output_dir_annotations = os.path.join(base_output_dir, "annotations", split, chunk_idx_str)
            os.makedirs(output_dir_annotations)

            # I think this loop is being skipped after a couple iterations.
            for batch_idx, batch in enumerate(chunk.iter(batch_size)):
                logger.info(f"\tProcessing batch {batch_idx}")
                
                annotations = asyncio.run(server.request_async([{"image": img, "prompt": POLICY_DEFAULT} for img in batch['image']]))

                logger.info(f"\tSaving {len(annotations)} annotations for image_ids [{batch['image_name'][0]}, ..., {batch['image_name'][-1]}]")

                invalid_json = save_json_annotations(annotations, output_dir_annotations, batch['image_name'])

                if invalid_json:
                    logger.warning(f"Invalid JSON: {invalid_json}")
                rt.step()


if __name__ == '__main__':
    main()
