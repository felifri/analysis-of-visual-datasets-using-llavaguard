import asyncio
import logging
import os
import rtpt
from datasets import load_dataset, IterableDataset, Image
import math
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.policy import POLICY_DEFAULT

base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/cifar-100/results/7B_25_03_26_01"

def main():
    # Try to create the output dir. If it already exists, throw an error to avoid overwriting data.
    os.makedirs(base_output_dir)

    logging.basicConfig(
        filename=f"{base_output_dir}/generate_annotations.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info("Starting image annotation: CIFAR-100...")

    logger.info(os.environ['HF_HOME'])

    if "finngu" in os.environ['HF_HOME']:
        logger.info("HF configured locally")
    else:
        logger.info("HF uses shared dir")

    dataset = load_dataset("uoft-cs/cifar100")

    logger.info("Successfully loaded all arrow files.")

    splits = ["train", "test"]
    chunk_size = 10000
    batch_size = 1000

    server = LlavaGuardServer()

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=1,
        port=10001,
        is_requests_wrapper=True
    )

    logger.info("LlavaGuard server ready.")

    try:
        for split in splits:
            logger.info(f"Generating annotations for split '{split}.'")

            len_split = len(dataset[split])
            num_chunks = math.ceil(len_split / chunk_size)
            chunk_ids_to_process = range(num_chunks) 
            
            iterable_ds = load_dataset("uoft-cs/cifar100", split=split).to_iterable_dataset()

            if not isinstance(iterable_ds, IterableDataset):
                raise ValueError("Dataset must be an IterableDataset for better performance.")

            iterable_ds = iterable_ds.map(
                function=lambda x, idx: {'image': x['img'], 'image_name': f"{str(idx).zfill(int(math.log10(len_split)) + 1)}"}, 
                with_indices=True
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
                    logger.warning(f"\tChunk {chunk_idx} is out of range. Skipping.")
                    continue

                # Create annotation dir and raise an error if they already exist to prevent data loss
                output_dir_annotations = os.path.join(base_output_dir, "annotations", split, chunk_idx_str)
                os.makedirs(output_dir_annotations)
                
                for batch_idx, batch in enumerate(chunk.iter(batch_size)):                
                    annotations = asyncio.run(server.request_async([{"image": img, "prompt": POLICY_DEFAULT} for img in batch['image']]))
                    invalid_json = save_json_annotations(annotations, output_dir_annotations, batch['image_name'])

                    if invalid_json:
                        logger.warning(f"Invalid JSON: {invalid_json}")
                    rt.step()
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        logger.info("Inference done, shutting down the server.")
        server.tearDownClass()


if __name__ == '__main__':
    main()
