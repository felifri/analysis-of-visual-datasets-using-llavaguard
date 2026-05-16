import asyncio
import subprocess
import logging
import os
from PIL import UnidentifiedImageError
import rtpt
from datasets import load_dataset
import math
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import compress_annotations, save_json_annotations
from util.policy import POLICY_DEFAULT

base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/cc12m/results/7B_25_04_25_02"
output_dir_annotations_compressed = os.path.join(base_output_dir, "annotations_compressed")

def main():
    # Create the output dir. If it already exists, throw an error to avoid overwriting data.
    os.makedirs(base_output_dir)
    os.makedirs(output_dir_annotations_compressed)
    
    logging.basicConfig(
        filename=f"{base_output_dir}/generate_annotations.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info("Starting image annotation: CC12M...")

    server = LlavaGuardServer()
    chunk_size = 100000
    batch_size = 1000

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=4,
        port=10000,
        is_requests_wrapper=True
    )

    logger.info("LlavaGuard server ready.")

    try:
        corrupted_indices = []
        with open("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/cc12m/corrupted_image_indices.txt", "r") as f:
            corrupted_indices = [int(line.strip()) for line in f.readlines()]
            
        logger.info(f"Loading dataset and excluding {len(corrupted_indices)} invalid indices")
        
        # Using Dataset as well as IterableDataset results in storing it twice on disk.
        # If storage is low, the Dataset can be deleted once the IterableDataset is created.
        # rm -r /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME/datasets/pixparse___cc12m-wds
        # The IterableDataset is stored in the 'hub' folder and not affected by the above command.
        dataset = load_dataset("pixparse/cc12m-wds", split="train")
        dataset = dataset.select((i for i in range(len(dataset)) if i not in set(corrupted_indices)))

        logger.info("Successfully loaded dataset.")

        num_chunks = math.ceil(len(dataset) / chunk_size)
        chunk_ids_to_process = range(107, num_chunks)  # range(97, 102)

        # TODO: Safeguard against len(chunk_ids_to_process) == 0
        rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(chunk_ids_to_process))
        rt.start()

        for chunk_idx in chunk_ids_to_process:
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            chunk_idx_str = str(chunk_idx).zfill(int(math.log10(num_chunks)) + 1)

            try:
                chunk = dataset.select(range(chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, len(dataset))))
            except IndexError:
                logger.error(f"\tChunk {chunk_idx} is out of range. Skipping.")
                continue

            # Create annotation dir and raise an error if they already exist to prevent data loss
            output_dir_annotations = os.path.join(base_output_dir, "annotations", chunk_idx_str)
            os.makedirs(output_dir_annotations)

            num_batches = math.ceil(len(chunk) / batch_size)

            for batch_idx in range(num_batches):
                start_idx_batch = batch_idx * batch_size
                batch = chunk.select(range(start_idx_batch, min(start_idx_batch + batch_size, len(chunk))))

                # Filter out problematic images
                valid_images = []
                valid_image_names = []
                for img, img_name in zip(batch['jpg'], batch['__key__']):
                    try:
                        # Attempt to open the image to ensure it's valid
                        img.verify()
                        valid_images.append(img)
                        valid_image_names.append(img_name)
                    except (UnidentifiedImageError, ValueError) as e:
                        logger.error(f"Skipping invalid image {img_name}: {e}")

                # Skip this batch if no valid images remain
                if not valid_images:
                    logger.warning(f"Batch {batch_idx + 1} contains no valid images. Skipping.")
                    continue

                annotations = asyncio.run(server.request_async([{"image": img, "prompt": POLICY_DEFAULT} for img in valid_images]))
                invalid_json = save_json_annotations(annotations, output_dir_annotations, valid_image_names)
            
                if invalid_json:
                    logger.error(f"Invalid JSON in batch {batch_idx + 1}: {invalid_json}")

            logger.info(f"Compressing annotations of chunk {chunk_idx + 1}/{num_chunks}")
            compress_annotations(annotation_dir=output_dir_annotations, output_dir=output_dir_annotations_compressed)
            
            # Archive (without comnpression) the annotation folder and delete the original folder afterwards
            subprocess.run(
                ["tar", "-cf", f"{output_dir_annotations}.tar", output_dir_annotations],
                check=True
            )
            subprocess.run(
                ["rm", "-r", output_dir_annotations],
                check=True
            )
            
            rt.step()
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        logger.info("Inference done, shutting down the server.")
        server.tearDownClass()


if __name__ == '__main__':
    main()
