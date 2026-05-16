import asyncio
import itertools
import logging
import os
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.file_utils import get_file_path_generator
from util.annotation_utils import save_json_annotations
from util.policy import POLICY_DEFAULT
from tqdm import tqdm

split = "train"
base_image_dir = f"/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/lsun/images/{split}"
base_output_dir = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/lsun/results/7B_25_03_21_01/{split}"

def main():
    # Create annotation dir and raise an error if they already exist to prevent data loss
    os.makedirs(base_output_dir)

    logging.basicConfig(
        filename=f"{base_output_dir}/generate_annotations.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info("Starting inference for LSUN...")

    server = LlavaGuardServer()

    # Will not actually start a server, but instead forward the requests to an already running instance of sglang.router
    # This provides almost linear data parallelisation up to 4 GPUs
    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        port=10001,
        is_requests_wrapper=True
    )

    # Categories can have up to 10,000,000 images
    chunk_size = 100000
    oom_chunks = 2
    batch_size = 1000

    logger.info("Server is up and running.")

    categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']

    try:
        for category in tqdm(categories):
            logger.info(f"Processing category {category}.")

            image_dir = os.path.join(base_image_dir, category)
            image_generator = get_file_path_generator(image_dir, file_extension='.jpg')
            image_counter = 0

            while True:
                batch = list(itertools.islice(image_generator, batch_size))
                if not batch:
                    break

                idx_chunk = image_counter // chunk_size
                idx_chunk_str = str(idx_chunk).zfill(oom_chunks)
                output_dir_annotations = os.path.join(base_output_dir, category, idx_chunk_str)


                if image_counter % chunk_size == 0:
                    logger.info(f"Processing chunk {idx_chunk_str}: {image_counter} images processed.")
                    
                    # Create annotation dir and raise an error if they already exist to prevent data loss
                    os.makedirs(output_dir_annotations)

                annotations = asyncio.run(server.request_async([{"image": img_path, "prompt": POLICY_DEFAULT} for img_path, _ in batch]))

                invalid_json = save_json_annotations(annotations, output_dir_annotations, [img_name for _, img_name in batch])

                if invalid_json:
                    logger.warning(f"Invalid JSON annotations: {invalid_json}")

                image_counter += len(batch)
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        logger.info("Inference done, shutting down the server.")
        server.tearDownClass()


if __name__ == '__main__':
    main()
