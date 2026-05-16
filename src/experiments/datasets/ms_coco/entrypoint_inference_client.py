import asyncio
import logging
import os
import rtpt
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.file_utils import get_file_paths
from util.policy import POLICY_DEFAULT

base_image_dir = f"/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/ms-coco/images"
base_output_dir = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/ms_coco/results/25_03_12_01/annotations"


def main():
    # Create the output dir. If it already exists, throw an error to avoid overwriting data.
    os.makedirs(base_output_dir)
    
    logging.basicConfig(
        filename=f"{base_output_dir}/generate_annotations.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info("Starting image annotation: MS COCO...")

    server = LlavaGuardServer()
    batch_size = 1000

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=4,
        port=10000,
        is_requests_wrapper=True
    )

    logger.info("LlavaGuard server ready.")

    try:
        for split in ["train2017", "test2017", "val2017", "unlabeled2017"]:
            logger.info(f"Generating annotations for split '{split}.'")

            image_dir = f"{base_image_dir}/{split}"
            output_dir = f"{base_output_dir}/{split}"
            os.makedirs(f"{output_dir}/image_annotations")

            image_paths, image_names = get_file_paths(image_dir, file_extension='.jpg')
            inputs = [
                {
                    'image': path,
                    'image_name': name,
                    'prompt': POLICY_DEFAULT,
                } for path, name in zip(image_paths, image_names)
            ]
            num_batches = str(len(inputs)//batch_size + 1).zfill(len(str(len(inputs)//batch_size)))

            logger.info(f"Total {len(inputs)} annotations will be generated in {num_batches} batches.")
            rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(inputs)//batch_size + 1)
            rt.start()

            for i in range(0, len(inputs), batch_size):
                idx_batch = str(i // batch_size).zfill(len(str(len(inputs)//batch_size)))
                logger.info(f"Running batch {idx_batch}/{num_batches}...")

                annotations = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))

                invalid_json = save_json_annotations(annotations, f"{output_dir}/image_annotations/batch_{idx_batch}", [input_['image_name'] for input_ in inputs[i:i+batch_size]])

                if invalid_json:
                    logger.warning(f"Invalid JSON annotations found in batch {idx_batch}/{num_batches}: {invalid_json}")

                rt.step()
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        logger.info("Inference done, shutting down the server.")
        server.tearDownClass()

if __name__ == '__main__':
    main()
