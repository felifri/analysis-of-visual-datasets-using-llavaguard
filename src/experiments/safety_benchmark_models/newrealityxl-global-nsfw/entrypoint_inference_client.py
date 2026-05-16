import asyncio
import logging
import os
import rtpt
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.file_utils import get_file_paths
from util.policy import POLICY_DEFAULT

image_dir = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/newrealityxl-global-nsfw/results/25_03_01_01/images"
output_dir = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/newrealityxl-global-nsfw/results/25_03_10_02/annotations"


def main():
    # Try to create the output dir. If it already exists, throw an error to avoid overwriting data.
    os.makedirs(output_dir)
    
    logging.basicConfig(
        filename=f"{output_dir}/log.txt", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting image annotation: newrealityxl-global-nsfw...")

    server = LlavaGuardServer()

    image_paths, image_names = get_file_paths(image_dir, file_extension='.jpg')
    inputs = [
        {
            'image': path,
            'image_name': name,
            'prompt': POLICY_DEFAULT,
        } for path, name in zip(image_paths, image_names)
    ]

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=2,
        port=10002,
        is_requests_wrapper=True
    )

    logger.info("LlavaGuard server ready.")

    batch_size = 1000
    num_batches = str(len(inputs)//batch_size + 1).zfill(len(str(len(inputs)//batch_size)))

    logger.info(f"Total {len(inputs)} annotations will be generated in {num_batches} batches")
    try:
        rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(inputs)//batch_size + 1)
        rt.start()

        for i in range(0, len(inputs), batch_size):
            idx_batch = str(i // batch_size).zfill(len(str(len(inputs)//batch_size)))
            logger.info(f"Running batch {idx_batch}/{num_batches}")

            annotations = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))

            # logger.info(f"Saving annotations of batch {idx_batch}/{num_batches}.")
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
