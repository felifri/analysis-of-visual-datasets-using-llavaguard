import asyncio
import math
import subprocess
import logging
import os
from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import compress_annotations, save_json_annotations
from util.policy import POLICY_DEFAULT
import webdataset as wds
from huggingface_hub import get_token

base_output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/open-images-v6/results/7B_25_04_30_05"
base_output_dir_annotations_compressed = os.path.join(base_output_dir, "annotations_compressed")

def is_valid_sample(sample):
    return sample is not None and all(value is not None for value in sample.values())

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

    logger.info("Starting image annotation: open-images-v6...")

    server = LlavaGuardServer()
    chunk_zfill = 3  # 9M images will result in about 90 chunks
    chunk_size = 100000
    batch_size = 1000

    server.setUpClass(
        model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
        dp_size=4,
        port=10001,
        is_requests_wrapper=True
    )

    logger.info("LlavaGuard server ready.")

    ds_splits = {
        "valid": f"https://huggingface.co/datasets/dalle-mini/open-images/resolve/main/data/valid/{{00000..00008}}.tar",
        "test": f"https://huggingface.co/datasets/dalle-mini/open-images/resolve/main/data/test/{{00000..00024}}.tar",
        "train": f"https://huggingface.co/datasets/dalle-mini/open-images/resolve/main/data/train/{{00000..01802}}.tar"
    }

    try:
        for split, url in ds_splits.items():
            logger.info(f"Loading dataset split {split}...")
            
            hf_token = get_token()
            url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"

            dataset = (
                wds.WebDataset(url, shardshuffle=False)
                .decode("pil")
                .map(lambda x: x if is_valid_sample(x) else None)
                .to_tuple("__key__", "jpg")
                .batched(batch_size)
            )

            logger.info("Successfully loaded dataset.")
            logger.info("Creating annotations now...")

            processed_samples = 0
            chunk_idx = 0

            output_dir_annotations_compressed = os.path.join(base_output_dir_annotations_compressed, split)
            os.makedirs(output_dir_annotations_compressed)

            for batch in dataset:
                chunk_idx_str = str(chunk_idx).zfill(chunk_zfill)

                output_dir_annotations = os.path.join(base_output_dir, "annotations", split, chunk_idx_str)
                os.makedirs(output_dir_annotations, exist_ok=True)

                keys, images = batch
                annotations = asyncio.run(server.request_async([{"image": img, "prompt": POLICY_DEFAULT} for img in images]))
                invalid_json = save_json_annotations(annotations, output_dir_annotations, keys)
            
                if invalid_json:
                    logger.error(f"Invalid JSON in batch {math.ceil(processed_samples / batch_size)}: {invalid_json}")
                else:
                    logger.info(f"Processed batch: {math.ceil(processed_samples / batch_size)}")
                
                processed_samples += batch_size

                if processed_samples >= chunk_size:
                    logger.info(f"Compressing annotations of chunk {chunk_idx}")
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
                    logger.info(f"Chunk {chunk_idx} done.")

                    chunk_idx += 1
                    processed_samples = 0
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        logger.info("Inference done, shutting down the server.")
        server.tearDownClass()


if __name__ == '__main__':
    main()
