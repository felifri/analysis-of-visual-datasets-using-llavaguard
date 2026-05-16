import os
import logging
import pandas as pd
from diffusers import DiffusionPipeline

exp_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/newrealityxl-global-nsfw/results/25_03_01_01"

def generate_images():
    # Create directory for experiment results
    os.makedirs(f"{exp_dir}/images", exist_ok=True)

    logging.basicConfig(
        filename=f"{exp_dir}/log.txt", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting image generation: newrealityxl-global-nsfw...")

    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/newrealityxl-global-nsfw").to("cuda")

    logger.info("Pretrained pipeline ready.")

    prompt_df = pd.read_csv("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/newrealityxl-global-nsfw/prompts.csv")

    logger.info(f"{len(prompt_df)} images will be generated.")

    for count, row in enumerate(prompt_df.itertuples(), start=1):
        try:
            idx = row.incremental_id
            
            image = pipe(row.prompt).images[0]
            image.save(f"{exp_dir}/images/{idx}.jpg")

            if count % 500 == 0:
                logger.info(f"Generated {count} images.")
        except Exception as e:
            logger.error(e, exc_info=True)

    logger.info("Done generating images.")

if __name__ == '__main__':
    generate_images()