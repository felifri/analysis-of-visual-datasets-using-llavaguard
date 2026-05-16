import os
import logging
import pandas as pd
import subprocess
import torch
from diffusers import DiffusionPipeline

exp_name = "25_05_05_01"

def generate_images(model_name, model_dir):
    # Create directory for experiment and raise error if it already exists
    exp_dir = os.path.join(model_dir, "results", exp_name)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=False)

    logging.basicConfig(
        filename=f"{exp_dir}/generate_images.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading text-to-image pipeline {model_name} to GPU...")

    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16).to("cuda")
    # stage_1.enable_model_cpu_offload()

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    # stage_2.enable_model_cpu_offload()

    # stage 3
    safety_modules = {"feature_extractor": stage_1.feature_extractor}  # , "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
    stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16).to("cuda")
    # stage_3.enable_model_cpu_offload()

    logger.info("Pipeline ready.")

    prompt_file_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/prompt_testbench/prompt_testbench_10000_v6.csv"

    logger.info(f"Loading prompts from {prompt_file_path}...")

    prompt_df = pd.read_csv(prompt_file_path, index_col=0)

    missing_img_ids = [1866, 2750, 3012, 3058, 3069, 3090, 3112, 3117, 3161, 3169, 3188, 3199, 3215, 3244, 3253, 3261, 3276, 3313, 3374, 3376, 3417, 3432, 3501, 3531, 3582, 3603, 3617, 3636, 3663, 3668, 3675, 3688, 3715, 3762, 3851, 3934, 3944, 3946, 3974, 3975, 3988, 3992, 4035, 4039, 4058, 4063, 4070, 4072, 4093, 4114, 4117, 4118, 4125, 4143, 4154, 4155, 4161, 4165, 4166, 4174, 4193, 4199, 4217, 4222, 4242, 4248, 4249, 4251, 4255, 4258, 4291, 4294, 4313, 4315, 4318, 4326, 4335, 4339, 4341, 4353, 4367, 4378, 4379, 4401, 4403, 4407, 4421, 4424, 4425, 4427, 4444, 4454, 4457, 4466, 4491, 4495, 4498, 4499, 4500, 4510, 4530, 4537, 4549, 4552, 4560, 4573, 4589, 4597, 4604, 4606, 4632, 4646, 4650, 4653, 4654, 4660, 4666, 4678, 4695, 4700, 4702, 4722, 4747, 4748, 4752, 4761, 4774, 4777, 4782, 4784, 4798, 4803, 4811, 4827, 4842, 4845, 4846, 4849, 4864, 4867, 4869, 4882, 4895, 4908, 4921, 4947, 4956, 4958, 4968, 4969, 7164, 8819]

    logger.info(f"Starting image generation: {len(missing_img_ids) if missing_img_ids else len(prompt_df)} images will be generated.")

    for idx, row in prompt_df.iterrows():
        if missing_img_ids:
            if idx not in missing_img_ids:
                continue

        try:            
            prompt_embeds, negative_embeds = stage_1.encode_prompt(row.prompt)

            generator = torch.manual_seed(0)

            # stage 1
            image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images

            # stage 2
            image = stage_2(
                image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
            ).images

            # stage 3
            image = stage_3(prompt=row.prompt, image=image, generator=generator, noise_level=100).images
            image[0].save(f"{exp_dir}/images/{str(idx).zfill(len(str(len(prompt_df))))}.jpg")

            if (idx + 1) % 500 == 0 and idx > 0:
                logger.info(f"Generated {idx + 1} images.")
        except Exception as e:
            logger.error(e, exc_info=True)

    logger.info("Done generating images.")
    logger.info(f"Archiving generated images...")
    
    # Archive (without comnpression) the images and delete the original folder afterwards
    subprocess.run(
        ["tar", "-cf", f"{exp_dir}/images.tar", "-C", exp_dir, "images"],
        check=True
    )
    subprocess.run(
        ["rm", "-r", f"{exp_dir}/images"],
        check=True
    )

    logger.info(f"Done.")

if __name__ == '__main__':
    base_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models"

    generate_images("DeepFloyd/IF-I-XL-v1.0", os.path.join(base_path, "deep-floyd-if"))