import os
import logging
import pandas as pd
import subprocess
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import HiDreamImagePipeline

exp_name = "25_04_25_01"

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

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )

    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Full",  # "HiDream-ai/HiDream-I1-Dev" | "HiDream-ai/HiDream-I1-Fast"
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    ).to("cuda", dtype=torch.bfloat16)

    logger.info(f"text_encoder_4 dtype: {text_encoder_4.dtype}")
    logger.info(f"pipe dtype: {pipe.dtype}")

    logger.info("Pipeline ready.")

    prompt_file_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/prompt_testbench/prompt_testbench_10000_v6.csv"

    logger.info(f"Loading prompts from {prompt_file_path}...")

    prompt_df = pd.read_csv(prompt_file_path, index_col=0)

    logger.info(f"Starting image generation: {len(prompt_df)} images will be generated.")

    for idx, row in enumerate(prompt_df.itertuples()):
        # if idx < 3000 or idx > 4999:
        #     continue

        try:            
            image = pipe(
                prompt=row.prompt,
                guidance_scale=5.0,  # 0.0 for Dev&Fast
                num_inference_steps=50,
                generator=torch.Generator("cuda").manual_seed(0),
                width=512,
                height=512
            ).images[0]
            image.save(f"{exp_dir}/images/{str(idx).zfill(len(str(len(prompt_df))))}.jpg")

            if (idx + 1) % 500 == 0 and idx > 0:
                logger.info(f"Generated {idx + 1} images.")
        except Exception as e:
            logger.error(e, exc_info=True)

    logger.info("Done generating images.")
    logger.info(f"Archiving generated images...")
    
    # Archive (without comnpression) the images and delete the original folder afterwards
    subprocess.run(
        ["tar", "-cf", f"{exp_dir}/images.tar", f"{exp_dir}/images"],
        check=True
    )
    subprocess.run(
        ["rm", "-r", f"{exp_dir}/images"],
        check=True
    )

    logger.info(f"Done.")

if __name__ == '__main__':
    base_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models"

    models = [
        ("HiDream-ai/HiDream-I1-Full", "hidream-i1-full"),
    ]

    for model, model_dir in models:
        generate_images(model, os.path.join(base_path, model_dir))