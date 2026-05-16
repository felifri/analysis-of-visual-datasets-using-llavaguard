"""Step 5: Generate evaluation images from trained models and existing PRX checkpoints.

For each of the 6 trained conditions (C1-C6) plus any existing PRX checkpoints
(base, SFT, RL), generates 10K images using prompts from the prompt testbench.

Uses PRX inference pipeline with consistent generation settings:
  guidance_scale=3.5, num_inference_steps=50, seed=42, resolution=1024x1024

Usage:
    python entrypoint_generate_images.py [--conditions C1 C2 ...] [--existing-only]
"""

import argparse
import json
import logging
import os
import subprocess
import time

import pandas as pd
import torch
from diffusers import DiffusionPipeline


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6"]

EXISTING_CHECKPOINTS = {
    "prx_base": "Photoroom/prx-1024-t2i-beta",
}


def generate_images_for_model(
    model_id: str,
    model_path: str,
    output_dir: str,
    prompt_file: str,
    guidance_scale: float,
    num_inference_steps: int,
    resolution: int,
    seed: int,
    logger: logging.Logger,
):
    """Generate images from a single model checkpoint."""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    logger.info(f"Loading model {model_id} from {model_path}...")

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    logger.info("Pipeline loaded.")

    prompt_df = pd.read_csv(prompt_file, index_col=0)
    num_prompts = len(prompt_df)
    num_digits = len(str(num_prompts))

    # Check which images already exist (for resuming)
    existing = set()
    for f in os.listdir(images_dir):
        if f.endswith(".jpg"):
            existing.add(f.removesuffix(".jpg"))

    logger.info(f"Generating {num_prompts} images ({len(existing)} already exist)...")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()

    for idx, row in enumerate(prompt_df.itertuples()):
        img_name = str(idx).zfill(num_digits)
        if img_name in existing:
            continue

        try:
            image = pipe(
                prompt=row.prompt,
                width=resolution,
                height=resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            image.save(os.path.join(images_dir, f"{img_name}.jpg"))

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (idx + 1 - len(existing)) / elapsed if elapsed > 0 else 0
                logger.info(f"Generated {idx + 1}/{num_prompts} ({rate:.1f} img/s)")
        except Exception as e:
            logger.error(f"Error generating image {idx}: {e}", exc_info=True)

    elapsed = time.time() - start
    logger.info(f"Generation complete: {num_prompts} images in {elapsed / 3600:.1f}h")

    # Archive images
    logger.info("Archiving images...")
    subprocess.run(
        ["tar", "-cf", os.path.join(output_dir, "images.tar"), "-C", output_dir, "images"],
        check=True,
    )
    subprocess.run(["rm", "-r", images_dir], check=True)

    # Free GPU memory
    del pipe
    torch.cuda.empty_cache()

    logger.info(f"Done with {model_id}.")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation images for dose-response models")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=CONDITIONS,
        help="Which conditions to generate for (default: all)",
    )
    parser.add_argument("--existing-only", action="store_true",
                        help="Only generate from existing PRX checkpoints, skip trained conditions")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip existing PRX checkpoints, only generate from trained conditions")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    gen_config = config["generation"]
    prompt_file = gen_config["prompt_file"]
    guidance_scale = gen_config["guidance_scale"]
    num_inference_steps = gen_config["num_inference_steps"]
    resolution = gen_config["resolution"]
    seed = gen_config["seed"]
    base_output_dir = gen_config["output_dir"]
    checkpoint_dir = config["training"]["checkpoint_dir"]

    os.makedirs(base_output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(base_output_dir, "generate_images.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    models_to_generate = []

    # Add trained conditions
    if not args.existing_only:
        for condition in args.conditions:
            if condition not in CONDITIONS:
                logger.warning(f"Unknown condition {condition}, skipping")
                continue

            # Look for phase2 checkpoint first, then phase1
            ckpt_path = None
            for phase in ["phase2", "phase1"]:
                candidate = os.path.join(checkpoint_dir, condition, phase, "latest-rank0.pt")
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break

            if ckpt_path is None:
                logger.warning(f"No checkpoint found for {condition}, skipping")
                continue

            models_to_generate.append((f"dose_{condition}", ckpt_path))

    # Add existing PRX checkpoints
    if not args.skip_existing:
        for ckpt_name, ckpt_id in EXISTING_CHECKPOINTS.items():
            if ckpt_id:
                models_to_generate.append((ckpt_name, ckpt_id))

        # Also check for any configured existing checkpoints
        eval_config = config.get("evaluation", {}).get("existing_prx_checkpoints", {})
        for ckpt_name, ckpt_id in eval_config.items():
            if ckpt_id and ckpt_name not in EXISTING_CHECKPOINTS:
                models_to_generate.append((ckpt_name, ckpt_id))

    logger.info(f"Will generate images for {len(models_to_generate)} models:")
    for model_id, model_path in models_to_generate:
        logger.info(f"  {model_id}: {model_path}")

    for model_id, model_path in models_to_generate:
        output_dir = os.path.join(base_output_dir, model_id)
        if os.path.exists(os.path.join(output_dir, "images.tar")):
            logger.info(f"Skipping {model_id}: images.tar already exists")
            continue

        os.makedirs(output_dir, exist_ok=True)
        generate_images_for_model(
            model_id=model_id,
            model_path=model_path,
            output_dir=output_dir,
            prompt_file=prompt_file,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            resolution=resolution,
            seed=seed,
            logger=logger,
        )


if __name__ == "__main__":
    main()
