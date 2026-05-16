"""Generate 10K evaluation images from existing PRX model checkpoints.

Optimized for H200 GPUs:
  - Batched generation (fills VRAM)
  - torch.compile on the transformer/UNet
  - Multi-GPU: splits prompts across all available GPUs

Usage:
    python entrypoint_generate_prx.py --model-index 0
    python entrypoint_generate_prx.py --model-index 0 --gpu-id 3
    python entrypoint_generate_prx.py --all
"""

import argparse
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from diffusers import DiffusionPipeline


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PRX_MODELS = [
    {"id": "Photoroom/prx-1024-t2i-beta",              "resolution": 1024, "short": "prx-1024-beta",          "batch_size": 8},
    {"id": "Photoroom/prx-512-t2i",                     "resolution": 512,  "short": "prx-512-base",           "batch_size": 32},
    {"id": "Photoroom/prx-512-t2i-sft",                 "resolution": 512,  "short": "prx-512-sft",            "batch_size": 32},
    {"id": "Photoroom/prx-512-t2i-sft-distilled",       "resolution": 512,  "short": "prx-512-sft-distilled",  "batch_size": 32},
    {"id": "Photoroom/prx-512-t2i-dc-ae",               "resolution": 512,  "short": "prx-512-dc-ae",          "batch_size": 32},
    {"id": "Photoroom/prx-256-t2i",                     "resolution": 256,  "short": "prx-256-base",           "batch_size": 64},
    {"id": "Photoroom/prx-256-t2i-sft",                 "resolution": 256,  "short": "prx-256-sft",            "batch_size": 64},
]


def generate_images_on_gpu(
    model_info: dict,
    output_base: str,
    prompt_file: str,
    gpu_id: int,
    prompt_indices: list[int] | None,
    logger: logging.Logger,
):
    """Generate images from a single PRX model on a specific GPU.

    Args:
        prompt_indices: If provided, only generate these prompt indices.
                       If None, generate all prompts.
    """
    model_id = model_info["id"]
    resolution = model_info["resolution"]
    short_name = model_info["short"]
    batch_size = model_info["batch_size"]

    output_dir = os.path.join(output_base, short_name)
    images_dir = os.path.join(output_dir, "images")
    tar_path = os.path.join(output_dir, "images.tar")

    # Skip if already done
    if os.path.exists(tar_path):
        logger.info(f"[GPU {gpu_id}] Skipping {short_name}: images.tar already exists")
        return

    os.makedirs(images_dir, exist_ok=True)

    device = f"cuda:{gpu_id}"
    logger.info(f"[GPU {gpu_id}] Loading model {model_id} (resolution={resolution}, batch_size={batch_size})...")

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Compile the denoiser for faster inference
    try:
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
            logger.info(f"[GPU {gpu_id}] Compiled transformer")
        elif hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
            logger.info(f"[GPU {gpu_id}] Compiled UNet")
    except Exception as e:
        logger.warning(f"[GPU {gpu_id}] torch.compile failed, continuing without: {e}")

    prompt_df = pd.read_csv(prompt_file, index_col=0)
    num_digits = len(str(len(prompt_df)))

    # Determine which prompts this GPU handles
    if prompt_indices is None:
        prompt_indices = list(range(len(prompt_df)))

    # Filter out already generated
    existing = set()
    if os.path.isdir(images_dir):
        for f in os.listdir(images_dir):
            if f.endswith(".jpg"):
                existing.add(f.removesuffix(".jpg"))

    todo = [(idx, prompt_df.iloc[idx]["prompt"]) for idx in prompt_indices
            if str(idx).zfill(num_digits) not in existing]

    logger.info(f"[GPU {gpu_id}] {short_name}: {len(todo)} images to generate ({len(existing)} exist)")

    if not todo:
        return

    generator = torch.Generator(device=device).manual_seed(42)
    start = time.time()
    generated = 0

    # Batch generation
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        batch_indices = [b[0] for b in batch]
        batch_prompts = [b[1] for b in batch]

        try:
            images = pipe(
                prompt=batch_prompts,
                width=resolution,
                height=resolution,
                guidance_scale=3.5,
                num_inference_steps=50,
                generator=generator,
            ).images

            for idx, image in zip(batch_indices, images):
                img_name = str(idx).zfill(num_digits)
                image.save(os.path.join(images_dir, f"{img_name}.jpg"))

            generated += len(images)

            if generated % 500 < batch_size:
                elapsed = time.time() - start
                rate = generated / elapsed if elapsed > 0 else 0
                total_done = len(existing) + generated
                logger.info(
                    f"[GPU {gpu_id}] {short_name}: {total_done}/{len(prompt_df)} "
                    f"({rate:.1f} img/s)"
                )
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Error in batch starting at {batch_indices[0]}: {e}", exc_info=True)
            # Fall back to single-image generation for this batch
            for idx, prompt in zip(batch_indices, batch_prompts):
                try:
                    image = pipe(
                        prompt=prompt,
                        width=resolution,
                        height=resolution,
                        guidance_scale=3.5,
                        num_inference_steps=50,
                        generator=generator,
                    ).images[0]
                    img_name = str(idx).zfill(num_digits)
                    image.save(os.path.join(images_dir, f"{img_name}.jpg"))
                    generated += 1
                except Exception:
                    pass

    elapsed = time.time() - start
    rate = generated / elapsed if elapsed > 0 else 0
    logger.info(f"[GPU {gpu_id}] {short_name}: {generated} generated in {elapsed / 60:.1f}m ({rate:.1f} img/s)")

    del pipe
    torch.cuda.empty_cache()


def generate_multi_gpu(
    model_info: dict,
    output_base: str,
    prompt_file: str,
    num_gpus: int,
    logger: logging.Logger,
):
    """Generate images using multiple GPUs in parallel.

    Loads model once, creates per-GPU copies sequentially to avoid meta tensor issues,
    then generates in parallel across GPUs.
    """
    import copy

    model_id = model_info["id"]
    short_name = model_info["short"]
    batch_size = model_info["batch_size"]
    resolution = model_info["resolution"]
    output_dir = os.path.join(output_base, short_name)
    images_dir = os.path.join(output_dir, "images")
    tar_path = os.path.join(output_dir, "images.tar")

    if os.path.exists(tar_path):
        logger.info(f"Skipping {short_name}: images.tar already exists")
        return

    os.makedirs(images_dir, exist_ok=True)

    prompt_df = pd.read_csv(prompt_file, index_col=0)
    num_digits = len(str(len(prompt_df)))
    all_indices = list(range(len(prompt_df)))

    # Filter out already generated
    existing = set()
    if os.path.isdir(images_dir):
        for f in os.listdir(images_dir):
            if f.endswith(".jpg"):
                existing.add(f.removesuffix(".jpg"))

    todo = [(idx, prompt_df.iloc[idx]["prompt"]) for idx in all_indices
            if str(idx).zfill(num_digits) not in existing]

    if not todo:
        logger.info(f"All images already generated for {short_name}")
        return

    logger.info(f"Loading model {model_id} once, then replicating to {num_gpus} GPUs...")
    base_pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Create per-GPU pipelines sequentially
    pipes = {}
    for gpu_id in range(num_gpus):
        logger.info(f"  Copying to GPU {gpu_id}...")
        pipe = copy.deepcopy(base_pipe).to(f"cuda:{gpu_id}")
        # Compile the denoiser
        try:
            if hasattr(pipe, "transformer"):
                pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
            elif hasattr(pipe, "unet"):
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        except Exception:
            pass
        pipes[gpu_id] = pipe

    del base_pipe

    # Split work across GPUs
    gpu_todos = [[] for _ in range(num_gpus)]
    for i, item in enumerate(todo):
        gpu_todos[i % num_gpus].append(item)

    logger.info(f"Generating {len(todo)} images across {num_gpus} GPUs (batch_size={batch_size})")

    def gpu_worker(gpu_id: int, gpu_todo: list):
        pipe = pipes[gpu_id]
        device = f"cuda:{gpu_id}"
        generator = torch.Generator(device=device).manual_seed(42)
        generated = 0
        start = time.time()

        for batch_start in range(0, len(gpu_todo), batch_size):
            batch = gpu_todo[batch_start:batch_start + batch_size]
            batch_indices = [b[0] for b in batch]
            batch_prompts = [b[1] for b in batch]

            try:
                images = pipe(
                    prompt=batch_prompts,
                    width=resolution, height=resolution,
                    guidance_scale=3.5, num_inference_steps=50,
                    generator=generator,
                ).images
                for idx, image in zip(batch_indices, images):
                    img_name = str(idx).zfill(num_digits)
                    image.save(os.path.join(images_dir, f"{img_name}.jpg"))
                generated += len(images)
            except Exception as e:
                logger.error(f"[GPU {gpu_id}] Batch error: {e}")
                for idx, prompt in zip(batch_indices, batch_prompts):
                    try:
                        img = pipe(prompt=prompt, width=resolution, height=resolution,
                                   guidance_scale=3.5, num_inference_steps=50,
                                   generator=generator).images[0]
                        img.save(os.path.join(images_dir, f"{str(idx).zfill(num_digits)}.jpg"))
                        generated += 1
                    except Exception:
                        pass

            if generated % 200 < batch_size:
                elapsed = time.time() - start
                rate = generated / elapsed if elapsed > 0 else 0
                logger.info(f"[GPU {gpu_id}] {short_name}: {generated}/{len(gpu_todo)} ({rate:.1f} img/s)")

        elapsed = time.time() - start
        rate = generated / elapsed if elapsed > 0 else 0
        logger.info(f"[GPU {gpu_id}] {short_name}: done {generated} in {elapsed/60:.1f}m ({rate:.1f} img/s)")

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(gpu_worker, gpu_id, gpu_todos[gpu_id])
                   for gpu_id in range(num_gpus)]
        for f in futures:
            f.result()

    # Cleanup
    for pipe in pipes.values():
        del pipe
    torch.cuda.empty_cache()

    # Archive
    num_images = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    logger.info(f"Archiving {short_name}: {num_images} images...")
    subprocess.run(["tar", "-cf", tar_path, "-C", output_dir, "images"], check=True)
    subprocess.run(["rm", "-r", images_dir], check=True)
    logger.info(f"{short_name}: done.")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation images from PRX models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Single GPU ID to use (default: use all available)")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    prompt_file = config["generation"]["prompt_file"]
    output_base = os.path.join(config["generation"]["output_dir"], "prx_existing")
    os.makedirs(output_base, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_base, "generate_prx.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    if args.model_index is not None:
        models = [PRX_MODELS[args.model_index]]
    elif args.model_id:
        models = [m for m in PRX_MODELS if m["id"] == args.model_id]
        if not models:
            logger.error(f"Unknown model: {args.model_id}")
            return
    elif args.all:
        models = PRX_MODELS
    else:
        logger.error("Specify --model-id, --model-index, or --all")
        return

    num_gpus = torch.cuda.device_count() if args.gpu_id is None else 1
    logger.info(f"Will generate for {len(models)} models using {num_gpus} GPUs")

    for model_info in models:
        if args.gpu_id is not None:
            generate_images_on_gpu(
                model_info, output_base, prompt_file, args.gpu_id, None, logger,
            )
            # Archive
            short_name = model_info["short"]
            output_dir = os.path.join(output_base, short_name)
            images_dir = os.path.join(output_dir, "images")
            tar_path = os.path.join(output_dir, "images.tar")
            if os.path.isdir(images_dir) and not os.path.exists(tar_path):
                num_images = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
                logger.info(f"Archiving {short_name}: {num_images} images...")
                subprocess.run(["tar", "-cf", tar_path, "-C", output_dir, "images"], check=True)
                subprocess.run(["rm", "-r", images_dir], check=True)
        else:
            generate_multi_gpu(model_info, output_base, prompt_file, num_gpus, logger)


if __name__ == "__main__":
    main()
