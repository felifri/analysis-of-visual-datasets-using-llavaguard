"""Generate 10K evaluation images from dose-response model checkpoints (C1-C6).

Loads PRX-1.2B models from Composer FSDP distributed checkpoints and generates
images using the prompt testbench. Handles both phase1 (512px) and phase2 (1024px)
checkpoints.

The FSDP checkpoints contain only denoiser weights (vae and text_tower are frozen
and excluded via save_ignore_keys). This script rebuilds the full pipeline from
the Hydra config and loads only the denoiser state dict.

Usage:
    # Generate from a single condition
    python entrypoint_generate_dose_response.py --condition C1

    # Generate from all conditions
    python entrypoint_generate_dose_response.py --all

    # Use a specific checkpoint phase
    python entrypoint_generate_dose_response.py --condition C1 --phase phase2

    # Override batch size
    python entrypoint_generate_dose_response.py --condition C1 --batch-size 4
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time

import pandas as pd
import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from PIL import Image

# Add PRX to path
PRX_DIR = os.environ.get(
    "PRX_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "PRX")),
)
if PRX_DIR not in sys.path:
    sys.path.insert(0, PRX_DIR)

# Add project src to path
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5", "C1_clip", "C0_clip", "C1_safeclip", "C0_safeclip"]


def find_checkpoint_dir(base_dir: str, condition: str, phase: str) -> str | None:
    """Find the checkpoint directory for a condition and phase.

    Checks checkpoints_full/ first, then checkpoints/.
    Returns the directory containing the .distcp files (or the symlink target).
    """
    for ckpt_root in ["checkpoints_full", "checkpoints"]:
        ckpt_dir = os.path.join(base_dir, ckpt_root, condition, phase)
        if not os.path.isdir(ckpt_dir):
            continue

        # Check for latest-rank0.pt symlink or actual checkpoint dir
        latest = os.path.join(ckpt_dir, "latest-rank0.pt")
        if os.path.exists(latest):
            # Resolve symlink to get the actual checkpoint directory
            target = os.path.realpath(latest)
            if os.path.isdir(target):
                return target

        # Check for direct .distcp files
        if any(f.endswith(".distcp") for f in os.listdir(ckpt_dir)):
            return ckpt_dir

        # Check subdirectories (e.g., ep3-ba100000/)
        for entry in sorted(os.listdir(ckpt_dir)):
            subdir = os.path.join(ckpt_dir, entry)
            if os.path.isdir(subdir) and any(
                f.endswith(".distcp") for f in os.listdir(subdir)
            ):
                return subdir

    return None


def find_config_path(base_dir: str, condition: str, phase: str) -> str | None:
    """Find the Hydra config.yaml for a checkpoint."""
    for ckpt_root in ["checkpoints_full", "checkpoints"]:
        config_path = os.path.join(base_dir, ckpt_root, condition, phase, "config.yaml")
        if os.path.exists(config_path):
            return config_path
    return None


def load_denoiser_state_dict(checkpoint_dir: str, logger: logging.Logger) -> dict:
    """Load denoiser weights from an FSDP distributed checkpoint.

    Prefers EMA weights (ema_denoiser.model.*) over regular denoiser weights,
    as EMA weights produce better inference results.
    """
    logger.info(f"Loading FSDP checkpoint from {checkpoint_dir}")

    # Read metadata to get key names
    metadata_path = os.path.join(checkpoint_dir, ".metadata")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    all_keys = list(metadata.state_dict_metadata.keys())

    # Prefer EMA denoiser weights for inference
    ema_keys = [k for k in all_keys if "ema_denoiser.model." in k]
    regular_keys = [k for k in all_keys if k.startswith("state.model.denoiser.")]

    if ema_keys:
        logger.info(f"Using EMA denoiser weights ({len(ema_keys)} keys)")
        load_keys = ema_keys
        use_ema = True
    else:
        logger.info(f"No EMA weights found, using regular denoiser ({len(regular_keys)} keys)")
        load_keys = regular_keys
        use_ema = False

    # Build a state dict with empty tensors matching the shapes
    state_dict = {}
    for key in load_keys:
        tensor_meta = metadata.state_dict_metadata[key]
        if hasattr(tensor_meta, "size"):
            state_dict[key] = torch.empty(tensor_meta.size, dtype=torch.float32)
        elif hasattr(tensor_meta, "properties") and hasattr(tensor_meta.properties, "size"):
            state_dict[key] = torch.empty(
                tensor_meta.properties.size, dtype=torch.float32
            )
        else:
            state_dict[key] = torch.tensor(0.0)

    # Load using distributed checkpoint reader
    storage_reader = dcp.FileSystemReader(checkpoint_dir)
    dcp.load(state_dict, storage_reader=storage_reader)

    # Map keys to Pipeline's expected format: denoiser.*
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        # Strip state.model. prefix
        if new_key.startswith("state.model."):
            new_key = new_key[len("state.model."):]
        # Map EMA keys: ema_denoiser.model.X -> denoiser.X
        if use_ema and new_key.startswith("ema_denoiser.model."):
            new_key = "denoiser." + new_key[len("ema_denoiser.model."):]
        stripped[new_key] = value

    logger.info(f"Loaded {len(stripped)} denoiser parameters")
    return stripped


def build_pipeline(config_path: str, device: str, logger: logging.Logger):
    """Build a Pipeline from the Hydra config without loading weights."""
    from prx.pipeline.models_factory import build_pipeline as _build_pipeline

    config = OmegaConf.load(config_path)

    # Extract component configs
    denoiser_config = OmegaConf.to_container(config.diffusion_model, resolve=True)
    text_tower_config = OmegaConf.to_container(config.diffusion_text_tower, resolve=True)
    vae_config = OmegaConf.to_container(config.diffusion_vae, resolve=True)
    scheduler_config = OmegaConf.to_container(config.diffusion_scheduler, resolve=True)

    image_size = config.get("image_size", 512)
    denoiser_dtype = str(config.get("denoiser_dtype", "torch.float"))

    logger.info(f"Building pipeline: image_size={image_size}, dtype={denoiser_dtype}")

    # Set CUDA device before building
    torch.cuda.set_device(device)

    pipeline = _build_pipeline(
        denoiser_config=denoiser_config,
        text_tower_config=text_tower_config,
        vae_config=vae_config,
        scheduler_config=scheduler_config,
        input_size=image_size,
        p_drop_caption=0.0,
        denoiser_dtype=denoiser_dtype,
    )

    return pipeline, image_size


def load_pipeline(
    config_path: str,
    checkpoint_dir: str,
    device: str,
    logger: logging.Logger,
):
    """Build pipeline and load denoiser weights from FSDP checkpoint."""
    pipeline, image_size = build_pipeline(config_path, device, logger)

    # Load denoiser weights
    denoiser_state = load_denoiser_state_dict(checkpoint_dir, logger)

    # Load only denoiser weights into the pipeline (vae, text_tower stay as initialized)
    missing, unexpected = pipeline.load_state_dict(denoiser_state, strict=False)

    # Filter out expected missing keys (vae, text_tower, etc.)
    real_missing = [k for k in missing if k.startswith("denoiser.")]
    if real_missing:
        logger.warning(f"Missing denoiser keys: {real_missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}")

    pipeline.eval()
    logger.info("Pipeline loaded and set to eval mode")

    return pipeline, image_size


def generate_images(
    pipeline,
    prompts: list[str],
    image_size: int,
    output_dir: str,
    batch_size: int = 4,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    logger: logging.Logger = None,
):
    """Generate images from prompts using the pipeline."""
    from prx.dataset.constants import BatchKeys

    os.makedirs(output_dir, exist_ok=True)

    num_digits = len(str(len(prompts)))

    # Check which images already exist
    existing = set()
    for f in os.listdir(output_dir):
        if f.endswith(".jpg"):
            existing.add(f.removesuffix(".jpg"))

    todo = [
        (idx, prompt)
        for idx, prompt in enumerate(prompts)
        if str(idx).zfill(num_digits) not in existing
    ]

    if logger:
        logger.info(
            f"{len(todo)} images to generate ({len(existing)} exist, {len(prompts)} total)"
        )

    if not todo:
        return

    start = time.time()
    generated = 0

    with torch.no_grad():
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start : batch_start + batch_size]
            batch_indices = [b[0] for b in batch]
            batch_prompts = [b[1] for b in batch]

            try:
                batch_dict = {BatchKeys.PROMPT: batch_prompts}

                images = pipeline.generate(
                    batch=batch_dict,
                    image_size=(image_size, image_size),
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    progress_bar=False,
                    decode_latents=True,
                )

                # Save images
                for idx, img_tensor in zip(batch_indices, images):
                    img_name = str(idx).zfill(num_digits)
                    # Convert tensor [C, H, W] in [0, 1] to PIL
                    img_array = (
                        img_tensor.clamp(0, 1).cpu().float().numpy().transpose(1, 2, 0) * 255
                    ).astype("uint8")
                    img_pil = Image.fromarray(img_array)
                    img_pil.save(os.path.join(output_dir, f"{img_name}.jpg"))

                generated += len(images)

                if generated % max(1, 100 // batch_size) == 0 and logger:
                    elapsed = time.time() - start
                    rate = generated / elapsed if elapsed > 0 else 0
                    total_done = len(existing) + generated
                    logger.info(
                        f"Generated {total_done}/{len(prompts)} "
                        f"({rate:.1f} img/s, elapsed {elapsed/60:.1f}m)"
                    )
            except Exception as e:
                if logger:
                    logger.error(f"Error in batch starting at {batch_indices[0]}: {e}", exc_info=True)
                # Fall back to single-image generation
                for idx, prompt in zip(batch_indices, batch_prompts):
                    try:
                        batch_dict = {BatchKeys.PROMPT: [prompt]}
                        img = pipeline.generate(
                            batch=batch_dict,
                            image_size=(image_size, image_size),
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            seed=seed,
                            progress_bar=False,
                            decode_latents=True,
                        )
                        img_name = str(idx).zfill(num_digits)
                        img_array = (
                            img[0].clamp(0, 1).cpu().float().numpy().transpose(1, 2, 0) * 255
                        ).astype("uint8")
                        Image.fromarray(img_array).save(
                            os.path.join(output_dir, f"{img_name}.jpg")
                        )
                        generated += 1
                    except Exception:
                        pass

    elapsed = time.time() - start
    rate = generated / elapsed if elapsed > 0 else 0
    if logger:
        logger.info(
            f"Generation complete: {generated} images in {elapsed/60:.1f}m ({rate:.1f} img/s)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation images from dose-response checkpoints"
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Condition to generate from (e.g. C1, C2, C1_seed_137)",
    )
    parser.add_argument("--all", action="store_true", help="Generate for all conditions")
    parser.add_argument(
        "--phase",
        type=str,
        default="auto",
        choices=["phase1", "phase2", "auto"],
        help="Checkpoint phase to use (default: auto = prefer phase2, fall back to phase1)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrained",
        choices=["pretrained", "sft"],
        help="Training stage to generate from (default: pretrained)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (for SFT/RL checkpoints)",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Override Hydra config.yaml path",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    parser.add_argument(
        "--guidance-scale", type=float, default=3.5, help="CFG guidance scale"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    prompt_file = config["generation"]["prompt_file"]
    # Output directory depends on stage
    if args.stage == "pretrained":
        output_base = os.path.join(config["generation"]["output_dir"], "dose_response")
    else:
        output_base = os.path.join(config["generation"]["output_dir"], args.stage)
    base_dir = os.path.dirname(config["training"]["checkpoint_dir"])
    os.makedirs(output_base, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_base, "generate_dose_response.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Load prompts
    prompt_df = pd.read_csv(prompt_file, index_col=0)
    prompts = prompt_df["prompt"].tolist()
    logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # Determine conditions to process
    if args.all:
        conditions = CONDITIONS
    elif args.condition:
        conditions = [args.condition]
    else:
        logger.error("Specify --condition or --all")
        return

    for condition in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing condition: {condition}")
        logger.info(f"{'='*60}")

        output_dir = os.path.join(output_base, condition, "images")
        tar_path = os.path.join(output_base, condition, "images.tar")

        # Skip if tar already exists
        if os.path.exists(tar_path):
            logger.info(f"Skipping {condition}: images.tar already exists")
            continue

        # Find checkpoint
        checkpoint_dir = args.checkpoint_dir
        config_path = args.config_yaml

        if checkpoint_dir is None:
            if args.stage == "sft":
                # Look in checkpoints_sft/
                sft_dir = os.path.join(base_dir, "checkpoints_sft", condition)
                if os.path.isdir(sft_dir):
                    # Check for denoiser.pt first (extracted weights)
                    denoiser_pt = os.path.join(sft_dir, "denoiser.pt")
                    if os.path.exists(denoiser_pt):
                        checkpoint_dir = sft_dir
                    elif os.path.exists(os.path.join(sft_dir, "latest-rank0.pt")):
                        checkpoint_dir = os.path.realpath(os.path.join(sft_dir, "latest-rank0.pt"))
                    else:
                        # Find latest subdirectory with .distcp files
                        for entry in sorted(os.listdir(sft_dir), reverse=True):
                            subdir = os.path.join(sft_dir, entry)
                            if os.path.isdir(subdir) and any(
                                f.endswith(".distcp") for f in os.listdir(subdir)
                            ):
                                checkpoint_dir = subdir
                                break
                if checkpoint_dir:
                    logger.info(f"Using SFT checkpoint: {checkpoint_dir}")
            else:
                # Original pretrained checkpoint discovery
                phases = (
                    ["phase2", "phase1"]
                    if args.phase == "auto"
                    else [args.phase]
                )
                for phase in phases:
                    checkpoint_dir = find_checkpoint_dir(base_dir, condition, phase)
                    config_path_found = find_config_path(base_dir, condition, phase)
                    if checkpoint_dir and config_path_found:
                        config_path = config_path_found
                        logger.info(f"Using {phase} checkpoint: {checkpoint_dir}")
                        break

        # Fall back to finding config from pretrained if not set
        if config_path is None:
            phases = ["phase1", "phase2"]
            for phase in phases:
                config_path = find_config_path(base_dir, condition, phase)
                if config_path:
                    break

        if not checkpoint_dir or not config_path:
            logger.error(
                f"No checkpoint found for {condition} (stage={args.stage}). "
                f"Looked in {base_dir}/checkpoints_*/{condition}/. Skipping."
            )
            continue

        # Load pipeline
        try:
            if os.path.exists(os.path.join(checkpoint_dir, "denoiser.pt")):
                # Extracted denoiser weights (SFT or RL)
                pipeline, image_size = build_pipeline(config_path, args.device, logger)
                state = torch.load(
                    os.path.join(checkpoint_dir, "denoiser.pt"),
                    map_location=args.device,
                )
                mapped = {}
                for k, v in state.items():
                    if k.startswith("denoiser."):
                        mapped[k] = v
                    else:
                        mapped[f"denoiser.{k}"] = v
                missing, unexpected = pipeline.load_state_dict(mapped, strict=False)
                real_missing = [k for k in missing if k.startswith("denoiser.")]
                if real_missing:
                    logger.warning(f"Missing denoiser keys: {real_missing[:10]}")
                pipeline.eval()
                logger.info(f"{args.stage} pipeline loaded from denoiser.pt")
            else:
                pipeline, image_size = load_pipeline(
                    config_path, checkpoint_dir, args.device, logger
                )
        except Exception as e:
            logger.error(f"Failed to load pipeline for {condition}: {e}", exc_info=True)
            continue

        # Generate images
        generate_images(
            pipeline,
            prompts,
            image_size,
            output_dir,
            batch_size=args.batch_size,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            logger=logger,
        )

        # Archive to tar
        num_images = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        logger.info(f"Archiving {condition}: {num_images} images to {tar_path}")
        condition_dir = os.path.join(output_base, condition)
        subprocess.run(
            ["tar", "-cf", tar_path, "-C", condition_dir, "images"], check=True
        )
        subprocess.run(["rm", "-r", output_dir], check=True)
        logger.info(f"{condition} complete: {num_images} images archived")

        # Cleanup GPU memory
        del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
