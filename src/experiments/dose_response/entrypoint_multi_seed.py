"""Generate 10K evaluation images from dose-response checkpoints with multiple seeds.

Loads each condition's model checkpoint once into GPU memory and generates
images for each specified seed. This measures run-to-run variance from
diffusion stochasticity while keeping the trained model fixed.

Existing seed=42 results are not regenerated — use this script only for
additional seeds.

Usage:
    # Generate seeds 137, 314, 789, 1331 for C1 and C2
    python entrypoint_multi_seed.py --conditions C1 C2 --seeds 137 314 789 1331

    # Smoke test: one condition, one seed, small batch
    python entrypoint_multi_seed.py --conditions C1 --seeds 137 --batch-size 2

    # Override checkpoint phase
    python entrypoint_multi_seed.py --conditions C1 --seeds 137 --phase phase2
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import pandas as pd
import torch

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

# Import generation utilities from the existing entrypoint
from entrypoint_generate_dose_response import (
    CONDITIONS,
    build_pipeline,
    find_checkpoint_dir,
    find_config_path,
    generate_images,
    load_denoiser_state_dict,
    load_pipeline,
)

DEFAULT_SEEDS = [137, 314, 789, 1331]


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation images with multiple diffusion seeds"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="Conditions to generate for (e.g. C1 C2 C1_seed_137)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Diffusion seeds to use (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="auto",
        choices=["phase1", "phase2", "auto"],
        help="Checkpoint phase to use (default: auto = prefer phase2)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrained",
        choices=["pretrained", "sft"],
        help="Training stage to generate from (default: pretrained)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    parser.add_argument(
        "--guidance-scale", type=float, default=3.5, help="CFG guidance scale"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    prompt_file = config["generation"]["prompt_file"]
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
            logging.FileHandler(os.path.join(output_base, "generate_multi_seed.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Load prompts
    prompt_df = pd.read_csv(prompt_file, index_col=0)
    prompts = prompt_df["prompt"].tolist()
    logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")
    logger.info(f"Conditions: {args.conditions}, Seeds: {args.seeds}")

    for condition in args.conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model for condition: {condition}")
        logger.info(f"{'='*60}")

        # Find checkpoint and config
        checkpoint_dir = None
        config_path = None

        if args.stage == "sft":
            sft_dir = os.path.join(base_dir, "checkpoints_sft", condition)
            if os.path.isdir(sft_dir):
                denoiser_pt = os.path.join(sft_dir, "denoiser.pt")
                if os.path.exists(denoiser_pt):
                    checkpoint_dir = sft_dir
                elif os.path.exists(os.path.join(sft_dir, "latest-rank0.pt")):
                    checkpoint_dir = os.path.realpath(
                        os.path.join(sft_dir, "latest-rank0.pt")
                    )
                else:
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
            phases = (
                ["phase2", "phase1"] if args.phase == "auto" else [args.phase]
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
            for phase in ["phase1", "phase2"]:
                config_path = find_config_path(base_dir, condition, phase)
                if config_path:
                    break

        if not checkpoint_dir or not config_path:
            logger.error(
                f"No checkpoint found for {condition} (stage={args.stage}). "
                f"Looked in {base_dir}/checkpoints_*/{condition}/. Skipping."
            )
            continue

        # Load pipeline once for this condition
        try:
            if os.path.exists(os.path.join(checkpoint_dir, "denoiser.pt")):
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
            logger.error(
                f"Failed to load pipeline for {condition}: {e}", exc_info=True
            )
            continue

        # Generate images for each seed
        for seed in args.seeds:
            seed_dir = os.path.join(output_base, condition, f"seed_{seed}")
            output_dir = os.path.join(seed_dir, "images")
            tar_path = os.path.join(seed_dir, "images.tar")

            if os.path.exists(tar_path):
                logger.info(f"Skipping {condition}/seed_{seed}: images.tar already exists")
                continue

            logger.info(f"\n--- Generating {condition} seed={seed} ---")

            generate_images(
                pipeline,
                prompts,
                image_size,
                output_dir,
                batch_size=args.batch_size,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=seed,
                logger=logger,
            )

            # Archive to tar and clean up raw images
            num_images = len(
                [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
            )
            logger.info(
                f"Archiving {condition}/seed_{seed}: {num_images} images to {tar_path}"
            )
            subprocess.run(
                ["tar", "-cf", tar_path, "-C", seed_dir, "images"], check=True
            )
            subprocess.run(["rm", "-r", output_dir], check=True)
            logger.info(f"{condition}/seed_{seed} complete: {num_images} images archived")

        # Free GPU memory before loading next condition
        del pipeline
        torch.cuda.empty_cache()
        logger.info(f"Released GPU memory for {condition}")

    logger.info("Multi-seed generation complete.")


if __name__ == "__main__":
    main()
