"""Step 6: Annotate generated images with LlavaGuard.

Runs LlavaGuard-7B on all generated images (6 conditions x 10K + existing
PRX checkpoints) using both POLICY_DEFAULT (binary Safe/Unsafe) and
optionally POLICY_SAFETY_GRANULAR (5-point scale).

Usage:
    python entrypoint_annotate_outputs.py [--models dose_C1 dose_C2 ...] [--granular]
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
import subprocess
import time

from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer
from util.annotation_utils import compress_annotations, save_json_annotations
from util.file_utils import get_file_paths
from util.policy import POLICY_DEFAULT, POLICY_SAFETY_GRANULAR


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def annotate_model_outputs(
    model_id: str,
    images_dir: str,
    output_dir: str,
    server: LlavaGuardServer,
    policy: str,
    batch_size: int,
    logger: logging.Logger,
):
    """Annotate all images in a directory using LlavaGuard."""
    os.makedirs(output_dir, exist_ok=True)

    image_paths, image_names = get_file_paths(images_dir, file_extension=".jpg")

    if not image_paths:
        logger.warning(f"No images found in {images_dir}")
        return

    # Skip already annotated
    existing = set()
    for f in os.listdir(output_dir):
        if f.endswith(".json") or f.endswith(".txt"):
            existing.add(f.rsplit(".", 1)[0])

    inputs = [
        {"image": path, "image_name": name}
        for path, name in zip(image_paths, image_names)
        if name not in existing
    ]

    if not inputs:
        logger.info(f"All {len(image_paths)} images already annotated for {model_id}")
        return

    logger.info(
        f"Annotating {len(inputs)} images for {model_id} "
        f"({len(existing)} already done, {len(image_paths)} total)"
    )

    total_invalid = 0
    start = time.time()

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : min(i + batch_size, len(inputs))]
        batch_idx = i // batch_size + 1
        total_batches = math.ceil(len(inputs) / batch_size)

        logger.info(f"  Batch {batch_idx}/{total_batches} for {model_id}")

        annotations = asyncio.run(
            server.request_async(
                [{"image": inp["image"], "prompt": policy} for inp in batch]
            )
        )

        invalid = save_json_annotations(
            annotations, output_dir, [inp["image_name"] for inp in batch]
        )

        if invalid:
            total_invalid += len(invalid)
            logger.warning(f"  Invalid JSON in batch: {len(invalid)} items")

    elapsed = time.time() - start
    logger.info(
        f"  {model_id}: {len(inputs)} annotations in {elapsed / 60:.1f}m "
        f"({total_invalid} invalid)"
    )

    return total_invalid


def main():
    parser = argparse.ArgumentParser(description="Annotate generated images with LlavaGuard")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific model IDs to annotate (default: all found)")
    parser.add_argument("--granular", action="store_true",
                        help="Also annotate with POLICY_SAFETY_GRANULAR (5-point scale)")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=10001)
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    generated_dir = config["generation"]["output_dir"]
    base_output_dir = config["evaluation"]["output_dir"]
    os.makedirs(base_output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(base_output_dir, "annotate_outputs.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # Discover models to annotate
    if args.models:
        model_ids = args.models
    else:
        model_ids = sorted([
            d for d in os.listdir(generated_dir)
            if os.path.isdir(os.path.join(generated_dir, d))
            or os.path.exists(os.path.join(generated_dir, d, "images.tar"))
        ])

        # Also discover multi-seed subdirectories:
        # {generated_dir}/dose_response/{condition}/seed_*/
        dose_response_dir = os.path.join(generated_dir, "dose_response")
        if os.path.isdir(dose_response_dir):
            for condition_name in sorted(os.listdir(dose_response_dir)):
                condition_path = os.path.join(dose_response_dir, condition_name)
                if not os.path.isdir(condition_path):
                    continue
                for entry in sorted(os.listdir(condition_path)):
                    if not entry.startswith("seed_"):
                        continue
                    seed_path = os.path.join(condition_path, entry)
                    if os.path.isdir(seed_path) and (
                        os.path.exists(os.path.join(seed_path, "images.tar"))
                        or os.path.isdir(os.path.join(seed_path, "images"))
                    ):
                        seed_num = entry.removeprefix("seed_")
                        seed_model_id = f"dose_{condition_name}_seed_{seed_num}"
                        if seed_model_id not in model_ids:
                            model_ids.append(seed_model_id)
        model_ids.sort()

    if not model_ids:
        logger.error(f"No models found in {generated_dir}")
        return

    logger.info(f"Found {len(model_ids)} models to annotate: {model_ids}")

    # Unpack tar archives where needed
    unpacked_dirs = {}
    for model_id in model_ids:
        # Resolve model_id to filesystem path
        # Seed model IDs like "dose_C1_seed_137" map to
        # {generated_dir}/dose_response/C1/seed_137/
        seed_match = re.match(r"dose_(.+)_seed_(\d+)$", model_id)
        if seed_match:
            condition = seed_match.group(1)
            seed_num = seed_match.group(2)
            model_dir = os.path.join(
                generated_dir, "dose_response", condition, f"seed_{seed_num}"
            )
        else:
            model_dir = os.path.join(generated_dir, model_id)

        images_dir = os.path.join(model_dir, "images")
        images_tar = os.path.join(model_dir, "images.tar")

        if os.path.isdir(images_dir) and os.listdir(images_dir):
            unpacked_dirs[model_id] = images_dir
        elif os.path.exists(images_tar):
            logger.info(f"Unpacking {images_tar}...")
            subprocess.run(
                ["tar", "-xf", images_tar, "-C", model_dir],
                check=True,
            )
            unpacked_dirs[model_id] = images_dir
        else:
            logger.warning(f"No images found for {model_id}, skipping")

    # Initialize LlavaGuard server
    server = LlavaGuardServer()
    server.setUpClass(
        model=config["llavaguard"]["model"],
        dp_size=args.dp_size,
        port=args.port,
        is_requests_wrapper=True,
    )
    logger.info("LlavaGuard server ready.")

    policies = [("default", POLICY_DEFAULT)]
    if args.granular:
        policies.append(("granular", POLICY_SAFETY_GRANULAR))

    try:
        for policy_name, policy_text in policies:
            logger.info(f"Running annotations with policy: {policy_name}")

            for model_id, images_dir in unpacked_dirs.items():
                output_dir = os.path.join(
                    base_output_dir, model_id, f"annotations_{policy_name}"
                )
                annotate_model_outputs(
                    model_id=model_id,
                    images_dir=images_dir,
                    output_dir=output_dir,
                    server=server,
                    policy=policy_text,
                    batch_size=args.batch_size,
                    logger=logger,
                )

                # Compress to Parquet
                parquet_dir = os.path.join(base_output_dir, model_id)
                compress_annotations(
                    annotation_dir=output_dir,
                    output_dir=parquet_dir,
                    parquet_size=100_000,
                )

                # Archive and clean up annotation JSONs
                annotations_parent = os.path.dirname(output_dir)
                annotations_name = os.path.basename(output_dir)
                subprocess.run(
                    ["tar", "-cf",
                     os.path.join(annotations_parent, f"{annotations_name}.tar"),
                     "-C", annotations_parent, annotations_name],
                    check=True,
                )
                subprocess.run(["rm", "-r", output_dir], check=True)

    except Exception as e:
        logger.error(f"Error during annotation: {e}", exc_info=True)
    finally:
        logger.info("Shutting down LlavaGuard server...")
        server.tearDownClass()

    # Clean up unpacked images
    logger.info("Cleaning up unpacked image directories...")
    for model_id, images_dir in unpacked_dirs.items():
        if os.path.isdir(images_dir):
            subprocess.run(["rm", "-r", images_dir], check=True)

    logger.info("Annotation complete.")


if __name__ == "__main__":
    main()
