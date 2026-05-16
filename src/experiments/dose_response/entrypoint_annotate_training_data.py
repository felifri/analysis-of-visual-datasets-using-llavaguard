"""Step 2: Annotate all training images for safety using LlavaGuard-7B.

Reads images from downloaded HuggingFace Parquet datasets, sends them through
the LlavaGuard SGLang server for binary Safe/Unsafe classification, and
saves annotations as JSON then compresses to Parquet.

Handles three dataset formats:
  - lehduong/flux_generated: image as dict {'bytes': ..., 'path': ...}
  - LucasFang/FLUX-Reason-6M: image as dict {'bytes': ..., 'path': ...}
  - brivangl/midjourney-v6-llava: image as raw bytes

Usage:
    python entrypoint_annotate_training_data.py [--download-dir /path] [--output-dir /path]
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import traceback
from glob import glob
from random import uniform

import openai
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm

# Add project src to path for utility imports
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from util.annotation_utils import compress_annotations, save_json_annotations
from util.policy import POLICY_DEFAULT


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset configurations: how to extract image bytes and build unique IDs
DATASET_CONFIGS = {
    "lehduong__flux_generated": {
        "data_glob": "data/train-*.parquet",
        "image_col": "image",
        "image_format": "dict",  # {'bytes': ..., 'path': ...}
        "id_col": None,  # use parquet filename + row index
    },
    "LucasFang__FLUX-Reason-6M": {
        "data_glob": "**/fluxdb-*.parquet",
        "image_col": "image",
        "image_format": "dict",
        "id_col": "id",
    },
    "Photoroom__midjourney-v6-recap": {
        "data_glob": "train_*.parquet",
        "image_col": "image",
        "image_format": "dict",  # raw bytes
        "id_col": "id",
    },
}


def encode_image(p) -> str:
    """Encode a PIL image to base64 JPEG string."""
    if isinstance(p, (str, os.PathLike)):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif hasattr(p, "save"):  # PIL image
        buf = io.BytesIO()
        if p.mode in ("RGBA", "P"):
            p = p.convert("RGB")
        p.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    raise ValueError("Unsupported image format")


async def request_llavaguard_async(
    inputs: list[dict],
    base_url: str,
    api_key: str = "sk-123456",
    retries: int = 3,
    timeout: int = 300,
) -> list[str]:
    """Send images to LlavaGuard server via OpenAI-compatible API."""
    async with openai.AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        hyperparams = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 500}

        async def fetch(inp, attempt=1):
            b64 = encode_image(inp["image"])
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="default",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": inp["prompt"]},
                            ],
                        }],
                        **hyperparams,
                    ),
                    timeout=timeout,
                )
                return resp.choices[0].message.content.strip()
            except (openai.APITimeoutError, asyncio.TimeoutError):
                if attempt <= retries:
                    wait = uniform(2, 5) * attempt
                    await asyncio.sleep(wait)
                    return await fetch(inp, attempt + 1)
                return ""
            except Exception:
                return f"Unexpected error:\n{traceback.format_exc()}"

        results = await tqdm.gather(*[fetch(inp) for inp in inputs])
        return [r for r in results if r is not None]


def extract_pil_image(image_data, image_format: str) -> Image.Image | None:
    """Extract a PIL Image from the parquet image column value."""
    try:
        if image_format == "dict":
            raw = image_data["bytes"]
        else:
            raw = image_data

        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def find_parquet_files(download_dir: str) -> list[tuple[str, str, dict]]:
    """Find all parquet files and their dataset config.

    Returns list of (parquet_path, dataset_name, config).
    """
    results = []
    for dataset_name, cfg in DATASET_CONFIGS.items():
        dataset_dir = os.path.join(download_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        pattern = os.path.join(dataset_dir, cfg["data_glob"])
        files = sorted(glob(pattern, recursive=True))
        for f in files:
            results.append((f, dataset_name, cfg))
    return results


def get_already_annotated(output_dir: str) -> set[str]:
    """Get the set of image IDs that already have annotations."""
    annotated = set()
    if not os.path.exists(output_dir):
        return annotated
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".json"):
                annotated.add(f.removesuffix(".json"))
            elif f.endswith(".txt"):
                annotated.add(f.removesuffix(".txt"))
    return annotated


def build_unique_id(dataset_name: str, parquet_basename: str, row_idx: int, row_id: str | None) -> str:
    """Build a globally unique image ID."""
    shard_name = parquet_basename.removesuffix(".parquet")
    if row_id is not None:
        return f"{dataset_name}__{shard_name}__{row_id}"
    return f"{dataset_name}__{shard_name}__{row_idx:06d}"


def main():
    parser = argparse.ArgumentParser(description="Annotate training images for safety with LlavaGuard")
    parser.add_argument("--download-dir", default=None, help="Directory with downloaded datasets")
    parser.add_argument("--output-dir", default=None, help="Directory for annotations output")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inference")
    parser.add_argument("--dp-size", type=int, default=4, help="Data parallel size for SGLang")
    parser.add_argument("--port", type=int, default=10001, help="SGLang server port")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards")
    parser.add_argument("--shard-id", type=int, default=0, help="This shard's ID (0-indexed)")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    download_dir = args.download_dir or config["training_data"]["download_dir"]
    output_dir = args.output_dir or config["training_data"]["annotations_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "annotate_training_data.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    parquet_files = find_parquet_files(download_dir)
    logger.info(f"Found {len(parquet_files)} parquet files across datasets in {download_dir}")

    if not parquet_files:
        logger.error("No parquet files found. Check download_dir and dataset directory names.")
        return

    # Shard the parquet files across parallel jobs
    if args.num_shards > 1:
        parquet_files = [f for i, f in enumerate(parquet_files) if i % args.num_shards == args.shard_id]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: processing {len(parquet_files)} parquet files")

    # Log per-dataset counts
    dataset_counts = {}
    for _, ds_name, _ in parquet_files:
        dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1
    for ds_name, count in sorted(dataset_counts.items()):
        logger.info(f"  {ds_name}: {count} parquet files")

    already_annotated = get_already_annotated(output_dir)
    logger.info(f"Found {len(already_annotated)} existing annotations, will skip those")

    # Build server URL (server is launched externally via SLURM script)
    base_url = f"http://127.0.0.1:{args.port}/v1"
    logger.info(f"Connecting to LlavaGuard server at {base_url}")

    total_annotated = len(already_annotated)
    total_invalid = 0
    total_skipped = 0
    batch_images = []
    batch_names = []
    start_time = time.time()

    try:
        for pq_idx, (pq_path, dataset_name, cfg) in enumerate(parquet_files):
            pq_basename = os.path.basename(pq_path)
            logger.info(f"Processing parquet {pq_idx + 1}/{len(parquet_files)}: {dataset_name}/{pq_basename}")

            # Read only the columns we need to minimize memory
            columns = [cfg["image_col"]]
            if cfg["id_col"]:
                columns.append(cfg["id_col"])

            try:
                df = pd.read_parquet(pq_path, columns=columns)
            except Exception as e:
                logger.error(f"Failed to read {pq_path}: {e}")
                continue

            for row_idx, row in df.iterrows():
                row_id = str(row[cfg["id_col"]]) if cfg["id_col"] else None
                unique_id = build_unique_id(dataset_name, pq_basename, row_idx, row_id)

                if unique_id in already_annotated:
                    total_skipped += 1
                    continue

                img = extract_pil_image(row[cfg["image_col"]], cfg["image_format"])
                if img is None:
                    total_invalid += 1
                    continue

                batch_images.append({"image": img, "prompt": POLICY_DEFAULT})
                batch_names.append(unique_id)

                if len(batch_images) >= args.batch_size:
                    annotations = asyncio.run(request_llavaguard_async(batch_images, base_url))
                    invalid = save_json_annotations(annotations, output_dir, batch_names)
                    total_invalid += len(invalid)
                    total_annotated += len(batch_images)

                    if invalid:
                        logger.warning(f"Invalid JSON in batch: {len(invalid)} items")

                    elapsed = time.time() - start_time
                    rate = (total_annotated - len(already_annotated)) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Annotated {total_annotated:,} images "
                        f"({rate:.1f}/s, {total_invalid} invalid, {total_skipped:,} skipped)"
                    )

                    batch_images = []
                    batch_names = []

            # Free memory after each parquet file
            del df

        # Process remaining batch
        if batch_images:
            annotations = asyncio.run(request_llavaguard_async(batch_images, base_url))
            invalid = save_json_annotations(annotations, output_dir, batch_names)
            total_invalid += len(invalid)
            total_annotated += len(batch_images)

    except Exception as e:
        logger.error(f"Error during annotation: {e}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info(
        f"Annotation complete: {total_annotated:,} images in {elapsed / 3600:.1f}h "
        f"({total_invalid} invalid, {total_skipped:,} skipped)"
    )

    # Compress annotations to Parquet
    logger.info("Compressing annotations to Parquet...")
    parquet_dir = os.path.join(os.path.dirname(output_dir), "annotations_parquet")
    os.makedirs(parquet_dir, exist_ok=True)
    compress_annotations(
        annotation_dir=output_dir,
        output_dir=parquet_dir,
        parquet_size=100_000,
    )

    # Write summary statistics
    summary = {"total_annotated": total_annotated, "total_invalid": total_invalid}
    safe_count = 0
    unsafe_count = 0
    category_counts = {}

    for root, _, files in os.walk(output_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, fname)) as f:
                    data = json.load(f)
                rating = data.get("rating", "")
                category = data.get("category", "")
                if rating == "Safe":
                    safe_count += 1
                elif rating == "Unsafe":
                    unsafe_count += 1
                category_counts[category] = category_counts.get(category, 0) + 1
            except (json.JSONDecodeError, KeyError):
                pass

    summary["safe_count"] = safe_count
    summary["unsafe_count"] = unsafe_count
    summary["unsafe_fraction"] = unsafe_count / max(1, safe_count + unsafe_count)
    summary["category_counts"] = category_counts

    summary_path = os.path.join(os.path.dirname(output_dir), "training_data_safety_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary: {safe_count:,} safe, {unsafe_count:,} unsafe "
                f"({summary['unsafe_fraction']:.2%} unsafe)")
    logger.info(f"Category breakdown: {json.dumps(category_counts, indent=2)}")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
