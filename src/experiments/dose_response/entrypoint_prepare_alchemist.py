"""Prepare Alchemist dataset for SFT: download images and convert to AR-bucketed MDS.

Downloads 3,350 images from URLs in the Alchemist CSV, then converts them to
the same AR-bucketed MDS format used by the dose-response pretraining pools.

Usage:
    python entrypoint_prepare_alchemist.py
    python entrypoint_prepare_alchemist.py --skip-download   # only MDS conversion
    python entrypoint_prepare_alchemist.py --skip-mds        # only download
    python entrypoint_prepare_alchemist.py --workers 16      # parallel downloads
"""

import argparse
import io
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image

try:
    from streaming import MDSWriter
except ImportError:
    MDSWriter = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MDS schema matching PRX's fine-t2i-to-mds.py and entrypoint_construct_subsets.py
MDS_COLUMNS: dict[str, str] = {
    "image": "jpeg",
    "width": "int",
    "height": "int",
    "original_width": "int",
    "original_height": "int",
    "id": "str",
    "prompt": "str",
    "enhanced_prompt": "str",
    "aesthetic_predictor_v_2_5_score": "str",
    "image_generator": "str",
    "prompt_generator": "str",
    "prompt_category": "str",
    "style": "str",
    "task": "str",
    "enhancer": "str",
    "image_aspect_ratio": "str",
    "image_generated_with_enhanced_prompt": "str",
    "length": "int",
    "enhanced_length": "int",
    "subset": "str",
}

JPEG_QUALITY = 95
SHARD_SIZE = 1 << 27  # 128 MB
DOWNLOAD_TIMEOUT = 30


# ---------------------------------------------------------------------------
# AR bucketing (same as PRX fine-t2i-to-mds.py and entrypoint_construct_subsets.py)
# ---------------------------------------------------------------------------

def build_ar_to_size(
    base: int = 1024, patch: int = 32,
    min_ar: float = 0.5, max_ar: float = 2.0, div: int = 16,
) -> dict[float, tuple[int, int]]:
    bp = base / patch
    sizes: set[tuple[int, int]] = set()
    for pw in range(
        math.ceil(math.sqrt(bp**2 * min_ar)),
        math.floor(math.sqrt(bp**2 * max_ar)) + 1,
    ):
        ph = math.floor(bp**2 / pw)
        w, h = pw * patch, ph * patch
        if w % div == 0 and h % div == 0:
            sizes.add((w, h))
    for ph in range(
        math.ceil(math.sqrt(bp**2 / max_ar)),
        math.floor(math.sqrt(bp**2 / min_ar)) + 1,
    ):
        pw = math.floor(bp**2 / ph)
        w, h = pw * patch, ph * patch
        if w % div == 0 and h % div == 0:
            sizes.add((w, h))
    return {w / h: (w, h) for w, h in sizes}


AR_TO_SIZE = build_ar_to_size()
AR_KEYS = tuple(sorted(AR_TO_SIZE.keys()))


def closest_bucket(w: int, h: int) -> tuple[float, int, int]:
    ar = w / h
    best = min(AR_KEYS, key=lambda x: abs(x - ar))
    tw, th = AR_TO_SIZE[best]
    return best, tw, th


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def download_image(url: str, output_path: str) -> bool:
    """Download a single image from URL. Returns True on success."""
    if os.path.exists(output_path):
        return True
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        # Validate it's actually an image
        img_bytes = resp.content
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception:
        return False


def download_all_images(
    df: pd.DataFrame, output_dir: str, workers: int, logger: logging.Logger,
) -> int:
    """Download all images from the Alchemist CSV. Returns count of successful downloads."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for _, row in df.iterrows():
        img_key = str(row["img_key"])
        url = str(row["url"])
        # Determine file extension from URL
        ext = url.rsplit(".", 1)[-1].lower() if "." in url.rsplit("/", 1)[-1] else "jpg"
        if ext not in ("jpg", "jpeg", "png", "webp", "bmp", "gif"):
            ext = "jpg"
        output_path = os.path.join(output_dir, f"{img_key}.{ext}")
        tasks.append((url, output_path, img_key))

    # Skip already downloaded
    existing = set(os.listdir(output_dir))
    todo = [(url, path, key) for url, path, key in tasks if os.path.basename(path) not in existing]

    logger.info(f"Download: {len(todo)} remaining ({len(tasks) - len(todo)} already exist)")

    if not todo:
        return len(tasks)

    success = len(tasks) - len(todo)
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, url, path): key
            for url, path, key in todo
        }
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1
            done = success + failed - (len(tasks) - len(todo))
            if done % 100 == 0:
                elapsed = time.time() - start
                logger.info(
                    f"  Downloaded {done}/{len(todo)} "
                    f"({failed} failed, {done / max(1, elapsed):.0f}/s)"
                )

    elapsed = time.time() - start
    logger.info(
        f"Download complete: {success} success, {failed} failed in {elapsed / 60:.1f}m"
    )
    return success


# ---------------------------------------------------------------------------
# MDS conversion
# ---------------------------------------------------------------------------

def convert_to_mds(
    df: pd.DataFrame,
    images_dir: str,
    output_dir: str,
    logger: logging.Logger,
) -> int:
    """Convert downloaded images to AR-bucketed MDS format."""
    if MDSWriter is None:
        raise ImportError("mosaicml-streaming required for MDS conversion")

    os.makedirs(output_dir, exist_ok=True)

    writers: dict[str, MDSWriter] = {}
    n_written = 0
    n_failed = 0
    start = time.time()

    for _, row in df.iterrows():
        img_key = str(row["img_key"])
        prompt = str(row["prompt"])
        url = str(row["url"])

        # Find the downloaded image
        ext = url.rsplit(".", 1)[-1].lower() if "." in url.rsplit("/", 1)[-1] else "jpg"
        if ext not in ("jpg", "jpeg", "png", "webp", "bmp", "gif"):
            ext = "jpg"
        img_path = os.path.join(images_dir, f"{img_key}.{ext}")

        if not os.path.exists(img_path):
            n_failed += 1
            continue

        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            orig_w, orig_h = img.size
            ar, tw, th = closest_bucket(orig_w, orig_h)
            ar_str = f"{ar:.3f}"

            img = img.resize((tw, th), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            buf.seek(0)
            jpeg_pil = Image.open(buf)
            jpeg_pil.load()

            sample = {
                "image": jpeg_pil,
                "width": tw,
                "height": th,
                "original_width": orig_w,
                "original_height": orig_h,
                "id": img_key,
                "prompt": prompt,
                "enhanced_prompt": "",
                "aesthetic_predictor_v_2_5_score": "",
                "image_generator": "alchemist",
                "prompt_generator": "",
                "prompt_category": "",
                "style": "",
                "task": "",
                "enhancer": "",
                "image_aspect_ratio": ar_str,
                "image_generated_with_enhanced_prompt": "",
                "length": len(prompt),
                "enhanced_length": 0,
                "subset": "alchemist",
            }

            if ar_str not in writers:
                out = os.path.join(output_dir, ar_str)
                os.makedirs(out, exist_ok=True)
                writers[ar_str] = MDSWriter(
                    out=out, columns=MDS_COLUMNS, size_limit=SHARD_SIZE,
                )
            writers[ar_str].write(sample)
            n_written += 1

        except Exception as e:
            n_failed += 1
            if n_failed <= 10:
                logger.warning(f"Failed to process {img_key}: {e}")

        if n_written > 0 and n_written % 500 == 0:
            elapsed = time.time() - start
            logger.info(
                f"  MDS: {n_written} written, {n_failed} failed, "
                f"{n_written / max(1, elapsed):.0f}/s"
            )

    for w in writers.values():
        w.finish()

    elapsed = time.time() - start
    logger.info(
        f"MDS conversion complete: {n_written} written, {n_failed} failed "
        f"in {elapsed / 60:.1f}m across {len(writers)} AR buckets"
    )
    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Alchemist dataset: download images and convert to MDS"
    )
    parser.add_argument("--csv", default=None, help="Path to alchemist CSV")
    parser.add_argument("--images-dir", default=None, help="Directory to store downloaded images")
    parser.add_argument("--mds-dir", default=None, help="Output MDS directory")
    parser.add_argument("--workers", type=int, default=32, help="Download worker threads")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-mds", action="store_true", help="Skip MDS conversion step")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    download_dir = config["training_data"]["download_dir"]
    mds_base = config["training_data"]["mds_dir"]

    csv_path = args.csv or os.path.join(
        download_dir, "yandex__alchemist", "data", "alchemist_3k_final.csv"
    )
    images_dir = args.images_dir or os.path.join(
        download_dir, "yandex__alchemist", "images"
    )
    mds_dir = args.mds_dir or os.path.join(mds_base, "alchemist")

    log_dir = config["base_output_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "prepare_alchemist.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Load CSV
    logger.info(f"Loading Alchemist CSV: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    logger.info(f"Loaded {len(df)} rows (columns: {list(df.columns)})")

    # Step 1: Download images
    if not args.skip_download:
        logger.info(f"Downloading images to {images_dir}")
        download_all_images(df, images_dir, args.workers, logger)
    else:
        logger.info("Skipping download (--skip-download)")

    # Step 2: Convert to MDS
    if not args.skip_mds:
        logger.info(f"Converting to MDS at {mds_dir}")
        convert_to_mds(df, images_dir, mds_dir, logger)
    else:
        logger.info("Skipping MDS conversion (--skip-mds)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
