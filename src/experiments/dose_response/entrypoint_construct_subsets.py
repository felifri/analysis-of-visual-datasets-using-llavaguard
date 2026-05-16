"""Step 3: Construct experimental subsets for each condition (C1-C5).

Builds shared MDS pools to avoid duplicating images across conditions:
  - safe_full:   All safe images (shared by C1, C2, C3, C0)
  - unsafe_full: All unsafe images (shared by C0, C6)
  - unsafe_c2:   Oversampled unsafe for C2's 5% target
  - unsafe_c3:   Oversampled unsafe for C3's 10% target
  - unsafe_c4:   Subsampled unsafe for C4's 1.21% at 1M
  - safe_c4:     Subsampled safe for C4's 1M scale
  - safe_c6:     Subsampled safe for C6's 1M scale

Each condition's dataset YAML points to the appropriate pools:
  C1: [safe_full]
  C2: [safe_full, unsafe_c2]   (safe_full provides ~7.85M, unsafe_c2 ~397K → total ~8.24M)
  C3: [safe_full, unsafe_c3]   (safe_full provides ~7.85M, unsafe_c3 ~872K → total ~8.72M)
  C0: [safe_full, unsafe_full]
  C4: [safe_c4, unsafe_c4]
  C6: [safe_c6, unsafe_full]

Usage:
    python entrypoint_construct_subsets.py --build-pools [--skip-mds]
    python entrypoint_construct_subsets.py --build-pool safe_full
"""

import argparse
import io
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from glob import glob
from typing import Any

import pandas as pd
from PIL import Image

try:
    from streaming import MDSWriter
except ImportError:
    MDSWriter = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MDS schema matching PRX's fine-t2i-to-mds.py
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

# Dataset configurations matching entrypoint_annotate_training_data.py
DATASET_CONFIGS = {
    "lehduong__flux_generated": {
        "data_glob": "data/train-*.parquet",
        "image_col": "image",
        "image_format": "dict",
        "id_col": None,
        "caption_col": "caption_caption",
    },
    "LucasFang__FLUX-Reason-6M": {
        "data_glob": "**/fluxdb-*.parquet",
        "image_col": "image",
        "image_format": "dict",
        "id_col": "id",
        "caption_col": "caption_detail",
    },
    "Photoroom__midjourney-v6-recap": {
        "data_glob": "train_*.parquet",
        "image_col": "image",
        "image_format": "dict",
        "id_col": "id",
        "caption_col": "gemini",
    },
}


# ---------------------------------------------------------------------------
# AR bucketing (same logic as PRX fine-t2i-to-mds.py)
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
# Annotation loading
# ---------------------------------------------------------------------------

def load_annotations_from_parquet(parquet_dir: str) -> pd.DataFrame:
    parquet_files = sorted(glob(os.path.join(parquet_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=False)


# ---------------------------------------------------------------------------
# Pool definitions
# ---------------------------------------------------------------------------

def define_pools(
    safe_ids: list[str],
    unsafe_ids: list[str],
    unsafe_by_category: dict[str, list[str]],
    rng: random.Random,
    logger: logging.Logger,
) -> dict[str, dict]:
    """Define the shared MDS pools and which IDs go into each."""
    total_size = len(safe_ids) + len(unsafe_ids)
    unsafe_fraction = len(unsafe_ids) / max(1, total_size)

    # C2 needs 5% unsafe at full scale
    c2_unsafe_count = int(total_size * 0.05)

    # C3 needs 10% unsafe at full scale
    c3_unsafe_count = int(total_size * 0.10)

    # C4 needs same proportion as original at 1M
    c4_unsafe_count = int(1_000_000 * unsafe_fraction)
    c4_safe_count = 1_000_000 - c4_unsafe_count

    # C6 needs all unsafe in 1M
    c6_safe_count = 1_000_000 - len(unsafe_ids)

    # Sample C2 oversampled unsafe (stratified across categories)
    c2_unsafe_selected = list(unsafe_ids)  # start with all
    remaining = c2_unsafe_count - len(unsafe_ids)
    logger.info(f"C2 unsafe oversampling: {remaining:,} additional from {len(unsafe_ids):,} pool")
    categories = sorted(unsafe_by_category.keys())
    per_cat = remaining // len(categories)
    remainder_n = remaining % len(categories)
    for i, cat in enumerate(categories):
        cat_ids = unsafe_by_category[cat]
        n = per_cat + (1 if i < remainder_n else 0)
        c2_unsafe_selected.extend(rng.choices(cat_ids, k=n))

    # Sample C3 oversampled unsafe (stratified across categories, 10% target)
    c3_unsafe_selected = list(unsafe_ids)  # start with all
    remaining = c3_unsafe_count - len(unsafe_ids)
    logger.info(f"C3 unsafe oversampling: {remaining:,} additional from {len(unsafe_ids):,} pool")
    per_cat = remaining // len(categories)
    remainder_n = remaining % len(categories)
    for i, cat in enumerate(categories):
        cat_ids = unsafe_by_category[cat]
        n = per_cat + (1 if i < remainder_n else 0)
        c3_unsafe_selected.extend(rng.choices(cat_ids, k=n))

    # Sample C4 unsafe (stratified subsample)
    c4_unsafe_selected = []
    per_cat = c4_unsafe_count // len(categories)
    remainder_n = c4_unsafe_count % len(categories)
    for i, cat in enumerate(categories):
        cat_ids = unsafe_by_category[cat]
        n = per_cat + (1 if i < remainder_n else 0)
        if n <= len(cat_ids):
            c4_unsafe_selected.extend(rng.sample(cat_ids, n))
        else:
            c4_unsafe_selected.extend(rng.choices(cat_ids, k=n))

    # Sample C4 safe (random subsample)
    c4_safe_selected = rng.sample(safe_ids, c4_safe_count)

    # Sample C6 safe (random subsample)
    c6_safe_selected = rng.sample(safe_ids, c6_safe_count)

    pools = {
        "safe_full": {
            "ids": safe_ids,
            "description": f"All safe images ({len(safe_ids):,})",
            "used_by": ["C1", "C2", "C3", "C0"],
        },
        "unsafe_full": {
            "ids": unsafe_ids,
            "description": f"All unsafe images ({len(unsafe_ids):,})",
            "used_by": ["C0", "C6"],
        },
        "unsafe_c2": {
            "ids": c2_unsafe_selected,
            "description": f"Oversampled unsafe for C2 5% ({len(c2_unsafe_selected):,}, {len(c2_unsafe_selected)/len(unsafe_ids):.1f}x)",
            "used_by": ["C2"],
        },
        "unsafe_c3": {
            "ids": c3_unsafe_selected,
            "description": f"Oversampled unsafe for C3 10% ({len(c3_unsafe_selected):,}, {len(c3_unsafe_selected)/len(unsafe_ids):.1f}x)",
            "used_by": ["C3"],
        },
        "unsafe_c4": {
            "ids": c4_unsafe_selected,
            "description": f"Subsampled unsafe for C4 ({len(c4_unsafe_selected):,})",
            "used_by": ["C4"],
        },
        "safe_c4": {
            "ids": c4_safe_selected,
            "description": f"Subsampled safe for C4 ({len(c4_safe_selected):,})",
            "used_by": ["C4"],
        },
        "safe_c6": {
            "ids": c6_safe_selected,
            "description": f"Subsampled safe for C6 ({len(c6_safe_selected):,})",
            "used_by": ["C6"],
        },
    }

    # Log pool definitions
    for name, pool in pools.items():
        unique = len(set(pool["ids"]))
        logger.info(f"  {name}: {len(pool['ids']):,} images ({unique:,} unique) — {pool['description']}")

    # Log condition compositions
    logger.info("Condition compositions:")
    compositions = {
        "C1": (["safe_full"], len(safe_ids), 0),
        "C2": (["safe_full", "unsafe_c2"], len(safe_ids), len(c2_unsafe_selected)),
        "C3": (["safe_full", "unsafe_c3"], len(safe_ids), len(c3_unsafe_selected)),
        "C0": (["safe_full", "unsafe_full"], len(safe_ids), len(unsafe_ids)),
        "C4": (["safe_c4", "unsafe_c4"], len(c4_safe_selected), len(c4_unsafe_selected)),
        "C6": (["safe_c6", "unsafe_full"], len(c6_safe_selected), len(unsafe_ids)),
    }
    for cid, (pool_names, n_safe, n_unsafe) in compositions.items():
        total = n_safe + n_unsafe
        frac = n_unsafe / max(1, total) * 100
        logger.info(f"  {cid}: {pool_names} → {total:,} total ({frac:.2f}% unsafe)")

    return pools


# ---------------------------------------------------------------------------
# Parquet image reading
# ---------------------------------------------------------------------------

def extract_pil_image(image_data, image_format: str) -> Image.Image | None:
    try:
        raw = image_data["bytes"] if image_format == "dict" else image_data
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def build_unique_id(dataset_name: str, parquet_basename: str, row_idx: int, row_id: str | None) -> str:
    shard_name = parquet_basename.removesuffix(".parquet")
    if row_id is not None:
        return f"{dataset_name}__{shard_name}__{row_id}"
    return f"{dataset_name}__{shard_name}__{row_idx:06d}"


def find_parquet_files(download_dir: str) -> list[tuple[str, str, dict]]:
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


# ---------------------------------------------------------------------------
# MDS writing
# ---------------------------------------------------------------------------

def process_image_to_mds(
    img: Image.Image, caption: str, image_id: str, subset: str, jpeg_quality: int,
) -> tuple[dict[str, Any], str] | None:
    try:
        orig_w, orig_h = img.size
        ar, tw, th = closest_bucket(orig_w, orig_h)
        ar_str = f"{ar:.3f}"

        img = img.resize((tw, th), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        jpeg_pil = Image.open(buf)
        jpeg_pil.load()

        sample = {
            "image": jpeg_pil, "width": tw, "height": th,
            "original_width": orig_w, "original_height": orig_h,
            "id": image_id, "prompt": caption or "",
            "enhanced_prompt": "", "aesthetic_predictor_v_2_5_score": "",
            "image_generator": "", "prompt_generator": "",
            "prompt_category": "", "style": "", "task": "",
            "enhancer": "", "image_aspect_ratio": "",
            "image_generated_with_enhanced_prompt": "",
            "length": len(caption) if caption else 0,
            "enhanced_length": 0, "subset": subset,
        }
        return sample, ar_str
    except Exception:
        return None


def write_pool_to_mds(
    pool_name: str,
    selected_ids: list[str],
    download_dir: str,
    output_root: str,
    logger: logging.Logger,
) -> int:
    """Write a pool of images to MDS format."""
    if MDSWriter is None:
        raise ImportError("mosaicml-streaming required for MDS conversion")

    os.makedirs(output_root, exist_ok=True)

    id_counts = Counter(selected_ids)
    unique_ids_needed = set(id_counts.keys())

    writers: dict[str, MDSWriter] = {}
    n_written = 0
    n_failed = 0
    start = time.time()

    source_files = find_parquet_files(download_dir)
    logger.info(
        f"  Writing MDS for {pool_name}: {len(selected_ids):,} images "
        f"({len(unique_ids_needed):,} unique) from {len(source_files)} parquet files"
    )

    for pq_path, dataset_name, cfg in source_files:
        if not unique_ids_needed:
            break

        pq_basename = os.path.basename(pq_path)
        columns = [cfg["image_col"]]
        if cfg["id_col"]:
            columns.append(cfg["id_col"])
        if cfg["caption_col"]:
            columns.append(cfg["caption_col"])

        try:
            df = pd.read_parquet(pq_path, columns=columns)
        except Exception as e:
            logger.error(f"Failed to read {pq_path}: {e}")
            continue

        for row_idx, row in df.iterrows():
            row_id = str(row[cfg["id_col"]]) if cfg["id_col"] else None
            unique_id = build_unique_id(dataset_name, pq_basename, row_idx, row_id)

            if unique_id not in unique_ids_needed:
                continue

            count = id_counts[unique_id]
            img = extract_pil_image(row[cfg["image_col"]], cfg["image_format"])
            if img is None:
                n_failed += 1
                unique_ids_needed.discard(unique_id)
                continue

            caption = str(row.get(cfg["caption_col"], "")) if cfg["caption_col"] else ""

            for copy_idx in range(count):
                sample_id = unique_id if copy_idx == 0 else f"{unique_id}__dup{copy_idx}"
                result = process_image_to_mds(img, caption, sample_id, dataset_name, JPEG_QUALITY)
                if result is None:
                    n_failed += 1
                    continue

                sample, ar_str = result
                if ar_str not in writers:
                    out = os.path.join(output_root, ar_str)
                    os.makedirs(out, exist_ok=True)
                    writers[ar_str] = MDSWriter(
                        out=out, columns=MDS_COLUMNS, size_limit=SHARD_SIZE,
                    )
                writers[ar_str].write(sample)
                n_written += 1

            unique_ids_needed.discard(unique_id)

        del df

        if n_written > 0 and n_written % 10000 < 500:
            elapsed = time.time() - start
            logger.info(
                f"    {pool_name}: {n_written:,} written, "
                f"{n_failed} failed, {n_written / max(1, elapsed):.0f}/s, "
                f"{len(unique_ids_needed):,} IDs remaining"
            )

    for w in writers.values():
        w.finish()

    elapsed = time.time() - start
    logger.info(
        f"  {pool_name} MDS complete: {n_written:,} written, "
        f"{n_failed} failed in {elapsed / 60:.1f}m"
    )

    if unique_ids_needed:
        logger.warning(f"  {pool_name}: {len(unique_ids_needed):,} IDs not found in source parquets")

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Construct shared MDS pools for dose-response experiment")
    parser.add_argument("--parquet-dir", default=None, help="Directory with annotation Parquet files")
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Base directory for manifests and ID lists")
    parser.add_argument("--mds-dir", default=None, help="Base directory for MDS outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-mds", action="store_true", help="Only create manifests, skip MDS conversion")
    parser.add_argument(
        "--build-pool",
        type=str, default=None,
        help="Build a single MDS pool by name (e.g. safe_full, unsafe_c2). "
             "If not specified, builds all pools.",
    )
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    download_dir = args.download_dir or config["training_data"]["download_dir"]
    output_dir = args.output_dir or config["training_data"]["subsets_dir"]
    mds_dir = args.mds_dir or config["training_data"]["mds_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mds_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "construct_subsets.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    rng = random.Random(args.seed)

    # Load annotations
    parquet_dir = args.parquet_dir or os.path.join(
        os.path.dirname(config["training_data"]["annotations_dir"]),
        "annotations_parquet",
    )
    logger.info(f"Loading annotations from Parquet: {parquet_dir}")
    annotations = load_annotations_from_parquet(parquet_dir)
    logger.info(f"Loaded {len(annotations):,} annotations")

    # Build safe/unsafe pools
    safe_ids = annotations[annotations["rating"] == "Safe"].index.tolist()
    unsafe_df = annotations[annotations["rating"] == "Unsafe"]
    unsafe_ids = unsafe_df.index.tolist()

    unsafe_by_category = defaultdict(list)
    for idx, row in unsafe_df.iterrows():
        cat = row.get("category", "NA: None applying")
        unsafe_by_category[cat].append(idx)

    logger.info(f"Image pools: {len(safe_ids):,} safe, {len(unsafe_ids):,} unsafe")

    # Define all pools
    pools = define_pools(safe_ids, unsafe_ids, unsafe_by_category, rng, logger)

    # Save pool manifests and ID lists
    for pool_name, pool in pools.items():
        manifest = {
            "pool_name": pool_name,
            "description": pool["description"],
            "total": len(pool["ids"]),
            "unique": len(set(pool["ids"])),
            "used_by": pool["used_by"],
            "seed": args.seed,
        }
        manifest_path = os.path.join(output_dir, f"pool_{pool_name}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        ids_path = os.path.join(output_dir, f"pool_{pool_name}_ids.json")
        with open(ids_path, "w") as f:
            json.dump(pool["ids"], f)

        logger.info(f"Saved {pool_name} manifest and IDs")

    # Save condition-to-pool mapping
    condition_map = {
        "C1": {"pools": ["safe_full"], "description": "0% unsafe, full scale"},
        "C2": {"pools": ["safe_full", "unsafe_c2"], "description": "5% unsafe, full scale"},
        "C3": {"pools": ["safe_full", "unsafe_c3"], "description": "10% unsafe, full scale"},
        "C0": {"pools": ["safe_full", "unsafe_full"], "description": "Original (1.21% unsafe), full scale"},
        "C4": {"pools": ["safe_c4", "unsafe_c4"], "description": "Same proportion (1.21%), 1M scale"},
        "C6": {"pools": ["safe_c6", "unsafe_full"], "description": "All unsafe (9.6%), 1M scale"},
    }
    with open(os.path.join(output_dir, "condition_pool_mapping.json"), "w") as f:
        json.dump(condition_map, f, indent=2)

    if args.skip_mds:
        logger.info("Skipping MDS conversion (--skip-mds)")
        return

    # Build MDS pools
    pools_to_build = [args.build_pool] if args.build_pool else list(pools.keys())

    for pool_name in pools_to_build:
        if pool_name not in pools:
            logger.warning(f"Unknown pool {pool_name}, skipping")
            continue

        pool_mds_dir = os.path.join(mds_dir, pool_name)
        if os.path.exists(pool_mds_dir) and os.listdir(pool_mds_dir):
            logger.info(f"Skipping {pool_name}: MDS directory already exists at {pool_mds_dir}")
            continue

        logger.info(f"Building MDS for pool: {pool_name}")
        write_pool_to_mds(
            pool_name, pools[pool_name]["ids"], download_dir, pool_mds_dir, logger
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
