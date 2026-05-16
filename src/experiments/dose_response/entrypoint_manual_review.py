"""Extract stratified sample images for manual validation of LlavaGuard annotations.

Samples 100 Safe + 100 Unsafe images per source (7 PRX models + training data),
extracts them to a review directory with a manifest CSV.

Usage:
    python entrypoint_manual_review.py
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import tarfile
from glob import glob

import pandas as pd
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same dataset configs as annotation script
DATASET_CONFIGS = {
    "lehduong__flux_generated": {
        "data_glob": "data/train-*.parquet",
        "image_col": "image",
        "image_format": "dict",
        "id_col": None,
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
        "image_format": "dict",
        "id_col": "id",
    },
}

PRX_MODELS = [
    "prx-1024-beta",
    "prx-512-base",
    "prx-512-sft",
    "prx-512-sft-distilled",
    "prx-512-dc-ae",
    "prx-256-base",
    "prx-256-sft",
]


def sample_ids(df: pd.DataFrame, n_safe: int, n_unsafe: int, rng: random.Random) -> pd.DataFrame:
    """Sample stratified IDs from annotation DataFrame."""
    safe = df[df["rating"] == "Safe"]
    unsafe = df[df["rating"] == "Unsafe"]

    n_safe = min(n_safe, len(safe))
    n_unsafe = min(n_unsafe, len(unsafe))

    safe_sample = safe.sample(n=n_safe, random_state=rng.randint(0, 2**31))
    unsafe_sample = unsafe.sample(n=n_unsafe, random_state=rng.randint(0, 2**31))

    return pd.concat([safe_sample, unsafe_sample])


def extract_prx_model(model_name: str, annotations_dir: str, images_dir: str,
                       review_dir: str, rng: random.Random):
    """Extract sample images for one PRX model."""
    parquet_path = os.path.join(annotations_dir, f"{model_name}.parquet")
    tar_path = os.path.join(images_dir, model_name, "images.tar")

    if not os.path.exists(parquet_path):
        print(f"  Skipping {model_name}: no annotation parquet at {parquet_path}")
        return
    if not os.path.exists(tar_path):
        print(f"  Skipping {model_name}: no images.tar at {tar_path}")
        return

    df = pd.read_parquet(parquet_path)
    sample = sample_ids(df, 100, 100, rng)
    sample_ids_set = set(str(idx) for idx in sample.index)

    print(f"  {model_name}: sampled {len(sample)} images "
          f"({(sample['rating'] == 'Safe').sum()} safe, {(sample['rating'] == 'Unsafe').sum()} unsafe)")

    # Create output directories
    safe_dir = os.path.join(review_dir, model_name, "safe")
    unsafe_dir = os.path.join(review_dir, model_name, "unsafe")
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)

    # Extract images from tar
    extracted = 0
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue
            img_id = os.path.basename(member.name).removesuffix(".jpg")
            if img_id not in sample_ids_set:
                continue

            rating = sample.loc[sample.index.astype(str) == img_id, "rating"].iloc[0]
            out_dir = safe_dir if rating == "Safe" else unsafe_dir

            f = tf.extractfile(member)
            if f:
                img_path = os.path.join(out_dir, f"{img_id}.jpg")
                with open(img_path, "wb") as out_f:
                    out_f.write(f.read())
                extracted += 1

    print(f"  {model_name}: extracted {extracted} images")

    # Write manifest
    manifest_path = os.path.join(review_dir, model_name, "manifest.csv")
    with open(manifest_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["image_id", "rating", "category", "rationale", "image_path"])
        for idx, row in sample.iterrows():
            img_id = str(idx)
            rating = row["rating"]
            subdir = "safe" if rating == "Safe" else "unsafe"
            img_path = os.path.join(review_dir, model_name, subdir, f"{img_id}.jpg")
            writer.writerow([img_id, rating, row.get("category", ""), row.get("rationale", ""), img_path])

    print(f"  {model_name}: manifest saved to {manifest_path}")


def extract_training_data(annotations_parquet_dir: str, raw_data_dir: str,
                           review_dir: str, rng: random.Random):
    """Extract sample images from training data."""
    # Load all training annotations
    parquet_files = sorted(glob(os.path.join(annotations_parquet_dir, "*.parquet")))
    if not parquet_files:
        print("  Skipping training data: no annotation parquets found")
        return

    print(f"  Loading {len(parquet_files)} annotation parquets...")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=False)

    sample = sample_ids(df, 100, 100, rng)
    sample_ids_set = set(sample.index.tolist())

    print(f"  Training data: sampled {len(sample)} images "
          f"({(sample['rating'] == 'Safe').sum()} safe, {(sample['rating'] == 'Unsafe').sum()} unsafe)")

    # Create output directories
    safe_dir = os.path.join(review_dir, "training_data", "safe")
    unsafe_dir = os.path.join(review_dir, "training_data", "unsafe")
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)

    # Parse sample IDs to find which source parquets to read
    # ID format: {dataset_name}__{shard_name}__{row_id_or_idx}
    id_to_source = {}
    for uid in sample_ids_set:
        parts = uid.split("__")
        if len(parts) >= 2:
            dataset_name = parts[0]
            shard_name = parts[1]
            id_to_source[uid] = (dataset_name, shard_name)

    # Group by source parquet
    source_groups = {}
    for uid, (ds, shard) in id_to_source.items():
        key = (ds, shard)
        if key not in source_groups:
            source_groups[key] = []
        source_groups[key].append(uid)

    extracted = 0

    for (dataset_name, shard_name), uids in source_groups.items():
        if dataset_name not in DATASET_CONFIGS:
            continue

        cfg = DATASET_CONFIGS[dataset_name]
        # Find the parquet file
        pattern = os.path.join(raw_data_dir, dataset_name, cfg["data_glob"])
        all_files = sorted(glob(pattern, recursive=True))

        target_file = None
        for f in all_files:
            if shard_name in os.path.basename(f).removesuffix(".parquet"):
                target_file = f
                break

        if target_file is None:
            print(f"  Could not find parquet for {dataset_name}/{shard_name}")
            continue

        # Read the parquet and extract matching images
        try:
            cols = [cfg["image_col"]]
            if cfg["id_col"]:
                cols.append(cfg["id_col"])
            pq_df = pd.read_parquet(target_file, columns=cols)
        except Exception as e:
            print(f"  Error reading {target_file}: {e}")
            continue

        uids_set = set(uids)
        pq_basename = os.path.basename(target_file)

        for row_idx, row in pq_df.iterrows():
            if cfg["id_col"]:
                row_id = str(row[cfg["id_col"]])
            else:
                row_id = None

            shard = pq_basename.removesuffix(".parquet")
            if row_id is not None:
                uid = f"{dataset_name}__{shard}__{row_id}"
            else:
                uid = f"{dataset_name}__{shard}__{row_idx:06d}"

            if uid not in uids_set:
                continue

            # Extract image
            try:
                img_data = row[cfg["image_col"]]
                if cfg["image_format"] == "dict":
                    raw = img_data["bytes"]
                else:
                    raw = img_data

                img = Image.open(io.BytesIO(raw))
                if img.mode != "RGB":
                    img = img.convert("RGB")

                rating = sample.loc[uid, "rating"]
                out_dir = safe_dir if rating == "Safe" else unsafe_dir
                # Use a safe filename
                safe_fname = uid.replace("/", "_").replace("__", "_") + ".jpg"
                img.save(os.path.join(out_dir, safe_fname))
                extracted += 1
            except Exception as e:
                print(f"  Error extracting {uid}: {e}")

    print(f"  Training data: extracted {extracted} images")

    # Write manifest
    manifest_path = os.path.join(review_dir, "training_data", "manifest.csv")
    with open(manifest_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["image_id", "rating", "category", "rationale", "image_path"])
        for idx, row in sample.iterrows():
            uid = str(idx)
            rating = row["rating"]
            subdir = "safe" if rating == "Safe" else "unsafe"
            safe_fname = uid.replace("/", "_").replace("__", "_") + ".jpg"
            img_path = os.path.join(review_dir, "training_data", subdir, safe_fname)
            writer.writerow([uid, rating, row.get("category", ""), row.get("rationale", ""), img_path])

    print(f"  Training data: manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract sample images for manual review")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    annotations_dir = config["evaluation"]["output_dir"]
    images_dir = os.path.join(config["generation"]["output_dir"], "prx_existing")
    annotations_parquet_dir = os.path.join(
        config["training_data"]["annotations_dir"],
        "annotations_parquet",
    )
    raw_data_dir = config["training_data"]["download_dir"]
    review_dir = os.path.join(os.path.dirname(config["base_output_dir"]), "review")

    os.makedirs(review_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Review output directory: {review_dir}")
    print(f"Annotations: {annotations_dir}")
    print(f"Images: {images_dir}")

    # Extract PRX model samples
    for model_name in PRX_MODELS:
        print(f"\nProcessing {model_name}...")
        extract_prx_model(model_name, annotations_dir, images_dir, review_dir, rng)

    # Extract training data samples
    print(f"\nProcessing training data...")
    extract_training_data(annotations_parquet_dir, raw_data_dir, review_dir, rng)

    print(f"\nDone. Review images at: {review_dir}")


if __name__ == "__main__":
    main()
