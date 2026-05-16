"""Cross-judge evaluation on training data sample.

Extracts 10K training images (5K safe + 5K unsafe per LlavaGuard) and
evaluates them with ShieldGemma and LlamaGuard-4 to validate the
training data safety annotations.

Usage:
    python entrypoint_cross_judge_training.py --judge shieldgemma
    python entrypoint_cross_judge_training.py --judge llamaguard4
"""

import argparse
import io
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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


def extract_training_sample(n_safe=5000, n_unsafe=5000, seed=42):
    """Extract a stratified sample of training images with their LlavaGuard labels."""
    # Load LlavaGuard annotations
    parquet_dir = "<your folder>"
    files = sorted(glob(os.path.join(parquet_dir, "*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs)

    rng = random.Random(seed)

    safe_ids = df[df["rating"] == "Safe"].index.tolist()
    unsafe_ids = df[df["rating"] == "Unsafe"].index.tolist()

    sample_safe = rng.sample(safe_ids, min(n_safe, len(safe_ids)))
    sample_unsafe = rng.sample(unsafe_ids, min(n_unsafe, len(unsafe_ids)))

    sample_ids = set(sample_safe + sample_unsafe)
    sample_labels = {}
    for uid in sample_safe:
        sample_labels[uid] = {"llavaguard_rating": "Safe", "llavaguard_category": df.loc[uid, "category"]}
    for uid in sample_unsafe:
        sample_labels[uid] = {"llavaguard_rating": "Unsafe", "llavaguard_category": df.loc[uid, "category"]}

    logger.info("Sampled %d safe + %d unsafe = %d training images", len(sample_safe), len(sample_unsafe), len(sample_ids))

    # Extract images from source parquets
    raw_dir = "<your folder>"
    images = {}

    for ds_name, cfg in DATASET_CONFIGS.items():
        ds_dir = os.path.join(raw_dir, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        pattern = os.path.join(ds_dir, cfg["data_glob"])
        pq_files = sorted(glob(pattern, recursive=True))

        for pq_path in tqdm(pq_files, desc=f"Reading {ds_name}", leave=False):
            pq_basename = os.path.basename(pq_path).removesuffix(".parquet")
            cols = [cfg["image_col"]]
            if cfg["id_col"]:
                cols.append(cfg["id_col"])

            try:
                pq_df = pd.read_parquet(pq_path, columns=cols)
            except Exception:
                continue

            for row_idx, row in pq_df.iterrows():
                if cfg["id_col"]:
                    row_id = str(row[cfg["id_col"]])
                    uid = f"{ds_name}__{pq_basename}__{row_id}"
                else:
                    uid = f"{ds_name}__{pq_basename}__{row_idx:06d}"

                if uid not in sample_ids or uid in images:
                    continue

                try:
                    img_data = row[cfg["image_col"]]
                    raw = img_data["bytes"] if cfg["image_format"] == "dict" else img_data
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    images[uid] = img
                except Exception:
                    pass

            if len(images) >= len(sample_ids):
                break

        if len(images) >= len(sample_ids):
            break

    logger.info("Extracted %d / %d training images", len(images), len(sample_ids))
    return images, sample_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=str, required=True,
                        choices=["llamaguard3", "shieldgemma", "sd_safety_checker"])
    parser.add_argument("--n-safe", type=int, default=5000)
    parser.add_argument("--n-unsafe", type=int, default=5000)
    args = parser.parse_args()

    output_base = "<your folder>"
    os.makedirs(output_base, exist_ok=True)

    out_path = os.path.join(output_base, f"{args.judge}_training_data.json")
    if os.path.exists(out_path):
        logger.info("Results already exist at %s", out_path)
        return

    # Extract training sample
    logger.info("Extracting training data sample...")
    images, labels = extract_training_sample(args.n_safe, args.n_unsafe)

    image_list = list(images.values())
    name_list = list(images.keys())

    # Import judge-specific function
    sys.path.insert(0, os.path.dirname(__file__))
    from entrypoint_cross_judge import evaluate_llamaguard3, evaluate_shieldgemma, evaluate_sd_safety_checker

    logger.info("Evaluating %d training images with %s...", len(image_list), args.judge)
    t0 = time.time()

    if args.judge == "llamaguard3":
        results = evaluate_llamaguard3(image_list, name_list)
    elif args.judge == "shieldgemma":
        results = evaluate_shieldgemma(image_list, name_list)
    elif args.judge == "sd_safety_checker":
        results = evaluate_sd_safety_checker(image_list, name_list)

    elapsed = time.time() - t0

    # Merge with LlavaGuard labels
    for r in results:
        uid = r["id"]
        if uid in labels:
            r["llavaguard_rating"] = labels[uid]["llavaguard_rating"]
            r["llavaguard_category"] = labels[uid]["llavaguard_category"]

    # Compute agreement
    agree = sum(1 for r in results if r.get("llavaguard_rating") and r["rating"] == r["llavaguard_rating"])
    disagree = sum(1 for r in results if r.get("llavaguard_rating") and r["rating"] != r["llavaguard_rating"])
    n_with_label = agree + disagree

    # Confusion matrix
    tp = sum(1 for r in results if r.get("llavaguard_rating") == "Unsafe" and r["rating"] == "Unsafe")
    fp = sum(1 for r in results if r.get("llavaguard_rating") == "Safe" and r["rating"] == "Unsafe")
    fn = sum(1 for r in results if r.get("llavaguard_rating") == "Unsafe" and r["rating"] == "Safe")
    tn = sum(1 for r in results if r.get("llavaguard_rating") == "Safe" and r["rating"] == "Safe")

    summary = {
        "judge": args.judge,
        "n_images": len(results),
        "n_unsafe_judge": sum(1 for r in results if r["rating"] == "Unsafe"),
        "n_unsafe_llavaguard": sum(1 for r in results if r.get("llavaguard_rating") == "Unsafe"),
        "agreement": agree,
        "disagreement": disagree,
        "agreement_rate": agree / max(1, n_with_label),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "elapsed_s": elapsed,
    }

    # Cohen's kappa
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(1, n_with_label ** 2)
    po = (tp + tn) / max(1, n_with_label)
    summary["cohens_kappa"] = (po - pe) / max(1e-10, 1 - pe)

    output = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("\nTraining data cross-judge results:")
    logger.info("  Agreement: %d/%d (%.1f%%)", agree, n_with_label, summary["agreement_rate"] * 100)
    logger.info("  Cohen's kappa: %.3f", summary["cohens_kappa"])
    logger.info("  Confusion: TP=%d, FP=%d, FN=%d, TN=%d", tp, fp, fn, tn)
    logger.info("  Saved to %s", out_path)


if __name__ == "__main__":
    main()
