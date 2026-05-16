"""Compute missing training-ref metrics on already-generated images.

Step 1: Precompute DINOv3 training reference stats (sample 30K from MDS)
Step 2: For each condition/model with training_generated/ images, compute missing metrics:
  - Dose-response: FDD, KDD (already have FID, KID, CLIP)
  - PRX models: KID, FDD, KDD (already have FID, CLIP)

Usage:
    python compute_missing_training_metrics.py
    python compute_missing_training_metrics.py --skip-precompute  # if DINOv3 stats already exist
"""
import argparse
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import torch
from glob import glob
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

os.environ["TORCH_HOME"] = "<your folder>"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
REF_STATS_DIR = "<your folder>"
METRICS_DIR = "<your folder>"
MDS_PATH = "<your folder>"
NUM_SAMPLES = 30000
SEED = 42

# Dose-response conditions
DOSE_CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5",
                   "C1_clip", "C0_clip", "C1_safeclip", "C0_safeclip"]

# PRX models
PRX_MODELS = ["prx-1024-beta", "prx-512-base", "prx-512-sft",
              "prx-512-sft-distilled", "prx-512-dc-ae", "prx-256-base", "prx-256-sft"]


# ============================================================
# Feature extraction
# ============================================================

def compute_inception_feats(img_dir, batch_size=64):
    """Compute Inception v3 features for images in a directory."""
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Inception"):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            all_feats.append(model(batch).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0).numpy()


def compute_dino_feats(img_dir, batch_size=32):
    """Compute DINOv3 features for images in a directory."""
    dino = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
    dino = dino.to(device).eval().to(torch.float32)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="DINOv3"):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            out = dino.forward_features(batch)
            all_feats.append(out["x_norm_clstoken"].cpu().float())
    del dino
    torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0).numpy()


def compute_fid(gen_feats, ref_stats_path):
    """Compute FID from generated features and cached reference stats."""
    ref = np.load(ref_stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]
    mu_gen = gen_feats.mean(axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))


def compute_kid(gen_feats, ref_feats_path, num_subsets=100, subset_size=1000):
    """Compute KID from generated features and cached reference features."""
    ref_feats = np.load(ref_feats_path)
    gen = torch.from_numpy(gen_feats)
    ref = torch.from_numpy(ref_feats)
    n = min(len(gen), len(ref), subset_size)
    if n < 2:
        return float("nan")
    kids = []
    for _ in range(num_subsets):
        idx_g = torch.randperm(len(gen))[:n]
        idx_r = torch.randperm(len(ref))[:n]
        x, y = gen[idx_g], ref[idx_r]
        d = x.shape[1]
        kxx = ((x @ x.T / d + 1) ** 3).mean()
        kyy = ((y @ y.T / d + 1) ** 3).mean()
        kxy = ((x @ y.T / d + 1) ** 3).mean()
        kids.append((kxx + kyy - 2 * kxy).item())
    return float(np.mean(kids))


# ============================================================
# Step 1: Precompute DINOv3 training reference stats
# ============================================================

def precompute_dino_training_stats():
    """Sample 30K training images from MDS, compute DINOv3 features."""
    dino_stats_path = os.path.join(REF_STATS_DIR, "training_30k_dinov3_stats.npz")
    dino_feats_path = os.path.join(REF_STATS_DIR, "training_30k_dinov3_feats.npy")

    if os.path.exists(dino_stats_path) and os.path.exists(dino_feats_path):
        logger.info("DINOv3 training ref stats already exist, skipping")
        return

    logger.info("=== Precomputing DINOv3 training ref stats ===")
    from streaming import StreamingDataset

    ds = StreamingDataset(local=MDS_PATH, shuffle=False)
    total = len(ds)
    rng = np.random.RandomState(SEED)
    indices = rng.choice(total, size=NUM_SAMPLES, replace=False)
    indices.sort()

    tmp_dir = tempfile.mkdtemp(prefix="training_30k_dino_")
    logger.info("Saving %d sampled images to %s", NUM_SAMPLES, tmp_dir)

    for count, idx in enumerate(tqdm(indices, desc="Sampling")):
        sample = ds[int(idx)]
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        img.save(os.path.join(tmp_dir, f"{count:05d}.jpg"))

    logger.info("Computing DINOv3 features...")
    feats = compute_dino_feats(tmp_dir)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)

    np.savez(dino_stats_path, mu=mu, sigma=sigma, n=NUM_SAMPLES)
    np.save(dino_feats_path, feats)
    logger.info("Saved DINOv3 stats: mu %s, sigma %s, feats %s", mu.shape, sigma.shape, feats.shape)

    shutil.rmtree(tmp_dir)


# ============================================================
# Step 2: Compute missing metrics
# ============================================================

def compute_missing_dose_response():
    """Compute FDD and KDD for dose-response conditions (already have FID, KID, CLIP)."""
    dino_stats = os.path.join(REF_STATS_DIR, "training_30k_dinov3_stats.npz")
    dino_feats = os.path.join(REF_STATS_DIR, "training_30k_dinov3_feats.npy")

    for condition in DOSE_CONDITIONS:
        gen_dir = os.path.join(METRICS_DIR, condition, "training_generated")
        result_path = os.path.join(METRICS_DIR, f"benchmark_results_training_{condition}.json")

        if not os.path.isdir(gen_dir):
            logger.warning("No training_generated for %s, skipping", condition)
            continue

        with open(result_path) as f:
            result = json.load(f)

        if "fdd_training_30k" in result and "kdd_training_30k" in result:
            logger.info("%s: FDD/KDD already computed, skipping", condition)
            continue

        logger.info("=== %s: computing FDD/KDD ===", condition)
        gen_dino_feats = compute_dino_feats(gen_dir)

        fdd_val = compute_fid(gen_dino_feats, dino_stats)
        result["fdd_training_30k"] = fdd_val
        logger.info("  FDD-training-30K: %.2f", fdd_val)

        kdd_val = compute_kid(gen_dino_feats, dino_feats)
        result["kdd_training_30k"] = kdd_val
        logger.info("  KDD-training-30K: %.6f", kdd_val)

        del gen_dino_feats

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("  Saved %s", result_path)


def compute_missing_prx():
    """Compute KID, FDD, KDD for PRX models (already have FID, CLIP)."""
    inception_feats = os.path.join(REF_STATS_DIR, "training_30k_inception_feats.npy")
    dino_stats = os.path.join(REF_STATS_DIR, "training_30k_dinov3_stats.npz")
    dino_feats = os.path.join(REF_STATS_DIR, "training_30k_dinov3_feats.npy")

    for model_short in PRX_MODELS:
        gen_dir = os.path.join(METRICS_DIR, model_short, "training_generated")
        result_path = os.path.join(METRICS_DIR, f"benchmark_training_{model_short}.json")

        if not os.path.isdir(gen_dir):
            logger.warning("No training_generated for %s, skipping", model_short)
            continue

        with open(result_path) as f:
            result = json.load(f)

        needs_kid = "kid_training_30k" not in result
        needs_fdd = "fdd_training_30k" not in result
        needs_kdd = "kdd_training_30k" not in result

        if not (needs_kid or needs_fdd or needs_kdd):
            logger.info("%s: all metrics already computed, skipping", model_short)
            continue

        logger.info("=== %s: computing missing metrics ===", model_short)

        # KID requires Inception features
        if needs_kid:
            logger.info("  Computing Inception features for KID...")
            gen_inception_feats = compute_inception_feats(gen_dir)
            kid_val = compute_kid(gen_inception_feats, inception_feats)
            result["kid_training_30k"] = kid_val
            logger.info("  KID-training-30K: %.6f", kid_val)
            del gen_inception_feats

        # FDD and KDD require DINOv3 features
        if needs_fdd or needs_kdd:
            logger.info("  Computing DINOv3 features for FDD/KDD...")
            gen_dino_feats = compute_dino_feats(gen_dir)

            if needs_fdd:
                fdd_val = compute_fid(gen_dino_feats, dino_stats)
                result["fdd_training_30k"] = fdd_val
                logger.info("  FDD-training-30K: %.2f", fdd_val)

            if needs_kdd:
                kdd_val = compute_kid(gen_dino_feats, dino_feats)
                result["kdd_training_30k"] = kdd_val
                logger.info("  KDD-training-30K: %.6f", kdd_val)

            del gen_dino_feats

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("  Saved %s", result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip DINOv3 training ref stats precomputation")
    args = parser.parse_args()

    if not args.skip_precompute:
        precompute_dino_training_stats()

    compute_missing_dose_response()
    compute_missing_prx()

    logger.info("Done computing all missing training-ref metrics.")


if __name__ == "__main__":
    main()
