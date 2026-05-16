"""Compute missing COCO FDD/KDD for dose-response conditions C1-C5."""
import json
import logging
import os

import numpy as np
import torch
from glob import glob
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from tqdm import tqdm

os.environ["TORCH_HOME"] = "<your folder>"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
REF_STATS_DIR = "<your folder>"
METRICS_DIR = "<your folder>"

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]


def compute_dino_feats(img_dir, batch_size=32):
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


def compute_fdd(gen_feats, ref_stats_path):
    ref = np.load(ref_stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]
    mu_gen = gen_feats.mean(axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))


def compute_kdd(gen_feats, ref_feats_path, num_subsets=100, subset_size=1000):
    ref_feats = np.load(ref_feats_path)
    gen = torch.from_numpy(gen_feats)
    ref = torch.from_numpy(ref_feats)
    n = min(len(gen), len(ref), subset_size)
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


def main():
    dino_stats = os.path.join(REF_STATS_DIR, "coco_train_dinov3_stats.npz")
    dino_feats = os.path.join(REF_STATS_DIR, "coco_train_dinov3_feats.npy")

    # Load combined results
    results_json = os.path.join(METRICS_DIR, "benchmark_results.json")
    with open(results_json) as f:
        all_results = json.load(f)

    for condition in CONDITIONS:
        if all_results.get(condition, {}).get("fdd_coco_30k", 0) != 0:
            logger.info("%s: FDD/KDD already computed, skipping", condition)
            continue

        gen_dir = os.path.join(METRICS_DIR, condition, "coco_generated")
        logger.info("=== %s: computing FDD/KDD (COCO ref) ===", condition)

        gen_dino_feats = compute_dino_feats(gen_dir)

        fdd_val = compute_fdd(gen_dino_feats, dino_stats)
        all_results[condition]["fdd_coco_30k"] = fdd_val
        logger.info("  FDD-COCO-30K: %.2f", fdd_val)

        kdd_val = compute_kdd(gen_dino_feats, dino_feats)
        all_results[condition]["kdd_coco_30k"] = kdd_val
        logger.info("  KDD-COCO-30K: %.6f", kdd_val)

        del gen_dino_feats

        # Save incrementally
        with open(results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("  Saved to %s", results_json)

    # Also update CSV
    import pandas as pd
    df = pd.DataFrame(all_results.values())
    df.to_csv(os.path.join(METRICS_DIR, "benchmark_results.csv"), index=False)
    logger.info("Done. Updated benchmark_results.json and .csv")


if __name__ == "__main__":
    main()
