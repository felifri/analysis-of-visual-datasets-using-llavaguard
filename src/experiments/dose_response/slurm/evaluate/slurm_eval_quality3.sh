#!/bin/bash
#SBATCH --job-name=dose-quality3
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Comprehensive quality metrics:
# - FID, KID (Inception features) vs C0 and vs training data sample
# - FDD, KDD (DINOv3 features) vs C0 and vs training data sample
# - Aesthetic Score
# - CLIP Score (already computed, included for completeness)

set -euo pipefail

echo "Quality evaluation 3 started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

$PYTHON << 'PYEOF'
import io
import json
import os
import sys
import time
import tarfile
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.environ.get("PRX_DIR", "<your folder>"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
output_dir = "<your folder>"
os.makedirs(output_dir, exist_ok=True)

MAX_IMAGES = 10000

# ============================================================
# Image loading
# ============================================================

def load_images_from_tar(tar_path, max_images=MAX_IMAGES):
    images = []
    with tarfile.open(tar_path, "r:") as tf:
        members = sorted(
            [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")],
            key=lambda m: m.name,
        )
        for member in tqdm(members[:max_images], desc=os.path.basename(os.path.dirname(tar_path)), leave=False):
            f = tf.extractfile(member)
            if f:
                images.append(Image.open(f).convert("RGB"))
    return images


def load_training_sample(n=10000, seed=42):
    """Load a random sample of training images as reference."""
    import random
    from glob import glob

    rng = random.Random(seed)
    raw_dir = "<your folder>"

    # Collect parquet files from all datasets
    DATASET_CONFIGS = {
        "lehduong__flux_generated": {"glob": "data/train-*.parquet", "img_col": "image", "fmt": "dict"},
        "LucasFang__FLUX-Reason-6M": {"glob": "**/fluxdb-*.parquet", "img_col": "image", "fmt": "dict"},
        "Photoroom__midjourney-v6-recap": {"glob": "train_*.parquet", "img_col": "image", "fmt": "dict"},
    }

    all_files = []
    for ds_name, cfg in DATASET_CONFIGS.items():
        ds_dir = os.path.join(raw_dir, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        from glob import glob as globfn
        files = sorted(globfn(os.path.join(ds_dir, cfg["glob"]), recursive=True))
        all_files.extend([(f, cfg) for f in files])

    rng.shuffle(all_files)

    images = []
    for pq_path, cfg in all_files:
        if len(images) >= n:
            break
        try:
            df = pd.read_parquet(pq_path, columns=[cfg["img_col"]])
            # Sample a few from each file
            sample_n = min(100, len(df), n - len(images))
            indices = rng.sample(range(len(df)), sample_n)
            for idx in indices:
                img_data = df.iloc[idx][cfg["img_col"]]
                raw = img_data["bytes"] if cfg["fmt"] == "dict" else img_data
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(img)
        except Exception as e:
            logger.warning(f"Error reading {pq_path}: {e}")
            continue

    logger.info(f"Loaded {len(images)} training reference images")
    return images[:n]


def get_tar_path(model_id):
    base = "<your folder>"
    if model_id.startswith("dose_"):
        cond = model_id.replace("dose_", "")
        return os.path.join(base, "dose_response", cond, "images.tar")
    else:
        return os.path.join(base, "prx_existing", model_id, "images.tar")


# ============================================================
# Feature extractors
# ============================================================

class InceptionFeatureExtractor:
    """Extract Inception v3 features for FID/KID."""
    def __init__(self, device="cuda"):
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
        self.model.fc = torch.nn.Identity()  # Remove classification head
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device

    @torch.no_grad()
    def extract(self, images, batch_size=64):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(img) for img in images[i:i+batch_size]]).to(self.device)
            feats = self.model(batch)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)


class DINOv3FeatureExtractor:
    """Extract DINOv3 features for FDD/KDD."""
    def __init__(self, device="cuda"):
        os.environ["TORCH_HOME"] = "<your folder>"
        self.model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
        self.model = self.model.to(device).eval().to(torch.float32)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device

    @torch.no_grad()
    def extract(self, images, batch_size=32):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(img) for img in images[i:i+batch_size]]).to(self.device)
            out = self.model.forward_features(batch)
            # Use CLS token
            feats = out["x_norm_clstoken"]
            all_feats.append(feats.cpu().float())
        return torch.cat(all_feats, dim=0)


# ============================================================
# Metrics
# ============================================================

def compute_fid_from_features(feats_gen, feats_ref):
    """Compute FID from pre-extracted features."""
    mu_gen, sigma_gen = feats_gen.mean(0).numpy(), np.cov(feats_gen.numpy(), rowvar=False)
    mu_ref, sigma_ref = feats_ref.mean(0).numpy(), np.cov(feats_ref.numpy(), rowvar=False)

    diff = mu_gen - mu_ref
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)
    return float(fid)


def compute_kid_from_features(feats_gen, feats_ref, num_subsets=100, subset_size=1000):
    """Compute KID (Kernel Inception Distance) from pre-extracted features."""
    n = min(len(feats_gen), len(feats_ref), subset_size)
    if n < 2:
        return float("nan")

    kids = []
    for _ in range(num_subsets):
        idx_gen = torch.randperm(len(feats_gen))[:n]
        idx_ref = torch.randperm(len(feats_ref))[:n]
        x = feats_gen[idx_gen]
        y = feats_ref[idx_ref]

        # Polynomial kernel k(x,y) = (x·y/d + 1)^3
        d = x.shape[1]
        kxx = ((x @ x.T / d + 1) ** 3).mean()
        kyy = ((y @ y.T / d + 1) ** 3).mean()
        kxy = ((x @ y.T / d + 1) ** 3).mean()
        kids.append((kxx + kyy - 2 * kxy).item())

    return float(np.mean(kids))


def compute_mmd_rbf(feats_gen, feats_ref, sigma=10.0):
    """Compute MMD with RBF kernel (used for CMMD/DINO-MMD style)."""
    n = min(5000, len(feats_gen), len(feats_ref))
    x = feats_gen[:n]
    y = feats_ref[:n]
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    kxx = torch.exp(-xx / (2 * sigma**2)).mean()
    kyy = torch.exp(-yy / (2 * sigma**2)).mean()
    kxy = torch.exp(-xy / (2 * sigma**2)).mean()
    return float((kxx + kyy - 2 * kxy) * 1000)


# ============================================================
# Main evaluation
# ============================================================

dose_models = ["dose_C1", "dose_C2", "dose_C0", "dose_C4", "dose_C6"]
prx_models = ["prx-1024-beta", "prx-512-base", "prx-512-sft",
               "prx-512-sft-distilled", "prx-512-dc-ae", "prx-256-base", "prx-256-sft"]
all_models = dose_models + prx_models

# Load reference sets
logger.info("Loading reference images...")

# Reference 1: C0 (original composition model output)
c0_images = load_images_from_tar(get_tar_path("dose_C0"), MAX_IMAGES)
logger.info(f"C0 reference: {len(c0_images)} images")

# Reference 2: Training data sample (use fewer to save memory)
train_images = load_training_sample(n=5000, seed=42)
logger.info(f"Training reference: {len(train_images)} images")

# Reference 3: COCO validation set
from glob import glob as globfn
coco_dir = "<your folder>"
coco_paths = sorted(globfn(os.path.join(coco_dir, "*.jpg")))[:5000]
coco_images = [Image.open(p).convert("RGB") for p in tqdm(coco_paths, desc="COCO", leave=False)]
logger.info(f"COCO reference: {len(coco_images)} images")

# === Phase 1: Inception features (extract all refs, then process models) ===
logger.info("=== Phase 1: Inception-based metrics ===")
all_results = {}
inception = InceptionFeatureExtractor(device)

logger.info("Extracting Inception reference features...")
c0_inception = inception.extract(c0_images)
train_inception = inception.extract(train_images)
coco_inception = inception.extract(coco_images)

# Process each model with Inception
for model_id in all_models:
    tar_path = get_tar_path(model_id)
    if not os.path.exists(tar_path):
        logger.warning(f"Skipping {model_id}: no tar")
        continue
    if model_id not in all_results:
        all_results[model_id] = {"model": model_id}

    logger.info(f"  {model_id}: Inception features...")
    images = load_images_from_tar(tar_path, MAX_IMAGES)
    all_results[model_id]["num_images"] = len(images)
    t0 = time.time()
    gen_inception = inception.extract(images)

    all_results[model_id]["fid_vs_c0"] = compute_fid_from_features(gen_inception, c0_inception)
    all_results[model_id]["fid_vs_train"] = compute_fid_from_features(gen_inception, train_inception)
    all_results[model_id]["fid_vs_coco"] = compute_fid_from_features(gen_inception, coco_inception)
    all_results[model_id]["kid_vs_c0"] = compute_kid_from_features(gen_inception, c0_inception)
    all_results[model_id]["kid_vs_train"] = compute_kid_from_features(gen_inception, train_inception)
    all_results[model_id]["kid_vs_coco"] = compute_kid_from_features(gen_inception, coco_inception)
    logger.info(f"  {model_id}: FID_c0={all_results[model_id]['fid_vs_c0']:.1f}, "
                f"FID_train={all_results[model_id]['fid_vs_train']:.1f}, "
                f"FID_coco={all_results[model_id]['fid_vs_coco']:.1f} ({time.time()-t0:.0f}s)")
    del images, gen_inception

# Free Inception
del inception, c0_inception, train_inception, coco_inception
torch.cuda.empty_cache()
import gc; gc.collect()

# === Phase 2: DINOv3 features ===
logger.info("\n=== Phase 2: DINOv3-based metrics ===")
dino = DINOv3FeatureExtractor(device)

logger.info("Extracting DINOv3 reference features...")
c0_dino = dino.extract(c0_images)
train_dino = dino.extract(train_images)
coco_dino = dino.extract(coco_images)

# Free reference images (no longer needed)
del c0_images, train_images, coco_images
gc.collect()

for model_id in all_models:
    tar_path = get_tar_path(model_id)
    if not os.path.exists(tar_path):
        continue

    logger.info(f"  {model_id}: DINOv3 features...")
    images = load_images_from_tar(tar_path, MAX_IMAGES)
    t0 = time.time()
    gen_dino = dino.extract(images)

    all_results[model_id]["fdd_vs_c0"] = compute_fid_from_features(gen_dino, c0_dino)
    all_results[model_id]["fdd_vs_train"] = compute_fid_from_features(gen_dino, train_dino)
    all_results[model_id]["fdd_vs_coco"] = compute_fid_from_features(gen_dino, coco_dino)
    all_results[model_id]["kdd_vs_c0"] = compute_kid_from_features(gen_dino, c0_dino)
    all_results[model_id]["kdd_vs_train"] = compute_kid_from_features(gen_dino, train_dino)
    all_results[model_id]["kdd_vs_coco"] = compute_kid_from_features(gen_dino, coco_dino)
    all_results[model_id]["mmd_dino_vs_c0"] = compute_mmd_rbf(gen_dino, c0_dino)
    all_results[model_id]["mmd_dino_vs_train"] = compute_mmd_rbf(gen_dino, train_dino)
    all_results[model_id]["mmd_dino_vs_coco"] = compute_mmd_rbf(gen_dino, coco_dino)
    logger.info(f"  {model_id}: FDD_c0={all_results[model_id]['fdd_vs_c0']:.1f}, "
                f"FDD_train={all_results[model_id]['fdd_vs_train']:.1f}, "
                f"FDD_coco={all_results[model_id]['fdd_vs_coco']:.1f} ({time.time()-t0:.0f}s)")
    del images, gen_dino

del dino, c0_dino, train_dino, coco_dino
torch.cuda.empty_cache()
gc.collect()

# === Phase 3: CLIP Score ===
logger.info("\n=== Phase 3: CLIP Score ===")
prompt_df = pd.read_csv(
    "<your folder>",
    index_col=0,
)
prompts = prompt_df["prompt"].tolist()

# CLIP model for CLIP score
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

for model_id in all_models:
    tar_path = get_tar_path(model_id)
    if not os.path.exists(tar_path):
        continue

    logger.info(f"  {model_id}: CLIP score...")
    images = load_images_from_tar(tar_path, MAX_IMAGES)
    t0 = time.time()
    clip_scores = []
    n = min(len(images), len(prompts))
    for i in range(0, n, 32):
        batch_imgs = images[i:i+32]
        batch_prompts = prompts[i:i+32]
        inputs = clip_processor(text=batch_prompts, images=batch_imgs, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            clip_scores.extend((img_e * txt_e).sum(dim=-1).cpu().tolist())

    all_results[model_id]["clip_score_mean"] = float(np.mean(clip_scores))
    all_results[model_id]["clip_score_std"] = float(np.std(clip_scores))
    logger.info(f"  {model_id}: CLIP={all_results[model_id]['clip_score_mean']:.4f} ({time.time()-t0:.0f}s)")
    del images

del clip_model, clip_processor
torch.cuda.empty_cache()

# Save results
with open(os.path.join(output_dir, "quality_full.json"), "w") as f:
    json.dump(all_results, f, indent=2)

rows = list(all_results.values())
df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, "quality_full.csv"), index=False)

logger.info(f"\nAll results saved to {output_dir}")
logger.info(f"\n{df.to_string()}")

PYEOF

echo "Quality evaluation 3 completed at $(date)"
