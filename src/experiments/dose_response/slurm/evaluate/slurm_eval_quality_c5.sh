#!/bin/bash
#SBATCH --job-name=dose-quality-c5
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00

# Compute quality metrics and ImageReward for C5 (missing from earlier runs)
# Appends C5 row to existing quality_full.csv and image_reward.csv

set -euo pipefail

echo "C5 quality evaluation started on $(hostname) at $(date)"
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
import gc

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.environ.get("PRX_DIR", "<your folder>"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
output_dir = "<your folder>"
MAX_IMAGES = 10000
MODEL_ID = "dose_C5"

tar_path = "<your folder>"
c0_tar = "<your folder>"

prompt_df = pd.read_csv(
    "<your folder>",
    index_col=0,
)
prompts = prompt_df["prompt"].tolist()


def load_images_from_tar(path, max_images=MAX_IMAGES):
    images = []
    with tarfile.open(path, "r:") as tf:
        members = sorted(
            [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")],
            key=lambda m: m.name,
        )
        for member in tqdm(members[:max_images], desc=os.path.basename(os.path.dirname(path)), leave=False):
            f = tf.extractfile(member)
            if f:
                images.append(Image.open(f).convert("RGB"))
    return images


def load_training_sample(n=5000, seed=42):
    import random
    from glob import glob as globfn
    rng = random.Random(seed)
    raw_dir = "<your folder>"
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
        files = sorted(globfn(os.path.join(ds_dir, cfg["glob"]), recursive=True))
        all_files.extend([(f, cfg) for f in files])
    rng.shuffle(all_files)
    images = []
    for pq_path, cfg in all_files:
        if len(images) >= n:
            break
        try:
            df = pd.read_parquet(pq_path, columns=[cfg["img_col"]])
            sample_n = min(100, len(df), n - len(images))
            indices = rng.sample(range(len(df)), sample_n)
            for idx in indices:
                img_data = df.iloc[idx][cfg["img_col"]]
                raw = img_data["bytes"] if cfg["fmt"] == "dict" else img_data
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(img)
        except Exception as e:
            logger.warning(f"Error reading {pq_path}: {e}")
    logger.info(f"Loaded {len(images)} training reference images")
    return images[:n]


# ============================================================
# Feature extractors (same as slurm_eval_quality3.sh)
# ============================================================

class InceptionFeatureExtractor:
    def __init__(self, device="cuda"):
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
        self.model.fc = torch.nn.Identity()
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
            feats = out["x_norm_clstoken"]
            all_feats.append(feats.cpu().float())
        return torch.cat(all_feats, dim=0)


def compute_fid_from_features(feats_gen, feats_ref):
    mu_gen, sigma_gen = feats_gen.mean(0).numpy(), np.cov(feats_gen.numpy(), rowvar=False)
    mu_ref, sigma_ref = feats_ref.mean(0).numpy(), np.cov(feats_ref.numpy(), rowvar=False)
    diff = mu_gen - mu_ref
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))


def compute_kid_from_features(feats_gen, feats_ref, num_subsets=100, subset_size=1000):
    n = min(len(feats_gen), len(feats_ref), subset_size)
    if n < 2:
        return float("nan")
    kids = []
    for _ in range(num_subsets):
        idx_gen = torch.randperm(len(feats_gen))[:n]
        idx_ref = torch.randperm(len(feats_ref))[:n]
        x, y = feats_gen[idx_gen], feats_ref[idx_ref]
        d = x.shape[1]
        kxx = ((x @ x.T / d + 1) ** 3).mean()
        kyy = ((y @ y.T / d + 1) ** 3).mean()
        kxy = ((x @ y.T / d + 1) ** 3).mean()
        kids.append((kxx + kyy - 2 * kxy).item())
    return float(np.mean(kids))


def compute_mmd_rbf(feats_gen, feats_ref, sigma=10.0):
    n = min(5000, len(feats_gen), len(feats_ref))
    x, y = feats_gen[:n], feats_ref[:n]
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    kxx = torch.exp(-xx / (2 * sigma**2)).mean()
    kyy = torch.exp(-yy / (2 * sigma**2)).mean()
    kxy = torch.exp(-xy / (2 * sigma**2)).mean()
    return float((kxx + kyy - 2 * kxy) * 1000)


# ============================================================
# Run: Quality metrics for C5
# ============================================================

logger.info("Loading C5 images...")
c5_images = load_images_from_tar(tar_path, MAX_IMAGES)
logger.info(f"Loaded {len(c5_images)} C5 images")

result = {"model": MODEL_ID, "num_images": len(c5_images)}

# Load references
logger.info("Loading reference images...")
c0_images = load_images_from_tar(c0_tar, MAX_IMAGES)
train_images = load_training_sample(n=5000, seed=42)

from glob import glob as globfn
coco_dir = "<your folder>"
coco_paths = sorted(globfn(os.path.join(coco_dir, "*.jpg")))[:5000]
coco_images = [Image.open(p).convert("RGB") for p in tqdm(coco_paths, desc="COCO", leave=False)]

# Phase 1: Inception
logger.info("=== Inception features ===")
inception = InceptionFeatureExtractor(device)
c0_inc = inception.extract(c0_images)
train_inc = inception.extract(train_images)
coco_inc = inception.extract(coco_images)
c5_inc = inception.extract(c5_images)

result["fid_vs_c0"] = compute_fid_from_features(c5_inc, c0_inc)
result["fid_vs_train"] = compute_fid_from_features(c5_inc, train_inc)
result["fid_vs_coco"] = compute_fid_from_features(c5_inc, coco_inc)
result["kid_vs_c0"] = compute_kid_from_features(c5_inc, c0_inc)
result["kid_vs_train"] = compute_kid_from_features(c5_inc, train_inc)
result["kid_vs_coco"] = compute_kid_from_features(c5_inc, coco_inc)
logger.info(f"FID_c0={result['fid_vs_c0']:.1f}, FID_train={result['fid_vs_train']:.1f}, FID_coco={result['fid_vs_coco']:.1f}")

del inception, c0_inc, train_inc, coco_inc, c5_inc
torch.cuda.empty_cache(); gc.collect()

# Phase 2: DINOv3
logger.info("=== DINOv3 features ===")
dino = DINOv3FeatureExtractor(device)
c0_dino = dino.extract(c0_images)
train_dino = dino.extract(train_images)
coco_dino = dino.extract(coco_images)
c5_dino = dino.extract(c5_images)

result["fdd_vs_c0"] = compute_fid_from_features(c5_dino, c0_dino)
result["fdd_vs_train"] = compute_fid_from_features(c5_dino, train_dino)
result["fdd_vs_coco"] = compute_fid_from_features(c5_dino, coco_dino)
result["kdd_vs_c0"] = compute_kid_from_features(c5_dino, c0_dino)
result["kdd_vs_train"] = compute_kid_from_features(c5_dino, train_dino)
result["kdd_vs_coco"] = compute_kid_from_features(c5_dino, coco_dino)
result["mmd_dino_vs_c0"] = compute_mmd_rbf(c5_dino, c0_dino)
result["mmd_dino_vs_train"] = compute_mmd_rbf(c5_dino, train_dino)
result["mmd_dino_vs_coco"] = compute_mmd_rbf(c5_dino, coco_dino)
logger.info(f"FDD_c0={result['fdd_vs_c0']:.1f}, FDD_train={result['fdd_vs_train']:.1f}, FDD_coco={result['fdd_vs_coco']:.1f}")

del dino, c0_dino, train_dino, coco_dino, c5_dino, c0_images, train_images, coco_images
torch.cuda.empty_cache(); gc.collect()

# Phase 3: CLIP Score
logger.info("=== CLIP Score ===")
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip_scores = []
n = min(len(c5_images), len(prompts))
for i in range(0, n, 32):
    batch_imgs = c5_images[i:i+32]
    batch_prompts = prompts[i:i+32]
    inputs = clip_processor(text=batch_prompts, images=batch_imgs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_e = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        clip_scores.extend((img_e * txt_e).sum(dim=-1).cpu().tolist())

result["clip_score_mean"] = float(np.mean(clip_scores))
result["clip_score_std"] = float(np.std(clip_scores))
logger.info(f"CLIP={result['clip_score_mean']:.4f}")

del clip_model, clip_processor
torch.cuda.empty_cache()

# Append to existing quality_full.csv
csv_path = os.path.join(output_dir, "quality_full.csv")
existing_df = pd.read_csv(csv_path)
# Remove any existing C5 row (in case of re-run)
existing_df = existing_df[existing_df["model"] != MODEL_ID]
new_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
new_df.to_csv(csv_path, index=False)

# Also update the JSON
json_path = os.path.join(output_dir, "quality_full.json")
with open(json_path) as f:
    all_results = json.load(f)
all_results[MODEL_ID] = result
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

logger.info(f"Appended C5 to {csv_path}")

# ============================================================
# Phase 4: ImageReward for C5
# ============================================================

logger.info("\n=== ImageReward ===")

# Reload C5 images (freed earlier)
c5_images = load_images_from_tar(tar_path, MAX_IMAGES)

sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip
_patch_imscore_blip()
from imscore.imreward.model import ImageReward

ir_model = ImageReward.from_pretrained("RE-N-Y/ImageReward").to(device).eval()

ir_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

ir_scores = []
n = min(len(c5_images), len(prompts))
t0 = time.time()
for i in range(0, n, 32):
    batch_imgs = c5_images[i:i+32]
    batch_prompts = prompts[i:i+32]
    batch_tensors = torch.stack([ir_transform(img) for img in batch_imgs]).to(device)
    with torch.no_grad():
        batch_scores = ir_model.score(batch_tensors, batch_prompts)
        if isinstance(batch_scores, torch.Tensor):
            ir_scores.extend(batch_scores.cpu().tolist())
        else:
            ir_scores.append(float(batch_scores))

ir_result = {
    "model": MODEL_ID,
    "image_reward_mean": float(np.mean(ir_scores)),
    "image_reward_std": float(np.std(ir_scores)),
    "image_reward_median": float(np.median(ir_scores)),
    "num_images": len(ir_scores),
}
logger.info(f"ImageReward: mean={ir_result['image_reward_mean']:.4f} ({time.time()-t0:.0f}s)")

# Append to existing image_reward.csv
ir_csv = os.path.join(output_dir, "image_reward.csv")
ir_existing = pd.read_csv(ir_csv)
ir_existing = ir_existing[ir_existing["model"] != MODEL_ID]
ir_new = pd.concat([ir_existing, pd.DataFrame([ir_result])], ignore_index=True)
ir_new.to_csv(ir_csv, index=False)

# Update JSON
ir_json = os.path.join(output_dir, "image_reward.json")
with open(ir_json) as f:
    ir_all = json.load(f)
ir_all[MODEL_ID] = ir_result
with open(ir_json, "w") as f:
    json.dump(ir_all, f, indent=2)

logger.info(f"Appended C5 to {ir_csv}")
logger.info("All C5 quality metrics complete.")

PYEOF

echo "C5 quality evaluation completed at $(date)"
