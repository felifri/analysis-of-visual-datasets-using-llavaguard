#!/bin/bash
#SBATCH --job-name=dose-quality2
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Compute additional quality metrics:
# - Aesthetic Score (reference-free)
# - FID between each condition and C0 (original) as reference
# - CMMD between each condition and C0

set -euo pipefail

echo "Quality evaluation 2 started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

$PYTHON << 'PYEOF'
import json
import os
import sys
import time
import tarfile
import logging

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
output_dir = "<your folder>"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def load_images_from_tar(tar_path, max_images=10000):
    images = []
    with tarfile.open(tar_path, "r:") as tf:
        members = sorted([m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")], key=lambda m: m.name)
        for member in tqdm(members[:max_images], desc=f"Loading {os.path.basename(os.path.dirname(tar_path))}", leave=False):
            f = tf.extractfile(member)
            if f:
                images.append(Image.open(f).convert("RGB"))
    return images

def get_tar_path(model_id):
    base = "<your folder>"
    if model_id.startswith("dose_"):
        cond = model_id.replace("dose_", "")
        return os.path.join(base, "dose_response", cond, "images.tar")
    else:
        return os.path.join(base, "prx_existing", model_id, "images.tar")

# ============================================================
# 1. Aesthetic Score
# ============================================================

def compute_aesthetic(images, device="cuda", batch_size=64):
    from transformers import CLIPModel, CLIPProcessor
    import torch.nn as nn

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    aesthetic_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/aesthetic_predictor_v2_5.pth")
    state = torch.load(aesthetic_path, map_location=device, weights_only=True)

    # Detect architecture from state dict
    keys = list(state.keys())
    first_weight_shape = state[keys[0]].shape
    in_dim = first_weight_shape[1] if len(first_weight_shape) == 2 else first_weight_shape[0]

    aesthetic_model = nn.Sequential(
        nn.Linear(in_dim, 1024), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    ).to(device)

    try:
        aesthetic_model.load_state_dict(state)
    except RuntimeError:
        logger.warning("Aesthetic predictor architecture mismatch, skipping")
        del clip_model, aesthetic_model
        return None

    aesthetic_model.eval()

    scores = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            embeds = clip_model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            score = aesthetic_model(embeds).squeeze(-1)
        scores.extend(score.cpu().tolist())

    del clip_model, aesthetic_model
    torch.cuda.empty_cache()
    return scores

# ============================================================
# 2. FID (using C0 as reference)
# ============================================================

def compute_fid(gen_images, ref_images, device="cuda", batch_size=64):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True).to(device)
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    for i in range(0, len(ref_images), batch_size):
        batch = torch.stack([transform(img) for img in ref_images[i:i+batch_size]]).to(device)
        fid.update(batch, real=True)

    for i in range(0, len(gen_images), batch_size):
        batch = torch.stack([transform(img) for img in gen_images[i:i+batch_size]]).to(device)
        fid.update(batch, real=False)

    val = fid.compute().item()
    del fid
    torch.cuda.empty_cache()
    return val

# ============================================================
# Run evaluations
# ============================================================

models = ["dose_C1", "dose_C2", "dose_C0", "dose_C4", "dose_C6"]
prx_models = ["prx-1024-beta", "prx-512-base", "prx-512-sft", "prx-512-sft-distilled",
               "prx-512-dc-ae", "prx-256-base", "prx-256-sft"]

all_results = {}

# Load C0 as reference for FID
logger.info("Loading C0 reference images for FID...")
c0_tar = get_tar_path("dose_C0")
c0_images = load_images_from_tar(c0_tar, max_images=10000)
logger.info(f"Loaded {len(c0_images)} C0 reference images")

for model_id in models + prx_models:
    tar_path = get_tar_path(model_id)
    if not os.path.exists(tar_path):
        logger.warning(f"Skipping {model_id}: no tar")
        continue

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating {model_id}")

    images = load_images_from_tar(tar_path, max_images=10000)
    result = {"model": model_id, "num_images": len(images)}

    # Aesthetic score
    logger.info("  Computing aesthetic score...")
    t0 = time.time()
    aes_scores = compute_aesthetic(images, device)
    if aes_scores:
        result["aesthetic_mean"] = float(np.mean(aes_scores))
        result["aesthetic_std"] = float(np.std(aes_scores))
        result["aesthetic_median"] = float(np.median(aes_scores))
        logger.info(f"  Aesthetic: {result['aesthetic_mean']:.3f} ({time.time()-t0:.0f}s)")
    else:
        result["aesthetic_mean"] = None
        logger.info("  Aesthetic: skipped (architecture mismatch)")

    # FID vs C0 (skip C0 vs itself)
    if model_id != "dose_C0":
        logger.info("  Computing FID vs C0...")
        t0 = time.time()
        fid_val = compute_fid(images, c0_images, device)
        result["fid_vs_c0"] = fid_val
        logger.info(f"  FID vs C0: {fid_val:.2f} ({time.time()-t0:.0f}s)")
    else:
        result["fid_vs_c0"] = 0.0

    all_results[model_id] = result
    del images

# Save
import pandas as pd
with open(os.path.join(output_dir, "quality_extended.json"), "w") as f:
    json.dump(all_results, f, indent=2)

rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in all_results.values()]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, "quality_extended.csv"), index=False)
logger.info(f"\nResults saved to {output_dir}")
logger.info(f"\n{df.to_string()}")

PYEOF

echo "Quality evaluation 2 completed at $(date)"
