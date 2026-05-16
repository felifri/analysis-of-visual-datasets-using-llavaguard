#!/bin/bash
#SBATCH --job-name=dose-fid30k
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6:00:00

# Compute proper FID-30K (COCO captions → COCO real) for all dose conditions
# Images already generated in benchmark_metrics/{condition}/coco_generated/

set -euo pipefail

echo "FID-30K computation started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>

$PYTHON << 'PYEOF'
import os, sys, gc, json, logging, csv, random
import numpy as np
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
ref_stats_dir = "<your folder>"
benchmark_dir = "<your folder>"

# Load precomputed COCO reference stats
ref = np.load(os.path.join(ref_stats_dir, "coco_train_inception_stats.npz"))
mu_ref, sigma_ref = ref["mu"], ref["sigma"]
ref_feats = np.load(os.path.join(ref_stats_dir, "coco_train_inception_feats.npy"))
logger.info(f"Loaded COCO ref stats: mu={mu_ref.shape}, ref_feats={ref_feats.shape}")

# Setup Inception
model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
model.fc = torch.nn.Identity()
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_dir, max_images=30000, batch_size=64):
    paths = sorted(glob(os.path.join(image_dir, "*.jpg")))[:max_images]
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def compute_fid(gen_feats, mu_ref, sigma_ref):
    mu_gen = gen_feats.mean(axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))

def compute_kid(gen_feats, ref_feats, num_subsets=100, subset_size=1000):
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

# Also compute CLIP scores on COCO-generated images
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def compute_clip_score(image_dir, prompts, max_images=30000, batch_size=32):
    paths = sorted(glob(os.path.join(image_dir, "*.jpg")))[:max_images]
    scores = []
    for i in tqdm(range(0, min(len(paths), len(prompts)), batch_size), desc="CLIP"):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        batch_prompts = prompts[i:i+batch_size]
        inputs = clip_processor(text=batch_prompts, images=batch_imgs, return_tensors="pt",
                                padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores.extend((img_e * txt_e).sum(dim=-1).cpu().tolist())
    return float(np.mean(scores)), float(np.std(scores))

# Load COCO captions (same sampling as generation)
coco_captions = []
with open("<your folder>") as f:
    for line in f:
        data = json.loads(line)
        coco_captions.append(data["conversations"][1]["value"])
rng = random.Random(42)
rng.shuffle(coco_captions)
coco_captions = coco_captions[:30000]

# Compute for all conditions
CONDITIONS = ["C1", "C2", "C0", "C4", "C6", "C5"]
results = []

for cid in CONDITIONS:
    coco_gen_dir = os.path.join(benchmark_dir, cid, "coco_generated")
    n_images = len(glob(os.path.join(coco_gen_dir, "*.jpg")))
    logger.info(f"\n{'='*60}")
    logger.info(f"Computing FID-30K for {cid} ({n_images} images)")

    feats = extract_features(coco_gen_dir, max_images=30000)
    fid = compute_fid(feats, mu_ref, sigma_ref)
    kid = compute_kid(feats, ref_feats)
    clip_mean, clip_std = compute_clip_score(coco_gen_dir, coco_captions)

    row = {
        "condition": cid,
        "fid_coco_30k": fid,
        "kid_coco_30k": kid,
        "clip_score_coco_mean": clip_mean,
        "clip_score_coco_std": clip_std,
        "num_images": n_images,
    }
    results.append(row)
    logger.info(f"  {cid}: FID={fid:.2f}, KID={kid:.6f}, CLIP={clip_mean:.4f}")

# Save
out_path = os.path.join(benchmark_dir, "benchmark_results.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)

out_json = os.path.join(benchmark_dir, "benchmark_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

logger.info(f"\nSaved to {out_path}")
print("\n" + "="*70)
print("FID-30K COCO Benchmark Results")
print("="*70)
for r in results:
    print(f"  {r['condition']}: FID={r['fid_coco_30k']:.2f}  KID={r['kid_coco_30k']:.6f}  CLIP={r['clip_score_coco_mean']:.4f}")

del clip_model, clip_processor, model
torch.cuda.empty_cache()

# Also compute ImageReward on COCO-generated images
logger.info("\n=== ImageReward on COCO-generated ===")
sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip
_patch_imscore_blip()
from imscore.imreward.model import ImageReward
import time

ir_model = ImageReward.from_pretrained("RE-N-Y/ImageReward").to(device).eval()
ir_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

for r in results:
    cid = r["condition"]
    coco_gen_dir = os.path.join(benchmark_dir, cid, "coco_generated")
    paths = sorted(glob(os.path.join(coco_gen_dir, "*.jpg")))[:30000]
    ir_scores = []
    n = min(len(paths), len(coco_captions))
    t0 = time.time()
    for i in range(0, n, 32):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i+32]]
        batch_prompts = coco_captions[i:i+32]
        batch_tensors = torch.stack([ir_transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            batch_scores = ir_model.score(batch_tensors, batch_prompts)
            if isinstance(batch_scores, torch.Tensor):
                ir_scores.extend(batch_scores.cpu().tolist())
            else:
                ir_scores.append(float(batch_scores))
    r["image_reward_coco_mean"] = float(np.mean(ir_scores))
    r["image_reward_coco_std"] = float(np.std(ir_scores))
    logger.info(f"  {cid}: ImageReward={r['image_reward_coco_mean']:.4f} ({time.time()-t0:.0f}s)")

# Re-save with ImageReward
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

logger.info("All benchmarks complete.")
PYEOF

echo "FID-30K computation completed at $(date)"
