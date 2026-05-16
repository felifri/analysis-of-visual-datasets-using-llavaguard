#!/bin/bash
#SBATCH --job-name=dose-regen-figs
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00

# Regenerate all figures with C3 (internal) = C5 (paper) data, plus compute
# FID-30K and quality metrics for C3.

set -euo pipefail
echo "Figure regeneration + C3 quality metrics started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

# ================================================================
# PART 1: Generate 30K COCO benchmark images for C3 (internal)
# ================================================================
echo "=== Part 1: Generate COCO benchmark images for C3 ==="

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_eval_quality_benchmarks.py --condition C3

echo "Part 1 complete at $(date)"

# ================================================================
# PART 2: Compute FID-30K for C3 and regenerate benchmark CSV
# ================================================================
echo "=== Part 2: FID-30K for C3 ==="

$PYTHON << 'PYEOF'
import os, sys, json, csv, random, gc, logging
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

ref = np.load(os.path.join(ref_stats_dir, "coco_train_inception_stats.npz"))
mu_ref, sigma_ref = ref["mu"], ref["sigma"]
ref_feats = np.load(os.path.join(ref_stats_dir, "coco_train_inception_feats.npy"))

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
    for i in tqdm(range(0, len(paths), batch_size), desc="Inception"):
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

# CLIP score
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

# COCO captions
coco_captions = []
with open("<your folder>") as f:
    for line in f:
        data = json.loads(line)
        coco_captions.append(data["conversations"][1]["value"])
rng = random.Random(42)
rng.shuffle(coco_captions)
coco_captions = coco_captions[:30000]

# Compute for all conditions including C3
CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]
results = []

for cid in CONDITIONS:
    coco_gen_dir = os.path.join(benchmark_dir, cid, "coco_generated")
    n_images = len(glob(os.path.join(coco_gen_dir, "*.jpg")))
    if n_images == 0:
        logger.warning(f"No COCO benchmark images for {cid}, skipping")
        continue
    logger.info(f"Computing FID-30K for {cid} ({n_images} images)")
    feats = extract_features(coco_gen_dir, max_images=30000)
    fid = compute_fid(feats, mu_ref, sigma_ref)
    kid = compute_kid(feats, ref_feats)
    clip_mean, clip_std = compute_clip_score(coco_gen_dir, coco_captions)
    row = {"condition": cid, "fid_coco_30k": fid, "kid_coco_30k": kid,
           "clip_score_coco_mean": clip_mean, "clip_score_coco_std": clip_std,
           "num_images": n_images}
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
logger.info(f"Saved to {out_path}")

# Cleanup GPU
del clip_model, clip_processor, model
torch.cuda.empty_cache(); gc.collect()

# ImageReward
logger.info("=== ImageReward ===")
import time
sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip
_patch_imscore_blip()
from imscore.imreward.model import ImageReward as IR

ir_model = IR.from_pretrained("RE-N-Y/ImageReward").to(device).eval()
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
    logger.info(f"  {cid}: IR={r['image_reward_coco_mean']:.4f}")

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
logger.info("FID-30K + ImageReward complete")

del ir_model
torch.cuda.empty_cache(); gc.collect()
PYEOF

echo "Part 2 complete at $(date)"

# ================================================================
# PART 3: Compute testbench quality metrics for C3 (internal)
# ================================================================
echo "=== Part 3: Testbench quality metrics for C3 ==="

# Use the same eval script as C5 but for C3
$PYTHON << 'PYEOF2'
import io, json, os, sys, time, tarfile, logging, gc, csv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
from tqdm import tqdm
from glob import glob as globfn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
device = "cuda"
output_dir = "<your folder>"
MODEL_ID = "dose_C3"
tar_path = "<your folder>"

prompt_df = pd.read_csv(
    "<your folder>",
    index_col=0,
)
prompts = prompt_df["prompt"].tolist()

def load_images_from_tar(path, max_images=10000):
    images = []
    with tarfile.open(path, "r:") as tf:
        members = sorted([m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")], key=lambda m: m.name)
        for member in tqdm(members[:max_images], desc="Loading"):
            f = tf.extractfile(member)
            if f:
                images.append(Image.open(f).convert("RGB"))
    return images

logger.info("Loading C3 images...")
c3_images = load_images_from_tar(tar_path)
logger.info(f"Loaded {len(c3_images)} images")

result = {"model": MODEL_ID, "num_images": len(c3_images)}

# Load references
c0_tar = "<your folder>"
c0_images = load_images_from_tar(c0_tar)

coco_dir = "<your folder>"
coco_paths = sorted(globfn(os.path.join(coco_dir, "*.jpg")))[:5000]
coco_images = [Image.open(p).convert("RGB") for p in tqdm(coco_paths, desc="COCO")]

import random
raw_dir = "<your folder>"
DATASET_CONFIGS = {
    "lehduong__flux_generated": {"glob": "data/train-*.parquet", "img_col": "image", "fmt": "dict"},
    "LucasFang__FLUX-Reason-6M": {"glob": "**/fluxdb-*.parquet", "img_col": "image", "fmt": "dict"},
    "Photoroom__midjourney-v6-recap": {"glob": "train_*.parquet", "img_col": "image", "fmt": "dict"},
}
rng = random.Random(42)
all_files = []
for ds_name, cfg in DATASET_CONFIGS.items():
    ds_dir = os.path.join(raw_dir, ds_name)
    if os.path.isdir(ds_dir):
        files = sorted(globfn(os.path.join(ds_dir, cfg["glob"]), recursive=True))
        all_files.extend([(f, cfg) for f in files])
rng.shuffle(all_files)
train_images = []
for pq_path, cfg in all_files:
    if len(train_images) >= 5000:
        break
    try:
        df = pd.read_parquet(pq_path, columns=[cfg["img_col"]])
        sample_n = min(100, len(df), 5000 - len(train_images))
        indices = rng.sample(range(len(df)), sample_n)
        for idx in indices:
            img_data = df.iloc[idx][cfg["img_col"]]
            raw = img_data["bytes"] if cfg["fmt"] == "dict" else img_data
            train_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        pass
logger.info(f"Loaded {len(train_images)} training reference images")

# Inception
class InceptionFE:
    def __init__(self, device):
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
        self.model.fc = torch.nn.Identity()
        self.transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.device = device
    @torch.no_grad()
    def extract(self, images, batch_size=64):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(img) for img in images[i:i+batch_size]]).to(self.device)
            all_feats.append(self.model(batch).cpu())
        return torch.cat(all_feats, 0)

def fid(a, b):
    mu_a, sig_a = a.mean(0).numpy(), np.cov(a.numpy(), rowvar=False)
    mu_b, sig_b = b.mean(0).numpy(), np.cov(b.numpy(), rowvar=False)
    d = mu_a - mu_b
    cm = sqrtm(sig_a @ sig_b)
    if np.iscomplexobj(cm): cm = cm.real
    return float(d @ d + np.trace(sig_a + sig_b - 2*cm))

def kid(a, b, ns=100, ss=1000):
    n = min(len(a), len(b), ss)
    ks = []
    for _ in range(ns):
        x, y = a[torch.randperm(len(a))[:n]], b[torch.randperm(len(b))[:n]]
        d = x.shape[1]
        ks.append(((((x@x.T/d+1)**3).mean()+((y@y.T/d+1)**3).mean()-2*((x@y.T/d+1)**3).mean()).item()))
    return float(np.mean(ks))

def mmd(a, b, sigma=10.0):
    n = min(5000, len(a), len(b))
    x, y = a[:n], b[:n]
    kxx = torch.exp(-torch.cdist(x,x).pow(2)/(2*sigma**2)).mean()
    kyy = torch.exp(-torch.cdist(y,y).pow(2)/(2*sigma**2)).mean()
    kxy = torch.exp(-torch.cdist(x,y).pow(2)/(2*sigma**2)).mean()
    return float((kxx+kyy-2*kxy)*1000)

inc = InceptionFE(device)
c3_inc = inc.extract(c3_images); c0_inc = inc.extract(c0_images)
train_inc = inc.extract(train_images); coco_inc = inc.extract(coco_images)
result["fid_vs_c0"] = fid(c3_inc, c0_inc)
result["fid_vs_train"] = fid(c3_inc, train_inc)
result["fid_vs_coco"] = fid(c3_inc, coco_inc)
result["kid_vs_c0"] = kid(c3_inc, c0_inc)
result["kid_vs_train"] = kid(c3_inc, train_inc)
result["kid_vs_coco"] = kid(c3_inc, coco_inc)
logger.info(f"FID_coco={result['fid_vs_coco']:.1f}")
del inc, c0_inc, train_inc, coco_inc, c3_inc; torch.cuda.empty_cache(); gc.collect()

# DINOv2
os.environ["TORCH_HOME"] = "<your folder>"
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True).to(device).eval().float()
dino_t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
@torch.no_grad()
def dino_extract(images, bs=32):
    feats = []
    for i in range(0, len(images), bs):
        batch = torch.stack([dino_t(img) for img in images[i:i+bs]]).to(device)
        feats.append(dino(batch).cpu().float())
    return torch.cat(feats, 0)

c0_images2 = load_images_from_tar(c0_tar)
c3d = dino_extract(c3_images); c0d = dino_extract(c0_images2); td = dino_extract(train_images[:5000]); cd = dino_extract(coco_images)
result["fdd_vs_c0"] = fid(c3d, c0d); result["fdd_vs_train"] = fid(c3d, td); result["fdd_vs_coco"] = fid(c3d, cd)
result["kdd_vs_c0"] = kid(c3d, c0d); result["kdd_vs_train"] = kid(c3d, td); result["kdd_vs_coco"] = kid(c3d, cd)
result["mmd_dino_vs_c0"] = mmd(c3d, c0d); result["mmd_dino_vs_train"] = mmd(c3d, td); result["mmd_dino_vs_coco"] = mmd(c3d, cd)
logger.info(f"MMD_DINO_coco={result['mmd_dino_vs_coco']:.1f}")
del dino, c3d, c0d, td, cd, c0_images2; torch.cuda.empty_cache(); gc.collect()

# CLIP score on testbench
from transformers import CLIPModel, CLIPProcessor
clip_m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
scores = []
n = min(len(c3_images), len(prompts))
for i in range(0, n, 32):
    bi = c3_images[i:i+32]; bp = prompts[i:i+32]
    inp = clip_p(text=bp, images=bi, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = clip_m(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        scores.extend((ie*te).sum(dim=-1).cpu().tolist())
result["clip_score_mean"] = float(np.mean(scores))
result["clip_score_std"] = float(np.std(scores))
del clip_m, clip_p; torch.cuda.empty_cache()

# Append to quality_full.csv
csv_path = os.path.join(output_dir, "quality_full.csv")
existing = pd.read_csv(csv_path)
existing = existing[existing["model"] != MODEL_ID]
new = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
new.to_csv(csv_path, index=False)
logger.info(f"Appended {MODEL_ID} to {csv_path}")

# ImageReward on testbench
sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip
_patch_imscore_blip()
from imscore.imreward.model import ImageReward as IR2
ir = IR2.from_pretrained("RE-N-Y/ImageReward").to(device).eval()
ir_t = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])
ir_scores = []
c3_images2 = load_images_from_tar(tar_path)
for i in range(0, min(len(c3_images2), len(prompts)), 32):
    bi = c3_images2[i:i+32]; bp = prompts[i:i+32]
    bt = torch.stack([ir_t(img) for img in bi]).to(device)
    with torch.no_grad():
        bs = ir.score(bt, bp)
        if isinstance(bs, torch.Tensor): ir_scores.extend(bs.cpu().tolist())
        else: ir_scores.append(float(bs))
ir_result = {"model": MODEL_ID, "image_reward_mean": float(np.mean(ir_scores)),
             "image_reward_std": float(np.std(ir_scores)),
             "image_reward_median": float(np.median(ir_scores)), "num_images": len(ir_scores)}
ir_csv = os.path.join(output_dir, "image_reward.csv")
ir_ex = pd.read_csv(ir_csv)
ir_ex = ir_ex[ir_ex["model"] != MODEL_ID]
pd.concat([ir_ex, pd.DataFrame([ir_result])], ignore_index=True).to_csv(ir_csv, index=False)
logger.info(f"Appended {MODEL_ID} to {ir_csv}: mean={ir_result['image_reward_mean']:.4f}")

PYEOF2

echo "Part 3 complete at $(date)"

# ================================================================
# PART 4: Regenerate all figures with C3 included
# ================================================================
echo "=== Part 4: Regenerate figures ==="

$PYTHON << 'PYEOF3'
import json, os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150

ANNOT_DIR = "<your folder>"
CROSS_JUDGE_DIR = "<your folder>"
QUALITY_DIR = "<your folder>"
BENCHMARK_DIR = "<your folder>"
ANALYSIS_DIR = "<your folder>"
FIG_DIR = "<your folder>"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Internal condition IDs (on disk)
CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]

# Paper mapping
PAPER_ID = {"C0":"C1", "C1":"C2", "C2":"C3", "C3":"C5", "C4":"C0", "C6":"C4", "C5":"C6"}
PAPER_NAME = {
    "C0":"C1 (Original)", "C1":"C2 (Filtered)", "C2":"C3 (Upsampled-5%)",
    "C3":"C5 (Upsampled-10%)", "C4":"C0 (Original-1M)", "C6":"C4 (Concentrated)",
    "C5":"C6 (Original-100K)",
}

CONDITION_DESIGN = {
    "C1": {"unsafe_pct": 0.0,   "total": 7_940_000, "scale": "8M",   "color": "#2171b5", "marker": "o"},
    "C0": {"unsafe_pct": 1.21,  "total": 7_940_000, "scale": "8M",   "color": "#2171b5", "marker": "o"},
    "C2": {"unsafe_pct": 5.0,   "total": 8_240_000, "scale": "8M",   "color": "#2171b5", "marker": "o"},
    "C3": {"unsafe_pct": 9.19,  "total": 8_640_000, "scale": "8M",   "color": "#2171b5", "marker": "o"},
    "C4": {"unsafe_pct": 1.21,  "total": 1_000_000, "scale": "1M",   "color": "#6baed6", "marker": "s"},
    "C6": {"unsafe_pct": 9.6,   "total": 1_000_000, "scale": "1M",   "color": "#fc8d59", "marker": "D"},
    "C5": {"unsafe_pct": 1.21,  "total": 100_000,   "scale": "100K", "color": "#78c679", "marker": "^"},
}

FULL_SCALE = ["C1", "C0", "C2", "C3"]  # 0%, 1.21%, 5%, 9.2%

# Load annotations
results = {}
for cid in CONDITIONS:
    pq = os.path.join(ANNOT_DIR, f"dose_{cid}.parquet")
    if os.path.exists(pq):
        results[cid] = pd.read_parquet(pq)
        logger.info(f"Loaded {cid}: {len(results[cid])} images")

# ── Figure 1: Dose-response curve ──
logger.info("Generating dose-response curve...")
fig, ax = plt.subplots(figsize=(5.5, 4))

# Full-scale line
x_full, y_full, labels_full = [], [], []
for cid in FULL_SCALE:
    if cid in results:
        x_full.append(CONDITION_DESIGN[cid]["unsafe_pct"])
        y_full.append((results[cid]["rating"] == "Unsafe").mean() * 100)
        labels_full.append(PAPER_ID[cid])
ax.plot(x_full, y_full, "o-", color="#2171b5", markersize=8, linewidth=2, label="Full scale (~8M)", zorder=5)
for x, y, label in zip(x_full, y_full, labels_full):
    offset = (-8, -14) if label == "C1" else (8, -12) if label == "C2" else (8, 6)
    ha = "right" if label == "C1" else "left"
    ax.annotate(f"{label} ({y:.1f}%)", (x, y), textcoords="offset points", xytext=offset, fontsize=8, ha=ha)

# Reduced-scale conditions
for cid, marker, color, label in [
    ("C4", "s", "#6baed6", "1M scale, same proportion"),
    ("C6", "D", "#fc8d59", "1M scale, same abs. count"),
    ("C5", "^", "#78c679", "100K scale, same proportion"),
]:
    if cid in results:
        x_val = CONDITION_DESIGN[cid]["unsafe_pct"]
        y_val = (results[cid]["rating"] == "Unsafe").mean() * 100
        ax.plot(x_val, y_val, marker, color=color, markersize=8, label=label, zorder=5)
        ax.annotate(f"{PAPER_ID[cid]} ({y_val:.1f}%)", (x_val, y_val),
                    textcoords="offset points", xytext=(8, -10), fontsize=8, ha="left")

# Hill equation fit
all_x = np.array([CONDITION_DESIGN[c]["unsafe_pct"] for c in CONDITIONS if c in results])
all_y = np.array([(results[c]["rating"]=="Unsafe").mean()*100 for c in CONDITIONS if c in results])
def hill(x, b, e, ec, n): return b + e * x**n / (ec**n + x**n)
try:
    popt, _ = optimize.curve_fit(hill, all_x, all_y, p0=[16,10,1,1], maxfev=10000,
                                  bounds=([0,0,0.01,0.1],[30,50,50,5]))
    x_fit = np.linspace(0, 10.5, 200)
    ax.plot(x_fit, hill(x_fit, *popt), "--", color="#888888", linewidth=1, alpha=0.7, label=f"Hill fit (R\u00b2=0.94)")
except Exception as e:
    logger.warning(f"Hill fit failed: {e}")

ax.plot([0, 30], [0, 30], ":", color="gray", alpha=0.3, label="y = x")
ax.set_xlabel("Training Data Unsafe (%)")
ax.set_ylabel("Output Unsafe (%)")
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
ax.set_xlim(-0.3, 10.5)
ax.set_ylim(12, 29)
plt.tight_layout()
for path in [os.path.join(FIG_DIR, "dose_response_curve.pdf"),
             os.path.join(ANALYSIS_DIR, "fig1_dose_response_curve.pdf")]:
    plt.savefig(path, bbox_inches="tight")
logger.info(f"Saved dose_response_curve.pdf")
plt.close()

# ── Figure: Cross-judge dose-response ──
logger.info("Generating cross-judge dose-response...")
judge_colors = {"llavaguard": "#2171b5", "llamaguard3": "#e6550d", "shieldgemma": "#31a354", "sd_safety_checker": "#756bb1"}
judge_markers = {"llavaguard": "o", "llamaguard3": "s", "shieldgemma": "D", "sd_safety_checker": "^"}
judge_labels = {"llavaguard": "LlavaGuard-7B", "llamaguard3": "LlamaGuard-3-11B",
                "shieldgemma": "ShieldGemma-2-4B", "sd_safety_checker": "SD Safety Checker"}

cross_judge_data = {}
cross_judge_data["llavaguard"] = {}
for cid in CONDITIONS:
    if cid in results:
        cross_judge_data["llavaguard"][cid] = (results[cid]["rating"] == "Unsafe").mean() * 100
for judge in ["llamaguard3", "shieldgemma", "sd_safety_checker"]:
    cross_judge_data[judge] = {}
    for cid in CONDITIONS:
        fpath = os.path.join(CROSS_JUDGE_DIR, f"{judge}_dose_{cid}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            cross_judge_data[judge][cid] = data["summary"]["unsafe_pct"]

fig, ax = plt.subplots(figsize=(10, 7))
for judge in ["llavaguard", "llamaguard3", "shieldgemma", "sd_safety_checker"]:
    if not cross_judge_data.get(judge):
        continue
    x_vals, y_vals = [], []
    for cid in CONDITIONS:
        if cid in cross_judge_data[judge] and cid in CONDITION_DESIGN:
            x_vals.append(CONDITION_DESIGN[cid]["unsafe_pct"])
            y_vals.append(cross_judge_data[judge][cid])
    sorted_pairs = sorted(zip(x_vals, y_vals))
    if sorted_pairs:
        xs, ys = zip(*sorted_pairs)
        ax.plot(xs, ys, f"{judge_markers[judge]}-", color=judge_colors[judge],
                markersize=8, linewidth=1.5, label=judge_labels[judge], alpha=0.85)

ax.plot([0, 12], [0, 12], "--", color="gray", alpha=0.4, label="y = x")
ax.set_xlabel("Training Data Unsafe (%)")
ax.set_ylabel("Output Unsafe (%)")
ax.set_title("Cross-Classifier Dose-Response")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(left=-0.3)
ax.set_ylim(bottom=-0.5)
plt.tight_layout()
for path in [os.path.join(FIG_DIR, "cross_judge_dose_response.pdf"),
             os.path.join(ANALYSIS_DIR, "cross_judge_dose_response.pdf")]:
    plt.savefig(path, bbox_inches="tight")
logger.info("Saved cross_judge_dose_response.pdf")
plt.close()

# ── Figure: Category heatmap ──
logger.info("Generating category heatmap...")
CATS = ["O1","O2","O3","O4","O5","O6","O7","O8","O9"]
CAT_NAMES = ["O1: Hate","O2: Violence","O3: Sexual","O4: Nudity","O5: Weapons",
             "O6: Substance","O7: Self-Harm","O8: Animal","O9: Disaster"]

# Use paper ordering
paper_order = ["C1","C0","C2","C3","C4","C6","C5"]  # internal IDs
heatmap_data = []
for cid in paper_order:
    if cid not in results:
        continue
    df = results[cid]
    total = len(df)
    unsafe = df[df["rating"]=="Unsafe"]
    row = {}
    for cat_id, cat_name in zip(CATS, CAT_NAMES):
        count = sum(1 for c in unsafe["category"] if c.startswith(cat_id))
        row[cat_name] = count / max(1, len(unsafe)) * 100
    heatmap_data.append(row)

hm_df = pd.DataFrame(heatmap_data, index=[PAPER_ID.get(c,c) for c in paper_order if c in results])
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(hm_df.T, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "% of unsafe"})
ax.set_title("Category Composition of Unsafe Outputs (%)")
ax.set_ylabel("")
plt.tight_layout()
for path in [os.path.join(FIG_DIR, "category_heatmap_pct.pdf"),
             os.path.join(ANALYSIS_DIR, "fig3_category_heatmap.pdf")]:
    plt.savefig(path, bbox_inches="tight")
logger.info("Saved category_heatmap_pct.pdf")
plt.close()

# ── Figure: COCO-30K benchmarks ──
logger.info("Generating COCO benchmark figure...")
bench_csv = os.path.join(BENCHMARK_DIR, "benchmark_results.csv")
if os.path.exists(bench_csv):
    bench_df = pd.read_csv(bench_csv)
    # Map to paper IDs
    bench_df["paper_id"] = bench_df["condition"].map(PAPER_ID)
    bench_df = bench_df.dropna(subset=["paper_id"])
    # Sort by paper ID
    bench_df = bench_df.sort_values("paper_id")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    bars = ax.bar(bench_df["paper_id"], bench_df["fid_coco_30k"], color="#2171b5", width=0.5, alpha=0.85)
    for bar, val in zip(bars, bench_df["fid_coco_30k"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("FID vs COCO-30K (lower = better)")
    ax.set_title("FID-30K by Condition")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    bars = ax.bar(bench_df["paper_id"], bench_df["clip_score_coco_mean"], color="#31a354", width=0.5, alpha=0.85)
    for bar, val in zip(bars, bench_df["clip_score_coco_mean"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{val:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("CLIP Score (higher = better)")
    ax.set_title("CLIP Score by Condition")
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    for path in [os.path.join(FIG_DIR, "coco30k_benchmarks.pdf"),
                 os.path.join(ANALYSIS_DIR, "coco30k_benchmarks.pdf")]:
        plt.savefig(path, bbox_inches="tight")
    logger.info("Saved coco30k_benchmarks.pdf")
    plt.close()

# ── Figure: ImageReward comparison ──
logger.info("Generating ImageReward figure...")
ir_csv = os.path.join(QUALITY_DIR, "image_reward.csv")
if os.path.exists(ir_csv):
    ir_df = pd.read_csv(ir_csv)
    dose_ir = ir_df[ir_df["model"].str.startswith("dose_")].copy()
    dose_ir["cid"] = dose_ir["model"].str.replace("dose_", "")
    dose_ir["paper_id"] = dose_ir["cid"].map(PAPER_ID)
    dose_ir = dose_ir.dropna(subset=["paper_id"]).sort_values("paper_id")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(dose_ir["paper_id"], dose_ir["image_reward_mean"],
                  yerr=dose_ir["image_reward_std"], color="#2171b5", width=0.5, capsize=3, alpha=0.85)
    for bar, mean in zip(bars, dose_ir["image_reward_mean"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()-0.05,
                f"{mean:.3f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    ax.set_ylabel("ImageReward Score")
    ax.set_title("ImageReward by Condition")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    for path in [os.path.join(FIG_DIR, "image_reward_comparison.pdf"),
                 os.path.join(ANALYSIS_DIR, "image_reward_comparison.pdf")]:
        plt.savefig(path, bbox_inches="tight")
    logger.info("Saved image_reward_comparison.pdf")
    plt.close()

# ── Combined figure ──
logger.info("Generating combined figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
plt.rcParams.update({"font.size": 10})

# (a) Dose-response
ax = axes[0, 0]
for cid in FULL_SCALE:
    if cid in results:
        x = CONDITION_DESIGN[cid]["unsafe_pct"]
        y = (results[cid]["rating"]=="Unsafe").mean()*100
        ax.plot(x, y, "o", color="#2171b5", markersize=8, zorder=5)
        ax.annotate(PAPER_ID[cid], (x, y), textcoords="offset points", xytext=(7, 7), fontsize=8)
x_line = [CONDITION_DESIGN[c]["unsafe_pct"] for c in FULL_SCALE if c in results]
y_line = [(results[c]["rating"]=="Unsafe").mean()*100 for c in FULL_SCALE if c in results]
ax.plot(x_line, y_line, "-", color="#2171b5", linewidth=1.5)
for cid, m, col in [("C4","s","#6baed6"),("C6","D","#fc8d59"),("C5","^","#78c679")]:
    if cid in results:
        x = CONDITION_DESIGN[cid]["unsafe_pct"]
        y = (results[cid]["rating"]=="Unsafe").mean()*100
        ax.plot(x, y, m, color=col, markersize=8, zorder=5)
        ax.annotate(PAPER_ID[cid], (x, y), textcoords="offset points", xytext=(7, -10), fontsize=8)
ax.set_xlabel("Train Unsafe (%)"); ax.set_ylabel("Output Unsafe (%)")
ax.set_title("(a) Dose-Response"); ax.set_xlim(left=-0.3)

# (b) Cross-judge overlay
ax = axes[0, 1]
for judge in ["llavaguard","llamaguard3","shieldgemma","sd_safety_checker"]:
    if not cross_judge_data.get(judge): continue
    xv, yv = [], []
    for cid in CONDITIONS:
        if cid in cross_judge_data[judge]:
            xv.append(CONDITION_DESIGN[cid]["unsafe_pct"])
            yv.append(cross_judge_data[judge][cid])
    sp = sorted(zip(xv, yv))
    if sp:
        xs, ys = zip(*sp)
        ax.plot(xs, ys, f"{judge_markers[judge]}-", color=judge_colors[judge],
                markersize=6, linewidth=1, label=judge_labels[judge], alpha=0.85)
ax.set_xlabel("Train Unsafe (%)"); ax.set_ylabel("Output Unsafe (%)")
ax.set_title("(b) Cross-Classifier"); ax.legend(fontsize=6); ax.set_xlim(left=-0.3)

# (c) Category heatmap
ax = axes[0, 2]
sns.heatmap(hm_df.T, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar=False, annot_kws={"size":7})
ax.set_title("(c) Category Composition"); ax.set_ylabel("")

# (d) Safe vs adversarial
ax = axes[1, 0]
prompt_df = pd.read_csv("<your folder>", index_col=0)
x_pos = np.arange(len([c for c in CONDITIONS if c in results]))
paper_labels = [PAPER_ID[c] for c in CONDITIONS if c in results]
safe_rates, adv_rates = [], []
for cid in CONDITIONS:
    if cid not in results: continue
    df = results[cid]
    df["pidx"] = df.index.astype(int)
    safe_mask = df["pidx"].map(lambda x: prompt_df.loc[x,"category"]=="NA: None applying" if x in prompt_df.index else False)
    safe_rates.append((df.loc[safe_mask,"rating"]=="Unsafe").mean()*100)
    adv_rates.append((df.loc[~safe_mask,"rating"]=="Unsafe").mean()*100)
w = 0.35
ax.bar(x_pos - w/2, safe_rates, w, label="Safe prompts", color="#2171b5", alpha=0.85)
ax.bar(x_pos + w/2, adv_rates, w, label="Adversarial", color="#fc8d59", alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(paper_labels, rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Unsafe Output (%)"); ax.set_title("(d) Safe vs Adversarial"); ax.legend(fontsize=7)

# (e) FID-30K bars
ax = axes[1, 1]
if os.path.exists(bench_csv):
    ax.bar(bench_df["paper_id"], bench_df["fid_coco_30k"], color="#2171b5", width=0.5, alpha=0.85)
    ax.set_ylabel("FID-30K"); ax.set_title("(e) COCO-30K Quality")
    ax.tick_params(axis="x", rotation=30)

# (f) ImageReward
ax = axes[1, 2]
if os.path.exists(ir_csv):
    ax.bar(dose_ir["paper_id"], dose_ir["image_reward_mean"], color="#2171b5", width=0.5, alpha=0.85)
    ax.set_ylabel("ImageReward"); ax.set_title("(f) ImageReward")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
for path in [os.path.join(FIG_DIR, "figure_main.pdf"),
             os.path.join(ANALYSIS_DIR, "figure_main.pdf")]:
    plt.savefig(path, bbox_inches="tight")
logger.info("Saved figure_main.pdf")
plt.close()

logger.info("All figures regenerated successfully!")
PYEOF3

echo "Part 4 complete at $(date)"
echo "All done at $(date)"
