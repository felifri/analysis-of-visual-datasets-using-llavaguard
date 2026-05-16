#!/bin/bash
#SBATCH --job-name=dose-ir-clip
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=4:00:00

# Compute ImageReward on COCO-30K generated images for CLIP/SafeCLIP conditions

set -euo pipefail

echo "ImageReward COCO-30K started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

$PYTHON << 'PYEOF'
import json, os, sys, time, logging
import numpy as np
import pandas as pd
import torch
from PIL import Image
from glob import glob
from torchvision import transforms as T

sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip
_patch_imscore_blip()
from imscore.imreward.model import ImageReward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
bench_dir = "<your folder>"

# Load COCO captions for prompts
coco_jsonl = "<your folder>"
captions = []
with open(coco_jsonl) as f:
    for line in f:
        data = json.loads(line)
        captions.append(data["conversations"][1]["value"])

logger.info("Loading ImageReward...")
model = ImageReward.from_pretrained("RE-N-Y/ImageReward").to(device).eval()
logger.info("ImageReward loaded")

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

conditions = ["C1_clip", "C0_clip", "C1_safeclip", "C0_safeclip"]

# Load existing benchmark results
results_path = os.path.join(bench_dir, "benchmark_results.json")
with open(results_path) as f:
    all_results = json.load(f)

for cond in conditions:
    img_dir = os.path.join(bench_dir, cond, "coco_generated")
    if not os.path.isdir(img_dir):
        logger.warning("No COCO images for %s at %s", cond, img_dir)
        continue

    paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if not paths:
        logger.warning("No jpg files in %s", img_dir)
        continue

    logger.info("Computing ImageReward for %s (%d images)...", cond, len(paths))
    n = min(len(paths), len(captions))
    scores = []
    t0 = time.time()

    for i in range(0, n, 1):
        img = Image.open(paths[i]).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model.score(img_tensor, [captions[i]])
            if isinstance(score, torch.Tensor):
                scores.append(float(score.cpu()))
            else:
                scores.append(float(score))
        if (i + 1) % 5000 == 0:
            logger.info("  %s: %d/%d (%.1f img/s)", cond, i+1, n, (i+1)/(time.time()-t0))

    elapsed = time.time() - t0
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    logger.info("  %s: IR=%.4f ± %.4f (%d images, %ds)", cond, mean_score, std_score, len(scores), elapsed)

    # Update benchmark results
    if cond in all_results:
        all_results[cond]["image_reward_coco_mean"] = mean_score
        all_results[cond]["image_reward_coco_std"] = std_score

# Save updated results
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

df = pd.DataFrame(all_results.values())
df.to_csv(os.path.join(bench_dir, "benchmark_results.csv"), index=False)
logger.info("Updated benchmark_results with ImageReward")
logger.info("\n%s", df[["condition", "fid_coco_30k", "clip_score_coco_mean", "image_reward_coco_mean"]].to_string())
PYEOF

echo "ImageReward completed at $(date)"
