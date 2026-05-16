#!/bin/bash
#SBATCH --job-name=dose-qual-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00

# Compute quality metrics for SFT dose-response models
# Reference-free: CLIP Score, ImageReward, Aesthetic Score
# Reference-based: FID, CMMD (using COCO real images)

set -euo pipefail

echo "SFT quality evaluation started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PRX_DIR=<your folder>

COCO_REAL=<your folder>
OUTPUT_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_eval_quality.py \
    --all-sft \
    --metrics clip_score,image_reward,fid,kid,fdd,kdd,cmmd \
    --reference-dir ${COCO_REAL} \
    --max-images 10000 \
    --batch-size 32 \
    --output-dir ${OUTPUT_DIR}

echo "SFT quality evaluation completed at $(date)"
