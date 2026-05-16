#!/bin/bash
#SBATCH --job-name=coco-missing-fdd
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00

# Compute missing COCO FDD/KDD for C1-C5 (images already generated)

set -euo pipefail

echo "Missing COCO DINOv3 metrics started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/compute_missing_coco_metrics.py

echo "Missing COCO DINOv3 metrics completed at $(date)"
