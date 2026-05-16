#!/bin/bash
#SBATCH --job-name=train-missing-metrics
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Compute missing training-ref metrics:
# 1. Precompute DINOv3 training ref stats (~30 min)
# 2. FDD/KDD for 11 dose-response conditions
# 3. KID/FDD/KDD for 7 PRX models

set -euo pipefail

echo "Missing training metrics started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/compute_missing_training_metrics.py

echo "Missing training metrics completed at $(date)"
