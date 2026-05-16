#!/bin/bash
#SBATCH --job-name=dose-quality
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00

# Compute quality metrics for all dose-response conditions and existing PRX models
# Metrics: CLIP Score (no reference needed), ImageReward, Aesthetic Score

set -euo pipefail

echo "Quality evaluation started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PRX_DIR=<your folder>

cd <your folder>

# Evaluate dose-response models (CLIP score only for now — no reference images needed)
echo "Evaluating dose-response models..."
$PYTHON experiments/dose_response/entrypoint_eval_quality.py \
    --all-dose \
    --metrics clip_score \
    --max-images 10000 \
    --batch-size 32

# Evaluate existing PRX models
echo "Evaluating existing PRX models..."
$PYTHON experiments/dose_response/entrypoint_eval_quality.py \
    --all-prx \
    --metrics clip_score \
    --max-images 10000 \
    --batch-size 32

echo "Quality evaluation completed at $(date)"
