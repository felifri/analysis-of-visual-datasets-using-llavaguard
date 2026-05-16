#!/bin/bash
#SBATCH --job-name=dose-bench-clip
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00

# COCO-30K benchmarks for CLIP/SafeCLIP text encoder ablation conditions
# Generates 30K COCO images per condition and computes FID, CLIP score, ImageReward

set -euo pipefail

echo "COCO-30K benchmarks for CLIP/SafeCLIP started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

for cond in C1_clip C0_clip C1_safeclip C0_safeclip; do
    echo "=== COCO-30K benchmark for $cond ==="
    $PYTHON experiments/dose_response/entrypoint_eval_quality_benchmarks.py \
        --condition $cond || echo "WARNING: benchmark for $cond failed"
done

echo "COCO-30K benchmarks completed at $(date)"
