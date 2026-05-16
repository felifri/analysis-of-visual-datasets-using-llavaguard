#!/bin/bash
#SBATCH --job-name=dose-bench-training
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00

# FID-30K against training data distribution
# Step 1: Precompute Inception ref stats from 30K MDS training images
# Step 2: Generate 30K images per condition and compute FID/KID/CLIP against training ref

set -euo pipefail

echo "Training-ref benchmarks started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

# Step 1: Precompute training reference stats (~20 min)
echo "=== Step 1: Precomputing training reference stats ==="
$PYTHON experiments/dose_response/precompute_ref_stats_training.py
echo "Reference stats done at $(date)"

# Step 2: Run training-ref benchmarks for all conditions
echo "=== Step 2: Training-ref benchmarks for all conditions ==="
$PYTHON experiments/dose_response/entrypoint_eval_quality_benchmarks.py \
    --all --reference training

echo "Training-ref benchmarks completed at $(date)"
