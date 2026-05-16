#!/bin/bash
#SBATCH --job-name=dose-eval-C3
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Full evaluation pipeline for C3 (10% unsafe):
#   1. LlavaGuard safety annotation
#   2. Cross-judge evaluation (LlamaGuard-3, ShieldGemma, SD Safety Checker)
#   3. Quality metrics (FID, KID, CLIP, DINO-MMD, ImageReward)

set -euo pipefail

echo "C3 full evaluation started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>
PYTHON_SGLANG=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

# Step 1: LlavaGuard safety annotation
echo "=== Step 1: LlavaGuard annotation ==="
$PYTHON_SGLANG experiments/dose_response/entrypoint_eval_quality.py \
    --condition C3 \
    --judge llavaguard

# Step 2: Cross-judge evaluation
echo "=== Step 2: Cross-judge evaluation ==="
for judge in llamaguard3 shieldgemma sd_safety_checker; do
    echo "Running $judge..."
    $PYTHON_SGLANG experiments/dose_response/entrypoint_cross_judge.py \
        --condition C3 \
        --judge $judge
done

# Step 3: Quality metrics
echo "=== Step 3: Quality metrics ==="
$PYTHON experiments/dose_response/entrypoint_eval_quality_benchmarks.py \
    --condition C3

echo "C3 full evaluation completed at $(date)"
