#!/bin/bash
#SBATCH --job-name=dose-eval-clip
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Cross-judge evaluation + testbench quality for CLIP/SafeCLIP text encoder ablation conditions
# Also computes missing C3 testbench quality metrics

set -euo pipefail

echo "CLIP/SafeCLIP evaluation started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>
PYTHON_SGLANG=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

# ── Step 1: Cross-judge evaluation for CLIP/SafeCLIP conditions ──
echo "=== Step 1: Cross-judge evaluation ==="
for cond in C1_clip C0_clip C1_safeclip C0_safeclip; do
    for judge in llamaguard3 shieldgemma sd_safety_checker; do
        echo "Running $judge on $cond..."
        $PYTHON experiments/dose_response/entrypoint_cross_judge.py \
            --condition $cond \
            --judge $judge || echo "WARNING: $judge on $cond failed"
    done
done

# ── Step 2: Testbench quality metrics for CLIP/SafeCLIP + C3 ──
echo "=== Step 2: Testbench quality metrics ==="
for cond in C1_clip C0_clip C1_safeclip C0_safeclip C3; do
    echo "Quality metrics for $cond..."
    $PYTHON experiments/dose_response/entrypoint_eval_quality.py \
        --model dose_$cond \
        --metrics clip_score \
        --max-images 10000 \
        --batch-size 32 || echo "WARNING: quality for $cond failed"
done

echo "CLIP/SafeCLIP evaluation completed at $(date)"
