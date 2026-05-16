#!/bin/bash
#SBATCH --job-name=dose-train-%a
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-10

# FID-30K against training data distribution — one condition per array task
# Ref stats must already exist (run precompute_ref_stats_training.py first)

set -euo pipefail

CONDITIONS=(C1 C2 C3 C0 C4 C6 C5 C1_clip C0_clip C1_safeclip C0_safeclip)
COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "Training-ref benchmark for $COND started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_eval_quality_benchmarks.py \
    --condition $COND --reference training

echo "Training-ref benchmark for $COND completed at $(date)"
