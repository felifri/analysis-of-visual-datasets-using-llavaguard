#!/bin/bash
#SBATCH --job-name=prx-train-fid-%a
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-6

# Training-FID for 7 existing PRX models — one model per array task
# Index mapping:
#   0: prx-1024-beta     3: prx-512-sft-distilled   6: prx-256-sft
#   1: prx-512-base      4: prx-512-dc-ae
#   2: prx-512-sft       5: prx-256-base

set -euo pipefail

echo "Training-ref benchmark for PRX model index $SLURM_ARRAY_TASK_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/eval_quality_benchmarks_prx.py \
    --model-index $SLURM_ARRAY_TASK_ID --reference training

echo "Training-ref benchmark for PRX model index $SLURM_ARRAY_TASK_ID completed at $(date)"
