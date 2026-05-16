#!/bin/bash
#SBATCH --job-name=prx-1024-train-fid
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Training-FID for prx-1024-beta using 4 GPUs (single-image fallback is slow)

set -euo pipefail

echo "Training-ref benchmark for prx-1024-beta (4 GPUs) started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/eval_quality_benchmarks_prx.py \
    --model-index 0 --reference training

echo "Training-ref benchmark for prx-1024-beta completed at $(date)"
