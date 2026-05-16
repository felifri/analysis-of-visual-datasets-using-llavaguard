#!/bin/bash
#SBATCH --job-name=dose-gen-prx
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --array=0-6

# 7 PRX models, one per task
# Each task loads model on all 8 GPUs, batched generation + torch.compile
# Index mapping:
#   0: prx-1024-t2i-beta      (batch=8,  8 GPUs)
#   1: prx-512-t2i             (batch=32, 8 GPUs)
#   2: prx-512-t2i-sft         (batch=32, 8 GPUs)
#   3: prx-512-t2i-sft-distilled (batch=32, 8 GPUs)
#   4: prx-512-t2i-dc-ae       (batch=32, 8 GPUs)
#   5: prx-256-t2i             (batch=64, 8 GPUs)
#   6: prx-256-t2i-sft         (batch=64, 8 GPUs)

set -euo pipefail

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

# Disable proxy, force offline
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_generate_prx.py \
    --model-index $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
