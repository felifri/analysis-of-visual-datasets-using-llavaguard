#!/bin/bash
#SBATCH --job-name=dose-gen-multiseed
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --array=0-1

# Generate 10K evaluation images with multiple diffusion seeds for C1 and C2
# Array mapping:
#   0: C1 (0% unsafe)
#   1: C2 (5% unsafe)
# Each task generates all 4 new seeds (137, 314, 789, 1331) sequentially.

set -euo pipefail

CONDITIONS=(C1 C2)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($CONDITION) started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

# Disable proxy, force offline for model loading
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>

# Set PRX directory
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_multi_seed.py \
    --conditions $CONDITION \
    --seeds 137 314 789 1331 \
    --phase phase1 \
    --batch-size 4 \
    --guidance-scale 3.5 \
    --num-inference-steps 50

echo "$CONDITION multi-seed generation completed at $(date)"
