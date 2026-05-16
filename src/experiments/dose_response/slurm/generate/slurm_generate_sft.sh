#!/bin/bash
#SBATCH --job-name=dose-gen-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --array=0-5

# Generate 10K evaluation images from SFT checkpoints
# Array mapping:
#   0: C1 (0% unsafe)
#   1: C2 (5% unsafe)
#   2: C3 (10% unsafe)
#   3: C0 (original ~1.21% unsafe)
#   4: C4 (1.21% at 1M scale)
#   5: C6 (9.6% at 1M scale)

set -euo pipefail

CONDITIONS=(C1 C2 C3 C0 C4 C6)
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

$PYTHON experiments/dose_response/entrypoint_generate_dose_response.py \
    --condition $CONDITION \
    --stage sft \
    --batch-size 4 \
    --guidance-scale 3.5 \
    --num-inference-steps 50 \
    --seed 42

echo "$CONDITION SFT generation completed at $(date)"
