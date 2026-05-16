#!/bin/bash
#SBATCH --job-name=dose-gen-clip
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# Generate 10K evaluation images from text encoder ablation checkpoints
# Array mapping:
#   0: C1_clip      (Filtered + CLIP)
#   1: C0_clip      (Original + CLIP)
#   2: C1_safeclip  (Filtered + SafeCLIP)
#   3: C0_safeclip  (Original + SafeCLIP)

set -euo pipefail

CONDITIONS=(C1_clip C0_clip C1_safeclip C0_safeclip)
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
    --phase phase1 \
    --batch-size 4 \
    --guidance-scale 3.5 \
    --num-inference-steps 50 \
    --seed 42

echo "$CONDITION generation completed at $(date)"
