#!/bin/bash
# SLURM array job for cross-judge evaluation of SFT models
# 3 judges × 7 conditions = 21 tasks
#
# Array mapping:
#   task_id // 7 = judge index (0=llamaguard3, 1=shieldgemma, 2=sd_safety_checker)
#   task_id % 7  = condition index (C1-C5)
#
# Usage:
#   sbatch slurm_cross_judge_sft.sh                  # all 21 tasks
#   sbatch --array=0-6 slurm_cross_judge_sft.sh      # llamaguard3 only
#   sbatch --array=7-13 slurm_cross_judge_sft.sh     # shieldgemma only
#   sbatch --array=14-20 slurm_cross_judge_sft.sh    # sd_safety_checker only
#
#SBATCH --job-name=xjudge-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --array=0-20

set -euo pipefail

JUDGES=(llamaguard3 shieldgemma sd_safety_checker)
CONDITIONS=(C1 C2 C3 C0 C4 C6 C5)

JUDGE_IDX=$((SLURM_ARRAY_TASK_ID / 7))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 7))

JUDGE=${JUDGES[$JUDGE_IDX]}
CONDITION=${CONDITIONS[$COND_IDX]}

echo "Cross-judge SFT: ${JUDGE} on sft_${CONDITION} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd <your folder>

$PYTHON entrypoint_cross_judge.py \
    --judge ${JUDGE} \
    --sft-condition ${CONDITION}

echo "Cross-judge ${JUDGE} sft_${CONDITION} complete at $(date)"
