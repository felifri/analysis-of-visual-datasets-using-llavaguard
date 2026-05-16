#!/bin/bash
# SLURM array job for annotating SFT generated images with LlavaGuard
# Annotates 10K images per model (7 SFT conditions)
#
# Usage:
#   sbatch slurm_annotate_post_training.sh
#
#SBATCH --job-name=dose-annot-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --array=0-6

set -euo pipefail

CONDITIONS=(C1 C2 C3 C0 C4 C6 C5)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "Annotating sft/${CONDITION} on $(hostname) at $(date)"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=<your folder>
SCRIPT_DIR="<your folder>"

cd ${SCRIPT_DIR}

MODEL_ID="sft/${CONDITION}"

$PYTHON entrypoint_annotate_outputs.py \
    --models "${MODEL_ID}" \
    --batch-size 1000 \
    --dp-size 4 \
    --port 10001

echo "Annotation for sft/${CONDITION} complete at $(date)"
