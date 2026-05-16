#!/bin/bash
# SLURM array job for generating images from SFT checkpoints
# Generates 10K images per checkpoint using the prompt testbench
#
# Usage:
#   sbatch slurm_generate_post_training.sh
#
#SBATCH --job-name=dose-gen-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-6

set -euo pipefail

CONDITIONS=(C1 C2 C3 C0 C4 C6 C5)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "Generating images for sft/${CONDITION} on $(hostname) at $(date)"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=<your folder>
CKPT_DIR="<your folder>"
SCRIPT_DIR="<your folder>"

CKPT_BASE="${CKPT_DIR}/checkpoints_sft/${CONDITION}"
OUTPUT_DIR="${CKPT_DIR}/generated_images/sft/${CONDITION}"

echo "Checkpoint: ${CKPT_BASE}"
echo "Output: ${OUTPUT_DIR}"

cd ${SCRIPT_DIR}

${PYTHON} entrypoint_generate_dose_response.py \
    --condition "${CONDITION}" \
    --stage sft \
    --checkpoint-dir "${CKPT_BASE}" \
    --batch-size 4

echo "Done at $(date)"
