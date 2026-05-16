#!/bin/bash
# SLURM array job for SFT training on Alchemist dataset
# Runs SFT for all 6 dose-response conditions (C1-C6)
#
# Usage:
#   sbatch slurm_sft_all.sh                    # submit all conditions
#   sbatch --array=0 slurm_sft_all.sh          # only C1
#   sbatch --array=0,3 slurm_sft_all.sh        # only C1 and C0
#
#SBATCH --job-name=dose-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --array=0-6

set -euo pipefail

CONDITIONS=(C1 C2 C3 C0 C4 C6 C5)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "SFT training for ${CONDITION} started on $(hostname) at $(date)"

export FSDP_VERSION=2
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export HYDRA_FULL_ERROR=1

COMPOSER=<your folder>
CKPT_DIR="<your folder>"
PRETRAINED_CKPT="${CKPT_DIR}/checkpoints_full/${CONDITION}/phase1/latest-rank0.pt"
SFT_SAVE_DIR="${CKPT_DIR}/checkpoints_sft/${CONDITION}"

cd <your folder>

echo "Loading pretrained checkpoint: ${PRETRAINED_CKPT}"
echo "Saving SFT checkpoint to: ${SFT_SAVE_DIR}"

$COMPOSER -n 8 -m prx.training.train \
    --config-name dose-response-sft \
    "dataset@dataset.train_dataset=train_alchemist" \
    "dataset@dataset.eval_dataset=train_alchemist" \
    name=${CONDITION}_sft group=dose-response-sft image_size=512 \
    global_batch_size=256 device_train_microbatch_size=32 \
    trainer.max_duration=20_000ba \
    trainer.save_folder=${SFT_SAVE_DIR} \
    +trainer.load_path=${PRETRAINED_CKPT} \
    trainer.load_weights_only=true \
    trainer.load_strict_model_weights=false \
    trainer.run_name=dose-response-sft-${CONDITION} \
    "~algorithms.pdino" \
    eval_first=false trainer.eval_interval=0 \
    optimizer.lr=5e-5 seed=42

echo "SFT training for ${CONDITION} complete at $(date)"
