#!/bin/bash
#SBATCH --job-name=dose-C4
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00

set -euo pipefail

echo "C4 training started on $(hostname) at $(date)"

export FSDP_VERSION=2

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export HYDRA_FULL_ERROR=1

COMPOSER=<your folder>
CKPT_DIR="<your folder>"

cd <your folder>

echo "Starting Phase 1 (single node, 8 GPUs)..."
$COMPOSER -n 8 -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_dose_c4" \
    "dataset@dataset.eval_dataset=train_dose_c4" \
    name=C4 group=dose-response image_size=512 \
    global_batch_size=256 device_train_microbatch_size=32 \
    trainer.max_duration=100_000ba \
    trainer.save_folder=${CKPT_DIR}/C4/phase1 \
    trainer.run_name=dose-response-C4-phase1 \
    "~algorithms.repa" \
    "~algorithms.pdino" \
    eval_first=false \
    trainer.eval_interval=0 \
    seed=42

echo "C4 Phase 1 complete at $(date)"
