#!/bin/bash
#SBATCH --job-name=dose-C2-full
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00

set -euo pipefail

echo "C2 full training started on $(hostname) at $(date)"

export FSDP_VERSION=2
export TORCH_HOME=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export HYDRA_FULL_ERROR=1

COMPOSER=<your folder>
CKPT_DIR="<your folder>"

cd <your folder>

# Phase 1: 512px, 100K steps
echo "Starting Phase 1..."
$COMPOSER -n 8 -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_dose_c2" \
    "dataset@dataset.eval_dataset=train_dose_c2" \
    name=C2 group=dose-response-full image_size=512 \
    global_batch_size=256 device_train_microbatch_size=32 \
    trainer.max_duration=100_000ba \
    trainer.save_folder=${CKPT_DIR}/C2/phase1 \
    trainer.run_name=dose-response-C2-full-phase1 \
    eval_first=false trainer.eval_interval=0 seed=42

echo "Phase 1 complete at $(date)"

# Phase 2: 1024px, 20K steps, no REPA
echo "Starting Phase 2..."
$COMPOSER -n 8 -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_dose_c2" \
    "dataset@dataset.eval_dataset=train_dose_c2" \
    name=C2 group=dose-response-full image_size=1024 \
    global_batch_size=128 device_train_microbatch_size=16 \
    trainer.max_duration=20_000ba \
    trainer.save_folder=${CKPT_DIR}/C2/phase2 \
    trainer.run_name=dose-response-C2-full-phase2 \
    +trainer.load_path=${CKPT_DIR}/C2/phase1/latest-rank0.pt \
    "~algorithms.repa" \
    eval_first=false trainer.eval_interval=0 seed=42

echo "C2 full training complete at $(date)"
