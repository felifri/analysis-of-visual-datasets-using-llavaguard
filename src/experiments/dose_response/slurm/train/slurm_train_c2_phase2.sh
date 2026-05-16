#!/bin/bash
#SBATCH --job-name=dose-C2-p2
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

set -euo pipefail
echo "C2 Phase 2 started on $(hostname) at $(date)"

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

$COMPOSER -n 8 -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_dose_c2" \
    "dataset@dataset.eval_dataset=train_dose_c2" \
    name=C2 group=dose-response-full image_size=1024 \
    global_batch_size=128 device_train_microbatch_size=16 \
    trainer.max_duration=20_000ba \
    trainer.save_folder=${CKPT_DIR}/C2/phase2 \
    trainer.run_name=dose-response-C2-full-phase2 \
    +trainer.load_weights_only=true +trainer.load_strict_model_weights=false +trainer.load_path=${CKPT_DIR}/C2/phase1/latest-rank0.pt \
    "~algorithms.repa" \
    eval_first=false trainer.eval_interval=0 seed=42

echo "C2 Phase 2 complete at $(date)"
