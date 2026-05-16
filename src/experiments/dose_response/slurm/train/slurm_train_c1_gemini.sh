#!/bin/bash
#SBATCH --job-name=dose-C1-gem
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00

set -euo pipefail
echo "C1 gemini training started on $(hostname) at $(date)"

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
    "dataset@dataset.train_dataset=train_dose_c1_gemini" \
    "dataset@dataset.eval_dataset=train_dose_c1_gemini" \
    name=C1 group=dose-response-gemini image_size=512 \
    global_batch_size=256 device_train_microbatch_size=32 \
    trainer.max_duration=100_000ba \
    trainer.save_folder=${CKPT_DIR}/C1/phase1 \
    trainer.run_name=dose-response-C1-gemini-phase1 \
    eval_first=false trainer.eval_interval=0 seed=42

echo "C1 gemini training complete at $(date)"
