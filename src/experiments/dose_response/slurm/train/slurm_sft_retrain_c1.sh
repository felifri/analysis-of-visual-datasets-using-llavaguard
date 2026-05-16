#!/bin/bash
# Re-run SFT for C1 (corrupt denoiser.pt)
#SBATCH --job-name=dose-sft-C1
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

set -euo pipefail
CONDITION=C1
echo "SFT re-training ${CONDITION} on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

export FSDP_VERSION=2
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1

COMPOSER=<your folder>
PYTHON=<your folder>
BASE="<your folder>"
SAVE_DIR="${BASE}/checkpoints_sft/${CONDITION}"
LOAD_PATH="${BASE}/checkpoints_full/${CONDITION}/phase1/latest-rank0.pt"
SCRIPT_DIR="<your folder>"

cd <your folder>

$COMPOSER -n 8 -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_alchemist" \
    "dataset@dataset.eval_dataset=train_alchemist" \
    name=${CONDITION}_sft group=dose-response image_size=512 \
    global_batch_size=256 device_train_microbatch_size=32 \
    trainer.max_duration=20_000ba \
    trainer.save_folder=${SAVE_DIR} \
    trainer.save_interval=5_000ba \
    trainer.run_name=dose-response-sft-${CONDITION} \
    +trainer.load_path=${LOAD_PATH} \
    +trainer.load_weights_only=true \
    +trainer.load_strict_model_weights=false \
    '+trainer.load_ignore_keys=["state/model/vae*","state/model/text_tower*"]' \
    "~algorithms.repa" "~algorithms.pdino" \
    eval_first=false trainer.eval_interval=0 seed=42 \
    +optimizer.lr=5e-5

echo "Training complete. Consolidating..."
$PYTHON ${SCRIPT_DIR}/consolidate_checkpoints.py \
    --ckpt_root "${BASE}/checkpoints_sft" \
    --conditions ${CONDITION} \
    --output_name denoiser.pt

echo "Done at $(date)"
