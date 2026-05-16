#!/bin/bash
# Re-run SFT training for C2-C6 to regenerate denoiser.pt
# SFT: 20K steps, batch 256, 512px, Alchemist dataset
# Starts from pretrained phase1 checkpoints (FSDP format)
# After training, consolidates FSDP checkpoint to denoiser.pt
#
# Usage:
#   sbatch slurm_sft_retrain.sh          # all C2-C6
#   sbatch --array=0 slurm_sft_retrain.sh  # only C2
#
#SBATCH --job-name=dose-sft-retrain
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --array=0-4

set -euo pipefail

CONDITIONS=(C2 C3 C0 C4 C6)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

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

# Verify pretrained checkpoint exists
if [ ! -e "${LOAD_PATH}" ]; then
    echo "ERROR: Pretrained checkpoint not found at ${LOAD_PATH}"
    exit 1
fi

cd <your folder>

echo "Phase: SFT on Alchemist, 20K steps from pretrained ${CONDITION}"
echo "Load path: ${LOAD_PATH}"
echo "Save dir: ${SAVE_DIR}"

# Key fix: use load_ignore_keys to skip keys the model has but the old
# checkpoint doesn't (e.g. vae._device_tracker added in newer model version).
# The glob pattern "state/model/vae*" removes all VAE keys from the target
# state dict before dcp.load(), so dcp won't error on missing keys.
# The VAE (IdentityVAE) has no learnable weights, so ignoring it is safe.
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

echo "SFT training complete at $(date). Consolidating checkpoint..."

# Find the latest FSDP checkpoint dir
LATEST_CKPT=$(readlink -f "${SAVE_DIR}/latest-rank0.pt" 2>/dev/null)
if [ -z "$LATEST_CKPT" ] || [ ! -d "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(ls -d ${SAVE_DIR}/ep*-ba* 2>/dev/null | sort | tail -1)
fi

if [ -z "$LATEST_CKPT" ] || [ ! -d "$LATEST_CKPT" ]; then
    echo "ERROR: No FSDP checkpoint dir found in ${SAVE_DIR}"
    exit 1
fi

echo "Consolidating FSDP checkpoint from ${LATEST_CKPT}..."
$PYTHON ${SCRIPT_DIR}/consolidate_checkpoints.py \
    --ckpt_root "${BASE}/checkpoints_sft" \
    --conditions ${CONDITION} \
    --output_name denoiser.pt

# Verify denoiser.pt was created
if [ -f "${SAVE_DIR}/denoiser.pt" ]; then
    SIZE=$(du -sh "${SAVE_DIR}/denoiser.pt" | cut -f1)
    echo "SUCCESS: denoiser.pt created (${SIZE})"
else
    echo "Retrying consolidation from ${LATEST_CKPT}..."
    $PYTHON -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from consolidate_checkpoints import consolidate
consolidate('${LATEST_CKPT}', '${SAVE_DIR}/denoiser.pt')
"
fi

echo "SFT re-training + consolidation for ${CONDITION} complete at $(date)"
