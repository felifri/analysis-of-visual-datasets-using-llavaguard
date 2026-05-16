#!/bin/bash
#SBATCH --job-name=dose-train
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --exclusive

set -euo pipefail

CONDITIONS=(C4 C6)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
CONDITION_LOWER=$(echo $CONDITION | tr '[:upper:]' '[:lower:]')

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($CONDITION) started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID * 100))
export WORLD_SIZE=$SLURM_NTASKS
export FSDP_VERSION=2

# Disable proxy, force offline
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export HYDRA_FULL_ERROR=1

PYTHON=<your folder>
CKPT_DIR="<your folder>"

cd <your folder>

# Phase 1: 512px, 100K steps, batch 1024
echo "Starting Phase 1 for $CONDITION..."
srun $PYTHON -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_dose_${CONDITION_LOWER}" \
    name=${CONDITION} \
    group=dose-response \
    image_size=512 \
    global_batch_size=1024 \
    device_train_microbatch_size=16 \
    trainer.max_duration=100_000ba \
    trainer.save_folder=${CKPT_DIR}/${CONDITION}/phase1 \
    trainer.run_name=dose-response-${CONDITION}-phase1 \
    seed=42

echo "Phase 1 complete for $CONDITION at $(date)"

# Phase 2: 1024px, 20K steps, batch 512, no REPA
echo "Starting Phase 2 for $CONDITION..."
srun $PYTHON -m prx.training.train \
    --config-name dose-response-speedrun \
    "dataset@dataset.train_dataset=train_dose_${CONDITION_LOWER}" \
    name=${CONDITION} \
    group=dose-response \
    image_size=1024 \
    global_batch_size=512 \
    device_train_microbatch_size=8 \
    trainer.max_duration=20_000ba \
    trainer.save_folder=${CKPT_DIR}/${CONDITION}/phase2 \
    trainer.run_name=dose-response-${CONDITION}-phase2 \
    trainer.load_path=${CKPT_DIR}/${CONDITION}/phase1/latest-rank0.pt \
    "~algorithms.repa" \
    seed=42

echo "$CONDITION training complete at $(date)"
