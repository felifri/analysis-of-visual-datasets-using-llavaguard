#!/bin/bash
#SBATCH --job-name=dose-train-ms
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --array=0-7

# Multi-seed training for C1 (original, 1.21% unsafe) and C0 (filtered, 0% unsafe)
# in internal naming. Paper: C1=internal C0, C2=internal C1.
# Array mapping:
#   0: C1 seed=137    (internal C1 = paper C2, 0% unsafe)
#   1: C1 seed=314
#   2: C1 seed=789
#   3: C1 seed=1331
#   4: C0 seed=137    (internal C0 = paper C1, 1.21% unsafe)
#   5: C0 seed=314
#   6: C0 seed=789
#   7: C0 seed=1331

set -euo pipefail

CONDITIONS=(C1 C1 C1 C1 C0 C0 C0 C0)
SEEDS=(137 314 789 1331 137 314 789 1331)
DATASETS=(train_dose_c1 train_dose_c1 train_dose_c1 train_dose_c1 train_dose_c0 train_dose_c0 train_dose_c0 train_dose_c0)

CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Training ${CONDITION} seed=${SEED} started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

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
    "dataset@dataset.train_dataset=${DATASET}" \
    "dataset@dataset.eval_dataset=${DATASET}" \
    name=${CONDITION}_seed_${SEED} group=dose-response-multiseed image_size=512 \
    global_batch_size=256 device_train_microbatch_size=32 \
    trainer.max_duration=100_000ba \
    trainer.save_folder=${CKPT_DIR}/${CONDITION}_seed_${SEED}/phase1 \
    trainer.run_name=dose-response-${CONDITION}-seed${SEED}-phase1 \
    "~algorithms.repa" "~algorithms.pdino" \
    eval_first=false trainer.eval_interval=0 seed=${SEED}

echo "${CONDITION} seed=${SEED} Phase 1 complete at $(date)"
