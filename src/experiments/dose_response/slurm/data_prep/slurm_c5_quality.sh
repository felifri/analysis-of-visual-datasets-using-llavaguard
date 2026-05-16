#!/bin/bash
# Quality metrics for C5 SFT
#
#SBATCH --job-name=qual-c5-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00

set -euo pipefail

echo "C5 SFT quality evaluation started on $(hostname) at $(date)"

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PRX_DIR=<your folder>

COCO_REAL=<your folder>
OUTPUT_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_eval_quality.py \
    --model sft_C5 \
    --metrics clip_score,image_reward,fid,kid,fdd,kdd \
    --reference-dir ${COCO_REAL} \
    --max-images 10000 \
    --batch-size 32 \
    --output-dir ${OUTPUT_DIR}

echo "C5 SFT quality evaluation completed at $(date)"
