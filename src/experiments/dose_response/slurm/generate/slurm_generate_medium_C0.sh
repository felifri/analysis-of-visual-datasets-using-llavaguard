#!/bin/bash
#SBATCH --job-name=gen-medium-C0
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00

# Generate 10K evaluation images from medium PRX-3.6B C0 checkpoint

set -euo pipefail

echo "Medium C0 generation started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

CKPT_ROOT="<your folder>"
CKPT_DIR="${CKPT_ROOT}/C0/ep3-ba100000"
CONFIG_YAML="${CKPT_ROOT}/C0/config.yaml"
OUTPUT_BASE="<your folder>"

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_generate_dose_response.py \
    --condition medium_C0 \
    --checkpoint-dir "$CKPT_DIR" \
    --config-yaml "$CONFIG_YAML" \
    --batch-size 4 \
    --guidance-scale 3.5 \
    --num-inference-steps 50 \
    --seed 42

echo "Medium C0 generation completed at $(date)"
