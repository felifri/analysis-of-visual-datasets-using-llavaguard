#!/bin/bash
#SBATCH --job-name=dose-gen-ts
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --array=0-7

# Generate 10K images from multi-seed trained checkpoints (training-seed variance)
# Uses fixed generation seed=42. Only the training seed varies.
# Internal C1 = paper C2 (0% unsafe), internal C0 = paper C1 (1.21% unsafe)
# Array mapping:
#   0: C1_seed_137     4: C0_seed_137
#   1: C1_seed_314     5: C0_seed_314
#   2: C1_seed_789     6: C0_seed_789
#   3: C1_seed_1331    7: C0_seed_1331

set -euo pipefail

CONDITIONS=(C1_seed_137 C1_seed_314 C1_seed_789 C1_seed_1331 C0_seed_137 C0_seed_314 C0_seed_789 C0_seed_1331)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($CONDITION) started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

# Disable proxy, force offline for model loading
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>

# Set PRX directory
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_generate_dose_response.py \
    --condition $CONDITION \
    --phase phase1 \
    --batch-size 4 \
    --guidance-scale 3.5 \
    --num-inference-steps 50 \
    --seed 42

echo "$CONDITION generation completed at $(date)"
