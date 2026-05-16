#!/bin/bash
#SBATCH --job-name=dose-gen-tsms
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --array=0-7

# Generate 10K images with 4 additional generation seeds for each training-seed checkpoint.
# Combined with existing gen seed=42, this gives 5 generation seeds per training seed.
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

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_multi_seed.py \
    --conditions $CONDITION \
    --seeds 137 314 789 1331 \
    --phase phase1 \
    --batch-size 4 \
    --guidance-scale 3.5 \
    --num-inference-steps 50

echo "$CONDITION multi-seed generation completed at $(date)"
