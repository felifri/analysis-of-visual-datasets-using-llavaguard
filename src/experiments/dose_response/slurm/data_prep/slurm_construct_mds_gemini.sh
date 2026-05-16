#!/bin/bash
#SBATCH --job-name=dose-mds-g
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-5

# Rebuild MDS pools with gemini captions (Photoroom/midjourney-v6-recap)
# Uses remapped annotation parquets

set -euo pipefail

POOLS=(safe_full unsafe_full unsafe_c2 unsafe_c4 safe_c4 safe_c6)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($POOL) started on $(hostname) at $(date)"

PYTHON=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_construct_subsets.py \
    --download-dir <your folder> \
    --parquet-dir <your folder> \
    --output-dir <your folder> \
    --mds-dir <your folder> \
    --seed 42 \
    --build-pool $POOL

echo "$POOL completed at $(date)"
