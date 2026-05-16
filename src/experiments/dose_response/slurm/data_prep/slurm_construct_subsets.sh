#!/bin/bash
#SBATCH --job-name=dose-subsets
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

PYTHON=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_construct_subsets.py \
    --download-dir <your folder> \
    --parquet-dir <your folder> \
    --output-dir <your folder> \
    --mds-dir <your folder> \
    --seed 42 \
    --conditions C1 C2 C0 C4 C6 \
    --skip-mds

echo "Job completed at $(date)"
