#!/bin/bash
#SBATCH --job-name=dose-C3-mds
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Build MDS pool for C3 (10% unsafe, ~872K oversampled unsafe images)

set -euo pipefail

echo "C3 MDS build started on $(hostname) at $(date)"

PYTHON=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_construct_subsets.py \
    --download-dir <your folder> \
    --parquet-dir <your folder> \
    --output-dir <your folder> \
    --mds-dir <your folder> \
    --seed 42 \
    --build-pool unsafe_c3

echo "C3 MDS build completed at $(date)"
