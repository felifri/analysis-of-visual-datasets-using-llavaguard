#!/bin/bash
#SBATCH --job-name=dose-review
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

echo "Manual review extraction started on $(hostname) at $(date)"

PYTHON=<your folder>

cd <your folder>

$PYTHON experiments/dose_response/entrypoint_manual_review.py --seed 42

echo "Done at $(date)"
