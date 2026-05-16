#!/bin/bash
# Prepare Alchemist dataset: download images and convert to MDS
#
# Usage:
#   sbatch slurm_prepare_alchemist.sh
#
#SBATCH --job-name=prep-alchemist
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=4:00:00

set -euo pipefail

echo "Preparing Alchemist dataset on $(hostname) at $(date)"

PYTHON=<your folder>
SCRIPT_DIR="<your folder>"

cd ${SCRIPT_DIR}

$PYTHON entrypoint_prepare_alchemist.py --workers 32

echo "Alchemist preparation complete at $(date)"
