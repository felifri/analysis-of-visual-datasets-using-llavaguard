#!/bin/bash
#SBATCH --job-name=dose-C5-mds
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=4:00:00

# Build MDS pools for C5 (100K images — should be fast)

set -euo pipefail

echo "C5 MDS build started on $(hostname) at $(date)"

PYTHON=<your folder>

$PYTHON << 'PYEOF'
import json, os, sys, logging

sys.path.insert(0, "<your folder>")
from experiments.dose_response.entrypoint_construct_subsets import write_pool_to_mds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

download_dir = "<your folder>"
mds_dir = "<your folder>"
subsets_dir = "<your folder>"

for pool_name in ["safe_c5", "unsafe_c5"]:
    ids_path = os.path.join(subsets_dir, f"pool_{pool_name}_ids.json")
    with open(ids_path) as f:
        ids = json.load(f)

    pool_mds_dir = os.path.join(mds_dir, pool_name)
    if os.path.exists(pool_mds_dir) and os.listdir(pool_mds_dir):
        logger.info(f"Skipping {pool_name}: already exists")
        continue

    logger.info(f"Building MDS for {pool_name}: {len(ids)} images")
    write_pool_to_mds(pool_name, ids, download_dir, pool_mds_dir, logger)

logger.info("Done.")
PYEOF

echo "C5 MDS build completed at $(date)"
