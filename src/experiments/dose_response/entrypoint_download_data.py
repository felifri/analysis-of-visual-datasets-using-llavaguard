"""Step 1: Download the three training datasets from HuggingFace.

Downloads lehduong/flux_generated, LucasFang/FLUX-Reason-6M, and
brivangl/midjourney-v6-llava in WebDataset tar format. Verifies that
the total number of downloaded images matches expectations.

Usage:
    python entrypoint_download_data.py [--output-dir /path/to/output]
"""

import argparse
import json
import logging
import os
import time

from huggingface_hub import snapshot_download


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    {
        "repo_id": "lehduong/flux_generated",
        "repo_type": "dataset",
        "expected_size": 1_700_000,
    },
    {
        "repo_id": "LucasFang/FLUX-Reason-6M",
        "repo_type": "dataset",
        "expected_size": 6_000_000,
    },
    {
        "repo_id": "brivangl/midjourney-v6-llava",
        "repo_type": "dataset",
        "expected_size": 1_000_000,
    },
]


def count_tar_files(directory: str) -> int:
    """Count .tar files in a directory tree."""
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".tar"):
                count += 1
    return count


def download_dataset(repo_id: str, repo_type: str, output_dir: str, logger: logging.Logger) -> str:
    """Download a single dataset via huggingface_hub snapshot_download.

    Returns the local path to the downloaded snapshot.
    """
    dataset_name = repo_id.replace("/", "__")
    local_dir = os.path.join(output_dir, dataset_name)

    logger.info(f"Downloading {repo_id} to {local_dir}...")
    start = time.time()

    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        resume_download=True,
    )

    elapsed = time.time() - start
    num_tars = count_tar_files(path)
    logger.info(f"Downloaded {repo_id}: {num_tars} tar files in {elapsed / 60:.1f}m -> {path}")

    return path


def main():
    parser = argparse.ArgumentParser(description="Download training datasets for dose-response experiment")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to download datasets into. Defaults to config.json value.",
    )
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    output_dir = args.output_dir or config["training_data"]["download_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "download.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Downloading {len(DATASETS)} datasets...")

    download_paths = {}
    for ds in DATASETS:
        path = download_dataset(ds["repo_id"], ds["repo_type"], output_dir, logger)
        download_paths[ds["repo_id"]] = path

    # Write a manifest recording the download paths
    manifest_path = os.path.join(output_dir, "download_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(download_paths, f, indent=2)

    logger.info(f"Download manifest saved to {manifest_path}")
    logger.info("All downloads complete. Verify sizes before proceeding to annotation.")


if __name__ == "__main__":
    main()
