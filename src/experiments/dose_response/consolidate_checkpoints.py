"""Consolidate FSDP distributed checkpoints into single .pt files.

Usage:
    python consolidate_checkpoints.py [--conditions C1 C2 C0 C4 C6] [--ckpt_root PATH]

Reads the distributed .distcp shards and saves a consolidated state_dict
as a single denoiser.pt file alongside the distributed checkpoint.
"""

import argparse
import logging
import os
import pickle

import torch
import torch.distributed.checkpoint as dcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CKPT_ROOT = "<your folder>"
DEFAULT_CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]


def consolidate(checkpoint_dir: str, output_path: str):
    """Load FSDP distributed checkpoint and save as a single .pt file."""
    logger.info(f"Loading FSDP checkpoint from {checkpoint_dir}")

    metadata_path = os.path.join(checkpoint_dir, ".metadata")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    all_keys = list(metadata.state_dict_metadata.keys())

    # Prefer EMA denoiser weights
    ema_keys = [k for k in all_keys if "ema_denoiser.model." in k]
    regular_keys = [k for k in all_keys if k.startswith("state.model.denoiser.")]

    if ema_keys:
        logger.info(f"Using EMA denoiser weights ({len(ema_keys)} keys)")
        load_keys = ema_keys
        use_ema = True
    else:
        logger.info(f"Using regular denoiser weights ({len(regular_keys)} keys)")
        load_keys = regular_keys
        use_ema = False

    # Build empty state dict matching shapes
    state_dict = {}
    for key in load_keys:
        tensor_meta = metadata.state_dict_metadata[key]
        if hasattr(tensor_meta, "size"):
            state_dict[key] = torch.empty(tensor_meta.size, dtype=torch.float32)
        elif hasattr(tensor_meta, "properties") and hasattr(tensor_meta.properties, "size"):
            state_dict[key] = torch.empty(tensor_meta.properties.size, dtype=torch.float32)
        else:
            state_dict[key] = torch.tensor(0.0)

    # Load distributed checkpoint
    storage_reader = dcp.FileSystemReader(checkpoint_dir)
    dcp.load(state_dict, storage_reader=storage_reader)

    # Remap keys to pipeline format: denoiser.*
    consolidated = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("state.model."):
            new_key = new_key[len("state.model."):]
        if use_ema and new_key.startswith("ema_denoiser.model."):
            new_key = "denoiser." + new_key[len("ema_denoiser.model."):]
        consolidated[new_key] = value

    logger.info(f"Saving {len(consolidated)} parameters to {output_path}")
    torch.save(consolidated, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved consolidated checkpoint: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Consolidate FSDP checkpoints")
    parser.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--output_name", default="denoiser.pt",
                        help="Filename for the consolidated checkpoint")
    args = parser.parse_args()

    for condition in args.conditions:
        # Try multiple directory layouts:
        #   1. {root}/{cond}/phase1/ep*   (checkpoints_full)
        #   2. {root}/{cond}/ep*          (checkpoints_sft, flat)
        #   3. {root}/{cond}/latest-rank0.pt symlink target
        checkpoint_dir = None
        output_dir = None

        # Layout 1: phase1 subdirectory
        phase1_dir = os.path.join(args.ckpt_root, condition, "phase1")
        if os.path.isdir(phase1_dir):
            candidates = [d for d in os.listdir(phase1_dir) if d.startswith("ep") and os.path.isdir(os.path.join(phase1_dir, d))]
            if candidates:
                checkpoint_dir = os.path.join(phase1_dir, sorted(candidates)[-1])
                output_dir = phase1_dir

        # Layout 2: flat (no phase subdir)
        if checkpoint_dir is None:
            cond_dir = os.path.join(args.ckpt_root, condition)
            if os.path.isdir(cond_dir):
                # Try latest-rank0.pt symlink
                latest = os.path.join(cond_dir, "latest-rank0.pt")
                if os.path.islink(latest):
                    target = os.path.join(cond_dir, os.readlink(latest))
                    if os.path.isdir(target):
                        checkpoint_dir = target
                        output_dir = cond_dir

                # Fall back to any ep* subdir
                if checkpoint_dir is None:
                    candidates = [d for d in os.listdir(cond_dir) if d.startswith("ep") and os.path.isdir(os.path.join(cond_dir, d))]
                    if candidates:
                        checkpoint_dir = os.path.join(cond_dir, sorted(candidates)[-1])
                        output_dir = cond_dir

        if checkpoint_dir is None or output_dir is None:
            logger.warning(f"No checkpoint found for {condition}, skipping")
            continue

        output_path = os.path.join(output_dir, args.output_name)
        logger.info(f"\n{'='*60}\nConsolidating {condition}\n{'='*60}")
        consolidate(checkpoint_dir, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
