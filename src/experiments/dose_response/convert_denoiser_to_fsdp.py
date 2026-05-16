#!/usr/bin/env python3
"""Convert denoiser.pt to FSDP-compatible distributed checkpoint format.

Creates a directory with .distcp shards + .metadata that torch.distributed.checkpoint
can load. This allows Composer with FSDP_VERSION=2 to load from denoiser.pt files.

Usage:
    python convert_denoiser_to_fsdp.py --conditions C2 C3 C0 C4 C6
"""

import argparse
import logging
import os
import pickle

import torch
import torch.distributed.checkpoint as dcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "<your folder>"


def convert(denoiser_pt_path: str, output_dir: str):
    """Convert denoiser.pt to a format loadable by Composer FSDP2."""
    logger.info(f"Loading {denoiser_pt_path}")
    state = torch.load(denoiser_pt_path, map_location="cpu", weights_only=True)

    # Remap keys: denoiser.X -> state.model.denoiser.X
    remapped = {}
    for k, v in state.items():
        if k.startswith("denoiser."):
            new_key = f"state.model.{k}"
        else:
            new_key = f"state.model.denoiser.{k}"
        remapped[new_key] = v

    os.makedirs(output_dir, exist_ok=True)

    # Save as a single-file "distributed" checkpoint
    # dcp.save() requires a process group, so we save manually in the expected format
    save_path = os.path.join(output_dir, "__0_0.distcp")
    torch.save(remapped, save_path)

    # Create .metadata file that dcp expects
    from torch.distributed.checkpoint.metadata import (
        BytesStorageMetadata,
        Metadata,
        TensorStorageMetadata,
    )
    from torch.distributed.checkpoint.planner import SavePlan, WriteItem

    state_dict_metadata = {}
    for key, tensor in remapped.items():
        if isinstance(tensor, torch.Tensor):
            state_dict_metadata[key] = TensorStorageMetadata(
                properties=torch.distributed.checkpoint.metadata.TensorProperties(
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    memory_format=torch.contiguous_format,
                    pin_memory=False,
                ),
                size=tensor.size(),
                chunks=[
                    torch.distributed.checkpoint.metadata.ChunkStorageMetadata(
                        offsets=torch.Size([0] * len(tensor.size())),
                        sizes=tensor.size(),
                    )
                ],
            )

    metadata = Metadata(state_dict_metadata=state_dict_metadata)
    metadata_path = os.path.join(output_dir, ".metadata")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Saved FSDP-compatible checkpoint to {output_dir} ({len(remapped)} keys)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["C2", "C3", "C0", "C4", "C6"])
    parser.add_argument("--ckpt_root", default=f"{BASE}/checkpoints_full")
    parser.add_argument("--output_root", default=f"{BASE}/checkpoints_full")
    args = parser.parse_args()

    for cond in args.conditions:
        denoiser_pt = os.path.join(args.ckpt_root, cond, "phase1", "denoiser.pt")
        output_dir = os.path.join(args.output_root, cond, "phase1", "fsdp_from_denoiser")

        if not os.path.exists(denoiser_pt):
            logger.warning(f"No denoiser.pt for {cond}, skipping")
            continue

        if os.path.exists(os.path.join(output_dir, ".metadata")):
            logger.info(f"{cond}: FSDP checkpoint already exists at {output_dir}, skipping")
            continue

        logger.info(f"\n{'='*60}\nConverting {cond}\n{'='*60}")
        convert(denoiser_pt, output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
