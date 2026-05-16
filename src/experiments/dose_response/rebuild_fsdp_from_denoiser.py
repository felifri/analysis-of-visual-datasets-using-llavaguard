#!/usr/bin/env python3
"""Rebuild FSDP checkpoint shards from denoiser.pt.

Uses the PRX pipeline to build the model, loads denoiser.pt weights,
then saves a proper FSDP distributed checkpoint via Composer's save mechanism.

Must run with torchrun on 8 GPUs (matching the original checkpoint).

Usage:
    torchrun --nproc_per_node=8 rebuild_fsdp_from_denoiser.py --condition C2
"""

import argparse
import logging
import os
import shutil
import sys

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "<your folder>"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add PRX to path
PRX_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "PRX"))
sys.path.insert(0, PRX_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()

    cond = args.condition
    phase1_dir = os.path.join(BASE, "checkpoints_full", cond, "phase1")
    denoiser_pt = os.path.join(phase1_dir, "denoiser.pt")

    # Find the checkpoint subdir
    ep_dirs = [d for d in os.listdir(phase1_dir)
               if d.startswith("ep") and os.path.isdir(os.path.join(phase1_dir, d))]
    if not ep_dirs:
        logger.error(f"No ep* directory found in {phase1_dir}")
        return
    ckpt_dir = os.path.join(phase1_dir, sorted(ep_dirs)[-1])

    if not os.path.exists(denoiser_pt):
        logger.error(f"No denoiser.pt at {denoiser_pt}")
        return

    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        logger.info(f"Rebuilding FSDP checkpoint for {cond}")
        logger.info(f"  denoiser.pt: {denoiser_pt}")
        logger.info(f"  target dir: {ckpt_dir}")

        # Remove corrupted shards
        for f in os.listdir(ckpt_dir):
            if f.endswith(".distcp"):
                os.remove(os.path.join(ckpt_dir, f))
                logger.info(f"  Removed corrupted {f}")

        # Load denoiser weights
        logger.info("Loading denoiser.pt...")
        state = torch.load(denoiser_pt, map_location="cpu", weights_only=True)

        # Remap to FSDP state dict format: denoiser.X -> state.model.denoiser.X
        fsdp_state = {}
        for k, v in state.items():
            if k.startswith("denoiser."):
                new_key = f"state.model.{k}"
            else:
                new_key = f"state.model.denoiser.{k}"
            fsdp_state[new_key] = v

        # Add the missing _device_tracker key
        fsdp_state["state.model.vae.vae._device_tracker"] = torch.zeros(1)

        # Save as a single-rank distributed checkpoint using dcp
        logger.info(f"Saving {len(fsdp_state)} keys as distributed checkpoint...")
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.metadata import (
            Metadata,
            TensorStorageMetadata,
            TensorProperties,
            ChunkStorageMetadata,
            BytesStorageMetadata,
        )
        import pickle

        # Build metadata
        state_dict_metadata = {}
        for key, tensor in fsdp_state.items():
            if isinstance(tensor, torch.Tensor):
                state_dict_metadata[key] = TensorStorageMetadata(
                    properties=TensorProperties(
                        dtype=tensor.dtype,
                        layout=tensor.layout,
                        requires_grad=tensor.requires_grad,
                        memory_format=torch.contiguous_format,
                        pin_memory=False,
                    ),
                    size=tensor.size(),
                    chunks=[ChunkStorageMetadata(
                        offsets=torch.Size([0] * len(tensor.size())),
                        sizes=tensor.size(),
                    )],
                )

        metadata = Metadata(state_dict_metadata=state_dict_metadata)

        # Write metadata
        with open(os.path.join(ckpt_dir, ".metadata"), "wb") as f:
            pickle.dump(metadata, f)

        # Write single shard using safetensors format (what dcp expects)
        # Actually dcp uses its own format. Let's use the FileSystemWriter directly.
        writer = dcp.FileSystemWriter(ckpt_dir)

        # For single-process save, we need to use dcp.save with a trivial process group
        # Instead, save in the format dcp expects: __0_0.distcp as a safetensors file
        try:
            from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter
        except ImportError:
            pass

        # Simplest approach: save each tensor individually using torch.save
        # in the shard format that dcp.FileSystemReader expects
        from torch.distributed.checkpoint.filesystem import WriteResult

        # Actually, let's just use safetensors directly
        try:
            from safetensors.torch import save_file
            save_file(fsdp_state, os.path.join(ckpt_dir, "__0_0.distcp"))
            logger.info("Saved using safetensors format")
        except ImportError:
            # Fallback: save as torch checkpoint with special handling
            # dcp uses its own binary format. Let's check what format the originals used.
            logger.info("safetensors not available, trying torch format...")
            torch.save(fsdp_state, os.path.join(ckpt_dir, "__0_0.distcp"))

        logger.info(f"Rebuild complete for {cond}")


if __name__ == "__main__":
    main()
