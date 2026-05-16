#!/bin/bash
# Upload consolidated and distributed checkpoints to HuggingFace with model cards.
# Usage: bash upload_checkpoints.sh
#
# Prerequisites:
#   pip install huggingface_hub[cli]
#   huggingface-cli login
#   python consolidate_checkpoints.py  (run first to create denoiser.pt files)

set -e

HF_USER="anonym371"
CKPT_ROOT="<your folder>"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A EPOCHS=(
  [C1]=ep3-ba100000
  [C2]=ep3-ba100000
  [C3]=ep2-ba90000
  [C0]=ep3-ba100000
  [C4]=ep25-ba100000
  [C6]=ep25-ba100000
  [C5]=ep266-ba100000
)

declare -A LABELS=(
  [C1]="0% unsafe, full scale"
  [C2]="5% unsafe, full scale"
  [C3]="10% unsafe, full scale"
  [C0]="Original composition (~1.21% unsafe), full scale"
  [C4]="Original proportion (~1.21% unsafe), 1M scale"
  [C6]="All unsafe included (~9.6% unsafe), 1M scale"
  [C5]="Original proportion (~1.21% unsafe), 100K scale"
)

declare -A UNSAFE_FRACS=(
  [C1]="0%"
  [C2]="5%"
  [C3]="10%"
  [C0]="~1.21% (original)"
  [C4]="~1.21% (original)"
  [C6]="~9.6%"
  [C5]="~1.21% (original)"
)

declare -A TOTAL_SIZES=(
  [C1]="~7.85M images"
  [C2]="~8.24M images"
  [C3]="~8.72M images"
  [C0]="~7.94M images"
  [C4]="1M images"
  [C6]="1M images"
  [C5]="100K images"
)

declare -A DESCRIPTIONS=(
  [C1]="All unsafe images removed. Training uses only the safe pool (7.85M safe images)."
  [C2]="5% unsafe content injected via oversampling (397K unsafe images added to 7.85M safe)."
  [C3]="10% unsafe content injected via oversampling (872K unsafe images added to 7.85M safe)."
  [C0]="Original dataset composition with no filtering. Contains the natural ~1.21% unsafe rate (96K unsafe in 7.94M total)."
  [C4]="Downscaled to 1M images while preserving the original ~1.21% unsafe proportion (12K unsafe, 988K safe)."
  [C6]="Downscaled to 1M images with all 96K original unsafe images included (~9.6% unsafe rate, 904K safe)."
  [C5]="Downscaled to 100K images while preserving the original ~1.21% unsafe proportion (1.2K unsafe, 98.8K safe)."
)

CONDITIONS=(C1 C2 C3 C0 C4 C6 C5)

for C in "${CONDITIONS[@]}"; do
  REPO="${HF_USER}/dose-response-${C,,}"
  EPOCH="${EPOCHS[$C]}"
  PHASE1="${CKPT_ROOT}/${C}/phase1"
  LOWER="${C,,}"

  echo ""
  echo "============================================================"
  echo "Uploading ${C} -> ${REPO}"
  echo "============================================================"

  # Create repo (--exist-ok avoids error if it already exists)
  huggingface-cli repo create "dose-response-${LOWER}" --type model --exist-ok

  # Generate README / model card
  cat > "/tmp/README_${LOWER}.md" <<READMEEOF
---
license: apache-2.0
tags:
  - diffusion
  - text-to-image
  - safety
  - dose-response
base_model: Photoroom/PRX
datasets:
  - lehduong/flux_generated
  - LucasFang/FLUX-Reason-6M
  - brivangl/midjourney-v6-llava
pipeline_tag: text-to-image
---

# Dose-Response ${C}: ${LABELS[$C]}

This model is part of a **dose-response experiment** studying how the fraction of unsafe content in training data affects the safety of generated images from text-to-image diffusion models.

## Model Details

| | |
|---|---|
| **Architecture** | PRX-1.2B (Photoroom diffusion model) |
| **Parameters** | 1.2B (denoiser only) |
| **Resolution** | 512px |
| **Condition** | ${C} — ${LABELS[$C]} |
| **Unsafe fraction** | ${UNSAFE_FRACS[$C]} |
| **Training set size** | ${TOTAL_SIZES[$C]} |
| **Training steps** | 100K batches |
| **Batch size** | 1024 (global) |
| **Precision** | bf16 |
| **Hardware** | 8x H200 GPUs |

## Condition Description

${DESCRIPTIONS[$C]}

## Dose-Response Conditions Overview

This model is one of 7 conditions in the dose-response experiment:

| Condition | Unsafe Fraction | Dataset Scale | Description |
|-----------|----------------|---------------|-------------|
| **C1** | 0% | Full (~7.85M) | All unsafe removed |
| **C2** | 5% | Full (~8.24M) | Unsafe oversampled to 5% |
| **C3** | 10% | Full (~8.72M) | Unsafe oversampled to 10% |
| **C0** | ~1.21% | Full (~7.94M) | Original composition |
| **C4** | ~1.21% | 1M | Original proportion, downscaled |
| **C6** | ~9.6% | 1M | All unsafe included, downscaled |
| **C5** | ~1.21% | 100K | Original proportion, small scale |

## Training Details

- **Base architecture**: [PRX](https://github.com/Photoroom/PRX) 1.2B
- **Text encoder**: T5-Gemma-2B (frozen)
- **VAE**: Identity (no compression)
- **Optimizer**: Muon
- **Algorithms**: TREAD + REPA-v3 + LPIPS + Perceptual DINO + EMA
- **EMA smoothing**: 0.999 (updated every 10 batches)
- **Training data sources**: \`lehduong/flux_generated\`, \`LucasFang/FLUX-Reason-6M\`, \`brivangl/midjourney-v6-llava\`
- **Safety annotations**: Training data annotated with [LlavaGuard-7B](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-7B-OV) to classify images as safe/unsafe

## Files

- \`denoiser.pt\` — Consolidated single-file checkpoint (EMA weights, ready for inference)
- \`distributed/\` — Original FSDP distributed checkpoint shards
- \`config.yaml\` — Full Hydra training configuration

## Usage

\`\`\`python
import torch

# Load consolidated checkpoint
state_dict = torch.load("denoiser.pt", map_location="cpu")
# Keys are in format: denoiser.*
\`\`\`

For the full generation pipeline, see the [diffusion_safety](https://github.com/<your github user>/diffusion_safety) repository.

## Citation

If you use these models, please cite the associated paper and the PRX architecture.

## License

Apache 2.0
READMEEOF

  # Upload README
  huggingface-cli upload "${REPO}" \
    "/tmp/README_${LOWER}.md" \
    README.md --repo-type model

  # Upload consolidated checkpoint
  if [ -f "${PHASE1}/denoiser.pt" ]; then
    echo "Uploading consolidated checkpoint..."
    huggingface-cli upload "${REPO}" \
      "${PHASE1}/denoiser.pt" \
      denoiser.pt --repo-type model
  else
    echo "WARNING: ${PHASE1}/denoiser.pt not found. Run consolidate_checkpoints.py first."
  fi

  # Upload distributed checkpoint
  if [ -d "${PHASE1}/${EPOCH}" ]; then
    echo "Uploading distributed checkpoint (${EPOCH})..."
    huggingface-cli upload "${REPO}" \
      "${PHASE1}/${EPOCH}" \
      distributed/ --repo-type model
  else
    echo "WARNING: ${PHASE1}/${EPOCH} not found, skipping distributed."
  fi

  # Upload config
  if [ -f "${PHASE1}/config.yaml" ]; then
    echo "Uploading config..."
    huggingface-cli upload "${REPO}" \
      "${PHASE1}/config.yaml" \
      config.yaml --repo-type model
  fi

  # Clean up temp README
  rm -f "/tmp/README_${LOWER}.md"

  echo "Done with ${C}."
done

echo ""
echo "All uploads complete!"
