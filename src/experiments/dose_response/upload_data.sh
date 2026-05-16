#!/bin/bash
# Upload generated images, output annotations, and dataset annotations to HuggingFace.
# Usage: bash upload_data.sh
#
# Prerequisites:
#   pip install huggingface_hub[cli]
#   huggingface-cli login
#   python normalize_annotations.py  (run first to create unified parquet files)
#
# Creates 3 HuggingFace dataset repos:
#   1. anonym371/dose-response-generated-images   (generated images + safety annotations)
#   2. anonym371/dose-response-training-annotations (training data safety annotations, no images)
#   3. anonym371/prx-generated-images              (existing PRX model generated images + annotations)

set -e

HF_USER="anonym371"
BASE="<your folder>"
NORMALIZED="${BASE}/results/normalized_annotations"

# ============================================================
# 1. Dose-response generated images + output annotations
# ============================================================
REPO1="${HF_USER}/dose-response-generated-images"
echo ""
echo "============================================================"
echo "Uploading dose-response generated images -> ${REPO1}"
echo "============================================================"

huggingface-cli repo create dose-response-generated-images --type dataset --exist-ok

# Create README
cat > /tmp/README_gen_images.md <<'READMEEOF'
---
license: apache-2.0
tags:
  - diffusion
  - text-to-image
  - safety
  - dose-response
  - image-generation
size_categories:
  - 10K<n<100K
---

# Dose-Response Generated Images

Generated images and safety annotations from the dose-response experiment studying how unsafe training data fraction affects text-to-image model output safety.

## Dataset Description

Each condition's model was used to generate 10,000 images from the same prompt set (1K safe + 9K unsafe prompts). All images were annotated for safety using 4 independent safety evaluators.

## Structure

```
images/
  C1/images.tar          # 10K images from C1 model (0% unsafe training data)
  C2/images.tar          # 10K images from C2 model (5% unsafe)
  C0/images.tar          # 10K images from C0 model (~1.21% unsafe, original)
  C4/images.tar          # 10K images from C4 model (~1.21% unsafe, 1M scale)
  C6/images.tar          # 10K images from C6 model (~9.6% unsafe, 1M scale)
  C5/images.tar          # 10K images from C5 model (~1.21% unsafe, 100K scale)
annotations/
  dose_C1.parquet        # Unified annotations from all 4 evaluators
  dose_C2.parquet
  dose_C0.parquet
  dose_C4.parquet
  dose_C6.parquet
  dose_C5.parquet
```

## Annotation Format

Each parquet file is indexed by image ID and contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `llavaguard_unsafe` | int | 0 = safe, 1 = unsafe (LlavaGuard-7B) |
| `llavaguard_category` | str | Safety category (e.g. "O3: Sexual Content", "NA: None applying") |
| `llamaguard3_unsafe` | int | 0 = safe, 1 = unsafe (LlamaGuard-3-11B-Vision) |
| `llamaguard3_category` | str | MLCommons category (e.g. "S12: Sexual Content", "NA") |
| `shieldgemma_unsafe` | int | 0 = safe, 1 = unsafe (ShieldGemma-2-4B) |
| `shieldgemma_category` | str | Highest-probability policy violated ("dangerous", "sexual", "violence", or "NA") |
| `shieldgemma_p_dangerous` | float | P(dangerous content) |
| `shieldgemma_p_sexual` | float | P(sexual content) |
| `shieldgemma_p_violence` | float | P(violent content) |
| `sd_safety_checker_unsafe` | int | 0 = safe, 1 = unsafe (SD Safety Checker) |

## Safety Evaluators

| Evaluator | Model | Categories |
|-----------|-------|------------|
| [LlavaGuard-7B](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-7B-OV) | Vision-language safety model | O1–O9 (9 categories) |
| [LlamaGuard-3-11B-Vision](https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision) | Meta's multimodal safety model | S1–S14 (MLCommons taxonomy) |
| [ShieldGemma-2-4B](https://huggingface.co/google/shieldgemma-2-4b) | Google's safety classifier | dangerous, sexual, violence |
| SD Safety Checker | CompVis CLIP-based NSFW classifier | binary (safe/unsafe) |

## Generation Settings

- **Prompts**: 10,000 (1K safe + 9K unsafe across 9 categories)
- **Guidance scale**: 3.5
- **Inference steps**: 50
- **Seed**: 42
- **Resolution**: 512px

## Related Resources

- [diffusion_safety](https://github.com/<your github user>/diffusion_safety) — experiment code
- Model checkpoints: `anonym371/dose-response-c1` through `anonym371/dose-response-c5`
READMEEOF

huggingface-cli upload "${REPO1}" /tmp/README_gen_images.md README.md --repo-type dataset

# Upload generated images
for C in C1 C2 C0 C4 C6 C5; do
  if [ -f "${BASE}/generated_images/dose_response/${C}/images.tar" ]; then
    echo "Uploading ${C} images..."
    huggingface-cli upload "${REPO1}" \
      "${BASE}/generated_images/dose_response/${C}/images.tar" \
      "images/${C}/images.tar" --repo-type dataset
  fi
done

# Upload normalized annotations
for C in C1 C2 C0 C4 C6 C5; do
  if [ -f "${NORMALIZED}/dose_${C}.parquet" ]; then
    echo "Uploading ${C} annotations..."
    huggingface-cli upload "${REPO1}" \
      "${NORMALIZED}/dose_${C}.parquet" \
      "annotations/dose_${C}.parquet" --repo-type dataset
  fi
done

rm -f /tmp/README_gen_images.md
echo "Done with dose-response generated images."

# ============================================================
# 2. PRX existing model generated images + annotations
# ============================================================
REPO2="${HF_USER}/prx-generated-images"
echo ""
echo "============================================================"
echo "Uploading PRX existing model images -> ${REPO2}"
echo "============================================================"

huggingface-cli repo create prx-generated-images --type dataset --exist-ok

cat > /tmp/README_prx_images.md <<'READMEEOF'
---
license: apache-2.0
tags:
  - diffusion
  - text-to-image
  - safety
  - prx
size_categories:
  - 10K<n<100K
---

# PRX Generated Images

Generated images and safety annotations from existing PRX model checkpoints, used as baselines in the dose-response experiment.

## Structure

```
images/
  prx-1024-beta/images.tar
  prx-256-base/images.tar
  prx-256-sft/images.tar
  prx-512-base/images.tar
  prx-512-dc-ae/images.tar
  prx-512-sft/images.tar
  prx-512-sft-distilled/images.tar
annotations/
  prx-1024-beta.parquet
  prx-256-base.parquet
  prx-256-sft.parquet
  prx-512-base.parquet
  prx-512-dc-ae.parquet
  prx-512-sft.parquet
  prx-512-sft-distilled.parquet
```

## Models

| Model | Source |
|-------|--------|
| prx-1024-beta | [Photoroom/prx-1024-t2i-beta](https://huggingface.co/Photoroom/prx-1024-t2i-beta) |
| prx-512-base | [Photoroom/prx-512-t2i](https://huggingface.co/Photoroom/prx-512-t2i) |
| prx-512-sft | [Photoroom/prx-512-t2i-sft](https://huggingface.co/Photoroom/prx-512-t2i-sft) |
| prx-512-sft-distilled | [Photoroom/prx-512-t2i-sft-distilled](https://huggingface.co/Photoroom/prx-512-t2i-sft-distilled) |
| prx-512-dc-ae | [Photoroom/prx-512-t2i-dc-ae](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae) |
| prx-256-base | [Photoroom/prx-256-t2i](https://huggingface.co/Photoroom/prx-256-t2i) |
| prx-256-sft | [Photoroom/prx-256-t2i-sft](https://huggingface.co/Photoroom/prx-256-t2i-sft) |

## Annotation Format

Each parquet file is indexed by image ID and contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `llavaguard_unsafe` | int | 0 = safe, 1 = unsafe (LlavaGuard-7B) |
| `llavaguard_category` | str | Safety category (e.g. "O3: Sexual Content", "NA: None applying") |
| `llamaguard3_unsafe` | int | 0 = safe, 1 = unsafe (LlamaGuard-3-11B-Vision) |
| `llamaguard3_category` | str | MLCommons category (e.g. "S12: Sexual Content", "NA") |
| `shieldgemma_unsafe` | int | 0 = safe, 1 = unsafe (ShieldGemma-2-4B) |
| `shieldgemma_category` | str | Highest-probability policy violated ("dangerous", "sexual", "violence", or "NA") |
| `shieldgemma_p_dangerous` | float | P(dangerous content) |
| `shieldgemma_p_sexual` | float | P(sexual content) |
| `shieldgemma_p_violence` | float | P(violent content) |
| `sd_safety_checker_unsafe` | int | 0 = safe, 1 = unsafe (SD Safety Checker) |

## Safety Evaluators

| Evaluator | Model | Categories |
|-----------|-------|------------|
| [LlavaGuard-7B](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-7B-OV) | Vision-language safety model | O1–O9 (9 categories) |
| [LlamaGuard-3-11B-Vision](https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision) | Meta's multimodal safety model | S1–S14 (MLCommons taxonomy) |
| [ShieldGemma-2-4B](https://huggingface.co/google/shieldgemma-2-4b) | Google's safety classifier | dangerous, sexual, violence |
| SD Safety Checker | CompVis CLIP-based NSFW classifier | binary (safe/unsafe) |

## Generation Settings

- **Prompts**: 10,000 (1K safe + 9K unsafe across 9 categories)
- **Guidance scale**: 3.5
- **Inference steps**: 50
- **Seed**: 42

## Related Resources

- [diffusion_safety](https://github.com/<your github user>/diffusion_safety) — experiment code
READMEEOF

huggingface-cli upload "${REPO2}" /tmp/README_prx_images.md README.md --repo-type dataset

PRX_MODELS=(prx-1024-beta prx-256-base prx-256-sft prx-512-base prx-512-dc-ae prx-512-sft prx-512-sft-distilled)
for M in "${PRX_MODELS[@]}"; do
  if [ -f "${BASE}/generated_images/prx_existing/${M}/images.tar" ]; then
    echo "Uploading ${M} images..."
    huggingface-cli upload "${REPO2}" \
      "${BASE}/generated_images/prx_existing/${M}/images.tar" \
      "images/${M}/images.tar" --repo-type dataset
  fi
  if [ -f "${NORMALIZED}/${M}.parquet" ]; then
    echo "Uploading ${M} annotations..."
    huggingface-cli upload "${REPO2}" \
      "${NORMALIZED}/${M}.parquet" \
      "annotations/${M}.parquet" --repo-type dataset
  fi
done

rm -f /tmp/README_prx_images.md
echo "Done with PRX generated images."

# ============================================================
# 3. Training data annotations (no images, just annotations with IDs)
# ============================================================
REPO3="${HF_USER}/dose-response-training-annotations"
echo ""
echo "============================================================"
echo "Uploading training data annotations -> ${REPO3}"
echo "============================================================"

huggingface-cli repo create dose-response-training-annotations --type dataset --exist-ok

cat > /tmp/README_train_annot.md <<'READMEEOF'
---
license: apache-2.0
tags:
  - safety
  - image-classification
  - llavaguard
  - dose-response
size_categories:
  - 1M<n<10M
---

# Dose-Response Training Data Safety Annotations

Safety annotations for the training datasets used in the dose-response experiment. Contains only annotation metadata indexed by image ID — **no images are included**.

## Dataset Description

~8.7M images from 3 source datasets were annotated for safety using [LlavaGuard-7B](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-7B-OV). A second full pass was done with Gemini. These annotations were used to construct training subsets with controlled unsafe content fractions (0%–10%).

Additionally, a 10K sample (5K safe + 5K unsafe per LlavaGuard) was cross-evaluated by all 4 safety judges.

## Source Datasets

| Dataset | Size | Source |
|---------|------|--------|
| [lehduong/flux_generated](https://huggingface.co/datasets/lehduong/flux_generated) | ~1.7M | FLUX-generated |
| [LucasFang/FLUX-Reason-6M](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M) | ~6M | FLUX-generated with reasoning |
| [brivangl/midjourney-v6-llava](https://huggingface.co/datasets/brivangl/midjourney-v6-llava) | ~1M | Midjourney v6 |

## Structure

```
llavaguard/                            # Full LlavaGuard-7B annotations (~8.7M images)
  shard_0.part_01_of_10.parquet
  ...
  shard_7.part_10_of_10.parquet
gemini/                                # Full Gemini annotations (~8.7M images)
  shard_0.part_01_of_10.parquet
  ...
  shard_7.part_10_of_10.parquet
cross_judge/
  cross_judge_training_sample.parquet  # 10K sample annotated by all 4 evaluators
```

## Annotation Format

### LlavaGuard / Gemini (full annotations)

Each parquet file is indexed by image ID and contains columns:
- `rating`: Safe / Unsafe
- `category`: O1–O9 safety category or "NA: None applying"
- `rationale`: Free-text explanation

### Cross-Judge Sample (unified format)

| Column | Type | Description |
|--------|------|-------------|
| `llavaguard_unsafe` | int | 0 = safe, 1 = unsafe (LlavaGuard-7B) |
| `llavaguard_category` | str | Safety category (e.g. "O3: Sexual Content") |
| `llamaguard3_unsafe` | int | 0 = safe, 1 = unsafe (LlamaGuard-3-11B-Vision) |
| `llamaguard3_category` | str | MLCommons category (e.g. "S12: Sexual Content") |
| `shieldgemma_unsafe` | int | 0 = safe, 1 = unsafe (ShieldGemma-2-4B) |
| `shieldgemma_category` | str | Primary violated policy ("dangerous", "sexual", "violence", or "NA") |
| `shieldgemma_p_dangerous` | float | P(dangerous content) |
| `shieldgemma_p_sexual` | float | P(sexual content) |
| `shieldgemma_p_violence` | float | P(violent content) |
| `sd_safety_checker_unsafe` | int | 0 = safe, 1 = unsafe (SD Safety Checker) |

Image IDs encode the source dataset and shard, e.g. `lehduong__flux_generated__train-00232-of-00271__004232`.

## Safety Evaluators

| Evaluator | Model | Categories |
|-----------|-------|------------|
| [LlavaGuard-7B](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-7B-OV) | Vision-language safety model | O1–O9 (9 categories) |
| [LlamaGuard-3-11B-Vision](https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision) | Meta's multimodal safety model | S1–S14 (MLCommons taxonomy) |
| [ShieldGemma-2-4B](https://huggingface.co/google/shieldgemma-2-4b) | Google's safety classifier | dangerous, sexual, violence |
| SD Safety Checker | CompVis CLIP-based NSFW classifier | binary (safe/unsafe) |

## Related Resources

- [diffusion_safety](https://github.com/<your github user>/diffusion_safety) — experiment code
- Generated images: [anonym371/dose-response-generated-images](https://huggingface.co/datasets/anonym371/dose-response-generated-images)
READMEEOF

huggingface-cli upload "${REPO3}" /tmp/README_train_annot.md README.md --repo-type dataset

# Upload LlavaGuard annotations (full)
echo "Uploading LlavaGuard training annotations..."
huggingface-cli upload "${REPO3}" \
  "${BASE}/annotations/annotations_parquet" \
  llavaguard/ --repo-type dataset

# Upload Gemini annotations (full)
echo "Uploading Gemini training annotations..."
huggingface-cli upload "${REPO3}" \
  "${BASE}/annotations/annotations_parquet_gemini" \
  gemini/ --repo-type dataset

# Upload cross-judge training sample
if [ -f "${NORMALIZED}/cross_judge_training_sample.parquet" ]; then
  echo "Uploading cross-judge training sample..."
  huggingface-cli upload "${REPO3}" \
    "${NORMALIZED}/cross_judge_training_sample.parquet" \
    "cross_judge/cross_judge_training_sample.parquet" --repo-type dataset
fi

rm -f /tmp/README_train_annot.md
echo "Done with training annotations."

echo ""
echo "============================================================"
echo "All uploads complete!"
echo "============================================================"
echo ""
echo "Repos created:"
echo "  https://huggingface.co/datasets/${REPO1}"
echo "  https://huggingface.co/datasets/${REPO2}"
echo "  https://huggingface.co/datasets/${REPO3}"
