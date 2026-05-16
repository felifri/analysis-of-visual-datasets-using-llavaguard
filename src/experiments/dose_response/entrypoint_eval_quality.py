"""Compute generation quality metrics on saved images.

Evaluates image quality, text-image alignment, and human preference scores
for generated images from dose-response conditions and existing PRX models.

Metrics implemented:
  - CLIP Score: text-image cosine similarity (via transformers CLIPModel)
  - FID: Fréchet Inception Distance (via torchmetrics, requires reference images)
  - CMMD: CLIP Maximum Mean Discrepancy (reuses PRX feature extractors)
  - DINO-MMD: DINOv2-based MMD (reuses PRX feature extractors)
  - ImageReward: learned human preference score (requires pip install image-reward)
  - Aesthetic Score: aesthetic quality predictor

Usage:
    # Evaluate a single model's generated images
    python entrypoint_eval_quality.py --model dose_C1 --metrics clip_score

    # Evaluate all dose-response conditions
    python entrypoint_eval_quality.py --all-dose --metrics clip_score,fid

    # Evaluate existing PRX models
    python entrypoint_eval_quality.py --all-prx --metrics clip_score
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add PRX to path for feature extractors
PRX_DIR = os.environ.get(
    "PRX_DIR",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "PRX")),
)
if PRX_DIR not in sys.path:
    sys.path.insert(0, PRX_DIR)

DOSE_CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]
PRX_MODELS = [
    "prx-1024-beta", "prx-512-base", "prx-512-sft",
    "prx-512-sft-distilled", "prx-512-dc-ae",
    "prx-256-base", "prx-256-sft",
]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images_from_tar(tar_path: str, max_images: int = None) -> list[Image.Image]:
    """Load PIL images from a tar archive."""
    images = []
    with tarfile.open(tar_path, "r:") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")]
        members = sorted(members, key=lambda m: m.name)
        if max_images:
            members = members[:max_images]
        for member in tqdm(members, desc=f"Loading {os.path.basename(tar_path)}", leave=False):
            f = tf.extractfile(member)
            if f:
                img = Image.open(f).convert("RGB")
                images.append(img)
    return images


def load_images_from_dir(img_dir: str, max_images: int = None) -> list[Image.Image]:
    """Load PIL images from a directory."""
    paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if max_images:
        paths = paths[:max_images]
    images = []
    for p in tqdm(paths, desc=f"Loading {os.path.basename(img_dir)}", leave=False):
        images.append(Image.open(p).convert("RGB"))
    return images


def get_images_and_prompts(
    model_id: str, config: dict, max_images: int = None,
) -> tuple[list[Image.Image], list[str]]:
    """Load generated images and their corresponding prompts."""
    # Try dose-response models first, then SFT/RL, then PRX existing
    if model_id.startswith("dose_") or model_id.startswith("C"):
        cond = model_id.replace("dose_", "")
        tar_path = os.path.join(
            config["generation"]["output_dir"], "dose_response", cond, "images.tar"
        )
    elif model_id.startswith("sft_") or model_id.startswith("sft/"):
        cond = model_id.replace("sft_", "").replace("sft/", "")
        tar_path = os.path.join(
            config["generation"]["output_dir"], "sft", cond, "images.tar"
        )
    else:
        tar_path = os.path.join(
            config["generation"]["output_dir"], "prx_existing", model_id, "images.tar"
        )

    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"No images.tar at {tar_path}")

    images = load_images_from_tar(tar_path, max_images)

    # Load prompts
    prompt_df = pd.read_csv(config["generation"]["prompt_file"], index_col=0)
    prompts = prompt_df["prompt"].tolist()
    if max_images:
        prompts = prompts[:max_images]

    # Ensure same length
    n = min(len(images), len(prompts))
    return images[:n], prompts[:n]


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------

def compute_clip_score(
    images: list[Image.Image],
    prompts: list[str],
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Compute CLIP score (text-image cosine similarity)."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    scores = []
    for i in tqdm(range(0, len(images), batch_size), desc="CLIP Score"):
        batch_imgs = images[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        inputs = processor(
            text=batch_prompts, images=batch_imgs,
            return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Per-pair cosine similarity
            img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (img_embeds * txt_embeds).sum(dim=-1)
            scores.extend(similarity.cpu().tolist())

    del model
    torch.cuda.empty_cache()

    return {
        "clip_score_mean": float(np.mean(scores)),
        "clip_score_std": float(np.std(scores)),
        "clip_score_median": float(np.median(scores)),
        "clip_scores": scores,
    }


# ---------------------------------------------------------------------------
# FID (requires reference images)
# ---------------------------------------------------------------------------

def compute_fid(
    generated_images: list[Image.Image],
    reference_images: list[Image.Image],
    device: str = "cuda",
    batch_size: int = 64,
) -> dict:
    """Compute FID between generated and reference image sets."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Add reference (real) images
    for i in tqdm(range(0, len(reference_images), batch_size), desc="FID (reference)"):
        batch = reference_images[i : i + batch_size]
        tensors = torch.stack([transform(img) for img in batch]).to(device)
        fid_metric.update(tensors, real=True)

    # Add generated (fake) images
    for i in tqdm(range(0, len(generated_images), batch_size), desc="FID (generated)"):
        batch = generated_images[i : i + batch_size]
        tensors = torch.stack([transform(img) for img in batch]).to(device)
        fid_metric.update(tensors, real=False)

    fid_value = fid_metric.compute().item()

    del fid_metric
    torch.cuda.empty_cache()

    return {"fid": fid_value}


# ---------------------------------------------------------------------------
# CMMD (reuses PRX CLIPFeatureExtractor)
# ---------------------------------------------------------------------------

def _rbf_kernel_sum_chunked(x: torch.Tensor, y: torch.Tensor, gamma: float, chunk_size: int = 1024) -> torch.Tensor:
    """Compute sum_{i,j} exp(-gamma * ||x_i - y_j||^2) in chunks to save memory.

    Uses the expansion ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x@y^T instead of
    torch.cdist to avoid materializing the full N×M distance matrix.
    Matches PRX/prx/callbacks/log_generation_metrics.py implementation.
    """
    n, d = x.shape
    y_sq = (y * y).sum(dim=1).unsqueeze(0)  # (1, m)

    total = x.new_zeros(())
    for i in range(0, n, chunk_size):
        x_chunk = x[i : i + chunk_size]  # (c, d)
        x_sq = (x_chunk * x_chunk).sum(dim=1).unsqueeze(1)  # (c, 1)
        dist2 = x_sq + y_sq - 2.0 * (x_chunk @ y.t())
        total = total + torch.exp(-gamma * dist2).sum()
    return total


def _cmmd_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 10.0,
    scale: float = 1000.0,
    chunk_size: int = 1024,
) -> float:
    """CMMD MMD estimator matching sayakpaul/cmmd-pytorch and PRX.

    Uses Gaussian RBF kernel with biased estimator (mean over ALL pairs
    including diagonal). Returns scale * (k_xx + k_yy - 2*k_xy).
    """
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return float("nan")

    gamma = 1.0 / (2.0 * sigma ** 2)

    s_xx = _rbf_kernel_sum_chunked(x, x, gamma=gamma, chunk_size=chunk_size)
    s_yy = _rbf_kernel_sum_chunked(y, y, gamma=gamma, chunk_size=chunk_size)
    s_xy = _rbf_kernel_sum_chunked(x, y, gamma=gamma, chunk_size=chunk_size)

    k_xx = s_xx / float(n * n)
    k_yy = s_yy / float(m * m)
    k_xy = s_xy / float(n * m)

    return (scale * (k_xx + k_yy - 2.0 * k_xy)).item()


def compute_cmmd(
    generated_images: list[Image.Image],
    reference_images: list[Image.Image],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Compute CLIP-based Maximum Mean Discrepancy (CMMD).

    Uses CLIP image_embeds WITHOUT L2 normalization, matching
    sayakpaul/cmmd-pytorch. L2-normalized features collapse the RBF
    kernel to ~1.0 with sigma=10, producing CMMD≈0.
    """
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    )

    def extract_features(images):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                # image_embeds: projected CLIP features, NOT L2-normalized
                feats = clip_model.get_image_features(**inputs)
            all_feats.append(feats.cpu().float())
        return torch.cat(all_feats, dim=0)

    gen_feats = extract_features(generated_images)
    ref_feats = extract_features(reference_images)

    cmmd_value = _cmmd_distance(gen_feats, ref_feats, sigma=10.0, scale=1000.0)

    del clip_model
    torch.cuda.empty_cache()

    return {"cmmd": cmmd_value}


# ---------------------------------------------------------------------------
# KID (Kernel Inception Distance)
# ---------------------------------------------------------------------------

def compute_kid(
    generated_images: list[Image.Image],
    reference_images: list[Image.Image],
    device: str = "cuda",
    batch_size: int = 64,
    num_subsets: int = 100,
    subset_size: int = 1000,
) -> dict:
    """Compute KID using Inception V3 features with polynomial kernel."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    model.fc = torch.nn.Identity()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(images):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            with torch.no_grad():
                feats = model(tensors)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    gen_feats = extract_features(generated_images)
    ref_feats = extract_features(reference_images)

    n = min(len(gen_feats), len(ref_feats), subset_size)
    kids = []
    for _ in range(num_subsets):
        idx_g = torch.randperm(len(gen_feats))[:n]
        idx_r = torch.randperm(len(ref_feats))[:n]
        x, y = gen_feats[idx_g], ref_feats[idx_r]
        d = x.shape[1]
        kxx = ((x @ x.T / d + 1) ** 3).mean()
        kyy = ((y @ y.T / d + 1) ** 3).mean()
        kxy = ((x @ y.T / d + 1) ** 3).mean()
        kids.append((kxx + kyy - 2 * kxy).item())

    del model
    torch.cuda.empty_cache()

    return {"kid": float(np.mean(kids)), "kid_std": float(np.std(kids))}


# ---------------------------------------------------------------------------
# FDD (Fréchet DINO Distance)
# ---------------------------------------------------------------------------

def compute_fdd(
    generated_images: list[Image.Image],
    reference_images: list[Image.Image],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Compute FDD using DINOv3 ViT-L/16 features."""
    from scipy.linalg import sqrtm

    os.environ.setdefault("TORCH_HOME", "<your folder>")
    dino_model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
    dino_model = dino_model.to(device).eval().to(torch.float32)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(images):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            with torch.no_grad():
                out = dino_model.forward_features(tensors)
                feats = out["x_norm_clstoken"]
            all_feats.append(feats.cpu().float())
        return torch.cat(all_feats, dim=0).numpy()

    gen_feats = extract_features(generated_images)
    ref_feats = extract_features(reference_images)

    mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)
    mu_ref, sigma_ref = ref_feats.mean(axis=0), np.cov(ref_feats, rowvar=False)

    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fdd_value = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))

    del dino_model
    torch.cuda.empty_cache()

    return {"fdd": fdd_value, "_fdd_gen_feats": gen_feats, "_fdd_ref_feats": ref_feats}


# ---------------------------------------------------------------------------
# KDD (Kernel DINO Distance)
# ---------------------------------------------------------------------------

def compute_kdd(
    generated_images: list[Image.Image],
    reference_images: list[Image.Image],
    device: str = "cuda",
    batch_size: int = 32,
    num_subsets: int = 100,
    subset_size: int = 1000,
) -> dict:
    """Compute KDD using DINOv3 features with polynomial kernel."""
    from scipy.linalg import sqrtm

    os.environ.setdefault("TORCH_HOME", "<your folder>")
    dino_model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
    dino_model = dino_model.to(device).eval().to(torch.float32)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(images):
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            with torch.no_grad():
                out = dino_model.forward_features(tensors)
                feats = out["x_norm_clstoken"]
            all_feats.append(feats.cpu().float())
        return torch.cat(all_feats, dim=0)

    gen_feats = extract_features(generated_images)
    ref_feats = extract_features(reference_images)

    n = min(len(gen_feats), len(ref_feats), subset_size)
    kids = []
    for _ in range(num_subsets):
        idx_g = torch.randperm(len(gen_feats))[:n]
        idx_r = torch.randperm(len(ref_feats))[:n]
        x, y = gen_feats[idx_g], ref_feats[idx_r]
        d = x.shape[1]
        kxx = ((x @ x.T / d + 1) ** 3).mean()
        kyy = ((y @ y.T / d + 1) ** 3).mean()
        kxy = ((x @ y.T / d + 1) ** 3).mean()
        kids.append((kxx + kyy - 2 * kxy).item())

    del dino_model
    torch.cuda.empty_cache()

    return {"kdd": float(np.mean(kids)), "kdd_std": float(np.std(kids))}


# ---------------------------------------------------------------------------
# ImageReward (requires pip install image-reward)
# ---------------------------------------------------------------------------

def compute_image_reward(
    images: list[Image.Image],
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Compute ImageReward score (learned human preference)."""
    import sys
    sys.path.insert(0, "<your folder>")
    from rewards import _patch_imscore_blip
    _patch_imscore_blip()
    from imscore.imreward.model import ImageReward

    model = ImageReward.from_pretrained("RE-N-Y/ImageReward").to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    scores = []
    for i in tqdm(range(0, len(images), batch_size), desc="ImageReward"):
        batch_imgs = images[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        batch_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            batch_scores = model.score(batch_tensors, batch_prompts)
            if isinstance(batch_scores, torch.Tensor):
                scores.extend(batch_scores.cpu().tolist())
            else:
                scores.append(float(batch_scores))

    del model
    torch.cuda.empty_cache()

    return {
        "image_reward_mean": float(np.mean(scores)),
        "image_reward_std": float(np.std(scores)),
        "image_reward_median": float(np.median(scores)),
        "image_reward_scores": scores,
    }


# ---------------------------------------------------------------------------
# Aesthetic Score
# ---------------------------------------------------------------------------

def compute_aesthetic_score(
    images: list[Image.Image],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Compute aesthetic quality score using the aesthetic predictor."""
    # Use CLIP + aesthetic predictor MLP
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Load aesthetic predictor weights if available
    aesthetic_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints",
        "aesthetic_predictor_v2_5.pth"
    )
    if not os.path.exists(aesthetic_path):
        return {"aesthetic_score": None, "error": f"Missing {aesthetic_path}"}

    # Simple MLP head (matches aesthetic predictor v2.5 architecture)
    import torch.nn as nn
    aesthetic_model = nn.Sequential(
        nn.Linear(1152, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Dropout(0.1),
        nn.Linear(16, 1),
    ).to(device)
    state = torch.load(aesthetic_path, map_location=device)
    # Handle state dicts with 'scoring_head.' prefix
    if any(k.startswith("scoring_head.") for k in state):
        state = {k.replace("scoring_head.", ""): v for k, v in state.items()}
    aesthetic_model.load_state_dict(state)
    aesthetic_model.eval()

    scores = []
    for i in tqdm(range(0, len(images), batch_size), desc="Aesthetic Score"):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            img_embeds = clip_model.get_image_features(**inputs)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            score = aesthetic_model(img_embeds).squeeze(-1)
        scores.extend(score.cpu().tolist())

    del clip_model, aesthetic_model
    torch.cuda.empty_cache()

    return {
        "aesthetic_score_mean": float(np.mean(scores)),
        "aesthetic_score_std": float(np.std(scores)),
        "aesthetic_score_median": float(np.median(scores)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

AVAILABLE_METRICS = {
    "clip_score": compute_clip_score,
    "image_reward": compute_image_reward,
    "aesthetic": compute_aesthetic_score,
    # FID and CMMD need reference images — handled separately
}

REFERENCE_METRICS = {"fid": compute_fid, "cmmd": compute_cmmd, "kid": compute_kid, "fdd": compute_fdd, "kdd": compute_kdd}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation quality metrics")
    parser.add_argument("--model", type=str, help="Model ID to evaluate")
    parser.add_argument("--all-dose", action="store_true", help="Evaluate all dose-response conditions")
    parser.add_argument("--all-sft", action="store_true", help="Evaluate all SFT conditions")
    parser.add_argument("--all-prx", action="store_true", help="Evaluate all existing PRX models")
    parser.add_argument(
        "--metrics", type=str, default="clip_score",
        help="Comma-separated metrics: clip_score,fid,kid,fdd,kdd,cmmd,image_reward,aesthetic",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Reference image directory for FID/CMMD")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    output_dir = args.output_dir or os.path.join(config["base_output_dir"], "quality_metrics")
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "eval_quality.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Determine models to evaluate
    models = []
    if args.all_dose:
        models.extend([f"dose_{c}" for c in DOSE_CONDITIONS])
    if args.all_sft:
        models.extend([f"sft_{c}" for c in DOSE_CONDITIONS])
    if args.all_prx:
        models.extend(PRX_MODELS)
    if args.model:
        models.append(args.model)
    if not models:
        logger.error("Specify --model, --all-dose, --all-sft, or --all-prx")
        return

    requested_metrics = [m.strip() for m in args.metrics.split(",")]
    logger.info(f"Evaluating {len(models)} models with metrics: {requested_metrics}")

    # Load reference images if needed for FID/CMMD
    reference_images = None
    if any(m in REFERENCE_METRICS for m in requested_metrics):
        if args.reference_dir:
            logger.info(f"Loading reference images from {args.reference_dir}")
            reference_images = load_images_from_dir(args.reference_dir, args.max_images)
        else:
            logger.warning("FID/CMMD requested but no --reference-dir provided, skipping")

    all_results = {}

    for model_id in models:
        logger.info(f"\nEvaluating {model_id}...")

        try:
            images, prompts = get_images_and_prompts(model_id, config, args.max_images)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {model_id}: {e}")
            continue

        logger.info(f"Loaded {len(images)} images")

        model_results = {"model": model_id, "num_images": len(images)}

        for metric_name in requested_metrics:
            logger.info(f"  Computing {metric_name}...")
            start = time.time()

            if metric_name in AVAILABLE_METRICS:
                fn = AVAILABLE_METRICS[metric_name]
                if metric_name in ("clip_score", "image_reward"):
                    result = fn(images, prompts, device=args.device)
                else:
                    result = fn(images, device=args.device)
            elif metric_name in REFERENCE_METRICS and reference_images:
                fn = REFERENCE_METRICS[metric_name]
                result = fn(images, reference_images, device=args.device)
            else:
                result = {metric_name: None, "error": "skipped or missing reference"}

            # Remove per-image scores and internal arrays from summary
            summary = {k: v for k, v in result.items()
                       if not k.endswith("_scores") and not k.startswith("_")}
            model_results.update(summary)

            elapsed = time.time() - start
            logger.info(f"  {metric_name}: {summary} ({elapsed:.1f}s)")

        all_results[model_id] = model_results

        # Save incrementally after each model
        results_path = os.path.join(output_dir, "quality_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Also save as CSV for easy viewing
    csv_rows = []
    for model_id, results in all_results.items():
        row = {k: v for k, v in results.items() if not isinstance(v, list)}
        csv_rows.append(row)

    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(output_dir, "quality_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to {results_path} and {csv_path}")
        logger.info(f"\n{df.to_string()}")


if __name__ == "__main__":
    main()
