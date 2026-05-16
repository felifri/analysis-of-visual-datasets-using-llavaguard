"""Generate from COCO captions and PartiPrompts, then compute quality metrics.

For each dose-response condition:
1. Generate 5K images from COCO val captions
2. Generate 1.6K images from PartiPrompts
3. Compute FID-5K (COCO generated vs COCO real)
4. Compute CLIP Score on PartiPrompts

Usage:
    python entrypoint_eval_quality_benchmarks.py --condition C1
    python entrypoint_eval_quality_benchmarks.py --all
    python entrypoint_eval_quality_benchmarks.py --all --reference training
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PRX_DIR = os.environ.get("PRX_DIR", "<your folder>")
sys.path.insert(0, PRX_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5",
              "C1_clip", "C0_clip", "C1_safeclip", "C0_safeclip"]

COCO_CAPTIONS_JSONL = "<your folder>"
COCO_REAL_DIR = "<your folder>"
NUM_COCO_CAPTIONS = 30000  # Sample 30K from 118K train captions
NUM_COCO_REF_IMAGES = 30000  # Use 30K real images as reference


def find_parti_prompts():
    cache = "<your folder>"
    if os.path.isdir(cache):
        snap = os.listdir(cache)[0]
        return os.path.join(cache, snap, "PartiPrompts.tsv")
    return None


def load_coco_captions():
    """Load and sample 30K COCO captions."""
    import random
    captions = []
    with open(COCO_CAPTIONS_JSONL) as f:
        for line in f:
            data = json.loads(line)
            caption = data["conversations"][1]["value"]
            captions.append(caption)
    rng = random.Random(42)
    rng.shuffle(captions)
    return captions[:NUM_COCO_CAPTIONS]


def load_training_captions():
    """Load 30K training captions from precomputed JSON."""
    ref_stats_dir = "<your folder>"
    captions_path = os.path.join(ref_stats_dir, "training_30k_captions.json")
    with open(captions_path) as f:
        captions = json.load(f)
    return captions


def load_parti_prompts():
    path = find_parti_prompts()
    if not path:
        return []
    df = pd.read_csv(path, sep="\t")
    return df["Prompt"].tolist()


def load_coco_real_images():
    """Load 30K random COCO real images as reference."""
    import random
    from glob import glob
    paths = sorted(glob(os.path.join(COCO_REAL_DIR, "*.jpg")))
    rng = random.Random(42)
    rng.shuffle(paths)
    paths = paths[:NUM_COCO_REF_IMAGES]
    return [Image.open(p).convert("RGB") for p in tqdm(paths, desc="COCO real", leave=False)]


def generate_from_checkpoint(condition, prompts, output_dir, resolution=512, batch_size=4,
                              guidance_scale=3.5, num_steps=50, seed=42):
    """Generate images using the dose-response checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    # Check existing
    existing = set(f.removesuffix(".jpg") for f in os.listdir(output_dir) if f.endswith(".jpg"))
    todo = [(i, p) for i, p in enumerate(prompts) if str(i).zfill(5) not in existing]

    if not todo:
        logger.info("  All %d images already generated", len(prompts))
        return

    # Find checkpoint
    base = "<your folder>"
    ckpt_dir = None
    config_path = None
    for root in ["checkpoints_full", "checkpoints"]:
        candidate = os.path.join(base, root, condition, "phase1")
        if os.path.isdir(candidate):
            latest = os.path.join(candidate, "latest-rank0.pt")
            if os.path.exists(latest):
                ckpt_dir = latest if os.path.isdir(latest) else candidate
                config_path = os.path.join(candidate, "config.yaml")
                break
            # Check subdirs
            for entry in sorted(os.listdir(candidate)):
                subdir = os.path.join(candidate, entry)
                if os.path.isdir(subdir) and any(f.endswith(".distcp") for f in os.listdir(subdir)):
                    ckpt_dir = subdir
                    config_path = os.path.join(candidate, "config.yaml")
                    break
        if ckpt_dir:
            break

    if not ckpt_dir or not config_path:
        logger.error("No checkpoint for %s", condition)
        return

    # Import PRX pipeline
    from omegaconf import OmegaConf
    from prx.pipeline.models_factory import build_pipeline
    from prx.dataset.constants import BatchKeys
    import torch.distributed.checkpoint as dcp

    config = OmegaConf.load(config_path)
    pipeline = build_pipeline(
        denoiser_config=OmegaConf.to_container(config.diffusion_model, resolve=True),
        text_tower_config=OmegaConf.to_container(config.diffusion_text_tower, resolve=True),
        vae_config=OmegaConf.to_container(config.diffusion_vae, resolve=True),
        scheduler_config=OmegaConf.to_container(config.diffusion_scheduler, resolve=True),
        input_size=resolution,
        p_drop_caption=0.0,
        denoiser_dtype=str(config.get("denoiser_dtype", "torch.float")),
    )

    # Load EMA weights
    metadata_path = os.path.join(ckpt_dir, ".metadata")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    all_keys = list(metadata.state_dict_metadata.keys())
    ema_keys = [k for k in all_keys if "ema_denoiser.model." in k]
    if not ema_keys:
        ema_keys = [k for k in all_keys if k.startswith("state.model.denoiser.")]

    state_dict = {}
    for key in ema_keys:
        tm = metadata.state_dict_metadata[key]
        if hasattr(tm, "size"):
            state_dict[key] = torch.empty(tm.size, dtype=torch.float32)
        elif hasattr(tm, "properties") and hasattr(tm.properties, "size"):
            state_dict[key] = torch.empty(tm.properties.size, dtype=torch.float32)
        else:
            state_dict[key] = torch.tensor(0.0)

    dcp.load(state_dict, storage_reader=dcp.FileSystemReader(ckpt_dir))

    stripped = {}
    for key, value in state_dict.items():
        new_key = key.replace("state.model.", "")
        if new_key.startswith("ema_denoiser.model."):
            new_key = "denoiser." + new_key[len("ema_denoiser.model."):]
        stripped[new_key] = value

    pipeline.load_state_dict(stripped, strict=False)
    pipeline = pipeline.to("cuda").eval()

    # Generate
    num_digits = 5
    generated = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start:batch_start + batch_size]
            batch_indices = [b[0] for b in batch]
            batch_prompts = [b[1] for b in batch]

            try:
                images = pipeline.generate(
                    batch={BatchKeys.PROMPT: batch_prompts},
                    image_size=(resolution, resolution),
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    seed=seed,
                    progress_bar=False,
                    decode_latents=True,
                )
                for idx, img_tensor in zip(batch_indices, images):
                    arr = (img_tensor.clamp(0, 1).cpu().float().numpy().transpose(1, 2, 0) * 255).astype("uint8")
                    Image.fromarray(arr).save(os.path.join(output_dir, f"{str(idx).zfill(num_digits)}.jpg"))
                generated += len(images)

                if generated % 500 < batch_size:
                    elapsed = time.time() - t0
                    logger.info("  Generated %d/%d (%.1f img/s)", generated, len(todo), generated / elapsed)
            except Exception as e:
                logger.error("Batch error at %d: %s", batch_indices[0], e)

    del pipeline
    torch.cuda.empty_cache()
    logger.info("  Generation complete: %d images in %.1fm", generated, (time.time() - t0) / 60)


def compute_fid_from_cached_stats(gen_dir, ref_stats_path, device="cuda", batch_size=64):
    """Compute FID using precomputed reference statistics (mu, sigma)."""
    from scipy.linalg import sqrtm
    from torchvision.models import inception_v3, Inception_V3_Weights

    ref = np.load(ref_stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from glob import glob
    gen_paths = sorted(glob(os.path.join(gen_dir, "*.jpg")))
    all_feats = []
    for i in range(0, len(gen_paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in gen_paths[i:i+batch_size]]
        batch = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0).numpy()
    mu_gen = all_feats.mean(axis=0)
    sigma_gen = np.cov(all_feats, rowvar=False)

    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))

    del model
    torch.cuda.empty_cache()
    return fid, all_feats


def compute_fdd_from_cached_stats(gen_dir, ref_stats_path, device="cuda", batch_size=32):
    """Compute FDD (DINOv3-based FID) using precomputed reference statistics."""
    from scipy.linalg import sqrtm

    ref = np.load(ref_stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]

    os.environ["TORCH_HOME"] = "<your folder>"
    dino_model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
    dino_model = dino_model.to(device).eval().to(torch.float32)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from glob import glob
    gen_paths = sorted(glob(os.path.join(gen_dir, "*.jpg")))
    all_feats = []
    for i in range(0, len(gen_paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in gen_paths[i:i+batch_size]]
        batch = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            out = dino_model.forward_features(batch)
            feats = out["x_norm_clstoken"]
        all_feats.append(feats.cpu().float())

    all_feats = torch.cat(all_feats, dim=0).numpy()
    mu_gen = all_feats.mean(axis=0)
    sigma_gen = np.cov(all_feats, rowvar=False)

    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fdd = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))

    del dino_model
    torch.cuda.empty_cache()
    return fdd, all_feats


def compute_kid_from_feats(gen_feats, ref_feats_path, num_subsets=100, subset_size=1000):
    """Compute KID using precomputed reference features."""
    ref_feats = np.load(ref_feats_path)
    gen = torch.from_numpy(gen_feats)
    ref = torch.from_numpy(ref_feats)
    n = min(len(gen), len(ref), subset_size)
    if n < 2:
        return float("nan")
    kids = []
    for _ in range(num_subsets):
        idx_g = torch.randperm(len(gen))[:n]
        idx_r = torch.randperm(len(ref))[:n]
        x, y = gen[idx_g], ref[idx_r]
        d = x.shape[1]
        kxx = ((x @ x.T / d + 1) ** 3).mean()
        kyy = ((y @ y.T / d + 1) ** 3).mean()
        kxy = ((x @ y.T / d + 1) ** 3).mean()
        kids.append((kxx + kyy - 2 * kxy).item())
    return float(np.mean(kids))


def compute_clip_score(gen_dir, prompts, device="cuda", batch_size=32):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    from glob import glob
    gen_paths = sorted(glob(os.path.join(gen_dir, "*.jpg")))
    n = min(len(gen_paths), len(prompts))

    scores = []
    for i in range(0, n, batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in gen_paths[i:i+batch_size]]
        batch_prompts = prompts[i:i+batch_size]
        inputs = processor(text=batch_prompts, images=batch_imgs, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores.extend((img_e * txt_e).sum(dim=-1).cpu().tolist())

    del model
    torch.cuda.empty_cache()
    return float(np.mean(scores)), float(np.std(scores))


def run_coco_benchmark(conditions, output_base):
    """Run COCO-30K benchmark (original behavior)."""
    coco_captions = load_coco_captions()
    parti_prompts = load_parti_prompts()
    logger.info("Loaded %d COCO captions, %d PartiPrompts", len(coco_captions), len(parti_prompts))

    ref_stats_dir = "<your folder>"
    inception_stats = os.path.join(ref_stats_dir, "coco_train_inception_stats.npz")
    inception_feats = os.path.join(ref_stats_dir, "coco_train_inception_feats.npy")
    dinov3_stats = os.path.join(ref_stats_dir, "coco_train_dinov3_stats.npz")
    dinov3_feats = os.path.join(ref_stats_dir, "coco_train_dinov3_feats.npy")

    if not os.path.exists(inception_stats):
        logger.error("Reference stats not found at %s. Run precompute_ref_stats.py first.", ref_stats_dir)
        return

    all_results = {}
    results_json = os.path.join(output_base, "benchmark_results.json")
    if os.path.exists(results_json):
        with open(results_json) as f:
            all_results = json.load(f)

    for condition in conditions:
        logger.info("\n" + "=" * 60)
        logger.info("Condition: %s", condition)

        result = {"condition": condition}

        # Generate from COCO captions
        coco_gen_dir = os.path.join(output_base, condition, "coco_generated")
        logger.info("Generating from COCO captions (30K)...")
        generate_from_checkpoint(condition, coco_captions, coco_gen_dir)

        # Compute FID-30K (Inception)
        logger.info("Computing FID-30K (Inception)...")
        t0 = time.time()
        fid_val, gen_inception_feats = compute_fid_from_cached_stats(coco_gen_dir, inception_stats)
        result["fid_coco_30k"] = fid_val
        logger.info("  FID-30K: %.2f (%.0fs)", fid_val, time.time() - t0)

        # Compute KID-30K (Inception)
        logger.info("Computing KID-30K (Inception)...")
        kid_val = compute_kid_from_feats(gen_inception_feats, inception_feats)
        result["kid_coco_30k"] = kid_val
        logger.info("  KID-30K: %.6f", kid_val)
        del gen_inception_feats

        # Compute FDD-30K (DINOv3)
        logger.info("Computing FDD-30K (DINOv3)...")
        t0 = time.time()
        fdd_val, gen_dino_feats = compute_fdd_from_cached_stats(coco_gen_dir, dinov3_stats)
        result["fdd_coco_30k"] = fdd_val
        logger.info("  FDD-30K: %.2f (%.0fs)", fdd_val, time.time() - t0)

        # Compute KDD-30K (DINOv3)
        logger.info("Computing KDD-30K (DINOv3)...")
        kdd_val = compute_kid_from_feats(gen_dino_feats, dinov3_feats)
        result["kdd_coco_30k"] = kdd_val
        logger.info("  KDD-30K: %.6f", kdd_val)
        del gen_dino_feats

        # Compute CLIP Score on COCO
        logger.info("Computing CLIP Score (COCO)...")
        clip_mean, clip_std = compute_clip_score(coco_gen_dir, coco_captions)
        result["clip_score_coco_mean"] = clip_mean
        result["clip_score_coco_std"] = clip_std
        logger.info("  CLIP (COCO): %.4f", clip_mean)

        # Generate from PartiPrompts
        parti_gen_dir = os.path.join(output_base, condition, "parti_generated")
        logger.info("Generating from PartiPrompts (1.6K)...")
        generate_from_checkpoint(condition, parti_prompts, parti_gen_dir)

        # Compute CLIP Score on PartiPrompts
        logger.info("Computing CLIP Score (PartiPrompts)...")
        clip_mean, clip_std = compute_clip_score(parti_gen_dir, parti_prompts)
        result["clip_score_parti_mean"] = clip_mean
        result["clip_score_parti_std"] = clip_std
        logger.info("  CLIP (PartiPrompts): %.4f", clip_mean)

        all_results[condition] = result

        # Save incrementally
        with open(os.path.join(output_base, "benchmark_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results.values())
    df.to_csv(os.path.join(output_base, "benchmark_results.csv"), index=False)
    logger.info("\nResults:\n%s", df.to_string())


def run_training_benchmark(conditions, output_base):
    """Run training-30K benchmark: FID/KID against training data distribution."""
    training_captions = load_training_captions()
    logger.info("Loaded %d training captions", len(training_captions))

    ref_stats_dir = "<your folder>"
    inception_stats = os.path.join(ref_stats_dir, "training_30k_inception_stats.npz")
    inception_feats = os.path.join(ref_stats_dir, "training_30k_inception_feats.npy")

    if not os.path.exists(inception_stats):
        logger.error("Training reference stats not found at %s. Run precompute_ref_stats_training.py first.",
                      ref_stats_dir)
        return

    all_results = {}
    results_json = os.path.join(output_base, "benchmark_results_training.json")
    if os.path.exists(results_json):
        with open(results_json) as f:
            all_results = json.load(f)

    for condition in conditions:
        logger.info("\n" + "=" * 60)
        logger.info("Condition (training ref): %s", condition)

        result = {"condition": condition}

        # Generate from training captions
        training_gen_dir = os.path.join(output_base, condition, "training_generated")
        logger.info("Generating from training captions (30K)...")
        generate_from_checkpoint(condition, training_captions, training_gen_dir)

        # Compute FID-30K (Inception, training ref)
        logger.info("Computing FID-30K against training (Inception)...")
        t0 = time.time()
        fid_val, gen_inception_feats = compute_fid_from_cached_stats(training_gen_dir, inception_stats)
        result["fid_training_30k"] = fid_val
        logger.info("  FID-training-30K: %.2f (%.0fs)", fid_val, time.time() - t0)

        # Compute KID-30K (Inception, training ref)
        logger.info("Computing KID-30K against training (Inception)...")
        kid_val = compute_kid_from_feats(gen_inception_feats, inception_feats)
        result["kid_training_30k"] = kid_val
        logger.info("  KID-training-30K: %.6f", kid_val)
        del gen_inception_feats

        # Compute CLIP Score on training captions
        logger.info("Computing CLIP Score (training captions)...")
        clip_mean, clip_std = compute_clip_score(training_gen_dir, training_captions)
        result["clip_score_training_mean"] = clip_mean
        result["clip_score_training_std"] = clip_std
        logger.info("  CLIP (training): %.4f", clip_mean)

        all_results[condition] = result

        # Save per-condition result (safe for parallel jobs)
        cond_result_path = os.path.join(output_base, f"benchmark_results_training_{condition}.json")
        with open(cond_result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved %s", cond_result_path)

    # Merge all per-condition results into combined files
    merged = {}
    for cond_file in sorted(os.listdir(output_base)):
        if cond_file.startswith("benchmark_results_training_") and cond_file.endswith(".json"):
            with open(os.path.join(output_base, cond_file)) as f:
                r = json.load(f)
                merged[r["condition"]] = r
    if merged:
        with open(os.path.join(output_base, "benchmark_results_training.json"), "w") as f:
            json.dump(merged, f, indent=2)
        df = pd.DataFrame(merged.values())
        df.to_csv(os.path.join(output_base, "benchmark_results_training.csv"), index=False)
        logger.info("\nTraining-ref results (%d conditions):\n%s", len(merged), df.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, choices=CONDITIONS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reference", type=str, choices=["coco", "training"], default="coco",
                        help="Reference distribution: 'coco' (default) or 'training'")
    args = parser.parse_args()

    conditions = CONDITIONS if args.all else [args.condition]
    output_base = "<your folder>"
    os.makedirs(output_base, exist_ok=True)

    if args.reference == "training":
        run_training_benchmark(conditions, output_base)
    else:
        run_coco_benchmark(conditions, output_base)


if __name__ == "__main__":
    main()
