"""Generate from COCO captions and PartiPrompts for existing PRX models, compute FID/FDD/CLIP.

Usage:
    python eval_quality_benchmarks_prx.py --model-index 0
    python eval_quality_benchmarks_prx.py --model-index 0 --reference training
"""
import json, os, sys, time, logging
import numpy as np
import pandas as pd
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
from tqdm import tqdm
from diffusers import DiffusionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRX_MODELS = [
    {"id": "Photoroom/prx-1024-t2i-beta", "resolution": 1024, "short": "prx-1024-beta"},
    {"id": "Photoroom/prx-512-t2i", "resolution": 512, "short": "prx-512-base"},
    {"id": "Photoroom/prx-512-t2i-sft", "resolution": 512, "short": "prx-512-sft"},
    {"id": "Photoroom/prx-512-t2i-sft-distilled", "resolution": 512, "short": "prx-512-sft-distilled"},
    {"id": "Photoroom/prx-512-t2i-dc-ae", "resolution": 512, "short": "prx-512-dc-ae"},
    {"id": "Photoroom/prx-256-t2i", "resolution": 256, "short": "prx-256-base"},
    {"id": "Photoroom/prx-256-t2i-sft", "resolution": 256, "short": "prx-256-sft"},
]

def load_coco_captions(n=30000):
    import random
    captions = []
    with open("<your folder>") as f:
        for line in f:
            data = json.loads(line)
            captions.append(data["conversations"][1]["value"])
    rng = random.Random(42)
    rng.shuffle(captions)
    return captions[:n]

def load_parti_prompts():
    cache = "<your folder>"
    snap = os.listdir(cache)[0]
    df = pd.read_csv(os.path.join(cache, snap, "PartiPrompts.tsv"), sep="\t")
    return df["Prompt"].tolist()

def generate_images(model_info, prompts, output_dir, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    existing = set(f.removesuffix(".jpg") for f in os.listdir(output_dir) if f.endswith(".jpg"))
    todo = [(i, p) for i, p in enumerate(prompts) if str(i).zfill(5) not in existing]
    if not todo:
        logger.info("  All %d images exist", len(prompts))
        return

    num_gpus = torch.cuda.device_count()
    res = model_info["resolution"]

    if num_gpus > 1:
        _generate_multi_gpu(model_info, todo, output_dir, batch_size, num_gpus)
    else:
        _generate_single_gpu(model_info, todo, output_dir, batch_size, 0)


def _generate_single_gpu(model_info, todo, output_dir, batch_size, gpu_id):
    """Generate images on a single GPU."""
    import copy
    device = f"cuda:{gpu_id}"
    res = model_info["resolution"]

    pipe = DiffusionPipeline.from_pretrained(model_info["id"], torch_dtype=torch.bfloat16).to(device)
    gen = torch.Generator(device=device).manual_seed(42)
    generated = 0
    t0 = time.time()

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start+batch_size]
        try:
            images = pipe(
                prompt=[b[1] for b in batch], width=res, height=res,
                guidance_scale=3.5, num_inference_steps=50, generator=gen,
            ).images
            for (idx, _), img in zip(batch, images):
                img.save(os.path.join(output_dir, f"{str(idx).zfill(5)}.jpg"))
            generated += len(images)
        except Exception as e:
            logger.warning("[GPU %d] Batch error (falling back to single): %s", gpu_id, e)
            for idx, prompt in batch:
                try:
                    img = pipe(
                        prompt=prompt, width=res, height=res,
                        guidance_scale=3.5, num_inference_steps=50, generator=gen,
                    ).images[0]
                    img.save(os.path.join(output_dir, f"{str(idx).zfill(5)}.jpg"))
                    generated += 1
                except Exception:
                    pass
        if generated % 500 < batch_size:
            logger.info("  [GPU %d] %d/%d (%.1f img/s)", gpu_id, generated, len(todo), generated/(time.time()-t0))

    del pipe
    torch.cuda.empty_cache()


def _generate_multi_gpu(model_info, todo, output_dir, batch_size, num_gpus):
    """Generate images across multiple GPUs in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    import copy

    res = model_info["resolution"]
    logger.info("  Multi-GPU generation: %d GPUs, %d images", num_gpus, len(todo))

    # Load model once, then deepcopy to each GPU
    logger.info("  Loading model once, then replicating to %d GPUs...", num_gpus)
    base_pipe = DiffusionPipeline.from_pretrained(model_info["id"], torch_dtype=torch.bfloat16)
    pipes = {}
    for gpu_id in range(num_gpus):
        logger.info("  Copying to GPU %d...", gpu_id)
        pipes[gpu_id] = copy.deepcopy(base_pipe).to(f"cuda:{gpu_id}")
    del base_pipe

    # Split work across GPUs
    gpu_todos = [[] for _ in range(num_gpus)]
    for i, item in enumerate(todo):
        gpu_todos[i % num_gpus].append(item)

    def gpu_worker(gpu_id):
        pipe = pipes[gpu_id]
        device = f"cuda:{gpu_id}"
        gen = torch.Generator(device=device).manual_seed(42)
        generated = 0
        t0 = time.time()

        for batch_start in range(0, len(gpu_todos[gpu_id]), batch_size):
            batch = gpu_todos[gpu_id][batch_start:batch_start+batch_size]
            try:
                images = pipe(
                    prompt=[b[1] for b in batch], width=res, height=res,
                    guidance_scale=3.5, num_inference_steps=50, generator=gen,
                ).images
                for (idx, _), img in zip(batch, images):
                    img.save(os.path.join(output_dir, f"{str(idx).zfill(5)}.jpg"))
                generated += len(images)
            except Exception as e:
                logger.warning("[GPU %d] Batch error (falling back to single): %s", gpu_id, e)
                for idx, prompt in batch:
                    try:
                        img = pipe(
                            prompt=prompt, width=res, height=res,
                            guidance_scale=3.5, num_inference_steps=50, generator=gen,
                        ).images[0]
                        img.save(os.path.join(output_dir, f"{str(idx).zfill(5)}.jpg"))
                        generated += 1
                    except Exception:
                        pass
            if generated % 500 < batch_size:
                logger.info("  [GPU %d] %d/%d (%.1f img/s)", gpu_id, generated, len(gpu_todos[gpu_id]), generated/(time.time()-t0))

        elapsed = time.time() - t0
        rate = generated / elapsed if elapsed > 0 else 0
        logger.info("  [GPU %d] Done: %d images in %.1fm (%.1f img/s)", gpu_id, generated, elapsed/60, rate)

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(gpu_worker, gid) for gid in range(num_gpus)]
        for f in futures:
            f.result()

    for pipe in pipes.values():
        del pipe
    torch.cuda.empty_cache()

def compute_fid_from_cached(gen_dir, stats_path):
    from scipy.linalg import sqrtm
    from torchvision.models import inception_v3, Inception_V3_Weights
    ref = np.load(stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to("cuda").eval()
    model.fc = torch.nn.Identity()
    transform = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    paths = sorted(glob(os.path.join(gen_dir, "*.jpg")))
    feats = []
    for i in range(0, len(paths), 64):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+64]]).to("cuda")
        with torch.no_grad(): feats.append(model(batch).cpu())
    feats = torch.cat(feats).numpy()
    mu_gen, sigma_gen = feats.mean(0), np.cov(feats, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean): covmean = covmean.real
    del model; torch.cuda.empty_cache()
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2*covmean))

def compute_fdd_from_cached(gen_dir, stats_path):
    from scipy.linalg import sqrtm
    os.environ["TORCH_HOME"] = "<your folder>"
    ref = np.load(stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]
    dino = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True).to("cuda").eval().float()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    paths = sorted(glob(os.path.join(gen_dir, "*.jpg")))
    feats = []
    for i in range(0, len(paths), 32):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+32]]).to("cuda")
        with torch.no_grad(): feats.append(dino.forward_features(batch)["x_norm_clstoken"].cpu().float())
    feats = torch.cat(feats).numpy()
    mu_gen, sigma_gen = feats.mean(0), np.cov(feats, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean): covmean = covmean.real
    del dino; torch.cuda.empty_cache()
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2*covmean))

def compute_clip(gen_dir, prompts):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    paths = sorted(glob(os.path.join(gen_dir, "*.jpg")))
    n = min(len(paths), len(prompts))
    scores = []
    for i in range(0, n, 32):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+32]]
        inputs = proc(text=prompts[i:i+32], images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            out = model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.extend((ie*te).sum(-1).cpu().tolist())
    del model; torch.cuda.empty_cache()
    return float(np.mean(scores))

def load_training_captions():
    """Load 30K training captions from precomputed JSON."""
    ref_stats_dir = "<your folder>"
    with open(os.path.join(ref_stats_dir, "training_30k_captions.json")) as f:
        return json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--reference", type=str, choices=["coco", "training"], default="coco",
                        help="Reference distribution: 'coco' (default) or 'training'")
    args = parser.parse_args()

    model_info = PRX_MODELS[args.model_index]
    short = model_info["short"]
    logger.info("Evaluating %s (reference=%s)", short, args.reference)

    output_base = "<your folder>"
    ref_dir = "<your folder>"
    os.makedirs(output_base, exist_ok=True)

    if args.reference == "training":
        training_captions = load_training_captions()
        result = {"model": short}

        # Generate from training captions
        train_dir = os.path.join(output_base, short, "training_generated")
        logger.info("Generating 30K from training captions...")
        generate_images(model_info, training_captions, train_dir)

        # FID-30K (training ref)
        logger.info("Computing FID-30K (training ref)...")
        result["fid_training_30k"] = compute_fid_from_cached(train_dir, os.path.join(ref_dir, "training_30k_inception_stats.npz"))
        logger.info("  FID: %.2f", result["fid_training_30k"])

        # CLIP on training captions
        result["clip_score_training_mean"] = compute_clip(train_dir, training_captions)
        logger.info("  CLIP training: %.4f", result["clip_score_training_mean"])

        # Save per-model result
        out_path = os.path.join(output_base, f"benchmark_training_{short}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved to %s", out_path)
    else:
        coco_captions = load_coco_captions()
        parti_prompts = load_parti_prompts()
        result = {"model": short}

        # COCO generation
        coco_dir = os.path.join(output_base, short, "coco_generated")
        logger.info("Generating 30K from COCO captions...")
        generate_images(model_info, coco_captions, coco_dir)

        # FID-30K
        logger.info("Computing FID-30K...")
        result["fid_coco_30k"] = compute_fid_from_cached(coco_dir, os.path.join(ref_dir, "coco_train_inception_stats.npz"))
        logger.info("  FID: %.2f", result["fid_coco_30k"])

        # FDD-30K
        logger.info("Computing FDD-30K...")
        result["fdd_coco_30k"] = compute_fdd_from_cached(coco_dir, os.path.join(ref_dir, "coco_train_dinov3_stats.npz"))
        logger.info("  FDD: %.2f", result["fdd_coco_30k"])

        # CLIP on COCO
        result["clip_coco"] = compute_clip(coco_dir, coco_captions)
        logger.info("  CLIP COCO: %.4f", result["clip_coco"])

        # PartiPrompts
        parti_dir = os.path.join(output_base, short, "parti_generated")
        logger.info("Generating 1.6K from PartiPrompts...")
        generate_images(model_info, parti_prompts, parti_dir)
        result["clip_parti"] = compute_clip(parti_dir, parti_prompts)
        logger.info("  CLIP Parti: %.4f", result["clip_parti"])

        # Save
        out_path = os.path.join(output_base, f"benchmark_{short}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved to %s", out_path)

if __name__ == "__main__":
    main()
