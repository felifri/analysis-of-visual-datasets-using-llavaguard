"""Precompute Inception and DINOv3 feature statistics (mu, sigma) for COCO train reference images."""
import os
import gc
import logging
import numpy as np
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

os.environ["TORCH_HOME"] = "<your folder>"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
output_dir = "<your folder>"
os.makedirs(output_dir, exist_ok=True)

COCO_TRAIN_DIR = "<your folder>"
paths = sorted(glob(os.path.join(COCO_TRAIN_DIR, "*.jpg")))
logger.info("Found %d COCO train images", len(paths))

batch_size = 64

# ============================================================
# Phase 1: Inception features
# ============================================================
logger.info("=== Phase 1: Inception v3 features ===")
model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
model.fc = torch.nn.Identity()

transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_feats = []
for i in tqdm(range(0, len(paths), batch_size), desc="Inception"):
    batch_paths = paths[i:i+batch_size]
    batch = torch.stack([transform_inception(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
    with torch.no_grad():
        feats = model(batch)
    all_feats.append(feats.cpu())

all_feats = torch.cat(all_feats, dim=0).numpy()
mu = all_feats.mean(axis=0)
sigma = np.cov(all_feats, rowvar=False)

np.savez(os.path.join(output_dir, "coco_train_inception_stats.npz"), mu=mu, sigma=sigma, n=len(paths))
np.save(os.path.join(output_dir, "coco_train_inception_feats.npy"), all_feats)
logger.info("Inception: saved mu %s, sigma %s, feats %s", mu.shape, sigma.shape, all_feats.shape)

del model, all_feats
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# Phase 2: DINOv3 features
# ============================================================
logger.info("=== Phase 2: DINOv3 features ===")
dino_model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
dino_model = dino_model.to(device).eval().to(torch.float32)

transform_dino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_feats = []
batch_size_dino = 32  # DINOv3 is larger
for i in tqdm(range(0, len(paths), batch_size_dino), desc="DINOv3"):
    batch_paths = paths[i:i+batch_size_dino]
    batch = torch.stack([transform_dino(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
    with torch.no_grad():
        out = dino_model.forward_features(batch)
        feats = out["x_norm_clstoken"]
    all_feats.append(feats.cpu().float())

all_feats = torch.cat(all_feats, dim=0).numpy()
mu = all_feats.mean(axis=0)
sigma = np.cov(all_feats, rowvar=False)

np.savez(os.path.join(output_dir, "coco_train_dinov3_stats.npz"), mu=mu, sigma=sigma, n=len(paths))
np.save(os.path.join(output_dir, "coco_train_dinov3_feats.npy"), all_feats)
logger.info("DINOv3: saved mu %s, sigma %s, feats %s", mu.shape, sigma.shape, all_feats.shape)

del dino_model, all_feats
torch.cuda.empty_cache()

logger.info("Done. Reference stats saved to %s", output_dir)
