"""Precompute Inception feature statistics (mu, sigma) for 30K training images sampled from the MDS safe_full pool."""
import json
import logging
import os
import tempfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

os.environ["TORCH_HOME"] = "<your folder>"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
output_dir = "<your folder>"
os.makedirs(output_dir, exist_ok=True)

MDS_PATH = "<your folder>"
NUM_SAMPLES = 30000
SEED = 42
batch_size = 64

# ============================================================
# Phase 1: Sample 30K images and captions from MDS
# ============================================================
logger.info("=== Phase 1: Sampling %d images from MDS pool ===", NUM_SAMPLES)

from streaming import StreamingDataset

ds = StreamingDataset(local=MDS_PATH, shuffle=False)
total = len(ds)
logger.info("MDS dataset has %d samples", total)

rng = np.random.RandomState(SEED)
indices = rng.choice(total, size=NUM_SAMPLES, replace=False)
indices.sort()
logger.info("Selected %d indices (min=%d, max=%d)", len(indices), indices[0], indices[-1])

# Save captions and images to temp dir
captions = []
tmp_dir = tempfile.mkdtemp(prefix="training_30k_")
logger.info("Saving sampled images to %s", tmp_dir)

for count, idx in enumerate(tqdm(indices, desc="Sampling")):
    sample = ds[int(idx)]
    captions.append(sample["prompt"])
    img = sample["image"]
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")
    img.save(os.path.join(tmp_dir, f"{count:05d}.jpg"))

captions_path = os.path.join(output_dir, "training_30k_captions.json")
with open(captions_path, "w") as f:
    json.dump(captions, f)
logger.info("Saved %d captions to %s", len(captions), captions_path)

# ============================================================
# Phase 2: Inception features
# ============================================================
logger.info("=== Phase 2: Inception v3 features ===")
model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
model.fc = torch.nn.Identity()

transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_paths = sorted(
    os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".jpg")
)
logger.info("Computing features for %d images", len(img_paths))

all_feats = []
for i in tqdm(range(0, len(img_paths), batch_size), desc="Inception"):
    batch_paths = img_paths[i : i + batch_size]
    batch = torch.stack(
        [transform_inception(Image.open(p).convert("RGB")) for p in batch_paths]
    ).to(device)
    with torch.no_grad():
        feats = model(batch)
    all_feats.append(feats.cpu())

all_feats = torch.cat(all_feats, dim=0).numpy()
mu = all_feats.mean(axis=0)
sigma = np.cov(all_feats, rowvar=False)

np.savez(
    os.path.join(output_dir, "training_30k_inception_stats.npz"),
    mu=mu,
    sigma=sigma,
    n=NUM_SAMPLES,
)
np.save(os.path.join(output_dir, "training_30k_inception_feats.npy"), all_feats)
logger.info("Inception: saved mu %s, sigma %s, feats %s", mu.shape, sigma.shape, all_feats.shape)

del model, all_feats
torch.cuda.empty_cache()

# ============================================================
# Cleanup
# ============================================================
import shutil

logger.info("Cleaning up temp dir %s", tmp_dir)
shutil.rmtree(tmp_dir)

logger.info("Done. Training reference stats saved to %s", output_dir)
