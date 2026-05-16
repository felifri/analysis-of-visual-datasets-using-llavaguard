import json, os, sys, time, tarfile, logging
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip
_patch_imscore_blip()

from imscore.imreward.model import ImageReward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
output_dir = "<your folder>"
os.makedirs(output_dir, exist_ok=True)

logger.info("Loading ImageReward...")
model = ImageReward.from_pretrained("RE-N-Y/ImageReward").to(device).eval()
logger.info("ImageReward loaded")

prompt_df = pd.read_csv(
    "<your folder>",
    index_col=0,
)
prompts = prompt_df["prompt"].tolist()

def load_images_from_tar(tar_path, max_images=10000):
    images = []
    with tarfile.open(tar_path, "r:") as tf:
        members = sorted([m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")], key=lambda m: m.name)
        for member in members[:max_images]:
            f = tf.extractfile(member)
            if f:
                images.append(Image.open(f).convert("RGB"))
    return images

def get_tar_path(model_id):
    base = "<your folder>"
    if model_id.startswith("dose_"):
        cond = model_id.replace("dose_", "")
        return os.path.join(base, "dose_response", cond, "images.tar")
    else:
        return os.path.join(base, "prx_existing", model_id, "images.tar")

all_models = ["dose_C1", "dose_C2", "dose_C0", "dose_C4", "dose_C6",
              "prx-1024-beta", "prx-512-base", "prx-512-sft",
              "prx-512-sft-distilled", "prx-512-dc-ae", "prx-256-base", "prx-256-sft"]

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

results = {}
for model_id in all_models:
    tar_path = get_tar_path(model_id)
    if not os.path.exists(tar_path):
        logger.warning("Skipping %s", model_id)
        continue

    logger.info("Evaluating %s...", model_id)
    images = load_images_from_tar(tar_path)
    n = min(len(images), len(prompts))
    scores = []
    t0 = time.time()

    for i in range(0, n, 32):
        batch_imgs = images[i:i+32]
        batch_prompts = prompts[i:i+32]
        batch_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            batch_scores = model.score(batch_tensors, batch_prompts)
            if isinstance(batch_scores, torch.Tensor):
                scores.extend(batch_scores.cpu().tolist())
            else:
                scores.append(float(batch_scores))

    elapsed = time.time() - t0
    mean_score = float(np.mean(scores))
    results[model_id] = {
        "model": model_id,
        "image_reward_mean": mean_score,
        "image_reward_std": float(np.std(scores)),
        "image_reward_median": float(np.median(scores)),
        "num_images": len(scores),
    }
    logger.info("  %s: mean=%.4f (%ds)", model_id, mean_score, elapsed)
    del images

with open(os.path.join(output_dir, "image_reward.json"), "w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results.values())
df.to_csv(os.path.join(output_dir, "image_reward.csv"), index=False)
logger.info("\n%s", df.to_string())
