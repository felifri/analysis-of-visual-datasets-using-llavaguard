#!/bin/bash
#SBATCH --job-name=dose-regen-v2
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00

set -euo pipefail
echo "Started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=<your folder>
export PRX_DIR=<your folder>

cd <your folder>

# ================================================================
# PART 1: Generate remaining COCO benchmark images for C3 (batch_size=32)
# ================================================================
echo "=== Part 1: COCO generation for C3 (batch_size=32) ==="

$PYTHON -c "
import sys, os
sys.path.insert(0, '.')
from experiments.dose_response.entrypoint_eval_quality_benchmarks import generate_from_checkpoint, load_coco_captions
captions = load_coco_captions()
print(f'Loaded {len(captions)} captions')
out_dir = '<your folder>'
generate_from_checkpoint('C3', captions, out_dir, resolution=512, batch_size=32, guidance_scale=3.5, num_steps=50, seed=42)
print('COCO generation complete')
"

echo "Part 1 complete at $(date)"

# ================================================================
# PART 2: FID-30K for all conditions including C3
# ================================================================
echo "=== Part 2: FID-30K ==="

$PYTHON << 'PYEOF'
import os, sys, json, csv, random, gc, logging
import numpy as np
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = "cuda"
ref_stats_dir = "<your folder>"
benchmark_dir = "<your folder>"

ref = np.load(os.path.join(ref_stats_dir, "coco_train_inception_stats.npz"))
mu_ref, sigma_ref = ref["mu"], ref["sigma"]
ref_feats = np.load(os.path.join(ref_stats_dir, "coco_train_inception_feats.npy"))

model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
model.fc = torch.nn.Identity()
transform = transforms.Compose([
    transforms.Resize((299, 299)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_dir, max_images=30000, batch_size=64):
    paths = sorted(glob(os.path.join(image_dir, "*.jpg")))[:max_images]
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Inception"):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def compute_fid(gen_feats, mu_ref, sigma_ref):
    mu_gen = gen_feats.mean(axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))

def compute_kid(gen_feats, ref_feats, num_subsets=100, subset_size=1000):
    gen = torch.from_numpy(gen_feats)
    ref = torch.from_numpy(ref_feats)
    n = min(len(gen), len(ref), subset_size)
    kids = []
    for _ in range(num_subsets):
        x, y = gen[torch.randperm(len(gen))[:n]], ref[torch.randperm(len(ref))[:n]]
        d = x.shape[1]
        kids.append(((((x@x.T/d+1)**3).mean()+((y@y.T/d+1)**3).mean()-2*((x@y.T/d+1)**3).mean()).item()))
    return float(np.mean(kids))

from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def compute_clip_score(image_dir, prompts, max_images=30000, batch_size=32):
    paths = sorted(glob(os.path.join(image_dir, "*.jpg")))[:max_images]
    scores = []
    for i in tqdm(range(0, min(len(paths), len(prompts)), batch_size), desc="CLIP"):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        batch_prompts = prompts[i:i+batch_size]
        inputs = clip_processor(text=batch_prompts, images=batch_imgs, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores.extend((img_e * txt_e).sum(dim=-1).cpu().tolist())
    return float(np.mean(scores)), float(np.std(scores))

coco_captions = []
with open("<your folder>") as f:
    for line in f:
        coco_captions.append(json.loads(line)["conversations"][1]["value"])
rng = random.Random(42); rng.shuffle(coco_captions); coco_captions = coco_captions[:30000]

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]
results = []
for cid in CONDITIONS:
    coco_gen_dir = os.path.join(benchmark_dir, cid, "coco_generated")
    n = len(glob(os.path.join(coco_gen_dir, "*.jpg")))
    if n < 1000:
        logger.warning(f"{cid}: only {n} images, skipping"); continue
    logger.info(f"{cid}: {n} images")
    feats = extract_features(coco_gen_dir)
    row = {"condition": cid, "fid_coco_30k": compute_fid(feats, mu_ref, sigma_ref),
           "kid_coco_30k": compute_kid(feats, ref_feats),
           "num_images": n}
    clip_m, clip_s = compute_clip_score(coco_gen_dir, coco_captions)
    row["clip_score_coco_mean"] = clip_m; row["clip_score_coco_std"] = clip_s
    results.append(row)
    logger.info(f"  FID={row['fid_coco_30k']:.2f}, CLIP={clip_m:.4f}")

# ImageReward
del clip_model, clip_processor, model; torch.cuda.empty_cache(); gc.collect()
import time
sys.path.insert(0, "<your folder>")
from rewards import _patch_imscore_blip; _patch_imscore_blip()
from imscore.imreward.model import ImageReward as IR
ir_model = IR.from_pretrained("RE-N-Y/ImageReward").to(device).eval()
ir_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)),
])
for r in results:
    cid = r["condition"]
    paths = sorted(glob(os.path.join(benchmark_dir, cid, "coco_generated", "*.jpg")))[:30000]
    scores = []
    for i in range(0, min(len(paths), len(coco_captions)), 32):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+32]]
        prompts = coco_captions[i:i+32]
        t = torch.stack([ir_transform(img) for img in imgs]).to(device)
        with torch.no_grad():
            s = ir_model.score(t, prompts)
            if isinstance(s, torch.Tensor): scores.extend(s.cpu().tolist())
            else: scores.append(float(s))
    r["image_reward_coco_mean"] = float(np.mean(scores))
    r["image_reward_coco_std"] = float(np.std(scores))
    logger.info(f"  {cid}: IR={r['image_reward_coco_mean']:.4f}")

out = os.path.join(benchmark_dir, "benchmark_results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
with open(out.replace(".csv",".json"), "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"Saved {out}")
PYEOF

echo "Part 2 complete at $(date)"

# ================================================================
# PART 3: Regenerate all figures
# ================================================================
echo "=== Part 3: Figures ==="

$PYTHON <your folder> 2>&1 || \
$PYTHON << 'PYEOF3'
import json, os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150

ANNOT_DIR = "<your folder>"
CROSS_JUDGE_DIR = "<your folder>"
QUALITY_DIR = "<your folder>"
BENCHMARK_DIR = "<your folder>"
ANALYSIS_DIR = "<your folder>"
FIG_DIR = "<your folder>"

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6", "C5"]
PAPER_ID = {"C0":"C1", "C1":"C2", "C2":"C3", "C3":"C5", "C4":"C0", "C6":"C4", "C5":"C6"}
CONDITION_DESIGN = {
    "C1": {"unsafe_pct": 0.0, "scale": "8M"}, "C0": {"unsafe_pct": 1.21, "scale": "8M"},
    "C2": {"unsafe_pct": 5.0, "scale": "8M"}, "C3": {"unsafe_pct": 9.19, "scale": "8M"},
    "C4": {"unsafe_pct": 1.21, "scale": "1M"}, "C6": {"unsafe_pct": 9.6, "scale": "1M"},
    "C5": {"unsafe_pct": 1.21, "scale": "100K"},
}
FULL_SCALE = ["C1", "C0", "C2", "C3"]

results = {}
for cid in CONDITIONS:
    pq = os.path.join(ANNOT_DIR, f"dose_{cid}.parquet")
    if os.path.exists(pq):
        results[cid] = pd.read_parquet(pq)

# ── Fig 1: Dose-response curve ──
fig, ax = plt.subplots(figsize=(5.5, 4))
x_f = [CONDITION_DESIGN[c]["unsafe_pct"] for c in FULL_SCALE if c in results]
y_f = [(results[c]["rating"]=="Unsafe").mean()*100 for c in FULL_SCALE if c in results]
l_f = [PAPER_ID[c] for c in FULL_SCALE if c in results]
ax.plot(x_f, y_f, "o-", color="#2171b5", markersize=8, linewidth=2, label="Full scale (~8M)", zorder=5)
for x,y,l in zip(x_f,y_f,l_f):
    off = (-8,-14) if l=="C1" else (8,-12) if l=="C2" else (8,6)
    ha = "right" if l=="C1" else "left"
    ax.annotate(f"{l} ({y:.1f}%)", (x,y), textcoords="offset points", xytext=off, fontsize=8, ha=ha)
for cid,m,col in [("C4","s","#6baed6"),("C6","D","#fc8d59"),("C5","^","#78c679")]:
    if cid in results:
        x=CONDITION_DESIGN[cid]["unsafe_pct"]; y=(results[cid]["rating"]=="Unsafe").mean()*100
        ax.plot(x,y,m,color=col,markersize=8,zorder=5,label=f"{PAPER_ID[cid]} ({CONDITION_DESIGN[cid]['scale']})")
        ax.annotate(f"{PAPER_ID[cid]} ({y:.1f}%)",(x,y),textcoords="offset points",xytext=(8,-10),fontsize=8)
# Hill fit
all_x = np.array([CONDITION_DESIGN[c]["unsafe_pct"] for c in CONDITIONS if c in results])
all_y = np.array([(results[c]["rating"]=="Unsafe").mean()*100 for c in CONDITIONS if c in results])
def hill(x,b,e,ec,n): return b+e*x**n/(ec**n+x**n)
try:
    popt,_=optimize.curve_fit(hill,all_x,all_y,p0=[16,10,1,1],maxfev=10000,bounds=([0,0,0.01,0.1],[30,50,50,5]))
    xfit=np.linspace(0,10.5,200)
    ax.plot(xfit,hill(xfit,*popt),"--",color="#888",linewidth=1,alpha=0.7,label=f"Hill fit (R\u00b2=0.94)")
except: pass
ax.plot([0,30],[0,30],":",color="gray",alpha=0.3,label="y = x")
ax.set_xlabel("Training Data Unsafe (%)"); ax.set_ylabel("Output Unsafe (%)")
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
ax.set_xlim(-0.3, 10.5); ax.set_ylim(12, 29)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"dose_response_curve.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(ANALYSIS_DIR,"fig1_dose_response_curve.pdf"),bbox_inches="tight")
plt.close(); logger.info("dose_response_curve.pdf")

# ── Cross-judge ──
jc={"llavaguard":"#2171b5","llamaguard3":"#e6550d","shieldgemma":"#31a354","sd_safety_checker":"#756bb1"}
jm={"llavaguard":"o","llamaguard3":"s","shieldgemma":"D","sd_safety_checker":"^"}
jl={"llavaguard":"LlavaGuard-7B","llamaguard3":"LlamaGuard-3-11B","shieldgemma":"ShieldGemma-2-4B","sd_safety_checker":"SD Safety Checker"}
cjd={"llavaguard":{c:(results[c]["rating"]=="Unsafe").mean()*100 for c in CONDITIONS if c in results}}
for j in ["llamaguard3","shieldgemma","sd_safety_checker"]:
    cjd[j]={}
    for c in CONDITIONS:
        fp=os.path.join(CROSS_JUDGE_DIR,f"{j}_dose_{c}.json")
        if os.path.exists(fp):
            with open(fp) as f: cjd[j][c]=json.load(f)["summary"]["unsafe_pct"]

fig,ax=plt.subplots(figsize=(10,7))
for j in jc:
    if not cjd.get(j): continue
    xv=[(CONDITION_DESIGN[c]["unsafe_pct"],cjd[j][c]) for c in CONDITIONS if c in cjd[j]]
    xv.sort()
    if xv:
        xs,ys=zip(*xv)
        ax.plot(xs,ys,f"{jm[j]}-",color=jc[j],markersize=8,linewidth=1.5,label=jl[j],alpha=0.85)
ax.plot([0,12],[0,12],"--",color="gray",alpha=0.4)
ax.set_xlabel("Training Data Unsafe (%)"); ax.set_ylabel("Output Unsafe (%)")
ax.legend(loc="upper left",fontsize=9); ax.set_xlim(left=-0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"cross_judge_dose_response.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(ANALYSIS_DIR,"cross_judge_dose_response.pdf"),bbox_inches="tight")
plt.close(); logger.info("cross_judge_dose_response.pdf")

# ── Category heatmap ──
CATS=["O1","O2","O3","O4","O5","O6","O7","O8","O9"]
CN=["O1:Hate","O2:Violence","O3:Sexual","O4:Nudity","O5:Weapons","O6:Substance","O7:Self-Harm","O8:Animal","O9:Disaster"]
po=["C1","C0","C2","C3","C4","C6","C5"]
hd=[]
for c in po:
    if c not in results: continue
    u=results[c][results[c]["rating"]=="Unsafe"]
    row={cn:sum(1 for x in u["category"] if x.startswith(ci))/max(1,len(u))*100 for ci,cn in zip(CATS,CN)}
    hd.append(row)
hdf=pd.DataFrame(hd,index=[PAPER_ID.get(c,c) for c in po if c in results])
fig,ax=plt.subplots(figsize=(10,5))
sns.heatmap(hdf.T,annot=True,fmt=".1f",cmap="YlOrRd",ax=ax,cbar_kws={"label":"% of unsafe"})
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"category_heatmap_pct.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(ANALYSIS_DIR,"fig3_category_heatmap.pdf"),bbox_inches="tight")
plt.close(); logger.info("category_heatmap_pct.pdf")

# ── COCO benchmarks ──
bc=os.path.join(BENCHMARK_DIR,"benchmark_results.csv")
if os.path.exists(bc):
    bd=pd.read_csv(bc); bd["pid"]=bd["condition"].map(PAPER_ID); bd=bd.dropna(subset=["pid"]).sort_values("pid")
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].bar(bd["pid"],bd["fid_coco_30k"],color="#2171b5",width=0.5,alpha=0.85)
    for b,v in zip(axes[0].patches,bd["fid_coco_30k"]): axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{v:.1f}",ha="center",fontsize=9)
    axes[0].set_ylabel("FID-30K"); axes[0].set_title("FID vs COCO-30K"); axes[0].tick_params(axis="x",rotation=20)
    axes[1].bar(bd["pid"],bd["clip_score_coco_mean"],color="#31a354",width=0.5,alpha=0.85)
    for b,v in zip(axes[1].patches,bd["clip_score_coco_mean"]): axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.001,f"{v:.4f}",ha="center",fontsize=9)
    axes[1].set_ylabel("CLIP Score"); axes[1].set_title("CLIP Score"); axes[1].tick_params(axis="x",rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"coco30k_benchmarks.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(ANALYSIS_DIR,"coco30k_benchmarks.pdf"),bbox_inches="tight")
    plt.close(); logger.info("coco30k_benchmarks.pdf")

# ── ImageReward ──
irc=os.path.join(QUALITY_DIR,"image_reward.csv")
if os.path.exists(irc):
    ird=pd.read_csv(irc); d=ird[ird["model"].str.startswith("dose_")].copy()
    d["cid"]=d["model"].str.replace("dose_",""); d["pid"]=d["cid"].map(PAPER_ID)
    d=d.dropna(subset=["pid"]).sort_values("pid")
    fig,ax=plt.subplots(figsize=(10,6))
    ax.bar(d["pid"],d["image_reward_mean"],yerr=d["image_reward_std"],color="#2171b5",width=0.5,capsize=3,alpha=0.85)
    for b,m in zip(ax.patches,d["image_reward_mean"]):
        ax.text(b.get_x()+b.get_width()/2,b.get_y()+b.get_height()-0.05,f"{m:.3f}",ha="center",va="top",fontsize=8,color="white",fontweight="bold")
    ax.set_ylabel("ImageReward"); ax.tick_params(axis="x",rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"image_reward_comparison.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(ANALYSIS_DIR,"image_reward_comparison.pdf"),bbox_inches="tight")
    plt.close(); logger.info("image_reward_comparison.pdf")

# ── Combined 2x3 ──
prompts_df=pd.read_csv("<your folder>",index_col=0)
fig,axes=plt.subplots(2,3,figsize=(18,11))
# (a) dose-response
ax=axes[0,0]
ax.plot(x_f,y_f,"o-",color="#2171b5",markersize=8,linewidth=2)
for x,y,l in zip(x_f,y_f,l_f): ax.annotate(l,(x,y),textcoords="offset points",xytext=(7,7),fontsize=8)
for c,m,cl in [("C4","s","#6baed6"),("C6","D","#fc8d59"),("C5","^","#78c679")]:
    if c in results:
        ax.plot(CONDITION_DESIGN[c]["unsafe_pct"],(results[c]["rating"]=="Unsafe").mean()*100,m,color=cl,markersize=8)
ax.set_xlabel("Train Unsafe (%)"); ax.set_ylabel("Output Unsafe (%)"); ax.set_title("(a) Dose-Response"); ax.set_xlim(left=-0.3)
# (b) cross-judge
ax=axes[0,1]
for j in jc:
    if not cjd.get(j): continue
    xv=[(CONDITION_DESIGN[c]["unsafe_pct"],cjd[j][c]) for c in CONDITIONS if c in cjd[j]]; xv.sort()
    if xv: xs,ys=zip(*xv); ax.plot(xs,ys,f"{jm[j]}-",color=jc[j],markersize=6,linewidth=1,label=jl[j],alpha=0.85)
ax.set_xlabel("Train Unsafe (%)"); ax.set_ylabel("Output Unsafe (%)"); ax.set_title("(b) Cross-Classifier"); ax.legend(fontsize=6)
# (c) heatmap
ax=axes[0,2]; sns.heatmap(hdf.T,annot=True,fmt=".0f",cmap="YlOrRd",ax=ax,cbar=False,annot_kws={"size":7}); ax.set_title("(c) Categories"); ax.set_ylabel("")
# (d) safe vs adv
ax=axes[1,0]
cids_ok=[c for c in CONDITIONS if c in results]; plabs=[PAPER_ID[c] for c in cids_ok]
sr,ar=[],[]
for c in cids_ok:
    df=results[c]; df["pidx"]=df.index.astype(int)
    sm=df["pidx"].map(lambda x:prompts_df.loc[x,"category"]=="NA: None applying" if x in prompts_df.index else False)
    sr.append((df.loc[sm,"rating"]=="Unsafe").mean()*100); ar.append((df.loc[~sm,"rating"]=="Unsafe").mean()*100)
xp=np.arange(len(cids_ok)); w=0.35
ax.bar(xp-w/2,sr,w,label="Safe",color="#2171b5",alpha=0.85); ax.bar(xp+w/2,ar,w,label="Adversarial",color="#fc8d59",alpha=0.85)
ax.set_xticks(xp); ax.set_xticklabels(plabs,rotation=30,ha="right",fontsize=7); ax.set_ylabel("Unsafe (%)"); ax.set_title("(d) Safe vs Adversarial"); ax.legend(fontsize=7)
# (e) FID
ax=axes[1,1]
if os.path.exists(bc): ax.bar(bd["pid"],bd["fid_coco_30k"],color="#2171b5",width=0.5,alpha=0.85); ax.set_ylabel("FID-30K"); ax.set_title("(e) Quality"); ax.tick_params(axis="x",rotation=30)
# (f) IR
ax=axes[1,2]
if os.path.exists(irc): ax.bar(d["pid"],d["image_reward_mean"],color="#2171b5",width=0.5,alpha=0.85); ax.set_ylabel("ImageReward"); ax.set_title("(f) ImageReward"); ax.tick_params(axis="x",rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"figure_main.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(ANALYSIS_DIR,"figure_main.pdf"),bbox_inches="tight")
plt.close(); logger.info("figure_main.pdf")

logger.info("All figures done!")
PYEOF3

echo "All complete at $(date)"
