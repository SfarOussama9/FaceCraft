"""Generate notebooks/01_data_and_preprocessing.ipynb (Deliverable 2) with nbformat.

    python scripts/build_notebook.py

Keeping the notebook as generated code means it is reproducible and reviewable in git diffs.
Run it afterwards with Jupyter ("Run All"). Environment variables understood by the notebook:
  FACECRAFT_DATA_CONFIG  path to the data config (default configs/data.yaml)
  FACECRAFT_FAKE_DATA=1  use a tiny synthetic dataset (smoke test, no real data needed)
"""
from pathlib import Path

import nbformat as nbf

CELLS = []


def md(text: str):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    CELLS.append(nbf.v4.new_code_cell(text.strip()))


# ----------------------------------------------------------------------------------------------
md("""
# FaceCraft — Deliverable 2: Dataset & Pre-processing
**CS495 Deep Learning · Fall 2026 · Theme 3: Generative AI & Intelligent Language Systems**
Team: Oussama Sfar, Ines Turki, Feriel Charrad, Yasmine Jedidi

**Project.** A user types a description of a face (*"a smiling young woman with blond hair and glasses"*) and our model,
trained from scratch, generates a matching 64×64 face. The core model is a text-conditioned diffusion model with a
Transformer text encoder; the baseline is a text-conditional GAN.

**This notebook documents:**
1. Data acquisition — 2. Cleaning — 3. Exploration — 4. Labeling (captions) — 5. Pre-processing pipeline —
6. Tokenization — 7. The PyTorch `Dataset` / `DataLoader` — 8. **First results**: attribute classifier and GAN baseline.

All the logic lives in the repository's Python modules (`data/`, `models/`, `training/`, `evaluation/`), which are
unit-tested (`pytest`). The notebook calls them and shows what they do.
""")

md("""
## 0. Setup
`RUN_TRAINING = False` loads results saved by the training scripts (`outputs/`) instead of retraining (training takes hours).
""")
code("""
import os, sys, json, glob, random, collections, re
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

from utils.config import default_data_config, load_config
from utils.seed import set_seed

RUN_TRAINING = False                     # True = run the training scripts from this notebook (hours)
FAKE = os.environ.get("FACECRAFT_FAKE_DATA") == "1"
set_seed(0)
pd.set_option("display.max_rows", 60)
plt.rcParams["figure.dpi"] = 110

if FAKE:   # tiny synthetic dataset with the real file formats (used to smoke-test this notebook)
    import tempfile, yaml
    from tests.conftest import write_fake_celeba
    from data.celeba import load_celeba_metadata as _lm
    from data.preprocess import build_cache as _bc
    tmp = Path(tempfile.mkdtemp())
    celeba = write_fake_celeba(tmp / "celeba", n_images=60)
    (tmp / "face2text").mkdir()
    (tmp / "face2text" / "clean_train_2.1.json").write_text(json.dumps(
        [{"response_id": k, "filename": f"{k:06d}.jpg", "description": "A young woman with long brown hair and a big smile."}
         for k in range(1, 30)]))
    (tmp / "dialog").mkdir()
    (tmp / "dialog" / "combined_annotation.txt").write_text("img_name Bangs Eyeglasses No_Beard Smiling Young\\n" +
        "".join(f"{k:06d}.jpg {k % 6} {k % 3} {k % 2} {k % 6} {k % 5}\\n" for k in range(1, 61)))
    ids = list(_lm(str(celeba)).index)
    _bc(str(celeba / "img_align_celeba" / "img_align_celeba"), ids, str(tmp / "cache.npy"), workers=1)
    cfg_fake = {"celeba_root": str(celeba), "face2text_dir": str(tmp / "face2text"),
                "dialog_labels": str(tmp / "dialog" / "combined_annotation.txt"), "cache_path": str(tmp / "cache.npy"),
                "image_size": 64, "crop": 140, "f2t_prob": 0.5, "dialog_prob": 0.3, "heldout_pairs": [], "seed": 0}
    DATA_CFG = str(tmp / "data.yaml")
    Path(DATA_CFG).write_text(yaml.safe_dump(cfg_fake))
else:
    DATA_CFG = default_data_config()        # configs/data_kaggle.yaml automatically on Kaggle

cfg = load_config(DATA_CFG)
print("data config:", DATA_CFG, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

# ----------------------------------------------------------------------------------------------
md("""
## 1. Data acquisition
All **images** come from **CelebA**. Two other datasets add **text** describing the *same* images:

| Source | What it gives | How we obtain it |
|---|---|---|
| **CelebA** (Liu et al., 2015) | 202,599 aligned celebrity faces, 40 binary attributes, official train/val/test split | Kaggle `jessicali9530/celeba-dataset` |
| **CelebA-Dialog** (Jiang et al., 2021) | 0–5 *degree* levels for Bangs, Eyeglasses, Beard, Smiling, Age | `combined_annotation.txt` (Google Drive, linked from the GitHub page) |
| **Face2Text v2.1** (Tanti et al., 2022) | ~18.7k descriptions **written by people** for ~12.2k CelebA images | Zenodo record 10973388 |

`scripts/download_data.py` downloads Face2Text automatically and explains how to get the other two.
""")
code("""
from data.celeba import ATTR_NAMES, load_celeba_metadata
from data.face2text import load_face2text
from data.dialog import load_dialog_levels

meta = load_celeba_metadata(cfg["celeba_root"])
f2t_files = sorted(glob.glob(os.path.join(cfg["face2text_dir"], "clean_*.json")))
f2t = load_face2text(f2t_files)
dialog = load_dialog_levels(cfg.get("dialog_labels"))

summary = pd.DataFrame({
    "images / rows": [len(meta), sum(len(v) for v in f2t.values()), 0 if dialog is None else len(dialog)],
    "distinct CelebA images": [len(meta), len(f2t), 0 if dialog is None else dialog.index.nunique()],
}, index=["CelebA (40 attributes)", "Face2Text (descriptions)", "CelebA-Dialog (5 levels)"])
display(summary)
display(meta["split"].value_counts().rename("images per official split").to_frame())
""")

# ----------------------------------------------------------------------------------------------
md("""
## 2. Cleaning
Checks performed:
* every image listed in the CSVs exists (and decodes — the cache build in §5 opens every file);
* Face2Text descriptions join onto CelebA (unmatched file names, empty or duplicated descriptions);
* CelebA-Dialog coverage;
* **contradictory labels** in CelebA. Some are physically impossible (bald *and* bangs, two hair colours), and
  facial-hair labels on women are mostly annotation noise. We don't delete images; our **caption rules** (§4)
  simply never state contradictory facts.
""")
code("""
img_dir = Path(cfg["celeba_root"]) / "img_align_celeba" / "img_align_celeba"
missing = [i for i in meta.index if not (img_dir / i).exists()]
print(f"images listed: {len(meta):,}   missing on disk: {len(missing)}")

raw_rows = []
for p in f2t_files:
    raw_rows += json.load(open(p, encoding="utf-8"))
f2t_df = pd.DataFrame(raw_rows)
f2t_df["description"] = f2t_df["description"].fillna("").str.strip()
print(f"Face2Text rows: {len(f2t_df):,}   empty: {(f2t_df.description == '').sum()}   "
      f"duplicated (same image + text): {f2t_df.duplicated(['filename', 'description']).sum()}   "
      f"file names not in CelebA: {(~f2t_df.filename.isin(meta.index)).sum()}")
if dialog is not None:
    print(f"CelebA images without Dialog labels: {len(meta.index.difference(dialog.index))}")

hair = meta[["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]].sum(1)
contradictions = pd.Series({
    "Bald and Bangs": int(((meta.Bald == 1) & (meta.Bangs == 1)).sum()),
    "Bald and a hair colour": int(((meta.Bald == 1) & (hair > 0)).sum()),
    "2+ hair colours": int((hair >= 2).sum()),
    "female with goatee/mustache/sideburns": int(((meta.Male == 0) & (meta[["Goatee", "Mustache", "Sideburns"]].sum(1) > 0)).sum()),
    "female with a beard (No_Beard = 0)": int(((meta.Male == 0) & (meta.No_Beard == 0)).sum()),
}, name="images")
display(contradictions.to_frame())
""")

# ----------------------------------------------------------------------------------------------
md("""
## 3. Exploration
### 3.1 Attribute frequencies — the dataset is very imbalanced
""")
code("""
freq = meta[ATTR_NAMES].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
freq.plot.barh(ax=ax, color="#4c72b0")
ax.set_xlabel("fraction of images with the attribute"); ax.set_title("CelebA attribute frequencies")
plt.tight_layout(); plt.show()
print("rarest:", ", ".join(f"{a} {v:.1%}" for a, v in freq.head(5).items()))
print("most common:", ", ".join(f"{a} {v:.1%}" for a, v in freq.tail(5).items()))
""")
md("""
### 3.2 Which attributes appear together?
Correlations matter for generation: if *Heavy_Makeup* almost always comes with *Wearing_Lipstick*, the model will
struggle to draw one without the other.
""")
code("""
from data.captions import CAPTION_ATTRS
corr = meta[CAPTION_ATTRS].astype(float).corr()
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(CAPTION_ATTRS))); ax.set_xticklabels(CAPTION_ATTRS, rotation=90, fontsize=7)
ax.set_yticks(range(len(CAPTION_ATTRS))); ax.set_yticklabels(CAPTION_ATTRS, fontsize=7)
plt.colorbar(im, fraction=0.04); ax.set_title("Correlation between caption attributes"); plt.tight_layout(); plt.show()
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values()
print("strongest positive:", pairs.tail(4).round(2).to_dict())
print("strongest negative:", pairs.head(4).round(2).to_dict())
""")
md("### 3.3 Demographic skew (gender × age)")
code("""
ct = pd.crosstab(meta.Male.map({0: "female", 1: "male"}), meta.Young.map({0: "not young", 1: "young"}), normalize="all")
display((ct * 100).round(1).astype(str) + " %")
print("CelebA is mostly young celebrities, with more women than men. Generated faces will inherit this bias;"
      " we will measure it on neutral prompts in the final evaluation.")
""")
md("### 3.4 Human-written descriptions (Face2Text)")
code("""
lengths = f2t_df.description.str.split().str.len()
words = collections.Counter(w for t in f2t_df.description.str.lower() for w in re.findall(r"[a-z']+", t))
stop = set("a an the and with of has is her his she he are to in on very at it this that has its have as".split())
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].hist(lengths, bins=40, color="#55a868"); ax[0].set_title("words per description"); ax[0].set_xlabel("words")
top = [(w, c) for w, c in words.most_common(80) if w not in stop][:20]
ax[1].barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color="#55a868"); ax[1].set_title("most frequent words")
plt.tight_layout(); plt.show()
print(f"median {lengths.median():.0f} words, 95th percentile {lengths.quantile(.95):.0f}, max {lengths.max()}")
""")
md("""
### 3.5 CelebA-Dialog levels — and a check of what they mean
Two column names in the file are misleading. By comparing each level with CelebA's own yes/no labels we found that
`No_Beard` actually stores the **amount of beard** (0 = none) and `Young` stores the **age** (0 = youngest).
The table shows, for each level, the fraction of images CelebA labels as having the attribute:
""")
code("""
if dialog is not None:
    j = dialog.join(meta[["Bangs", "Eyeglasses", "No_Beard", "Smiling", "Young"]], rsuffix="_celeba")
    check = pd.DataFrame({
        "Bangs: % CelebA Bangs": j.groupby("Bangs")["Bangs_celeba"].mean(),
        "Eyeglasses: % CelebA Eyeglasses": j.groupby("Eyeglasses")["Eyeglasses_celeba"].mean(),
        "Beard: % CelebA *No*_Beard": j.groupby("Beard")["No_Beard"].mean(),
        "Smiling: % CelebA Smiling": j.groupby("Smiling")["Smiling_celeba"].mean(),
        "Age: % CelebA Young": j.groupby("Age")["Young"].mean(),
    }).rename_axis("level")
    display((check * 100).round(1))
    dialog.apply(lambda c: c.value_counts().sort_index()).plot.bar(figsize=(10, 3), title="images per level")
    plt.tight_layout(); plt.show()
else:
    print("CelebA-Dialog labels not available: Dialog captions are skipped.")
""")

# ----------------------------------------------------------------------------------------------
md("""
## 4. Labeling: from labels to sentences
Each training image gets a **new caption every time it is loaded**, drawn from up to three sources:

| Source | Example | Used when |
|---|---|---|
| **Template** (`data/captions.py`) | *"a photo of a young man with black hair and stubble, wearing glasses"* | always available |
| **Dialog** (`data/dialog.py`) | *"a middle-aged man with a slight smile and thick-framed glasses"* | 30 % of the time when no human text |
| **Face2Text** (human) | *"A young boy with short dark hair and thick arched eyebrows…"* | 50 % of the time for images that have one |

**Template engine:** 15 sentence patterns, synonyms (*glasses / eyeglasses / spectacles*), random order of details.
**Rules:** never mention *Attractive, Blurry, Wearing_Necktie, Wearing_Necklace*; at most one hair colour;
*Bald* removes hair colour, bangs and texture; facial hair only for men.
""")
code("""
from data.preprocess import load_cache
from data.captions import template_caption
from data.dialog import dialog_caption

cache, cache_ids = load_cache(cfg["cache_path"])
pos = {i: k for k, i in enumerate(cache_ids)}
rng = random.Random(3)
with_f2t = [i for i in meta.index if i in f2t]
show = (with_f2t[:3] + [i for i in meta.index[::max(1, len(meta) // 50)] if i not in f2t][:3])[:6]
fig, axes = plt.subplots(len(show), 1, figsize=(12, 2.0 * len(show)))
for ax, i in zip(np.atleast_1d(axes), show):
    t = template_caption(meta.loc[i, ATTR_NAMES].to_dict(), rng)
    d = dialog_caption(dialog.loc[i].to_dict(), meta.loc[i, "Male"] == 1, rng) if dialog is not None and i in dialog.index else "—"
    h = f2t[i][0] if i in f2t else "—"
    ax.imshow(np.array(cache[pos[i]])); ax.set_xticks([]); ax.set_yticks([])
    ax.text(70, 10, f"TEMPLATE: {t}\\nDIALOG:   {d}\\nHUMAN:    {h[:140]}", fontsize=8, va="top", family="monospace")
    ax.set_xlim(0, 700); ax.set_title(i, loc="left", fontsize=8)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------------
md("""
## 5. Pre-processing pipeline
1. **Crop:** the aligned 178×218 photo → centered 140×140 square (keeps the face, drops most background).
2. **Resize:** 140×140 → **64×64** (bicubic, anti-aliased).
3. **Cache:** all 202,599 images stored once as a single `uint8` array (~2.5 GB, `data/preprocess.py`), read by
   memory-mapping during training — decoding JPEGs is no longer a bottleneck.
4. **At load time:** scale to **[−1, 1]**, random **horizontal flip** (training split only).
""")
code("""
i = meta.index[0]
orig = Image.open(img_dir / i).convert("RGB")
w, h = orig.size; l, t = (w - 140) // 2, (h - 140) // 2
fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
ax[0].imshow(orig); ax[0].add_patch(plt.Rectangle((l, t), 140, 140, fill=False, ec="red", lw=2)); ax[0].set_title(f"original {w}x{h}")
ax[1].imshow(orig.crop((l, t, l + 140, t + 140))); ax[1].set_title("crop 140x140")
ax[2].imshow(np.array(cache[pos[i]])); ax[2].set_title("cached 64x64")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
print(f"cache: {cfg['cache_path']}  shape {cache.shape}  dtype {cache.dtype}  size {cache.nbytes / 1e9:.2f} GB")
""")
md("""
### 5.1 Splits and the held-out attribute pairs (compositional test)
We keep the **official CelebA split** everywhere; Face2Text images inherit their CelebA split (so no human description
of a test face is ever seen in training).

For the **compositional generalization** test we choose 3 attribute pairs and remove every *training* image that has
both (generators see each attribute alone, never together). Rules: ≥ 200 test images with the pair, ≤ 2 % of training
images, never a pair used in the demo (`data/splits.py`).
""")
code("""
from data.splits import choose_heldout_pairs, heldout_mask
pairs = [tuple(p) for p in cfg["heldout_pairs"]]
if not FAKE:
    chosen = choose_heldout_pairs(meta, k=3, min_test=200, seed=0)
    print("re-derived pairs:", chosen, "| matches config:", sorted(chosen) == sorted(pairs))
rows = []
for a, b in pairs:
    both = (meta[a] == 1) & (meta[b] == 1)
    rows.append({"pair": f"{a} + {b}", **{s: int((both & (meta.split == s)).sum()) for s in ["train", "val", "test"]}})
display(pd.DataFrame(rows))
m = heldout_mask(meta, pairs)
split_table = meta.split.value_counts().rename("all images").to_frame()
split_table["used by generators"] = split_table["all images"] - (m & (meta.split == "train")).groupby(meta.split).sum().reindex(split_table.index).fillna(0).astype(int)
split_table["with human (Face2Text) text"] = meta[meta.index.isin(f2t)].split.value_counts()
display(split_table)
""")

# ----------------------------------------------------------------------------------------------
md("""
## 6. Tokenization
Captions are converted to token ids with **CLIP's BPE tokenizer**: just a vocabulary lookup table (no network
weights), so the *from scratch* rule holds — our own Transformer learns what the tokens mean. Every caption becomes
exactly **77 ids** (padded / truncated).
""")
code("""
from data.tokenizer import make_tokenizer
from training.common import build_dataset
tok = make_tokenizer("hash" if FAKE else "clip")
train_ds = build_dataset(cfg, "train", tok)
rng = random.Random(0)
by_source = collections.defaultdict(list)
for i in rng.sample(train_ds.ids, min(3000, len(train_ds.ids))):
    c, s = train_ds.caption_for(i, rng)
    by_source[s].append(c)
lens = {s: (tok(cs) != tok.pad_id).sum(1).numpy() + 1 for s, cs in by_source.items()}
fig, ax = plt.subplots(figsize=(8, 3))
for s, l in lens.items():
    ax.hist(l, bins=30, alpha=.6, label=f"{s} (n={len(l)})")
ax.axvline(77, color="red", ls="--", label="max 77"); ax.legend(); ax.set_xlabel("tokens per caption")
plt.tight_layout(); plt.show()
allc = [c for cs in by_source.values() for c in cs]
raw = [len(tok.tok(c)["input_ids"]) for c in allc] if hasattr(tok, "tok") else [len(c.split()) for c in allc]
print(f"captions longer than 77 tokens (truncated): {np.mean(np.array(raw) > 77):.2%}")
""")

# ----------------------------------------------------------------------------------------------
md("""
## 7. The PyTorch Dataset and DataLoader
`FaceCraftDataset` (`data/dataset.py`) returns, for every index, the tuple shared by all models:
`(image [3,64,64] in [−1,1], token_ids [77], attrs [40] ∈ {0,1}, caption_source)`.
""")
code("""
from torch.utils.data import DataLoader
print(f"train items: {len(train_ds):,}  (held-out pairs removed)  | with human text: {len(train_ds.f2t):,}")
dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
images, token_ids, attrs, sources = next(iter(dl))
print("images", tuple(images.shape), images.dtype, f"range [{images.min():.2f}, {images.max():.2f}]")
print("token_ids", tuple(token_ids.shape), token_ids.dtype, "| attrs", tuple(attrs.shape), "| sources", collections.Counter(sources))
rng = random.Random(1)
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for ax, k in zip(axes.flat, range(8)):
    img, _, _, _ = train_ds[k]
    cap, src = train_ds.caption_for(train_ds.ids[k], rng)
    ax.imshow(((img.permute(1, 2, 0) + 1) / 2).clamp(0, 1)); ax.axis("off")
    words = cap.split(); lines = [" ".join(words[j:j + 6]) for j in range(0, min(len(words), 24), 6)]
    ax.set_title(f"[{src}]\\n" + "\\n".join(lines), fontsize=7)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------------
md("""
## 8. First results
### 8.1 Attribute classifier ("the judge"), trained from scratch
A ResNet-18-style CNN (`models/classifier.py`) trained on real CelebA 64×64 to predict the 40 attributes. Later it
checks whether generated faces contain what the prompt asked for (**attribute accuracy**). Baseline: always
predicting each attribute's most frequent value.
""")
code("""
import subprocess
if RUN_TRAINING:
    subprocess.run([sys.executable, "-m", "training.train_classifier", "--config", "configs/classifier.yaml"], check=True)
p = Path("outputs/classifier/metrics.json")
if p.exists():
    clf = json.loads(p.read_text())
    print(f"test mean accuracy: {clf['mean_acc']:.4f}   majority baseline: {clf['majority_baseline_mean_acc']:.4f}")
    per = pd.DataFrame({"classifier": clf["per_attr_acc"], "majority": clf["majority_baseline_per_attr"]})
    per["gain"] = per.classifier - per.majority
    per.sort_values("classifier").plot.barh(y=["majority", "classifier"], figsize=(9, 9), width=.8,
                                            title="test accuracy per attribute")
    plt.tight_layout(); plt.show()
    display(per.sort_values("classifier").head(8).round(3))
else:
    print("No classifier results yet: run  python -m training.train_classifier --config configs/classifier.yaml")
""")
md("""
### 8.2 Baseline generator: text-conditional GAN, trained from scratch
Generator with **conditional batch norm**, discriminator with **spectral normalization** and a **projection** head,
hinge loss, and the matching-aware trick (real image + wrong caption counts as fake). Its text encoder is a small
Transformer trained from scratch together with the GAN (`models/gan.py`, `models/text_encoder.py`).
**Preliminary FID/KID** on 5,000 test captions (final evaluation will use 10,000).
""")
code("""
if RUN_TRAINING:
    subprocess.run([sys.executable, "-m", "training.train_gan", "--config", "configs/gan.yaml"], check=True)
gan_dir = Path("outputs/gan")
if (gan_dir / "losses.csv").exists():
    L = pd.read_csv(gan_dir / "losses.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
    L.plot(x="step", y=["loss_D", "loss_G"], ax=ax[0], title="GAN losses", alpha=.8)
    L.plot(x="step", y=["d_real", "d_fake", "d_wrong"], ax=ax[1], title="discriminator scores", alpha=.8)
    plt.tight_layout(); plt.show()
    grids = sorted(glob.glob(str(gan_dir / "samples" / "step_*.png")))
    picks = [grids[0], grids[len(grids) // 2], grids[-1]] if len(grids) >= 3 else grids
    prompts = load_config("configs/gan.yaml")["sample_prompts"]
    for g in picks:
        plt.figure(figsize=(7, 7)); plt.imshow(Image.open(g)); plt.axis("off")
        plt.title(Path(g).stem + " — one row per prompt:\\n" + "\\n".join(prompts), fontsize=7); plt.show()
    if (gan_dir / "metrics.json").exists():
        display(pd.Series(json.loads((gan_dir / "metrics.json").read_text())).to_frame("GAN baseline"))
else:
    print("No GAN results yet: run  python -m training.train_gan --config configs/gan.yaml")
""")

# ----------------------------------------------------------------------------------------------
md("""
## 9. Summary and next steps
**Done for Deliverable 2**
* Three data sources acquired, cleaned and joined on CelebA image ids; official split preserved.
* Caption pipeline with three sources (template, Dialog degrees, human descriptions) and truthfulness rules.
* 64×64 pre-processing + memory-mapped cache; `FaceCraftDataset` / `DataLoader` with the shared interface.
* Held-out attribute pairs fixed for the compositional test.
* First results: attribute classifier (the evaluation judge) and the GAN baseline with preliminary FID/KID.

**Next (towards Deliverable 3)**
* Core model: text-conditioned **diffusion U-Net with cross-attention**, Transformer text encoder (6 layers),
  classifier-free guidance, DDIM sampling, EMA.
* Experiments: diffusion vs. GAN; our text encoder vs. CLIP; cross-attention vs. pooled conditioning;
  template-only vs. human captions; guidance-scale / sampling-step sweep.
* Metrics: FID/KID (10k samples), attribute accuracy, compositional accuracy, human-caption test, human vote.
* Gradio demo.
""")


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_and_preprocessing.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
