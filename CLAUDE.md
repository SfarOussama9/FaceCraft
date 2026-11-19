# FaceCraft — project context for AI assistants

Read this first. Full design: `docs/superpowers/specs/2026-11-08-facecraft-design.md`.

## What we are building
Text-to-face generation: a user types *"a smiling young woman with blond hair and glasses"* and our model generates a matching 64×64 face.
- **Core model (from scratch):** text-conditioned diffusion U-Net (DDPM training, DDIM sampling, cross-attention to text, classifier-free guidance, EMA).
- **Text encoder (from scratch):** small Transformer encoder, trained jointly with the U-Net.
- **Baseline (from scratch):** conditional GAN (conditional batch norm in G; spectral norm + projection head + hinge loss in D).
- **Judge (from scratch):** ResNet-style CNN predicting the 40 CelebA attributes, used only for evaluation.
- **Only pretrained parts allowed:** CLIP text encoder (one ablation only), CLIP BPE tokenizer (lookup table), InceptionV3 inside FID/KID.

## Course rules (CS495 Deep Learning, Fall 2026, MedTech/SMU — theme: Generative AI)
- Core model must be implemented and trained **from scratch**. Pretrained models = complementary only.
- Must compare with at least one baseline; quantitative + qualitative evaluation.
- Reproducible: fixed seeds, configs, `requirements.txt`, training/eval/inference code.
- Gradio demo required. Final report 8–10 pages (LaTeX).
- Team of 4: Oussama Sfar, Ines Turki, Feriel Charrad, Yasmine Jedidi. 65% of the grade is individual (oral defense) → **every member must understand every line**.

## Deadlines
- D1 proposal — done (8 Nov).
- **D2 — Sun 22 Nov 23:59:** dataset & preprocessing notebook + first baseline results.
- D3 — 21–23 Dec (TBC): code + report + presentation + live demo.
- Late = −5% per deliverable. Submit via Moodle only.

## Data (all images come from CelebA)
| Source | Covers | Use |
|---|---|---|
| CelebA (Kaggle `jessicali9530/celeba-dataset`) | 202,599 images, 40 binary attrs, official split | images + template captions |
| CelebA-Dialog (github.com/ziqihuangg/CelebA-Dialog) | same images, 0–5 levels for Bangs/Eyeglasses/Beard/Smiling/Age | degree captions (template-based) |
| Face2Text v2 (zenodo.org/records/6583553) | ~17k human-written captions, 10,559 images | human language: training + held-out test |

Preprocessing: center-crop 140×140 → resize 64×64 → [−1, 1], random h-flip. Official CelebA split everywhere. Exclude `Attractive`, `Blurry` from captions; no contradictory captions (bald + bangs, two hair colors). Three attribute pairs are held out of **all** generator training (compositional test) — never pairs used in demo prompts.

## Shared interfaces (do not change without telling the team)
- Dataset item: `(image: float[3,64,64] in [-1,1], token_ids: long[77], attrs: float[40], caption_source: str)`
- Text encoder: `forward(token_ids) -> (tokens: [B,77,D], pooled: [B,D])` — same for ours and the CLIP wrapper.
- Generators: `sample(prompts: list[str], n, seed, guidance, steps) -> uint8[N,64,64,3]` (GAN ignores guidance/steps).
- Checkpoint: one `.pt` with `model`, `ema`, `optimizer`, `step` + the config `.yaml` next to it.

## Repo layout
```
data/        download/prep scripts, caption engine, Dataset classes
models/      unet.py, text_encoder.py, diffusion.py, gan.py, classifier.py
training/    train_diffusion.py, train_gan.py, train_classifier.py
evaluation/  fid_kid.py, attribute_accuracy.py, compositional.py, attention_maps.py
demo/        app.py (Gradio)
notebooks/   01_data_and_preprocessing.ipynb (D2), results notebooks
configs/     one YAML per experiment
scripts/     helper shell/python scripts (Kaggle setup, etc.)
docs/        design + plans
```

## Compute
Kaggle (T4/P100, ~30 GPU-h/week/account, 12 h sessions) + Colab backup. Mixed precision. **Checkpoint every ~2k steps and auto-resume** — sessions die.

## How to work on this code
- Explain things in simple terms; the team must be able to defend everything orally.
- Small, readable files. Every module starts with a docstring explaining *what and why*. Comment every non-obvious line in diffusion/attention code.
- Prefer clarity over cleverness; no premature abstraction.
- Fixed seeds; all hyperparameters in YAML configs, not hard-coded.
- Pin exact package versions in `requirements.txt` once the Kaggle environment is confirmed.
- Never commit datasets, checkpoints or large outputs (see `.gitignore`).
