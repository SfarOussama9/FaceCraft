# FaceCraft — Design

**Text-to-face generation with a diffusion model and a Transformer text encoder, both trained from scratch**

CS495 Deep Learning, Fall 2026 (MedTech / SMU). Theme 3: Generative AI & Intelligent Language Systems (GA). Team of 4.
Written 8 Nov 2026. Decisions marked *(Claude's call)* were made by Claude after the team handed over the decision; everything else comes from the course rules or the team.

---

## 1. Intent

You type a description of a face, for example *"a smiling young woman with wavy blond hair and glasses"*, and the system generates a face that matches it. It does this with a model the team built and trained itself, not an API or a downloaded model.

**Why this project.** The team wants three things, in this order:
1. **Impressive.** It should be strong to demo, and strong on a CV: "built a text-to-image diffusion model from scratch".
2. **Understandable.** All four members must be able to pitch it and answer technical questions on any part. 65% of the grade is individual: contribution, understanding and oral defense.
3. **Feasible.** It has to work with free Kaggle/Colab GPUs, about 6 weeks of work, and heavy use of AI coding assistants.

**One-sentence pitch:** "We taught a neural network to remove noise from images. To create a face, it starts from pure static and removes the noise step by step, guided by your sentence, until the face appears. We built the network, the language model that reads the sentence, and a GAN to compare against, all from scratch."

**The research question the report answers:** how faithfully does a small from-scratch diffusion model follow text describing a face, compared with a GAN? And which design choices matter most: the text encoder, the conditioning mechanism, and the kind of training captions?

### Goals
- A text-conditioned **diffusion model trained from scratch** (core model) that generates recognizable 64×64 faces matching the prompt.
- A **Transformer text encoder trained from scratch**. This is the "language" half of the theme. It also covers the Transformers chapter of the individual written exam.
- A **GAN baseline** trained from scratch on the same data.
- **Evaluation that measures whether the face follows the text**, not only whether it looks real.
- A **Gradio demo** the jury can type into live.

### Non-goals
- Photorealistic high resolution. 64×64 is the target, and 128×128 is a stretch at most.
- Beating published state of the art.
- Fine-tuning large pretrained generators (Stable Diffusion, StyleGAN2-ADA). They are cut from scope.
- Generating specific real people.

### Anti-goals (failure even if it "works")
- **The core is secretly pretrained.** If the demo's quality comes from CLIP or a downloaded generator, the project fails the course's main rule.
- **A member can't explain their part, or the whole pipeline.** Code that only an AI assistant understands is a liability at the oral exam.
- **Scope creep that leaves nothing finished.** A finished small system beats an unfinished big one.
- **Faces presented as real people**, or prompts that target real identities.

---

## 2. Course constraints (fixed)
- At least one core model **implemented and trained from scratch**. Pretrained models are allowed only as complementary parts.
- The project must go beyond classification and regression, include meaningful experiments, and compare against **at least one baseline**.
- **Reproducible:** a training, evaluation and inference codebase, fixed seeds, documented environment. Jupyter notebooks are strongly recommended.
- **Minimal interface:** Gradio or Streamlit, with a live demo at the presentation.
- **Report:** 8–10 pages using the course LaTeX template.
- **Deadlines:**
  - Deliverable 2, dataset and preprocessing notebook plus first baseline results: **Sun 22 Nov 2026, 23:59**.
  - Deliverable 3, code, report, presentation and demo: **21–23 Dec 2026** (TBC).
  - Each late deliverable costs −5% of the project grade. Submit on Moodle only.
- Individual written exam on Sequence Models and Transformers: 1 or 4 Dec.

---

## 3. Data *(Claude's call, after web research)*

| Dataset | What it gives us | Role |
|---|---|---|
| **CelebA** — 202,599 aligned celebrity faces, 40 binary attributes, official split 162,770 / 19,867 / 19,962 ([Kaggle mirror](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)) | Images, plus attributes that we turn into captions | **Main training data**, for the generators and the attribute classifier |
| **CelebA-Dialog** — fine-grained levels (0–5) for Bangs, Eyeglasses, Beard, Smiling, Age on the same images ([GitHub](https://github.com/ziqihuangg/CelebA-Dialog)) | Degree words: "slight smile" vs. "laughing", "thin beard" vs. "full beard" | Extra caption source mixed into training. Its captions are template-generated, not human-written |
| **Face2Text v2** — 17,022 descriptions written by hired annotators for 10,559 CelebA images ([Zenodo](https://zenodo.org/records/6583553); use the newest version on the record; images come from CelebA) | **Real human language** about the same faces | The key addition. Used (a) mixed into training so the text encoder learns real phrasing, and (b) as a **held-out human-caption test set** |
| **Our own prompt set** — 60 prompts written by the team | Prompts nobody has seen, including unusual phrasing | Demo prompts and the qualitative figure |

**Rejected datasets, and why:**
- **FFHQ:** no text, so it adds nothing to a text-to-image model.
- **UTKFace:** the bias analysis would need an ethnicity classifier, which is sensitive and noisy. CelebA's own Male, Young and Pale_Skin attributes are enough for a basic fairness check.
- **FaceCaption-15M:** it ships URLs to scraped web photos, many links are dead, and its captions are auto-generated.
- **CelebAText-HQ:** good human captions, but no reliable public download was found.
- **MM-CelebA-HQ:** template captions again, and only needed for 128×128. It stays optional, only if the stretch goal happens.

Why this combination is the best one: CelebA gives scale (200k images, enough for a diffusion model to learn faces). Templates and Dialog give controllable, checkable attributes. Face2Text adds what every other student project on CelebA lacks: **real human sentences**. That makes it possible to show the model handles how people actually talk, not only our templates. That is the "intelligent language" angle, and it's a finding worth putting on a CV.

### Preprocessing
- **Images:** center-crop the aligned 178×218 CelebA image to 140×140, resize to **64×64**, scale to [−1, 1], random horizontal flip. Cache the whole dataset as one uint8 tensor or `.npy` file (about 2.5 GB at 64×64) so Kaggle sessions don't spend time decoding JPEGs.
- **Template captions:** convert the 40 attributes into sentences with randomized templates. Use about 20 sentence patterns, synonyms ("eyeglasses", "glasses", "spectacles"), and random attribute order, and a fresh caption is sampled every epoch. Include only visually checkable attributes, roughly 25 of them. Drop subjective attributes (Attractive) and attributes about the photo rather than the face (Blurry). Rules prevent contradictions: no "bald with bangs", and only one hair color per caption.
- **Dialog captions:** turn the 0–5 levels into degree phrases, using the same template engine.
- **Caption sampling during training:** for each image, pick one available caption source at random. Face2Text captions are used only for the ~10k images that have them, and never for the test split.
- **Tokenizer:** reuse the **CLIP BPE tokenizer** (a word-piece lookup table with no learned network weights), max length 77. It handles unknown words gracefully and makes the CLIP ablation a clean swap *(Claude's call)*.
- **Splits:** use CelebA's official partition for everything. Face2Text images inherit their CelebA split. Its test-split images form the human-caption test set.
- **Held-out attribute combinations (compositional test):** remove all training images containing 3 chosen attribute pairs. This tests whether the model can combine concepts it never saw together. *(Claude's call:)*
  - **One shared training split for every generator run:** the main diffusion model, the GAN and all ablations E2–E4. The pairs are removed once, in the data pipeline, so every run is comparable and no extra "compositional-only" run is needed.
  - Because the main and demo model never sees these pairs, **choose pairs that are not natural demo prompts.** For example *Goatee + Wearing_Hat*, *Bangs + Eyeglasses*, *Gray_Hair + Smiling*. Do **not** hold out anything in the team's demo prompts (blond + glasses, smiling + young, etc.).
  - Rule: each pair has at least 200 test images. The pairs are fixed in the Deliverable 2 notebook, before any real training run.

---

## 4. Models

### 4.1 Core: text-conditioned diffusion model (from scratch)
- **Denoiser:** a U-Net at 64×64 with
  - channel widths of about 128, 256 and 256
  - 2 residual blocks per resolution, GroupNorm, SiLU
  - sinusoidal timestep embedding
  - self-attention at 16×16 and 8×8
  - **cross-attention to the text tokens** at 16×16 and 8×8.

  The target size is about **35M parameters**. The exact widths are left to the builder, within the compute budget below.
- **Diffusion process:** standard DDPM with T = 1000. Cosine noise schedule, the network predicts the noise (ε), MSE loss. *(Claude's call: ε-prediction and cosine schedule, the most documented combination.)*
- **Sampling:** DDIM with 50 steps by default, with a demo slider from 10 to 100.
- **Classifier-free guidance:** drop the caption, replacing it with the empty prompt, 10% of the time during training. At sampling, `ε = ε_uncond + w·(ε_cond − ε_uncond)`, with the default w around 3 and a demo slider.
- **EMA of weights** (decay 0.9999) for sampling. It's standard, cheap, and noticeably improves samples.
- **Training:** AdamW with lr 1e-4 (or 2e-4) and warmup, batch 64–128, mixed precision (fp16 on T4/P100). Target **~150k steps** for the main model. Checkpoint every ~2k steps to persistent storage and resume automatically.

### 4.2 Text encoder (from scratch)
- A small Transformer encoder: 6 layers, width 512, 8 heads, learned positional embeddings, about 20M parameters.
- It outputs the full sequence of token vectors (used by cross-attention), plus a pooled vector (used by the pooled ablation).
- It is trained **jointly with the U-Net** through the diffusion loss, with no separate pretraining. That keeps it simple and makes it fully from scratch.

### 4.3 Baseline: conditional GAN (from scratch)
One baseline, built properly so the comparison is fair *(Claude's call: one well-built GAN, not a weak DCGAN plus a strong one)*:
- **Generator:** DCGAN-style upsampling with **conditional batch norm** driven by the text vector.
- **Discriminator:** conv layers with **spectral normalization** and a **projection** text-conditioning head. Hinge loss.
- **Text input:** the pooled vector from the same tokenizer, with its own small Transformer encoder trained with the GAN *(the builder may instead reuse a frozen copy of the trained diffusion text encoder, if GAN training is unstable; document which)*.
- EMA of generator weights. Train to roughly the same GPU-hours as the diffusion model.

### 4.4 Evaluation model: attribute classifier (from scratch)
- A ResNet-18-style CNN trained from scratch on real CelebA 64×64 images to predict the 40 attributes.
- Expected mean accuracy is about 90% on the real test set. Report it, because it shows the judge is trustworthy.
- It is used **only for measurement**: it checks whether generated faces contain the requested attributes. It also runs in the demo, showing ✓/✗ for each attribute in the prompt.

### 4.5 Complementary pretrained parts (allowed by the rules, and clearly labeled)
- **Frozen CLIP text encoder:** used only in one ablation (it replaces our text encoder).
- **InceptionV3:** used only inside FID/KID, as is standard.
- The CLIP tokenizer (a lookup table, see §3).

---

## 5. Experiments

All experiments use fixed seeds and the same evaluation protocol. Ablation runs may be shorter (e.g. 60k steps) as long as every run being compared uses the same step count.

| # | Comparison | What it shows |
|---|---|---|
| E1 | **Diffusion vs. GAN** (main result) | Is diffusion actually better at realism and at following text? |
| E2 | Our Transformer text encoder vs. **frozen CLIP** text encoder | How far can a from-scratch language model get vs. one pretrained on 400M pairs? |
| E3 | **Cross-attention** vs. a single pooled text vector added to the time embedding | Why modern text-to-image models use cross-attention |
| E4 | Training captions: **templates only** vs. **templates + Dialog + Face2Text human captions**, tested on human-written test captions | Does real human language in training help on real human prompts? (the "language" finding) |
| E5 | Guidance scale w ∈ {1, 2, 3, 5, 7.5} × DDIM steps ∈ {10, 25, 50, 100} | The quality vs. diversity vs. speed trade-off. Inference only, so it costs no extra training |

### Metrics
- **FID and KID** on 10,000 generated samples vs. real CelebA test images, at 64×64. Use the same protocol and library for every model (e.g. `torchmetrics` or `clean-fid`).
- **Attribute accuracy** (primary text-fidelity metric): for every attribute named in the prompt, does the classifier detect it? Report it per attribute and on average.
- **Compositional accuracy:** attribute accuracy on the held-out attribute pairs only.
- **Human-caption test:** FID/KID and attribute accuracy using Face2Text test captions as prompts. *(Claude's call)* Which attributes are scored:
  - A simple **keyword matcher** (synonym lists plus basic negation, e.g. "no beard", "clean-shaven") finds the attributes a caption mentions.
  - An attribute is scored only if the matcher finds it **and** the image's CelebA label agrees. This removes matcher mistakes without penalizing traits the prompt never asked for.
  - Validate the matcher by hand on 100 captions and report its precision. It lives in `evaluation/` and is owned by Member 4, with Member 1 supplying the synonym lists from the caption engine.
- **Human preference vote:** about 20 classmates, blind A/B choice between diffusion and GAN on the same prompts.
- **Explicitly not used:** Inception Score (a poor fit for faces), and CLIPScore (CLIP is weak on 64px faces, and it's also an ablation component).

### Qualitative figures for the report and slides
- A grid of fixed prompts × models, using the same seeds.
- A denoising strip from noise to face.
- **Cross-attention maps** showing that the word "glasses" lights up the eyes and "smiling" lights up the mouth. This is the single most convincing figure for the jury.
- Prompt interpolation, e.g. "young man" → "old man".
- Failure cases, shown honestly.

### Ethics and bias (short report section)
- Compare the attribute distribution of generated faces for neutral prompts ("a person") against CelebA, using the Male, Young and Pale_Skin attributes. Discuss CelebA's skew, since it is mostly young, light-skinned celebrities.
- Outputs are synthetic and labeled as such in the demo. Identity prompts (celebrity names) are not supported or encouraged. Data is used under the non-commercial research licenses.

---

## 6. Demo (Gradio)
- A prompt box. Example prompts are pre-filled from the team's prompt set.
- Sliders for guidance scale and DDIM steps, and a seed field. Generate 4 faces per prompt.
- An **animated GIF of the selected face emerging from noise**.
- An attribute check under each face, from the classifier: "smiling ✓ glasses ✓ blond ✗".
- A **side-by-side view: diffusion vs. GAN** for the same prompt.
- An **Edit tab:** take a generated face, partially re-noise it (about 50–60%), and denoise with an edited prompt ("…now with a beard"). This is SDEdit-style editing and needs no extra training.
- It must run on a laptop CPU, slowly, or a free GPU. Fallback: a pre-generated gallery in case the live demo fails.

---

## 7. Compute and tooling
- **Budget:** Kaggle gives about 30 GPU-h per week per account, and there are 4 accounts. Colab is the backup. Sessions are about 12 h at most, so **resuming from checkpoints is mandatory**.
- **Rough training costs** (estimates, to be confirmed in week 1):
  - main diffusion model: about 15–25 GPU-h
  - each ablation: about 6–10 GPU-h
  - GAN: about 8–12 GPU-h
  - classifier: about 1–2 GPU-h.

  The total fits comfortably in the budget.
- **Stack:** PyTorch, YAML configs, one entry-point script per stage, `requirements.txt` with pinned versions, fixed seeds, Weights & Biases (or TensorBoard) logging, checkpoints saved to Kaggle Datasets or Google Drive.
- **Repo layout:**
  ```
  FaceCraft/
    data/          # download scripts, caption templates, dataset classes, caching
    models/        # unet.py, text_encoder.py, diffusion.py, gan.py, classifier.py
    training/      # train_diffusion.py, train_gan.py, train_classifier.py
    evaluation/    # fid_kid.py, attribute_accuracy.py, compositional.py, attention_maps.py
    demo/          # app.py (Gradio)
    notebooks/     # 01_data_and_preprocessing.ipynb (Deliverable 2), results notebooks
    configs/       # *.yaml, one per experiment
    docs/
  ```
- **Code style:** small files, clear names, a docstring at the top of every module explaining *what and why* in plain words, and comments on every non-obvious line in the diffusion and attention code. Each member must be able to read every file.

---

## 8. Team split *(Claude's call; each person owns a defensible technical core)*

| Member | Owns | Also leads |
|---|---|---|
| **1: Data & Language** | Dataset download and caching, caption template engine, Dialog and Face2Text integration, tokenizer, **Transformer text encoder** | Deliverable 2 notebook; dataset section of the report |
| **2: Diffusion core** | U-Net with self- and cross-attention, noise schedule, DDPM training loop, DDIM sampler, CFG, EMA, checkpoint and resume | Main training runs; method section of the report |
| **3: Baseline & Judge** | Conditional GAN (CBN, SN, projection, hinge), **attribute classifier** | GAN results; baseline section of the report |
| **4: Experiments & Demo** | FID/KID, attribute and compositional evaluation, ablations E2–E5, attention maps, Gradio app, edit tab | Results section, slides, live demo |

There is a weekly 30-minute sync where each member explains their part to the others. This is the oral exam insurance.

### How the team builds it
- The **four students build it**, each with their own AI coding assistant, as four parallel work packages, one per row above. AI writes drafts. The owner must be able to explain every line before it is merged.
- To avoid people blocking each other, the **shared interfaces are fixed in week 1**, as small Python stubs plus docstrings in the repo:
  - **Dataset:** `__getitem__` returns `(image: float tensor 3×64×64 in [−1,1], token_ids: long tensor [77], attrs: float tensor [40])`, plus a `caption_source` tag.
  - **Text encoder:** `forward(token_ids) -> (tokens: [B,77,D], pooled: [B,D])`, with the same signature for our encoder and the CLIP wrapper.
  - **Generator sampling:** `sample(prompts: list[str], n, seed, guidance, steps) -> uint8 images [N,64,64,3]`. Diffusion and GAN both implement it (the GAN ignores guidance and steps). Evaluation and demo code depend only on this.
  - **Checkpoint:** one `.pt` file with model, EMA, optimizer and step, plus a config `.yaml` saved next to it.
- The plan covers **code, training runs** (which run goes on whose Kaggle account), **the report, and rehearsals**, not just code.

---

## 9. Milestones
- **9–15 Nov:**
  - repo, data download, cache at 64×64, caption engine
  - the attribute classifier trains
  - GAN and diffusion first runs, unconditional or with simple captions, to prove the pipelines work.
- **Sun 22 Nov, Deliverable 2:** a notebook with
  - data exploration: attribute frequencies and imbalance, caption examples from all three sources, Face2Text statistics, the held-out pairs
  - the preprocessing pipeline
  - **first results:** classifier accuracy, first GAN samples with a first FID, and first diffusion samples.
- **23 Nov – 6 Dec:**
  - main text-conditioned diffusion run with CFG
  - GAN trained to the same budget
  - first full metrics
  - (exam on 1 or 4 Dec)
- **7–15 Dec:** ablations E2–E4, E5 sweep, human vote, attention maps, demo, report draft.
- **16–20 Dec:** final numbers, report polish, rehearsals (each member presents their part plus answers on any part).
- **21–23 Dec:** Deliverable 3.

## 10. Risks
| Risk | Plan |
|---|---|
| Diffusion training slower than estimated | Measure it/s in week 1; shrink the U-Net widths or the step count; ablations run shorter at equal steps |
| GAN collapses | SN, hinge loss and EMA are already in; fall back to a frozen text encoder; a weak but honest GAN is still a valid baseline |
| Face2Text download or alignment problems | E4 degrades to templates vs. templates + Dialog, and our own 60 prompts become the human test set |
| Kaggle session dies mid-run | Checkpoints every ~2k steps, automatic resume |
| Someone can't explain a part | Weekly explain-back syncs; heavy comments; every member reruns the full pipeline once |

## 11. Stretch goals (only after the core is done)
1. 128×128 with MM-CelebA-HQ / CelebA-HQ (a second, upsampling diffusion stage, or retraining at 128).
2. Precision and recall metrics.
3. Hugging Face Space hosting the demo, for the CV link.

## 12. Left to the builder
- Exact U-Net widths, attention placement and batch size, within the compute budget.
- Learning-rate schedule details, and the logging tool (W&B vs. TensorBoard).
- The exact caption template wording, and which ~25 attributes are "checkable".
- The choice of the 3 held-out attribute pairs (rule: each must have at least 200 test images).
- The library for FID/KID, as long as it is the same for all models.
