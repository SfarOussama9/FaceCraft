# Deliverable 2 — Data Pipeline, Preprocessing Notebook & First Baselines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** By Sun 22 Nov 23:59, deliver `notebooks/01_data_and_preprocessing.ipynb`. It documents data acquisition, cleaning, labeling and preprocessing, and shows first results: attribute-classifier accuracy and first text-conditional GAN samples with a preliminary FID.

**Architecture:**
- Small, single-purpose modules in `data/` build:
  - one metadata table (CelebA attributes + split + Face2Text + optional Dialog levels)
  - a uint8 image cache at 64×64
  - a caption engine
  - a PyTorch `Dataset` that returns the shared interface tuple.
- `models/`, `training/` and `evaluation/` contain the classifier, the GAN with a from-scratch Transformer text encoder, and FID/KID.
- Training scripts are YAML-configured. They write `outputs/<run>/metrics.json` and sample images, which the notebook loads.
- Local unit tests run on CPU with synthetic data. Real runs happen on Kaggle.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, pandas, numpy, Pillow, PyYAML, `transformers` (CLIP tokenizer only), `torchmetrics[image]` (FID/KID), matplotlib, nbformat, pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-facecraft-design.md` (sections 3, 4.2–4.4, 5 Metrics, 7, 9). Also `CLAUDE.md`.

## Global Constraints
- Images: center-crop the aligned 178×218 CelebA image to **140×140**, resize to **64×64** (bilinear, antialias), store as uint8. At load time, scale to **[−1, 1]** and apply a random horizontal flip (train only).
- Use the official CelebA partition everywhere: `list_eval_partition.csv` (0 = train 162,770 / 1 = val 19,867 / 2 = test 19,962). Face2Text images inherit their CelebA split. **Ignore Face2Text's own train/dev/test files' split meaning.** Load all three and re-split by CelebA partition.
- Face2Text source: **v2.1**, Zenodo record 10973388, `face2text_v2.1.zip`. Use `clean_{train,dev,test}_2.1.json`, fields `filename`, `description`. Expected totals: 18,690 captions on 12,227 images. Under the CelebA split: train 15,070 / val 1,844 / test 1,776 captions.
- Captions never mention `Attractive`, `Blurry`, `Wearing_Necktie` or `Wearing_Necklace` (the last two are mostly outside the 140×140 crop).
- Caption contradiction rules:
  - at most one hair color is mentioned (if 2+ are set, mention none)
  - `Bald` suppresses hair color, bangs and hair texture
  - facial-hair attributes are only mentioned when `Male = 1`.
- Tokenizer: `openai/clip-vit-base-patch32` CLIP tokenizer, `max_length=77`, padding `"max_length"`, truncation on.
- Dataset item, exactly: `(image: float32[3,64,64] in [-1,1], token_ids: int64[77], attrs: float32[40] in {0,1}, caption_source: str)`. The attribute order is CelebA's CSV column order. `caption_source ∈ {"template", "dialog", "face2text"}`.
- Caption sampling per item:
  - if the image has Face2Text captions → `"face2text"` with probability `f2t_prob` (default **0.5**)
  - otherwise → `"dialog"` with probability `dialog_prob` (default **0.3**) if Dialog levels exist for that image
  - otherwise → `"template"`.
- Held-out attribute pairs: 3 pairs, each with **≥ 200 test images**. They must not include any pair from `DEMO_PAIRS`. They are removed from the **train** split of every generator dataset (not from the classifier's data).
- Seeds fixed through `utils/seed.py:set_seed(seed)`. All hyperparameters live in `configs/*.yaml`.
- Checkpoints: one `.pt` with keys `model`, `ema` (or `None`), `optimizer`, `step`, plus a copy of the config YAML next to it. Save every `ckpt_every` steps (default **2000**) and **auto-resume** from `<out_dir>/last.pt` if it exists.
- Never commit data, caches, checkpoints or outputs (`.gitignore` already covers them).

## Review Focus
1. **Odd JPEGs.** Grayscale or CMYK CelebA files must be converted to RGB, never crash the cache build. → test in Task 2.
2. **An image whose caption-eligible attributes are all 0** must still get a non-empty, grammatical caption (e.g. "a photo of a woman"). → test in Task 3.
3. **Face2Text descriptions longer than 77 tokens** (max is 75 words) must be truncated, not error. → test in Task 4.
4. **A Kaggle session dying mid-run.** Restarting the same command must resume from `last.pt` at the saved step. → test in Task 6.
5. **The CelebA-Dialog label file is missing.** Google Drive currently rate-limits it. The whole pipeline must run without it, and the "dialog" source is simply never sampled. → test in Task 4.

---

## File Structure

| File | Responsibility |
|---|---|
| `utils/seed.py` | `set_seed` |
| `utils/config.py` | `load_config(path) -> dict` (YAML) |
| `data/celeba.py` | Constants (`ATTR_NAMES`, `SPLITS`) and `load_celeba_metadata` |
| `data/face2text.py` | `load_face2text` |
| `data/dialog.py` | `load_dialog_levels` (tolerant parser) and `dialog_caption` |
| `data/captions.py` | Template caption engine |
| `data/splits.py` | Held-out pair selection and masking |
| `data/preprocess.py` | Crop/resize one image; build the uint8 cache |
| `data/tokenizer.py` | CLIP tokenizer wrapper |
| `data/dataset.py` | `FaceCraftDataset` |
| `models/classifier.py` | ResNet-style 40-attribute classifier |
| `models/text_encoder.py` | From-scratch Transformer encoder |
| `models/gan.py` | Conditional generator (CBN) + discriminator (SN + projection) |
| `training/common.py` | Checkpoint save/resume, AMP helpers |
| `training/train_classifier.py` | Classifier training and evaluation |
| `training/train_gan.py` | GAN training, sampling, sample grids |
| `evaluation/fid_kid.py` | FID/KID on N samples vs. the real test cache |
| `scripts/download_data.py` | Face2Text from Zenodo, Dialog via gdown with fallback, CelebA instructions |
| `scripts/build_notebook.py` | Generates the D2 notebook with nbformat |
| `configs/{data,classifier,gan}.yaml` | Hyperparameters |
| `tests/` | pytest, CPU-only, synthetic data |

---

### Task 1: Project utilities + metadata loaders (CelebA, Face2Text, Dialog)

**Files:**
- Create: `utils/__init__.py`, `utils/seed.py`, `utils/config.py`, `data/celeba.py`, `data/face2text.py`, `data/dialog.py`, `configs/data.yaml`, `tests/conftest.py`, `tests/test_metadata.py`

**Interfaces:**
- Produces:
  - `ATTR_NAMES: list[str]` (40, CSV order)
  - `load_celeba_metadata(root: str) -> pd.DataFrame`. It reads `list_attr_celeba.csv` and `list_eval_partition.csv` (Kaggle layout). The index is `image_id` (e.g. `"000001.jpg"`). It has 40 attribute columns as int 0/1 (CSV −1 → 0) and a `split` column in {"train", "val", "test"}.
  - `load_face2text(paths: list[str]) -> dict[str, list[str]]`: filename → stripped descriptions; empty descriptions are dropped.
  - `load_dialog_levels(path: str | None) -> pd.DataFrame | None`: index `image_id`, columns `["Bangs", "Eyeglasses", "Beard", "Smiling", "Age"]` (ints 0–5). Returns `None` if the path is missing.
  - `dialog_caption(levels: dict[str, int], male: bool, rng: random.Random) -> str`
  - `set_seed(seed: int) -> None`
  - `load_config(path: str) -> dict`
- `configs/data.yaml` keys: `celeba_root`, `face2text_dir`, `dialog_labels` (nullable), `cache_path`, `image_size: 64`, `crop: 140`, `f2t_prob: 0.5`, `dialog_prob: 0.3`, `heldout_pairs: []` (filled in Task 3), `seed: 0`.

- [ ] **Step 1: Write failing tests.** `tests/conftest.py` builds a tiny fake CelebA root in `tmp_path`: 6 images listed in both CSVs (Kaggle headers: `image_id,5_o_Clock_Shadow,...` and `image_id,partition`), plus a 2-file fake Face2Text set.

```python
def test_celeba_metadata_shapes(fake_celeba):
    df = load_celeba_metadata(fake_celeba)
    assert list(df.columns[:40]) == ATTR_NAMES and len(ATTR_NAMES) == 40
    assert set(df["split"]) <= {"train", "val", "test"}
    assert df[ATTR_NAMES].isin([0, 1]).all().all()

def test_face2text_merges_files_and_drops_empty(fake_f2t):
    caps = load_face2text(fake_f2t)
    assert caps["000001.jpg"] == ["A man with short hair.", "He is smiling."]
    assert "000002.jpg" not in caps          # its only description was "   "

def test_dialog_parser_tolerates_header_lines(tmp_path):
    p = tmp_path / "fg.txt"
    p.write_text("202599\nBangs Eyeglasses No_Beard Smiling Young\n000001.jpg 0 0 0 2 1\n000002.jpg 5 3 0 0 4\n")
    df = load_dialog_levels(str(p))
    assert df.loc["000002.jpg"].tolist() == [5, 3, 0, 0, 4]

def test_dialog_missing_file_returns_none():
    assert load_dialog_levels("does/not/exist.txt") is None

def test_dialog_caption_mentions_degree():
    s = dialog_caption({"Bangs": 0, "Eyeglasses": 0, "Beard": 0, "Smiling": 5, "Age": 1}, male=False, rng=random.Random(0))
    assert "laugh" in s.lower() and s.strip()
```

- [ ] **Step 2: Run them to verify they fail.** Run `pytest tests/test_metadata.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement the modules.**
  - Dialog parser: keep only lines whose first token ends in `.jpg` and is followed by ≥ 5 integers; take the first 5. The column meaning follows Talk-to-Edit `configs/attributes_5.json` (Bangs, Eyeglasses, No_Beard→"Beard" amount, Smiling, Young→"Age").
  - `dialog_caption` uses our own short degree phrases per level 0–5, e.g. Smiling: no smile / faint smile / slight smile / smile / big smile / laughing; Eyeglasses: none / rimless glasses / thin-frame glasses / thick-frame glasses / thin sunglasses / thick sunglasses. Mention only non-zero Bangs, Eyeglasses and Beard; always mention Smiling and Age. The level→word direction for Beard and Age is **verified in the notebook** (Task 7) against CelebA `No_Beard`/`Young`.
- [ ] **Step 4: Run the tests.** Run `pytest tests/test_metadata.py -v`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add utils data configs tests && git commit -m "feat(data): metadata loaders for CelebA, Face2Text, CelebA-Dialog"`

### Task 2: Image preprocessing + cache

**Files:**
- Create: `data/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Produces:
  - `preprocess_image(path: str, crop: int = 140, size: int = 64) -> np.ndarray` returns uint8 `[64, 64, 3]`
  - `build_cache(image_dir: str, image_ids: list[str], out_path: str, workers: int = 4) -> None` writes a `.npy` uint8 `[N, 64, 64, 3]` in `image_ids` order, plus `<out_path>.ids.txt` with one id per line. It is resumable: if `out_path` exists with matching ids, it skips the work.
  - `load_cache(out_path: str) -> tuple[np.ndarray (mmap), list[str]]`

- [ ] **Step 1: Write failing tests.**

```python
def test_preprocess_center_crop_and_size(tmp_path):
    img = Image.new("RGB", (178, 218), (0, 0, 0)); img.paste((255, 0, 0), (19, 39, 159, 179))   # 140x140 red center
    p = tmp_path / "a.jpg"; img.save(p, quality=100)
    out = preprocess_image(str(p))
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert out[..., 0].mean() > 240 and out[..., 1].mean() < 15          # crop is exactly the red square

def test_preprocess_converts_grayscale_and_cmyk(tmp_path):
    for mode in ["L", "CMYK"]:
        p = tmp_path / f"{mode}.jpg"; Image.new(mode, (178, 218)).save(p)
        assert preprocess_image(str(p)).shape == (64, 64, 3)

def test_build_and_load_cache_roundtrip(tmp_path, fake_celeba_images):
    ids = ["000001.jpg", "000002.jpg"]
    build_cache(fake_celeba_images, ids, str(tmp_path / "c.npy"), workers=1)
    arr, got_ids = load_cache(str(tmp_path / "c.npy"))
    assert arr.shape == (2, 64, 64, 3) and got_ids == ids
```

- [ ] **Step 2: Run them to verify they fail.** Run `pytest tests/test_preprocess.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Use `Image.open(...).convert("RGB")`, a centered 140×140 box, `resize((64, 64), Image.BICUBIC)`, and a `multiprocessing.Pool` for the cache.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(data): 64x64 preprocessing and uint8 cache"`

### Task 3: Template caption engine + held-out pairs

**Files:**
- Create: `data/captions.py`, `data/splits.py`, `tests/test_captions.py`

**Interfaces:**
- Produces:
  - `CAPTION_ATTRS: list[str]`: every attribute in `ATTR_NAMES` except `Attractive`, `Blurry`, `Wearing_Necktie`, `Wearing_Necklace`
  - `template_caption(attrs: dict[str, int], rng: random.Random) -> str`: uses at least 15 sentence patterns, a synonym table (e.g. Eyeglasses → glasses / eyeglasses / spectacles), random order of attribute phrases, and the subject from `Male`/`Young` ("a young woman", "an older man", …)
  - `DEMO_PAIRS: list[tuple[str, str]] = [("Blond_Hair", "Eyeglasses"), ("Smiling", "Young"), ("Smiling", "Eyeglasses"), ("Black_Hair", "Smiling"), ("Male", "Eyeglasses")]`
  - `choose_heldout_pairs(df: pd.DataFrame, k: int = 3, min_test: int = 200, exclude: list = DEMO_PAIRS, seed: int = 0) -> list[tuple[str, str]]`: candidate pairs come from `CAPTION_ATTRS` minus `Male`/`Young`; both attributes = 1; the pair has ≥ `min_test` test images and ≤ 2 % of train images, so removing it loses little data. Pick `k` deterministically from the qualifying pairs with `seed`.
  - `heldout_mask(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.Series[bool]`: True where any pair is fully present.

- [ ] **Step 1: Write failing tests.**

```python
def test_caption_respects_exclusions_and_rules():
    rng = random.Random(0)
    a = {n: 0 for n in ATTR_NAMES} | {"Male": 0, "Attractive": 1, "Blurry": 1, "Bald": 1, "Blond_Hair": 1, "Bangs": 1, "Goatee": 1}
    for _ in range(50):
        s = template_caption(a, rng).lower()
        assert "attractive" not in s and "blurry" not in s
        assert "blond" not in s and "bangs" not in s       # bald suppresses hair
        assert "goatee" not in s                           # facial hair only if Male=1

def test_two_hair_colors_mentions_none():
    a = {n: 0 for n in ATTR_NAMES} | {"Black_Hair": 1, "Blond_Hair": 1}
    s = template_caption(a, random.Random(1)).lower()
    assert "black" not in s and "blond" not in s

def test_all_zero_attrs_gives_nonempty_caption():
    s = template_caption({n: 0 for n in ATTR_NAMES}, random.Random(0))
    assert len(s.split()) >= 3 and "woman" in s.lower()

def test_captions_vary():
    a = {n: 0 for n in ATTR_NAMES} | {"Male": 1, "Smiling": 1, "Eyeglasses": 1, "Black_Hair": 1}
    assert len({template_caption(a, random.Random(i)) for i in range(30)}) >= 10

def test_heldout_pairs_exclude_demo_pairs(synthetic_attr_df):     # conftest: 5k random rows with a test split
    pairs = choose_heldout_pairs(synthetic_attr_df, k=3, min_test=20)
    assert len(pairs) == 3 and not set(pairs) & set(DEMO_PAIRS)
    m = heldout_mask(synthetic_attr_df, pairs)
    for a, b in pairs: assert m[(synthetic_attr_df[a] == 1) & (synthetic_attr_df[b] == 1)].all()
```

- [ ] **Step 2: Run them to verify they fail.** Run `pytest tests/test_captions.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `data/captions.py` and `data/splits.py`.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(data): template caption engine and held-out attribute pairs"`

### Task 4: Tokenizer + FaceCraftDataset

**Files:**
- Create: `data/tokenizer.py`, `data/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: the Task 1–3 functions.
- Produces:
  - `Tokenizer` with `__call__(texts: list[str]) -> torch.LongTensor [B, 77]` and attributes `pad_id: int`, `vocab_size: int`
  - `FaceCraftDataset(meta: pd.DataFrame, cache: np.ndarray, cache_ids: list[str], split: str, tokenizer: Tokenizer, f2t: dict[str, list[str]], dialog: pd.DataFrame | None, heldout_pairs: list[tuple[str, str]], f2t_prob: float = 0.5, dialog_prob: float = 0.3, train_aug: bool = True, drop_heldout: bool = True, seed: int = 0)`. `__getitem__` returns the Global Constraints tuple. The random flip applies only when `split == "train" and train_aug`. Held-out rows are dropped only when `split == "train" and drop_heldout`.
  - `caption_for(image_id: str, rng) -> tuple[str, str]` (caption, source). It is public so the notebook can show examples.

- [ ] **Step 1: Write failing tests** (the tokenizer test downloads the CLIP tokenizer once; mark it `@pytest.mark.network`).

```python
def test_item_matches_interface(tiny_ds):
    img, tok, attrs, src = tiny_ds[0]
    assert img.shape == (3, 64, 64) and img.dtype == torch.float32 and -1 <= img.min() and img.max() <= 1
    assert tok.shape == (77,) and tok.dtype == torch.int64
    assert attrs.shape == (40,) and set(attrs.unique().tolist()) <= {0.0, 1.0}
    assert src in {"template", "dialog", "face2text"}

@pytest.mark.network
def test_long_caption_truncates():
    t = Tokenizer(); assert t(["word " * 200]).shape == (1, 77)

def test_no_dialog_never_samples_dialog(tiny_ds_no_dialog):
    assert all(tiny_ds_no_dialog.caption_for(i, random.Random(k))[1] != "dialog" for k in range(50) for i in tiny_ds_no_dialog.ids)

def test_f2t_prob_one_always_uses_human_caption(tiny_ds_f2t_prob_1):
    ds = tiny_ds_f2t_prob_1; i = next(i for i in ds.ids if i in ds.f2t)
    assert {ds.caption_for(i, random.Random(k))[1] for k in range(20)} == {"face2text"}

def test_heldout_dropped_from_train_only(tiny_meta_with_pair):
    tr = make_ds(tiny_meta_with_pair, "train"); te = make_ds(tiny_meta_with_pair, "test")
    assert "pair_img.jpg" not in tr.ids and "pair_img_test.jpg" in te.ids
```

  The fixtures in `conftest.py` use a fake tokenizer (hash words to ids, pad to 77) so the non-network tests run offline.
- [ ] **Step 2: Run them to verify they fail.** Run `pytest tests/test_dataset.py -v -m "not network"`. Expected: FAIL.
- [ ] **Step 3: Implement.** The tokenizer wraps `CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")`. The dataset converts with `torch.from_numpy(cache[idx]).permute(2, 0, 1).float() / 127.5 - 1`.
- [ ] **Step 4: Run the tests.** Run `pytest tests -v`. Expected: PASS, network test included when online.
- [ ] **Step 5: Commit.** `git commit -m "feat(data): CLIP tokenizer wrapper and FaceCraftDataset"`

### Task 5: Attribute classifier (model + training script)

**Files:**
- Create: `models/classifier.py`, `training/common.py`, `training/train_classifier.py`, `configs/classifier.yaml`, `tests/test_classifier.py`

**Interfaces:**
- Produces:
  - `AttributeClassifier(num_attrs: int = 40, width: int = 64)`: a ResNet-18-style network (4 stages, 2 BasicBlocks each, widths 64/128/256/512, stem conv3×3 stride 1 for 64×64 inputs). `forward(x: [B,3,64,64]) -> logits [B,40]`.
  - `training/common.py`:
    - `save_ckpt(path, model, ema, optimizer, step, cfg)` writes `.pt` + `.yaml`
    - `load_latest(out_dir, model, ema, optimizer) -> int` returns the step, 0 if none
    - `amp_context(device)`
  - CLI: `python -m training.train_classifier --config configs/classifier.yaml`. It trains with BCEWithLogits, evaluates on test, and writes `outputs/classifier/metrics.json` = `{"mean_acc": float, "per_attr_acc": {name: float}, "majority_baseline_mean_acc": float}` and `outputs/classifier/best.pt`.
- `configs/classifier.yaml`: `epochs: 6`, `batch_size: 256`, `lr: 1e-3`, `weight_decay: 1e-4`, AdamW, cosine schedule, `amp: true`, `ckpt_every: 2000`, `out_dir: outputs/classifier`, `seed: 0`. The data comes from `FaceCraftDataset` (captions unused, `drop_heldout=False`).

- [ ] **Step 1: Write failing tests.**

```python
def test_classifier_output_shape():
    assert AttributeClassifier()(torch.randn(2, 3, 64, 64)).shape == (2, 40)

def test_classifier_overfits_tiny_batch():
    torch.manual_seed(0); m = AttributeClassifier(width=16); x = torch.randn(8, 3, 64, 64); y = (torch.rand(8, 40) > .5).float()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(150): opt.zero_grad(); l = F.binary_cross_entropy_with_logits(m(x), y); l.backward(); opt.step()
    assert ((m(x) > 0).float() == y).float().mean() > 0.95

def test_ckpt_roundtrip(tmp_path):
    m = AttributeClassifier(width=8); opt = torch.optim.Adam(m.parameters())
    save_ckpt(tmp_path / "last.pt", m, None, opt, 1234, {"a": 1})
    assert load_latest(tmp_path, AttributeClassifier(width=8), None, torch.optim.Adam(m.parameters())) == 1234
```

- [ ] **Step 2: Run them to verify they fail.** Run `pytest tests/test_classifier.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the model, `common.py` and the training script. The script also supports `--max_steps` and `--subset N` for a quick local smoke run.
- [ ] **Step 4: Run tests + smoke run.**
  - Run `pytest tests/test_classifier.py -v`. Expected: PASS.
  - If a local cache exists: `python -m training.train_classifier --config configs/classifier.yaml --subset 512 --max_steps 20`. Expected: `metrics.json` written.
- [ ] **Step 5: Commit.** `git commit -m "feat(classifier): ResNet attribute classifier + training script"`

### Task 6: Text encoder + conditional GAN + FID (model + training script)

**Files:**
- Create: `models/text_encoder.py`, `models/gan.py`, `training/train_gan.py`, `evaluation/fid_kid.py`, `configs/gan.yaml`, `tests/test_gan.py`

**Interfaces:**
- Consumes: `Tokenizer`, `FaceCraftDataset`, `save_ckpt`, `load_latest`.
- Produces:
  - `TextEncoder(vocab_size: int, dim: int = 512, layers: int = 6, heads: int = 8, max_len: int = 77, pad_id: int)`. `forward(token_ids: [B,77]) -> (tokens: [B,77,dim], pooled: [B,dim])`. Learned positional embeddings, pre-LN `nn.TransformerEncoder`, a padding mask, and `pooled` = the masked mean of the tokens. For the GAN, `configs/gan.yaml` uses `dim: 256, layers: 4`.
  - `Generator(z_dim: int = 128, cond_dim: int = 256, ch: int = 64)`:
    - `forward(z, cond) -> images [B,3,64,64]` in [−1,1] (tanh)
    - structure: linear → 4×4 → 4 upsample blocks, each `Upsample + conv3×3 + ConditionalBatchNorm(cond) + ReLU`.
  - `Discriminator(cond_dim: int = 256, ch: int = 64)`:
    - `forward(images, cond) -> score [B]`
    - structure: spectral-normed downsample conv blocks → global sum pool `h` → `linear(h) + (embed(cond) · h)` (projection).
  - `GANSampler` implementing the shared interface: `sample(prompts: list[str], n: int, seed: int, guidance=None, steps=None) -> np.uint8 [N,64,64,3]`.
  - CLI `python -m training.train_gan --config configs/gan.yaml [--max_steps N]`:
    - hinge loss, Adam(2e-4, betas (0.0, 0.9)), 1 D step per G step
    - EMA of G (0.999)
    - checkpoints every `ckpt_every`, with auto-resume
    - every `sample_every` (2000) steps: a fixed-prompt 8×8 grid → `outputs/gan/samples/step_XXXXXX.png`
    - at the end: `outputs/gan/metrics.json` = `{"step": int, "fid": float, "kid_mean": float, "kid_std": float, "n_samples": int}` and `outputs/gan/losses.csv`.
  - `compute_fid_kid(fake_uint8: np.ndarray, real_uint8: np.ndarray, device) -> dict`: torchmetrics `FrechetInceptionDistance(feature=2048, normalize=False)` and `KernelInceptionDistance(subset_size=1000)`, batched. The real images are the test split cache. D2 uses `n_samples: 5000` (preliminary); the final evaluation will use 10k.
  - The text encoder is trained jointly with G and D. Text features are detached for D's update, so only G's loss trains the encoder.
- `configs/gan.yaml`: `steps: 20000`, `batch_size: 128`, `z_dim: 128`, `ch: 64`, `text_dim: 256`, `text_layers: 4`, `lr: 2e-4`, `ckpt_every: 2000`, `sample_every: 2000`, `fid_samples: 5000`, `out_dir: outputs/gan`, `seed: 0`, `amp: false` (keep GAN in fp32 for stability).

- [ ] **Step 1: Write failing tests.**

```python
def test_text_encoder_shapes_and_padding_invariance():
    enc = TextEncoder(vocab_size=1000, dim=32, layers=2, heads=4, pad_id=0).eval()
    a = torch.tensor([[5, 6, 7] + [0] * 74]); tok, pooled = enc(a)
    assert tok.shape == (1, 77, 32) and pooled.shape == (1, 32)
    b = a.clone(); b[0, 50] = 0                      # changing padding does nothing
    assert torch.allclose(enc(b)[1], pooled, atol=1e-5)

def test_gan_shapes():
    G, D = Generator(z_dim=16, cond_dim=32, ch=8), Discriminator(cond_dim=32, ch=8)
    x = G(torch.randn(4, 16), torch.randn(4, 32)); assert x.shape == (4, 3, 64, 64) and x.abs().max() <= 1
    assert D(x, torch.randn(4, 32)).shape == (4,)

def test_gan_resumes_from_checkpoint(tmp_path, tiny_gan_cfg):     # conftest: tiny cfg + tiny dataset
    run_gan(tiny_gan_cfg | {"out_dir": str(tmp_path), "steps": 4, "ckpt_every": 2})
    assert run_gan(tiny_gan_cfg | {"out_dir": str(tmp_path), "steps": 6, "ckpt_every": 2}) == 4  # returns start step

def test_fid_identical_sets_near_zero():
    x = np.random.RandomState(0).randint(0, 255, (64, 64, 64, 3), dtype=np.uint8)
    assert compute_fid_kid(x, x, "cpu")["fid"] < 1.0
```

  `train_gan.py` exposes `run_gan(cfg: dict) -> int`, which returns the step it started from. The CLI wraps it.
- [ ] **Step 2: Run them to verify they fail.** Run `pytest tests/test_gan.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the four files.
- [ ] **Step 4: Run tests.** Run `pytest tests -v`. Expected: all PASS. (The FID test downloads the Inception weights once.)
- [ ] **Step 5: Commit.** `git commit -m "feat(gan): Transformer text encoder, conditional SN-GAN baseline, FID/KID"`

### Task 7: Download script + Deliverable 2 notebook

**Files:**
- Create: `scripts/download_data.py`, `scripts/build_notebook.py`, `notebooks/01_data_and_preprocessing.ipynb` (generated), `docs/KAGGLE.md`

**Interfaces:**
- Consumes: everything above.
- `scripts/download_data.py --out datasets/`:
  - downloads and unzips Face2Text v2.1 (`https://zenodo.org/api/records/10973388/files/face2text_v2.1.zip/content`)
  - tries `gdown` for the Dialog fine-grained label file (id `1wZcVEjJ5LwP1Ciuc3j_RFw9Vcusj4UEU`), and on failure prints the browser link and continues
  - prints the instructions for attaching the Kaggle CelebA dataset (`jessicali9530/celeba-dataset`, images under `img_align_celeba/img_align_celeba/`).
- `docs/KAGGLE.md`: the exact steps to run on Kaggle:
  1. Attach the CelebA dataset.
  2. Upload the repo, or `git clone`.
  3. `pip install -r requirements.txt`.
  4. Run the download script.
  5. Build the cache.
  6. Run the classifier and the GAN as background "Save & Run All" sessions.
  7. Download `outputs/`.
- The notebook is built by `build_notebook.py` with `nbformat`, so it can be regenerated. Sections, each with markdown that explains *what and why* in plain words:
  1. **Overview.** Goal, the three text sources, and how the data feeds the models.
  2. **Acquisition.** Paths, the download script, counts: 202,599 images, 40 attributes, the split sizes, Face2Text 18,690 / 12,227, and Dialog coverage (or a "not available" note).
  3. **Cleaning.**
     - Check that every listed image exists and opens (count failures).
     - Join Face2Text to CelebA (count unmatched filenames).
     - Count empty or duplicate descriptions.
     - Count the contradictory attribute combinations found in the raw labels (Bald+Bangs, 2+ hair colors, female + facial hair).
  4. **Exploration.**
     - Bar chart of attribute frequencies.
     - Co-occurrence heatmap for the caption attributes.
     - Gender and age skew.
     - Face2Text description-length histogram and top words.
     - Dialog level distributions, plus the **check of level direction vs. the CelebA `No_Beard`/`Young` binary labels**.
  5. **Labeling: captions.** For 6 sample faces, show the image next to its template, Dialog and Face2Text captions. Also explain the contradiction rules and the sampling probabilities.
  6. **Preprocessing.**
     - Before/after crop visualization.
     - Cache build (or load) and its size.
     - Split table.
     - **Held-out pairs table**: chosen pairs with train/test counts, which are then written back into `configs/data.yaml`.
  7. **Tokenization.** Token-length histogram per source and the truncation rate at 77.
  8. **DataLoader.** One batch shown as a grid with captions, plus a check of tensor shapes and ranges.
  9. **First results.** A `RUN_TRAINING = False` flag:
     - if True, run the two training scripts
     - otherwise load `outputs/`.

     Show classifier mean and per-attribute accuracy vs. the majority baseline, the GAN loss curves, sample grids at several steps, and FID/KID.
  10. **Next steps.** Diffusion model and ablations.

- [ ] **Step 1: Write the download script and `docs/KAGGLE.md`.** Run `python scripts/download_data.py --out datasets/`. Expected: Face2Text unzipped; a Dialog success or fallback message.
- [ ] **Step 2: Write `build_notebook.py` and generate the notebook.** Run `python scripts/build_notebook.py`. Expected: the `.ipynb` is written and `python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/01_data_and_preprocessing.ipynb', 4))"` passes.
- [ ] **Step 3: Smoke-execute the notebook against the fake data.** Run `jupyter nbconvert --to notebook --execute` with env `FACECRAFT_FAKE_DATA=1`, which makes section 2 build the conftest-style fake dataset. Expected: no cell errors.
- [ ] **Step 4: Commit.** `git commit -m "feat(notebook): Deliverable 2 data & preprocessing notebook + Kaggle guide"`

### Task 8: Real runs on Kaggle and submission (team, not agent)

- [ ] **Step 1:** Push the repo to GitHub, or zip it and upload it to Kaggle. Run `docs/KAGGLE.md` steps 1–5. Expected: the cache is about 2.5 GB with 202,599 ids.
- [ ] **Step 2:** Classifier run. Expected: `mean_acc ≈ 0.90` on test, above the majority baseline (≈ 0.80).
- [ ] **Step 3:** GAN run, 20k steps (about 3–5 h on a T4). Expected: recognizable faces in the late grids; record FID/KID.
- [ ] **Step 4:** Set `RUN_TRAINING = False`, place the downloaded `outputs/` next to the notebook, run all cells on Kaggle, and save the executed notebook.
- [ ] **Step 5:** Read-through by all four members. Submit the executed `.ipynb` (and an HTML export) on Moodle **before Sun 22 Nov 23:59**.
