"""Train the text-conditional GAN baseline (with its from-scratch Transformer text encoder).

    python -m training.train_gan --config configs/gan.yaml
    python -m training.train_gan --config configs/gan.yaml --max_steps 200      # quick check

One training step:
  1. encode the captions -> sentence vectors (text encoder)
  2. D step (hinge loss): real image + its caption -> "real"; generated image -> "fake";
                          real image + a WRONG caption -> "fake"   (matching-aware, Reed et al. 2016)
  3. G step: make D believe G's images match their captions. G's loss also trains the text encoder.
  4. update the EMA copies of G and the text encoder (smoother weights, used for sampling)

Outputs in out_dir: last.pt (auto-resume), samples/step_XXXXXX.png, losses.csv, metrics.json (FID/KID).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from data.tokenizer import make_tokenizer
from models.gan import Discriminator, Generator
from models.text_encoder import TextEncoder
from training.common import build_dataset, ema_update, load_latest, make_ema, save_ckpt
from utils.config import load_config
from utils.seed import set_seed


class GANSampler:
    """Shared generator interface: sample(prompts, n, seed) -> uint8 [len(prompts) * n, 64, 64, 3]."""

    def __init__(self, generator, text_encoder, tokenizer, z_dim: int, device):
        self.G, self.text, self.tok, self.z_dim, self.device = generator, text_encoder, tokenizer, z_dim, device

    @torch.no_grad()
    def sample(self, prompts: list, n: int = 1, seed: int = 0, guidance=None, steps=None, batch: int = 256):
        self.G.eval()
        self.text.eval()
        all_prompts = [p for p in prompts for _ in range(n)]          # n images per prompt, grouped
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = []
        for i in range(0, len(all_prompts), batch):
            chunk = all_prompts[i:i + batch]
            cond = self.text(self.tok(chunk).to(self.device))[1]
            z = torch.randn(len(chunk), self.z_dim, generator=gen).to(self.device)
            img = self.G(z, cond)
            out.append(((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)


def save_grid(images: np.ndarray, rows: int, path: Path) -> None:
    """images [rows*cols, 64, 64, 3] -> one PNG grid (one row per prompt)."""
    cols = len(images) // rows
    grid = images.reshape(rows, cols, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(rows * 64, cols * 64, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


def _infinite(loader):
    while True:
        yield from loader


def run_gan(cfg: dict) -> int:
    """Train (or resume) until cfg["steps"]; returns the step training started from."""
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = load_config(cfg["data_config"])
    tok = make_tokenizer(cfg.get("tokenizer", "clip"))
    train_ds = build_dataset(data_cfg, "train", tok)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=device == "cuda",
                        persistent_workers=cfg["num_workers"] > 0)

    text = TextEncoder(tok.vocab_size, cfg["text_dim"], cfg["text_layers"], cfg["text_heads"],
                       pad_id=tok.pad_id).to(device)
    G = Generator(cfg["z_dim"], cfg["text_dim"], cfg["ch"]).to(device)
    D = Discriminator(cfg["text_dim"], cfg["ch"]).to(device)
    nets = nn.ModuleDict({"G": G, "D": D, "text": text})
    ema = nn.ModuleDict({"G": make_ema(G), "text": make_ema(text)})
    opt = {"G": torch.optim.Adam(list(G.parameters()) + list(text.parameters()), cfg["lr"], betas=(0.0, 0.9)),
           "D": torch.optim.Adam(D.parameters(), cfg["lr"], betas=(0.0, 0.9))}

    out = Path(cfg["out_dir"])
    start = step = load_latest(out, nets, ema, opt)
    if start:
        print(f"resumed from step {start}")
    sampler = GANSampler(ema["G"], ema["text"], tok, cfg["z_dim"], device)
    prompts = cfg["sample_prompts"]
    out.mkdir(parents=True, exist_ok=True)
    new_log = start == 0 or not (out / "losses.csv").exists()
    log_f = open(out / "losses.csv", "w" if start == 0 else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss_D", "loss_G", "d_real", "d_fake", "d_wrong"])

    batches, t0 = _infinite(loader), time.time()
    while step < cfg["steps"]:
        # --- n_dis D steps per G step (SN-GAN recipe, Miyato et al. 2018). With only 1, G caught up
        #     with D within a few steps and D went blind to the images (see tests/test_gan.py). ---
        for _ in range(cfg.get("n_dis", 5)):
            images, token_ids, _, _ = next(batches)
            images, token_ids = images.to(device, non_blocking=True), token_ids.to(device, non_blocking=True)
            with torch.no_grad():                         # text vectors are inputs for D, not trained by D
                c = text(token_ids)[1]
                fake = G(torch.randn(len(images), cfg["z_dim"], device=device), c)
            d_real = D(images, c)
            d_fake = D(fake, c)
            d_wrong = D(images, c.roll(1, dims=0))        # real image, caption of another image
            loss_D = torch.relu(1 - d_real).mean() + 0.5 * (torch.relu(1 + d_fake).mean() + torch.relu(1 + d_wrong).mean())
            opt["D"].zero_grad(set_to_none=True)
            loss_D.backward()
            opt["D"].step()

        # --- G step (also updates the text encoder through `cond`) ---
        cond = text(token_ids)[1]
        z = torch.randn(len(images), cfg["z_dim"], device=device)
        loss_G = -D(G(z, cond), cond).mean()
        opt["G"].zero_grad(set_to_none=True)
        loss_G.backward()
        opt["G"].step()

        ema_update(ema["G"], G, cfg["ema_decay"])
        ema_update(ema["text"], text, cfg["ema_decay"])
        step += 1
        if step % 50 == 0 or step == cfg["steps"]:
            log.writerow([step, f"{loss_D.item():.4f}", f"{loss_G.item():.4f}", f"{d_real.mean().item():.3f}",
                          f"{d_fake.mean().item():.3f}", f"{d_wrong.mean().item():.3f}"])
            log_f.flush()
        if step % 200 == 0:
            print(f"step {step}/{cfg['steps']}  D {loss_D.item():.3f}  G {loss_G.item():.3f}  {time.time() - t0:.0f}s", flush=True)
        if step % cfg["sample_every"] == 0:
            save_grid(sampler.sample(prompts, n=8, seed=123), len(prompts), out / "samples" / f"step_{step:06d}.png")
        if step % cfg["ckpt_every"] == 0:
            save_ckpt(out / "last.pt", nets, ema, opt, step, cfg)
    save_ckpt(out / "last.pt", nets, ema, opt, step, cfg)
    log_f.close()

    metrics = {"step": step, "train_seconds_this_session": round(time.time() - t0)}
    if cfg.get("fid_samples", 0) > 0:
        metrics.update(evaluate_fid(cfg, data_cfg, sampler, tok, device))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics", metrics)
    return start


def evaluate_fid(cfg, data_cfg, sampler, tok, device) -> dict:
    """Generate fid_samples images from TEST captions and compare with the same number of real test images."""
    import random
    from evaluation.fid_kid import compute_fid_kid
    test_ds = build_dataset(data_cfg, "test", tok)
    n = min(cfg["fid_samples"], len(test_ds))
    rng = random.Random(0)
    ids = test_ds.ids[:n]
    prompts = [test_ds.caption_for(i, rng)[0] for i in ids]
    fake = sampler.sample(prompts, n=1, seed=0)
    real = np.stack([np.array(test_ds.cache[p]) for p in test_ds.cache_pos[:n]])
    return compute_fid_kid(fake, real, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gan.yaml")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fid_samples", type=int, default=None)
    a = ap.parse_args()
    c = load_config(a.config)
    if a.max_steps:
        c["steps"] = a.max_steps
    if a.out_dir:
        c["out_dir"] = a.out_dir
    if a.fid_samples is not None:
        c["fid_samples"] = a.fid_samples
    run_gan(c)
