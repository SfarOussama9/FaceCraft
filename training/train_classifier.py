"""Train the 40-attribute classifier on real CelebA and report its test accuracy.

    python -m training.train_classifier --config configs/classifier.yaml
    python -m training.train_classifier --config configs/classifier.yaml --subset 512 --max_steps 20   # smoke test

Loss: binary cross-entropy for each of the 40 attributes (multi-label classification).
Outputs (in out_dir):
  last.pt / last.yaml   checkpoint for auto-resume
  best.pt               final weights (used later as the judge for generated images)
  metrics.json          {"mean_acc", "per_attr_acc", "majority_baseline_mean_acc", "history", ...}
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.celeba import ATTR_NAMES
from data.tokenizer import make_tokenizer
from models.classifier import AttributeClassifier
from training.common import amp_context, build_dataset, load_latest, save_ckpt
from utils.config import load_config
from utils.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> np.ndarray:
    """Return per-attribute accuracy [40] over the whole loader."""
    model.eval()
    correct, total = torch.zeros(40, device=device), 0
    for images, _, attrs, _ in loader:
        images, attrs = images.to(device, non_blocking=True), attrs.to(device)
        with amp_context(device, amp):
            logits = model(images)
        correct += ((logits.float() > 0).float() == attrs).sum(0)
        total += len(attrs)
    model.train()
    return (correct / total).cpu().numpy()


def main(cfg: dict, max_steps=None, subset=None) -> dict:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = load_config(cfg["data_config"])
    tok = make_tokenizer(cfg.get("tokenizer", "hash"))
    # The judge sees ALL real images (no held-out removal): it must recognise every attribute.
    train_ds = build_dataset(data_cfg, "train", tok, drop_heldout=False)
    test_ds = build_dataset(data_cfg, "test", tok)
    val_ds = build_dataset(data_cfg, "val", tok)
    if subset:
        train_ds, test_ds, val_ds = (Subset(d, range(min(subset, len(d)))) for d in (train_ds, test_ds, val_ds))
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=device == "cuda",
              persistent_workers=cfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl, test_dl = DataLoader(val_ds, **kw), DataLoader(test_ds, **kw)

    model = AttributeClassifier(width=cfg["width"]).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_dl)
    total_steps = max_steps or cfg["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device == "cuda")
    out = Path(cfg["out_dir"])
    step = load_latest(out, model, None, opt)
    for _ in range(step):
        sched.step()
    if step:
        print(f"resumed from step {step}")

    history, t0 = [], time.time()
    model.train()
    while step < total_steps:
        running = 0.0
        for images, _, attrs, _ in train_dl:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            attrs = attrs.to(device, non_blocking=True)
            with amp_context(device, cfg["amp"]):
                loss = F.binary_cross_entropy_with_logits(model(images).float(), attrs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            if step % 100 == 0:
                print(f"step {step}/{total_steps}  loss {running / 100:.4f}  {time.time() - t0:.0f}s", flush=True)
                running = 0.0
            if step % cfg["ckpt_every"] == 0:
                save_ckpt(out / "last.pt", model, None, opt, step, cfg)
            if step >= total_steps:
                break
        val_acc = evaluate(model, val_dl, device, cfg["amp"])
        history.append({"step": step, "val_mean_acc": float(val_acc.mean())})
        print(f"[val] step {step}  mean acc {val_acc.mean():.4f}", flush=True)
        save_ckpt(out / "last.pt", model, None, opt, step, cfg)

    test_acc = evaluate(model, test_dl, device, cfg["amp"])
    # Majority baseline: always predict the most common value of each attribute in the training set.
    train_rate = np.asarray(train_ds.dataset.attrs if subset else train_ds.attrs).mean(0)
    test_attrs = np.asarray(test_ds.dataset.attrs[: len(test_ds)] if subset else test_ds.attrs)
    majority = ((test_attrs == (train_rate > 0.5)).mean(0))
    metrics = {"mean_acc": float(test_acc.mean()),
               "per_attr_acc": {n: float(a) for n, a in zip(ATTR_NAMES, test_acc)},
               "majority_baseline_mean_acc": float(majority.mean()),
               "majority_baseline_per_attr": {n: float(a) for n, a in zip(ATTR_NAMES, majority)},
               "steps": step, "history": history, "train_seconds": round(time.time() - t0)}
    torch.save({"model": model.state_dict(), "width": cfg["width"]}, out / "best.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[test] mean acc {metrics['mean_acc']:.4f}   majority baseline {metrics['majority_baseline_mean_acc']:.4f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    c = load_config(a.config)
    if a.out_dir:
        c["out_dir"] = a.out_dir
    main(c, a.max_steps, a.subset)
