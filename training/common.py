"""Shared training helpers: checkpoints with auto-resume, mixed precision, EMA, building datasets.

Checkpoint format (one for the whole project): <out_dir>/last.pt holds
  {"model": state_dict, "ema": state_dict or None, "optimizer": state_dict (or dict of them), "step": int}
and <out_dir>/last.yaml holds the config that produced it.
Kaggle/Colab sessions die after ~12 h, so every script saves often and resumes from last.pt.
"""
import contextlib
import copy
import glob
import os
from pathlib import Path

import torch
import yaml


def _opt_state(optimizer):
    if optimizer is None:
        return None
    if isinstance(optimizer, dict):
        return {k: o.state_dict() for k, o in optimizer.items()}
    return optimizer.state_dict()


def save_ckpt(path, model, ema, optimizer, step: int, cfg: dict) -> None:
    """Write the checkpoint atomically (a crash mid-save never corrupts the previous one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
             "optimizer": _opt_state(optimizer), "step": step}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    with open(path.with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_latest(out_dir, model, ema, optimizer) -> int:
    """If <out_dir>/last.pt exists, restore everything and return its step; otherwise return 0."""
    path = Path(out_dir) / "last.pt"
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    if optimizer is not None and state.get("optimizer") is not None:
        if isinstance(optimizer, dict):
            for k, o in optimizer.items():
                o.load_state_dict(state["optimizer"][k])
        else:
            optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def amp_context(device: str, enabled: bool = True):
    """float16 autocast on GPU (about 2x faster), plain float32 otherwise."""
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_ema(model: torch.nn.Module) -> torch.nn.Module:
    """A frozen copy of the model whose weights are a running average of the trained weights."""
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """ema = decay * ema + (1 - decay) * model   (buffers such as BatchNorm statistics are copied)."""
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.detach(), alpha=1 - decay)
    for e, m in zip(ema.buffers(), model.buffers()):
        e.copy_(m)


def build_dataset(data_cfg: dict, split: str, tokenizer, **overrides):
    """Construct FaceCraftDataset for `split` from a data config (configs/data.yaml)."""
    from data.celeba import load_celeba_metadata
    from data.dataset import FaceCraftDataset
    from data.dialog import load_dialog_levels
    from data.face2text import load_face2text
    from data.preprocess import load_cache

    meta = load_celeba_metadata(data_cfg["celeba_root"])
    cache, ids = load_cache(data_cfg["cache_path"])
    f2t_files = sorted(glob.glob(os.path.join(data_cfg["face2text_dir"], "clean_*.json")))
    args = dict(meta=meta, cache=cache, cache_ids=ids, split=split, tokenizer=tokenizer,
                f2t=load_face2text(f2t_files), dialog=load_dialog_levels(data_cfg.get("dialog_labels")),
                heldout_pairs=data_cfg.get("heldout_pairs", []), f2t_prob=data_cfg.get("f2t_prob", 0.5),
                dialog_prob=data_cfg.get("dialog_prob", 0.3), seed=data_cfg.get("seed", 0))
    args.update(overrides)
    return FaceCraftDataset(**args)
