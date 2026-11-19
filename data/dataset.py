"""FaceCraftDataset: the one PyTorch Dataset every model in the project trains on.

Each item is (image, token_ids, attrs, caption_source):
  image          float32 [3, 64, 64] in [-1, 1]   (from the uint8 cache)
  token_ids      int64   [77]                     (caption -> CLIP BPE ids)
  attrs          float32 [40] of 0/1              (CelebA labels, ATTR_NAMES order; used for evaluation)
  caption_source "template" | "dialog" | "face2text"

A NEW caption is drawn every time an image is loaded, so across epochs the model sees the same face
described in many ways. How the caption is chosen:
  1. the image has human Face2Text captions  -> use one with probability f2t_prob
  2. else, the image has Dialog levels       -> degree-aware Dialog caption with probability dialog_prob
  3. else                                    -> random template caption from the 40 labels
"""
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.captions import template_caption
from data.celeba import ATTR_NAMES
from data.dialog import dialog_caption
from data.splits import heldout_mask


class FaceCraftDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, cache: np.ndarray, cache_ids: list, split: str, tokenizer,
                 f2t: dict, dialog, heldout_pairs: list, f2t_prob: float = 0.5, dialog_prob: float = 0.3,
                 train_aug: bool = True, drop_heldout: bool = True, seed: int = 0):
        rows = meta[meta["split"] == split]
        if split == "train" and drop_heldout and heldout_pairs:
            rows = rows[~heldout_mask(rows, [tuple(p) for p in heldout_pairs])]
        position = {image_id: k for k, image_id in enumerate(cache_ids)}
        rows = rows[rows.index.isin(position)]                 # only images present in the cache
        self.ids = list(rows.index)
        self.cache_pos = np.array([position[i] for i in self.ids], dtype=np.int64)
        self.attrs = rows[ATTR_NAMES].to_numpy(dtype=np.float32)
        self.male = dict(zip(self.ids, rows["Male"].to_numpy() == 1))
        self.label_dicts = rows[ATTR_NAMES].to_dict("index")   # image_id -> {attr: 0/1}
        self.cache = cache
        self.tokenizer = tokenizer
        self.f2t = {i: f2t[i] for i in self.ids if i in f2t}
        self.dialog = None if dialog is None else dialog[dialog.index.isin(self.ids)].to_dict("index")
        self.f2t_prob, self.dialog_prob = f2t_prob, dialog_prob
        self.flip = split == "train" and train_aug
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def caption_for(self, image_id: str, rng: random.Random):
        """Return (caption, source) for one image, following the sampling rule in the module docstring."""
        if image_id in self.f2t and rng.random() < self.f2t_prob:
            return rng.choice(self.f2t[image_id]), "face2text"
        if self.dialog is not None and image_id in self.dialog and rng.random() < self.dialog_prob:
            return dialog_caption(self.dialog[image_id], self.male[image_id], rng), "dialog"
        return template_caption(self.label_dicts[image_id], rng), "template"

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        pixels = np.array(self.cache[self.cache_pos[idx]])              # copy out of the memmap
        image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
        if self.flip and self.rng.random() < 0.5:
            image = image.flip(-1)                                      # horizontal flip
        caption, source = self.caption_for(image_id, self.rng)
        token_ids = self.tokenizer([caption])[0]
        return image, token_ids, torch.from_numpy(self.attrs[idx]), source

    # --- pickling (DataLoader workers on Windows) -------------------------------------------------
    # A memory-mapped cache must travel as its file path, not as 2.5 GB of pixels.
    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.cache, np.memmap) and self.cache.filename:
            state["cache"] = None
            state["_cache_file"] = self.cache.filename
        return state

    def __setstate__(self, state):
        cache_file = state.pop("_cache_file", None)
        self.__dict__.update(state)
        if cache_file is not None:
            self.cache = np.load(cache_file, mmap_mode="r")
