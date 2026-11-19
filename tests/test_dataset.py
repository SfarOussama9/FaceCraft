"""Tests for the tokenizer wrapper and FaceCraftDataset (the shared data interface)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.celeba import ATTR_NAMES, load_celeba_metadata
from data.dataset import FaceCraftDataset
from data.face2text import load_face2text
from data.preprocess import build_cache, load_cache


class FakeTokenizer:
    """Offline stand-in for the CLIP tokenizer: hashes words to ids, pads to 77."""
    pad_id = 0
    vocab_size = 1000

    def __call__(self, texts):
        out = torch.zeros(len(texts), 77, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [1 + (hash(w) % 998) for w in t.split()][:77]
            out[i, :len(ids)] = torch.tensor(ids)
        return out


@pytest.fixture
def parts(fake_celeba, fake_f2t, tmp_path):
    meta = load_celeba_metadata(fake_celeba)
    img_dir = str(Path(fake_celeba) / "img_align_celeba" / "img_align_celeba")
    build_cache(img_dir, list(meta.index), str(tmp_path / "cache.npy"), workers=1)
    cache, ids = load_cache(str(tmp_path / "cache.npy"))
    dialog = pd.DataFrame({"Bangs": 0, "Eyeglasses": 0, "Beard": 0, "Smiling": 3, "Age": 1},
                          index=list(meta.index))
    return dict(meta=meta, cache=cache, cache_ids=ids, f2t=load_face2text(fake_f2t), dialog=dialog)


def make_ds(parts, split="train", **kw):
    args = dict(meta=parts["meta"], cache=parts["cache"], cache_ids=parts["cache_ids"], split=split,
                tokenizer=FakeTokenizer(), f2t=parts["f2t"], dialog=parts["dialog"], heldout_pairs=[])
    args.update(kw)
    return FaceCraftDataset(**args)


def test_item_matches_interface(parts):
    ds = make_ds(parts)
    img, tok, attrs, src = ds[0]
    assert img.shape == (3, 64, 64) and img.dtype == torch.float32
    assert -1 <= img.min() and img.max() <= 1
    assert tok.shape == (77,) and tok.dtype == torch.int64
    assert attrs.shape == (40,) and attrs.dtype == torch.float32
    assert set(attrs.unique().tolist()) <= {0.0, 1.0}
    assert src in {"template", "dialog", "face2text"}


def test_split_selects_rows(parts):
    assert make_ds(parts, "train").ids == ["000001.jpg", "000002.jpg", "000003.jpg"]
    assert make_ds(parts, "test").ids == ["000005.jpg", "000006.jpg"]


def test_attrs_match_metadata(parts):
    ds = make_ds(parts, "test")
    _, _, attrs, _ = ds[0]
    expected = parts["meta"].loc["000005.jpg", ATTR_NAMES].to_numpy(dtype=np.float32)
    assert np.array_equal(attrs.numpy(), expected)


def test_no_flip_outside_train(parts):
    ds = make_ds(parts, "test")
    a = ds[0][0]
    b = ds[0][0]
    assert torch.equal(a, b)
    direct = torch.from_numpy(np.array(parts["cache"][parts["cache_ids"].index("000005.jpg")]))
    assert torch.allclose(a, direct.permute(2, 0, 1).float() / 127.5 - 1)


def test_no_dialog_never_samples_dialog(parts):
    ds = make_ds(parts, dialog=None)
    assert all(ds.caption_for(i, random.Random(k))[1] != "dialog" for k in range(50) for i in ds.ids)


def test_f2t_prob_one_always_uses_human_caption(parts):
    ds = make_ds(parts, f2t_prob=1.0)
    assert {ds.caption_for("000001.jpg", random.Random(k))[1] for k in range(20)} == {"face2text"}
    assert ds.caption_for("000001.jpg", random.Random(0))[0] in parts["f2t"]["000001.jpg"]


def test_f2t_prob_zero_never_uses_human_caption(parts):
    ds = make_ds(parts, f2t_prob=0.0)
    assert all(ds.caption_for("000001.jpg", random.Random(k))[1] != "face2text" for k in range(30))


def test_image_without_f2t_or_dialog_uses_template(parts):
    ds = make_ds(parts, dialog=None, f2t_prob=1.0)
    assert {ds.caption_for("000003.jpg", random.Random(k))[1] for k in range(20)} == {"template"}


def test_heldout_dropped_from_train_only(parts):
    meta = parts["meta"].copy()
    meta.loc[:, "Goatee"] = 0
    meta.loc[:, "Wearing_Hat"] = 0
    meta.loc[["000002.jpg", "000006.jpg"], ["Goatee", "Wearing_Hat"]] = 1
    p = dict(parts, meta=meta)
    tr = make_ds(p, "train", heldout_pairs=[("Goatee", "Wearing_Hat")])
    te = make_ds(p, "test", heldout_pairs=[("Goatee", "Wearing_Hat")])
    assert "000002.jpg" not in tr.ids and "000006.jpg" in te.ids
    keep = make_ds(p, "train", heldout_pairs=[("Goatee", "Wearing_Hat")], drop_heldout=False)
    assert "000002.jpg" in keep.ids


def test_pickling_does_not_copy_the_image_cache(parts):
    # DataLoader workers on Windows receive the dataset by pickling; a memmap must be reopened, not copied
    import pickle
    ds = make_ds(parts)
    blob = pickle.dumps(ds)
    assert len(blob) < 200_000
    ds2 = pickle.loads(blob)
    assert torch.equal(ds2[0][2], ds[0][2]) and ds2[0][0].shape == (3, 64, 64)


@pytest.mark.network
def test_clip_tokenizer_truncates_long_caption():
    from data.tokenizer import Tokenizer
    t = Tokenizer()
    out = t(["word " * 200, "a smiling woman"])
    assert out.shape == (2, 77) and out.dtype == torch.int64
    assert (out[1] == t.pad_id).sum() > 60
