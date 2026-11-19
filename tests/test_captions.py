"""Tests for the template caption engine and held-out attribute pairs."""
import random

import numpy as np
import pandas as pd
import pytest

from data.captions import CAPTION_ATTRS, template_caption
from data.celeba import ATTR_NAMES
from data.splits import DEMO_PAIRS, choose_heldout_pairs, heldout_mask


def zeros(**on):
    a = {n: 0 for n in ATTR_NAMES}
    a.update(on)
    return a


def test_caption_attrs_exclusions():
    for bad in ["Attractive", "Blurry", "Wearing_Necktie", "Wearing_Necklace"]:
        assert bad not in CAPTION_ATTRS
    assert len(CAPTION_ATTRS) == 36


def test_caption_respects_exclusions_and_rules():
    rng = random.Random(0)
    a = zeros(Male=0, Attractive=1, Blurry=1, Bald=1, Blond_Hair=1, Bangs=1, Goatee=1, Wavy_Hair=1)
    for _ in range(50):
        s = template_caption(a, rng).lower()
        assert "attractive" not in s and "blurry" not in s
        assert "blond" not in s and "bangs" not in s and "wavy" not in s   # bald suppresses hair
        assert "goatee" not in s                                           # facial hair only if Male=1


def test_two_hair_colors_mentions_none():
    a = zeros(Black_Hair=1, Blond_Hair=1)
    for k in range(20):
        s = template_caption(a, random.Random(k)).lower()
        assert "black" not in s and "blond" not in s


def test_all_zero_attrs_gives_nonempty_caption():
    s = template_caption(zeros(), random.Random(0))
    assert len(s.split()) >= 3 and "woman" in s.lower()


def test_male_young_subject():
    s = template_caption(zeros(Male=1, Young=1), random.Random(0)).lower()
    assert "man" in s and "woman" not in s


def test_mentions_requested_attribute():
    for k in range(20):
        s = template_caption(zeros(Male=1, Eyeglasses=1), random.Random(k)).lower()
        assert any(w in s for w in ["glasses", "spectacles"])


def test_stubble_is_not_called_a_beard():
    # Review finding: CelebA No_Beard=0 mostly means stubble (Dialog: 49% level 1, 2.4% full beard)
    a = zeros(Male=1, No_Beard=0, **{"5_o_Clock_Shadow": 1})
    for k in range(50):
        s = template_caption(a, random.Random(k)).lower()
        assert "beard" not in s


def test_never_claims_a_full_beard():
    a = zeros(Male=1, No_Beard=0)
    for k in range(50):
        assert "full beard" not in template_caption(a, random.Random(k)).lower()


def test_captions_vary():
    a = zeros(Male=1, Smiling=1, Eyeglasses=1, Black_Hair=1)
    assert len({template_caption(a, random.Random(i)) for i in range(30)}) >= 10


@pytest.fixture
def synthetic_attr_df():
    """5,000 random faces with realistic-ish attribute rates and a split column."""
    rng = np.random.RandomState(0)
    n = 5000
    rates = {name: 0.15 for name in ATTR_NAMES}
    rates.update(Male=0.4, Young=0.75, Smiling=0.5, No_Beard=0.8)
    data = {name: (rng.rand(n) < rates[name]).astype(int) for name in ATTR_NAMES}
    df = pd.DataFrame(data, index=[f"{i:06d}.jpg" for i in range(1, n + 1)])
    df["split"] = rng.choice(["train", "val", "test"], size=n, p=[0.8, 0.1, 0.1])
    return df


def test_heldout_pairs_exclude_demo_pairs(synthetic_attr_df):
    pairs = choose_heldout_pairs(synthetic_attr_df, k=3, min_test=5, max_train_frac=0.05)
    demo = {tuple(sorted(p)) for p in DEMO_PAIRS}
    assert len(pairs) == 3 and not {tuple(sorted(p)) for p in pairs} & demo
    for a, b in pairs:
        assert "Male" not in (a, b) and "Young" not in (a, b)
        both = (synthetic_attr_df[a] == 1) & (synthetic_attr_df[b] == 1)
        assert (both & (synthetic_attr_df["split"] == "test")).sum() >= 5
    m = heldout_mask(synthetic_attr_df, pairs)
    for a, b in pairs:
        assert m[(synthetic_attr_df[a] == 1) & (synthetic_attr_df[b] == 1)].all()


def test_heldout_pairs_with_int8_labels(synthetic_attr_df):
    # load_celeba_metadata stores labels as int8: co-occurrence counts must not overflow at 127
    df = synthetic_attr_df.copy()
    df[ATTR_NAMES] = df[ATTR_NAMES].astype("int8")
    assert choose_heldout_pairs(df, k=3, min_test=5, max_train_frac=0.05) == \
        choose_heldout_pairs(synthetic_attr_df, k=3, min_test=5, max_train_frac=0.05)


def test_heldout_pairs_deterministic(synthetic_attr_df):
    p1 = choose_heldout_pairs(synthetic_attr_df, k=3, min_test=5, max_train_frac=0.05, seed=0)
    p2 = choose_heldout_pairs(synthetic_attr_df, k=3, min_test=5, max_train_frac=0.05, seed=0)
    assert p1 == p2


def test_heldout_mask_empty_pairs(synthetic_attr_df):
    assert not heldout_mask(synthetic_attr_df, []).any()
