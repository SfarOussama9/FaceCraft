"""Held-out attribute pairs for the compositional generalization test.

Idea: pick a few combinations, e.g. (Goatee, Wearing_Hat), and remove EVERY training image that has
both. The generators still see goatees and hats separately, never together. At test time we ask for
"a man with a goatee wearing a hat" and measure whether the model can combine the two concepts.

We never hold out combinations used in the live demo (DEMO_PAIRS), otherwise the demo would suffer.
"""
import itertools
import random

import pandas as pd

from data.captions import CAPTION_ATTRS

DEMO_PAIRS = [("Blond_Hair", "Eyeglasses"), ("Smiling", "Young"), ("Smiling", "Eyeglasses"),
              ("Black_Hair", "Smiling"), ("Male", "Eyeglasses")]

# Attributes that make poor held-out pairs: subject words, the inverted beard label, and hair labels
# whose combinations are contradictory by our caption rules.
_NOT_CANDIDATES = {"Male", "Young", "No_Beard"}
_CONTRADICTORY = {frozenset(p) for p in itertools.combinations(
    ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald"], 2)} | {
    frozenset(("Bald", x)) for x in ["Bangs", "Wavy_Hair", "Straight_Hair", "Receding_Hairline"]}


def choose_heldout_pairs(df: pd.DataFrame, k: int = 3, min_test: int = 200, exclude: list = DEMO_PAIRS,
                         seed: int = 0, max_train_frac: float = 0.02) -> list:
    """Pick k attribute pairs that (1) have >= min_test test images with both attributes,
    (2) cover <= max_train_frac of train images (removing them costs little data),
    (3) are not demo pairs, and (4) share no attribute with each other. Deterministic given seed."""
    excluded = {frozenset(p) for p in exclude}
    candidates = [a for a in CAPTION_ATTRS if a not in _NOT_CANDIDATES]
    # int64: labels are stored as int8, and counts above 127 would silently overflow
    train = df[df["split"] == "train"][candidates].astype("int64")
    test = df[df["split"] == "test"][candidates].astype("int64")
    co_train = train.T @ train            # co_train.loc[a, b] = number of train images with a AND b
    co_test = test.T @ test
    qualified = []
    for a, b in itertools.combinations(candidates, 2):
        pair = frozenset((a, b))
        if pair in excluded or pair in _CONTRADICTORY:
            continue
        if co_test.loc[a, b] >= min_test and co_train.loc[a, b] <= max_train_frac * len(train):
            qualified.append((a, b))
    rng = random.Random(seed)
    rng.shuffle(qualified)
    chosen, used = [], set()
    for a, b in qualified:
        if a in used or b in used:
            continue
        chosen.append((a, b))
        used |= {a, b}
        if len(chosen) == k:
            break
    if len(chosen) < k:
        raise ValueError(f"only {len(chosen)} qualifying pairs found (wanted {k})")
    return chosen


def heldout_mask(df: pd.DataFrame, pairs: list) -> pd.Series:
    """True for every image that contains at least one held-out pair."""
    mask = pd.Series(False, index=df.index)
    for a, b in pairs:
        mask |= (df[a] == 1) & (df[b] == 1)
    return mask
