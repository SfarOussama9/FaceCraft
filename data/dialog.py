"""CelebA-Dialog fine-grained labels (Jiang et al., 2021) and the degree-aware captions built from them.

The file combined_annotation.txt gives each CelebA image 5 levels from 0 to 5:
    img_name Bangs Eyeglasses No_Beard Smiling Young
    000001.jpg 0 0 0 3 1
Two column names are misleading. We checked them against CelebA's own yes/no labels:
  - "No_Beard" is really the AMOUNT of beard (0 = none ... 5 = full beard)  -> we call it "Beard"
  - "Young"    is really the AGE             (0 = youngest ... 5 = oldest) -> we call it "Age"
For the other three, level 0 means "absent" and 1-5 are increasing degrees.
"""
import random
from pathlib import Path
from typing import Optional

import pandas as pd

DIALOG_COLUMNS = ["Bangs", "Eyeglasses", "Beard", "Smiling", "Age"]

# Our own wording for each level (index = level 0..5).
BANGS = ["", "very short bangs", "short bangs", "medium-length bangs", "long bangs", "very long bangs"]
GLASSES = ["", "rimless glasses", "thin-framed glasses", "thick-framed glasses",
           "thin-framed sunglasses", "thick-framed sunglasses"]
BEARD = ["", "light stubble", "a short beard", "a beard", "a thick beard", "a full, bushy beard"]
SMILE = ["a neutral expression", "a faint smile", "a slight smile", "a smile", "a big smile", "laughing"]
AGE = ["a teenage", "a young", "a young adult", "a middle-aged", "an older", "an elderly"]


def load_dialog_levels(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Parse the fine-grained label file. Returns None if the file is not available.

    Tolerant on purpose: any line that does not start with "<name>.jpg" followed by at least
    5 integers (headers, counts) is skipped.
    """
    if not path or not Path(path).exists():
        return None
    ids, rows = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6 or not parts[0].endswith(".jpg"):
                continue
            try:
                levels = [int(p) for p in parts[1:6]]
            except ValueError:
                continue
            ids.append(parts[0])
            rows.append(levels)
    return pd.DataFrame(rows, index=pd.Index(ids, name="image_id"), columns=DIALOG_COLUMNS)


def dialog_caption(levels: dict, male: bool, rng: random.Random) -> str:
    """Build a degree-aware caption, e.g. "a middle-aged man with a slight smile and thick-framed glasses"."""
    person = "man" if male else "woman"
    subject = f"{AGE[levels['Age']]} {person}"
    details = []
    if levels["Bangs"] > 0:
        details.append(BANGS[levels["Bangs"]])
    if levels["Eyeglasses"] > 0:
        details.append(GLASSES[levels["Eyeglasses"]])
    if male and levels["Beard"] > 0:
        details.append(BEARD[levels["Beard"]])
    rng.shuffle(details)
    if levels["Smiling"] == 5:                       # "an older man laughing, with A and B"
        head = f"{subject} laughing"
        return head if not details else f"{head}, with {_join(details)}"
    # "a young woman with a slight smile, short bangs and thin-framed glasses"
    return f"{subject} with {_join([SMILE[levels['Smiling']]] + details)}"


def _join(phrases: list) -> str:
    """["A"] -> "A";  ["A", "B", "C"] -> "A, B and C"."""
    return phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
