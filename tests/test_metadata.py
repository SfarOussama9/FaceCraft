"""Tests for the metadata loaders (CelebA attributes/split, Face2Text, CelebA-Dialog)."""
import random

import pytest

from data.celeba import ATTR_NAMES, load_celeba_metadata
from data.dialog import dialog_caption, load_dialog_levels
from data.face2text import load_face2text
from tests.conftest import CELEBA_ATTRS


def test_celeba_metadata_shapes(fake_celeba):
    df = load_celeba_metadata(fake_celeba)
    assert list(df.columns[:40]) == ATTR_NAMES and len(ATTR_NAMES) == 40
    assert ATTR_NAMES == CELEBA_ATTRS
    assert set(df["split"]) <= {"train", "val", "test"}
    assert df[ATTR_NAMES].isin([0, 1]).all().all()
    assert df.index[0] == "000001.jpg"


def test_celeba_split_mapping(fake_celeba):
    df = load_celeba_metadata(fake_celeba)
    assert df.loc["000001.jpg", "split"] == "train"
    assert df.loc["000004.jpg", "split"] == "val"
    assert df.loc["000006.jpg", "split"] == "test"


def test_face2text_merges_files_and_drops_empty(fake_f2t):
    caps = load_face2text(fake_f2t)
    assert caps["000001.jpg"] == ["A man with short hair.", "He is smiling."]
    assert "000002.jpg" not in caps          # its only description was "   "


def test_dialog_parser_tolerates_header_lines(tmp_path):
    p = tmp_path / "fg.txt"
    p.write_text("202599\nBangs Eyeglasses No_Beard Smiling Young\n000001.jpg 0 0 0 2 1\n000002.jpg 5 3 0 0 4\n")
    df = load_dialog_levels(str(p))
    assert list(df.columns) == ["Bangs", "Eyeglasses", "Beard", "Smiling", "Age"]
    assert df.loc["000002.jpg"].tolist() == [5, 3, 0, 0, 4]


def test_dialog_parser_real_header_format(tmp_path):
    # Exact format of the real combined_annotation.txt
    p = tmp_path / "combined_annotation.txt"
    p.write_text("img_name Bangs Eyeglasses No_Beard Smiling Young\n000001.jpg 0 0 0 3 1\n202599.jpg 0 0 0 0 1\n")
    df = load_dialog_levels(str(p))
    assert len(df) == 2 and df.loc["000001.jpg", "Smiling"] == 3


def test_dialog_missing_file_returns_none():
    assert load_dialog_levels("does/not/exist.txt") is None
    assert load_dialog_levels(None) is None


def test_dialog_caption_mentions_degree():
    s = dialog_caption({"Bangs": 0, "Eyeglasses": 0, "Beard": 0, "Smiling": 5, "Age": 1},
                       male=False, rng=random.Random(0))
    assert "laugh" in s.lower() and s.strip()


def test_dialog_caption_beard_and_age_direction():
    # Beard 5 = full beard; Age 5 = oldest (verified against CelebA No_Beard / Young labels)
    s = dialog_caption({"Bangs": 0, "Eyeglasses": 0, "Beard": 5, "Smiling": 0, "Age": 5},
                       male=True, rng=random.Random(0)).lower()
    assert "beard" in s and ("old" in s or "elderly" in s or "senior" in s)
    s0 = dialog_caption({"Bangs": 0, "Eyeglasses": 0, "Beard": 0, "Smiling": 0, "Age": 0},
                        male=True, rng=random.Random(0)).lower()
    assert "beard" not in s0
