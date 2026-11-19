"""Shared pytest fixtures: tiny FAKE datasets so tests run anywhere, without the real 1.4 GB CelebA.

The fake data mimics the real file layouts exactly (Kaggle CelebA CSVs, Face2Text JSON,
CelebA-Dialog TXT), so the loaders are tested against the same formats they will meet for real.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make the repo root importable (data/, models/, ... are top-level packages).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The 40 CelebA attributes, in the exact column order of list_attr_celeba.csv.
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
    "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
    "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
    "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]

# image_id -> CelebA partition (0 train, 1 val, 2 test)
FAKE_PARTITION = {"000001.jpg": 0, "000002.jpg": 0, "000003.jpg": 0, "000004.jpg": 1,
                  "000005.jpg": 2, "000006.jpg": 2}


def write_fake_celeba(root: Path, n_images: int = 6, seed: int = 0) -> Path:
    """Create a fake CelebA root in the Kaggle layout and return it."""
    rng = np.random.RandomState(seed)
    img_dir = root / "img_align_celeba" / "img_align_celeba"
    img_dir.mkdir(parents=True, exist_ok=True)
    ids = [f"{i:06d}.jpg" for i in range(1, n_images + 1)]
    attr_lines = ["image_id," + ",".join(CELEBA_ATTRS)]
    part_lines = ["image_id,partition"]
    for k, image_id in enumerate(ids):
        Image.fromarray(rng.randint(0, 255, (218, 178, 3), dtype=np.uint8)).save(img_dir / image_id)
        values = rng.choice([-1, 1], size=40)
        attr_lines.append(image_id + "," + ",".join(str(v) for v in values))
        part_lines.append(f"{image_id},{FAKE_PARTITION.get(image_id, k % 3)}")
    (root / "list_attr_celeba.csv").write_text("\n".join(attr_lines) + "\n")
    (root / "list_eval_partition.csv").write_text("\n".join(part_lines) + "\n")
    return root


@pytest.fixture
def fake_celeba(tmp_path) -> str:
    return str(write_fake_celeba(tmp_path / "celeba"))


@pytest.fixture
def fake_celeba_images(fake_celeba) -> str:
    """Directory that directly contains the fake .jpg files."""
    return str(Path(fake_celeba) / "img_align_celeba" / "img_align_celeba")


@pytest.fixture
def tiny_gan_cfg(tmp_path_factory) -> dict:
    """A complete, tiny GAN config on fake data (offline tokenizer, CPU-sized model, no FID)."""
    import yaml
    from data.celeba import load_celeba_metadata
    from data.preprocess import build_cache

    root = tmp_path_factory.mktemp("tinydata")
    celeba = write_fake_celeba(root / "celeba", n_images=12)
    f2t_dir = root / "face2text"
    f2t_dir.mkdir()
    (f2t_dir / "clean_train_2.1.json").write_text(
        json.dumps([{"response_id": 1, "filename": "000001.jpg", "description": "A smiling man."}]))
    ids = list(load_celeba_metadata(str(celeba)).index)
    cache = root / "cache" / "c.npy"
    build_cache(str(celeba / "img_align_celeba" / "img_align_celeba"), ids, str(cache), workers=1)
    data_cfg = {"celeba_root": str(celeba), "face2text_dir": str(f2t_dir), "dialog_labels": None,
                "cache_path": str(cache), "f2t_prob": 0.5, "dialog_prob": 0.3, "heldout_pairs": [], "seed": 0}
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data_cfg))
    return {"data_config": str(data_yaml), "tokenizer": "hash", "z_dim": 8, "ch": 8, "text_dim": 16,
            "text_layers": 1, "text_heads": 2, "batch_size": 2, "num_workers": 0, "lr": 2e-4,
            "ema_decay": 0.999, "steps": 4, "ckpt_every": 2, "sample_every": 2, "fid_samples": 0,
            "sample_prompts": ["a smiling woman", "a man with glasses"], "amp": False, "seed": 0}


@pytest.fixture
def fake_f2t(tmp_path) -> list:
    """Two Face2Text-style JSON files (same schema as clean_*_2.1.json)."""
    a = [{"response_id": 1, "filename": "000001.jpg", "description": " A man with short hair. "},
         {"response_id": 2, "filename": "000002.jpg", "description": "   "}]
    b = [{"response_id": 3, "filename": "000001.jpg", "description": "He is smiling."},
         {"response_id": 4, "filename": "000005.jpg", "description": "A woman with long blond hair and glasses."}]
    paths = []
    for name, rows in [("clean_train_2.1.json", a), ("clean_test_2.1.json", b)]:
        p = tmp_path / name
        p.write_text(json.dumps(rows), encoding="utf-8")
        paths.append(str(p))
    return paths
