"""Test the command-line generator: prompt in, PNG grid out."""
from PIL import Image


def test_generate_from_checkpoint_writes_grid(tmp_path, tiny_gan_cfg):
    from training.train_gan import run_gan
    from scripts.generate import generate
    run_dir = tmp_path / "gan"
    run_gan(dict(tiny_gan_cfg, out_dir=str(run_dir), steps=2, ckpt_every=2))
    out = generate(["a smiling woman", "a bald man"], n=3, seed=0, run_dir=str(run_dir), out=str(tmp_path / "g.png"))
    img = Image.open(out)
    assert img.size == (3 * 64, 2 * 64)          # one row per prompt, n columns


def test_same_seed_same_faces(tmp_path, tiny_gan_cfg):
    from training.train_gan import run_gan
    from scripts.generate import generate
    import numpy as np
    run_dir = tmp_path / "gan"
    run_gan(dict(tiny_gan_cfg, out_dir=str(run_dir), steps=2, ckpt_every=2))
    a = np.array(Image.open(generate(["a man"], n=2, seed=5, run_dir=str(run_dir), out=str(tmp_path / "a.png"))))
    b = np.array(Image.open(generate(["a man"], n=2, seed=5, run_dir=str(run_dir), out=str(tmp_path / "b.png"))))
    assert (a == b).all()
