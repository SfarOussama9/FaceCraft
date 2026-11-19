"""Tests for the from-scratch Transformer text encoder, the conditional GAN, and FID/KID."""
import numpy as np
import pytest
import torch

from models.gan import Discriminator, Generator
from models.text_encoder import TextEncoder


def test_text_encoder_shapes_and_padding_invariance():
    torch.manual_seed(0)
    enc = TextEncoder(vocab_size=1000, dim=32, layers=2, heads=4, pad_id=0).eval()
    a = torch.tensor([[5, 6, 7] + [0] * 74])
    tok, pooled = enc(a)
    assert tok.shape == (1, 77, 32) and pooled.shape == (1, 32)
    b = a.clone()
    b[0, 50] = 0                      # changing padding does nothing
    assert torch.allclose(enc(b)[1], pooled, atol=1e-5)
    c = a.clone()
    c[0, 1] = 9                       # changing a real word changes the output
    assert not torch.allclose(enc(c)[1], pooled, atol=1e-4)


def test_text_encoder_all_padding_row_is_finite():
    enc = TextEncoder(vocab_size=100, dim=16, layers=1, heads=2, pad_id=0).eval()
    tok, pooled = enc(torch.zeros(2, 77, dtype=torch.long))
    assert torch.isfinite(tok).all() and torch.isfinite(pooled).all()


def test_gan_shapes():
    G, D = Generator(z_dim=16, cond_dim=32, ch=8), Discriminator(cond_dim=32, ch=8)
    x = G(torch.randn(4, 16), torch.randn(4, 32))
    assert x.shape == (4, 3, 64, 64) and x.abs().max() <= 1
    assert D(x, torch.randn(4, 32)).shape == (4,)


def test_discriminator_uses_spectral_norm():
    D = Discriminator(cond_dim=32, ch=8)
    assert any(hasattr(m, "weight_orig") for m in D.modules())


def test_generator_depends_on_condition():
    torch.manual_seed(0)
    G = Generator(z_dim=16, cond_dim=32, ch=8).eval()
    z = torch.randn(2, 16)
    assert not torch.allclose(G(z, torch.randn(2, 32)), G(z, torch.randn(2, 32)))


def test_gan_resumes_from_checkpoint(tmp_path, tiny_gan_cfg):
    from training.train_gan import run_gan
    assert run_gan(dict(tiny_gan_cfg, out_dir=str(tmp_path), steps=4, ckpt_every=2)) == 0
    assert run_gan(dict(tiny_gan_cfg, out_dir=str(tmp_path), steps=6, ckpt_every=2)) == 4
    assert (tmp_path / "last.pt").exists() and (tmp_path / "metrics.json").exists()


def test_gan_does_n_dis_discriminator_steps_per_generator_step(tmp_path, tiny_gan_cfg):
    # Regression: with Adam(beta1=0) and 1 D step per G step the discriminator collapsed (went blind to the
    # image) within ~50 steps. The SN-GAN recipe (Miyato et al. 2018) uses n_dis = 5.
    from training.train_gan import run_gan
    run_gan(dict(tiny_gan_cfg, out_dir=str(tmp_path), steps=2, n_dis=3, ckpt_every=100))
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)["optimizer"]
    d_steps = next(iter(state["D"]["state"].values()))["step"]
    g_steps = next(iter(state["G"]["state"].values()))["step"]
    assert float(g_steps) == 2 and float(d_steps) == 6


def test_gan_config_uses_reference_n_dis():
    from utils.config import load_config
    assert load_config("configs/gan.yaml")["n_dis"] == 5


@pytest.mark.network
def test_fid_identical_sets_near_zero():
    from evaluation.fid_kid import compute_fid_kid
    x = np.random.RandomState(0).randint(0, 255, (64, 64, 64, 3), dtype=np.uint8)
    out = compute_fid_kid(x, x, "cpu", kid_subset=32)
    assert out["fid"] < 1.0 and set(out) >= {"fid", "kid_mean", "kid_std"}
