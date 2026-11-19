"""Tests for the attribute classifier and the shared checkpoint helpers."""
import torch
import torch.nn.functional as F

from models.classifier import AttributeClassifier
from training.common import load_latest, save_ckpt


def test_classifier_output_shape():
    assert AttributeClassifier()(torch.randn(2, 3, 64, 64)).shape == (2, 40)


def test_classifier_overfits_tiny_batch():
    torch.manual_seed(0)
    m = AttributeClassifier(width=16)
    x = torch.randn(8, 3, 64, 64)
    y = (torch.rand(8, 40) > .5).float()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(x), y)
        loss.backward()
        opt.step()
    m.eval()
    assert ((m(x) > 0).float() == y).float().mean() > 0.95


def test_ckpt_roundtrip(tmp_path):
    m = AttributeClassifier(width=8)
    opt = torch.optim.Adam(m.parameters())
    save_ckpt(tmp_path / "last.pt", m, None, opt, 1234, {"a": 1})
    assert (tmp_path / "last.yaml").exists()
    m2 = AttributeClassifier(width=8)
    assert load_latest(tmp_path, m2, None, torch.optim.Adam(m2.parameters())) == 1234
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p, q)


def test_load_latest_without_checkpoint_returns_zero(tmp_path):
    m = AttributeClassifier(width=8)
    assert load_latest(tmp_path, m, None, torch.optim.Adam(m.parameters())) == 0


def test_ckpt_supports_module_dict_and_optimizer_dict(tmp_path):
    mods = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
    opts = {"a": torch.optim.Adam(mods["a"].parameters()), "b": torch.optim.Adam(mods["b"].parameters())}
    save_ckpt(tmp_path / "last.pt", mods, None, opts, 7, {})
    mods2 = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
    opts2 = {"a": torch.optim.Adam(mods2["a"].parameters()), "b": torch.optim.Adam(mods2["b"].parameters())}
    assert load_latest(tmp_path, mods2, None, opts2) == 7
    assert torch.equal(mods["b"].weight, mods2["b"].weight)
