"""Tests for picking the right data config (laptop vs. Kaggle)."""
from utils.config import default_data_config


def test_default_data_config_local(tmp_path):
    assert default_data_config(kaggle_input=str(tmp_path / "nope")) == "configs/data.yaml"


def test_default_data_config_on_kaggle(tmp_path):
    (tmp_path / "input").mkdir()
    assert default_data_config(kaggle_input=str(tmp_path / "input")) == "configs/data_kaggle.yaml"


def test_env_var_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("FACECRAFT_DATA_CONFIG", "configs/custom.yaml")
    assert default_data_config(kaggle_input=str(tmp_path)) == "configs/custom.yaml"
