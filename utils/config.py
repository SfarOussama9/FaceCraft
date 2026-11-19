"""Load experiment settings from YAML files (all hyperparameters live in configs/, never in code)."""
import os
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Read a YAML config file and return it as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_data_config(kaggle_input: str = "/kaggle/input") -> str:
    """Which data config to use: $FACECRAFT_DATA_CONFIG if set, the Kaggle paths on Kaggle, else the local one."""
    if os.environ.get("FACECRAFT_DATA_CONFIG"):
        return os.environ["FACECRAFT_DATA_CONFIG"]
    return "configs/data_kaggle.yaml" if Path(kaggle_input).exists() else "configs/data.yaml"
