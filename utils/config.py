"""Load experiment settings from YAML files (all hyperparameters live in configs/, never in code)."""
import yaml


def load_config(path: str) -> dict:
    """Read a YAML config file and return it as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
