"""Reproducibility: one function that fixes every random number generator we use."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs) so runs can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
