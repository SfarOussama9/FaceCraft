"""Type a prompt, get faces: generate images from a trained model checkpoint.

    python scripts/generate.py "a smiling young woman with blond hair"
    python scripts/generate.py "a bald older man" "a young man with glasses" --n 8 --seed 42
    python scripts/generate.py "a woman with bangs" --run outputs/gan --out my_faces.png

Output: one PNG grid, one row per prompt, --n faces per row (default 8), opened automatically.
The same --seed always gives the same faces. Currently supports the GAN baseline (outputs/gan);
the diffusion model will be added here later.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow running as a plain script

from data.tokenizer import make_tokenizer
from models.gan import Generator
from models.text_encoder import TextEncoder
from training.train_gan import GANSampler, save_grid
from utils.config import load_config


def generate(prompts: list, n: int = 8, seed: int = 0, run_dir: str = "outputs/gan", out: str = "generated.png") -> str:
    """Load the EMA generator + text encoder from <run_dir>/last.pt, draw n faces per prompt, save a grid."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "last.yaml")                     # the config saved with the checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = make_tokenizer(cfg.get("tokenizer", "clip"))
    text = TextEncoder(tok.vocab_size, cfg["text_dim"], cfg["text_layers"], cfg["text_heads"], pad_id=tok.pad_id)
    G = Generator(cfg["z_dim"], cfg["text_dim"], cfg["ch"])
    ema = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)["ema"]   # smoothed weights
    G.load_state_dict({k[2:]: v for k, v in ema.items() if k.startswith("G.")})
    text.load_state_dict({k[5:]: v for k, v in ema.items() if k.startswith("text.")})
    sampler = GANSampler(G.to(device), text.to(device), tok, cfg["z_dim"], device)
    images = sampler.sample(prompts, n=n, seed=seed)
    save_grid(images, len(prompts), Path(out))
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate faces from text prompts.")
    ap.add_argument("prompts", nargs="+", help='one or more prompts in quotes, e.g. "a smiling woman with glasses"')
    ap.add_argument("--n", type=int, default=8, help="faces per prompt")
    ap.add_argument("--seed", type=int, default=0, help="change it to get different faces")
    ap.add_argument("--run", default="outputs/gan", help="folder with last.pt / last.yaml")
    ap.add_argument("--out", default="generated.png")
    a = ap.parse_args()
    path = generate(a.prompts, a.n, a.seed, a.run, a.out)
    print(f"saved {path}")
    if os.name == "nt":
        os.startfile(os.path.abspath(path))                       # open the image on Windows
