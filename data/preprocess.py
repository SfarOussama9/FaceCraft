"""Image preprocessing: aligned CelebA photo (178x218) -> centered 140x140 crop -> 64x64 RGB.

Why 140x140? CelebA's aligned photos put the face in the middle; a 140-pixel square around the
center keeps the whole face (hair, chin) and drops most background. This is the standard crop used
by DCGAN / diffusion papers on CelebA 64x64.

Why a cache? Decoding 200k JPEGs every epoch is slow (especially on Kaggle). We decode once and store
all images as one uint8 array on disk (~2.5 GB). Training then reads it via memory-mapping, which is
almost free.
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(path: str, crop: int = 140, size: int = 64) -> np.ndarray:
    """Open any image file, center-crop `crop`x`crop`, resize to `size`x`size`. Returns uint8 [size, size, 3]."""
    with Image.open(path) as img:
        img = img.convert("RGB")                       # handles grayscale / CMYK files safely
        w, h = img.size
        left, top = (w - crop) // 2, (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((size, size), Image.BICUBIC)  # Pillow anti-aliases when downscaling
        return np.asarray(img, dtype=np.uint8)


def _ids_path(out_path: str) -> Path:
    return Path(str(out_path) + ".ids.txt")


def build_cache(image_dir: str, image_ids: list, out_path: str, workers: int = 4) -> None:
    """Preprocess every image in `image_ids` (in that order) into one .npy file + a list of ids.

    Skips all work if a finished cache with exactly these ids already exists.
    """
    out_path = Path(out_path)
    ids_file = _ids_path(out_path)
    if out_path.exists() and ids_file.exists() and ids_file.read_text().split() == list(image_ids):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(image_ids), 64, 64, 3))
    paths = [os.path.join(image_dir, i) for i in image_ids]
    if workers <= 1:
        for k, p in enumerate(paths):
            arr[k] = preprocess_image(p)
    else:
        with Pool(workers) as pool:
            for k, img in enumerate(pool.imap(preprocess_image, paths, chunksize=256)):
                arr[k] = img
    arr.flush()
    del arr
    os.replace(tmp, out_path)                      # only a complete cache gets the final name
    ids_file.write_text("\n".join(image_ids))


def load_cache(out_path: str):
    """Return (images as a read-only memory-mapped uint8 array [N,64,64,3], list of image ids)."""
    arr = np.load(out_path, mmap_mode="r")
    ids = _ids_path(out_path).read_text().split()
    return arr, ids


if __name__ == "__main__":
    # Build the full CelebA cache from the paths in a data config:
    #   python -m data.preprocess --config configs/data.yaml [--limit 1000] [--workers 8]
    import argparse
    import time

    from data.celeba import load_celeba_metadata
    from utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    ap.add_argument("--out", default=None, help="override cache_path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ids = list(load_celeba_metadata(cfg["celeba_root"]).index)[: args.limit]
    out = args.out or cfg["cache_path"]
    t0 = time.time()
    build_cache(os.path.join(cfg["celeba_root"], "img_align_celeba", "img_align_celeba"), ids, out, args.workers)
    print(f"cache: {out}  images: {len(ids)}  time: {time.time() - t0:.0f}s")
