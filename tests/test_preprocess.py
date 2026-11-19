"""Tests for image preprocessing (center crop 140 -> resize 64) and the uint8 image cache."""
import numpy as np
from PIL import Image

from data.preprocess import build_cache, load_cache, preprocess_image


def test_preprocess_center_crop_and_size(tmp_path):
    img = Image.new("RGB", (178, 218), (0, 0, 0))
    img.paste((255, 0, 0), (19, 39, 159, 179))          # 140x140 red square, exactly centered
    p = tmp_path / "a.png"                              # lossless so the border stays sharp
    img.save(p)
    out = preprocess_image(str(p))
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert out[..., 0].mean() > 240 and out[..., 1].mean() < 15   # crop is exactly the red square


def test_preprocess_converts_grayscale_and_cmyk(tmp_path):
    for mode in ["L", "CMYK"]:
        p = tmp_path / f"{mode}.jpg"
        Image.new(mode, (178, 218)).save(p)
        assert preprocess_image(str(p)).shape == (64, 64, 3)


def test_build_and_load_cache_roundtrip(tmp_path, fake_celeba_images):
    ids = ["000001.jpg", "000002.jpg"]
    out = str(tmp_path / "c.npy")
    build_cache(fake_celeba_images, ids, out, workers=1)
    arr, got_ids = load_cache(out)
    assert arr.shape == (2, 64, 64, 3) and arr.dtype == np.uint8 and got_ids == ids
    # same pixels as preprocessing the file directly
    direct = preprocess_image(f"{fake_celeba_images}/000002.jpg")
    assert np.array_equal(np.asarray(arr[1]), direct)


def test_build_cache_skips_when_already_built(tmp_path, fake_celeba_images):
    ids = ["000001.jpg"]
    out = str(tmp_path / "c.npy")
    build_cache(fake_celeba_images, ids, out, workers=1)
    mtime = (tmp_path / "c.npy").stat().st_mtime_ns
    build_cache(fake_celeba_images, ids, out, workers=1)   # second call must be a no-op
    assert (tmp_path / "c.npy").stat().st_mtime_ns == mtime
