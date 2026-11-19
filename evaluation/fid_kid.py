"""FID and KID: how close the distribution of generated faces is to real faces (lower = better).

Both pass images through a pretrained InceptionV3 network (a standard, *measurement-only* use of a
pretrained model) and compare the statistics of the 2048-d features:
  FID (Heusel et al., 2017): distance between two Gaussians fitted to real vs. fake features.
  KID (Binkowski et al., 2018): kernel (MMD) distance; unbiased, more reliable with few samples.
Same protocol for every model: N generated images vs. N real TEST images, both 64x64 uint8.
"""
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def _batches(arr: np.ndarray, batch: int):
    for i in range(0, len(arr), batch):
        yield torch.from_numpy(np.ascontiguousarray(arr[i:i + batch])).permute(0, 3, 1, 2)   # [B,3,H,W] uint8


@torch.no_grad()
def compute_fid_kid(fake_uint8: np.ndarray, real_uint8: np.ndarray, device, batch: int = 256,
                    kid_subset: int = 1000) -> dict:
    """fake_uint8, real_uint8: [N, 64, 64, 3] uint8 arrays. Returns {"fid", "kid_mean", "kid_std", "n_fake", "n_real"}."""
    subset = min(kid_subset, len(fake_uint8), len(real_uint8))
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)
    for arr, real in ((real_uint8, True), (fake_uint8, False)):
        for x in _batches(arr, batch):
            x = x.to(device)
            fid.update(x, real=real)
            kid.update(x, real=real)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(kid_mean), "kid_std": float(kid_std),
            "n_fake": int(len(fake_uint8)), "n_real": int(len(real_uint8))}
