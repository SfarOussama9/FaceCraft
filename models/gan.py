"""Text-conditional GAN baseline (trained from scratch).

Generator G(z, text) -> 64x64 image:
  [noise z ; sentence vector] -> linear -> 4x4 feature map -> 4 x (upsample, conv, CONDITIONAL BatchNorm, ReLU) -> 64x64
  Conditional BatchNorm: the text decides the scale/shift of every feature map, so the sentence steers
  the image at every resolution (not only at the input).

Discriminator D(image, text) -> realism score:
  4 x (spectral-normalised conv stride 2, LeakyReLU) -> 4x4 features -> sum-pool to vector h
  score = linear(h) + <embed(text), h>        ("projection discriminator", Miyato & Koyama 2018)
  The inner product rewards images whose features AGREE with the sentence.
  Spectral normalisation (Miyato et al. 2018) limits how sharply D can change -> much more stable training.

Trained with the hinge loss and the "matching-aware" trick of Reed et al. (2016): a real image paired
with the WRONG sentence is also shown to D as fake, so D (and therefore G) must care about the text.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm whose scale (gamma) and shift (beta) are predicted from the condition vector."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)          # start as plain BatchNorm (gamma = 1, beta = 0)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        g = 1 + self.gamma(cond)[:, :, None, None]
        b = self.beta(cond)[:, :, None, None]
        return g * self.bn(x) + b


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.cbn = ConditionalBatchNorm2d(c_out, cond_dim)

    def forward(self, x, cond):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return torch.relu(self.cbn(self.conv(x), cond))


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 256, ch: int = 64):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(z_dim + cond_dim, 8 * ch * 4 * 4)
        self.blocks = nn.ModuleList([UpBlock(8 * ch, 8 * ch, cond_dim),   # 4  -> 8
                                     UpBlock(8 * ch, 4 * ch, cond_dim),   # 8  -> 16
                                     UpBlock(4 * ch, 2 * ch, cond_dim),   # 16 -> 32
                                     UpBlock(2 * ch, ch, cond_dim)])      # 32 -> 64
        self.out = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z, cond):
        x = self.fc(torch.cat([z, cond], dim=1)).view(-1, 8 * self.ch, 4, 4)
        for block in self.blocks:
            x = block(x, cond)
        return self.out(x)                       # [B, 3, 64, 64] in [-1, 1]


class Discriminator(nn.Module):
    def __init__(self, cond_dim: int = 256, ch: int = 64):
        super().__init__()
        chans = [3, ch, 2 * ch, 4 * ch, 8 * ch]
        layers = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):                     # 64 -> 32 -> 16 -> 8 -> 4
            layers += [SN(nn.Conv2d(c_in, c_out, 4, 2, 1)), nn.LeakyReLU(0.2)]
        layers += [SN(nn.Conv2d(8 * ch, 8 * ch, 3, 1, 1)), nn.LeakyReLU(0.2)]
        self.features = nn.Sequential(*layers)
        self.linear = SN(nn.Linear(8 * ch, 1))
        self.embed = SN(nn.Linear(cond_dim, 8 * ch))

    def forward(self, x, cond):
        h = self.features(x).sum(dim=(2, 3))                                 # [B, 8ch]
        return self.linear(h).squeeze(1) + (self.embed(cond) * h).sum(1)     # projection term
