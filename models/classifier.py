"""Attribute classifier ("the judge"): a ResNet-18-style CNN, trained from scratch on real CelebA.

Input: a 64x64 face. Output: 40 logits, one per CelebA attribute (sigmoid > 0.5 means "present").
We use it ONLY to evaluate generators: "the prompt asked for glasses; does the generated face have glasses?"

Architecture (He et al., 2016, adapted to 64x64 inputs):
  stem: 3x3 conv (stride 1, no max-pool - the image is already small)
  4 stages x 2 BasicBlocks, widths w, 2w, 4w, 8w, the last 3 stages halve the resolution (64 -> 8)
  global average pool -> linear layer -> 40 logits
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = relu(F(x) + shortcut(x))."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:      # match shape when resolution or channels change
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class AttributeClassifier(nn.Module):
    def __init__(self, num_attrs: int = 40, width: int = 64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        layers, c_in = [], w
        for c_out, stride in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers += [BasicBlock(c_in, c_out, stride), BasicBlock(c_out, c_out, 1)]
            c_in = c_out
        self.stages = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, num_attrs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.stem(x))          # [B, 8w, 8, 8]
        h = h.mean(dim=(2, 3))                 # global average pool -> [B, 8w]
        return self.head(h)                    # [B, 40] logits
