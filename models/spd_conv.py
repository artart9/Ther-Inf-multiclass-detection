"""SPD-Conv: space-to-depth downsampling + non-strided convolution.

Replaces strided Conv/pooling so no fine-grained information is discarded
by sampling. Spatial size is reduced by `factor`; channels grow by `factor**2`,
then a stride-1 Conv projects back to the target channel count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


def space_to_depth(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """(B, C, H, W) -> (B, C * factor**2, H / factor, W / factor)."""
    return F.pixel_unshuffle(x, factor)


class SPDConv(nn.Module):
    """Space-to-depth followed by a non-strided Conv (SPD-Conv).

    YAML / Ultralytics args match Conv: ``[c2, k, factor]`` so the stem line
    ``[-1, 1, SPDConv, [64, 3, 2]]`` replaces ``[-1, 1, Conv, [64, 3, 2]]``.
    The third arg is the space-to-depth factor (not a convolution stride).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, act: bool = True):
        super().__init__()
        self.factor = int(s)
        self.conv = Conv(c1 * self.factor * self.factor, c2, k, s=1, p=p, g=g, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(space_to_depth(x, self.factor))
