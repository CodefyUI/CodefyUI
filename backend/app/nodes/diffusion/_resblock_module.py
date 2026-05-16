"""Shared residual block ``nn.Module`` used by the diffusion family.

Both the production ``DiffusionUNet`` meta-node and the educational
``EduResBlock`` (which ships in the ``c3`` chapter plugin) instantiate
this same convolutional block. Keeping it as a private sibling helper
of ``DiffusionUNet`` keeps the production node self-sufficient even
when the educational plugin isn't installed.

The block follows the standard SD/DDPM shape::

    h = SiLU(GroupNorm(x))
    h = Conv3x3(h)                       # in_channels → out_channels
    if time_emb is not None:
        h = h + time_proj(time_emb)      # FiLM-style additive bias
    h = SiLU(GroupNorm(h))
    h = Conv3x3(h)
    skip = Conv1x1(x) if shape mismatch else x
    return h + skip
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlockModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
        time_emb_dim: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        gen = torch.Generator()
        gen.manual_seed(int(seed))

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim > 0 else None
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Seed the conv/linear weights so the same params produce the same module.
        scale = 1.0 / math.sqrt(max(in_channels, 1))
        with torch.no_grad():
            for layer in (self.conv1, self.conv2):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=gen) * scale
                )
                layer.bias.zero_()
            if self.time_proj is not None:
                self.time_proj.weight.copy_(
                    torch.randn(self.time_proj.weight.shape, generator=gen) * scale
                )
                self.time_proj.bias.zero_()
            if isinstance(self.skip, nn.Conv2d):
                self.skip.weight.copy_(
                    torch.randn(self.skip.weight.shape, generator=gen) * scale
                )
                self.skip.bias.zero_()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if time_emb is not None and self.time_proj is not None:
            h = h + self.time_proj(time_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)
