"""EduResBlockNode — residual block, the workhorse of diffusion U-Nets.

The standard SD/DDPM residual block:

    h = SiLU(GroupNorm(x))
    h = Conv3x3(h)            # in_channels → out_channels
    if time_emb provided:
        h = h + Linear(time_emb)[:, :, None, None]    # FiLM-style additive bias
    h = SiLU(GroupNorm(h))
    h = Conv3x3(h)
    if in_channels != out_channels:
        skip = Conv1x1(x)
    else:
        skip = x
    return h + skip

The time-embedding injection is what makes it "diffusion-flavoured" —
it lets the same convolutional weights behave differently at different
denoising steps. Without `time_emb` the node degrades to a plain
ResNet-style block, useful for the segmentation U-Net preset.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.stateful_module import StatefulModuleMixin


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


class EduResBlockNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "EduResBlock"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Residual block: GN→SiLU→Conv → (+time emb projection) → GN→SiLU→Conv "
        "→ Add(skip). The building unit of a diffusion U-Net. Connect "
        "`time_emb` from a `TimestepEmbedding` node to make it "
        "time-conditioned; leave it unconnected for a plain ResNet block."
    )

    structural_params = ("in_channels", "out_channels", "groups", "time_emb_dim", "seed")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input feature map [N, in_channels, H, W].",
            ),
            PortDefinition(
                name="time_emb",
                data_type=DataType.TENSOR,
                description="Optional [N, time_emb_dim] timestep embedding for conditioning.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Output feature map [N, out_channels, H, W] (skip-connected).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="in_channels",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Channels of the input feature map.",
            ),
            ParamDefinition(
                name="out_channels",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Channels of the output. When ≠ in_channels, a 1×1 conv is added on the skip path.",
            ),
            ParamDefinition(
                name="groups",
                param_type=ParamType.INT,
                default=4,
                min_value=1,
                description="GroupNorm groups. Must divide both in_channels and out_channels.",
            ),
            ParamDefinition(
                name="time_emb_dim",
                param_type=ParamType.INT,
                default=32,
                min_value=0,
                description="Dimension of the optional `time_emb` input. Set 0 to disable the time-projection layer entirely.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the conv/linear initialisation.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        in_channels = int(params.get("in_channels", 8))
        out_channels = int(params.get("out_channels", 8))
        groups = int(params.get("groups", 4))
        time_emb_dim = int(params.get("time_emb_dim", 32))
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"EduResBlock: groups={groups} must divide both in_channels={in_channels} and out_channels={out_channels}."
            )
        return _ResBlockModule(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            time_emb_dim=time_emb_dim,
            seed=int(params.get("seed", 42)),
        )

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x = inputs.get("tensor")
        if x is None:
            raise ValueError("EduResBlock requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.ndim != 4:
            raise ValueError(
                f"EduResBlock expects [N, C, H, W] shape; got {tuple(x.shape)}."
            )

        in_channels = int(params.get("in_channels", 8))
        time_emb_dim = int(params.get("time_emb_dim", 32))
        groups = int(params.get("groups", 4))
        out_channels = int(params.get("out_channels", 8))
        if x.shape[1] != in_channels:
            raise ValueError(
                f"EduResBlock: input has {x.shape[1]} channels but in_channels={in_channels}."
            )
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"EduResBlock: groups={groups} must divide both in_channels={in_channels} and out_channels={out_channels}."
            )

        time_emb = inputs.get("time_emb")
        if time_emb is not None:
            if not isinstance(time_emb, torch.Tensor):
                time_emb = torch.as_tensor(time_emb, dtype=torch.float32)
            time_emb = time_emb.float()
            if time_emb.ndim == 1:
                # Promote a single [D] to [1, D].
                time_emb = time_emb.unsqueeze(0)
            if time_emb.shape[-1] != time_emb_dim:
                raise ValueError(
                    f"EduResBlock: time_emb last dim {time_emb.shape[-1]} doesn't match time_emb_dim={time_emb_dim}."
                )

        module = self.get_or_build_module(context, params)
        out = module(x, time_emb)
        return {"tensor": out}
