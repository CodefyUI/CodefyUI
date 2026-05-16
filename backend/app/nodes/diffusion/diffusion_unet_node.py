"""DiffusionUNetNode — toy multi-level U-Net wired in one node.

This is the *meta-node* counterpart to the expanded preset that wires
each ResBlock and skip connection by hand. Both presets ship with the
project — students can pick whichever fits their learning style.

Architecture (with default channel_mult="1,2,4"):

    x [N, C_in, H, W]
        ──> stem Conv3x3 ─────────────────────────> h0 [N, base, H, W]
                ↓ ResBlock(base→base*1) ──skip──┐
                ↓ MaxPool 2×                    │
                ↓ ResBlock(base*1→base*2) ──skip┐
                ↓ MaxPool 2×                    ││
                ↓ Bottleneck ResBlock           ││
                ↑ Upsample 2×                   ││
                ↑ Concat skip ──────────────────┘│
                ↑ ResBlock(base*4→base*2)        │
                ↑ Upsample 2×                    │
                ↑ Concat skip ───────────────────┘
                ↑ ResBlock(base*2→base*1)
                ↑ final Conv3x3 ───> ε [N, C_in, H, W]

Every ResBlock receives the same `time_emb` so the whole network is
time-conditioned. The output channel count matches the input channel
count, which is the noise-prediction convention.
"""

from __future__ import annotations

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
from ._resblock_module import _ResBlockModule
from .timestep_embedding_node import _TimestepMLP


def _parse_mult(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("DiffusionUNet: channel_mult is empty.")
    try:
        mults = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"DiffusionUNet: invalid channel_mult {s!r}.") from exc
    if any(m <= 0 for m in mults):
        raise ValueError("DiffusionUNet: channel multipliers must be positive.")
    return mults


class _DiffusionUNetModule(nn.Module):
    """Toy U-Net: time-conditioned, configurable depth via channel_mult."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mult: tuple[int, ...],
        time_emb_dim: int,
        num_groups: int,
        seed: int,
    ) -> None:
        super().__init__()

        if any((base_channels * m) % num_groups for m in channel_mult):
            raise ValueError(
                f"DiffusionUNet: num_groups={num_groups} must divide every base*mult — "
                f"base={base_channels}, mult={channel_mult}."
            )

        # Use a stem and a chain of "down" stages then a chain of "up" stages.
        # We deliberately reuse the existing _ResBlockModule + _TimestepMLP so
        # the meta-node and the expanded preset stay numerically aligned.
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim

        # Time MLP — produces time_emb from raw timesteps.
        self.time_mlp = _TimestepMLP(
            embed_dim=time_emb_dim, max_period=10000, seed=seed
        )

        # Stem — bring input up to base_channels.
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down path: at each level, one ResBlock + (MaxPool except last).
        ch = base_channels
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        skip_channels: list[int] = []
        for level, mult in enumerate(channel_mult[:-1]):
            target = base_channels * mult
            self.down_blocks.append(
                _ResBlockModule(
                    in_channels=ch,
                    out_channels=target,
                    groups=num_groups,
                    time_emb_dim=time_emb_dim,
                    seed=seed + 100 + level,
                )
            )
            ch = target
            skip_channels.append(ch)
            self.down_pools.append(nn.MaxPool2d(2))

        # Bottleneck.
        bottleneck_target = base_channels * channel_mult[-1]
        self.bottleneck = _ResBlockModule(
            in_channels=ch,
            out_channels=bottleneck_target,
            groups=num_groups,
            time_emb_dim=time_emb_dim,
            seed=seed + 200,
        )
        ch = bottleneck_target

        # Up path: mirror the down path. Each up-step does Upsample → Concat
        # with the matching skip → ResBlock.
        self.up_blocks = nn.ModuleList()
        for level, skip_ch in enumerate(reversed(skip_channels)):
            target = skip_ch  # collapse back down to skip's channel count
            self.up_blocks.append(
                _ResBlockModule(
                    in_channels=ch + skip_ch,  # post-concat
                    out_channels=target,
                    groups=num_groups,
                    time_emb_dim=time_emb_dim,
                    seed=seed + 300 + level,
                )
            )
            ch = target

        # Final 1x1 conv back to input channels — that's the noise prediction.
        self.head = nn.Conv2d(ch, in_channels, kernel_size=1)

        # Seed all conv weights once via the buried RNG (each ResBlock seeds
        # its own internals; we only need to seed the bare convs we add here).
        gen = torch.Generator()
        gen.manual_seed(seed + 999)
        with torch.no_grad():
            for layer in (self.stem, self.head):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=gen) * 0.1
                )
                layer.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Project timestep into the embedding space the ResBlocks expect.
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        time_emb = self.time_mlp(t)

        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for block, pool in zip(self.down_blocks, self.down_pools):
            h = block(h, time_emb)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h, time_emb)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            if h.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    f"DiffusionUNet: spatial mismatch on skip — h={tuple(h.shape)} skip={tuple(skip.shape)}. "
                    f"Input H/W must be divisible by 2^(num_levels-1)."
                )
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
        return self.head(h)


class DiffusionUNetNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "DiffusionUNet"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "A complete toy diffusion U-Net packaged as a single node. Outputs an "
        "nn.Module that maps `(x, t) → predicted_noise` of the same shape as "
        "x. Compose with `DDPMSampler` to run reverse diffusion. For students "
        "who want to see the architecture wired up explicitly, see the "
        "`Mini-UNet-Expanded` preset."
    )

    structural_params = (
        "in_channels",
        "base_channels",
        "channel_mult",
        "time_emb_dim",
        "num_groups",
        "seed",
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Callable nn.Module taking (x, t) → predicted noise.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="in_channels",
                param_type=ParamType.INT,
                default=3,
                min_value=1,
                description="Channels of the noisy input (3 for RGB, 4 for typical SD latents).",
            ),
            ParamDefinition(
                name="base_channels",
                param_type=ParamType.INT,
                default=16,
                min_value=1,
                description="Channel count after the stem. Each level multiplies this by the matching `channel_mult` entry.",
            ),
            ParamDefinition(
                name="channel_mult",
                param_type=ParamType.STRING,
                default="1,2,4",
                description="Comma-separated channel multipliers per level. Length determines depth (down-blocks + bottleneck).",
            ),
            ParamDefinition(
                name="time_emb_dim",
                param_type=ParamType.INT,
                default=32,
                min_value=2,
                description="Width of the timestep embedding (must be even).",
            ),
            ParamDefinition(
                name="num_groups",
                param_type=ParamType.INT,
                default=4,
                min_value=1,
                description="GroupNorm groups inside every ResBlock. Must divide all level channel counts.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for all weight initialisation.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        return _DiffusionUNetModule(
            in_channels=int(params.get("in_channels", 3)),
            base_channels=int(params.get("base_channels", 16)),
            channel_mult=_parse_mult(str(params.get("channel_mult", "1,2,4"))),
            time_emb_dim=int(params.get("time_emb_dim", 32)),
            num_groups=int(params.get("num_groups", 4)),
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
        # Validate channel_mult eagerly (build_module would too, but a clean
        # error message at graph load time is friendlier).
        _parse_mult(str(params.get("channel_mult", "1,2,4")))
        model = self.get_or_build_module(context, params)
        return {"model": model}
