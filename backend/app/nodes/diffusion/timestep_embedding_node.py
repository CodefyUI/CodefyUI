"""TimestepEmbeddingNode — turn a diffusion timestep into a vector.

The U-Net needs to know "which step are we on" to denoise differently
near t=0 (almost-clean image, small adjustments) vs. near t=T (pure
noise, big adjustments). The standard trick from DDPM (Ho et al. 2020)
is the same sinusoidal frequency-bank used for transformer positions,
followed by a small MLP (Linear → SiLU → Linear) to give the model room
to nonlinearly remap the time signal:

    freq[i] = exp(-ln(max_period) * i / (D/2)),  i = 0..D/2-1
    sin/cos:   sin(t · freq[i])  cos(t · freq[i])
    project:   Linear(D, 4D) → SiLU → Linear(4D, D)

Output is broadcast-ready: a [B, D] tensor that ResBlocks can project
further and add channel-wise to feature maps.
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


class _TimestepMLP(nn.Module):
    """Frequency embedding + 2-layer projection."""

    def __init__(self, embed_dim: int, max_period: int, seed: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        scale = 1.0 / math.sqrt(embed_dim)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=gen) * scale
                )
                layer.bias.zero_()

    def freq_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding identical to DDPM/Vaswani — half sin, half cos."""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )  # [half]
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.freq_embed(timesteps)
        return self.fc2(F.silu(self.fc1(emb)))


class TimestepEmbeddingNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "TimestepEmbedding"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Encode a diffusion timestep $t$ into a vector that conditions U-Net "
        "blocks. Sinusoidal frequency bank (à la Vaswani) followed by "
        "Linear→SiLU→Linear, the standard DDPM recipe."
    )

    structural_params = ("embed_dim", "max_period", "seed")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="timestep",
                data_type=DataType.TENSOR,
                description="Scalar int, 0-d tensor, or [B] tensor of timesteps.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embedding",
                data_type=DataType.TENSOR,
                description="Float32 tensor of shape [B, embed_dim].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="embed_dim",
                param_type=ParamType.INT,
                default=32,
                min_value=2,
                description="Dimension of the time vector. Must be even (sin/cos halves).",
            ),
            ParamDefinition(
                name="max_period",
                param_type=ParamType.INT,
                default=10000,
                min_value=1,
                description="Largest period in the frequency bank — controls how many distinct timesteps the embedding can resolve.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the projection-layer initialisation.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        embed_dim = int(params.get("embed_dim", 32))
        if embed_dim < 2:
            raise ValueError(f"TimestepEmbedding: embed_dim must be ≥ 2, got {embed_dim}.")
        if embed_dim % 2 != 0:
            raise ValueError(f"TimestepEmbedding: embed_dim must be even, got {embed_dim}.")
        return _TimestepMLP(
            embed_dim=embed_dim,
            max_period=int(params.get("max_period", 10000)),
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
        t_in = inputs.get("timestep")
        if t_in is None:
            raise ValueError("TimestepEmbedding requires a `timestep` input.")

        if isinstance(t_in, torch.Tensor):
            timesteps = t_in
        else:
            timesteps = torch.as_tensor(t_in)
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)

        # Validate params here too (build_module won't run if mixin caches an
        # earlier shape-incompatible build).
        embed_dim = int(params.get("embed_dim", 32))
        if embed_dim < 2:
            raise ValueError(f"TimestepEmbedding: embed_dim must be ≥ 2, got {embed_dim}.")
        if embed_dim % 2 != 0:
            raise ValueError(f"TimestepEmbedding: embed_dim must be even, got {embed_dim}.")

        module = self.get_or_build_module(context, params)
        embedding = module(timesteps)
        return {"embedding": embedding}
