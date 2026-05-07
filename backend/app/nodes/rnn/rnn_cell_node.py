"""RNNCellNode — single-step vanilla recurrent cell.

The simplest sequence operator, used to introduce the recurrent idea
before the more involved LSTM and GRU gates:

    h_t = nonlinearity( W_ih @ x_t + W_hh @ h_{t-1} + b )

Wraps ``nn.RNNCell`` so the same module can be reused across time
steps if the user manually unrolls the loop on the canvas. The
default ``hidden`` input is zeros — students unrolling 3 steps wire
each cell's ``hidden`` output into the next cell's ``hidden`` input.

(This is the textbook C4-1 lesson — once students see the recurrent
idea here, ``LSTM`` and ``GRU`` are introduced as more sophisticated
versions of the same pattern.)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.stateful_module import StatefulModuleMixin


class _SeededRNNCell(nn.Module):
    """nn.RNNCell with deterministic init from a seed."""

    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str, seed: int) -> None:
        super().__init__()
        self.cell = nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
        )
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        scale = 1.0 / math.sqrt(max(1, hidden_size))
        with torch.no_grad():
            for p in self.cell.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=gen) * scale)
                else:
                    p.zero_()

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        return self.cell(x, h)


class RNNCellNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "RNNCell"
    CATEGORY = "RNN"
    DESCRIPTION = (
        "Single-step vanilla RNN cell: $h_t = \\phi(W_{ih} x_t + W_{hh} h_{t-1} + b)$. "
        "Wraps nn.RNNCell. Connect the `hidden` output of one instance into "
        "the `hidden` input of the next to manually unroll the recurrence."
    )

    structural_params = ("input_size", "hidden_size", "nonlinearity", "seed")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input at this time step [batch, input_size].",
            ),
            PortDefinition(
                name="hidden",
                data_type=DataType.TENSOR,
                description="Hidden state from previous step [batch, hidden_size]. Defaults to zeros.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="hidden",
                data_type=DataType.TENSOR,
                description="New hidden state [batch, hidden_size].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="input_size",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Dimension of the input vector at each time step.",
            ),
            ParamDefinition(
                name="hidden_size",
                param_type=ParamType.INT,
                default=16,
                min_value=1,
                description="Dimension of the hidden state.",
            ),
            ParamDefinition(
                name="nonlinearity",
                param_type=ParamType.SELECT,
                default="tanh",
                options=["tanh", "relu"],
                description="Activation applied to the recurrence output.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the W_ih / W_hh / bias initialisation.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        nonlin = str(params.get("nonlinearity", "tanh"))
        if nonlin not in ("tanh", "relu"):
            raise ValueError(f"RNNCell: unknown nonlinearity {nonlin!r}")
        return _SeededRNNCell(
            input_size=int(params.get("input_size", 8)),
            hidden_size=int(params.get("hidden_size", 16)),
            nonlinearity=nonlin,
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
            raise ValueError("RNNCell requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        input_size = int(params.get("input_size", 8))
        hidden_size = int(params.get("hidden_size", 16))
        nonlin = str(params.get("nonlinearity", "tanh"))
        if nonlin not in ("tanh", "relu"):
            raise ValueError(f"RNNCell: unknown nonlinearity {nonlin!r}")

        if x.ndim != 2 or x.shape[-1] != input_size:
            raise ValueError(
                f"RNNCell: input shape {tuple(x.shape)} doesn't match "
                f"[batch, input_size={input_size}]."
            )

        h = inputs.get("hidden")
        if h is not None:
            if not isinstance(h, torch.Tensor):
                h = torch.as_tensor(h, dtype=torch.float32)
            h = h.float()
            if h.shape != (x.shape[0], hidden_size):
                raise ValueError(
                    f"RNNCell: hidden shape {tuple(h.shape)} doesn't match "
                    f"[batch={x.shape[0]}, hidden_size={hidden_size}]."
                )

        module = self.get_or_build_module(context, params)
        out = module(x, h)
        return {"hidden": out}
