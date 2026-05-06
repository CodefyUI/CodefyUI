"""EduFFNNode — toy feed-forward block for transformer demos.

Inside a transformer block, after attention has mixed information across
positions, a position-wise FFN expands each token's vector into a wider
hidden space, applies a non-linearity, and projects back. The classical
"4× expansion" pattern (d → 4d → d) gives the model capacity to combine
the now-mixed information.

This educational variant keeps everything small (default ``embed_dim=8``,
``hidden_dim=16``) and exposes the post-activation hidden state as a
separate output so a future viz can show "what each neuron lit up for".

Two modes for the non-linearity: ``relu`` and ``gelu``. Real GPT-2 used
GELU; the simpler ReLU is easier for first-time readers.
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
from ...core.step_trace import StepRecorder


class _EduFFNModule(nn.Module):
    """Two linear layers with a tap point for the post-activation hidden state."""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: str, seed: int) -> None:
        super().__init__()
        # Local generator so we don't perturb the global RNG; gives reproducible
        # weights across runs with the same seed.
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=gen) * (1.0 / max(1, layer.weight.shape[1]) ** 0.5)
                )
                layer.bias.zero_()
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc1(x)
        if self.activation_name == "relu":
            a = F.relu(h)
        elif self.activation_name == "gelu":
            a = F.gelu(h)
        else:
            raise ValueError(f"Unknown EduFFN activation: {self.activation_name!r}")
        out = self.fc2(a)
        return out, a


class EduFFNNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "EduFFN"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Toy feed-forward block: x → Linear(D, hidden) → activation → Linear(hidden, D). "
        "Defaults are tiny (D=8, hidden=16) for inspectability. Exposes the "
        "post-activation hidden state as a separate output."
    )

    structural_params = ("embed_dim", "hidden_dim", "activation", "seed")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input of shape [..., embed_dim].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Output, same shape as input.",
            ),
            PortDefinition(
                name="activations",
                data_type=DataType.TENSOR,
                description="Post-activation hidden state of shape [..., hidden_dim] — useful for visualisation.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="embed_dim",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Input/output dimension. Must match the upstream tensor's last dim.",
            ),
            ParamDefinition(
                name="hidden_dim",
                param_type=ParamType.INT,
                default=16,
                min_value=1,
                description="Hidden dimension (typically 4×embed_dim in production).",
            ),
            ParamDefinition(
                name="activation",
                param_type=ParamType.SELECT,
                default="relu",
                options=["relu", "gelu"],
                description="Non-linearity between the two linear layers.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the linear-layer initialisation. Same seed → same weights.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        return _EduFFNModule(
            embed_dim=int(params.get("embed_dim", 8)),
            hidden_dim=int(params.get("hidden_dim", 16)),
            activation=str(params.get("activation", "relu")),
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
            raise ValueError("EduFFN requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        embed_dim = int(params.get("embed_dim", 8))
        if x.shape[-1] != embed_dim:
            raise ValueError(
                f"EduFFN: input last dim {x.shape[-1]} does not match embed_dim={embed_dim}."
            )

        activation = str(params.get("activation", "relu"))
        if activation not in ("relu", "gelu"):
            raise ValueError(f"Unknown EduFFN activation: {activation!r}")

        module = self.get_or_build_module(context, params)
        out, activations = module(x)

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "input",
                f"Input of shape {tuple(x.shape)}.",
                tensor=x,
            )
            recorder.record(
                "linear1",
                f"Project to hidden space: x @ W1.T + b1, hidden dim = {params.get('hidden_dim', 16)}.",
                hidden=module.fc1(x),
            )
            recorder.record(
                "activation",
                f"Apply {activation} non-linearity element-wise.",
                activations=activations,
            )
            recorder.record(
                "linear2",
                "Project back to embed_dim: a @ W2.T + b2.",
                output=out,
            )
            return {"tensor": out, "activations": activations, "__steps__": recorder.steps}

        return {"tensor": out, "activations": activations}
