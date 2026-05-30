"""EduLayerNormNode — layer normalization written out step by step.

This is the educational counterpart to the production ``LayerNorm`` node.
Where that one wraps the framework primitive, this one exposes the four
pieces of one layer-norm over the last dimension ``D`` so students can watch
the statistics get computed and applied (textbook lesson **I4-3**):

    mean = x.mean(-1, keepdim=True)                 # per-row average
    var  = x.var(-1, unbiased=False, keepdim=True)  # biased (population) var
    xhat = (x - mean) / sqrt(var + eps)             # standardise to ~N(0,1)
    y    = gamma * xhat + beta                       # learnable affine

The normalization is always over the **last** dimension; everything to its
left (batch, sequence, …) is treated as independent rows. ``mean`` and
``var`` are surfaced as ``[*, 1]`` display-only outputs so the inspector can
show the statistics that drove the standardisation.
"""

from __future__ import annotations

from typing import Any

import torch

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduLayerNormNode(BaseNode):
    NODE_NAME = "Edu-LayerNorm"
    CATEGORY = "Transformer"
    DESCRIPTION = (
        "Layer normalization over the last dim D, stepped out: mean = x.mean(-1); "
        "var = x.var(-1) (biased); xhat = (x - mean) / sqrt(var + eps); "
        "y = gamma * xhat + beta. Exposes mean and var as display-only outputs so "
        "students see the statistics that standardise each row before the affine."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x",
                data_type=DataType.TENSOR,
                description="Input tensor of shape [*, D]; normalized over the last dim D.",
            ),
            PortDefinition(
                name="gamma",
                data_type=DataType.TENSOR,
                description="Optional affine scale of shape [D]. Defaults to ones.",
                optional=True,
            ),
            PortDefinition(
                name="beta",
                data_type=DataType.TENSOR,
                description="Optional affine shift of shape [D]. Defaults to zeros.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="y",
                data_type=DataType.TENSOR,
                description="Normalized (and optionally affine-transformed) output, same shape as x.",
            ),
            PortDefinition(
                name="mean",
                data_type=DataType.TENSOR,
                description="Per-row mean over the last dim, shape [*, 1]. Display-only.",
            ),
            PortDefinition(
                name="var",
                data_type=DataType.TENSOR,
                description="Per-row biased variance over the last dim, shape [*, 1]. Display-only.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="normalized_dim",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Size D of the normalized last dimension. 0 means infer D from "
                    "x's last dim; any other value is checked against x and gamma/beta."
                ),
            ),
            ParamDefinition(
                name="eps",
                param_type=ParamType.FLOAT,
                default=1e-5,
                min_value=0.0,
                description="Added to the variance before sqrt for numerical stability.",
            ),
            ParamDefinition(
                name="elementwise_affine",
                param_type=ParamType.BOOL,
                default=True,
                description="If true apply y = gamma * xhat + beta; if false output xhat directly.",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x = inputs.get("x")
        if x is None:
            raise ValueError("EduLayerNorm requires an `x` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        if x.ndim < 1:
            raise ValueError(
                f"EduLayerNorm: x must be at least 1-D; got shape {tuple(x.shape)}."
            )

        d = x.shape[-1]
        normalized_dim = int(params.get("normalized_dim", 0) or 0)
        if normalized_dim and normalized_dim != d:
            raise ValueError(
                f"EduLayerNorm: normalized_dim={normalized_dim} does not match x's "
                f"last dim {d}. Set normalized_dim=0 to infer it from x."
            )

        eps = float(params.get("eps", 1e-5))
        elementwise_affine = bool(params.get("elementwise_affine", True))

        gamma = self._coerce_affine(inputs.get("gamma"), d, name="gamma")
        beta = self._coerce_affine(inputs.get("beta"), d, name="beta")

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        mean = x.mean(dim=-1, keepdim=True)
        if recorder is not None:
            recorder.record(
                "mean",
                "Per-row mean over the last dim: $\\mu = \\frac{1}{D}\\sum_i x_i$.",
                scalars={"D": float(d), "eps": eps},
                mean=mean,
            )

        var = x.var(dim=-1, unbiased=False, keepdim=True)
        if recorder is not None:
            recorder.record(
                "var",
                "Per-row biased variance over the last dim: "
                "$\\sigma^2 = \\frac{1}{D}\\sum_i (x_i - \\mu)^2$.",
                var=var,
            )

        xhat = (x - mean) / torch.sqrt(var + eps)
        if recorder is not None:
            recorder.record(
                "normalize",
                "Standardise each row: $\\hat{x} = (x - \\mu) / \\sqrt{\\sigma^2 + \\epsilon}$.",
                xhat=xhat,
            )

        if elementwise_affine:
            y = gamma * xhat + beta
        else:
            y = xhat
        if recorder is not None:
            recorder.record(
                "affine",
                "Apply the learnable affine: $y = \\gamma \\hat{x} + \\beta$."
                if elementwise_affine
                else "elementwise_affine is off, so $y = \\hat{x}$.",
                y=y,
            )

        result: dict[str, Any] = {"y": y, "mean": mean, "var": var}
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _coerce_affine(value: Any, d: int, *, name: str) -> torch.Tensor:
        """Validate / default an affine parameter (gamma or beta) to shape [D]."""
        if value is None:
            return torch.ones(d) if name == "gamma" else torch.zeros(d)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        value = value.float()
        if value.ndim != 1 or value.shape[0] != d:
            raise ValueError(
                f"EduLayerNorm: {name} must have shape [{d}] to match x's last dim; "
                f"got {tuple(value.shape)}."
            )
        return value
