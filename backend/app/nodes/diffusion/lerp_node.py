"""LerpNode — element-wise linear interpolation: ``α a + (1-α) b``.

The forward diffusion equation
    x_t = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε
is *almost* a lerp; specifically it's an interpolation with an unusual
weighting that ensures the variance of x_t matches x_0's. For teaching
purposes a plain lerp captures the spirit — students see "x_t is a
weighted blend of clean signal and noise" before we introduce the exact
sqrt-based scaling.

``alpha`` accepts a scalar param, a 0-d tensor, or a broadcastable tensor
input — the last form lets a batch share one tensor of per-sample
alphas, which is how schedulers feed the diffusion equation in practice.
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class LerpNode(BaseNode):
    NODE_NAME = "Lerp"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Linear interpolation: $\\alpha\\,a + (1-\\alpha)\\,b$. "
        "When $\\alpha=1$ the output is $a$; when $\\alpha=0$ it's $b$. "
        "Use as a teaching stand-in for the diffusion forward equation."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor_a",
                data_type=DataType.TENSOR,
                description="First tensor (weighted by alpha).",
            ),
            PortDefinition(
                name="tensor_b",
                data_type=DataType.TENSOR,
                description="Second tensor (weighted by 1 - alpha).",
            ),
            PortDefinition(
                name="alpha",
                data_type=DataType.TENSOR,
                description="Optional scalar or broadcastable tensor — overrides the `alpha` param when connected.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Interpolated tensor, shape determined by broadcasting.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="alpha",
                param_type=ParamType.FLOAT,
                default=0.5,
                description="Interpolation weight (0..1). Used only when the `alpha` input is not connected.",
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
        a = inputs.get("tensor_a")
        b = inputs.get("tensor_b")
        if a is None or b is None:
            raise ValueError("Lerp requires both `tensor_a` and `tensor_b` inputs.")
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=torch.float32)

        alpha_in = inputs.get("alpha")
        if alpha_in is not None:
            alpha = alpha_in if isinstance(alpha_in, torch.Tensor) else torch.as_tensor(alpha_in, dtype=torch.float32)
        else:
            alpha = torch.as_tensor(float(params.get("alpha", 0.5)), dtype=a.dtype)

        out = alpha * a + (1.0 - alpha) * b
        return {"tensor": out}
