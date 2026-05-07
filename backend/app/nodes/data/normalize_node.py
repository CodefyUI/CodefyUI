"""NormalizeNode — feature scaling for tabular data.

Three standard recipes covered:

* ``zscore`` — subtract mean, divide by std. Most common for ML where
  features should be on comparable scales without bounded range.
* ``minmax`` — subtract min, divide by (max - min). Output lives in
  [0, 1]. Useful when the model expects bounded inputs.
* ``unit_norm`` — divide by row L2-norm. Used when only the *direction*
  of the feature vector matters (cosine similarity, attention).

The ``stats`` output exposes the per-column / per-row statistics that
were used so students can verify the normalisation step concretely.
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


class NormalizeNode(BaseNode):
    NODE_NAME = "Normalize"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Scale a tensor along the chosen axis. zscore = $(x-\\mu)/\\sigma$, "
        "minmax = $(x-\\min)/(\\max-\\min)$, unit_norm = $x/\\|x\\|_2$. "
        "Use axis=0 for per-column tabular normalisation, axis=1 for "
        "per-row sample normalisation."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input tensor; the normalisation runs along `axis`.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Normalised tensor, same shape as input.",
            ),
            PortDefinition(
                name="stats",
                data_type=DataType.LIST,
                description="The per-column statistics used (mean/std for zscore, min/max for minmax, etc.).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="zscore",
                options=["zscore", "minmax", "unit_norm"],
                description="Normalisation recipe.",
            ),
            ParamDefinition(
                name="axis",
                param_type=ParamType.INT,
                default=0,
                description="Axis along which to compute statistics. 0 = per-column, 1 = per-row.",
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
        x = inputs.get("tensor")
        if x is None:
            raise ValueError("Normalize requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        mode = str(params.get("mode", "zscore"))
        axis = int(params.get("axis", 0))

        stats: dict[str, list[float]] = {}

        if mode == "zscore":
            mean = x.mean(dim=axis, keepdim=True)
            std = x.std(dim=axis, unbiased=False, keepdim=True)
            # Constant columns have std=0 — replace with 1 so the centred value
            # passes through as zero rather than NaN.
            safe_std = torch.where(std == 0, torch.ones_like(std), std)
            out = (x - mean) / safe_std
            stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}
        elif mode == "minmax":
            mn = x.min(dim=axis, keepdim=True).values
            mx = x.max(dim=axis, keepdim=True).values
            rng = mx - mn
            safe_rng = torch.where(rng == 0, torch.ones_like(rng), rng)
            out = (x - mn) / safe_rng
            stats = {"min": mn.squeeze().tolist(), "max": mx.squeeze().tolist()}
        elif mode == "unit_norm":
            norms = x.norm(dim=axis, keepdim=True)
            safe_norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            out = x / safe_norms
            stats = {"norm": norms.squeeze().tolist()}
        else:
            raise ValueError(f"Unknown Normalize mode: {mode!r}")

        return {"tensor": out, "stats": stats}
