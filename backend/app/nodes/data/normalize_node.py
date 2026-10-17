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

Wiring that ``stats`` dict into a second Normalize's ``stats`` input makes
the second node apply those statistics instead of fitting its own. That is
how a test split is scaled with the training split's mean and std, so no
statistic of the test rows leaks into training.
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
    DESCRIPTION = "Normalise a tensor along one axis; outputs the stats"
    DETAILS = (
        "zscore = $(x-\\mu)/\\sigma$, minmax = $(x-\\min)/(\\max-\\min)$, unit_norm = "
        "$x/\\|x\\|_2$. axis=0 computes per column, axis=1 per row. A constant "
        "column is divided by 1 rather than 0, so it comes out as zeros instead of "
        "NaN. With `stats` wired from another Normalize, those statistics are "
        "applied as-is and nothing is fitted: wire the training split's stats "
        "into the Normalize of the test split. The recipe then follows the stats "
        "and `mode` is ignored. Per-row statistics (axis 1) belong to the rows "
        "they came from, so with axis 1 the wired stats only choose the recipe "
        "and each incoming row is normalised with its own statistics. Both "
        "nodes must use the same axis."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input tensor; the normalisation runs along `axis`.",
            ),
            PortDefinition(
                name="stats",
                data_type=DataType.ANY,
                description=(
                    "Optional `stats` output of another Normalize (e.g. fitted on "
                    "the training split). When wired, those statistics are applied "
                    "instead of fitting new ones."
                ),
                optional=True,
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
                data_type=DataType.ANY,
                description=(
                    "Dict of the statistics used (mean/std for zscore, min/max for "
                    "minmax, norm for unit_norm); wire it into another Normalize's "
                    "`stats` input to reuse them."
                ),
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
        try:
            axis = int(params.get("axis", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Normalize: axis must be a whole number, got {params.get('axis')!r}."
            ) from exc
        if x.dim() == 0 or not -x.dim() <= axis < x.dim():
            raise ValueError(
                f"Normalize: axis={axis} is out of range for a {x.dim()}-D tensor "
                f"of shape {tuple(x.shape)}. Use 0 for per-column statistics or "
                f"1 for per-row."
            )

        fitted = inputs.get("stats")
        if fitted is not None:
            return self._apply(x, fitted, axis)

        out, stats = self._fit(x, mode, axis)
        return {"tensor": out, "stats": stats}

    @staticmethod
    def _fit(x: torch.Tensor, mode: str, axis: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Fit *mode*'s statistics along *axis* and apply them."""
        stats: dict[str, Any]
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
            raise ValueError(
                f"Normalize: unknown mode {mode!r}; use zscore, minmax or unit_norm."
            )
        # Recorded so a Normalize that applies these stats can tell per-row
        # statistics from per-column ones (axis 0, reusable on other rows).
        # Axis 0 stays implicit, so its stats dict is the one it always was.
        if axis % x.dim() != 0:
            stats["axis"] = axis
        return out, stats

    @staticmethod
    def _apply(x: torch.Tensor, stats: Any, axis: int) -> dict[str, Any]:
        """Normalise ``x`` with statistics fitted elsewhere (no refit)."""
        if not isinstance(stats, dict):
            raise ValueError(
                f"Normalize: the `stats` input must be the stats dict of another "
                f"Normalize, got {type(stats).__name__}."
            )
        fitted_axis = stats.get("axis", 0)
        if int(fitted_axis) % x.dim() != axis % x.dim():
            raise ValueError(
                f"Normalize: the `stats` input was fitted with axis={fitted_axis}, "
                f"but this node has axis={axis}. Set both Normalize nodes to the "
                f"same axis."
            )
        if axis % x.dim() != 0:
            # Per-row statistics describe the rows they were fitted on, so
            # training row i's mean has nothing to say about test row i. The
            # recipe carries over; the statistics are recomputed on these rows.
            if "mean" in stats and "std" in stats:
                recipe = "zscore"
            elif "min" in stats and "max" in stats:
                recipe = "minmax"
            elif "norm" in stats:
                recipe = "unit_norm"
            else:
                raise ValueError(
                    f"Normalize: the `stats` input has keys {sorted(stats)}; expected "
                    f"mean/std, min/max or norm from another Normalize."
                )
            out, own = NormalizeNode._fit(x, recipe, axis)
            return {"tensor": out, "stats": own}

        # Statistics were reduced along `axis` with keepdim and then squeezed;
        # restore the keepdim shape so they broadcast against `x`.
        shape = list(x.shape)
        shape[axis] = 1

        def stat(key: str) -> torch.Tensor:
            # The stats are plain lists (made with .tolist()), so as_tensor
            # puts them on the CPU; on a GPU run ``x`` is on the GPU (#808).
            t = torch.as_tensor(stats[key], dtype=torch.float32, device=x.device)
            expected = 1
            for d in shape:
                expected *= d
            if t.numel() != expected:
                raise ValueError(
                    f"Normalize: stats[{key!r}] has {t.numel()} value(s), but a "
                    f"tensor of shape {tuple(x.shape)} along axis={axis} needs "
                    f"{expected}. Wire stats fitted on data with the same columns "
                    f"and use the same axis."
                )
            return t.reshape(shape)

        if "mean" in stats and "std" in stats:
            mean, std = stat("mean"), stat("std")
            out = (x - mean) / torch.where(std == 0, torch.ones_like(std), std)
        elif "min" in stats and "max" in stats:
            mn, mx = stat("min"), stat("max")
            rng = mx - mn
            out = (x - mn) / torch.where(rng == 0, torch.ones_like(rng), rng)
        elif "norm" in stats:
            norms = stat("norm")
            out = x / torch.where(norms == 0, torch.ones_like(norms), norms)
        else:
            raise ValueError(
                f"Normalize: the `stats` input has keys {sorted(stats)}; expected "
                f"mean/std, min/max or norm from another Normalize."
            )
        return {"tensor": out, "stats": stats}
