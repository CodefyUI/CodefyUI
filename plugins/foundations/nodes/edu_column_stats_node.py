"""EduColumnStatsNode — column-wise descriptive statistics for tabular data.

Supports textbook lesson **C1-2 (表格資料)**: students start with a 2D tensor
``[rows, columns]`` and compute mean, std, min, max for each column.

Instead of one opaque "Mean" node, this expands the computation into named
steps for the Teaching Inspector:

    1. col_sum      = sum along rows
    2. means        = col_sum / row_count
    3. deviations²  = (x - means)²
    4. stds         = sqrt(mean(deviations²))           (population std)
    5. mins / maxs  = column-wise min and max

so a student can see exactly where each statistic comes from.
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

from ._params import param_bool


class EduColumnStatsNode(BaseNode):
    NODE_NAME = "Edu-ColumnStats"
    CATEGORY = "Classical"
    DESCRIPTION = "Per-column mean, std, min, max and row count"
    DETAILS = (
        "The standard deviation is the population std by default; set `unbiased` "
        "to divide by N-1 for the sample std instead (NaN for a single row). "
        "float64 and integer tables are computed in float64. With Verbose internals on it "
        "also records each intermediate step — column sum, divide by the row "
        "count, squared deviations, then the square root — so the Inspector can "
        "show where each statistic came from."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="2D tensor of shape [rows, columns].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="means", data_type=DataType.TENSOR,
                           description="Per-column mean, shape [columns]."),
            PortDefinition(name="stds", data_type=DataType.TENSOR,
                           description="Per-column std (population, or sample when unbiased), shape [columns]."),
            PortDefinition(name="mins", data_type=DataType.TENSOR,
                           description="Per-column min, shape [columns]."),
            PortDefinition(name="maxs", data_type=DataType.TENSOR,
                           description="Per-column max, shape [columns]."),
            PortDefinition(name="counts", data_type=DataType.TENSOR,
                           description="Row count repeated for each column, shape [columns]."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="unbiased",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "If true, divide variance by N-1 (sample std). Default is the "
                    "population std (N) which matches what a student computing "
                    "'sum of squared deviations / count' would write."
                ),
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
        x = inputs.get("table")
        if x is None:
            raise ValueError("Edu-ColumnStats requires a `table` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        # float64 and integer tables are summed in float64: float32 loses the
        # digits a std of large values lives in (1e7+1, 1e7+2, 1e7+3 gave
        # 1.29 where 0.82 is true). MPS has no float64, so float32 there.
        if x.dtype == torch.float64 or not (x.is_floating_point() or x.dtype == torch.bool):
            x = x.to(torch.float32 if x.device.type == "mps" else torch.float64)
        else:
            x = x.float()
        if x.ndim != 2:
            raise ValueError(
                f"Edu-ColumnStats expects a 2D [rows, columns] tensor; got shape {tuple(x.shape)}."
            )
        if x.shape[0] == 0:
            raise ValueError("Edu-ColumnStats: table has zero rows.")

        unbiased = param_bool(params, "unbiased", False, "Edu-ColumnStats")
        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        n_rows = float(x.shape[0])
        if recorder is not None:
            recorder.record(
                "input", "Start with the 2D table.",
                scalars={"rows": n_rows, "columns": float(x.shape[1])},
                table=x,
            )

        col_sum = x.sum(dim=0)
        if recorder is not None:
            recorder.record("col_sum", "Σ along rows for each column.", col_sum=col_sum)

        counts = torch.full_like(col_sum, n_rows)
        means = col_sum / counts
        if recorder is not None:
            recorder.record(
                "means", "Divide each column sum by the row count.",
                means=means, counts=counts,
            )

        deviations = x - means.unsqueeze(0)
        if recorder is not None:
            recorder.record("deviations", "x − μ for every entry.", deviations=deviations)

        squared = deviations * deviations
        # The sample std of a single row is undefined (0/0): NaN, as pandas
        # and numpy give, rather than a confident 0.
        denom = (n_rows - 1.0) if unbiased else n_rows
        if denom > 0:
            variances = squared.sum(dim=0) / denom
        else:
            variances = torch.full_like(col_sum, float("nan"))
        stds = variances.sqrt()
        if recorder is not None:
            recorder.record(
                "stds",
                f"Variance = mean(deviations²) (÷{denom:g}); std = √variance.",
                squared=squared, variances=variances, stds=stds,
                scalars={"denominator": denom},
            )

        mins = x.min(dim=0).values
        maxs = x.max(dim=0).values
        if recorder is not None:
            recorder.record(
                "range", "Per-column min and max for the range.",
                mins=mins, maxs=maxs,
            )

        result: dict[str, Any] = {
            "means": means,
            "stds": stds,
            "mins": mins,
            "maxs": maxs,
            "counts": counts,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
