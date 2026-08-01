"""StatsPercentileNode — the q-th percentile of a tensor, along a chosen axis.

Linear interpolation between order statistics, which is numpy's and pandas'
shared default, so a value from this node lines up with ``np.percentile`` and
``DataFrame.quantile`` on the same data.

NaN is skipped per series, as everywhere in this pack. That is deliberately
*not* what ``np.percentile`` does — it propagates NaN across the whole
reduction — and matches ``np.nanpercentile`` instead, minus the RuntimeWarning
that one raises on an all-NaN slice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._stats_core import (
    as_matrix,
    format_percentile,
    parse_percentiles,
    percentile_of,
    present,
    resolve_columns,
    to_tensor,
)

_AXES = ("all", "columns", "rows")


class StatsPercentileNode(BaseNode):
    NODE_NAME = "Stats-Percentile"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Percentiles of a tensor: one row per requested q. `all` reduces the "
        "whole tensor to a 1D result, `columns` gives a percentile per column, "
        "`rows` per row. NaN is skipped; interpolation is linear."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Values to take percentiles of. 1D or 2D [rows, columns].",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional column names, used when `axis` is `columns`.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="percentiles",
                data_type=DataType.TENSOR,
                description=(
                    "float32 [q] for axis `all`, otherwise [q, series] "
                    "(one column per described column or row)."
                ),
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Series names of the result, in result order.",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="One label per requested percentile, e.g. 25%, 50%, 75%.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="q",
                param_type=ParamType.STRING,
                default="50",
                description="Comma-separated percentiles in 0-100, e.g. '5,50,95'.",
            ),
            ParamDefinition(
                name="axis",
                param_type=ParamType.SELECT,
                default="all",
                options=list(_AXES),
                description=(
                    "`all` reduces every value to one number per q; `columns` "
                    "reduces down each column; `rows` reduces across each row."
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
        table = as_matrix(inputs.get("tensor"), node=self.NODE_NAME, port="tensor")

        axis = str(params.get("axis", "all") or "all")
        if axis not in _AXES:
            raise ValueError(
                f"{self.NODE_NAME}: unknown axis {axis!r}; expected one of {list(_AXES)}."
            )

        quantiles = sorted(set(parse_percentiles(
            params.get("q", "50"), node=self.NODE_NAME, param="q",
        )))
        if not quantiles:
            raise ValueError(
                f"{self.NODE_NAME}: `q` is empty — give at least one percentile, e.g. '50'."
            )

        if axis == "columns":
            work = table
            names = resolve_columns(inputs.get("columns"), table.shape[1])
        elif axis == "rows":
            work = table.T
            names = [f"r{i}" for i in range(table.shape[0])]
        else:
            work = table.reshape(-1, 1)
            names = ["all"]

        out = np.empty((len(quantiles), work.shape[1]), dtype=np.float64)
        for c in range(work.shape[1]):
            values = present(work[:, c])
            out[:, c] = [percentile_of(values, q) for q in quantiles]

        # `all` collapses to the 1D tensor the name promises; the axis forms
        # keep the [q, series] table shape so they feed TableView directly.
        result = out.reshape(-1) if axis == "all" else out
        return {
            "percentiles": to_tensor(result),
            "columns": list(names),
            "row_labels": [format_percentile(q) for q in quantiles],
        }
