"""StatsCorrelationNode — a Pearson or Spearman correlation matrix, plus a heatmap.

Matches ``pandas.DataFrame.corr`` on both counts:

* **Pairwise NaN handling.** With ``drop_nan`` on (the default, and pandas'
  default) each cell is computed from the rows where *both* of its columns are
  present, so one hole in column C does not shrink the A-B correlation. Turn it
  off to let NaN propagate the way ``numpy.corrcoef`` does.
* **Spearman is Pearson on ranks**, with tied values sharing their average
  rank — the same definition ``scipy.stats.rankdata(method="average")`` uses.

The heatmap is pinned to a -1..1 diverging scale rather than the data's own
extent: a matrix whose strongest off-diagonal correlation is 0.3 must not look
like one full of perfect correlations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.node_base import (
    MEDIA_CHART,
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._stats_core import as_matrix, heatmap_chart, resolve_columns, to_tensor

_METHODS = ("pearson", "spearman")

#: Below this many complete pairs a correlation is not defined (a single point
#: has no spread, so the denominator is zero).
_MIN_PAIRS = 2


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` ascending, ties sharing their average rank.

    Hand-rolled rather than pulled from scipy: the pack is numpy + torch only,
    and this is the one scipy function it would otherwise need.
    """
    n = values.size
    if n == 0:
        return values.astype(np.float64)
    # mergesort => stable, so equal values keep input order and the tie scan
    # below walks contiguous runs.
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ordered = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        # Ranks are 1-based, so the run [i..j] averages to (i+1 + j+1) / 2.
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r for two equal-length, already-cleaned vectors."""
    if x.size < _MIN_PAIRS:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(float(xc @ xc) * float(yc @ yc)))
    if denom == 0.0:
        # A constant column has no variance, so no correlation is defined.
        return float("nan")
    return float(float(xc @ yc) / denom)


class StatsCorrelationNode(BaseNode):
    NODE_NAME = "Stats-Correlation"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Correlation matrix across a table's columns, Pearson or Spearman. "
        "NaN is handled pairwise by default, so one gap does not shrink every "
        "other pair. Outputs the matrix plus a -1..1 heatmap."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="2D [rows, columns] table to correlate column-wise.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional column names, in table order.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="matrix",
                data_type=DataType.TENSOR,
                description="float32 [columns, columns] correlation matrix.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Column names, matching both axes of the matrix.",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="Same names as `columns` — the matrix is symmetric.",
            ),
            PortDefinition(
                name="chart",
                data_type=DataType.ANY,
                description="Heatmap spec for Stats-ChartView; also rendered directly.",
                media=MEDIA_CHART,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="method",
                param_type=ParamType.SELECT,
                default="pearson",
                options=list(_METHODS),
                description=(
                    "`pearson` measures a linear relationship; `spearman` ranks "
                    "first, so it catches any monotonic one and shrugs off outliers."
                ),
            ),
            ParamDefinition(
                name="drop_nan",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Compute each pair from the rows where both columns are present "
                    "(pandas' default). Off lets NaN propagate into the result."
                ),
            ),
            ParamDefinition(
                name="title",
                param_type=ParamType.STRING,
                default="Correlation",
                description="Chart title.",
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
        table = as_matrix(inputs.get("table"), node=self.NODE_NAME)
        names = resolve_columns(inputs.get("columns"), table.shape[1])

        method = str(params.get("method", "pearson") or "pearson")
        if method not in _METHODS:
            raise ValueError(
                f"{self.NODE_NAME}: unknown method {method!r}; "
                f"expected one of {list(_METHODS)}."
            )
        drop_nan = bool(params.get("drop_nan", True))

        n_cols = table.shape[1]
        matrix = np.full((n_cols, n_cols), float("nan"), dtype=np.float64)
        missing = np.isnan(table)

        for i in range(n_cols):
            for j in range(i, n_cols):
                if drop_nan:
                    keep = ~(missing[:, i] | missing[:, j])
                    x = table[keep, i]
                    y = table[keep, j]
                else:
                    x = table[:, i]
                    y = table[:, j]
                if method == "spearman" and not (
                    np.isnan(x).any() or np.isnan(y).any()
                ):
                    x = average_ranks(x)
                    y = average_ranks(y)
                # A NaN left in place here is deliberate: ranking would give
                # it a finite rank and turn "undefined" into a number, so with
                # drop_nan off Spearman propagates exactly as Pearson does.
                r = _pearson(x, y)
                matrix[i, j] = r
                matrix[j, i] = r

        chart = heatmap_chart(
            matrix,
            row_labels=names,
            col_labels=names,
            colormap="RdBu",
            title=str(params.get("title", "Correlation") or "Correlation"),
            # Read against a fixed diverging scale, never the data's own extent.
            vmin=-1.0,
            vmax=1.0,
        )

        return {
            "matrix": to_tensor(matrix),
            "columns": list(names),
            "row_labels": list(names),
            "chart": chart,
        }
