"""StatsHistogramNode — bin a tensor and hand back both a table and a chart.

Wraps ``numpy.histogram`` with the two knobs that matter in practice: a fixed
range (so two runs are comparable frame to frame) and ``density`` (so a
histogram of 10 000 samples is comparable with one of 100).

Non-finite values are dropped before binning and counted in ``dropped``.
numpy would otherwise refuse the auto range outright — "autodetected range of
[nan, nan] is not finite" — and, given an explicit range, would silently drop
them anyway because every comparison against a NaN is False. Reporting the
number is the difference between a quiet lie and a fact the reader can check.
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

from ._stats_core import as_vector, bar_chart, float_param, int_param, to_tensor

_RANGE_MODES = ("auto", "manual")


class StatsHistogramNode(BaseNode):
    NODE_NAME = "Stats-Histogram"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Bin a tensor into a histogram. Outputs a [bins, 3] table of "
        "(bin_start, bin_end, count) plus a bar chart that renders in the "
        "Results panel. Set `density` for a probability density instead of counts."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Values to bin. Any shape; flattened first.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="float32 [bins, 3]: bin_start, bin_end, count (or density).",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="['bin_start', 'bin_end', 'count'] — or 'density'.",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="One '[lo, hi)' label per bin.",
            ),
            PortDefinition(
                name="chart",
                data_type=DataType.ANY,
                description="Bar-chart spec for Stats-ChartView; also rendered directly.",
                media=MEDIA_CHART,
            ),
            PortDefinition(
                name="dropped",
                data_type=DataType.SCALAR,
                description="How many non-finite values (NaN / ±Inf) were excluded.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="bins",
                param_type=ParamType.INT,
                default=20,
                min_value=1,
                max_value=1000,
                description="Number of equal-width bins.",
            ),
            ParamDefinition(
                name="range_mode",
                param_type=ParamType.SELECT,
                default="auto",
                options=list(_RANGE_MODES),
                description=(
                    "`auto` spans the data's own min..max; `manual` pins the range "
                    "so successive runs are comparable."
                ),
            ),
            ParamDefinition(
                name="range_min",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Low edge of the first bin (manual range only).",
                visible_when={"range_mode": "manual"},
            ),
            ParamDefinition(
                name="range_max",
                param_type=ParamType.FLOAT,
                default=1.0,
                description="High edge of the last bin (manual range only).",
                visible_when={"range_mode": "manual"},
            ),
            ParamDefinition(
                name="density",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Report a probability density (bin areas sum to 1) instead of "
                    "raw counts."
                ),
            ),
            ParamDefinition(
                name="title",
                param_type=ParamType.STRING,
                default="Histogram",
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
        values = as_vector(inputs.get("tensor"), node=self.NODE_NAME, port="tensor")
        finite = values[np.isfinite(values)]
        dropped = int(values.size - finite.size)

        bins = int_param(params, "bins", 20, node=self.NODE_NAME)
        if bins < 1:
            raise ValueError(f"{self.NODE_NAME}: `bins` must be at least 1; got {bins}.")

        density = bool(params.get("density", False))
        mode = str(params.get("range_mode", "auto") or "auto")
        if mode not in _RANGE_MODES:
            raise ValueError(
                f"{self.NODE_NAME}: unknown range_mode {mode!r}; "
                f"expected one of {list(_RANGE_MODES)}."
            )

        if mode == "manual":
            lo = float_param(params, "range_min", 0.0, node=self.NODE_NAME)
            hi = float_param(params, "range_max", 1.0, node=self.NODE_NAME)
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                raise ValueError(
                    f"{self.NODE_NAME}: manual range needs finite range_min < range_max; "
                    f"got [{lo}, {hi}]."
                )
            span = (lo, hi)
        elif finite.size == 0:
            # numpy cannot autodetect a range from nothing. A unit range keeps
            # the output well-formed (all-zero counts) instead of raising on a
            # graph whose upstream simply produced no data yet.
            span = (0.0, 1.0)
        else:
            span = None

        if span is None:
            counts, edges = np.histogram(finite, bins=bins, density=density)
        else:
            counts, edges = np.histogram(
                finite, bins=bins, range=span, density=density
            )

        value_name = "density" if density else "count"
        table = np.stack(
            [edges[:-1], edges[1:], counts.astype(np.float64)], axis=1
        )
        labels = [f"[{edges[i]:g}, {edges[i + 1]:g})" for i in range(len(counts))]

        chart = bar_chart(
            labels,
            counts,
            title=str(params.get("title", "Histogram") or "Histogram"),
            x_label="bin",
            y_label=value_name,
        )
        if dropped:
            chart["note"] = f"{dropped} non-finite value(s) excluded"

        return {
            "table": to_tensor(table),
            "columns": ["bin_start", "bin_end", value_name],
            "row_labels": labels,
            "chart": chart,
            "dropped": dropped,
        }
