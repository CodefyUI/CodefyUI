"""StatsDescribeNode — the one-node answer to "what is in this data?".

Reproduces ``pandas.DataFrame.describe()`` on the numeric half of a table:
count, mean, sample std (ddof=1), min, the requested percentiles, and max —
in that order and with pandas' row labels, so a reader who knows the pandas
output can read this one without a legend.

**NaN is missing, ±Inf is a value** — pandas' rule, pinned against the real
library in ``backend/tests/test_stats_describe_node.py``: ``count`` is the
non-NaN count, and a column holding an ``inf`` has an ``inf`` mean.

The node reports exactly the percentiles it was asked for, sorted and
de-duplicated. pandas ≤2.x silently appended the median to any percentile
list; pandas 3.0 removed that (it was deprecated in 2.2), and since the CI
matrix spans both — 3.10 resolves pandas 2.x, 3.11+ resolves 3.x — matching
the old behaviour would make this node's output depend on which pandas
happened to be installed next to it. It follows the surviving rule.
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
    max_of,
    mean_of,
    min_of,
    parse_percentiles,
    percentile_of,
    present,
    resolve_columns,
    std_of,
    to_tensor,
)

_AXES = ("columns", "rows", "all")


class StatsDescribeNode(BaseNode):
    NODE_NAME = "Stats-Describe"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Descriptive statistics for every column of a table: count, mean, std, "
        "min, percentiles and max, laid out exactly like pandas' describe(). "
        "Set `axis` to describe rows instead, or the table as a whole."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="2D [rows, columns] table, or a 1D vector treated as one column.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional column names, in table order (e.g. CSVReader's `columns`).",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="[statistics, columns] float32 table of the computed statistics.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Column names of the output table (the described columns).",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="Statistic names: count, mean, std, min, <percentiles>, max.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="axis",
                param_type=ParamType.SELECT,
                default="columns",
                options=list(_AXES),
                description=(
                    "`columns` describes each column (the pandas default), `rows` "
                    "describes each row, `all` describes every value as one series."
                ),
            ),
            ParamDefinition(
                name="percentiles",
                param_type=ParamType.STRING,
                default="25,50,75",
                description=(
                    "Comma-separated percentiles in 0-100. Reported exactly as "
                    "asked, sorted and de-duplicated."
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
        table = as_matrix(inputs.get("table"), node=self.NODE_NAME)

        axis = str(params.get("axis", "columns") or "columns")
        if axis not in _AXES:
            raise ValueError(
                f"{self.NODE_NAME}: unknown axis {axis!r}; expected one of {list(_AXES)}."
            )

        quantiles = sorted(set(parse_percentiles(
            params.get("percentiles", "25,50,75"),
            node=self.NODE_NAME,
            param="percentiles",
        )))

        if axis == "columns":
            work = table
            names = resolve_columns(inputs.get("columns"), table.shape[1])
        elif axis == "rows":
            work = table.T
            names = [f"r{i}" for i in range(table.shape[0])]
        else:
            work = table.reshape(-1, 1)
            names = ["all"]

        row_labels = (
            ["count", "mean", "std", "min"]
            + [format_percentile(q) for q in quantiles]
            + ["max"]
        )

        out = np.empty((len(row_labels), work.shape[1]), dtype=np.float64)
        for c in range(work.shape[1]):
            values = present(work[:, c])
            stats = [float(values.size), mean_of(values), std_of(values), min_of(values)]
            stats.extend(percentile_of(values, q) for q in quantiles)
            stats.append(max_of(values))
            out[:, c] = stats

        return {
            "table": to_tensor(out),
            "columns": list(names),
            "row_labels": row_labels,
        }
