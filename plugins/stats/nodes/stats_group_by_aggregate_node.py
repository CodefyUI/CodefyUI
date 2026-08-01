"""StatsGroupByAggregateNode — one row per group, one aggregate per column.

Two ways to say what a group is, because CodefyUI tables carry their
categorical column *beside* the numeric matrix rather than inside it:

* connect ``keys`` — a per-row label list, which is exactly what
  ``CSVReader``'s ``labels`` output is; or
* set ``group_by`` to one or more of the table's own columns, by name or by
  index, for data that is already numerically coded.

``group_by`` wins when both are given, so a graph that leaves a ``keys`` edge
connected while experimenting with a column grouping gets the thing it just
typed rather than the thing it forgot to unplug.

Aggregates follow pandas' ``groupby`` semantics: NaN is skipped, ``count`` is
the number of *present* values in that column and group (which is why the
separate ``counts`` output carries the group's row count), and ``std`` is the
sample std, ddof=1.
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
    AGGREGATIONS,
    as_matrix,
    max_of,
    mean_of,
    min_of,
    parse_names,
    present,
    resolve_columns,
    std_of,
    to_tensor,
)


def _aggregate(values: np.ndarray, how: str) -> float:
    present_values = present(values)
    if how == "count":
        return float(present_values.size)
    if how == "sum":
        # pandas sums an all-NaN group to 0.0, not NaN.
        return float(present_values.sum()) if present_values.size else 0.0
    if how == "mean":
        return mean_of(present_values)
    if how == "min":
        return min_of(present_values)
    if how == "max":
        return max_of(present_values)
    return std_of(present_values)


class StatsGroupByAggregateNode(BaseNode):
    NODE_NAME = "Stats-GroupByAggregate"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Collapse a table to one row per group. Group by a per-row label list "
        "on `keys` (e.g. CSVReader's labels) or by the table's own columns via "
        "`group_by`, then aggregate every remaining column with mean / sum / "
        "count / min / max / std."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="2D [rows, columns] table of values to aggregate.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional column names, in table order.",
                optional=True,
            ),
            PortDefinition(
                name="keys",
                data_type=DataType.LIST,
                description=(
                    "Optional per-row group label, one entry per table row "
                    "(e.g. CSVReader's `labels`)."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="float32 [groups, aggregated columns].",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Aggregated column names, each suffixed with its aggregate.",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="One group key per row, in output order.",
            ),
            PortDefinition(
                name="counts",
                data_type=DataType.TENSOR,
                description="float32 [groups] — how many table rows fell in each group.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="group_by",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Comma-separated column names or indices to group by. Those "
                    "columns leave the value set. Empty means group by the `keys` input."
                ),
            ),
            ParamDefinition(
                name="agg",
                param_type=ParamType.SELECT,
                default="mean",
                options=list(AGGREGATIONS),
                description="Aggregate applied to every value column.",
            ),
            ParamDefinition(
                name="agg_overrides",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Per-column exceptions as 'column=agg' pairs, comma-separated — "
                    "e.g. 'price=max, qty=sum'."
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
        names = resolve_columns(inputs.get("columns"), table.shape[1])

        default_agg = str(params.get("agg", "mean") or "mean")
        if default_agg not in AGGREGATIONS:
            raise ValueError(
                f"{self.NODE_NAME}: unknown agg {default_agg!r}; "
                f"expected one of {list(AGGREGATIONS)}."
            )
        overrides = self._parse_overrides(params.get("agg_overrides", ""), names)

        group_by = parse_names(params.get("group_by", ""))
        if group_by:
            key_columns = [self._resolve_column(token, names) for token in group_by]
            value_columns = [c for c in range(len(names)) if c not in set(key_columns)]
            keys = [
                tuple(float(table[r, c]) for c in key_columns)
                for r in range(table.shape[0])
            ]
            labels = ["|".join(f"{v:g}" for v in key) for key in keys]
        else:
            raw_keys = inputs.get("keys")
            if not isinstance(raw_keys, (list, tuple)):
                raise ValueError(
                    f"{self.NODE_NAME}: nothing to group by — connect a per-row "
                    "`keys` list or set the `group_by` param to one of the table's "
                    "columns."
                )
            if len(raw_keys) != table.shape[0]:
                raise ValueError(
                    f"{self.NODE_NAME}: `keys` has {len(raw_keys)} entries but the "
                    f"table has {table.shape[0]} rows — they must be aligned."
                )
            labels = [str(k) for k in raw_keys]
            keys = [(label,) for label in labels]
            value_columns = list(range(len(names)))

        if not value_columns:
            raise ValueError(
                f"{self.NODE_NAME}: every column is a grouping column, so there is "
                "nothing left to aggregate."
            )

        # Ordered by key so a re-run of the same graph produces the same rows.
        # Sorting the tuples, not the rendered labels, keeps numeric groups in
        # numeric order (10 after 2, not before it).
        members: dict[Any, list[int]] = {}
        label_of: dict[Any, str] = {}
        for row, (key, label) in enumerate(zip(keys, labels)):
            members.setdefault(key, []).append(row)
            label_of.setdefault(key, label)
        ordered = sorted(members)

        out = np.empty((len(ordered), len(value_columns)), dtype=np.float64)
        counts = np.empty(len(ordered), dtype=np.float64)
        for r, key in enumerate(ordered):
            rows = members[key]
            counts[r] = float(len(rows))
            for c, column in enumerate(value_columns):
                how = overrides.get(names[column], default_agg)
                out[r, c] = _aggregate(table[rows, column], how)

        return {
            "table": to_tensor(out),
            "columns": [
                f"{names[c]} [{overrides.get(names[c], default_agg)}]"
                for c in value_columns
            ],
            "row_labels": [label_of[key] for key in ordered],
            "counts": to_tensor(counts),
        }

    def _parse_overrides(self, raw: Any, names: list[str]) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for pair in parse_names(raw):
            column, sep, how = pair.partition("=")
            column, how = column.strip(), how.strip()
            if not sep or not column or not how:
                raise ValueError(
                    f"{self.NODE_NAME}: `agg_overrides` entry {pair!r} is not a "
                    "'column=agg' pair."
                )
            if how not in AGGREGATIONS:
                raise ValueError(
                    f"{self.NODE_NAME}: `agg_overrides` names unknown aggregate "
                    f"{how!r}; expected one of {list(AGGREGATIONS)}."
                )
            if column not in names:
                raise ValueError(
                    f"{self.NODE_NAME}: `agg_overrides` names unknown column "
                    f"{column!r}; the table has {names}."
                )
            overrides[column] = how
        return overrides

    def _resolve_column(self, token: str, names: list[str]) -> int:
        if token in names:
            return names.index(token)
        try:
            index = int(token)
        except ValueError:
            raise ValueError(
                f"{self.NODE_NAME}: `group_by` names unknown column {token!r}; "
                f"the table has {names}."
            ) from None
        if not 0 <= index < len(names):
            raise ValueError(
                f"{self.NODE_NAME}: `group_by` index {index} is out of range for a "
                f"table with {len(names)} column(s)."
            )
        return index
