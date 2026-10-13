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
sample std, ddof=1. A row whose group key is missing (NaN in a ``group_by``
column, ``None`` or NaN in ``keys``) belongs to no group, as with pandas'
default ``dropna=True``; the node says how many rows it left out.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from app.core.advisories import emit_advisory
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
    distinct_number_labels,
    max_of,
    mean_of,
    min_of,
    parse_names,
    present,
    resolve_columns,
    std_of,
    to_tensor,
)


logger = logging.getLogger(__name__)

GROUP_BY_NOTE_PREFIX = "[Stats-GroupByAggregate] "
NAN_KEY_WARNING_KIND = "stats_group_key_missing"


def _is_missing_key(value: Any) -> bool:
    """pandas drops a ``None`` or NaN group key; every other value is a key."""
    if value is None:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def _all_numeric(labels: list[str]) -> bool:
    try:
        for label in labels:
            float(label)
    except ValueError:
        return False
    return True


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
    DESCRIPTION = "One row per group: mean, sum, count, min, max or std"
    DETAILS = (
        "Group by the per-row `keys` list (CSVReader's `labels`) or by the table's "
        "own columns via `group_by`, which wins when both are given; a grouping "
        "column leaves the value set. `agg_overrides` takes `column=agg` pairs for "
        "exceptions. NaN is skipped as in pandas, so `count` is the number of "
        "present values, while the separate `counts` output holds each group's row "
        "count. Rows with a missing group key are left out and counted in the log."
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
        skipped = 0
        rows_kept: list[int] = []
        keys: list[tuple[Any, ...]] = []
        order_key: Any = None
        if group_by:
            key_columns = [self._resolve_column(token, names) for token in group_by]
            value_columns = [c for c in range(len(names)) if c not in set(key_columns)]
            for r in range(table.shape[0]):
                key = tuple(float(table[r, c]) for c in key_columns)
                # NaN != NaN, so a NaN key would open a new group on every
                # row and make sorted() return an unsorted order.
                if any(math.isnan(v) for v in key):
                    skipped += 1
                    continue
                keys.append(key)
                rows_kept.append(r)
            # Per key column: integers exact, other values with just enough
            # digits that two different keys never print the same.
            per_column = [
                distinct_number_labels([key[i] for key in keys])
                for i in range(len(key_columns))
            ]
            labels = [
                "|".join(column[n] for column in per_column) for n in range(len(keys))
            ]
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
            labels = []
            for r, k in enumerate(raw_keys):
                if _is_missing_key(k):
                    skipped += 1
                    continue
                labels.append(str(k))
                rows_kept.append(r)
            keys = [(label,) for label in labels]
            value_columns = list(range(len(names)))
            # Text keys that are all numbers sort as numbers (2 before 10),
            # matching the group_by path and pandas.
            if _all_numeric(labels):
                order_key = lambda key: (float(key[0]), key[0])  # noqa: E731

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
        for row, key, label in zip(rows_kept, keys, labels):
            members.setdefault(key, []).append(row)
            label_of.setdefault(key, label)
        ordered = sorted(members, key=order_key)

        out = np.empty((len(ordered), len(value_columns)), dtype=np.float64)
        counts = np.empty(len(ordered), dtype=np.float64)
        for r, key in enumerate(ordered):
            rows = members[key]
            counts[r] = float(len(rows))
            for c, column in enumerate(value_columns):
                how = overrides.get(names[column], default_agg)
                out[r, c] = _aggregate(table[rows, column], how)

        result: dict[str, Any] = {
            "table": to_tensor(out),
            "columns": [
                f"{names[c]} [{overrides.get(names[c], default_agg)}]"
                for c in value_columns
            ],
            "row_labels": [label_of[key] for key in ordered],
            "counts": to_tensor(counts),
        }
        if skipped:
            result["__log__"] = emit_advisory(
                f"{skipped} of {table.shape[0]} row(s) have a missing group key "
                "(NaN or None) and belong to no group, as in pandas' groupby.",
                kind=NAN_KEY_WARNING_KIND,
                prefix=GROUP_BY_NOTE_PREFIX,
                context=context,
                logger=logger,
            )
        return result

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
