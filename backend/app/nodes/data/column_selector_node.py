"""ColumnSelectorNode — pick a subset of columns from a 2D tensor.

Common pattern in tabular ML: load a CSV with N columns, then keep only
a few feature columns for a particular experiment. Selecting by index
works regardless of column names; selecting by name requires a
``columns`` LIST input (typically the ``columns`` output of
``CSVReader``).

When both ``indices`` and ``names`` params are set, ``names`` wins —
it's the more readable form and rarely set by accident.
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


def parse_indices(raw: Any, node: str) -> list[int]:
    """Comma-separated whole numbers from the ``indices`` param.

    ``int('a')`` or ``int('1.0')`` raises a message naming neither the node
    nor the param, so each piece is checked here and the bad one quoted.
    """
    text = str(raw if raw is not None else "").strip()
    indices: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            indices.append(int(piece))
        except ValueError:
            raise ValueError(
                f"{node}: indices has {piece!r}, which is not a whole number. "
                f"Write positions as comma-separated whole numbers, for "
                f"example '0,2,3'."
            ) from None
    return indices


class ColumnSelectorNode(BaseNode):
    NODE_NAME = "ColumnSelector"
    CATEGORY = "Data"
    DESCRIPTION = "Pick columns from a 2D tensor, by index or name"
    DETAILS = (
        "Selecting by name needs the `columns` input wired, and `names` wins when "
        "`indices` is also set."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input 2D tensor [rows, cols].",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional list of column names — required when selecting by `names`.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Tensor of shape [rows, len(selected_cols)].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="indices",
                param_type=ParamType.STRING,
                default="0",
                description="Comma-separated column indices, e.g. '0,2,3'. Used when `names` is empty.",
            ),
            ParamDefinition(
                name="names",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Comma-separated column names. When set, takes precedence over "
                    "`indices` and requires the `columns` input to be connected."
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
        x = inputs.get("tensor")
        if x is None:
            raise ValueError("ColumnSelector requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        n_cols = x.shape[-1] if x.ndim >= 1 else 0

        names_param = str(params.get("names", "")).strip()
        if names_param:
            cols_in = inputs.get("columns")
            if not cols_in:
                raise ValueError(
                    "ColumnSelector: `names` is set but no `columns` input is connected — "
                    "either connect a list of column names or use `indices` instead."
                )
            cols_list = [str(c).strip() for c in cols_in]
            if len(cols_list) != n_cols:
                # Names are looked up in this list and used as indices into
                # the tensor: a list describing another tensor (the CSV's,
                # wired to an already-selected tensor) picks the wrong column.
                raise ValueError(
                    f"ColumnSelector: the `columns` input has {len(cols_list)} names "
                    f"but the tensor has {n_cols} columns, so the names cannot say "
                    f"which column is which. Wire the column list that describes "
                    f"this tensor, or select by `indices`."
                )
            requested = [c.strip() for c in names_param.split(",") if c.strip()]
            missing = [c for c in requested if c not in cols_list]
            if missing:
                raise ValueError(
                    f"ColumnSelector: names {missing} not in columns input {cols_list}"
                )
            indices = [cols_list.index(c) for c in requested]
        else:
            indices = parse_indices(params.get("indices", ""), "ColumnSelector")
            for i in indices:
                if i < 0 or i >= n_cols:
                    raise ValueError(
                        f"ColumnSelector: indices contains out-of-range value {i} (tensor has {n_cols} cols)"
                    )

        if not indices:
            return {"tensor": x[..., :0]}
        return {"tensor": x[..., indices]}
