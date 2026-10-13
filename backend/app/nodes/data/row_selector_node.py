"""RowSelectorNode — 從 2D tensor 取出指定的列（rows）。

ColumnSelector 在「欄」維度做的事，RowSelector 在「列」維度照做一遍：
載入一張表後，只保留特定幾列。用索引選列不需要列名；用名稱選列則需要
連上 ``labels`` LIST 輸入（通常是某個欄位轉成的列標籤）。

當 ``indices`` 與 ``names`` 同時設定時，``names`` 優先——它比較好讀、
也比較少不小心設到。這是純粹的「依位置/名稱挑列」基礎算子，不含任何
條件判斷；要依條件（例如 英文 > 80）篩列請改用 Edu-FilterRows。
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
from .column_selector_node import parse_indices


def _label_text(value: Any) -> str:
    """A row label as stripped text; integral floats read as '1', not '1.0'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


class RowSelectorNode(BaseNode):
    NODE_NAME = "RowSelector"
    CATEGORY = "Data"
    DESCRIPTION = "Pick rows from a 2D tensor, by index or name"
    DETAILS = (
        "Selecting by name needs the `labels` input wired, with one label per "
        "row of the tensor, and `names` wins when `indices` is also set. A name "
        "selects every row with that label, so names=cat on labels [cat, dog, "
        "cat] gives two rows; several names give their rows in the order the "
        "names are listed. Filtering by a condition such as score > 80 is "
        "FilterRows, in the EDU pack."
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
                name="labels",
                data_type=DataType.LIST,
                description="Optional list of row labels. Required when selecting rows by `names`.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Tensor of shape [len(selected_rows), cols].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="indices",
                param_type=ParamType.STRING,
                default="0",
                description="Comma-separated row indices, for example '0,2,3'. Used when `names` is empty.",
            ),
            ParamDefinition(
                name="names",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Comma-separated row names; each selects every row with that label. "
                    "Takes precedence over `indices` and needs the `labels` input connected."
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
            raise ValueError("RowSelector requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        n_rows = x.shape[0] if x.ndim >= 1 else 0

        names_param = str(params.get("names", "")).strip()
        if names_param:
            labels_in = inputs.get("labels")
            if isinstance(labels_in, torch.Tensor):
                labels_in = labels_in.reshape(-1).tolist()
            if labels_in is None or len(labels_in) == 0:
                raise ValueError(
                    "RowSelector: `names` is set but no `labels` input is connected — "
                    "either connect a list of row labels or use `indices` instead."
                )
            labels_list = [_label_text(c) for c in labels_in]
            if len(labels_list) != n_rows:
                # Names are looked up in this list and used as indices into
                # the tensor: a list describing another tensor picks the
                # wrong rows, or runs past the end.
                raise ValueError(
                    f"RowSelector: the `labels` input has {len(labels_list)} labels "
                    f"but the tensor has {n_rows} rows, so the labels cannot say "
                    f"which row is which. Wire the label list that describes "
                    f"this tensor, or select by `indices`."
                )
            requested = [c.strip() for c in names_param.split(",") if c.strip()]
            missing = [c for c in requested if c not in labels_list]
            if missing:
                raise ValueError(
                    f"RowSelector: names {missing} not in labels input {labels_list}"
                )
            # Every row carrying a requested label, grouped by name in the
            # order the names are given, rows in table order within a name.
            indices = [i for c in requested for i, label in enumerate(labels_list) if label == c]
        else:
            indices = parse_indices(params.get("indices", ""), "RowSelector")
            for i in indices:
                if i < 0 or i >= n_rows:
                    raise ValueError(
                        f"RowSelector: indices contains out-of-range value {i} (tensor has {n_rows} rows)"
                    )

        if not indices:
            return {"tensor": x[:0]}
        return {"tensor": x[indices]}
