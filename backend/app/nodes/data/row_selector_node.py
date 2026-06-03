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


class RowSelectorNode(BaseNode):
    NODE_NAME = "RowSelector"
    CATEGORY = "Data"
    DESCRIPTION = (
        "從 2D tensor 取出一部分的列（rows）。用 `indices` 依位置選列，"
        "或用 `names`（需連上 `labels` 輸入）依列名選列。對應 ColumnSelector，"
        "只是作用在「列」這個維度；純挑列、不含條件判斷。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="輸入的 2D tensor [rows, cols]。",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="可選的列標籤清單——用 `names` 選列時需要連上。",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="shape [len(selected_rows), cols] 的 tensor。",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="indices",
                param_type=ParamType.STRING,
                default="0",
                description="逗號分隔的列索引，例如 '0,2,3'。當 `names` 留空時使用。",
            ),
            ParamDefinition(
                name="names",
                param_type=ParamType.STRING,
                default="",
                description="逗號分隔的列名稱。設定時優先於 `indices`，且需要連上 `labels` 輸入。",
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
            if not labels_in:
                raise ValueError(
                    "RowSelector: `names` is set but no `labels` input is connected — "
                    "either connect a list of row labels or use `indices` instead."
                )
            labels_list = [str(c) for c in labels_in]
            requested = [c.strip() for c in names_param.split(",") if c.strip()]
            missing = [c for c in requested if c not in labels_list]
            if missing:
                raise ValueError(
                    f"RowSelector: names {missing} not in labels input {labels_list}"
                )
            indices = [labels_list.index(c) for c in requested]
        else:
            idx_param = str(params.get("indices", "")).strip()
            indices = [int(i) for i in idx_param.split(",") if i.strip()] if idx_param else []
            for i in indices:
                if i < 0 or i >= n_rows:
                    raise ValueError(
                        f"RowSelector: indices contains out-of-range value {i} (tensor has {n_rows} rows)"
                    )

        if not indices:
            return {"tensor": x[:0]}
        return {"tensor": x[indices]}
