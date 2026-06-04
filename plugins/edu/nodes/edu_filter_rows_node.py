"""EduFilterRowsNode — 依條件篩選表格的列。

對應教材 **I1-1（表格資料動手）**。學生從一張 2D 表 ``[rows, columns]`` 出發，
問「哪些列滿足 `欄 op 門檻`？」（例如 英文 > 80）。本節點回傳篩選後的表、
布林遮罩、以及通過的列數。

「依條件篩選列」概念上是三步——取出該欄、跟門檻比較、用遮罩去索引原表。
pandas 把它藏在 ``df[df["英文"] > 80]`` 一行裡；本節點把步驟攤開，讓
Teaching Inspector 能一格一格顯示：

    1. pick_column = table[:, col]
    2. mask        = pick_column op threshold
    3. count       = mask.sum()
    4. filtered    = table[mask]

它比 RowSelector / ColumnSelector「多做事」（帶條件判斷），所以歸在教學用的
`edu` plugin 而非平台內建；面板上歸到 EDU 分類。
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

_OPS = {
    ">": lambda c, t: c > t,
    ">=": lambda c, t: c >= t,
    "<": lambda c, t: c < t,
    "<=": lambda c, t: c <= t,
    "==": lambda c, t: c == t,
    "!=": lambda c, t: c != t,
}


class EduFilterRowsNode(BaseNode):
    NODE_NAME = "FilterRows"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "依條件篩選 2D 表格的列：只保留「指定欄 op 門檻」成立的列（例如 英文 > 80）。"
        "可用欄名選欄（需連 `columns` 輸入）或用位置 `column_index`。"
        "輸出篩選後的表、布林列遮罩 mask、以及通過的列數 count。"
        "verbose 模式會記錄 取欄 → 比較 → 計數 → 索引 四個步驟。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="2D tensor，shape [rows, columns]。",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="可選的欄名清單；用 `column_name` 篩選時需要連上（例如 CSVReader.columns）。",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="篩選後的表，shape [k, columns]（k = 通過的列數）。",
            ),
            PortDefinition(
                name="mask",
                data_type=DataType.TENSOR,
                description="布林遮罩，shape [rows]；通過條件的列為 True。",
            ),
            PortDefinition(
                name="count",
                data_type=DataType.SCALAR,
                description="通過條件的列數（k）。",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="column_name",
                param_type=ParamType.STRING,
                default="",
                description="要比較的欄位名稱。設定時需連 `columns` 輸入；優先於 `column_index`。",
            ),
            ParamDefinition(
                name="column_index",
                param_type=ParamType.INT,
                default=0,
                description="要比較的欄位位置索引，當 `column_name` 留空時使用。",
                min_value=0,
            ),
            ParamDefinition(
                name="op",
                param_type=ParamType.SELECT,
                default=">",
                options=list(_OPS.keys()),
                description="對選定欄逐元素套用的比較運算子。",
            ),
            ParamDefinition(
                name="threshold",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="與欄位比較的門檻值。",
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
            raise ValueError("FilterRows requires a `table` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.ndim != 2:
            raise ValueError(
                f"FilterRows expects a 2D [rows, columns] tensor; got shape {tuple(x.shape)}."
            )

        column_name = str(params.get("column_name", "")).strip()
        if column_name:
            cols_in = inputs.get("columns")
            if not cols_in:
                raise ValueError(
                    "FilterRows: `column_name` is set but no `columns` input is connected — "
                    "connect a list of column names (e.g. CSVReader.columns) or clear "
                    "`column_name` to filter by `column_index`."
                )
            cols_list = [str(c) for c in cols_in]
            if column_name not in cols_list:
                raise ValueError(
                    f"FilterRows: column_name {column_name!r} not found in columns {cols_list}."
                )
            col_idx = cols_list.index(column_name)
        else:
            col_idx = int(params.get("column_index", 0))
            if col_idx < 0 or col_idx >= x.shape[1]:
                raise ValueError(
                    f"FilterRows: column_index {col_idx} is out of range for a table "
                    f"with {x.shape[1]} columns."
                )

        op = str(params.get("op", ">")).strip()
        if op not in _OPS:
            raise ValueError(
                f"FilterRows: unknown op {op!r}. Choose one of {list(_OPS)}."
            )
        threshold = float(params.get("threshold", 0.0))

        column = x[:, col_idx]
        mask = _OPS[op](column, threshold)
        filtered = x[mask]
        count = int(mask.sum().item())

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            rec = StepRecorder()
            rec.record(
                "input", "Receive the 2D table.",
                scalars={"rows": float(x.shape[0]), "columns": float(x.shape[1])},
                table=x,
            )
            label = f" ({column_name!r})" if column_name else ""
            rec.record(
                "pick_column", f"Take column index {col_idx}{label}.",
                column=column,
            )
            rec.record(
                "compare", f"Compare each value with {op} {threshold:g}.",
                mask=mask.float(),
            )
            rec.record(
                "count", "Sum the True entries to get the surviving row count.",
                scalars={"count": float(count)},
            )
            rec.record(
                "index", "Index the original table with the mask to keep passing rows.",
                filtered=filtered,
            )
            return {"tensor": filtered, "mask": mask, "count": count, "__steps__": rec.steps}

        return {"tensor": filtered, "mask": mask, "count": count}
