"""StatsTableViewNode — render a table as fixed-width text in the Results log.

The terminal end of a stats chain. It emits the reserved ``__log__`` key,
which the ``output_kind`` channel turns into a ``text`` entry (#117), so the
rendered table appears in the Results panel with no new frontend surface at
all — and also hands the same string back on a ``text`` port for anything
downstream that wants it.

Rows past ``max_rows`` are dropped with a trailing "... N more row(s)" line
rather than silently, because a table that stops without saying so reads as
the whole answer.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._stats_core import as_matrix, format_text_table, int_param, resolve_columns


class StatsTableViewNode(BaseNode):
    NODE_NAME = "Stats-TableView"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Show a table in the Results panel as aligned text, with its column "
        "names and row labels. Caps the output at `max_rows` and says how many "
        "rows it left out."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="2D [rows, columns] table to render.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional column names, in table order.",
                optional=True,
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="Optional row names — e.g. Stats-Describe's statistic names.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="The rendered table, as it appears in the Results panel.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="max_rows",
                param_type=ParamType.INT,
                default=20,
                min_value=0,
                max_value=1000,
                description="Rows to show before truncating. 0 means every row.",
            ),
            ParamDefinition(
                name="precision",
                param_type=ParamType.INT,
                default=4,
                min_value=0,
                max_value=12,
                description="Decimal places for non-integer values.",
            ),
            ParamDefinition(
                name="title",
                param_type=ParamType.STRING,
                default="",
                description="Optional heading printed above the table.",
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
        raw_rows = inputs.get("row_labels")
        row_labels = (
            [str(r) for r in raw_rows] if isinstance(raw_rows, (list, tuple)) else []
        )

        rendered = format_text_table(
            table,
            names,
            row_labels,
            max_rows=int_param(params, "max_rows", 20, node=self.NODE_NAME),
            precision=int_param(params, "precision", 4, node=self.NODE_NAME),
        )
        title = str(params.get("title", "") or "").strip()
        text = f"{title}\n{rendered}" if title else rendered

        # `__log__` is the reserved key the output_kind channel reads as a
        # `text` entry; `text` is the same string as ordinary port data.
        return {"text": text, "__log__": text}
