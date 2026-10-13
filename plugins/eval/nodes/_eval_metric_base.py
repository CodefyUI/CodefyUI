"""The shape every eval metric node shares.

A metric reads a prediction table, scores the ``ok`` rows that have a
label, reports how many rows it counted and how many it skipped (an error,
a cancelled item or an unlabelled row is skipped, never scored as wrong),
logs the value as a run metric so a sweep can read it, and hands out a
one-row table in both transports -- the stats split table for
Stats-TableView, and a record table whose meta names the metric node, its
params and the run it measured.
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

from ._eval_core import (
    PREDICTION_COLUMNS,
    STATUS_OK,
    metric_result,
    require_table,
    table_records,
)


class EvalMetricNode(BaseNode):
    """Base class; NODE_NAME stays empty so the registry skips it."""

    NODE_NAME = ""
    CATEGORY = "Training"

    # Logs a run metric, which a cache hit would silently skip (#254).
    cacheable = False
    align_inputs = False

    #: The metric's default series name.
    METRIC = ""

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.ANY,
                description="Prediction table from Eval-BatchInference.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="value",
                data_type=DataType.SCALAR,
                description="The metric over the counted rows.",
            ),
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="One-row [1, 3] table: the metric, rows counted, rows skipped.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Column names of `table`.",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="The run the row measures: model@revision.",
            ),
            PortDefinition(
                name="metrics",
                data_type=DataType.ANY,
                description="The same row as a record table, with the metric's producer in its meta.",
            ),
        ]

    @classmethod
    def metric_params(cls) -> list[ParamDefinition]:
        return []

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            *cls.metric_params(),
            ParamDefinition(
                name="metric_name",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Name the value is logged under as a run metric. Empty uses the "
                    "metric's own name; give A and B runs different names."
                ),
            ),
            ParamDefinition(
                name="step",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Step the run metric is logged at.",
                advanced=True,
            ),
        ]

    def metric_name(self, params: dict[str, Any]) -> str:
        return self.METRIC

    def score(
        self, rows: list[dict[str, Any]], params: dict[str, Any]
    ) -> tuple[float, int]:
        """(value, counted) over the scorable rows."""
        raise NotImplementedError

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        node = self.NODE_NAME
        table = require_table(
            inputs.get("predictions"), node=node, port="predictions",
            columns=PREDICTION_COLUMNS,
        )
        records = table_records(table)
        scorable = [
            r for r in records
            if r.get("status") == STATUS_OK and r.get("label") is not None
        ]
        if not scorable:
            raise ValueError(
                f"{node}: no row is both ok and labelled, so there is nothing to "
                f"score ({len(records)} row(s) in the table)."
            )
        value, counted = self.score(scorable, params)
        skipped = len(records) - counted
        metric = self.metric_name(params)
        series = str(params.get("metric_name", "") or "").strip() or metric
        if context is not None:
            step = max(0, int(params.get("step", 0) or 0))
            context.log_metric(series, value, step)
        used = {k: params.get(k) for k in [p.name for p in self.define_params()]}
        return metric_result(
            node=node, metric=metric, value=value, counted=counted,
            skipped=skipped, table=table, params=used,
        )
