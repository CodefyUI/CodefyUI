"""EvalF1ScoreNode -- micro or macro F1 of the answers against the labels.

Checked against ``sklearn.metrics.f1_score`` in
``backend/tests/test_eval_nodes.py``; see ``_eval_metric_base`` for what
every eval metric shares.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import ParamDefinition, ParamType

from ._eval_core import f1_score
from ._eval_metric_base import EvalMetricNode

AVERAGES = ("macro", "micro")


class EvalF1ScoreNode(EvalMetricNode):
    NODE_NAME = "Eval-F1Score"
    DESCRIPTION = "Micro or macro F1 of the answers against the labels"
    DETAILS = (
        "macro averages per-class F1 over every class seen in the answers or "
        "the labels, so a rare class counts as much as a common one; micro "
        "pools every decision, which for single-label classification equals "
        "exact match. A class with no true or predicted item scores 0, as in "
        "sklearn's f1_score. Skips error, cancelled and unlabelled rows."
    )
    METRIC = "f1"

    @classmethod
    def metric_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="average",
                param_type=ParamType.SELECT,
                default="macro",
                options=list(AVERAGES),
                description="macro: mean of per-class F1. micro: F1 over all decisions pooled.",
            ),
        ]

    def _average(self, params: dict[str, Any]) -> str:
        average = str(params.get("average", "macro") or "macro")
        if average not in AVERAGES:
            raise ValueError(
                f"{self.NODE_NAME}: unknown average {average!r}; expected one of {list(AVERAGES)}."
            )
        return average

    def metric_name(self, params: dict[str, Any]) -> str:
        return f"f1_{self._average(params)}"

    def score(self, rows: list[dict[str, Any]], params: dict[str, Any]) -> tuple[float, int]:
        value = f1_score(
            [r["answer"] for r in rows], [r["label"] for r in rows], self._average(params)
        )
        return value, len(rows)
