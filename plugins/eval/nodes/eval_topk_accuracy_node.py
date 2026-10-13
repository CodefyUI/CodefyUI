"""EvalTopKAccuracyNode -- the share of ok rows whose label is in the top k.

Checked against ``sklearn.metrics.top_k_accuracy_score`` in
``backend/tests/test_eval_nodes.py``; see ``_eval_metric_base`` for what
every eval metric shares.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import ParamDefinition, ParamType

from ._eval_core import topk_hits
from ._eval_infer import MAX_TOPK
from ._eval_metric_base import EvalMetricNode


class EvalTopKAccuracyNode(EvalMetricNode):
    NODE_NAME = "Eval-TopKAccuracy"
    DESCRIPTION = "Share of ok rows whose label is in the top k answers"
    DETAILS = (
        "Reads the ranked candidates Eval-BatchInference keeps in the `topk` "
        "column in classification mode (up to five, most likely first). A row "
        "without candidates -- a generation row -- is skipped, not counted as "
        "a miss. k = 1 equals exact match. Matches sklearn's "
        "top_k_accuracy_score when the scores have no ties."
    )
    METRIC = "topk_accuracy"

    @classmethod
    def metric_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="k",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                max_value=MAX_TOPK,
                description="How many of the ranked answers may contain the label.",
            ),
        ]

    def _k(self, params: dict[str, Any]) -> int:
        k = int(params.get("k", 2) or 2)
        if not 1 <= k <= MAX_TOPK:
            raise ValueError(f"{self.NODE_NAME}: `k` must be 1-{MAX_TOPK}; got {k}.")
        return k

    def metric_name(self, params: dict[str, Any]) -> str:
        return f"top{self._k(params)}_accuracy"

    def score(self, rows: list[dict[str, Any]], params: dict[str, Any]) -> tuple[float, int]:
        hits, counted = topk_hits(
            [r.get("topk") for r in rows], [r["label"] for r in rows], self._k(params)
        )
        if not counted:
            raise ValueError(
                f"{self.NODE_NAME}: no row has ranked candidates in a `topk` "
                "column. Eval-BatchInference writes them in classification mode only."
            )
        return hits / counted, counted
