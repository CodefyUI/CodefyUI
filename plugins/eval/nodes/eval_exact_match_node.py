"""EvalExactMatchNode -- the share of ok rows whose answer equals the label.

Checked against ``sklearn.metrics.accuracy_score`` in
``backend/tests/test_eval_nodes.py``; see ``_eval_metric_base`` for what
every eval metric shares.
"""

from __future__ import annotations

from typing import Any

from ._eval_core import exact_match
from ._eval_metric_base import EvalMetricNode


class EvalExactMatchNode(EvalMetricNode):
    NODE_NAME = "Eval-ExactMatch"
    DESCRIPTION = "Share of ok rows whose answer equals the label"
    DETAILS = (
        "Compares the normalised answer with the normalised label, as "
        "sklearn's accuracy_score. Rows with status error or cancelled, and "
        "rows without a label, are skipped and counted in `skipped` -- never "
        "scored as wrong. Logs the value as a run metric under `metric_name`."
    )
    METRIC = "exact_match"

    def score(self, rows: list[dict[str, Any]], params: dict[str, Any]) -> tuple[float, int]:
        return exact_match([r["answer"] for r in rows], [r["label"] for r in rows]), len(rows)
