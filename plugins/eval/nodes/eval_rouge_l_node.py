"""EvalRougeLNode -- mean ROUGE-L of the answers against the labels.

Computed by ``_eval_text_metrics.rouge_l``, which reproduces
``rouge_score.rouge_scorer.RougeScorer(["rougeL"])`` and is checked against
it in ``backend/tests/test_eval_text_metrics.py``; see ``_eval_metric_base``
for what every eval metric shares.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import ParamDefinition, ParamType

from ._eval_metric_base import EvalMetricNode
from ._eval_text_metrics import ROUGE_MEASURES, ROUGE_TOKENIZERS, rouge_l


class EvalRougeLNode(EvalMetricNode):
    NODE_NAME = "Eval-ROUGE-L"
    DESCRIPTION = "Mean ROUGE-L of the answers against the labels"
    DETAILS = (
        "ROUGE-L scores each row by the longest common subsequence of its "
        "answer and label tokens, as rouge-score's RougeScorer(['rougeL']) "
        "without stemming, and the value is the mean over the counted rows "
        "(0 to 1). tokenize rouge is rouge-score's tokenizer: lowercase, and "
        "anything outside a-z and 0-9 separates tokens, so text with no ASCII "
        "letters or digits (Chinese, for one) has no tokens and scores 0; "
        "use char for such text. A row with no tokens on either side scores 0. "
        "Skips error, cancelled and unlabelled rows."
    )
    METRIC = "rouge_l"

    @classmethod
    def metric_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="measure",
                param_type=ParamType.SELECT,
                default="fmeasure",
                options=list(ROUGE_MEASURES),
                description="fmeasure, precision (share of the answer in the LCS) or recall (share of the label).",
            ),
            ParamDefinition(
                name="tokenize",
                param_type=ParamType.SELECT,
                default="rouge",
                options=list(ROUGE_TOKENIZERS),
                description="rouge: rouge-score's tokenizer (ASCII only). char: characters. whitespace: split on spaces.",
            ),
        ]

    def _choice(self, params: dict[str, Any], name: str, default: str, options) -> str:
        value = str(params.get(name, default) or default)
        if value not in options:
            raise ValueError(
                f"{self.NODE_NAME}: unknown {name} {value!r}; expected one of {list(options)}."
            )
        return value

    def metric_name(self, params: dict[str, Any]) -> str:
        measure = self._choice(params, "measure", "fmeasure", ROUGE_MEASURES)
        return "rouge_l" if measure == "fmeasure" else f"rouge_l_{measure}"

    def score(self, rows: list[dict[str, Any]], params: dict[str, Any]):
        measure = self._choice(params, "measure", "fmeasure", ROUGE_MEASURES)
        tokenize = self._choice(params, "tokenize", "rouge", ROUGE_TOKENIZERS)
        scores = [rouge_l(r["label"], r["answer"] or "", tokenize) for r in rows]
        value = sum(s[measure] for s in scores) / len(scores)
        zero = sum(1 for s in scores if s["fmeasure"] == 0.0)
        details = {
            "compatible_with": "rouge_score.rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)",
            "tokenize": tokenize,
            "measure": measure,
            "aggregation": "mean over rows",
            "rows_scoring_zero": zero,
        }
        return value, len(rows), details
