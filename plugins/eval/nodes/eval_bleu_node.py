"""EvalBLEUNode -- corpus BLEU of the answers against the labels.

Computed by ``_eval_text_metrics.corpus_bleu``, which reproduces
``sacrebleu.corpus_bleu`` and is checked against it in
``backend/tests/test_eval_text_metrics.py``; see ``_eval_metric_base`` for
what every eval metric shares.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import ParamDefinition, ParamType

from ._eval_metric_base import EvalMetricNode
from ._eval_text_metrics import (
    BLEU_SMOOTHING,
    BLEU_TOKENIZERS,
    bleu_signature,
    corpus_bleu,
)


class EvalBLEUNode(EvalMetricNode):
    NODE_NAME = "Eval-BLEU"
    DESCRIPTION = "Corpus BLEU of the answers against the labels"
    DETAILS = (
        "Corpus-level BLEU on a 0 to 100 scale, as sacrebleu's corpus_bleu "
        "computes it with one reference per item: n-grams up to 4 pooled over "
        "every counted row, a brevity penalty, and the chosen smoothing. The "
        "answer and label are the normalised ones from Eval-BatchInference, so "
        "normalize=strip_lower there gives case-insensitive BLEU. tokenize 13a "
        "is sacrebleu's default; use zh or char for Chinese text. Skips error, "
        "cancelled and unlabelled rows. The metrics table's meta records the "
        "sacrebleu signature and the n-gram counts."
    )
    METRIC = "bleu"

    @classmethod
    def metric_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="tokenize",
                param_type=ParamType.SELECT,
                default="13a",
                options=list(BLEU_TOKENIZERS),
                description="13a: sacrebleu's default. zh: Chinese characters split. char: every character. none: spaces only.",
            ),
            ParamDefinition(
                name="smooth_method",
                param_type=ParamType.SELECT,
                default="exp",
                options=list(BLEU_SMOOTHING),
                description="sacrebleu's smoothing for n-gram orders with no match. exp is its default.",
                advanced=True,
            ),
        ]

    def _choice(self, params: dict[str, Any], name: str, default: str, options) -> str:
        value = str(params.get(name, default) or default)
        if value not in options:
            raise ValueError(
                f"{self.NODE_NAME}: unknown {name} {value!r}; expected one of {list(options)}."
            )
        return value

    def score(self, rows: list[dict[str, Any]], params: dict[str, Any]):
        tokenize = self._choice(params, "tokenize", "13a", BLEU_TOKENIZERS)
        smooth = self._choice(params, "smooth_method", "exp", BLEU_SMOOTHING)
        result = corpus_bleu(
            [r["answer"] or "" for r in rows], [r["label"] for r in rows],
            tokenize=tokenize, smooth_method=smooth,
        )
        details = {
            "signature": bleu_signature(tokenize, smooth),
            "compatible_with": "sacrebleu.corpus_bleu",
            "precisions": result["precisions"],
            "bp": result["bp"],
            "sys_len": result["sys_len"],
            "ref_len": result["ref_len"],
            "counts": result["counts"],
            "totals": result["totals"],
        }
        return result["score"], len(rows), details
