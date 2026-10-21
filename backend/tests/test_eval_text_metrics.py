"""Eval-BLEU and Eval-ROUGE-L against the reference implementations (#139).

The pack computes both metrics itself (``_eval_text_metrics``) so that
installing it brings no nltk, absl or lxml. These tests are what makes that
safe: every number is compared with ``sacrebleu.corpus_bleu`` and
``rouge_score.rouge_scorer.RougeScorer``, which the backend's ``dev`` extra
installs for exactly this (test-only, like tensorboard for core#136).

Fixtures are small and random, drawn from a vocabulary chosen to hit every
rule the tokenizers have: punctuation next to digits and letters, an HTML
entity, a hyphen after a digit, Chinese characters, and a character inside
the two ranges sacrebleu's ``zh`` table writes as five-digit escapes.
"""

from __future__ import annotations

import random

import pytest
import sacrebleu
from rouge_score import rouge_scorer

from cdui_plugins.eval.nodes._eval_core import (
    PREDICTION_COLUMNS,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    input_hash,
    make_table,
)
from cdui_plugins.eval.nodes._eval_text_metrics import (
    BLEU_SMOOTHING,
    BLEU_TOKENIZERS,
    bleu_signature,
    corpus_bleu,
    rouge_l,
    rouge_tokens,
)
from cdui_plugins.eval.nodes.eval_bleu_node import EvalBLEUNode
from cdui_plugins.eval.nodes.eval_rouge_l_node import EvalRougeLNode

VOCAB = [
    "the", "a", "cat", "dog", "sat", "on", "mat", "red", "2", "3.5", "-", "U.S.",
    "&amp;", "don't", "貓", "在", "墊子", "上", ",", ".", "x-ray", "1,000", "Hello!",
    "e.g.", "9-5", "(ok)", "‐", "⨀", "ＡＢ", "naïve",
]


def _sentence(rng: random.Random, low: int = 0, high: int = 12) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(rng.randint(low, high)))


def _corpus(seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    return [_sentence(rng) for _ in range(n)], [_sentence(rng) for _ in range(n)]


class _Tokenizer:
    """A rouge-score tokenizer object wrapping one of the pack's tokenizers."""

    def __init__(self, mode: str):
        self.mode = mode

    def tokenize(self, text):
        return rouge_tokens(text, self.mode)


# ── BLEU ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("tokenize", BLEU_TOKENIZERS)
@pytest.mark.parametrize("smooth", BLEU_SMOOTHING)
def test_corpus_bleu_equals_sacrebleu(seed, tokenize, smooth):
    hypotheses, references = _corpus(seed)
    ours = corpus_bleu(hypotheses, references, tokenize=tokenize, smooth_method=smooth)
    ref = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize, smooth_method=smooth)
    assert ours["score"] == pytest.approx(ref.score, abs=1e-9)
    assert ours["bp"] == pytest.approx(ref.bp, abs=1e-12)
    assert (ours["sys_len"], ours["ref_len"]) == (ref.sys_len, ref.ref_len)
    assert ours["precisions"] == pytest.approx(ref.precisions, abs=1e-9)


def test_bleu_on_a_hand_checked_fixture():
    hypotheses = ["the cat sat on the mat", "a dog"]
    references = ["the cat sat on a mat", "a red dog"]
    ours = corpus_bleu(hypotheses, references)
    ref = sacrebleu.corpus_bleu(hypotheses, [references])
    assert ours["score"] == pytest.approx(ref.score, abs=1e-9)
    # 8 hypothesis tokens, 9 reference tokens: a brevity penalty applies.
    assert (ours["sys_len"], ours["ref_len"]) == (8, 9)
    assert ours["bp"] == pytest.approx(0.8824969, abs=1e-6)
    assert corpus_bleu(["same words here ok"], ["same words here ok"])["score"] == pytest.approx(100.0)


def test_the_signature_matches_sacrebleus_without_its_version():
    for tokenize in BLEU_TOKENIZERS:
        for smooth in BLEU_SMOOTHING:
            metric = sacrebleu.BLEU(tokenize=tokenize, smooth_method=smooth)
            metric.corpus_score(["a"], [["a"]])
            theirs = str(metric.get_signature()).rsplit("|version:", 1)[0]
            assert bleu_signature(tokenize, smooth) == theirs


def test_bleu_refuses_unknown_settings():
    with pytest.raises(ValueError, match="tokenizer"):
        corpus_bleu(["a"], ["a"], tokenize="intl")
    with pytest.raises(ValueError, match="smoothing"):
        corpus_bleu(["a"], ["a"], smooth_method="lin")


# ── ROUGE-L ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(80))
def test_rouge_l_equals_rouge_score(seed):
    rng = random.Random(seed)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    target, prediction = _sentence(rng), _sentence(rng)
    ours = rouge_l(target, prediction)
    ref = scorer.score(target, prediction)["rougeL"]
    assert ours["precision"] == pytest.approx(ref.precision, abs=1e-12)
    assert ours["recall"] == pytest.approx(ref.recall, abs=1e-12)
    assert ours["fmeasure"] == pytest.approx(ref.fmeasure, abs=1e-12)


@pytest.mark.parametrize("mode", ["char", "whitespace"])
@pytest.mark.parametrize("seed", range(20))
def test_rouge_l_with_another_tokenizer_equals_rouge_scores_lcs(mode, seed):
    """rouge-score accepts a tokenizer object; its LCS then scores our tokens."""
    rng = random.Random(seed)
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=_Tokenizer(mode))
    target, prediction = _sentence(rng), _sentence(rng)
    ref = scorer.score(target, prediction)["rougeL"]
    assert rouge_l(target, prediction, mode)["fmeasure"] == pytest.approx(ref.fmeasure, abs=1e-12)


def test_the_rouge_tokenizer_drops_chinese_and_char_keeps_it():
    assert rouge_tokens("貓在墊子上") == []
    assert rouge_l("貓在墊子上", "貓在上")["fmeasure"] == 0.0
    assert rouge_tokens("貓在 上", "char") == ["貓", "在", "上"]
    assert rouge_l("貓在墊子上", "貓在上", "char")["fmeasure"] == pytest.approx(2 * 0.6 * 1.0 / 1.6)


# ── the nodes ────────────────────────────────────────────────────────────────

def _table(rows: list[dict]) -> dict:
    full = []
    for r in rows:
        text = f"input of {r['id']}"
        full.append([r["id"], text, input_hash(text), r.get("answer"), r.get("answer"),
                     r.get("label"), r.get("status", STATUS_OK), r.get("error")])
    return make_table(PREDICTION_COLUMNS, full, {
        "producer": {"model_id": "org/lm", "model_revision": "c" * 40}})


def _generation_rows(seed: int = 0, n: int = 12):
    rng = random.Random(seed)
    ok = [{"id": f"g{i:02d}", "answer": _sentence(rng, 1), "label": _sentence(rng, 1)}
          for i in range(n)]
    broken = [
        {"id": "x0", "answer": None, "label": "a cat", "status": STATUS_ERROR, "error": "boom"},
        {"id": "x1", "answer": None, "label": "a dog", "status": STATUS_CANCELLED},
        {"id": "x2", "answer": "an answer", "label": None},
    ]
    return ok + broken, ok


class _Metrics:
    def __init__(self):
        self.logged = []

    def log_metric(self, name, value, step, node_id=None):
        self.logged.append((name, float(value), int(step)))


@pytest.mark.parametrize("tokenize", BLEU_TOKENIZERS)
def test_the_bleu_node_scores_the_ok_labelled_rows_as_sacrebleu(tokenize):
    rows, ok = _generation_rows()
    context = _Metrics()
    out = EvalBLEUNode().execute({"predictions": _table(rows)},
                                 {"tokenize": tokenize, "metric_name": "bleu_a"}, context=context)
    expected = sacrebleu.corpus_bleu([r["answer"] for r in ok], [[r["label"] for r in ok]],
                                     tokenize=tokenize).score
    assert out["value"] == pytest.approx(expected, abs=1e-9)
    assert out["columns"] == ["bleu", "counted", "skipped"]
    assert out["table"].tolist()[0][1:] == [12.0, 3.0]
    assert context.logged == [("bleu_a", pytest.approx(expected), 0)]
    details = out["metrics"]["meta"]["metric"]["details"]
    assert details["signature"] == f"nrefs:1|case:mixed|eff:no|tok:{tokenize}|smooth:exp"
    assert details["compatible_with"] == "sacrebleu.corpus_bleu"
    assert len(details["precisions"]) == 4


@pytest.mark.parametrize(("normalize", "case"), [("strip_lower", "lc"), ("strip", "mixed"),
                                                  ("none", "mixed")])
def test_the_bleu_signature_follows_the_tables_normalisation(normalize, case):
    """BatchInference's default strip_lower lowercases answers and labels, so
    the default BLEU is sacrebleu's case:lc, and its signature says so."""
    rows, ok = _generation_rows(seed=2)
    table = _table(rows)
    table["meta"]["producer"]["normalize"] = normalize
    out = EvalBLEUNode().execute({"predictions": table}, {})
    details = out["metrics"]["meta"]["metric"]["details"]
    assert details["normalize"] == normalize
    assert details["signature"] == f"nrefs:1|case:{case}|eff:no|tok:13a|smooth:exp"
    metric = sacrebleu.BLEU(lowercase=case == "lc")
    hypotheses = [r["answer"] for r in ok]
    references = [r["label"] for r in ok]
    if case == "lc":  # what strip_lower hands the node
        hypotheses = [h.lower() for h in hypotheses]
        references = [r.lower() for r in references]
        table = _table([{**r, "answer": r["answer"] and r["answer"].lower(),
                         "label": r["label"] and r["label"].lower()} for r in rows])
        table["meta"]["producer"]["normalize"] = normalize
        out = EvalBLEUNode().execute({"predictions": table}, {})
    score = metric.corpus_score(hypotheses, [references])
    assert out["value"] == pytest.approx(score.score, abs=1e-9)
    theirs = str(metric.get_signature()).rsplit("|version:", 1)[0]
    assert details["signature"] == theirs


def test_the_bleu_node_refuses_an_unknown_tokenizer():
    rows, _ = _generation_rows()
    with pytest.raises(ValueError, match="unknown tokenize"):
        EvalBLEUNode().execute({"predictions": _table(rows)}, {"tokenize": "intl"})


@pytest.mark.parametrize(("measure", "name"), [("fmeasure", "rouge_l"),
                                                ("precision", "rouge_l_precision"),
                                                ("recall", "rouge_l_recall")])
def test_the_rouge_l_node_is_the_mean_of_rouge_score_over_ok_rows(measure, name):
    rows, ok = _generation_rows(seed=4)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [getattr(scorer.score(r["label"], r["answer"])["rougeL"], measure) for r in ok]
    out = EvalRougeLNode().execute({"predictions": _table(rows)}, {"measure": measure})
    assert out["value"] == pytest.approx(sum(scores) / len(scores), abs=1e-12)
    assert out["columns"][0] == name
    assert out["table"].tolist()[0][1:] == [12.0, 3.0]
    details = out["metrics"]["meta"]["metric"]["details"]
    assert details["tokenize"] == "rouge"
    assert details["rows_scoring_zero"] == sum(
        1 for r in ok if scorer.score(r["label"], r["answer"])["rougeL"].fmeasure == 0.0)


def test_the_rouge_l_node_scores_chinese_only_with_a_tokenizer_that_sees_it():
    rows = [{"id": "z1", "answer": "貓在上", "label": "貓在墊子上"}]
    out = EvalRougeLNode().execute({"predictions": _table(rows)}, {})
    assert out["value"] == 0.0
    out = EvalRougeLNode().execute({"predictions": _table(rows)}, {"tokenize": "char"})
    assert out["value"] == pytest.approx(0.75)
