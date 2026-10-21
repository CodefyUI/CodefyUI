"""Corpus BLEU and ROUGE-L for the eval pack, without third-party packages.

Both follow a reference implementation exactly, and
``backend/tests/test_eval_text_metrics.py`` checks them against it:

* **BLEU** is ``sacrebleu.corpus_bleu(hypotheses, [references])`` with
  sacrebleu's defaults (``lowercase=False``, ``max_ngram_order=4``,
  ``effective_order=False``) and its four smoothing methods. The tokenizers
  are sacrebleu's ``13a`` (its default, the mteval-v13a rules), ``zh``,
  ``char`` and ``none``, copied rule for rule -- including ``zh``'s range
  table as sacrebleu writes it.
* **ROUGE-L** is ``rouge_score.rouge_scorer.RougeScorer(["rougeL"])`` with
  ``use_stemmer=False``: the longest common subsequence of the two token
  lists, as precision, recall and F-measure. ``rouge`` tokenizes the way
  rouge-score does (lowercase, every run of characters outside ``a-z0-9`` is
  a separator), which drops every non-ASCII letter, so a Chinese answer has
  no tokens at all; ``char`` splits into characters for such text, and
  ``whitespace`` splits on whitespace.

Why not depend on the two packages: rouge-score brings nltk and absl, and
sacrebleu brings lxml and portalocker, for what is a page of arithmetic.
Computing them here keeps ``cdui plugin install eval`` at the one
dependency the loaders need (transformers), and the tests pin the parity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

MAX_NGRAM_ORDER = 4

BLEU_TOKENIZERS = ("13a", "zh", "char", "none")
BLEU_SMOOTHING = ("exp", "floor", "add-k", "none")
#: sacrebleu's ``BLEU.SMOOTH_DEFAULTS``.
SMOOTH_DEFAULTS = {"none": None, "floor": 0.1, "add-k": 1, "exp": None}

ROUGE_TOKENIZERS = ("rouge", "char", "whitespace")
ROUGE_MEASURES = ("fmeasure", "precision", "recall")


# ── sacrebleu's tokenizers ───────────────────────────────────────────────────

# sacrebleu.tokenizers.tokenizer_re.TokenizerRegexp
_RE_RULES = (
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
)


def _regexp_post(line: str) -> str:
    for pattern, replacement in _RE_RULES:
        line = pattern.sub(replacement, line)
    return " ".join(line.split())


def _tok_13a(line: str) -> str:
    line = line.replace("<skipped>", "")
    line = line.replace("-\n", "")
    line = line.replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"')
        line = line.replace("&amp;", "&")
        line = line.replace("&lt;", "<")
        line = line.replace("&gt;", ">")
    return _regexp_post(f" {line} ")


# sacrebleu.tokenizers.tokenizer_zh._UCODE_RANGES, verbatim. Two entries are
# written there as " 0" and "⾀0": a four-digit escape followed by a
# literal digit, so they are two-character strings, and sacrebleu compares
# characters against them as strings. Reproduced as written, because parity
# is the point.
_ZH_RANGES = (
    ("㐀", "䶵"),
    ("一", "龥"),
    ("龦", "龻"),
    ("豈", "鶴"),
    ("侮", "頻"),
    ("並", "龎"),
    (" 0", "⩭6"),
    ("⾀0", "⾡d"),
    ("＀", "￯"),
    ("⺀", "⻿"),
    ("　", "〿"),
    ("㇀", "㇯"),
    ("⼀", "⿟"),
    ("⿰", "⿿"),
    ("㄀", "ㄯ"),
    ("ㆠ", "ㆿ"),
    ("︐", "︟"),
    ("︰", "﹏"),
    ("☀", "⛿"),
    ("✀", "➿"),
    ("㈀", "㋿"),
    ("㌀", "㏿"),
)


def _is_chinese_char(char: str) -> bool:
    return any(start <= char <= end for start, end in _ZH_RANGES)


def _tok_zh(line: str) -> str:
    line = line.strip()
    out = []
    for char in line:
        out.append(f" {char} " if _is_chinese_char(char) else char)
    return _regexp_post("".join(out))


def _tok_char(line: str) -> str:
    return " ".join(char for char in line)


def _tok_none(line: str) -> str:
    return line


_BLEU_TOKENIZE = {"13a": _tok_13a, "zh": _tok_zh, "char": _tok_char, "none": _tok_none}


def _ngrams(line: str, max_order: int) -> tuple[Counter, int]:
    tokens = line.split()
    grams: list[tuple[str, ...]] = []
    for n in range(1, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            grams.append(tuple(tokens[i:i + n]))
    return Counter(grams), len(tokens)


def bleu_signature(tokenize: str, smooth_method: str) -> str:
    """sacrebleu's signature for these settings, without its version field."""
    smooth = smooth_method
    default = SMOOTH_DEFAULTS[smooth_method]
    if default is not None:
        smooth += f"[{default:.2f}]"
    return f"nrefs:1|case:mixed|eff:no|tok:{tokenize}|smooth:{smooth}"


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    tokenize: str = "13a",
    smooth_method: str = "exp",
) -> dict:
    """Corpus BLEU (0 to 100) with one reference per hypothesis.

    Returns ``{"score", "precisions", "bp", "sys_len", "ref_len", "counts",
    "totals"}``, the fields of sacrebleu's ``BLEUScore``.
    """
    if tokenize not in _BLEU_TOKENIZE:
        raise ValueError(f"unknown BLEU tokenizer {tokenize!r}; expected one of {list(BLEU_TOKENIZERS)}")
    if smooth_method not in SMOOTH_DEFAULTS:
        raise ValueError(f"unknown smoothing {smooth_method!r}; expected one of {list(BLEU_SMOOTHING)}")
    if len(hypotheses) != len(references):
        raise ValueError("BLEU needs one reference per hypothesis")
    tok = _BLEU_TOKENIZE[tokenize]
    order = MAX_NGRAM_ORDER
    correct = [0] * order
    total = [0] * order
    sys_len = ref_len = 0
    for hypothesis, reference in zip(hypotheses, references):
        hyp_grams, hyp_len = _ngrams(tok(str(hypothesis).rstrip()), order)
        ref_grams, this_ref_len = _ngrams(tok(str(reference).rstrip()), order)
        sys_len += hyp_len
        ref_len += this_ref_len
        for gram, count in hyp_grams.items():
            n = len(gram) - 1
            total[n] += count
            if gram in ref_grams:
                correct[n] += min(count, ref_grams[gram])
    return _compute_bleu(correct, total, sys_len, ref_len, smooth_method)


def _compute_bleu(correct, total, sys_len, ref_len, smooth_method) -> dict:
    """sacrebleu's ``BLEU.compute_bleu`` (``effective_order=False``)."""
    correct = list(correct)
    total = list(total)
    smooth_value = SMOOTH_DEFAULTS[smooth_method]
    bp = 1.0
    if sys_len < ref_len:
        bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
    order = MAX_NGRAM_ORDER
    precisions = [0.0] * order
    result = {"precisions": precisions, "bp": bp, "sys_len": sys_len, "ref_len": ref_len,
              "counts": correct, "totals": total}
    if not any(correct):
        return {**result, "score": 0.0}
    smooth_mteval = 1.0
    for n in range(1, order + 1):
        if smooth_method == "add-k" and n > 1:
            correct[n - 1] += smooth_value
            total[n - 1] += smooth_value
        if total[n - 1] == 0:
            break
        if correct[n - 1] == 0:
            if smooth_method == "exp":
                smooth_mteval *= 2
                precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
            elif smooth_method == "floor":
                precisions[n - 1] = 100.0 * smooth_value / total[n - 1]
        else:
            precisions[n - 1] = 100.0 * correct[n - 1] / total[n - 1]
    log_sum = sum(-9999999999 if p == 0.0 else math.log(p) for p in precisions)
    score = bp * math.exp(log_sum / order)
    return {**result, "score": score}


# ── ROUGE-L ──────────────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SPACES = re.compile(r"\s+")
_VALID = re.compile(r"^[a-z0-9]+$")


def rouge_tokens(text: str, tokenize: str = "rouge") -> list[str]:
    """Tokens as ROUGE-L sees them; ``rouge`` is rouge-score's tokenizer."""
    if tokenize == "rouge":
        text = _NON_ALNUM.sub(" ", text.lower())
        return [t for t in _SPACES.split(text) if _VALID.match(t)]
    if tokenize == "char":
        return [c for c in text.lower() if not c.isspace()]
    if tokenize == "whitespace":
        return text.lower().split()
    raise ValueError(f"unknown ROUGE tokenizer {tokenize!r}; expected one of {list(ROUGE_TOKENIZERS)}")


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            current[j] = previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(target: str, prediction: str, tokenize: str = "rouge") -> dict:
    """``{"precision", "recall", "fmeasure"}`` of one prediction against its target."""
    target_tokens = rouge_tokens(str(target), tokenize)
    prediction_tokens = rouge_tokens(str(prediction), tokenize)
    if not target_tokens or not prediction_tokens:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    lcs = _lcs_length(target_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}
