"""Shared plumbing for the eval pack: record tables, metrics, alignment.

Pure Python plus ``torch`` for the one numeric projection. Nothing here
imports ``transformers``, so every rule in this file is testable on a CI
runner that has never installed it. See ``plugins/eval/README.md`` for the
third-party-facing version of the contracts below.

**The record table.** The stats pack's table contract is a numeric 2D
``TENSOR`` plus a ``columns`` LIST, and it says plainly that non-numeric
data never enters ``table``. A prediction is text -- an id, an input, an
answer, an error message -- so the eval pack carries its tables as one
JSON-safe dict on an ``ANY`` port::

    {"format": "codefyui.eval.table", "version": 1,
     "columns": ["id", "input", ...],     # names, in row order
     "rows": [["s01", "...", ...], ...],  # one list per row, column-aligned
     "meta": {...}}                       # provenance, counts, identity

and every node that produces one ALSO produces the numeric split-table
projection (``TENSOR`` + ``columns`` + ``row_labels``) that
``Stats-TableView`` and ``Stats-ChartView`` render. Same column/row
vocabulary, two transports: text where the cell is text, the stats contract
where the cell is a number.

**Prediction tables** carry :data:`PREDICTION_COLUMNS` first, in that order,
then any extras (``score``, ``topk``). ``status`` is one of
:data:`STATUSES`; a metric counts only ``ok`` rows and says how many it
skipped.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import torch

#: The ``format`` tag every record table carries.
TABLE_FORMAT = "codefyui.eval.table"
TABLE_VERSION = 1

#: The ``format`` tag of an eval dataset (JSONLDataset / PromptTemplate).
DATASET_FORMAT = "codefyui.eval.dataset"

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUSES = (STATUS_OK, STATUS_ERROR, STATUS_CANCELLED)

#: The fixed head of a prediction table, in order. ``input_hash`` is the
#: input's identity: the first 16 hex digits of the sha256 of the exact text
#: the model saw (and, for a VQA item, of the image's sha256 with it; see
#: :func:`item_input_hash`), so two runs over "the same" item can prove
#: they were.
PREDICTION_COLUMNS = (
    "id",
    "input",
    "input_hash",
    "raw_answer",
    "answer",
    "label",
    "status",
    "error",
)

#: Answer normalisation modes BatchInference offers.
NORMALIZE_MODES = ("strip_lower", "strip", "none")

#: Cells longer than this are cut in the text rendering (never in the table).
TEXT_CELL_WIDTH = 32


# ── hashing ──────────────────────────────────────────────────────────────────

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_hash(text: str) -> str:
    """The identity of one model input: 16 hex digits of its sha256."""
    return sha256_text(text)[:16]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def item_input_hash(text: str, image_sha256: str | None = None) -> str:
    """The identity of what the model saw: the text, plus the image if it read one.

    Without an image this is :func:`input_hash` of the text, so a text-only
    table keeps the hashes it always had. With one, the image's sha256 joins
    the text, so two runs over "q07" can prove they asked the same question
    about the same picture.
    """
    if not image_sha256:
        return input_hash(text)
    return input_hash(f"{text}\0image-sha256:{image_sha256}")


# ── record tables ────────────────────────────────────────────────────────────

def make_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a record table, checking every row has one cell per column."""
    names = [str(c) for c in columns]
    if len(set(names)) != len(names):
        raise ValueError(f"record table has duplicate column names: {names}")
    out_rows: list[list[Any]] = []
    for index, row in enumerate(rows):
        cells = list(row)
        if len(cells) != len(names):
            raise ValueError(
                f"record table row {index} has {len(cells)} cells for "
                f"{len(names)} columns."
            )
        out_rows.append(cells)
    return {
        "format": TABLE_FORMAT,
        "version": TABLE_VERSION,
        "columns": names,
        "rows": out_rows,
        "meta": dict(meta or {}),
    }


def require_table(
    value: Any, *, node: str, port: str, columns: Sequence[str] = ()
) -> dict[str, Any]:
    """Check *value* is a record table holding *columns*; return it."""
    if value is None:
        raise ValueError(f"{node} requires a `{port}` input.")
    if not isinstance(value, dict) or value.get("format") != TABLE_FORMAT:
        raise ValueError(
            f"{node}: `{port}` must be an eval record table (from "
            f"Eval-BatchInference or another Eval node); got "
            f"{type(value).__name__}."
        )
    have = list(value.get("columns") or [])
    missing = [c for c in columns if c not in have]
    if missing:
        raise ValueError(
            f"{node}: `{port}` has no column(s) {missing}; it has {have}."
        )
    return value


def table_records(table: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows of a record table as dicts keyed by column name."""
    columns = list(table["columns"])
    return [dict(zip(columns, row)) for row in table["rows"]]


def table_column(table: dict[str, Any], name: str) -> list[Any]:
    index = list(table["columns"]).index(name)
    return [row[index] for row in table["rows"]]


def numeric_projection(
    rows: Sequence[Sequence[Any]], columns: Sequence[str]
) -> torch.Tensor:
    """A float32 ``[rows, columns]`` tensor; ``None`` becomes NaN (missing).

    NaN is the stats contract's spelling of "missing", which is exactly what
    an item that was never compared is.
    """
    data = [
        [float("nan") if cell is None else float(cell) for cell in row]
        for row in rows
    ]
    if not data:
        return torch.zeros((0, len(columns)), dtype=torch.float32)
    return torch.tensor(data, dtype=torch.float32)


# ── eval datasets ────────────────────────────────────────────────────────────

def make_dataset(items: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """An eval dataset: items with ``id``, ``input`` and ``label``."""
    return {"format": DATASET_FORMAT, "version": 1, "items": items, "meta": meta}


def require_dataset(value: Any, *, node: str, port: str = "dataset") -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{node} requires a `{port}` input.")
    if not isinstance(value, dict) or value.get("format") != DATASET_FORMAT:
        raise ValueError(
            f"{node}: `{port}` must be an eval dataset from Eval-JSONLDataset "
            f"or Eval-PromptTemplate; got {type(value).__name__}."
        )
    return value


def first_duplicates(values: Iterable[Any], limit: int = 5) -> list[Any]:
    """The first *limit* values that occur more than once, in first-seen order."""
    seen: set[Any] = set()
    reported: list[Any] = []
    for value in values:
        if value in seen and value not in reported:
            reported.append(value)
            if len(reported) >= limit:
                break
        seen.add(value)
    return reported


# ── normalisation ────────────────────────────────────────────────────────────

def parse_label_map(raw: Any, *, node: str) -> dict[str, str]:
    """Parse ``"LABEL_0=negative, LABEL_1=positive"`` into a dict."""
    out: dict[str, str] = {}
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"{node}: `label_map` entries are `from=to` pairs separated by "
                f"commas; {part!r} has no '='."
            )
        source, _, target = part.partition("=")
        out[source.strip()] = target.strip()
    return out


def normalize_answer(
    value: Any, mode: str = "strip_lower", label_map: dict[str, str] | None = None
) -> str:
    """The normalised form of a raw answer (or a label).

    ``label_map`` applies to the stripped raw text BEFORE case folding, so a
    map written against a model's own label names (``LABEL_0``) matches what
    the model actually said.
    """
    text = "" if value is None else str(value)
    if mode != "none":
        text = text.strip()
    if label_map:
        text = label_map.get(text, text)
    if mode == "strip_lower":
        text = text.lower()
    return text


# ── metrics ──────────────────────────────────────────────────────────────────

def scored_rows(table: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """The ``ok`` rows of a prediction table, and how many were skipped."""
    records = table_records(table)
    ok = [r for r in records if r.get("status") == STATUS_OK]
    return ok, len(records) - len(ok)


def exact_match(answers: Sequence[str], labels: Sequence[str]) -> float:
    if not answers:
        return 0.0
    return sum(1 for a, y in zip(answers, labels) if a == y) / len(answers)


def _sorted_classes(values: Iterable[str]) -> list[str]:
    """sklearn's class order: sorted, numerically where every label is a number."""
    unique = sorted(set(values))
    try:
        return sorted(unique, key=float)
    except ValueError:
        return unique


def f1_score(answers: Sequence[str], labels: Sequence[str], average: str) -> float:
    """F1 as ``sklearn.metrics.f1_score(labels, answers, average=...)``.

    ``micro`` pools every decision (for single-label classification it is
    the accuracy); ``macro`` is the unweighted mean of per-class F1 over the
    union of classes seen in either list, with 0/0 counted as 0 -- sklearn's
    ``zero_division`` default, minus the warning.
    """
    if average not in ("micro", "macro"):
        raise ValueError(f"unknown average {average!r}; expected micro or macro")
    if not answers:
        return 0.0
    classes = _sorted_classes(list(labels) + list(answers))
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    for predicted, actual in zip(answers, labels):
        if predicted == actual:
            tp[actual] += 1
        else:
            fp[predicted] += 1
            fn[actual] += 1
    if average == "micro":
        num = 2 * sum(tp.values())
        den = num + sum(fp.values()) + sum(fn.values())
        return num / den if den else 0.0
    scores = []
    for c in classes:
        den = 2 * tp[c] + fp[c] + fn[c]
        scores.append((2 * tp[c] / den) if den else 0.0)
    return sum(scores) / len(scores)


def topk_hits(
    candidates: Sequence[Sequence[str] | None], labels: Sequence[str], k: int
) -> tuple[int, int]:
    """(hits, counted): rows whose label is among their first *k* candidates.

    A row with no candidate list is not counted at all -- it is not a miss,
    it is a row this metric cannot judge.
    """
    hits = counted = 0
    for ranked, label in zip(candidates, labels):
        if not isinstance(ranked, (list, tuple)):
            continue
        counted += 1
        if label in list(ranked)[:k]:
            hits += 1
    return hits, counted


def producer_label(meta: dict[str, Any]) -> str:
    """A short, human row label for the run a table came from."""
    producer = meta.get("producer") or {}
    model = str(producer.get("model_id") or "run")
    revision = str(producer.get("model_revision") or "")
    if revision.startswith("local-sha256:"):
        revision = revision[len("local-sha256:"):]
    return f"{model}@{revision[:7]}" if revision else model


def metric_result(
    *,
    node: str,
    metric: str,
    value: float,
    counted: int,
    skipped: int,
    table: dict[str, Any],
    params: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The outputs every metric node returns, in one shape.

    ``table``/``columns``/``row_labels`` are the stats split table (one row);
    ``metrics`` is the same row as a record table whose meta names the
    metric's producer -- the metric node and its params -- and the run it
    measured, copied from the prediction table's own provenance. *details*
    (BLEU's signature and n-gram counts, say) joins the metric's entry.
    """
    label = producer_label(table.get("meta") or {})
    columns = [metric, "counted", "skipped"]
    row = [float(value), int(counted), int(skipped)]
    source = table.get("meta") or {}
    metrics = make_table(
        ["run", *columns],
        [[label, *row]],
        meta={
            "metric": {"name": metric, "node": node, "params": dict(params),
                       **({"details": json_safe(details)} if details else {})},
            "producer": source.get("producer"),
            "dataset": source.get("dataset"),
            "seed": source.get("seed"),
        },
    )
    text = (
        f"{metric} = {value:.4f} over {counted} ok row(s)"
        + (f"; {skipped} row(s) skipped (status error/cancelled)" if skipped else "")
        + f" -- {label}"
    )
    return {
        "value": float(value),
        "table": numeric_projection([row], columns),
        "columns": columns,
        "row_labels": [label],
        "metrics": metrics,
        "__log__": text,
    }


# ── rendering ────────────────────────────────────────────────────────────────

def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        text = "|".join(str(v) for v in value)
    else:
        text = str(value)
    text = text.replace("\n", " ")
    if len(text) > TEXT_CELL_WIDTH:
        text = text[: TEXT_CELL_WIDTH - 3] + "..."
    return text


def format_records(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_rows: int = 20,
    title: str = "",
) -> str:
    """Render record rows as fixed-width text for the Results log."""
    shown = list(rows) if max_rows <= 0 else list(rows)[:max_rows]
    cells = [[_cell_text(c) for c in row] for row in shown]
    widths = [len(str(c)) for c in columns]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    if title:
        lines.append(title)
    lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)).rstrip())
    for row in cells:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    left = len(rows) - len(shown)
    if left > 0:
        lines.append(f"... {left} more row(s)")
    return "\n".join(lines)


# ── charts ───────────────────────────────────────────────────────────────────

def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def heatmap_spec(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any]:
    """A ``MEDIA_CHART`` heatmap, per the stats pack's chart contract."""
    values = [[_finite(v) for v in row] for row in matrix]
    flat = [v for row in values for v in row]
    return {
        "kind": "heatmap",
        "title": str(title),
        "x_label": str(x_label),
        "y_label": str(y_label),
        "matrix": values,
        "row_labels": [str(r) for r in row_labels],
        "col_labels": [str(c) for c in col_labels],
        "vmin": 0.0,
        "vmax": max(flat) if flat else 1.0,
        "colormap": "blues",
    }


# ── provenance ───────────────────────────────────────────────────────────────

def is_commit_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def is_pinned_revision(value: Any) -> bool:
    """A revision that names exactly one set of weights."""
    return is_commit_sha(value) or str(value or "").startswith("local-sha256:")


def reproducibility(
    producer: dict[str, Any],
    dataset: dict[str, Any],
    generation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Whether a run can be reproduced from what its table records, and why not.

    A model id alone never qualifies: ``main`` moves. What does is a pinned
    model revision, a pinned processor revision, the dataset's content hash,
    and -- for sampled generation -- the seed.
    """
    reasons = []
    if not is_pinned_revision(producer.get("model_revision")):
        reasons.append("model revision is not pinned to a commit or content hash")
    if not is_pinned_revision(producer.get("processor_revision")):
        reasons.append("processor revision is not pinned to a commit or content hash")
    if not dataset.get("sha256"):
        reasons.append("dataset has no content hash")
    if generation and generation.get("do_sample") and generation.get("seed") is None:
        reasons.append("sampled generation without a seed")
    return {"reproducible": not reasons, "reasons": reasons}


def json_safe(value: Any) -> Any:
    """Round-trip through JSON so a meta dict can never carry a live object."""
    return json.loads(json.dumps(value, default=str))
