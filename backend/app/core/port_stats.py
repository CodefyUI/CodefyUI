"""Summary statistics for one captured port value, computed where the data is.

Why this module exists
----------------------
``/api/execution/outputs/{run}/{node}/{port}`` ships VALUES. That is the right
answer for a [2, 3] tensor and the wrong one for a 2 GB activation: the only
honest way to answer "what does this data look like" at that scale is to
summarise it on the server and send back a couple of kilobytes. This module is
that summary, and nothing here ever returns a payload proportional to the input.

What is exact and what is not
-----------------------------
The split is deliberate, because the stats people debug with are not the stats
that are expensive:

* ALWAYS EXACT, at any size — ``count``, ``min``, ``max``, ``nan_count``,
  ``inf_count``, ``zero_frac``, and integer class balance. These are O(n)
  reductions with no sort, so there is no reason to approximate them, and a
  *sampled* NaN count is close to useless: "3 NaNs in a 1M sample" is not an
  answer anybody can act on, whereas "3 NaNs, exactly" is.
* SAMPLED above ``sample_threshold`` — ``mean``, ``std``, the quantiles and the
  histogram. These describe a distribution, and a distribution is exactly the
  thing a sample estimates well.

The sample is drawn with a seeded CPU generator, so the same tensor and the
same seed give byte-identical output — a cached stat and a recomputed one can
never disagree.

Why the threshold is 4M and not something rounder
-------------------------------------------------
Quantiles need a sort, and ``torch.sort`` costs ~0.18 µs/element (measured on
a 16-thread CPU): 0.7 s at 4M, 3 s at 16M, 9 s at 50M. 4M is therefore the
largest tensor this can summarise exactly while still feeling like a click
rather than a job. (``torch.quantile`` also refuses inputs over 16.7M, which
is why :func:`_quantiles_by_sort` interpolates by hand rather than calling it
— corroborating evidence for the number, not the reason for it.)

An earlier draft kept the threshold at 50M and estimated quantiles above 4M by
inverting a 16384-bin histogram — O(n), no sort. That is more accurate than
sampling on well-behaved data and catastrophically worse on the data this
endpoint exists to debug: bin width is set by the full range, so one diverged
value at 1e6 in a standard-normal tensor moved the reported median from 0.0012
to 25.3. A 1M-element sample put it at 0.0023. Outliers are the whole reason
someone opens this panel, so the estimator that breaks on them is the wrong
one, and sampling took its place along with ~45 lines of it.

Mean and std ride the same working set as the quantiles; ``min``/``max`` do
not, so an outlier is always visible in the stat table even when the
distribution around it was sampled.

JSON safety
-----------
Every float goes through :func:`_num`, which turns NaN/±Inf into ``None``. That
is the convention ``run_store.json_safe`` established in #119, and it is
load-bearing rather than cosmetic: Starlette renders responses with
``allow_nan=False``, so one leaked NaN is a 500, and NaN-laden tensors are
precisely what this endpoint exists to describe.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from typing import Any, Iterator

# ── tunables that are not worth a config knob ────────────────────────────────

#: Bars in the returned histogram. 64 reads well at inspector width and keeps
#: the payload around 1.5 KB.
HISTOGRAM_BINS = 64

#: An integer tensor with at most this many distinct values is a label tensor,
#: and a label tensor wants class balance rather than a histogram.
CLASS_BALANCE_MAX = 64

#: Rows in a tabular column's ``top`` list.
TOP_K = 10

#: Elements per chunk in the full-tensor passes. Bounds the temporaries those
#: passes allocate (a bool mask, an int64 cast) to ~64 MB. When one slice of
#: the first splittable axis is itself over budget, :func:`_iter_chunks`
#: recurses onto the next axis, so the only shape that exceeds this is a
#: single element with no axis to split at all.
CHUNK_ELEMENTS = 8_000_000

#: Rows scanned when working out which columns a list-of-records has. Bounded
#: separately from the cell budget because discovering names is itself a
#: Python loop over every key of every row; a column that first appears after
#: this many records is not described.
RECORD_NAME_PROBE_ROWS = 1_000

#: Cap on one ``top`` value's rendered length. A text column can hold whole
#: paragraphs, and ten of those across sixty columns is the difference between
#: a 2 KB payload and a 600 KB one.
TOP_VALUE_MAX_CHARS = 120

#: Ceiling on the ``torch.unique`` fallback that catches label tensors whose
#: values are few but far apart. Above this the fallback is skipped rather
#: than approximated: a sort of the whole tensor is exactly the cost this
#: module is built to avoid, and a *guessed* class balance is worse than none.
UNIQUE_EXACT_MAX = 1_000_000

#: Columns described in a tabular payload, and rows scanned per column.
DEFAULT_TABULAR_MAX_COLUMNS = 64
DEFAULT_TABULAR_MAX_ROWS = 200_000

QUANTILE_LEVELS: tuple[tuple[str, float], ...] = (
    ("p1", 0.01),
    ("p5", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
    ("p99", 0.99),
)


# ── number formatting ────────────────────────────────────────────────────────


def _num(value: Any) -> float | None:
    """The value as a plain float, or None when it is not finite.

    None rather than NaN/±Inf is the whole point (see the JSON-safety note in
    the module docstring): an all-NaN tensor has no mean, and saying so with
    ``null`` beats emitting a token the encoder refuses.

    Deliberately NOT rounded. An earlier version trimmed to six significant
    figures for compactness, which flattened an int64 tensor near 1e15 into a
    single repeated value and collapsed all 65 edges of a narrow range at a
    large magnitude (1e6 to 1e6+0.5) onto one number — the payload lying about
    the data to save a few hundred bytes. Python emits the shortest string
    that round-trips, so ``0.1`` still costs three characters; only the values
    that genuinely need the digits pay for them, and the frontend rounds for
    display anyway.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── tensor helpers ───────────────────────────────────────────────────────────


def _split_axis(t: Any) -> int | None:
    """First dimension with more than one element, or None if there is none.

    NOT hard-coded to dim 0. A single-sample activation permuted to
    channels-last — ``x.permute(0, 2, 3, 1)`` on ``[1, C, H, W]`` — is
    non-contiguous with ``shape[0] == 1``, which is both the commonest thing
    anyone inspects and the one shape a dim-0 split cannot cut.
    """
    for dim in range(t.dim()):
        if t.shape[dim] > 1:
            return dim
    return None


def _iter_chunks(t: Any, chunk_elements: int = CHUNK_ELEMENTS) -> Iterator[Any]:
    """Walk `t` in pieces of roughly `chunk_elements` elements.

    A contiguous tensor is walked through a free 1-D view. A non-contiguous one
    is walked in slices along {@link _split_axis} instead, because flattening it
    would copy the whole thing — the exact allocation the chunking exists to
    avoid. A tensor with no splittable axis is a single element; it is yielded
    whole because there is nothing to cut.
    """
    if t.numel() == 0:
        return
    if t.is_contiguous():
        flat = t.reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            yield flat[start : start + chunk_elements]
        return
    axis = _split_axis(t)
    if axis is None:
        yield t
        return
    length = t.shape[axis]
    per_slice = max(1, t.numel() // length)
    step = max(1, chunk_elements // per_slice)
    for start in range(0, length, step):
        piece = t.narrow(axis, start, min(step, length - start))
        if piece.numel() > chunk_elements and piece.shape[axis] == 1:
            # One slice of this axis is already over budget — a `[1, 3, 16384,
            # 16384]` image batch splits into three 268M-element pieces, and
            # `_scan_tensor` would then build a bool mask over each. Hand the
            # piece back to the next splittable axis. Terminating: this branch
            # only runs once `step` has collapsed to a single slice, so every
            # recursion retires one axis.
            yield from _iter_chunks(piece, chunk_elements)
        else:
            yield piece


def _dtype_kind(t: Any) -> str:
    """One of ``float`` / ``int`` / ``bool`` / ``other`` (complex, quantized).

    A non-strided layout (sparse, nested) also lands in ``other``: every pass
    below calls ``is_contiguous`` or ``amin``, and those raise rather than
    answer for a tensor that is not a dense buffer.
    """
    import torch

    if t.layout != torch.strided:
        return "other"
    if t.dtype == torch.bool:
        return "bool"
    if t.is_floating_point():
        return "float"
    if t.is_complex():
        return "other"
    try:
        # Integral dtypes are the ones that survive an int64 cast unchanged.
        # Asking this way (rather than listing dtypes) keeps future int types
        # working, and quantized dtypes fall out as "other" because they raise.
        torch.zeros(1, dtype=t.dtype).to(torch.int64)
    except (RuntimeError, TypeError):
        return "other"
    return "int"


class _Scan:
    """Result of the single full-tensor pass."""

    __slots__ = ("nan", "inf", "zero", "lo", "hi", "finite")

    def __init__(self) -> None:
        self.nan = 0
        self.inf = 0
        self.zero = 0
        self.lo: float | int | None = None
        self.hi: float | int | None = None
        self.finite = 0


def _scan_tensor(t: Any, kind: str) -> _Scan:
    """One pass over every element: health counts plus the finite extremes.

    NaN and Inf are excluded from min/max — a diverged tensor whose max reads
    ``inf`` tells you nothing about the values you still have.
    """
    import torch

    out = _Scan()
    lo_t = None
    hi_t = None
    for chunk in _iter_chunks(t):
        n = chunk.numel()
        if kind == "float":
            nan_mask = torch.isnan(chunk)
            inf_mask = torch.isinf(chunk)
            n_nan = int(nan_mask.sum())
            n_inf = int(inf_mask.sum())
            out.nan += n_nan
            out.inf += n_inf
            out.finite += n - n_nan - n_inf
            if n_nan or n_inf:
                good = ~(nan_mask | inf_mask)
                if not bool(good.any()):
                    continue
                # `where` rather than a boolean index: masked selection would
                # allocate a second copy of the chunk's surviving elements.
                c_lo = torch.where(good, chunk, torch.inf).amin()
                c_hi = torch.where(good, chunk, -torch.inf).amax()
            else:
                c_lo = chunk.amin()
                c_hi = chunk.amax()
        else:
            out.finite += n
            c_lo = chunk.amin()
            c_hi = chunk.amax()
        lo_t = c_lo if lo_t is None else torch.minimum(lo_t, c_lo)
        hi_t = c_hi if hi_t is None else torch.maximum(hi_t, c_hi)
        out.zero += int((chunk == 0).sum())

    if lo_t is not None:
        if kind == "int":
            out.lo = int(lo_t)
            out.hi = int(hi_t)
        elif kind == "bool":
            out.lo = int(bool(lo_t))
            out.hi = int(bool(hi_t))
        else:
            out.lo = float(lo_t)
            out.hi = float(hi_t)
    return out


def _flat_view(t: Any) -> Any:
    """1-D view of `t`, copying only when the layout leaves no choice."""
    try:
        return t.view(-1)
    except RuntimeError:
        return t.reshape(-1)


def _narrow_for_sampling(t: Any, target: int, seed: int) -> Any:
    """Pre-thin a non-contiguous tensor before flattening it.

    Flattening a non-contiguous 2 GB tensor copies 2 GB, which is exactly the
    allocation a sampled path must not make. Keeping a subset of slices along
    {@link _split_axis} first means only the survivors get copied.

    The slices are CHOSEN AT RANDOM, not strided. An earlier version kept every
    n-th slice, which on a ``[1, 64, 1024, 1024]`` activation meant channels
    0, 16, 32 and 48 and nothing else — a systematically selected quarter of
    the data reported under ``"sampled": true``, which reads as an unbiased
    draw. Any axis that carries structure (channel, class, time) would have
    been summarised from whichever slices the stride happened to land on.

    Drawn with ``randint`` rather than ``randperm`` because the axis can be
    long: a permutation of 500M rows is a 4 GB allocation to pick four of them.
    ``unique`` sorts and de-duplicates, so the survivors stay in order and no
    slice is counted twice.
    """
    import torch

    if t.is_contiguous() or t.numel() <= target * 4:
        return t
    axis = _split_axis(t)
    if axis is None:
        return t
    length = t.shape[axis]
    per_slice = max(1, t.numel() // length)
    keep = max(1, -(-target * 4 // per_slice))
    if keep >= length:
        return t
    generator = torch.Generator()
    generator.manual_seed(seed)
    picks = torch.unique(
        torch.randint(0, length, (keep,), generator=generator, dtype=torch.long)
    )
    return t.index_select(axis, picks.to(t.device))


def _seeded_sample(flat: Any, size: int, seed: int) -> Any:
    """`size` elements drawn uniformly (with replacement) from `flat`.

    The index draw happens on CPU whatever device the data is on, so a CUDA
    tensor and its CPU copy sample the same positions.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randint(0, flat.numel(), (size,), generator=generator, dtype=torch.long)
    if flat.device.type != "cpu":
        idx = idx.to(flat.device)
    return flat.index_select(0, idx)


def _finite_only(x: Any, kind: str) -> Any:
    """Drop NaN/Inf. Returns `x` untouched when there is nothing to drop."""
    import torch

    if kind != "float":
        return x
    mask = torch.isfinite(x)
    if bool(mask.all()):
        return x
    return x[mask]


def _quantiles_by_sort(work: Any, levels=QUANTILE_LEVELS) -> dict[str, float | None]:
    """Exact quantiles, linearly interpolated — ``numpy.quantile`` semantics."""
    import torch

    ordered, _ = torch.sort(work.to(torch.float64))
    n = ordered.numel()
    out: dict[str, float | None] = {}
    for name, q in levels:
        pos = q * (n - 1)
        low = int(math.floor(pos))
        high = min(low + 1, n - 1)
        frac = pos - low
        value = float(ordered[low]) * (1.0 - frac) + float(ordered[high]) * frac
        out[name] = _num(value)
    return out


def _histogram_counts(work: Any, bins: int, lo: float, hi: float) -> Any:
    """``torch.histc`` over `work`, cast to something histc accepts."""
    import torch

    if work.dtype not in (torch.float32, torch.float64):
        work = work.to(torch.float32)
    return torch.histc(work, bins=bins, min=lo, max=hi).to(torch.int64)


def _exact_value_counts(t: Any, lo: int, span: int) -> list[dict[str, Any]]:
    """Exact per-value counts over EVERY element, via a chunked bincount.

    O(n) with no sort, so class balance stays exact even for a tensor whose
    distribution had to be sampled — which matters, because "is my training set
    balanced" is not a question anybody wants answered approximately.
    """
    import torch

    total = None
    for chunk in _iter_chunks(t):
        shifted = chunk.reshape(-1).to(torch.int64) - lo
        counts = torch.bincount(shifted, minlength=span)
        total = counts if total is None else total + counts
    if total is None:
        return []
    return [
        {"value": lo + i, "count": int(c)}
        for i, c in enumerate(total.tolist())
        if c > 0
    ]


# ── the tensor branch ────────────────────────────────────────────────────────


def _tensor_stats(
    value: Any,
    *,
    sample_threshold: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    t = value.detach()
    numel = t.numel()
    kind = _dtype_kind(t)

    stats: dict[str, Any] = {
        "kind": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "count": numel,
        "sampled": False,
        "sample_size": None,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "quantiles": {},
        "nan_count": 0,
        "inf_count": 0,
        "zero_frac": None,
        "histogram": None,
        "value_counts": None,
    }

    if numel == 0:
        stats["zero_frac"] = 0.0
        return stats
    if kind == "other":
        # Complex / quantized: the shape and dtype are still worth showing, but
        # every stat below assumes a total order that these dtypes do not have.
        return stats

    scan = _scan_tensor(t, kind)
    stats["nan_count"] = scan.nan
    stats["inf_count"] = scan.inf
    stats["zero_frac"] = _num(scan.zero / numel)
    stats["min"] = _num(scan.lo) if kind == "float" else scan.lo
    stats["max"] = _num(scan.hi) if kind == "float" else scan.hi

    if scan.finite == 0 or scan.lo is None:
        # Everything is NaN/Inf. Reporting `mean: null` beats reporting NaN,
        # and the counts above already say what happened.
        return stats

    # ── the working set: what the distribution stats actually read ──────────
    sampled = numel > sample_threshold
    if sampled:
        narrowed = _narrow_for_sampling(t, sample_size, seed)
        work = _seeded_sample(_flat_view(narrowed), sample_size, seed)
    else:
        work = _flat_view(t)
    if work.device.type != "cpu":
        work = work.to("cpu")
    work = _finite_only(work, kind)

    if work.numel() == 0:
        # A sample that happened to land entirely on NaNs.
        stats["sampled"] = sampled
        stats["sample_size"] = 0 if sampled else None
        return stats

    stats["sampled"] = sampled
    stats["sample_size"] = work.numel() if sampled else None

    # float64 for integers, not float32: a mantissa of 24 bits silently
    # rounds anything past 2^24, and an int64 tensor around 1e15 would report
    # a mean sitting outside the p25-p75 spread printed beside it.
    if work.dtype == torch.float64:
        work_f = work
    else:
        work_f = work.to(torch.float64 if kind == "int" else torch.float32)
    var, mean = torch.var_mean(work_f, correction=0)
    stats["mean"] = _num(mean)
    # Population std (ddof=0), matching `numpy.std` rather than torch's own
    # default of Bessel-corrected — this module checks itself against numpy.
    stats["std"] = _num(math.sqrt(max(float(var), 0.0)))

    lo = float(scan.lo)
    hi = float(scan.hi)
    stats["quantiles"] = (
        _quantiles_by_sort(work_f)
        if hi > lo
        else {name: _num(lo) for name, _ in QUANTILE_LEVELS}
    )

    # ── class balance, or a histogram ──────────────────────────────────────
    if kind in ("int", "bool"):
        span = int(scan.hi) - int(scan.lo) + 1
        if 1 <= span <= CLASS_BALANCE_MAX:
            counts = _exact_value_counts(t, int(scan.lo), span)
        elif not sampled and work.numel() <= UNIQUE_EXACT_MAX:
            # Few distinct values, far apart — labels carrying their original
            # ids rather than 0..C-1. Only taken when `work` IS the whole
            # tensor, so these counts are exact like the bincount ones; a
            # sampled class balance would be a guess wearing a count's clothes.
            unique, unique_counts = torch.unique(work, return_counts=True)
            counts = (
                [
                    {"value": int(v), "count": int(c)}
                    for v, c in zip(unique.tolist(), unique_counts.tolist())
                ]
                if unique.numel() <= CLASS_BALANCE_MAX
                else []
            )
        else:
            counts = []
        if counts:
            if kind == "bool":
                counts = [{"value": bool(e["value"]), "count": e["count"]} for e in counts]
            stats["value_counts"] = counts
            return stats

    if hi > lo:
        counts = _histogram_counts(work_f, HISTOGRAM_BINS, lo, hi)
        width = (hi - lo) / HISTOGRAM_BINS
        stats["histogram"] = {
            "bins": HISTOGRAM_BINS,
            "edges": [_num(lo + i * width) for i in range(HISTOGRAM_BINS + 1)],
            "counts": [int(c) for c in counts.tolist()],
        }
    else:
        # One value, one bar. A 64-bin histogram of a constant is 63 lies.
        stats["histogram"] = {
            "bins": 1,
            "edges": [_num(lo), _num(hi)],
            "counts": [int(work.numel())],
        }
    return stats


# ── the tabular branch ───────────────────────────────────────────────────────


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def _top_value(v: Any) -> str | int | float | bool:
    """One `top` entry, with long text cut to a length a table can show."""
    if isinstance(v, (int, float, bool)):
        return v
    text = v if isinstance(v, str) else str(v)
    if len(text) <= TOP_VALUE_MAX_CHARS:
        return text
    return text[:TOP_VALUE_MAX_CHARS] + "..."


def _column_stats(name: str, values: list[Any], max_rows: int) -> dict[str, Any]:
    """count / unique / top-k / missing for one column, plus numeric moments.

    ``unique`` and ``top`` are computed from a plain ``dict`` counter rather
    than a set-plus-sort, so one walk answers both. Unhashable cell values
    (a list in a cell) are counted by their repr — a stat is better than a
    crash, and the repr is what the UI would show anyway.
    """
    scanned = values if len(values) <= max_rows else values[:max_rows]
    counter: dict[Any, int] = {}
    missing = 0
    numeric: list[float] = []
    all_numeric = True
    saw_bool = False
    saw_float = False
    saw_int = False
    saw_str = False

    for v in scanned:
        if _is_missing(v):
            missing += 1
            continue
        if isinstance(v, bool):
            saw_bool = True
            numeric.append(float(v))
        elif isinstance(v, int):
            saw_int = True
            numeric.append(float(v))
        elif isinstance(v, float):
            saw_float = True
            numeric.append(v)
        else:
            all_numeric = False
            if isinstance(v, str):
                saw_str = True
        try:
            counter[v] = counter.get(v, 0) + 1
        except TypeError:
            key = repr(v)
            counter[key] = counter.get(key, 0) + 1

    if saw_bool and not (saw_int or saw_float or saw_str):
        dtype = "bool"
    elif all_numeric and saw_float:
        dtype = "float"
    elif all_numeric and saw_int:
        dtype = "int"
    elif saw_str and all(isinstance(k, str) for k in counter):
        dtype = "str"
    elif not counter:
        dtype = "empty"
    else:
        dtype = "mixed"

    top = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:TOP_K]
    column: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "count": len(scanned) - missing,
        "missing": missing,
        "unique": len(counter),
        "top": [{"value": _top_value(v), "count": c} for v, c in top],
        "mean": None,
        "min": None,
        "max": None,
    }
    if all_numeric and numeric:
        mean = sum(numeric) / len(numeric)
        column["mean"] = _num(mean)
        column["min"] = _num(min(numeric))
        column["max"] = _num(max(numeric))
    return column


def _row_budget(columns: int, max_rows: int) -> int:
    """Rows to scan PER COLUMN, given a budget over the whole table.

    The column walk is interpreted Python, so a 64-column table at 200k rows
    each would be 12.8M dict operations - seconds of wall clock for a summary
    that is supposed to be instant. Spreading the budget keeps the cost flat in
    the column count, with a floor so a very wide table still sees enough rows
    per column for its top-k to mean something.
    """
    return max(1000, max_rows // max(1, columns)) if columns else max_rows


def _as_columns(
    value: Any, *, max_rows: int, max_columns: int
) -> tuple[list[tuple[str, list[Any]]], int, int] | None:
    """Normalise a tabular-ish value into ``([(name, cells)], rows, columns)``.

    Three shapes are recognised, all of them things a node in this repo
    actually emits: a columnar dict (``{"age": [...], "city": [...]}``), a list
    of record dicts, and a flat list of primitives (``CSVReader.labels``), which
    becomes a single column so class balance over string labels works.

    The caps are applied HERE, at the slice, not downstream where the summary
    is computed. A 10M-row CSV transposed in full is over a gigabyte of list
    slots and ten seconds on the GIL — spent building rows nobody then reads.
    The returned row and column counts are still the true ones; only the cells
    handed on are cut.
    """
    if isinstance(value, dict):
        if not value:
            return [], 0, 0
        rows = 0
        # Distinct dict keys can share a `str()` (1 and "1"). Left alone they
        # become two columns with one name and a duplicate React key.
        keep: list[tuple[str, Any]] = []
        names: list[str] = []
        seen: set[str] = set()
        for key, cells in value.items():
            if not isinstance(cells, (list, tuple)):
                return None
            rows = max(rows, len(cells))
            name = str(key)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(keep) < max_columns:
                keep.append((name, cells))
        budget = _row_budget(len(keep), max_rows)
        columns = [(name, list(cells[:budget])) for name, cells in keep]
        return columns, rows, len(names)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [], 0, 0
        probe = value[:RECORD_NAME_PROBE_ROWS]
        if all(isinstance(row, dict) for row in probe):
            record_names: list[str] = []
            seen_names: set[str] = set()
            for row in probe:
                for key in row:
                    if str(key) not in seen_names:
                        seen_names.add(str(key))
                        record_names.append(str(key))
            kept = record_names[:max_columns]
            budget = _row_budget(len(kept), max_rows)
            head = value[:budget]
            columns = [(name, [row.get(name) for row in head]) for name in kept]
            return columns, len(value), len(record_names)
        primitive = (str, int, float, bool, type(None))
        if all(isinstance(x, primitive) for x in probe):
            budget = _row_budget(1, max_rows)
            return [("values", list(value[:budget]))], len(value), 1
    return None


def _tabular_stats(
    columns: list[tuple[str, list[Any]]],
    rows: int,
    column_count: int,
    *,
    max_rows: int,
) -> dict[str, Any]:
    # ``max_rows`` is a budget over the whole TABLE, not per column. The column
    # walk is interpreted Python, so a 64-column table at 200k rows each would
    # be 12.8M dict operations — seconds of wall clock for a summary that is
    # supposed to be instant. Spreading the budget keeps the cost flat in the
    # column count, with a floor so a very wide table still sees enough rows
    # per column for its top-k to mean something.
    per_column = max(1000, max_rows // max(1, len(columns))) if columns else max_rows
    sampled = rows > per_column
    return {
        "kind": "tabular",
        "rows": rows,
        "column_count": column_count,
        "columns_truncated": column_count > len(columns),
        "sampled": sampled,
        "sample_size": min(rows, per_column) if sampled else None,
        "columns": [_column_stats(name, cells, per_column) for name, cells in columns],
    }


# ── entry point ──────────────────────────────────────────────────────────────


def compute_port_stats(
    value: Any,
    *,
    sample_threshold: int | None = None,
    sample_size: int | None = None,
    tabular_max_rows: int | None = None,
    tabular_max_columns: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarise one captured port value.

    The returned dict is JSON-safe by construction and bounded in size: a
    64-bin histogram or at most ``CLASS_BALANCE_MAX`` value counts for a
    tensor, at most ``tabular_max_columns`` column summaries for a table.

    Every limit is a parameter rather than a settings read so tests can drive
    the branches with small inputs. The route fills the two that users have a
    reason to tune (``sample_threshold``, ``sample_size``) from ``settings``;
    the tabular caps have no knob because nothing has needed one.
    """
    from ..config import settings

    if sample_threshold is None:
        sample_threshold = settings.STATS_SAMPLE_THRESHOLD
    if sample_size is None:
        sample_size = settings.STATS_SAMPLE_SIZE
    if tabular_max_rows is None:
        tabular_max_rows = DEFAULT_TABULAR_MAX_ROWS
    if tabular_max_columns is None:
        tabular_max_columns = DEFAULT_TABULAR_MAX_COLUMNS

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _tensor_stats(
                value,
                sample_threshold=max(1, sample_threshold),
                sample_size=max(1, sample_size),
                seed=seed,
            )
        if isinstance(value, torch.nn.Module):
            return {"kind": "unsupported", "type": type(value).__name__}
    except ImportError:  # pragma: no cover - torch is a hard dependency here
        pass

    max_rows = max(1, tabular_max_rows)
    max_columns = max(1, tabular_max_columns)
    shaped = _as_columns(value, max_rows=max_rows, max_columns=max_columns)
    if shaped is not None:
        columns, rows, column_count = shaped
        return _tabular_stats(columns, rows, column_count, max_rows=max_rows)

    return {"kind": "unsupported", "type": type(value).__name__}


# ── cache ────────────────────────────────────────────────────────────────────


def payload_bytes(payload: Any) -> int:
    """Serialized size of a stats payload, in bytes.

    Mirrors ``run_service.json_size`` deliberately rather than importing it:
    that module pulls in RunService and the whole database layer, and this one
    is meant to stay a leaf. ``ensure_ascii`` defaults to True, so ``len`` is a
    byte count.
    """
    try:
        return len(json.dumps(payload, separators=(",", ":"), default=str))
    except (TypeError, ValueError, RecursionError):  # pragma: no cover
        return 0


class PortStatsCache:
    """A bytes-bounded LRU for computed stat payloads.

    Bounded by BYTES, not by entry count. An entry-count bound would be a
    guess dressed as a limit here: a 64-bin histogram is ~1.5 KB and a wide
    tabular summary is ~50 KB, so "200 entries" means anywhere from 300 KB to
    10 MB depending on what the user happened to inspect.

    Guarded by a ``threading.Lock`` rather than an ``asyncio.Lock`` because
    the compute that fills it runs in a worker thread.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[0]

    def put(self, key: tuple, payload: Any) -> None:
        size = payload_bytes(payload)
        with self._lock:
            if size > self._max_bytes:
                # Caching it would evict everything else and still not fit.
                self._drop(key)
                return
            self._drop(key)
            self._entries[key] = (payload, size)
            self._total += size
            while self._total > self._max_bytes and self._entries:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._total -= evicted

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total = 0

    def _drop(self, key: tuple) -> None:
        """Remove `key` if present. Caller holds the lock."""
        old = self._entries.pop(key, None)
        if old is not None:
            self._total -= old[1]
