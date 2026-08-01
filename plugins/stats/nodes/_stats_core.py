"""Shared plumbing for the stats pack: the table contract and chart specs.

Two contracts live here, and both are normative for the pack — see
``plugins/stats/README.md`` for the third-party-facing version.

**The table contract.** CodefyUI has no ``TABLE`` port type (``DataType`` is a
closed Enum in core). What ``CSVReader`` and ``SyntheticDataset`` actually
emit, and what ``ColumnSelector`` / ``RowSelector`` / ``Edu-ColumnStats``
actually accept, is a *split table*: a 2D numeric ``TENSOR`` carrying the
values plus a parallel ``LIST`` of column names. This pack produces and
consumes exactly that, and additionally carries a ``row_labels`` LIST — the
one thing no core node round-trips, and the reason a ``describe()`` table is
readable at all.

**Missing values.** NaN means missing; ±Inf is a value. That is pandas' rule
(``count()`` ignores NaN but counts Inf, so a column holding an Inf has an Inf
mean), and matching it is what makes the parity test in
``backend/tests/test_stats_describe_node.py`` meaningful. Every reduction here
is written against an explicit ``~isnan`` mask rather than ``np.nanmean`` and
friends, because those warn on an all-NaN column and this code runs on engine
worker threads, where ``warnings.catch_warnings`` is process-global and so not
safe to use.

**Chart specs.** ``MEDIA_CHART`` payloads must be JSON-safe *and* small: the
run store serialises events with ``allow_nan=False`` and elides any payload
over ``RUN_EVENT_PAYLOAD_CAP_BYTES`` (128 KB). So every number goes through
:func:`_num`, which returns a plain finite Python ``float`` — a ``np.float32``
is not JSON-serialisable, and the run store's ``json_safe`` does not catch it
either, since only ``np.float64`` subclasses Python ``float``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

# ── chart payload caps ───────────────────────────────────────────────────────
# Sized so a spec stays well under RUN_EVENT_PAYLOAD_CAP_BYTES (128 KB) even
# with long labels. Past these the producer downsamples and says so in the
# spec's ``note``, which the renderer prints under the title: silently drawing
# the first N points would be a lie about the data.
MAX_BARS = 200
MAX_SERIES = 8
MAX_POINTS_PER_SERIES = 2000
MAX_SCATTER_POINTS = 2000
MAX_HEATMAP_SIDE = 64

#: Chart kinds the frontend renderer knows. Extending this list means teaching
#: ``frontend/src/components/shared/ChartView.tsx`` a new branch.
CHART_KINDS = ("bar", "line", "scatter", "heatmap")

#: Aggregations ``Stats-GroupByAggregate`` understands.
AGGREGATIONS = ("mean", "sum", "count", "min", "max", "std")


# ── table coercion ───────────────────────────────────────────────────────────

def as_matrix(value: Any, *, node: str, port: str = "table") -> np.ndarray:
    """Coerce a table port value to a 2D float64 numpy array.

    float64 rather than the float32 of the tensor convention: every statistic
    is computed here and only the *result* goes back out as float32, so
    accumulating a 150-row mean does not lose digits the parity test would
    then blame on the algorithm. A 1D input becomes a single column, which is
    what lets ``Stats-Describe`` sit directly on a vector.
    """
    if value is None:
        raise ValueError(f"{node} requires a `{port}` input.")
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().to(torch.float64).numpy()
    else:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{node}: `{port}` must be numeric; got {type(value).__name__}, "
                f"which numpy could not read as numbers ({exc})."
            ) from exc
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"{node} expects a 2D [rows, columns] table on `{port}`; "
            f"got shape {tuple(arr.shape)}."
        )
    return arr


def as_vector(value: Any, *, node: str, port: str = "tensor") -> np.ndarray:
    """Coerce a port value to a flat float64 numpy array."""
    if value is None:
        raise ValueError(f"{node} requires a `{port}` input.")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float64).numpy().reshape(-1)
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{node}: `{port}` must be numeric; got {type(value).__name__} ({exc})."
        ) from exc


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Hand a computed array back as float32, the pack's output dtype.

    ``ascontiguousarray`` because ``torch.from_numpy`` refuses a negative
    stride, which is exactly what a reversed or transposed view hands it.
    """
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def resolve_columns(names: Any, n_cols: int, *, prefix: str = "c") -> list[str]:
    """Column names for a table, padded or trimmed to ``n_cols``.

    A missing or short ``columns`` input is not an error: it is the normal
    state downstream of a ``ColumnSelector``, which drops the metadata it was
    given. Generated names carry the column index, so a graph that later
    reconnects the real names does not silently renumber anything.

    Duplicates are disambiguated with a ``#n`` suffix rather than letting the
    first one win — a duplicate name makes a rendered table unreadable and an
    ``agg_overrides`` lookup ambiguous.
    """
    resolved = [str(c) for c in names] if isinstance(names, (list, tuple)) else []
    if len(resolved) > n_cols:
        resolved = resolved[:n_cols]
    resolved.extend(f"{prefix}{i}" for i in range(len(resolved), n_cols))
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in resolved:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}#{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def as_labels(value: Any, *, node: str, port: str) -> list[str]:
    """Read a per-row label list from a LIST port or a 1D/2D tensor.

    Accepts what real graphs produce: ``CSVReader.labels`` (already strings), a
    class-index tensor from an ``argmax``, or a one-hot / logit matrix (row
    argmax is taken, which is what a classifier's raw output is). Everything
    becomes ``str`` because a confusion matrix is indexed by class *name*.
    """
    if value is None:
        raise ValueError(f"{node} requires a `{port}` input.")
    if isinstance(value, torch.Tensor):
        work = value.detach().cpu()
        if work.ndim >= 2 and work.shape[-1] > 1:
            work = work.argmax(dim=-1)
        flat = work.reshape(-1).tolist()
        return [_label_of(v) for v in flat]
    if isinstance(value, np.ndarray):
        work_np = value
        if work_np.ndim >= 2 and work_np.shape[-1] > 1:
            work_np = work_np.argmax(axis=-1)
        return [_label_of(v) for v in work_np.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [_label_of(v) for v in value]
    raise ValueError(
        f"{node}: `{port}` must be a list of labels or a tensor of class "
        f"indices; got {type(value).__name__}."
    )


def _label_of(value: Any) -> str:
    """Stringify one class label, keeping integral floats readable.

    A class-index tensor arrives as float (``2.0``), and ``"2.0"`` would not
    match the ``"2"`` a sibling list-of-strings port carries for the same
    class. Collapsing integral floats is what lets predictions and ground
    truth come from different node types.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_names(raw: Any) -> list[str]:
    """Split a comma-separated STRING param into trimmed, non-empty names."""
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_percentiles(raw: Any, *, node: str, param: str) -> list[float]:
    """Parse a comma-separated percentile list, in percent (0-100).

    Percent rather than pandas' fractions because the value is typed into a
    text box: ``25,50,75`` is unambiguous where ``0.25,0.5,0.75`` invites
    someone to write ``50`` and get the 50th of a percent.
    """
    values: list[float] = []
    for part in parse_names(raw):
        try:
            q = float(part)
        except ValueError as exc:
            raise ValueError(
                f"{node}: `{param}` entry {part!r} is not a number "
                f"(expected percentages such as '25,50,75')."
            ) from exc
        if not 0.0 <= q <= 100.0:
            raise ValueError(
                f"{node}: `{param}` entry {q:g} is outside 0-100."
            )
        values.append(q)
    return values


def _number_param(params: Any, name: str, default: Any, node: str, cast: Any) -> Any:
    """Read a numeric param without the ``or default`` falsy-zero trap.

    ``int(params.get("bins", 20) or 20)`` reads fine and silently turns a
    deliberate ``0`` into ``20``, so an out-of-range value the node means to
    reject never reaches the check. Only ``None`` and ``""`` — a cleared
    widget — fall back to the default here.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        return cast(default)
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{node}: `{name}` must be a number; got {raw!r} ({exc})."
        ) from exc


def int_param(params: Any, name: str, default: int, *, node: str) -> int:
    return _number_param(params, name, default, node, int)


def float_param(params: Any, name: str, default: float, *, node: str) -> float:
    return _number_param(params, name, default, node, float)


def format_percentile(q: float) -> str:
    """pandas' label for a percentile: ``25%``, ``2.5%``, ``99.9%``."""
    text = f"{q:g}"
    return f"{text}%"


# ── NaN-aware reductions (NaN is missing, ±Inf is a value) ───────────────────

def present(column: np.ndarray) -> np.ndarray:
    """The non-missing entries of one column."""
    return column[~np.isnan(column)]


def mean_of(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def std_of(values: np.ndarray, ddof: int = 1) -> float:
    """Sample std by default — pandas' ``describe`` uses ddof=1."""
    if values.size <= ddof:
        return float("nan")
    return float(values.std(ddof=ddof))


def min_of(values: np.ndarray) -> float:
    return float(values.min()) if values.size else float("nan")


def max_of(values: np.ndarray) -> float:
    return float(values.max()) if values.size else float("nan")


def percentile_of(values: np.ndarray, q: float) -> float:
    """Linear-interpolated percentile — numpy's and pandas' shared default."""
    if not values.size:
        return float("nan")
    return float(np.percentile(values, q))


# ── chart specs ──────────────────────────────────────────────────────────────

def _num(value: Any) -> float:
    """A plain, finite Python float — the only numeric type a spec may carry."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def _base(kind: str, title: str, x_label: str, y_label: str) -> dict[str, Any]:
    spec: dict[str, Any] = {"kind": kind, "title": str(title)}
    if x_label:
        spec["x_label"] = str(x_label)
    if y_label:
        spec["y_label"] = str(y_label)
    return spec


def _note(spec: dict[str, Any], text: str) -> None:
    """Record a downsampling / substitution notice on the spec."""
    existing = spec.get("note")
    spec["note"] = f"{existing} {text}".strip() if existing else text


def bar_chart(
    labels: Sequence[Any],
    values: Sequence[Any],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict[str, Any]:
    """A categorical bar chart: one labelled bar per value."""
    spec = _base("bar", title, x_label, y_label)
    pairs = list(zip([str(label) for label in labels], [_num(v) for v in values]))
    if len(pairs) > MAX_BARS:
        _note(spec, f"showing the first {MAX_BARS} of {len(pairs)} bars")
        pairs = pairs[:MAX_BARS]
    spec["bars"] = [{"label": label, "value": value} for label, value in pairs]
    return spec


def line_chart(
    series: Sequence[tuple[str, Sequence[Any], Sequence[Any]]],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict[str, Any]:
    """One or more named lines, each given as ``(name, xs, ys)``.

    Points with a non-finite x or y are dropped rather than pinned to 0: a
    gap in a curve is honest, a spike to the origin is not.
    """
    spec = _base("line", title, x_label, y_label)
    kept = list(series)
    if len(kept) > MAX_SERIES:
        _note(spec, f"showing {MAX_SERIES} of {len(kept)} series")
        kept = kept[:MAX_SERIES]
    out: list[dict[str, Any]] = []
    for name, xs, ys in kept:
        points = [
            [float(x), float(y)]
            for x, y in zip(xs, ys)
            if np.isfinite(float(x)) and np.isfinite(float(y))
        ]
        if len(points) > MAX_POINTS_PER_SERIES:
            step = (len(points) + MAX_POINTS_PER_SERIES - 1) // MAX_POINTS_PER_SERIES
            _note(spec, f"series sampled every {step} points")
            points = points[::step]
        out.append({"name": str(name), "points": points})
    spec["series"] = out
    return spec


def scatter_chart(
    xs: Sequence[Any],
    ys: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    clusters: Sequence[Any] | None = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict[str, Any]:
    """A point cloud. Non-finite points are dropped, as in :func:`line_chart`."""
    spec = _base("scatter", title, x_label, y_label)
    points: list[dict[str, Any]] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        fx, fy = float(x), float(y)
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        point: dict[str, Any] = {"x": fx, "y": fy}
        if labels is not None and i < len(labels):
            point["label"] = str(labels[i])
        if clusters is not None and i < len(clusters):
            point["cluster"] = int(clusters[i])
        points.append(point)
    if len(points) > MAX_SCATTER_POINTS:
        step = (len(points) + MAX_SCATTER_POINTS - 1) // MAX_SCATTER_POINTS
        _note(spec, f"sampled every {step} of {len(points)} points")
        points = points[::step]
    spec["points"] = points
    return spec


def heatmap_chart(
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    colormap: str = "viridis",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    value_format: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Any]:
    """A labelled matrix with an explicit value range for colour scaling.

    ``vmin``/``vmax`` travel with the spec because the renderer must colour a
    raw count matrix (0..N) and a row-normalised one (0..1) on the same
    component without inventing a scale of its own. They default to the data's
    own extent; pass them to pin a scale the data does not reach — a
    correlation matrix is read against a fixed -1..1 whether or not this
    particular one contains a -1.
    """
    spec = _base("heatmap", title, x_label, y_label)
    work = np.asarray(matrix, dtype=np.float64)
    if work.ndim != 2:
        raise ValueError(
            f"heatmap needs a 2D matrix; got shape {tuple(work.shape)}."
        )
    rows = list(row_labels or [])
    cols = list(col_labels or [])
    if work.shape[0] > MAX_HEATMAP_SIDE or work.shape[1] > MAX_HEATMAP_SIDE:
        _note(
            spec,
            f"showing the first {MAX_HEATMAP_SIDE}x{MAX_HEATMAP_SIDE} of "
            f"{work.shape[0]}x{work.shape[1]} cells",
        )
        work = work[:MAX_HEATMAP_SIDE, :MAX_HEATMAP_SIDE]
        rows = rows[:MAX_HEATMAP_SIDE]
        cols = cols[:MAX_HEATMAP_SIDE]
    finite = work[np.isfinite(work)]
    if finite.size < work.size:
        _note(spec, "non-finite cells drawn as 0")
    spec["matrix"] = [[_num(v) for v in row] for row in work.tolist()]
    spec["row_labels"] = [str(r) for r in rows]
    spec["col_labels"] = [str(c) for c in cols]
    spec["vmin"] = vmin if vmin is not None else (
        float(finite.min()) if finite.size else 0.0
    )
    spec["vmax"] = vmax if vmax is not None else (
        float(finite.max()) if finite.size else 1.0
    )
    spec["colormap"] = colormap
    if value_format:
        spec["value_format"] = value_format
    return spec


# ── text rendering ───────────────────────────────────────────────────────────

def format_number(value: float, precision: int) -> str:
    """One cell of a rendered table. NaN prints as the pandas-ish ``NaN``."""
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.{precision}f}"


def format_text_table(
    matrix: np.ndarray,
    columns: Sequence[str],
    row_labels: Sequence[str],
    *,
    max_rows: int,
    precision: int,
) -> str:
    """Render a table as fixed-width text for the Results log.

    Columns are right-aligned on their widest cell, which is what makes a
    column of numbers scannable; the row-label gutter is left-aligned.
    """
    rows = matrix.shape[0]
    shown = rows if max_rows <= 0 else min(rows, max_rows)
    labels = [str(r) for r in row_labels[:shown]]
    labels.extend(str(i) for i in range(len(labels), shown))
    gutter = max([len(label) for label in labels] + [0])

    cells: list[list[str]] = []
    for r in range(shown):
        cells.append([format_number(float(v), precision) for v in matrix[r]])

    widths = []
    for c, name in enumerate(columns):
        widest = max([len(row[c]) for row in cells] + [len(str(name))])
        widths.append(widest)

    lines = [
        " ".join([" " * gutter] + [str(n).rjust(w) for n, w in zip(columns, widths)])
    ]
    for label, row in zip(labels, cells):
        lines.append(
            " ".join([label.ljust(gutter)] + [v.rjust(w) for v, w in zip(row, widths)])
        )
    if shown < rows:
        lines.append(f"... {rows - shown} more row(s) of {rows}")
    return "\n".join(lines)
