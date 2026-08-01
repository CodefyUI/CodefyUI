"""Unit tests for ``app.core.port_stats`` — the summary-statistics compute.

The reference for every numeric assertion is numpy, computed over the same
values, because "is this number right" must not be answered by the code under
test. Where the module deliberately deviates from a naive numpy call (NaN and
Inf are excluded from mean/std/quantiles rather than poisoning them) the test
builds the filtered array explicitly.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from app.core.port_stats import (
    CLASS_BALANCE_MAX,
    HISTOGRAM_BINS,
    PortStatsCache,
    compute_port_stats,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(a, p))


def _assert_json_safe(payload: dict) -> str:
    """Round-trip through the same encoder Starlette's JSONResponse uses."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


# ── tensor branch: exactness against numpy ───────────────────────────────────


#: Stats are emitted as plain float64 (``port_stats._num`` does not round), so
#: the only slack needed against numpy is torch's own accumulation order.
ROUNDING_REL = 1e-12


def test_float_tensor_matches_numpy_reference():
    a = np.linspace(-3.5, 7.25, 501, dtype=np.float64)
    stats = compute_port_stats(torch.from_numpy(a.copy()))

    assert stats["kind"] == "tensor"
    assert stats["sampled"] is False
    assert stats["sample_size"] is None
    assert stats["shape"] == [501]
    assert stats["dtype"] == "torch.float64"
    assert stats["device"] == "cpu"
    assert stats["count"] == 501

    assert stats["mean"] == pytest.approx(float(np.mean(a)), rel=ROUNDING_REL)
    assert stats["std"] == pytest.approx(float(np.std(a)), rel=ROUNDING_REL)
    assert stats["min"] == pytest.approx(float(np.min(a)), rel=ROUNDING_REL)
    assert stats["max"] == pytest.approx(float(np.max(a)), rel=ROUNDING_REL)

    for name, p in (
        ("p1", 0.01),
        ("p5", 0.05),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p95", 0.95),
        ("p99", 0.99),
    ):
        assert stats["quantiles"][name] == pytest.approx(_q(a, p), rel=ROUNDING_REL)

    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0
    assert stats["zero_frac"] == pytest.approx(0.0)


def test_multi_dim_tensor_reports_full_shape_and_flattened_count():
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    stats = compute_port_stats(t)
    assert stats["shape"] == [2, 3, 4]
    assert stats["count"] == 24
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(23.0)


def test_histogram_shape_and_totals():
    a = np.random.default_rng(7).normal(size=5000)
    stats = compute_port_stats(torch.from_numpy(a))

    hist = stats["histogram"]
    assert stats["value_counts"] is None
    assert hist["bins"] == HISTOGRAM_BINS
    assert len(hist["edges"]) == HISTOGRAM_BINS + 1
    assert len(hist["counts"]) == HISTOGRAM_BINS
    assert sum(hist["counts"]) == 5000
    assert hist["edges"][0] == pytest.approx(float(a.min()), rel=ROUNDING_REL)
    assert hist["edges"][-1] == pytest.approx(float(a.max()), rel=ROUNDING_REL)


def test_constant_tensor_collapses_to_one_bin():
    stats = compute_port_stats(torch.full((100,), 2.5))
    assert stats["min"] == pytest.approx(2.5)
    assert stats["max"] == pytest.approx(2.5)
    assert stats["std"] == pytest.approx(0.0)
    hist = stats["histogram"]
    assert hist["bins"] == 1
    assert hist["counts"] == [100]


def test_zero_fraction_counts_exact_zeros():
    t = torch.tensor([0.0, 0.0, 1.0, -1.0])
    stats = compute_port_stats(t)
    assert stats["zero_frac"] == pytest.approx(0.5)


# ── NaN / Inf: the headline debugging stat ───────────────────────────────────


def test_nan_and_inf_are_counted_and_excluded_from_moments():
    raw = [1.0, 2.0, float("nan"), float("inf"), -float("inf"), 3.0, float("nan")]
    stats = compute_port_stats(torch.tensor(raw, dtype=torch.float64))
    finite = np.array([1.0, 2.0, 3.0])

    assert stats["nan_count"] == 2
    assert stats["inf_count"] == 2
    assert stats["count"] == 7
    assert stats["mean"] == pytest.approx(float(np.mean(finite)))
    assert stats["std"] == pytest.approx(float(np.std(finite)))
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)
    _assert_json_safe(stats)


def test_all_nan_tensor_reports_none_rather_than_nan():
    stats = compute_port_stats(torch.full((16,), float("nan")))
    assert stats["nan_count"] == 16
    assert stats["mean"] is None
    assert stats["std"] is None
    assert stats["min"] is None
    assert stats["max"] is None
    assert stats["histogram"] is None
    assert stats["quantiles"] == {}
    _assert_json_safe(stats)


def test_inf_only_tensor_reports_none_rather_than_inf():
    t = torch.tensor([float("inf"), -float("inf"), float("inf")])
    stats = compute_port_stats(t)
    assert stats["inf_count"] == 3
    assert stats["nan_count"] == 0
    assert stats["min"] is None
    assert stats["max"] is None
    assert stats["mean"] is None
    assert stats["histogram"] is None
    _assert_json_safe(stats)


def test_empty_tensor_is_described_not_crashed():
    stats = compute_port_stats(torch.zeros(0))
    assert stats["kind"] == "tensor"
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["histogram"] is None
    _assert_json_safe(stats)


def test_zero_dim_tensor():
    stats = compute_port_stats(torch.tensor(4.5))
    assert stats["shape"] == []
    assert stats["count"] == 1
    assert stats["mean"] == pytest.approx(4.5)


def test_tensor_requiring_grad_is_detached():
    t = torch.ones(4, requires_grad=True) * 2
    stats = compute_port_stats(t)
    assert stats["mean"] == pytest.approx(2.0)


# ── sampling ─────────────────────────────────────────────────────────────────


def test_threshold_switches_to_sampled_mode():
    t = torch.arange(5000, dtype=torch.float32)

    exact = compute_port_stats(t, sample_threshold=10_000, sample_size=500)
    assert exact["sampled"] is False
    assert exact["sample_size"] is None

    sampled = compute_port_stats(t, sample_threshold=1000, sample_size=500)
    assert sampled["sampled"] is True
    assert sampled["sample_size"] == 500
    # count stays the TRUE element count; only the distribution is sampled.
    assert sampled["count"] == 5000


def test_same_seed_gives_the_same_sample():
    t = torch.randn(20_000)
    a = compute_port_stats(t, sample_threshold=100, sample_size=256, seed=99)
    b = compute_port_stats(t, sample_threshold=100, sample_size=256, seed=99)
    assert a == b


def test_different_seed_gives_a_different_sample():
    t = torch.randn(20_000)
    a = compute_port_stats(t, sample_threshold=100, sample_size=256, seed=1)
    b = compute_port_stats(t, sample_threshold=100, sample_size=256, seed=2)
    assert a["mean"] != b["mean"]


def test_sampled_min_max_and_nan_counts_stay_exact():
    """Sampling degrades the distribution, never the outlier/health counts."""
    t = torch.zeros(10_000)
    t[4321] = 999.0
    t[8765] = float("nan")

    stats = compute_port_stats(t, sample_threshold=100, sample_size=128)
    assert stats["sampled"] is True
    assert stats["max"] == pytest.approx(999.0)
    assert stats["min"] == pytest.approx(0.0)
    assert stats["nan_count"] == 1


def test_a_single_sample_activation_is_still_chunked_and_thinned():
    """`shape[0] == 1` is the commonest thing anyone inspects.

    A `[1, C, H, W]` activation permuted to channels-last is non-contiguous
    with a leading dim of 1. Splitting on dim 0 cannot cut it, so both the
    chunked scan and the pre-sampling thin have to look for the first axis
    that actually has length.
    """
    from app.core.port_stats import _iter_chunks, _narrow_for_sampling, _split_axis

    t = torch.randn(1, 64, 40, 40).permute(0, 2, 3, 1)
    assert not t.is_contiguous()
    assert _split_axis(t) == 1

    chunks = list(_iter_chunks(t, chunk_elements=10_000))
    assert len(chunks) > 1
    assert max(c.numel() for c in chunks) <= 40 * 40 * 64
    assert sum(c.numel() for c in chunks) == t.numel()

    thinned = _narrow_for_sampling(t, 1000, 0)
    assert thinned.numel() < t.numel()

    stats = compute_port_stats(t, sample_threshold=1000, sample_size=512)
    assert stats["count"] == t.numel()
    assert stats["sampled"] is True


def _periodic_axis_tensor(rows: int = 64, cols: int = 4096, period: int = 16):
    """Non-contiguous, with periodic structure ON the axis thinning splits.

    Row ``i`` is entirely ``1.0`` when ``i % period == 0`` and ``0.0``
    otherwise, so the true mean is ``1 / period``. Built transposed so
    :func:`_split_axis` picks the 64-long structured axis rather than the long
    featureless one - get that backwards and a stride looks unbiased, because
    it is then striding across the axis that carries no signal.
    """
    per_row = [1.0 if i % period == 0 else 0.0 for i in range(rows)]
    base = torch.stack([torch.full((cols,), v) for v in per_row], dim=1)
    return base.t()


def test_thinning_a_structured_axis_does_not_pick_a_stride():
    """A fixed stride reports whichever slices it happens to land on.

    The old ``t[::step]`` kept 4 of 64 rows at a step of 16 - every one of them
    a spike row - and reported a mean of 1.0 for data whose mean is 0.0625,
    under ``"sampled": true``. Aliasing is the failure a stride has and a draw
    does not, which is why this fixture is periodic rather than a ramp: a
    stride estimates a ramp perfectly well, so a ramp proves nothing.
    """
    from app.core.port_stats import _split_axis

    t = _periodic_axis_tensor()
    assert not t.is_contiguous()
    assert _split_axis(t) == 0, "the structured axis must be the split axis"
    assert t.mean().item() == pytest.approx(1 / 16)

    for seed in range(6):
        stats = compute_port_stats(
            t, sample_threshold=1000, sample_size=4096, seed=seed
        )
        assert stats["sampled"] is True
        # The stride returns exactly 1.0 for every seed; a draw of 4 rows from
        # 64 lands on all four spikes with probability ~1.5e-5.
        assert stats["mean"] < 0.5, f"seed {seed} looks strided: {stats['mean']}"


def test_thinning_stays_deterministic_under_a_seed():
    t = _periodic_axis_tensor(cols=2048)
    a = compute_port_stats(t, sample_threshold=1000, sample_size=2048, seed=7)
    b = compute_port_stats(t, sample_threshold=1000, sample_size=2048, seed=7)
    assert a == b


@pytest.mark.parametrize(
    "make, budget",
    [
        # Split axis is the 3-long channel dim, so one slice is 40k elements
        # against a 10k budget - the pre-fix chunker yielded it whole.
        (lambda: torch.zeros(1, 3, 200, 400)[:, :, :, ::2], 10_000),
        # Still over budget after the first recursion, so it has to go again.
        (lambda: torch.zeros(1, 3, 4, 20_000)[:, :, :, ::2], 1_000),
    ],
    ids=["one-oversized-slice", "two-level-recursion"],
)
def test_an_oversized_slice_recurses_onto_the_next_axis(make, budget):
    """A slice bigger than the budget used to be yielded whole.

    ``step`` clamps to 1 once a single slice already exceeds the budget, and
    the pre-fix chunker then handed that slice on - ``_scan_tensor`` would
    build a bool mask over all of it, which is the allocation chunking exists
    to prevent. Measured pre-fix: 40k-element chunks against both budgets.
    """
    from app.core.port_stats import _iter_chunks

    t = make()
    assert not t.is_contiguous()

    chunks = list(_iter_chunks(t, chunk_elements=budget))
    assert max(c.numel() for c in chunks) <= budget
    # Nothing skipped or double-counted, however deep the recursion went.
    assert sum(c.numel() for c in chunks) == t.numel()
    assert len(chunks) >= t.numel() // budget



def test_sampling_handles_non_contiguous_tensors():
    t = torch.randn(400, 50).t()  # non-contiguous view
    assert not t.is_contiguous()
    stats = compute_port_stats(t, sample_threshold=100, sample_size=256)
    assert stats["sampled"] is True
    assert stats["count"] == 20_000
    assert stats["mean"] is not None


def test_sampled_quantiles_track_the_exact_ones():
    """A sample must describe the same distribution the full sort does."""
    a = np.random.default_rng(3).normal(size=200_000)
    t = torch.from_numpy(a)

    exact = compute_port_stats(t)
    sampled = compute_port_stats(t, sample_threshold=1000, sample_size=50_000)

    assert exact["sampled"] is False
    assert sampled["sampled"] is True
    for name in ("p5", "p25", "p50", "p75", "p95"):
        assert sampled["quantiles"][name] == pytest.approx(
            exact["quantiles"][name], abs=0.05
        )


def test_a_lone_outlier_does_not_move_the_median():
    """The regression that killed the histogram-based quantile estimator.

    A diverged value is the normal reason to open this panel, so an estimator
    whose resolution is set by the full range is the wrong one.
    """
    a = np.random.default_rng(11).normal(size=300_000)
    a[123] = 1e6
    stats = compute_port_stats(torch.from_numpy(a), sample_threshold=1000, sample_size=50_000)

    assert stats["max"] == pytest.approx(1e6)  # the outlier is still visible
    assert abs(stats["quantiles"]["p50"]) < 0.05
    assert stats["quantiles"]["p99"] == pytest.approx(2.33, abs=0.1)


# ── integer / class-balance branch ───────────────────────────────────────────


def test_integer_labels_switch_to_class_balance():
    t = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int64)
    stats = compute_port_stats(t)

    assert stats["histogram"] is None
    assert stats["value_counts"] == [
        {"value": 0, "count": 1},
        {"value": 1, "count": 2},
        {"value": 2, "count": 3},
    ]
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0


def test_class_balance_counts_stay_exact_when_sampled():
    t = torch.tensor([0] * 900 + [1] * 100, dtype=torch.int64)
    stats = compute_port_stats(t, sample_threshold=10, sample_size=32)
    assert stats["sampled"] is True
    assert stats["value_counts"] == [
        {"value": 0, "count": 900},
        {"value": 1, "count": 100},
    ]


def test_bool_tensor_class_balance():
    t = torch.tensor([True, False, True, True])
    stats = compute_port_stats(t)
    assert stats["value_counts"] == [
        {"value": False, "count": 1},
        {"value": True, "count": 3},
    ]


def test_sparse_integer_values_still_class_balance():
    """Few distinct values, wide range — the span test misses, unique catches."""
    t = torch.tensor([0, 5000, 5000, 0, 5000], dtype=torch.int32)
    stats = compute_port_stats(t)
    assert stats["value_counts"] == [
        {"value": 0, "count": 2},
        {"value": 5000, "count": 3},
    ]


def test_sparse_integer_class_balance_is_dropped_rather_than_estimated():
    """A class balance that cannot be exact must not be shipped as one.

    The span test only catches labels numbered 0..C-1. For labels that are few
    but far apart, the fallback reads the whole tensor — so once the tensor is
    sampled there is no honest count to report and the payload falls back to a
    histogram instead of quoting a scaled-down guess.
    """
    t = torch.tensor([0, 5000] * 5000, dtype=torch.int64)

    exact = compute_port_stats(t)
    assert exact["value_counts"] == [
        {"value": 0, "count": 5000},
        {"value": 5000, "count": 5000},
    ]

    sampled = compute_port_stats(t, sample_threshold=100, sample_size=256)
    assert sampled["sampled"] is True
    assert sampled["value_counts"] is None
    assert sampled["histogram"] is not None


def test_class_balance_over_a_contiguous_span_survives_sampling():
    """Labels numbered 0..C-1 still get exact counts — bincount needs no sort."""
    t = torch.tensor([0] * 700 + [1] * 200 + [2] * 100, dtype=torch.int64)
    stats = compute_port_stats(t, sample_threshold=100, sample_size=64)
    assert stats["sampled"] is True
    assert stats["value_counts"] == [
        {"value": 0, "count": 700},
        {"value": 1, "count": 200},
        {"value": 2, "count": 100},
    ]


def test_large_integers_keep_their_precision():
    """float32 has 24 mantissa bits; an int64 near 1e15 would round flat."""
    base = 10**15
    t = torch.tensor([base, base + 1, base + 2, base + 3], dtype=torch.int64)
    stats = compute_port_stats(t)
    assert stats["mean"] == pytest.approx(base + 1.5, rel=1e-15)
    # The mean must not land outside the spread printed next to it.
    assert stats["quantiles"]["p25"] <= stats["mean"] <= stats["quantiles"]["p75"]


def test_wide_integer_tensor_uses_a_histogram():
    t = torch.arange(CLASS_BALANCE_MAX * 4, dtype=torch.int64)
    stats = compute_port_stats(t)
    assert stats["value_counts"] is None
    assert stats["histogram"]["bins"] == HISTOGRAM_BINS
    assert sum(stats["histogram"]["counts"]) == CLASS_BALANCE_MAX * 4


def test_exactly_sixty_four_distinct_values_is_still_class_balance():
    t = torch.arange(CLASS_BALANCE_MAX, dtype=torch.int64)
    stats = compute_port_stats(t)
    assert stats["value_counts"] is not None
    assert len(stats["value_counts"]) == CLASS_BALANCE_MAX
    assert stats["histogram"] is None


def test_sixty_five_distinct_values_tips_over_to_a_histogram():
    t = torch.arange(CLASS_BALANCE_MAX + 1, dtype=torch.int64)
    stats = compute_port_stats(t)
    assert stats["value_counts"] is None
    assert stats["histogram"] is not None


def test_negative_integer_labels():
    t = torch.tensor([-1, -1, 0, 1], dtype=torch.int64)
    stats = compute_port_stats(t)
    assert stats["value_counts"] == [
        {"value": -1, "count": 2},
        {"value": 0, "count": 1},
        {"value": 1, "count": 1},
    ]


# ── tabular branch ───────────────────────────────────────────────────────────


def test_columnar_dict_per_column_stats():
    value = {
        "age": [30, 40, 40, None],
        "city": ["taipei", "tainan", "taipei", "taipei"],
    }
    stats = compute_port_stats(value)

    assert stats["kind"] == "tabular"
    assert stats["rows"] == 4
    by_name = {c["name"]: c for c in stats["columns"]}

    age = by_name["age"]
    assert age["dtype"] == "int"
    assert age["count"] == 3
    assert age["missing"] == 1
    assert age["unique"] == 2
    assert age["min"] == pytest.approx(30.0)
    assert age["max"] == pytest.approx(40.0)
    assert age["mean"] == pytest.approx(110 / 3)

    city = by_name["city"]
    assert city["dtype"] == "str"
    assert city["missing"] == 0
    assert city["unique"] == 2
    assert city["top"][0] == {"value": "taipei", "count": 3}
    assert city["mean"] is None


def test_records_list_is_transposed_to_columns():
    value = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3}]
    stats = compute_port_stats(value)
    assert stats["kind"] == "tabular"
    assert stats["rows"] == 3
    by_name = {c["name"]: c for c in stats["columns"]}
    assert by_name["a"]["count"] == 3
    # The third record has no "b" at all — that is a missing value, not a row.
    assert by_name["b"]["count"] == 2
    assert by_name["b"]["missing"] == 1


def test_plain_list_becomes_one_column():
    value = ["setosa"] * 3 + ["virginica"]
    stats = compute_port_stats(value)
    assert stats["kind"] == "tabular"
    assert stats["rows"] == 4
    assert len(stats["columns"]) == 1
    col = stats["columns"][0]
    assert col["unique"] == 2
    assert col["top"] == [
        {"value": "setosa", "count": 3},
        {"value": "virginica", "count": 1},
    ]


def test_float_nan_in_a_column_counts_as_missing():
    stats = compute_port_stats({"x": [1.0, float("nan"), 3.0]})
    col = stats["columns"][0]
    assert col["missing"] == 1
    assert col["count"] == 2
    assert col["mean"] == pytest.approx(2.0)
    _assert_json_safe(stats)


def test_top_k_is_bounded_and_ordered():
    stats = compute_port_stats({"x": [str(i % 40) for i in range(400)]})
    col = stats["columns"][0]
    assert col["unique"] == 40
    assert len(col["top"]) <= 10
    counts = [entry["count"] for entry in col["top"]]
    assert counts == sorted(counts, reverse=True)


def test_empty_list_is_tabular_with_no_rows():
    stats = compute_port_stats([])
    assert stats["kind"] == "tabular"
    assert stats["rows"] == 0
    assert stats["columns"] == []


def test_wide_table_truncates_columns():
    value = {f"c{i}": [1, 2] for i in range(400)}
    stats = compute_port_stats(value)
    assert stats["columns_truncated"] is True
    assert stats["column_count"] == 400
    assert len(stats["columns"]) < 400


def test_row_cap_applies_before_the_table_is_materialised():
    """The cap has to bound the ALLOCATION, not just the analysis.

    Transposing a ten-million-row CSV in full costs a gigabyte of list slots
    and seconds on the GIL, all to build rows the summary then discards.
    """
    from app.core.port_stats import _as_columns, _row_budget

    records = [{"a": i, "b": i} for i in range(50_000)]
    columns, rows, column_count = _as_columns(records, max_rows=8000, max_columns=8)
    assert rows == 50_000  # the reported total stays honest
    assert column_count == 2
    assert all(len(cells) == _row_budget(2, 8000) for _, cells in columns)

    columnar = {f"c{i}": list(range(50_000)) for i in range(20)}
    columns, rows, column_count = _as_columns(columnar, max_rows=8000, max_columns=4)
    assert rows == 50_000
    assert column_count == 20
    assert len(columns) == 4
    assert all(len(cells) == _row_budget(4, 8000) for _, cells in columns)


def test_the_row_budget_is_applied_before_the_transpose_not_after():
    """The old order built every column at the whole-table budget, then cut.

    Counted through a list subclass, because "we sliced early" is only
    observable as cells never read. 64 columns x 200k rows is 12.8M reads to
    produce cells the summary then throws away.
    """
    from app.core.port_stats import _as_columns

    reads = 0

    class CountingDict(dict):
        def get(self, key, default=None):  # noqa: A003 - dict API
            nonlocal reads
            reads += 1
            return super().get(key, default)

    records = [CountingDict({f"c{i}": 1 for i in range(64)}) for _ in range(20_000)]
    columns, rows, column_count = _as_columns(records, max_rows=64_000, max_columns=64)

    assert rows == 20_000
    assert column_count == 64
    # Budget is 64_000 // 64 = 1000 rows per column, over 64 columns.
    assert reads == 64 * 1000
    assert all(len(cells) == 1000 for _, cells in columns)


def test_long_top_values_are_truncated():
    from app.core.port_stats import TOP_VALUE_MAX_CHARS

    essay = "x" * 5000
    stats = compute_port_stats({"notes": [essay, essay, "short"]})
    top = stats["columns"][0]["top"]
    assert len(top[0]["value"]) <= TOP_VALUE_MAX_CHARS + 3
    assert top[0]["value"].endswith("...")
    assert len(_assert_json_safe(stats)) < 2000


def test_duplicate_stringified_keys_collapse_to_one_column():
    stats = compute_port_stats({1: [1, 2], "1": [3, 4]})
    names = [c["name"] for c in stats["columns"]]
    assert names == ["1"]
    assert stats["column_count"] == 1


def test_long_table_samples_rows():
    value = {"x": list(range(50_000))}
    stats = compute_port_stats(value, tabular_max_rows=1000)
    assert stats["rows"] == 50_000
    assert stats["sampled"] is True
    assert stats["sample_size"] == 1000
    assert stats["columns"][0]["count"] == 1000


def test_row_budget_is_shared_across_columns():
    """The scan is interpreted Python — its cost must not scale with width."""
    narrow = compute_port_stats(
        {"a": list(range(40_000))}, tabular_max_rows=20_000, tabular_max_columns=8
    )
    wide = compute_port_stats(
        {f"c{i}": list(range(40_000)) for i in range(8)},
        tabular_max_rows=20_000,
        tabular_max_columns=8,
    )
    assert narrow["columns"][0]["count"] == 20_000
    # Same budget, eight columns -> an eighth of the rows each.
    assert wide["columns"][0]["count"] == 2500
    assert all(c["count"] == 2500 for c in wide["columns"])


# ── unsupported values ───────────────────────────────────────────────────────


def test_module_value_is_reported_as_unsupported():
    stats = compute_port_stats(torch.nn.Linear(2, 2))
    assert stats["kind"] == "unsupported"
    assert stats["type"] == "Linear"


def test_scalar_value_is_reported_as_unsupported():
    stats = compute_port_stats(3)
    assert stats["kind"] == "unsupported"
    assert stats["type"] == "int"


def test_complex_tensor_reports_metadata_only():
    stats = compute_port_stats(torch.ones(4, dtype=torch.complex64))
    assert stats["kind"] == "tensor"
    assert stats["shape"] == [4]
    assert stats["mean"] is None
    assert stats["histogram"] is None


def test_list_of_tensors_is_unsupported():
    stats = compute_port_stats([torch.ones(2), torch.ones(2)])
    assert stats["kind"] == "unsupported"


def test_sparse_tensor_reports_metadata_without_crashing():
    """Every pass here assumes a dense buffer; a sparse one must not 500."""
    # `to_sparse` rather than `sparse_coo_tensor`: the latter warns that it is
    # skipping invariant checks, and this suite keeps its output clean.
    sparse = torch.tensor([[0.0, 3.0], [4.0, 0.0]]).to_sparse()
    stats = compute_port_stats(sparse)
    assert stats["kind"] == "tensor"
    assert stats["shape"] == [2, 2]
    assert stats["mean"] is None


def test_half_precision_tensor_with_nans():
    t = torch.tensor([1.0, 2.0, float("nan")], dtype=torch.float16)
    stats = compute_port_stats(t)
    assert stats["nan_count"] == 1
    assert stats["mean"] == pytest.approx(1.5, rel=1e-3)
    assert stats["max"] == pytest.approx(2.0, rel=1e-3)


# ── payload budget ───────────────────────────────────────────────────────────


def test_payload_is_small_for_a_large_tensor():
    t = torch.randn(2_000_000)
    stats = compute_port_stats(t, sample_threshold=1_000_000, sample_size=100_000)
    assert stats["sampled"] is True
    assert len(_assert_json_safe(stats)) < 100_000


# ── bytes-bounded cache ──────────────────────────────────────────────────────


def _payload(n: int) -> dict:
    """A payload whose serialized size grows with n."""
    return {"blob": "x" * n}


def test_cache_returns_the_stored_payload():
    cache = PortStatsCache(max_bytes=10_000)
    cache.put(("r", "n", "p"), _payload(10))
    assert cache.get(("r", "n", "p")) == _payload(10)
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_miss_counts():
    cache = PortStatsCache(max_bytes=10_000)
    assert cache.get(("r", "n", "p")) is None
    assert cache.misses == 1


def test_cache_evicts_by_bytes_not_by_count():
    cache = PortStatsCache(max_bytes=1000)
    for i in range(20):
        cache.put(("r", "n", str(i)), _payload(200))

    assert cache.total_bytes <= 1000
    # 20 entries of ~215 bytes each cannot all fit in 1000 bytes.
    assert len(cache) < 20
    assert cache.get(("r", "n", "0")) is None
    assert cache.get(("r", "n", "19")) is not None


def test_cache_evicts_least_recently_used_first():
    # Two ~211-byte payloads fit, three do not.
    cache = PortStatsCache(max_bytes=500)
    cache.put(("a",), _payload(200))
    cache.put(("b",), _payload(200))
    cache.get(("a",))  # "a" is now the most recent
    cache.put(("c",), _payload(200))

    assert cache.get(("b",)) is None
    assert cache.get(("a",)) is not None
    assert cache.get(("c",)) is not None


def test_cache_refuses_an_entry_larger_than_the_budget():
    cache = PortStatsCache(max_bytes=100)
    cache.put(("big",), _payload(5000))
    assert cache.get(("big",)) is None
    assert cache.total_bytes == 0


def test_cache_replacing_a_key_updates_the_byte_total():
    cache = PortStatsCache(max_bytes=10_000)
    cache.put(("k",), _payload(100))
    first = cache.total_bytes
    cache.put(("k",), _payload(500))
    assert len(cache) == 1
    assert cache.total_bytes > first


def test_cache_clear_resets_bytes():
    cache = PortStatsCache(max_bytes=10_000)
    cache.put(("k",), _payload(100))
    cache.clear()
    assert len(cache) == 0
    assert cache.total_bytes == 0


def test_cache_is_disabled_by_a_zero_budget():
    cache = PortStatsCache(max_bytes=0)
    cache.put(("k",), _payload(1))
    assert cache.get(("k",)) is None


def test_no_stat_is_a_bare_nan():
    """Every float in the payload must survive ``allow_nan=False``."""
    t = torch.tensor([float("nan"), float("inf"), 0.0, 1.0])
    stats = compute_port_stats(t)
    text = _assert_json_safe(stats)
    assert "NaN" not in text and "Infinity" not in text
    assert math.isfinite(stats["zero_frac"])
