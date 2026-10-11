"""Tests for DataMixDatasetNode (#300)."""

from __future__ import annotations

import pytest

from torch.utils.data import Dataset

from app.nodes.llm.data_mix_dataset_node import DataMixDatasetNode


class LocalTextListDataset(Dataset):
    """Minimal rows-of-strings corpus for the tests."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _corpora(*sizes, prefix=("a", "b", "c", "d")):
    return [
        LocalTextListDataset([f"{prefix[i]}{row}" for row in range(size)])
        for i, size in enumerate(sizes)
    ]


def _mix(corpora, params=None):
    inputs = {f"corpus_{i + 1}": corpus for i, corpus in enumerate(corpora)}
    return DataMixDatasetNode().execute(
        inputs, {"sources": len(corpora), **(params or {})})


def test_node_metadata_and_dynamic_ports():
    assert DataMixDatasetNode.NODE_NAME == "DataMixDataset"
    assert DataMixDatasetNode.CATEGORY == "LLM"
    assert DataMixDatasetNode.cacheable is False
    assert [p.name for p in DataMixDatasetNode.define_inputs_dynamic({"sources": 4})] == [
        "corpus_1", "corpus_2", "corpus_3", "corpus_4"]
    assert len(DataMixDatasetNode.define_inputs_dynamic({"sources": 99})) == 6


def test_concat_is_an_ordered_curriculum():
    result = _mix(_corpora(3, 2), {"mode": "concat"})
    assert result["num_rows"] == 5
    assert [result["dataset"][i] for i in range(5)] == ["a0", "a1", "a2", "b0", "b1"]


def test_interleave_is_deterministic_per_seed_and_exhaustive():
    first = _mix(_corpora(20, 10), {"seed": 7, "weights": "0.5, 0.5"})
    second = _mix(_corpora(20, 10), {"seed": 7, "weights": "0.5, 0.5"})
    third = _mix(_corpora(20, 10), {"seed": 8, "weights": "0.5, 0.5"})
    rows = [first["dataset"][i] for i in range(first["num_rows"])]
    assert rows == [second["dataset"][i] for i in range(second["num_rows"])]
    assert rows != [third["dataset"][i] for i in range(third["num_rows"])]
    # Without replacement: every row appears exactly once.
    assert sorted(rows) == sorted(f"a{i}" for i in range(20)) + sorted(
        f"b{i}" for i in range(10)) or len(set(rows)) == 30
    assert first["num_rows"] == 30


def test_interleave_respects_the_weights_roughly():
    result = _mix(_corpora(8000, 8000), {"weights": "0.8, 0.2", "seed": 3})
    head = [result["dataset"][i] for i in range(2000)]
    share_a = sum(1 for row in head if row.startswith("a")) / len(head)
    assert 0.74 <= share_a <= 0.86


def test_rows_within_a_source_keep_their_order():
    result = _mix(_corpora(50, 50), {"seed": 5})
    seen_a = [row for row in (result["dataset"][i] for i in range(100))
              if row.startswith("a")]
    assert seen_a == [f"a{i}" for i in range(50)]


def test_exhausted_source_hands_the_tail_to_the_rest():
    result = _mix(_corpora(3, 300), {"weights": "0.5, 0.5", "seed": 1})
    tail = [result["dataset"][i] for i in range(result["num_rows"])][-200:]
    assert all(row.startswith("b") for row in tail)


def test_weight_count_mismatch_and_bad_values_are_refused():
    with pytest.raises(ValueError, match="one weight per corpus"):
        _mix(_corpora(2, 2), {"weights": "1"})
    with pytest.raises(ValueError, match="numbers"):
        _mix(_corpora(2, 2), {"weights": "a, b"})
    with pytest.raises(ValueError, match="positive"):
        _mix(_corpora(2, 2), {"weights": "1, 0"})


def test_missing_corpus_and_empty_corpus_are_refused():
    with pytest.raises(ValueError, match="corpus_2 is not connected"):
        DataMixDatasetNode().execute(
            {"corpus_1": LocalTextListDataset(["x"])}, {"sources": 2})
    with pytest.raises(RuntimeError, match="corpus_2 has no rows"):
        _mix([LocalTextListDataset(["x"]), LocalTextListDataset([])])


# ── #695: ratio mode, where the weights set the output share ────────────


def _rows_of(result):
    return [result["dataset"][i] for i in range(result["num_rows"])]


def _share(rows, prefix):
    return sum(1 for row in rows if row.startswith(prefix)) / len(rows)


def test_ratio_mode_hits_the_weights_on_equal_corpora():
    """The issue's repro: two 1000-row corpora at 0.95/0.05. interleave
    returns 1000 of each; ratio returns 95% from corpus_1."""
    rows = _rows_of(_mix(_corpora(1000, 1000),
                         {"mode": "ratio", "weights": "0.95, 0.05",
                          "total_rows": 2000, "seed": 0}))
    assert len(rows) == 2000
    assert sum(row.startswith("a") for row in rows) == 1900
    assert sum(row.startswith("b") for row in rows) == 100


def test_ratio_mode_total_rows_zero_means_the_sum_of_the_corpus_sizes():
    result = _mix(_corpora(300, 100), {"mode": "ratio", "weights": "1, 3"})
    rows = _rows_of(result)
    assert result["num_rows"] == 400
    assert _share(rows, "a") == pytest.approx(0.25)


def test_ratio_mode_repeats_a_small_corpus_evenly():
    """A corpus asked for more rows than it has repeats them, and no row
    repeats before every row of that corpus has been used once."""
    rows = _rows_of(_mix(_corpora(5, 1000),
                         {"mode": "ratio", "weights": "0.5, 0.5",
                          "total_rows": 23, "seed": 4}))
    from_a = [row for row in rows if row.startswith("a")]
    assert len(from_a) == 12
    counts = {row: from_a.count(row) for row in set(from_a)}
    assert set(counts) == {f"a{i}" for i in range(5)}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_ratio_mode_subsamples_without_repeats_and_shuffles():
    rows = _rows_of(_mix(_corpora(1000, 1000),
                         {"mode": "ratio", "weights": "0.5, 0.5",
                          "total_rows": 200, "seed": 1}))
    assert len(set(rows)) == 200
    # Mixed through the whole output, not one corpus then the other.
    assert 0.3 <= _share(rows[:100], "a") <= 0.7


def test_ratio_mode_is_deterministic_per_seed():
    params = {"mode": "ratio", "weights": "0.7, 0.3", "total_rows": 50}
    first = _rows_of(_mix(_corpora(40, 40), {**params, "seed": 2}))
    again = _rows_of(_mix(_corpora(40, 40), {**params, "seed": 2}))
    other = _rows_of(_mix(_corpora(40, 40), {**params, "seed": 3}))
    assert first == again
    assert first != other


def test_ratio_mode_quotas_sum_to_total_rows_for_three_sources():
    rows = _rows_of(_mix(_corpora(10, 10, 10),
                         {"sources": 3, "mode": "ratio", "weights": "1, 1, 1",
                          "total_rows": 10, "seed": 0}))
    assert len(rows) == 10
    assert sorted(sum(r.startswith(p) for r in rows) for p in "abc") == [3, 3, 4]


def test_interleave_keeps_the_corpus_sizes_whatever_the_weights():
    """interleave stays available and does what its text says: every row
    once, so the weights cannot change the share."""
    rows = _rows_of(_mix(_corpora(1000, 1000),
                         {"weights": "0.95, 0.05", "seed": 0}))
    assert len(rows) == 2000
    assert _share(rows, "a") == 0.5


def test_the_descriptions_say_what_the_weights_control():
    params = {p.name: p for p in DataMixDatasetNode.define_params()}
    assert params["mode"].options == ["interleave", "ratio", "concat"]
    assert params["mode"].default == "interleave"
    assert params["total_rows"].default == 0
    weights_text = params["weights"].description
    assert "share" in weights_text and "only" in weights_text
    assert "corpus sizes" in DataMixDatasetNode.DETAILS
