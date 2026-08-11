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
