"""Tests for TextCorpusDatasetNode.

HF Hub loading is metadata-only here (network integration belongs to the
extension suite, same policy as HuggingFaceDataset); the local-file path and
the HF adapter shape are exercised for real.
"""

from __future__ import annotations

import pytest

from app.nodes.data._hf_adapter import HFTorchTextDataset
from app.nodes.llm.text_corpus_dataset_node import TextCorpusDatasetNode


def test_node_metadata():
    assert TextCorpusDatasetNode.NODE_NAME == "TextCorpusDataset"
    assert TextCorpusDatasetNode.CATEGORY == "LLM"
    assert TextCorpusDatasetNode.define_inputs() == []
    assert TextCorpusDatasetNode.cacheable is False


def test_default_params_target_tinystories():
    params = {p.name: p for p in TextCorpusDatasetNode.define_params()}
    assert params["dataset_name"].default == "roneneldan/TinyStories"
    assert params["text_column"].default == "text"
    assert params["source"].options == ["huggingface", "local_file"]


def test_local_file_loads_one_sample_per_line(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("first story\n\nsecond story\nthird story\n", encoding="utf-8")
    result = TextCorpusDatasetNode().execute(
        {}, {"source": "local_file", "local_file": str(corpus)},
    )
    dataset = result["dataset"]
    assert result["num_rows"] == 3
    assert len(dataset) == 3
    assert dataset[0] == "first story"
    assert dataset[2] == "third story"


def test_local_file_max_rows_caps_the_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(f"row {i}" for i in range(10)), encoding="utf-8")
    result = TextCorpusDatasetNode().execute(
        {}, {"source": "local_file", "local_file": str(corpus), "max_rows": 4},
    )
    assert result["num_rows"] == 4


def test_missing_local_file_has_actionable_error(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        TextCorpusDatasetNode().execute(
            {}, {"source": "local_file", "local_file": str(tmp_path / "nope.txt")},
        )


def test_blank_local_file_param_is_refused():
    with pytest.raises(ValueError, match="local_file"):
        TextCorpusDatasetNode().execute({}, {"source": "local_file"})


def test_hf_text_adapter_returns_plain_strings():
    class FakeHFDataset:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

    ds = HFTorchTextDataset(
        FakeHFDataset([{"text": "hello"}, {"text": "world"}]), text_column="text",
    )
    assert len(ds) == 2
    assert ds[0] == "hello"
    assert ds[1] == "world"
