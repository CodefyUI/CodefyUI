"""Tests for TextCorpusDatasetNode (#290).

Network-free throughout: the ``huggingface`` source is exercised with
``datasets.load_dataset`` monkeypatched to build an in-memory
``datasets.Dataset``, so the assertions cover the argument the node PASSES (the
split-slicing string, ``trust_remote_code=False``) and how it classifies a
failure, which is where the bugs actually live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.nodes.llm.text_corpus_dataset_node import TextCorpusDatasetNode


def _local(path: Path, **params) -> dict:
    return TextCorpusDatasetNode().execute(
        {}, {"source": "local_file", "local_path": str(path), **params})


def _hf(**params) -> dict:
    return TextCorpusDatasetNode().execute(
        {}, {"source": "huggingface", **params})


@pytest.fixture
def fake_load_dataset(monkeypatch):
    """Replace ``datasets.load_dataset`` with a recorder over fixed rows.

    Returns the call log; the rows and the exception it raises are settable so
    one fixture covers the happy path, the missing-column path and both
    failure classifications.
    """
    from datasets import Dataset

    calls: list[dict] = []
    state = {
        "rows": {"text": ["alpha", "beta", "gamma"]},
        "error": None,
    }

    def fake(name, subset=None, *, split=None, cache_dir=None,
             trust_remote_code=None):
        calls.append({
            "name": name, "subset": subset, "split": split,
            "cache_dir": cache_dir, "trust_remote_code": trust_remote_code,
        })
        if state["error"] is not None:
            raise state["error"]
        return Dataset.from_dict(state["rows"])

    monkeypatch.setattr("datasets.load_dataset", fake)
    fake.calls = calls
    fake.state = state
    return fake


# ── metadata ────────────────────────────────────────────────────────────


def test_node_metadata():
    assert TextCorpusDatasetNode.NODE_NAME == "TextCorpusDataset"
    assert TextCorpusDatasetNode.CATEGORY == "LLM"
    assert TextCorpusDatasetNode.define_inputs() == []
    assert [p.name for p in TextCorpusDatasetNode.define_outputs()] == [
        "dataset", "num_rows"]


def test_hf_params_are_hidden_in_local_mode_and_the_reverse():
    params = {p.name: p for p in TextCorpusDatasetNode.define_params()}
    for name in ("dataset_name", "subset", "split", "text_column", "cache_dir"):
        assert params[name].visible_when == {"source": "huggingface"}, name
    for name in ("local_path", "split_lines"):
        assert params[name].visible_when == {"source": "local_file"}, name
    # max_rows applies to both sources, so it must NOT be conditional.
    assert params["max_rows"].visible_when is None
    assert params["dataset_name"].default == "roneneldan/TinyStories"


def test_the_node_is_not_cacheable():
    assert TextCorpusDatasetNode.cacheable is False


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        TextCorpusDatasetNode().execute({}, {"source": "s3"})


# ── local_file source ───────────────────────────────────────────────────


def test_local_file_is_one_document_by_default(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line one\nline two\n", encoding="utf-8")

    res = _local(corpus)
    assert res["num_rows"] == 1
    assert len(res["dataset"]) == 1
    assert res["dataset"][0] == "line one\nline two\n"


def test_split_lines_makes_one_row_per_line(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("one\ntwo\n\n  \nthree\n", encoding="utf-8")

    res = _local(corpus, split_lines=True)
    # Blank and whitespace-only lines are dropped: an empty document would
    # contribute nothing but an EOS token to the packed stream downstream.
    assert res["num_rows"] == 3
    assert [res["dataset"][i] for i in range(3)] == ["one", "two", "three"]


def test_max_rows_truncates_a_line_split_file(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a\nb\nc\nd\n", encoding="utf-8")

    res = _local(corpus, split_lines=True, max_rows=2)
    assert res["num_rows"] == 2
    assert [res["dataset"][i] for i in range(2)] == ["a", "b"]


def test_max_rows_zero_means_every_row(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a\nb\nc\n", encoding="utf-8")

    assert _local(corpus, split_lines=True, max_rows=0)["num_rows"] == 3


def test_an_empty_file_yields_no_rows(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("   \n\n", encoding="utf-8")

    res = _local(corpus)
    assert res["num_rows"] == 0
    assert len(res["dataset"]) == 0


def test_a_missing_local_path_names_the_param():
    with pytest.raises(ValueError, match="local_path"):
        TextCorpusDatasetNode().execute(
            {}, {"source": "local_file", "local_path": "  "})


def test_a_nonexistent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _local(tmp_path / "nope.txt")


def test_a_bare_filename_resolves_against_the_upload_directory(
        tmp_path, monkeypatch):
    uploads = tmp_path / "files"
    uploads.mkdir()
    (uploads / "uploaded.txt").write_text("from the dropdown", encoding="utf-8")
    monkeypatch.setattr(settings, "DATA_FILES_DIR", uploads)

    res = TextCorpusDatasetNode().execute(
        {}, {"source": "local_file", "local_path": "uploaded.txt"})
    assert res["dataset"][0] == "from the dropdown"


def test_a_non_utf8_file_says_so(tmp_path):
    corpus = tmp_path / "latin1.txt"
    corpus.write_bytes("caf\xe9".encode("latin-1"))

    with pytest.raises(ValueError, match="UTF-8"):
        _local(corpus)


def test_the_local_dataset_is_indexable_and_sized(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a\nb\n", encoding="utf-8")

    dataset = _local(corpus, split_lines=True)["dataset"]
    assert len(dataset) == 2
    assert list(dataset) == ["a", "b"]
    with pytest.raises(IndexError):
        dataset[2]


# ── huggingface source ──────────────────────────────────────────────────


def test_hf_rows_come_back_as_strings(fake_load_dataset):
    res = _hf()
    assert res["num_rows"] == 3
    assert res["dataset"][0] == "alpha"
    assert all(isinstance(res["dataset"][i], str) for i in range(3))


def test_hf_defaults_pass_no_subset_and_refuse_remote_code(fake_load_dataset):
    _hf()
    call = fake_load_dataset.calls[0]
    assert call["name"] == "roneneldan/TinyStories"
    # Empty string -> None: `datasets` treats "" as a config named "".
    assert call["subset"] is None
    assert call["split"] == "train"
    assert call["cache_dir"] is None
    # The dataset-script RCE guard, same as HuggingFaceDataset.
    assert call["trust_remote_code"] is False


def test_hf_max_rows_becomes_split_slicing(fake_load_dataset):
    fake_load_dataset.state["rows"] = {"text": ["alpha", "beta"]}
    _hf(max_rows=2)
    # Sliced in the SPLIT rather than after loading: the point of the cap is
    # not downloading the other 2 million stories.
    assert fake_load_dataset.calls[0]["split"] == "train[:2]"


def test_hf_max_rows_leaves_an_already_sliced_split_alone(fake_load_dataset):
    _hf(split="train[:3]", max_rows=2)
    # "train[:3][:2]" is not valid HF split syntax; the explicit slice wins and
    # the cap is applied to the rows that came back instead.
    assert fake_load_dataset.calls[0]["split"] == "train[:3]"


def test_hf_max_rows_still_caps_rows_a_sliced_split_returned(fake_load_dataset):
    res = _hf(split="train[:3]", max_rows=2)
    assert res["num_rows"] == 2
    assert [res["dataset"][i] for i in range(2)] == ["alpha", "beta"]


def test_hf_missing_text_column_lists_what_is_there(fake_load_dataset):
    with pytest.raises(RuntimeError, match=r"story.*\['text'\]"):
        _hf(text_column="story")


def test_hf_auth_failure_points_at_hf_token(fake_load_dataset):
    class GatedRepoError(Exception):
        pass

    fake_load_dataset.state["error"] = GatedRepoError("gated dataset")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _hf(dataset_name="meta/private")


def test_hf_401_is_also_read_as_an_auth_failure(fake_load_dataset):
    fake_load_dataset.state["error"] = OSError("401 Client Error: Unauthorized")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _hf()


def test_hf_other_failures_are_not_relabelled(fake_load_dataset):
    fake_load_dataset.state["error"] = ValueError("unknown split 'trian'")
    with pytest.raises(ValueError, match="unknown split"):
        _hf(split="trian")


def test_hf_none_in_the_text_column_becomes_an_empty_string(fake_load_dataset):
    # A real corpus column is nullable; str(None) == "None" would inject the
    # literal word into the training stream.
    fake_load_dataset.state["rows"] = {"text": ["alpha", None]}
    res = _hf()
    assert res["dataset"][1] == ""
