"""Tests for LMTokenizedDatasetNode (offline: a fake tokenizer drives packing)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from app.nodes.data._hf_adapter import LocalTextListDataset
from app.nodes.llm.lm_tokenized_dataset_node import LMTokenizedDatasetNode


class FakeTokenizer:
    """Deterministic stand-in honoring the LMTokenizer handle contract."""

    encoding_name = "fake-1"
    eos_id = 99

    def __init__(self):
        self.batch_calls = 0

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 90 for ch in text]

    def encode_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        self.batch_calls += 1
        return [self.encode(t) for t in texts]


def run(rows, params=None, tokenizer=None):
    tokenizer = tokenizer or FakeTokenizer()
    result = LMTokenizedDatasetNode().execute(
        {"dataset": LocalTextListDataset(rows), "tokenizer": tokenizer},
        {"cache": False, **(params or {})},
    )
    return result, tokenizer


def test_node_metadata():
    assert LMTokenizedDatasetNode.NODE_NAME == "LMTokenizedDataset"
    assert LMTokenizedDatasetNode.CATEGORY == "LLM"
    assert LMTokenizedDatasetNode.cacheable is False


ROW_16 = "abcdefghijklmnop"  # 16 chars -> 16 fake tokens


def test_pack_math_and_shift_by_one():
    rows = [ROW_16, ROW_16.upper()]  # 16 tokens each + EOS => 17-token blocks
    result, _ = run(rows, {"seq_len": 16})  # block=17 -> exactly 2 blocks
    assert result["total_tokens"] == 34
    assert result["num_blocks"] == 2
    dataset = result["dataset"]
    inputs, labels = dataset[0]
    assert inputs.dtype == torch.int64
    assert inputs.shape == (16,)
    # labels are inputs shifted by one within the same block
    fake = FakeTokenizer()
    stream = fake.encode(rows[0]) + [99] + fake.encode(rows[1]) + [99]
    assert inputs.tolist() == stream[0:16]
    assert labels.tolist() == stream[1:17]
    inputs2, labels2 = dataset[1]
    assert inputs2.tolist() == stream[17:33]
    assert labels2.tolist() == stream[18:34]


def test_append_eos_false_drops_document_separators():
    rows = [ROW_16 + "q", ROW_16 + "r"]  # 17 tokens each, no EOS -> 2 blocks
    result, _ = run(rows, {"seq_len": 16, "append_eos": False})
    assert result["total_tokens"] == 34
    assert result["num_blocks"] == 2
    assert 99 not in result["dataset"].blocks.flatten().tolist()


def test_remainder_tokens_are_dropped():
    rows = [ROW_16 + "qrst"]  # 20 tokens + EOS = 21; block=17 -> 1 block, 4 dropped
    result, _ = run(rows, {"seq_len": 16})
    assert result["num_blocks"] == 1
    assert result["total_tokens"] == 21


def test_max_tokens_caps_the_stream():
    rows = ["abcdefghij" * 10]  # 100 tokens + EOS
    result, _ = run(rows, {"seq_len": 16, "max_tokens": 40})
    assert result["total_tokens"] == 40
    assert result["num_blocks"] == 2  # 40 // 17


def test_too_small_corpus_names_seq_len():
    with pytest.raises(RuntimeError, match="seq_len"):
        run(["ab"], {"seq_len": 16})


def test_missing_tokenizer_contract_is_refused():
    with pytest.raises(ValueError, match="LMTokenizer"):
        LMTokenizedDatasetNode().execute(
            {"dataset": LocalTextListDataset(["abc"]), "tokenizer": object()}, {},
        )


def test_dataloader_collates_batches():
    rows = ["abcdefgh" * 4] * 8
    result, _ = run(rows, {"seq_len": 16})
    loader = DataLoader(result["dataset"], batch_size=4, shuffle=False)
    inputs, labels = next(iter(loader))
    assert inputs.shape == (4, 16)
    assert labels.shape == (4, 16)
    assert inputs.dtype == torch.int64


def test_cache_roundtrip_skips_retokenization(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    rows = ["abcdefgh" * 8] * 6
    tokenizer = FakeTokenizer()
    first = LMTokenizedDatasetNode().execute(
        {"dataset": LocalTextListDataset(rows), "tokenizer": tokenizer},
        {"seq_len": 16, "cache": True},
    )
    calls_after_first = tokenizer.batch_calls
    assert calls_after_first > 0
    second = LMTokenizedDatasetNode().execute(
        {"dataset": LocalTextListDataset(rows), "tokenizer": tokenizer},
        {"seq_len": 16, "cache": True},
    )
    assert tokenizer.batch_calls == calls_after_first  # cache hit, no re-encode
    assert torch.equal(first["dataset"].blocks, second["dataset"].blocks)
    assert second["total_tokens"] == first["total_tokens"]
    cache_files = list((tmp_path / "lm_token_cache").glob("lmpack-*.pt"))
    assert len(cache_files) == 1


def test_cache_key_changes_with_seq_len(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    rows = ["abcdefgh" * 8] * 6
    LMTokenizedDatasetNode().execute(
        {"dataset": LocalTextListDataset(rows), "tokenizer": FakeTokenizer()},
        {"seq_len": 16, "cache": True},
    )
    LMTokenizedDatasetNode().execute(
        {"dataset": LocalTextListDataset(rows), "tokenizer": FakeTokenizer()},
        {"seq_len": 32, "cache": True},
    )
    cache_files = list((tmp_path / "lm_token_cache").glob("lmpack-*.pt"))
    assert len(cache_files) == 2
