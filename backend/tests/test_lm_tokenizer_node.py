"""Tests for LMTokenizerNode (offline: gpt2 BPE ranks ship in the tiktoken cache)."""

from __future__ import annotations

import pickle

import pytest

from app.nodes.llm.lm_tokenizer_node import LMTokenizerHandle, LMTokenizerNode


def test_node_metadata():
    assert LMTokenizerNode.NODE_NAME == "LMTokenizer"
    assert LMTokenizerNode.CATEGORY == "LLM"
    assert LMTokenizerNode.define_inputs() == []
    assert LMTokenizerNode.cacheable is False


def test_unknown_encoding_is_refused():
    with pytest.raises(ValueError, match="encoding"):
        LMTokenizerNode().execute({}, {"encoding": "made-up"})


def test_gpt2_contract_roundtrip():
    result = LMTokenizerNode().execute({}, {"encoding": "gpt2"})
    handle = result["tokenizer"]
    assert result["vocab_size"] == 50257
    assert handle.vocab_size == 50257
    assert handle.eos_id == 50256
    ids = handle.encode("Once upon a time")
    assert ids and all(isinstance(i, int) for i in ids)
    assert handle.decode(ids) == "Once upon a time"


def test_encode_batch_matches_single_encode():
    handle = LMTokenizerNode().execute({}, {})["tokenizer"]
    texts = ["Hello world", "The quick brown fox"]
    assert handle.encode_batch(texts) == [handle.encode(t) for t in texts]


def test_special_token_literals_are_treated_as_plain_text():
    handle = LMTokenizerNode().execute({}, {})["tokenizer"]
    ids = handle.encode("before <|endoftext|> after")
    # encode_ordinary must not inject the real EOS id for the literal.
    assert handle.eos_id not in ids


def test_handle_pickles_by_name_and_reloads():
    handle = LMTokenizerHandle("gpt2")
    ids_before = handle.encode("pickle me")
    clone = pickle.loads(pickle.dumps(handle))
    assert clone.encoding_name == "gpt2"
    assert clone.encode("pickle me") == ids_before
