"""Tests for LMTokenizerNode (#290).

Network-free: ``gpt2``'s BPE files are already in tiktoken's on-disk cache in
this environment (``test_tokenizer_node.py`` exercises the same loader), and
the one test that needs a load FAILURE monkeypatches the loader rather than
going offline for real.
"""

from __future__ import annotations

import pytest

from app.nodes.llm import lm_tokenizer_node
from app.nodes.llm.lm_tokenizer_node import LMTokenizerNode


def _run(encoding: str = "gpt2") -> dict:
    return LMTokenizerNode().execute({}, {"encoding": encoding})


def test_node_metadata():
    assert LMTokenizerNode.NODE_NAME == "LMTokenizer"
    assert LMTokenizerNode.CATEGORY == "LLM"
    assert LMTokenizerNode.define_inputs() == []
    assert [p.name for p in LMTokenizerNode.define_outputs()] == [
        "tokenizer", "vocab_size"]
    params = {p.name: p for p in LMTokenizerNode.define_params()}
    assert params["encoding"].default == "gpt2"
    assert params["encoding"].options == [
        "gpt2", "p50k_base", "cl100k_base", "o200k_base"]


def test_gpt2_vocab_size_and_eos_id():
    res = _run("gpt2")
    tokenizer = res["tokenizer"]
    # The two numbers every downstream node reads off this port: vocab_size
    # has to match CausalLMModel's vocab_size param, and eos_id is what
    # LMTokenizedDataset joins documents with.
    assert res["vocab_size"] == 50257
    assert tokenizer.vocab_size == 50257
    assert tokenizer.eos_id == 50256


def test_eos_id_is_derived_not_hardcoded():
    # cl100k_base shares the <|endoftext|> literal with gpt2 but numbers it
    # differently -- a hardcoded 50256 would pass the gpt2 test and be wrong
    # here.
    assert _run("cl100k_base")["tokenizer"].eos_id == 100257


def test_encode_decode_round_trip():
    tokenizer = _run("gpt2")["tokenizer"]
    text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(text)
    assert ids and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == text


def test_encode_returns_a_plain_list():
    # tiktoken hands back a list already, but LMTokenizedDataset extends its
    # stream with the result, so anything list-like would do -- the contract
    # promises list[int] and callers are allowed to rely on it.
    ids = _run("gpt2")["tokenizer"].encode("hello")
    assert type(ids) is list


def test_special_token_literal_does_not_raise():
    # A corpus row containing the literal "<|endoftext|>" is ordinary text as
    # far as this tokenizer is concerned; tiktoken's default would raise.
    tokenizer = _run("gpt2")["tokenizer"]
    ids = tokenizer.encode("before <|endoftext|> after")
    assert tokenizer.decode(ids) == "before <|endoftext|> after"


def test_decode_accepts_any_int_iterable():
    tokenizer = _run("gpt2")["tokenizer"]
    ids = tokenizer.encode("round trip")
    assert tokenizer.decode(tuple(ids)) == "round trip"


def test_unknown_encoding_raises():
    with pytest.raises(ValueError, match="unknown encoding"):
        _run("not-a-real-encoding")


def test_an_hf_tokenizer_family_is_rejected():
    # bert-base-uncased IS a family the shared _load_encoder knows, so a node
    # that passed its param straight through would return a `tokenizers`
    # Tokenizer -- which has no eos_id, no vocab_size and a different encode()
    # signature. The param is validated against this node's own options first.
    with pytest.raises(ValueError, match="unknown encoding"):
        _run("bert-base-uncased")


def test_reuses_the_shared_encoder_cache():
    from app.nodes.llm.tokenizer_node import _ENCODER_CACHE, _load_encoder

    _run("gpt2")
    assert "gpt2" in _ENCODER_CACHE
    # Same object, not merely an equal one: a second cache would double the
    # BPE tables in memory and download them twice on a cold machine.
    tokenizer = _run("gpt2")["tokenizer"]
    assert tokenizer.encode("x") == list(
        _load_encoder("gpt2")[1].encode("x", disallowed_special=()))


def test_a_load_failure_is_reported_as_an_offline_hint(monkeypatch):
    def explode(family: str):
        raise ConnectionError("getaddrinfo failed")

    monkeypatch.setattr(lm_tokenizer_node, "_load_encoder", explode)
    with pytest.raises(RuntimeError, match="offline"):
        _run("gpt2")


def test_empty_text_encodes_to_nothing():
    tokenizer = _run("gpt2")["tokenizer"]
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
