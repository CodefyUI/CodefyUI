"""Tests for the LLM TokenizerNode."""

from __future__ import annotations

import pytest

from app.nodes.llm.tokenizer_node import TokenizerNode


def _run(family: str, text: str, show_special: bool = False) -> dict:
    return TokenizerNode().execute(
        {},
        {"family": family, "text": text, "show_special_tokens": show_special},
    )


def test_node_metadata():
    assert TokenizerNode.NODE_NAME == "Tokenizer"
    assert TokenizerNode.CATEGORY == "LLM"
    inputs = TokenizerNode.define_inputs()
    outputs = TokenizerNode.define_outputs()
    assert [p.name for p in inputs] == ["text"]
    assert [p.name for p in outputs] == ["tokens", "token_ids", "offsets"]


def test_cl100k_basic_ascii():
    res = _run("cl100k_base", "Hello, world!")
    assert res["tokens"] == ["Hello", ",", " world", "!"]
    assert res["token_ids"] == [9906, 11, 1917, 0]
    assert res["offsets"][0] == [0, 5]


def test_offsets_cover_full_text():
    text = "The quick brown fox"
    res = _run("cl100k_base", text)
    # Concatenating offset slices should reconstruct the input exactly for ASCII.
    reconstructed = "".join(text[s:e] for s, e in res["offsets"])
    assert reconstructed == text


def test_empty_text_returns_empty_lists():
    res = _run("cl100k_base", "")
    assert res["tokens"] == []
    assert res["token_ids"] == []
    assert res["offsets"] == []


def test_input_port_overrides_param():
    res = TokenizerNode().execute(
        {"text": "from input"},
        {"family": "cl100k_base", "text": "from param", "show_special_tokens": False},
    )
    reconstructed = "".join("from input"[s:e] for s, e in res["offsets"])
    assert reconstructed == "from input"


def test_batch_input_takes_first_item():
    # Single-item demo path for batch inputs; explicit batch iteration is RAG's job.
    res = TokenizerNode().execute(
        {"text": ["hello", "ignored"]},
        {"family": "cl100k_base", "text": "fallback", "show_special_tokens": False},
    )
    reconstructed = "".join("hello"[s:e] for s, e in res["offsets"])
    assert reconstructed == "hello"


def test_unicode_and_emoji_do_not_crash():
    res = _run("cl100k_base", "café 世界 🌏")
    assert len(res["tokens"]) == len(res["token_ids"]) == len(res["offsets"])
    # Sanity: at least one token, IDs are ints.
    assert all(isinstance(i, int) for i in res["token_ids"])


def test_tiktoken_special_token_literal_does_not_raise():
    # cl100k_base has e.g. <|endoftext|>; our config allows it as plain text.
    res = _run("cl100k_base", "before <|endoftext|> after")
    assert "before" in "".join(res["tokens"])
    assert "after" in "".join(res["tokens"])


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown tokenizer family"):
        _run("not-a-real-family", "x")


def test_o200k_differs_from_cl100k_for_same_input():
    cl = _run("cl100k_base", "the quick brown fox")
    o2 = _run("o200k_base", "the quick brown fox")
    # Different families should usually produce different ID sequences for the
    # same input. (Both share many basic merges so this is not guaranteed for
    # every string, but for "the quick brown fox" it holds across model families.)
    assert cl["token_ids"] != o2["token_ids"]


def test_step_trace_emitted_when_verbose():
    from dataclasses import dataclass

    @dataclass
    class FakeContext:
        verbose: bool = True

    res = TokenizerNode().execute(
        {},
        {"family": "cl100k_base", "text": "hi", "show_special_tokens": False},
        context=FakeContext(),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert step_names == ["input_text", "tokenize"]
