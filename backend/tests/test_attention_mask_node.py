"""Tests for AttentionMaskNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.llm.attention_mask_node import AttentionMaskNode


def _run(*, tensor=None, tokens=None, **params):
    p = {"mode": "causal", "pad_token": ""}
    p.update(params)
    inputs: dict = {}
    if tensor is not None:
        inputs["tensor"] = tensor
    if tokens is not None:
        inputs["tokens"] = tokens
    return AttentionMaskNode().execute(inputs, p)


def test_node_metadata():
    assert AttentionMaskNode.NODE_NAME == "AttentionMask"
    assert AttentionMaskNode.CATEGORY == "LLM"
    out_names = [p.name for p in AttentionMaskNode.define_outputs()]
    assert out_names == ["mask"]


def test_causal_mask_shape_from_tensor():
    res = _run(tensor=torch.zeros(4, 8), mode="causal")
    assert res["mask"].shape == (4, 4)
    assert res["mask"].dtype == torch.bool


def test_causal_mask_is_strictly_upper_triangular():
    res = _run(tensor=torch.zeros(5, 8), mode="causal")
    m = res["mask"]
    expected = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.equal(m, expected)


def test_causal_mask_diagonal_is_visible():
    """Each token can see itself (diagonal must be False = unblocked)."""
    res = _run(tensor=torch.zeros(3, 8), mode="causal")
    diag = torch.diagonal(res["mask"])
    assert torch.all(~diag)


def test_seq_len_from_tokens_list():
    res = _run(tokens=["a", "b", "c"], mode="causal")
    assert res["mask"].shape == (3, 3)


def test_seq_len_from_3d_tensor():
    """[seq, batch, embed] shape — should derive seq from dim 0."""
    res = _run(tensor=torch.zeros(6, 1, 8), mode="causal")
    assert res["mask"].shape == (6, 6)


def test_padding_mask_blocks_pad_tokens():
    res = _run(tokens=["the", "cat", "<pad>", "<pad>"], mode="padding", pad_token="<pad>")
    m = res["mask"]
    # Columns 2 and 3 (pad positions) blocked everywhere; cols 0, 1 unblocked.
    assert torch.all(m[:, 2])
    assert torch.all(m[:, 3])
    assert torch.all(~m[:, 0])
    assert torch.all(~m[:, 1])


def test_padding_mask_with_no_pad_tokens_returns_all_false():
    res = _run(tokens=["a", "b", "c"], mode="padding", pad_token="<pad>")
    assert torch.all(~res["mask"])


def test_padding_mode_requires_tokens():
    with pytest.raises(ValueError, match="padding"):
        _run(tensor=torch.zeros(3, 8), mode="padding", pad_token="<pad>")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown"):
        _run(tensor=torch.zeros(3, 8), mode="not-a-mode")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        AttentionMaskNode().execute({}, {"mode": "causal", "pad_token": ""})
