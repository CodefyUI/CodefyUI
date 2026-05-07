"""Tests for PositionalEncodingNode."""

from __future__ import annotations

import math

import pytest
import torch

from app.nodes.llm.positional_encoding_node import PositionalEncodingNode


def _run(tensor, **params):
    p = {"mode": "sinusoidal", "max_len": 512, "seed": 42}
    p.update(params)
    return PositionalEncodingNode().execute({"tensor": tensor}, p)


def test_node_metadata():
    assert PositionalEncodingNode.NODE_NAME == "PositionalEncoding"
    assert PositionalEncodingNode.CATEGORY == "LLM"
    out_names = [p.name for p in PositionalEncodingNode.define_outputs()]
    assert out_names == ["tensor", "pe"]


def test_sinusoidal_pe_shape_2d_input():
    """[seq, D] in → [seq, D] tensor + [seq, D] pe."""
    x = torch.zeros(5, 8)
    res = _run(x, mode="sinusoidal")
    assert res["tensor"].shape == (5, 8)
    assert res["pe"].shape == (5, 8)


def test_sinusoidal_pe_shape_3d_input():
    """[seq, batch, D] preserved."""
    x = torch.zeros(5, 2, 8)
    res = _run(x, mode="sinusoidal")
    assert res["tensor"].shape == (5, 2, 8)
    assert res["pe"].shape == (5, 8)


def test_sinusoidal_pe_first_position_first_dim_is_zero():
    """sin(0) = 0 — first position, even dim 0 must be zero."""
    res = _run(torch.zeros(10, 16), mode="sinusoidal")
    assert pytest.approx(res["pe"][0, 0].item(), abs=1e-7) == 0.0


def test_sinusoidal_pe_first_position_odd_dim_is_one():
    """cos(0) = 1 — first position, odd dim 1 must be one."""
    res = _run(torch.zeros(10, 16), mode="sinusoidal")
    assert pytest.approx(res["pe"][0, 1].item(), abs=1e-7) == 1.0


def test_sinusoidal_pe_matches_vaswani_formula():
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)."""
    d = 8
    seq = 4
    res = _run(torch.zeros(seq, d), mode="sinusoidal")
    pe = res["pe"]
    for pos in range(seq):
        for i in range(d // 2):
            denom = 10000 ** (2 * i / d)
            expected_sin = math.sin(pos / denom)
            expected_cos = math.cos(pos / denom)
            assert pytest.approx(pe[pos, 2 * i].item(), abs=1e-5) == expected_sin
            assert pytest.approx(pe[pos, 2 * i + 1].item(), abs=1e-5) == expected_cos


def test_sinusoidal_pe_added_to_input():
    """Output tensor should be input + PE."""
    x = torch.full((4, 8), 3.0)
    res = _run(x, mode="sinusoidal")
    expected = x + res["pe"]
    assert torch.allclose(res["tensor"], expected, atol=1e-6)


def test_sinusoidal_pe_added_to_3d_input_broadcasts_over_batch():
    x = torch.zeros(4, 3, 8)
    res = _run(x, mode="sinusoidal")
    # PE [seq, D] should broadcast across batch dim, every batch sees same PE.
    for b in range(3):
        assert torch.allclose(res["tensor"][:, b, :], res["pe"], atol=1e-6)


def test_learnable_pe_is_deterministic_given_seed():
    x = torch.zeros(4, 8)
    a = _run(x, mode="learnable", seed=123)
    b = _run(x, mode="learnable", seed=123)
    assert torch.allclose(a["pe"], b["pe"])


def test_learnable_pe_changes_with_seed():
    x = torch.zeros(4, 8)
    a = _run(x, mode="learnable", seed=1)
    b = _run(x, mode="learnable", seed=2)
    assert not torch.allclose(a["pe"], b["pe"])


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown"):
        _run(torch.zeros(4, 8), mode="not-a-mode")


def test_seq_len_exceeds_max_len_raises():
    with pytest.raises(ValueError, match="max_len"):
        _run(torch.zeros(20, 8), mode="sinusoidal", max_len=10)
