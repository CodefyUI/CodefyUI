"""Tests for RNNCellNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.rnn.rnn_cell_node import RNNCellNode


def _run(tensor, *, hidden=None, **params):
    p = {"input_size": 8, "hidden_size": 16, "nonlinearity": "tanh", "seed": 42}
    p.update(params)
    inputs: dict = {"tensor": tensor}
    if hidden is not None:
        inputs["hidden"] = hidden
    return RNNCellNode().execute(inputs, p)


def test_node_metadata():
    assert RNNCellNode.NODE_NAME == "RNNCell"
    assert RNNCellNode.CATEGORY == "RNN"
    out_names = [p.name for p in RNNCellNode.define_outputs()]
    assert out_names == ["hidden"]


def test_output_shape_matches_hidden_size():
    """Input [B, input_size] → output [B, hidden_size]."""
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, input_size=8, hidden_size=16)
    assert res["hidden"].shape == (4, 16)


def test_zero_hidden_init_default():
    """Without hidden input, defaults to zeros — output should still be valid."""
    x = torch.randn(2, 8)
    res = _run(x)
    assert torch.isfinite(res["hidden"]).all()


def test_explicit_hidden_input_used():
    """Same x with two different hidden states → different outputs."""
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(0))
    h1 = torch.zeros(2, 16)
    h2 = torch.randn(2, 16, generator=torch.Generator().manual_seed(1))
    a = _run(x, hidden=h1)
    b = _run(x, hidden=h2)
    assert not torch.allclose(a["hidden"], b["hidden"])


def test_tanh_output_in_range():
    """tanh nonlinearity → output values in [-1, 1]."""
    x = torch.randn(8, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, nonlinearity="tanh")
    assert res["hidden"].min().item() >= -1.0
    assert res["hidden"].max().item() <= 1.0


def test_relu_output_non_negative():
    x = torch.randn(8, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, nonlinearity="relu")
    assert res["hidden"].min().item() >= 0.0


def test_deterministic_given_seed():
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=42)
    b = _run(x, seed=42)
    assert torch.allclose(a["hidden"], b["hidden"])


def test_different_seeds_give_different_outputs():
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=1)
    b = _run(x, seed=2)
    assert not torch.allclose(a["hidden"], b["hidden"])


def test_input_size_mismatch_raises():
    with pytest.raises(ValueError, match="input_size"):
        _run(torch.zeros(2, 4), input_size=8)


def test_unknown_nonlinearity_raises():
    with pytest.raises(ValueError, match="nonlinearity"):
        _run(torch.zeros(2, 8), nonlinearity="not-a-function")


def test_can_chain_unrolled():
    """Manually unroll 3 steps — verify that h_3 depends on x_1, x_2, x_3."""
    cell = RNNCellNode()
    p = {"input_size": 4, "hidden_size": 4, "nonlinearity": "tanh", "seed": 42}
    x1 = torch.randn(1, 4, generator=torch.Generator().manual_seed(0))
    x2 = torch.randn(1, 4, generator=torch.Generator().manual_seed(1))
    x3 = torch.randn(1, 4, generator=torch.Generator().manual_seed(2))
    h0 = torch.zeros(1, 4)
    h1 = cell.execute({"tensor": x1, "hidden": h0}, p)["hidden"]
    h2 = cell.execute({"tensor": x2, "hidden": h1}, p)["hidden"]
    h3 = cell.execute({"tensor": x3, "hidden": h2}, p)["hidden"]
    # Different x1 should propagate to different h3.
    h1_alt = cell.execute({"tensor": x1 + 1.0, "hidden": h0}, p)["hidden"]
    h2_alt = cell.execute({"tensor": x2, "hidden": h1_alt}, p)["hidden"]
    h3_alt = cell.execute({"tensor": x3, "hidden": h2_alt}, p)["hidden"]
    assert not torch.allclose(h3, h3_alt)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        RNNCellNode().execute(
            {},
            {"input_size": 8, "hidden_size": 16, "nonlinearity": "tanh", "seed": 42},
        )
