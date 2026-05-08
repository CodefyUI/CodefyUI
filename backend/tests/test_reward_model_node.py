"""Tests for RewardModelNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.rl.reward_model_node import RewardModelNode


def _run(hidden_states=None, **params):
    p = {"input_dim": 8, "hidden_dim": 16, "seed": 42}
    p.update(params)
    inputs = {}
    if hidden_states is not None:
        inputs["hidden_states"] = hidden_states
    return RewardModelNode().execute(inputs, p)


def test_node_metadata():
    assert RewardModelNode.NODE_NAME == "RewardModel"
    assert RewardModelNode.CATEGORY == "RL"
    out_names = [p.name for p in RewardModelNode.define_outputs()]
    assert "model" in out_names
    assert "rewards" in out_names


def test_returns_module_when_no_input():
    res = _run()
    assert isinstance(res["model"], nn.Module)


def test_scores_2d_hidden_states():
    """[B, H] → scalar reward per item, shape [B]."""
    h = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    res = _run(h)
    assert res["rewards"].shape == (4,)


def test_uses_last_token_for_3d_input():
    """[B, T, H] → uses the last token to score the sequence, [B]."""
    h = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    res = _run(h)
    assert res["rewards"].shape == (2,)


def test_seed_makes_init_reproducible():
    h = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
    a = _run(h, seed=42)
    b = _run(h, seed=42)
    assert torch.allclose(a["rewards"], b["rewards"])
    c = _run(h, seed=99)
    assert not torch.allclose(a["rewards"], c["rewards"])


def test_dim_mismatch_raises():
    h = torch.randn(2, 16)
    with pytest.raises(RuntimeError):
        _run(h, input_dim=8)
