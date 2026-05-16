"""Tests for DQNNode."""

from __future__ import annotations

import torch

from app.nodes.rl.dqn_node import DQNNode


def test_node_metadata():
    assert DQNNode.NODE_NAME == "DQN"
    assert DQNNode.CATEGORY == "RL"


def test_creates_model_with_correct_shape():
    res = DQNNode().execute({}, {"state_dim": 4, "action_dim": 2, "hidden_dim": 32})
    model = res["model"]
    # Sanity: model should accept (B, state_dim) and emit (B, action_dim)
    state = torch.zeros(1, 4)
    out = model(state)
    assert out.shape == (1, 2)


def test_no_state_input_emits_zero_qvalues():
    res = DQNNode().execute({}, {"state_dim": 4, "action_dim": 5, "hidden_dim": 16})
    assert torch.equal(res["action"], torch.zeros(5))


def test_with_state_input_emits_qvalues():
    state = torch.randn(1, 4)
    res = DQNNode().execute(
        {"state": state},
        {"state_dim": 4, "action_dim": 2, "hidden_dim": 8},
    )
    assert res["action"].shape == (1, 2)
    # Q-values from inference should NOT have grad enabled
    assert res["action"].requires_grad is False


def test_hidden_dim_affects_param_count():
    res_small = DQNNode().execute({}, {"state_dim": 4, "action_dim": 2, "hidden_dim": 8})
    res_large = DQNNode().execute({}, {"state_dim": 4, "action_dim": 2, "hidden_dim": 128})
    n_small = sum(p.numel() for p in res_small["model"].parameters())
    n_large = sum(p.numel() for p in res_large["model"].parameters())
    assert n_large > n_small
