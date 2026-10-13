"""Tests for DQNNode."""

from __future__ import annotations

import pytest
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


# ── 2026-10-13 functional-correctness audit ─────────────────────────────



def test_string_sizes_are_coerced():
    res = DQNNode().execute({}, {"state_dim": "4", "action_dim": "2", "hidden_dim": "8"})
    assert res["model"](torch.zeros(1, 4)).shape == (1, 2)


@pytest.mark.parametrize("name", ["state_dim", "action_dim", "hidden_dim"])
@pytest.mark.parametrize("value", [0, -3, "four", 2.5])
def test_bad_sizes_are_node_named_errors(name, value):
    params = {"state_dim": 4, "action_dim": 2, "hidden_dim": 8, name: value}
    with pytest.raises(ValueError, match=f"DQN: {name} must be"):
        DQNNode().execute({}, params)


def test_sizes_declare_a_minimum_of_one():
    assert all(p.min_value == 1 for p in DQNNode.define_params())
