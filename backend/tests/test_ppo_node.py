"""Tests for PPONode (actor-critic network)."""

from __future__ import annotations

import torch

from app.nodes.rl.ppo_node import PPONode


def test_node_metadata():
    assert PPONode.NODE_NAME == "PPO"
    assert PPONode.CATEGORY == "RL"


def test_creates_actor_critic_model():
    res = PPONode().execute({}, {"state_dim": 4, "action_dim": 2, "hidden_dim": 16})
    model = res["model"]
    state = torch.zeros(1, 4)
    probs, value = model(state)
    assert probs.shape == (1, 2)
    assert value.shape == (1, 1)


def test_action_probs_sum_to_one():
    state = torch.randn(2, 4)
    res = PPONode().execute({"state": state}, {"state_dim": 4, "action_dim": 3, "hidden_dim": 16})
    # Forward through model to get probs
    probs, _ = res["model"](state)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_no_state_emits_zero_probs():
    res = PPONode().execute({}, {"state_dim": 4, "action_dim": 5, "hidden_dim": 16})
    assert torch.equal(res["action"], torch.zeros(5))


def test_with_state_emits_action_probs():
    state = torch.randn(1, 4)
    res = PPONode().execute({"state": state}, {"state_dim": 4, "action_dim": 2, "hidden_dim": 8})
    assert res["action"].shape == (1, 2)
    # Probs sum to 1
    assert torch.isclose(res["action"].sum(), torch.tensor(1.0))
