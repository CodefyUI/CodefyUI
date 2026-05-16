"""Tests for EduPolicyGradientNode (chapter pack C5)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.c5.nodes.edu_policy_gradient_node import EduPolicyGradientNode


def _run(logits, actions, rewards, **params):
    base = {"baseline": "mean", "temperature": 1.0}
    base.update(params)
    return EduPolicyGradientNode().execute(
        {"logits": logits, "actions": actions, "rewards": rewards}, base
    )


def test_node_metadata():
    assert EduPolicyGradientNode.NODE_NAME == "EduPolicyGradient"
    assert EduPolicyGradientNode.CATEGORY == "RL"
    out_names = [p.name for p in EduPolicyGradientNode.define_outputs()]
    assert out_names == ["log_probs", "advantages", "loss", "probs"]


def test_probs_sum_to_one():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    res = _run(logits, torch.tensor([0, 1]), torch.tensor([1.0, 1.0]))
    summed = res["probs"].sum(dim=-1)
    assert torch.allclose(summed, torch.ones(2), atol=1e-6)


def test_log_probs_match_gathered_log_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    actions = torch.tensor([2])
    res = _run(logits, actions, torch.tensor([0.0]))
    expected = F.log_softmax(logits, dim=-1)[0, 2]
    assert res["log_probs"].item() == pytest.approx(expected.item(), abs=1e-6)


def test_baseline_mean_subtracts_average_reward():
    logits = torch.zeros(3, 2)
    actions = torch.tensor([0, 1, 0])
    rewards = torch.tensor([2.0, 4.0, 6.0])
    res = _run(logits, actions, rewards, baseline="mean")
    # mean reward = 4 → advantages = [-2, 0, 2]
    assert res["advantages"].tolist() == pytest.approx([-2.0, 0.0, 2.0])


def test_baseline_none_keeps_raw_rewards():
    logits = torch.zeros(3, 2)
    actions = torch.tensor([0, 1, 0])
    rewards = torch.tensor([2.0, 4.0, 6.0])
    res = _run(logits, actions, rewards, baseline="none")
    assert res["advantages"].tolist() == pytest.approx([2.0, 4.0, 6.0])


def test_temperature_makes_distribution_more_uniform():
    logits = torch.tensor([[10.0, 0.0]])
    cold = _run(logits, torch.tensor([0]), torch.tensor([1.0]), temperature=0.5)
    hot = _run(logits, torch.tensor([0]), torch.tensor([1.0]), temperature=5.0)
    # Lower T → sharper distribution → action 0 prob closer to 1.
    assert cold["probs"][0, 0].item() > hot["probs"][0, 0].item()


def test_loss_is_scalar_and_has_gradient():
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    actions = torch.tensor([2])
    rewards = torch.tensor([1.0])
    res = _run(logits, actions, rewards)
    loss = res["loss"]
    assert loss.ndim == 0  # scalar
    # We can backprop through the loss to logits.
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_rejects_out_of_range_action():
    logits = torch.zeros(2, 3)
    actions = torch.tensor([0, 5])  # 5 ≥ num_actions=3
    rewards = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="out of range"):
        _run(logits, actions, rewards)


def test_rejects_temperature_zero():
    with pytest.raises(ValueError, match="temperature"):
        _run(torch.zeros(1, 2), torch.tensor([0]), torch.tensor([1.0]), temperature=0.0)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduPolicyGradientNode().execute({"logits": torch.zeros(1, 2)}, {})
