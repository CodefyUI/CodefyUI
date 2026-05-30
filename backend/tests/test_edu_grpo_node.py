"""Tests for EduGRPONode (lesson C5-4 / I5: group-relative policy optimization)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.rl.nodes.edu_grpo_node import EduGRPONode


def _run(logits, actions, rewards, *, context=None, **params):
    base = {"temperature": 1.0, "std_eps": 1e-6}
    base.update(params)
    return EduGRPONode().execute(
        {"logits": logits, "actions": actions, "rewards": rewards},
        base,
        context=context,
    )


class _Ctx:
    """Minimal stand-in for ExecutionContext with verbose mode on."""

    verbose = True
    weights_persistent = False
    node_state_store = None
    current_node_id = None


def test_node_metadata():
    assert EduGRPONode.NODE_NAME == "Edu-GRPO"
    assert EduGRPONode.CATEGORY == "RL"
    out_names = [p.name for p in EduGRPONode.define_outputs()]
    assert out_names == ["loss", "advantages", "log_probs", "group_mean", "group_std"]


def test_advantages_zero_mean_unit_std_for_spread_rewards():
    # A spread-out reward vector → group-normalized advantages should be a
    # z-score: ~zero mean and (with a tiny std_eps) ~unit population std.
    logits = torch.zeros(4, 3)
    actions = torch.tensor([0, 1, 2, 0])
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0])
    res = _run(logits, actions, rewards, std_eps=1e-12)
    adv = res["advantages"]
    assert adv.mean().item() == pytest.approx(0.0, abs=1e-5)
    # Population std of a true z-score is 1.
    assert adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)


def test_group_mean_and_std_match_rewards():
    logits = torch.zeros(3, 2)
    actions = torch.tensor([0, 1, 0])
    rewards = torch.tensor([2.0, 4.0, 6.0])
    res = _run(logits, actions, rewards)
    assert res["group_mean"].item() == pytest.approx(4.0)
    # Population std of [2,4,6] about mean 4 = sqrt((4+0+4)/3) = sqrt(8/3).
    assert res["group_std"].item() == pytest.approx(math.sqrt(8.0 / 3.0), abs=1e-6)


def test_identical_rewards_give_zero_advantages_and_no_nan():
    # group_std == 0 here; std_eps must prevent a divide-by-zero NaN/inf.
    logits = torch.zeros(3, 2)
    actions = torch.tensor([0, 1, 0])
    rewards = torch.tensor([5.0, 5.0, 5.0])
    res = _run(logits, actions, rewards)
    adv = res["advantages"]
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)
    assert res["group_std"].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(res["loss"]).all()


def test_hand_computed_small_case():
    # Two grouped samples, 2 actions, deterministic by hand.
    # rewards = [0, 2] → group_mean = 1, group_std = 1 (population).
    # advantages = (rewards - 1) / (1 + 1e-6) ≈ [-1, +1].
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    actions = torch.tensor([0, 1])
    rewards = torch.tensor([0.0, 2.0])
    res = _run(logits, actions, rewards, std_eps=1e-6)

    assert res["group_mean"].item() == pytest.approx(1.0)
    assert res["group_std"].item() == pytest.approx(1.0, abs=1e-6)
    assert res["advantages"].tolist() == pytest.approx([-1.0, 1.0], abs=1e-5)

    # Uniform logits → prob 0.5 for the taken action → log_prob = log(0.5).
    expected_logp = math.log(0.5)
    assert res["log_probs"].tolist() == pytest.approx([expected_logp, expected_logp], abs=1e-6)

    # loss = -mean(log_probs * advantages)
    #      = -mean([log(0.5)*-1, log(0.5)*+1]) = -mean([+0.693, -0.693]) = 0.
    assert res["loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_log_probs_match_gathered_log_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
    actions = torch.tensor([2, 0])
    rewards = torch.tensor([1.0, 4.0])
    res = _run(logits, actions, rewards)
    expected = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    assert torch.allclose(res["log_probs"], expected, atol=1e-6)


def test_loss_is_finite_scalar_and_has_gradient():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], requires_grad=True)
    actions = torch.tensor([2, 1])
    rewards = torch.tensor([1.0, 3.0])
    res = _run(logits, actions, rewards)
    loss = res["loss"]
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss).all()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_rejects_group_of_size_one():
    logits = torch.zeros(1, 3)
    actions = torch.tensor([0])
    rewards = torch.tensor([1.0])
    with pytest.raises(ValueError, match="at least two"):
        _run(logits, actions, rewards)


def test_rejects_out_of_range_action():
    logits = torch.zeros(2, 3)
    actions = torch.tensor([0, 5])  # 5 ≥ num_actions=3
    rewards = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="out of range"):
        _run(logits, actions, rewards)


def test_rejects_shape_mismatch():
    logits = torch.zeros(3, 2)
    actions = torch.tensor([0, 1, 0])
    rewards = torch.tensor([1.0, 2.0])  # only 2, expected 3
    with pytest.raises(ValueError, match="rewards must have shape"):
        _run(logits, actions, rewards)


def test_rejects_temperature_zero():
    with pytest.raises(ValueError, match="temperature"):
        _run(torch.zeros(2, 2), torch.tensor([0, 1]), torch.tensor([1.0, 2.0]), temperature=0.0)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduGRPONode().execute({"logits": torch.zeros(2, 2)}, {})


def test_steps_present_when_verbose():
    res = _run(
        torch.zeros(3, 2),
        torch.tensor([0, 1, 0]),
        torch.tensor([1.0, 2.0, 3.0]),
        context=_Ctx(),
    )
    assert "__steps__" in res
    names = [s.name for s in res["__steps__"]]
    assert names == ["softmax", "log_probs", "group_stats", "advantages", "loss"]


def test_steps_absent_without_verbose():
    res = _run(
        torch.zeros(3, 2),
        torch.tensor([0, 1, 0]),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert "__steps__" not in res
