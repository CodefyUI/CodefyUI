"""Tests for EduPPOClipNode (lesson C5-2 / I5: the PPO clipped surrogate)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.rl.nodes.edu_ppo_clip_node import EduPPOClipNode


def _run(logits_new, logits_old, actions, advantages, *, context=None, **params):
    base = {"clip_epsilon": 0.2, "temperature": 1.0}
    base.update(params)
    return EduPPOClipNode().execute(
        {
            "logits_new": logits_new,
            "logits_old": logits_old,
            "actions": actions,
            "advantages": advantages,
        },
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
    assert EduPPOClipNode.NODE_NAME == "Edu-PPOClip"
    assert EduPPOClipNode.CATEGORY == "RL"
    out_names = [p.name for p in EduPPOClipNode.define_outputs()]
    assert out_names == ["loss", "ratio", "clipped_ratio", "objective"]


def test_identical_policies_give_unit_ratio_and_negative_mean_advantage():
    # When the new policy equals the old (behaviour) policy, every ratio is
    # exactly 1, nothing clips, and the surrogate reduces to the plain
    # advantage so loss = -mean(advantages).
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 2.0], [3.0, 0.0, 0.0]])
    actions = torch.tensor([0, 2, 1])
    advantages = torch.tensor([1.0, -2.0, 0.5])
    res = _run(logits, logits.clone(), actions, advantages)

    assert torch.allclose(res["ratio"], torch.ones(3), atol=1e-6)
    # No movement → clipped ratio is also all ones.
    assert torch.allclose(res["clipped_ratio"], torch.ones(3), atol=1e-6)
    assert res["loss"].item() == pytest.approx((-advantages.mean()).item(), abs=1e-6)


def test_clip_binds_for_large_positive_advantage_ratio():
    # action 0, two actions. old logits [0,0] → p_old(0) = 0.5.
    # new logits [1,0] → p_new(0) = sigmoid(1). ratio = sigmoid(1)/0.5 ≈ 1.462,
    # which exceeds 1 + eps = 1.2, so the clip binds.
    logits_new = torch.tensor([[1.0, 0.0]])
    logits_old = torch.tensor([[0.0, 0.0]])
    actions = torch.tensor([0])
    advantages = torch.tensor([2.0])
    eps = 0.2
    res = _run(logits_new, logits_old, actions, advantages, clip_epsilon=eps)

    closed_form_ratio = (1.0 / (1.0 + math.exp(-1.0))) / 0.5
    assert res["ratio"].item() == pytest.approx(closed_form_ratio, abs=1e-6)
    assert res["ratio"].item() > 1.0 + eps  # the clip actually has something to do

    # Clip binds at the upper edge of the trust region.
    assert res["clipped_ratio"].item() == pytest.approx(1.0 + eps, abs=1e-6)

    # Positive advantage + ratio > 1+eps → min() selects the clipped (smaller)
    # branch, so the objective uses clip·A, not ratio·A.
    surr1 = closed_form_ratio * 2.0  # unclipped, larger
    surr2 = (1.0 + eps) * 2.0  # clipped, smaller
    assert surr2 < surr1
    assert res["objective"].item() == pytest.approx(surr2, abs=1e-6)
    # loss = -mean(objective) = -(1.2 * 2.0) = -2.4
    assert res["loss"].item() == pytest.approx(-surr2, abs=1e-6)


def test_clipped_ratio_stays_within_trust_region():
    # A spread of new/old logits so several ratios stray outside [1-eps, 1+eps];
    # clipped_ratio must always be confined to the trust region.
    logits_new = torch.tensor([[4.0, 0.0], [-3.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    logits_old = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    actions = torch.tensor([0, 0, 0, 0])
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    eps = 0.2
    res = _run(logits_new, logits_old, actions, advantages, clip_epsilon=eps)

    clipped = res["clipped_ratio"]
    assert bool((clipped >= 1.0 - eps - 1e-6).all())
    assert bool((clipped <= 1.0 + eps + 1e-6).all())
    # At least one raw ratio left the region, proving the clamp did real work.
    assert bool(((res["ratio"] > 1.0 + eps) | (res["ratio"] < 1.0 - eps)).any())


def test_ratio_matches_gathered_log_softmax_difference():
    logits_new = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
    logits_old = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
    actions = torch.tensor([2, 0])
    advantages = torch.tensor([1.0, -1.0])
    res = _run(logits_new, logits_old, actions, advantages)

    logp_new = F.log_softmax(logits_new, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    logp_old = F.log_softmax(logits_old, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    expected_ratio = torch.exp(logp_new - logp_old)
    assert torch.allclose(res["ratio"], expected_ratio, atol=1e-6)


def test_temperature_scales_both_policies():
    # With non-unit temperature the ratio uses log_softmax(logits / T) for both
    # the new and old policies.
    logits_new = torch.tensor([[2.0, 0.0]])
    logits_old = torch.tensor([[0.0, 1.0]])
    actions = torch.tensor([0])
    advantages = torch.tensor([1.0])
    T = 2.0
    res = _run(logits_new, logits_old, actions, advantages, temperature=T)

    logp_new = F.log_softmax(logits_new / T, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    logp_old = F.log_softmax(logits_old / T, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    expected_ratio = torch.exp(logp_new - logp_old)
    assert res["ratio"].item() == pytest.approx(expected_ratio.item(), abs=1e-6)


def test_loss_is_scalar_and_has_gradient_through_new_policy():
    logits_new = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], requires_grad=True)
    logits_old = torch.tensor([[0.5, 0.5, 1.0], [1.0, 0.0, 0.0]])
    actions = torch.tensor([2, 1])
    advantages = torch.tensor([1.0, -0.5])
    res = _run(logits_new, logits_old, actions, advantages)
    loss = res["loss"]
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss).all()
    loss.backward()
    assert logits_new.grad is not None
    assert logits_new.grad.shape == logits_new.shape


def test_deterministic_with_explicit_tensors():
    logits_new = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
    logits_old = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    actions = torch.tensor([0, 2])
    advantages = torch.tensor([1.5, -0.5])
    a = _run(logits_new, logits_old, actions, advantages)
    b = _run(logits_new, logits_old, actions, advantages)
    assert torch.equal(a["ratio"], b["ratio"])
    assert torch.equal(a["clipped_ratio"], b["clipped_ratio"])
    assert torch.equal(a["objective"], b["objective"])
    assert a["loss"].item() == b["loss"].item()


def test_rejects_out_of_range_action():
    logits = torch.zeros(2, 3)
    actions = torch.tensor([0, 5])  # 5 >= num_actions=3
    advantages = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="out of range"):
        _run(logits, logits.clone(), actions, advantages)


def test_rejects_shape_mismatch_between_policies():
    logits_new = torch.zeros(2, 3)
    logits_old = torch.zeros(2, 4)  # mismatched num_actions
    actions = torch.tensor([0, 1])
    advantages = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        _run(logits_new, logits_old, actions, advantages)


def test_rejects_advantages_shape_mismatch():
    logits = torch.zeros(3, 2)
    actions = torch.tensor([0, 1, 0])
    advantages = torch.tensor([1.0, 2.0])  # only 2, expected 3
    with pytest.raises(ValueError, match="advantages must have shape"):
        _run(logits, logits.clone(), actions, advantages)


def test_rejects_non_2d_logits():
    bad = torch.zeros(3)  # 1-D
    actions = torch.tensor([0, 1, 0])
    advantages = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"\[B, A\]"):
        _run(bad, torch.zeros(3, 2), actions, advantages)


def test_rejects_negative_clip_epsilon():
    logits = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="clip_epsilon"):
        _run(logits, logits.clone(), torch.tensor([0, 1]), torch.tensor([1.0, 2.0]), clip_epsilon=-0.1)


def test_rejects_temperature_zero():
    logits = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="temperature"):
        _run(logits, logits.clone(), torch.tensor([0, 1]), torch.tensor([1.0, 2.0]), temperature=0.0)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduPPOClipNode().execute({"logits_new": torch.zeros(2, 2)}, {})


def test_steps_present_when_verbose():
    res = _run(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([1.0, -1.0]),
        context=_Ctx(),
    )
    assert "__steps__" in res
    names = [s.name for s in res["__steps__"]]
    assert names == ["log_probs", "ratio", "clip", "surrogates", "objective", "loss"]
    # The ratio step exposes the clip_epsilon scalar so the lesson can label it.
    ratio_step = res["__steps__"][1]
    assert ratio_step.scalars["clip_epsilon"] == pytest.approx(0.2)


def test_steps_absent_without_verbose():
    res = _run(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([1.0, -1.0]),
    )
    assert "__steps__" not in res
