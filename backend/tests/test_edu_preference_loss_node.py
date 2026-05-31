"""Tests for EduPreferenceLossNode (chapter pack C5-3 / I5)."""

from __future__ import annotations

import math

import pytest
import torch

from cdui_plugins.rl.nodes.edu_preference_loss_node import EduPreferenceLossNode


def _run(reward_chosen, reward_rejected, **params):
    base = {"beta": 1.0}
    base.update(params)
    return EduPreferenceLossNode().execute(
        {"reward_chosen": reward_chosen, "reward_rejected": reward_rejected}, base
    )


def test_node_metadata():
    assert EduPreferenceLossNode.NODE_NAME == "Edu-PreferenceLoss"
    assert EduPreferenceLossNode.CATEGORY == "RL"
    out_names = [p.name for p in EduPreferenceLossNode.define_outputs()]
    assert out_names == ["loss", "prob_preferred", "margin", "accuracy"]


def test_chosen_much_greater_drives_loss_to_zero():
    # Chosen scores dominate rejected → model is confident and correct.
    reward_chosen = torch.tensor([10.0, 12.0, 8.0])
    reward_rejected = torch.tensor([-10.0, -8.0, -12.0])
    res = _run(reward_chosen, reward_rejected)
    assert res["loss"].item() == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(res["prob_preferred"], torch.ones(3), atol=1e-4)
    assert res["accuracy"].item() == pytest.approx(1.0)


def test_margin_is_difference():
    reward_chosen = torch.tensor([3.0, 1.0, 5.0])
    reward_rejected = torch.tensor([1.0, 4.0, 2.0])
    res = _run(reward_chosen, reward_rejected)
    assert res["margin"].tolist() == pytest.approx([2.0, -3.0, 3.0])


def test_equal_scores_give_half_and_log2_loss():
    reward_chosen = torch.tensor([1.0, 2.0, 3.0])
    reward_rejected = torch.tensor([1.0, 2.0, 3.0])
    res = _run(reward_chosen, reward_rejected)
    # margin = 0 → sigmoid(0) = 0.5 → loss = -log(0.5) = log(2) ≈ 0.6931.
    assert res["loss"].item() == pytest.approx(-math.log(0.5), abs=1e-6)
    assert res["loss"].item() == pytest.approx(0.6931, abs=1e-3)
    assert torch.allclose(res["prob_preferred"], torch.full((3,), 0.5), atol=1e-6)


def test_accuracy_counts_correctly_ordered_pairs():
    # 2 of 4 pairs have chosen > rejected.
    reward_chosen = torch.tensor([5.0, 1.0, 9.0, 0.0])
    reward_rejected = torch.tensor([2.0, 4.0, 3.0, 7.0])
    res = _run(reward_chosen, reward_rejected)
    assert res["accuracy"].item() == pytest.approx(0.5)


def test_prob_preferred_strictly_in_unit_interval():
    # Finite (non-saturating) margins, including a strongly negative one, all
    # map to a probability strictly inside (0, 1).
    reward_chosen = torch.tensor([8.0, -8.0, 0.0])
    reward_rejected = torch.tensor([-8.0, 8.0, 0.0])
    res = _run(reward_chosen, reward_rejected)
    p = res["prob_preferred"]
    assert torch.all(p > 0.0)
    assert torch.all(p < 1.0)


def test_beta_sharpens_preference_probability():
    reward_chosen = torch.tensor([1.0])
    reward_rejected = torch.tensor([0.0])
    soft = _run(reward_chosen, reward_rejected, beta=0.5)
    sharp = _run(reward_chosen, reward_rejected, beta=5.0)
    # Same positive margin → larger β pushes P(chosen > rejected) closer to 1.
    assert sharp["prob_preferred"][0].item() > soft["prob_preferred"][0].item()


def test_loss_is_scalar_and_has_gradient():
    reward_chosen = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    reward_rejected = torch.tensor([0.5, 2.5, 1.0])
    res = _run(reward_chosen, reward_rejected)
    loss = res["loss"]
    assert loss.ndim == 0  # scalar
    loss.backward()
    assert reward_chosen.grad is not None
    assert reward_chosen.grad.shape == reward_chosen.shape


def test_rejects_shape_mismatch():
    reward_chosen = torch.tensor([1.0, 2.0, 3.0])
    reward_rejected = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        _run(reward_chosen, reward_rejected)


def test_rejects_non_1d_input():
    reward_chosen = torch.zeros(2, 3)
    reward_rejected = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="1-D"):
        _run(reward_chosen, reward_rejected)


def test_rejects_negative_beta():
    with pytest.raises(ValueError, match="beta"):
        _run(torch.tensor([1.0]), torch.tensor([0.0]), beta=-1.0)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduPreferenceLossNode().execute({"reward_chosen": torch.tensor([1.0])}, {})


def test_no_steps_without_verbose_context():
    res = _run(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]))
    assert "__steps__" not in res


def test_step_trace_emitted_when_verbose():
    class _Ctx:
        verbose = True
        weights_persistent = False
        node_state_store = None

    res = EduPreferenceLossNode().execute(
        {
            "reward_chosen": torch.tensor([2.0, 1.0]),
            "reward_rejected": torch.tensor([0.0, 1.5]),
        },
        {"beta": 1.0},
        context=_Ctx(),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    # Expect margin → scaled → sigmoid → loss progression.
    assert step_names == ["margin", "scaled", "sigmoid", "loss"]
