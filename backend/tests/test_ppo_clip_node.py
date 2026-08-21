"""Tests for PPOClipObjectiveNode.

The three-row table is C5-2 example 5-2-a, reproduced digit for digit in
I5-2. If these move, the textbook is wrong.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.rl.ppo_clip_node import PPOClipObjectiveNode


def _run(advantages, ratio=None, log_probs_new=None, log_probs_old=None, **params):
    inputs = {"advantages": torch.as_tensor(advantages, dtype=torch.float32)}
    if ratio is not None:
        inputs["ratio"] = torch.as_tensor(ratio, dtype=torch.float32)
    if log_probs_new is not None:
        inputs["log_probs_new"] = torch.as_tensor(log_probs_new, dtype=torch.float32)
    if log_probs_old is not None:
        inputs["log_probs_old"] = torch.as_tensor(log_probs_old, dtype=torch.float32)
    p = {"epsilon": 0.2}
    p.update(params)
    return PPOClipObjectiveNode().execute(inputs, p)


def test_node_metadata():
    assert PPOClipObjectiveNode.NODE_NAME == "PPOClipObjective"
    assert PPOClipObjectiveNode.CATEGORY == "RL"
    out_names = [p.name for p in PPOClipObjectiveNode.define_outputs()]
    for expected in ("objective", "unclipped", "clipped", "ratio", "was_clipped"):
        assert expected in out_names


def test_reproduces_c5_2_worked_example():
    """C5-2 example 5-2-a, all three rows at once."""
    out = _run(advantages=[0.5, 0.8, -0.3], ratio=[1.05, 1.40, 0.55])

    assert out["unclipped"].tolist() == pytest.approx([0.525, 1.12, -0.165], abs=1e-4)
    assert out["clipped"].tolist() == pytest.approx([0.525, 0.96, -0.24], abs=1e-4)
    assert out["objective"].tolist() == pytest.approx([0.525, 0.96, -0.24], abs=1e-4)
    # Row 1 is inside the interval; rows 2 and 3 are truncated.
    assert out["was_clipped"].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert out["clip_fraction"] == pytest.approx(2 / 3)


def test_ratio_inside_the_interval_is_untouched():
    out = _run(advantages=[1.0], ratio=[1.1])
    assert out["objective"].tolist() == pytest.approx(out["unclipped"].tolist())
    assert out["was_clipped"].tolist() == [0.0]


def test_positive_advantage_is_capped_above():
    """Pushing the probability past 1+eps stops paying."""
    at_edge = _run(advantages=[1.0], ratio=[1.2])["objective"]
    way_past = _run(advantages=[1.0], ratio=[5.0])["objective"]
    assert at_edge.tolist() == pytest.approx(way_past.tolist())


def test_negative_advantage_is_capped_below():
    """The other direction: pushing it down past 1-eps stops paying too."""
    at_edge = _run(advantages=[-1.0], ratio=[0.8])["objective"]
    way_past = _run(advantages=[-1.0], ratio=[0.01])["objective"]
    assert at_edge.tolist() == pytest.approx(way_past.tolist())


def test_clip_does_not_clamp_the_ratio_it_reports():
    """Clipping flattens the objective; the ratio itself is reported as given."""
    out = _run(advantages=[1.0], ratio=[5.0])
    assert out["ratio"].tolist() == pytest.approx([5.0])


def test_zero_advantage_counts_as_unclipped():
    """Both branches agree at A = 0, so nothing was actually truncated."""
    out = _run(advantages=[0.0], ratio=[3.0])
    assert out["was_clipped"].tolist() == [0.0]
    assert out["objective"].tolist() == pytest.approx([0.0])


def test_ratio_computed_from_log_probs():
    """exp(new - old); identical log-probs mean the policy has not moved."""
    out = _run(advantages=[1.0, 1.0], log_probs_new=[-1.0, -2.0], log_probs_old=[-1.0, -1.0])
    assert out["ratio"].tolist() == pytest.approx([1.0, torch.exp(torch.tensor(-1.0)).item()], abs=1e-5)


def test_first_inner_epoch_has_ratio_exactly_one_and_clips_nothing():
    """A real PPO step reuses one batch: on the first pass new == old, so the
    clip provably cannot engage. It only bites from the second epoch on."""
    lp = [-0.5, -1.2, -0.9]
    out = _run(advantages=[1.0, -1.0, 0.5], log_probs_new=lp, log_probs_old=lp)
    assert out["ratio"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert out["clip_fraction"] == pytest.approx(0.0)


def test_loss_is_negative_mean_objective():
    out = _run(advantages=[0.5, 0.8, -0.3], ratio=[1.05, 1.40, 0.55])
    assert float(out["loss"]) == pytest.approx(-float(out["objective"].mean()))


def test_epsilon_zero_pins_the_ratio_to_one():
    """No room to move: the clipped branch is always just A."""
    out = _run(advantages=[2.0], ratio=[1.9], epsilon=0.0)
    assert out["clipped"].tolist() == pytest.approx([2.0])
    assert out["objective"].tolist() == pytest.approx([2.0])


def test_larger_epsilon_clips_fewer_samples():
    ratios = [1.05, 1.25, 1.45]
    tight = _run(advantages=[1.0] * 3, ratio=ratios, epsilon=0.1)["clip_fraction"]
    loose = _run(advantages=[1.0] * 3, ratio=ratios, epsilon=0.5)["clip_fraction"]
    assert tight > loose


def test_missing_both_ratio_and_log_probs_is_an_error():
    with pytest.raises(ValueError, match="ratio"):
        _run(advantages=[1.0])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="per-sample"):
        _run(advantages=[1.0, 2.0], ratio=[1.0])


def test_negative_epsilon_is_rejected():
    with pytest.raises(ValueError, match="epsilon"):
        _run(advantages=[1.0], ratio=[1.0], epsilon=-0.1)
