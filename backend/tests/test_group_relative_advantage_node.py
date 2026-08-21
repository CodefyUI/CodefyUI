"""Tests for GroupRelativeAdvantageNode.

GRPO's one change to PPO. The 8-sample group here is I5-4's flagship run, so
the baseline 0.375 and the +0.625 / -0.375 split are published numbers.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.rl.group_relative_advantage_node import GroupRelativeAdvantageNode


def _run(rewards, group_ids=None, expand_index=None, **params):
    inputs = {"rewards": torch.as_tensor(rewards, dtype=torch.float32)}
    if group_ids is not None:
        inputs["group_ids"] = torch.as_tensor(group_ids)
    if expand_index is not None:
        inputs["expand_index"] = torch.as_tensor(expand_index)
    p = {"normalize": False}
    p.update(params)
    return GroupRelativeAdvantageNode().execute(inputs, p)


#: I5-4's rollout: 8 episodes in the gridworld, 3 of them reach the goal.
I5_4_RETURNS = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_node_metadata():
    assert GroupRelativeAdvantageNode.NODE_NAME == "GroupRelativeAdvantage"
    assert GroupRelativeAdvantageNode.CATEGORY == "RL"
    out_names = [p.name for p in GroupRelativeAdvantageNode.define_outputs()]
    for expected in ("advantages", "baseline", "group_mean", "group_std", "report",
                     "advantages_expanded"):
        assert expected in out_names


def test_reproduces_i5_4_flagship_group():
    """3 successes out of 8 -> baseline is 3/8, exactly."""
    out = _run(I5_4_RETURNS)
    assert out["group_mean"] == pytest.approx(0.375)
    assert out["advantages"].tolist() == pytest.approx(
        [-0.375, -0.375, -0.375, -0.375, 0.625, 0.625, -0.375, 0.625]
    )


def test_advantages_sum_to_zero():
    """Subtracting the mean is what that means. The fastest wiring check."""
    out = _run(I5_4_RETURNS)
    assert float(out["advantages"].sum()) == pytest.approx(0.0, abs=1e-6)


def test_report_states_the_zero_sum_check():
    assert "should be 0" in _run(I5_4_RETURNS)["report"]


def test_baseline_is_computed_not_estimated():
    """No network, no training: the baseline is the arithmetic mean."""
    out = _run(I5_4_RETURNS)
    assert out["baseline"].tolist() == pytest.approx([0.375] * 8)


@pytest.mark.parametrize("rewards", [[1.0] * 8, [0.0] * 8, [-1.0] * 5])
def test_a_group_with_no_variance_teaches_nothing(rewards):
    """All-same scores -> every advantage is 0 -> zero gradient. This is the
    constraint GRPO runs under, and I5-4 demonstrates it deliberately."""
    out = _run(rewards)
    assert out["group_std"] == pytest.approx(0.0)
    assert out["advantages"].tolist() == pytest.approx([0.0] * len(rewards))


def test_i5_4_degenerate_contrast_has_variance_on_one_side_only():
    """The controlled comparison: same everything, one config has signal."""
    good = _run(I5_4_RETURNS)
    dead = _run([0.0] * 8)
    assert good["group_std"] > 0
    assert dead["group_std"] == pytest.approx(0.0)


def test_group_ids_partition_the_batch():
    """Two prompts batched together get their own baselines."""
    out = _run([0.0, 1.0, 10.0, 20.0], group_ids=[0, 0, 1, 1])
    assert out["baseline"].tolist() == pytest.approx([0.5, 0.5, 15.0, 15.0])
    assert out["advantages"].tolist() == pytest.approx([-0.5, 0.5, -5.0, 5.0])


def test_each_group_sums_to_zero_independently():
    out = _run([0.0, 1.0, 10.0, 20.0], group_ids=[0, 0, 1, 1])
    adv = out["advantages"]
    assert float(adv[:2].sum()) == pytest.approx(0.0)
    assert float(adv[2:].sum()) == pytest.approx(0.0)


def test_group_mean_reports_the_first_group():
    out = _run([0.0, 1.0, 10.0, 20.0], group_ids=[0, 0, 1, 1])
    assert out["group_mean"] == pytest.approx(0.5)


def test_normalize_divides_by_group_std():
    out = _run([0.0, 1.0], normalize=True)
    # mean 0.5, population std 0.5 -> advantages become -1 and +1.
    assert out["advantages"].tolist() == pytest.approx([-1.0, 1.0])


def test_normalize_leaves_a_constant_group_alone():
    """std is 0; dividing by an epsilon would invent signal that is not there."""
    out = _run([3.0, 3.0, 3.0], normalize=True)
    assert out["advantages"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_empty_rewards_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        _run([])


def test_mismatched_group_ids_are_rejected():
    with pytest.raises(ValueError, match="same samples"):
        _run([1.0, 2.0, 3.0], group_ids=[0, 0])


def test_accepts_a_plain_list():
    out = GroupRelativeAdvantageNode().execute({"rewards": [0.0, 1.0]}, {})
    assert out["group_mean"] == pytest.approx(0.5)


def test_expand_index_broadcasts_advantages_to_samples():
    """Real GRPO gives every token of a response that response's advantage.
    I5-4 wires PolicyRollout's episode_ids in here to do exactly that."""
    out = _run([0.0, 1.0], group_ids=None, expand_index=[0, 0, 0, 1, 1])
    assert out["advantages_expanded"].tolist() == pytest.approx(
        [-0.5, -0.5, -0.5, 0.5, 0.5])


def test_expanded_defaults_to_the_unexpanded_advantages():
    out = _run(I5_4_RETURNS)
    assert out["advantages_expanded"].tolist() == pytest.approx(
        out["advantages"].tolist())


def test_expand_index_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="expand_index"):
        _run([0.0, 1.0], expand_index=[0, 1, 2])


def test_expanded_advantages_still_sum_to_zero_when_groups_are_balanced():
    """Equal-length episodes keep the zero-sum property after broadcasting."""
    out = _run([0.0, 1.0], expand_index=[0, 0, 1, 1])
    assert float(out["advantages_expanded"].sum()) == pytest.approx(0.0, abs=1e-6)
