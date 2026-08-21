"""Tests for DiscountNode.

The numbers pinned here are printed in the textbook (I5-1), so a change that
moves them breaks a published claim, not just a test.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.rl.discount_node import DiscountNode


def _run(rewards, episode_ids=None, **params):
    inputs = {"rewards": torch.as_tensor(rewards, dtype=torch.float32)}
    if episode_ids is not None:
        inputs["episode_ids"] = torch.as_tensor(episode_ids)
    p = {"gamma": 0.9}
    p.update(params)
    return DiscountNode().execute(inputs, p)


def test_node_metadata():
    assert DiscountNode.NODE_NAME == "Discount"
    assert DiscountNode.CATEGORY == "RL"
    out_names = [p.name for p in DiscountNode.define_outputs()]
    assert "returns" in out_names
    assert "first_return" in out_names


def test_reproduces_c5_1_example_short_path():
    """C5-1 example 5-1-a: reaching the goal in 5 steps gives 0.9^4 = 0.6561."""
    rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert _run(rewards)["first_return"] == pytest.approx(0.6561, abs=1e-4)


def test_reproduces_c5_1_example_long_path():
    """The same goal reached in 8 steps is worth 0.9^7 = 0.4783 -- less."""
    rewards = [0.0] * 7 + [1.0]
    assert _run(rewards)["first_return"] == pytest.approx(0.4783, abs=1e-4)


def test_discount_makes_the_faster_route_worth_more():
    """The whole point of gamma: same reward, sooner is better."""
    fast = _run([0.0] * 4 + [1.0])["first_return"]
    slow = _run([0.0] * 7 + [1.0])["first_return"]
    assert fast > slow


@pytest.mark.parametrize(
    "gamma,expected",
    [(1.0, 1.0000), (0.9, 0.2542), (0.5, 0.0001)],
)
def test_i5_1_gamma_sweep_on_the_14_step_episode(gamma, expected):
    """I5-1's gamma table. Reward lands on step 13, so G_0 = gamma ** 13."""
    rewards = [0.0] * 13 + [1.0]
    assert _run(rewards, gamma=gamma)["first_return"] == pytest.approx(expected, abs=1e-4)


def test_gamma_one_passes_the_reward_back_undiscounted():
    out = _run([0.0, 0.0, 5.0], gamma=1.0)
    assert out["returns"].tolist() == pytest.approx([5.0, 5.0, 5.0])


def test_episode_ids_reset_the_fold_at_boundaries():
    """Without the reset, episode 0 would inherit episode 1's reward."""
    rewards = [0.0, 1.0, 0.0, 1.0]
    ids = [0, 0, 1, 1]
    returns = _run(rewards, episode_ids=ids, gamma=1.0)["returns"].tolist()
    assert returns == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_without_episode_ids_everything_is_one_episode():
    """Same rewards, no ids: the second episode's reward now leaks backwards."""
    returns = _run([0.0, 1.0, 0.0, 1.0], gamma=1.0)["returns"].tolist()
    assert returns == pytest.approx([2.0, 2.0, 1.0, 1.0])


def test_first_return_is_the_first_entry_of_returns():
    out = _run([0.0] * 4 + [1.0])
    assert out["first_return"] == pytest.approx(float(out["returns"][0]))


def test_accepts_a_plain_list():
    """Graphs hand this node whatever the upstream port produced."""
    out = DiscountNode().execute({"rewards": [0.0, 0.0, 1.0]}, {"gamma": 1.0})
    assert out["first_return"] == pytest.approx(1.0)
