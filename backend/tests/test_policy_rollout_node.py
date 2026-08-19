"""Tests for PolicyRolloutNode.

The seed-8 / seed-0 episodes and the 8-episode group are the runs I5-1 and
I5-4 print verbatim, so they are pinned here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.rl.gridworld_env_node import GridWorldEnvNode
from app.nodes.rl.policy_rollout_node import PolicyRolloutNode


def _policy(seed: int = 0):
    """I5-1's agent: one Linear, one weight per (cell, direction)."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 4))


def _env(**params):
    p = {"size": 4, "traps": "1,1", "max_steps": 30}
    p.update(params)
    return GridWorldEnvNode().execute({}, p)["env"]


def _run(model=None, env=None, **params):
    p = {"episodes": 1, "temperature": 1.0, "seed": 0}
    p.update(params)
    inputs = {"model": model if model is not None else _policy(),
              "env": env if env is not None else _env()}
    return PolicyRolloutNode().execute(inputs, p)


def test_node_metadata():
    assert PolicyRolloutNode.NODE_NAME == "PolicyRollout"
    assert PolicyRolloutNode.CATEGORY == "RL"
    out_names = [p.name for p in PolicyRolloutNode.define_outputs()]
    for expected in ("states", "actions", "rewards", "logits", "returns",
                     "episode_lengths", "episode_ids", "success_rate", "report",
                     "trajectory", "log_probs"):
        assert expected in out_names


def test_i5_1_flagship_episode_reaches_the_goal_in_14_steps():
    out = _run(seed=8)
    assert out["report"] == "episode 0:  14 steps, return +1.000, ended at goal"
    assert int(out["episode_lengths"][0]) == 14
    assert float(out["returns"][0]) == pytest.approx(1.0)


def test_i5_1_seed_zero_walks_into_the_trap():
    """Same model, same world, same start -- a different seed, a different fate."""
    out = _run(seed=0)
    assert out["report"] == "episode 0:   5 steps, return -1.000, ended at trap"


def test_the_reward_is_sparse_and_terminal():
    """14 actions, one non-zero reward, and it is the last one."""
    rewards = _run(seed=8)["rewards"]
    assert float(rewards[-1]) == pytest.approx(1.0)
    assert rewards[:-1].abs().sum() == pytest.approx(0.0)


def test_same_seed_is_reproducible():
    a = _run(seed=8)
    b = _run(seed=8)
    assert a["actions"].tolist() == b["actions"].tolist()


def test_sampling_not_argmax_makes_seeds_diverge():
    """pi(a|s) is a distribution, not a lookup table. This is GRPO's basis."""
    a = _run(seed=8)["actions"].tolist()
    b = _run(seed=0)["actions"].tolist()
    assert a != b


def test_i5_4_group_of_eight_has_three_successes():
    out = _run(env=_env(traps="", max_steps=40), episodes=8, seed=0)
    assert out["success_rate"] == pytest.approx(0.375)
    assert [float(x) for x in out["returns"]] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    )


def test_i5_4_degenerate_group_has_no_successes():
    """max_steps 40 -> 20 is the single variable I5-4 changes; signal vanishes."""
    out = _run(env=_env(traps="", max_steps=20), episodes=8, seed=0)
    assert out["success_rate"] == pytest.approx(0.0)
    assert [float(x) for x in out["returns"]] == pytest.approx([0.0] * 8)


def test_shapes_line_up_across_outputs():
    out = _run(episodes=3, seed=8)
    n = out["actions"].shape[0]
    assert out["states"].shape == (n, 16)
    assert out["rewards"].shape == (n,)
    assert out["logits"].shape == (n, 4)
    assert out["episode_ids"].shape == (n,)
    assert out["returns"].shape == (3,)
    assert out["episode_lengths"].shape == (3,)


def test_episode_ids_match_the_episode_lengths():
    out = _run(episodes=3, seed=8)
    counts = torch.bincount(out["episode_ids"], minlength=3)
    assert counts.tolist() == out["episode_lengths"].tolist()


def test_report_has_one_line_per_episode():
    assert len(_run(episodes=4, seed=8)["report"].splitlines()) == 4


def test_actions_are_valid_indices():
    actions = _run(episodes=4, seed=8)["actions"]
    assert int(actions.min()) >= 0
    assert int(actions.max()) <= 3


def test_accepts_an_actor_critic_returning_a_tuple():
    """PPO-style models hand back (probs, value); the node takes the policy half."""

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Linear(16, 4)
            self.v = nn.Linear(16, 1)

        def forward(self, x):
            return torch.softmax(self.pi(x), dim=-1), self.v(x)

    torch.manual_seed(0)
    out = _run(model=ActorCritic(), seed=8)
    assert out["logits"].shape[1] == 4
    assert torch.isfinite(out["logits"]).all()


def test_model_is_left_in_training_mode_if_it_started_there():
    model = _policy()
    model.train()
    _run(model=model, seed=8)
    assert model.training is True


def test_an_env_without_reset_is_rejected():
    with pytest.raises(ValueError, match="does not look like an environment"):
        PolicyRolloutNode().execute({"model": _policy(), "env": object()}, {})


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        _run(temperature=0.0)


def test_low_temperature_still_produces_a_valid_rollout():
    """Exploit-only. I5-1 shows this fails to reach the goal, which is the point."""
    out = _run(env=_env(traps="", max_steps=40), episodes=5, temperature=0.1, seed=0)
    assert 0.0 <= out["success_rate"] <= 1.0
    assert out["actions"].shape[0] > 0


def test_trajectory_lists_every_step_of_the_first_episode():
    """I5-1 prints this table; its shape is what makes the sparse reward
    visible -- 14 rows, one non-zero in the reward column, at the bottom."""
    out = _run(seed=8)
    lines = out["trajectory"].splitlines()
    assert lines[0].split() == ["step", "|", "state", "|", "action", "|", "reward"]
    body = lines[1:-1]
    assert len(body) == 14
    assert body[-1].endswith("+1.000")
    assert all(row.endswith("+0.000") for row in body[:-1])
    assert lines[-1] == "episode 0 ended at goal"


def test_trajectory_state_column_is_the_hot_index():
    out = _run(seed=8)
    first_row = out["trajectory"].splitlines()[1]
    assert int(first_row.split("|")[1]) == 0  # the start cell


def test_trajectory_covers_only_the_first_episode():
    out = _run(episodes=3, seed=8)
    body = out["trajectory"].splitlines()[1:-1]
    assert len(body) == int(out["episode_lengths"][0])


def test_trajectory_names_the_ending_for_a_trap():
    assert _run(seed=0)["trajectory"].splitlines()[-1] == "episode 0 ended at trap"


def test_log_probs_are_the_behaviour_policy_record():
    """PPO's ratio is measured against these; once the policy moves they are
    gone, so the rollout has to record them at sampling time."""
    out = _run(seed=8)
    lp = out["log_probs"]
    assert lp.shape == out["actions"].shape
    assert bool((lp < 0).all())          # log of a probability
    assert torch.isfinite(lp).all()


def test_log_probs_match_a_hand_recomputation():
    """log pi(a_t | s_t) for the action actually taken, at the sampling temperature."""
    model = _policy()
    out = _run(model=model, seed=8)
    with torch.no_grad():
        expected = torch.log_softmax(out["logits"], dim=-1).gather(
            1, out["actions"].unsqueeze(1)).squeeze(1)
    assert out["log_probs"].tolist() == pytest.approx(expected.tolist(), abs=1e-5)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_log_probs_are_taken_at_the_sampling_temperature(temperature):
    """The recorded log-prob must come from the distribution actually sampled
    from, not the raw logits -- otherwise PPO's ratio has the wrong denominator.

    Comparing two temperatures head-to-head would not be a controlled test:
    changing the temperature changes which states get visited, so the two runs
    are different trajectories. This checks the invariant on one run instead.
    """
    out = _run(seed=8, temperature=temperature)
    expected = torch.log_softmax(out["logits"] / temperature, dim=-1).gather(
        1, out["actions"].unsqueeze(1)).squeeze(1)
    assert out["log_probs"].tolist() == pytest.approx(expected.tolist(), abs=1e-5)
