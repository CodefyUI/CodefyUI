"""Tests for GridWorldEnvNode.

The 4x4 layout, the one-hot observation and the three terminal outcomes are
all quoted in I5-1, and the policy in that chapter is a single Linear sized
against ``state_dim`` / ``action_dim``.
"""

from __future__ import annotations

import pytest

from app.nodes.rl.gridworld_env_node import GridWorldEnvNode


def _run(**params):
    p = {"size": 4, "traps": "1,1", "step_reward": 0.0,
         "goal_reward": 1.0, "trap_reward": -1.0, "max_steps": 30}
    p.update(params)
    return GridWorldEnvNode().execute({}, p)


def test_node_metadata():
    assert GridWorldEnvNode.NODE_NAME == "GridWorldEnv"
    assert GridWorldEnvNode.CATEGORY == "RL"
    out_names = [p.name for p in GridWorldEnvNode.define_outputs()]
    for expected in ("env", "state_dim", "action_dim", "layout"):
        assert expected in out_names


def test_not_cacheable():
    """The env carries position state; a cache hit would replay a used world."""
    assert GridWorldEnvNode.cacheable is False


def test_dims_match_the_grid():
    out = _run()
    assert out["state_dim"] == 16
    assert out["action_dim"] == 4


def test_layout_is_the_picture_i5_1_prints():
    assert _run()["layout"] == "S . . .\n. X . .\n. . . .\n. . . G"


def test_layout_without_traps():
    assert _run(traps="")["layout"] == "S . . .\n. . . .\n. . . .\n. . . G"


def test_reset_returns_one_hot_at_the_start_cell():
    obs = _run()["env"].reset()
    assert obs.shape == (16,)
    assert float(obs.sum()) == pytest.approx(1.0)
    assert int(obs.argmax()) == 0


def test_reaching_the_goal_ends_the_episode_with_the_goal_reward():
    env = _run(traps="")["env"]
    env.reset()
    for action in (2, 2, 2, 1, 1):  # down x3, right x2
        obs, reward, done, info = env.step(action)
        assert not done
    obs, reward, done, info = env.step(1)  # the last step right lands on (3,3)
    assert done and info["outcome"] == "goal"
    assert reward == pytest.approx(1.0)
    assert int(obs.argmax()) == 15


def test_stepping_into_a_trap_ends_the_episode_with_the_trap_reward():
    env = _run()["env"]
    env.reset()
    env.step(2)  # down to (1,0)
    _, reward, done, info = env.step(1)  # right into the trap at (1,1)
    assert done and info["outcome"] == "trap"
    assert reward == pytest.approx(-1.0)


def test_running_out_of_steps_ends_the_episode_as_a_timeout():
    env = _run(traps="", max_steps=3)["env"]
    env.reset()
    env.step(1)
    env.step(3)
    _, reward, done, info = env.step(1)
    assert done and info["outcome"] == "timeout"
    assert reward == pytest.approx(0.0)


def test_walking_into_a_wall_costs_a_step_and_does_not_move():
    """I5-1's trajectory table has four of these; they are not a special case."""
    env = _run(traps="")["env"]
    env.reset()
    obs, reward, done, info = env.step(0)  # up, from the top row
    assert int(obs.argmax()) == 0
    assert not done and info["outcome"] == "step"


def test_wall_bumps_still_count_towards_max_steps():
    env = _run(traps="", max_steps=2)["env"]
    env.reset()
    env.step(0)
    _, _, done, info = env.step(0)
    assert done and info["outcome"] == "timeout"


def test_step_reward_is_paid_on_ordinary_steps():
    env = _run(traps="", step_reward=-0.05)["env"]
    env.reset()
    _, reward, _, _ = env.step(2)
    assert reward == pytest.approx(-0.05)


def test_reset_clears_position_and_step_count():
    env = _run(traps="", max_steps=3)["env"]
    env.reset()
    env.step(2)
    env.step(2)
    obs = env.reset()
    assert int(obs.argmax()) == 0
    _, _, done, _ = env.step(1)
    assert not done  # the earlier steps did not carry over


def test_several_traps_can_be_given():
    env = _run(traps="1,1; 2,3")["env"]
    assert env.traps == {(1, 1), (2, 3)}


def test_an_invalid_action_is_rejected():
    env = _run()["env"]
    env.reset()
    with pytest.raises(ValueError, match="0-3"):
        env.step(9)


@pytest.mark.parametrize(
    "spec,match",
    [
        ("1", "row,col"),
        ("a,b", "two integers"),
        ("9,9", "outside"),
        ("0,0", "start cell"),
        ("3,3", "goal cell"),
    ],
)
def test_bad_trap_specs_fail_on_this_node_with_a_named_cause(spec, match):
    with pytest.raises(ValueError, match=match):
        _run(traps=spec)


def test_a_larger_grid_scales_state_dim():
    out = _run(size=5, traps="")
    assert out["state_dim"] == 25
    assert out["action_dim"] == 4
