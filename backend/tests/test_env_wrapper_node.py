"""Tests for EnvWrapperNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.rl.env_wrapper_node import EnvWrapperNode


def test_node_metadata():
    assert EnvWrapperNode.NODE_NAME == "EnvWrapper"
    assert EnvWrapperNode.CATEGORY == "RL"
    out_names = [p.name for p in EnvWrapperNode.define_outputs()]
    assert "env" in out_names
    assert "observation" in out_names


def test_default_creates_cartpole_env():
    res = EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})
    assert res["env"] is not None
    # CartPole observation is 4D
    assert res["observation"].shape == (4,)
    assert res["observation"].dtype == torch.float32


def test_observation_is_tensor_type():
    res = EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})
    assert isinstance(res["observation"], torch.Tensor)


def test_unknown_env_raises():
    with pytest.raises(Exception):
        EnvWrapperNode().execute({}, {"env_name": "DefinitelyNotARealEnv-v99"})


def test_env_can_step():
    res = EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})
    env = res["env"]
    obs, reward, terminated, truncated, info = env.step(0)
    assert len(obs) == 4


# ── the run seed reaches the environment (#706) ──────────────────────────


def _rollout_graph() -> tuple[list[dict], list[dict]]:
    """EnvWrapper(CartPole-v1) -> PolicyRollout, as the issue wires it."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "env", "type": "EnvWrapper",
         "data": {"params": {"env_name": "CartPole-v1"}}},
        {"id": "dqn", "type": "DQN",
         "data": {"params": {"state_dim": 4, "action_dim": 2,
                             "hidden_dim": 8}}},
        {"id": "roll", "type": "PolicyRollout",
         "data": {"params": {"episodes": 2, "temperature": 1.0, "seed": 0}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "env",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "dqn",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e1", "source": "env", "target": "roll",
         "sourceHandle": "env", "targetHandle": "env"},
        {"id": "e2", "source": "dqn", "target": "roll",
         "sourceHandle": "model", "targetHandle": "model"},
    ]
    return nodes, edges


async def _run(seed: int | None, *, rollout: bool = False) -> dict:
    """Execute EnvWrapper alone, or the whole rollout graph."""
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    nodes, edges = _rollout_graph()
    if not rollout:
        nodes, edges = nodes[:2], edges[:1]
    return await execute_graph(
        nodes, edges, context=ExecutionContext(device="cpu", seed=seed))


async def test_same_run_seed_gives_the_same_observation_and_rollout():
    first, second = await _run(3), await _run(3)
    assert torch.equal(first["env"]["observation"],
                       second["env"]["observation"])
    first, second = await _run(3, rollout=True), await _run(3, rollout=True)
    assert torch.equal(first["roll"]["states"], second["roll"]["states"])
    assert torch.equal(first["roll"]["actions"], second["roll"]["actions"])


async def test_different_run_seeds_give_different_observations():
    first, second = await _run(3), await _run(4)
    assert not torch.equal(first["env"]["observation"],
                           second["env"]["observation"])


async def test_an_unseeded_run_still_varies():
    first, second = await _run(None), await _run(None)
    assert not torch.equal(first["env"]["observation"],
                           second["env"]["observation"])


def test_a_seeded_run_seeds_the_action_space():
    from app.core.execution_context import ExecutionContext

    def samples() -> list[int]:
        ctx = ExecutionContext(seed=3)
        ctx.current_node_id = "env"
        env = EnvWrapperNode().execute(
            {}, {"env_name": "CartPole-v1"}, context=ctx)["env"]
        return [int(env.action_space.sample()) for _ in range(32)]

    assert samples() == samples()
