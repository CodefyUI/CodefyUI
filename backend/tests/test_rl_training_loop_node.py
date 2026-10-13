"""Tests for RLTrainingLoopNode.

The GridWorld runs are the fast correctness checks for each algorithm. The
CartPole-v1 runs are the paper reproductions (REINFORCE, DQN, PPO) with the
recipes the node's PR documents; each stops at ``target_return`` and is
asserted with margin below it, so a platform whose floats send training down
a slightly different path still passes as long as it learns.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from app.nodes.rl.dqn_node import DQNNode
from app.nodes.rl.env_wrapper_node import EnvWrapperNode
from app.nodes.rl.gridworld_env_node import GridWorldEnvNode
from app.nodes.rl.ppo_node import PPONode
from app.nodes.rl.rl_training_loop_node import RLTrainingLoopNode
from app.nodes.training.optimizer_node import OptimizerNode


class _Ctx:
    """The slice of ExecutionContext the node uses, with a stop switch."""

    device = "cpu"

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int]] = []
        self.stop = False

    def should_stop(self) -> bool:
        return self.stop

    def log_metric(self, name, value, step, node_id=None) -> None:
        self.metrics.append((name, float(value), int(step)))


def _grid():
    return GridWorldEnvNode().execute(
        {}, {"size": 4, "traps": "1,1", "max_steps": 30})["env"]


def _cartpole():
    return EnvWrapperNode().execute({}, {"env_name": "CartPole-v1"})["env"]


def _build(kind: str, state_dim: int, action_dim: int, hidden: int, seed: int = 0):
    """A model built under a forked RNG, so tests never move the global one."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if kind == "linear":
            return nn.Sequential(nn.Linear(state_dim, action_dim))
        node = DQNNode if kind == "dqn" else PPONode
        return node().execute({}, {"state_dim": state_dim, "action_dim": action_dim,
                                   "hidden_dim": hidden})["model"]


def _train(model, env, lr, progress=None, context=None, **params):
    optimizer = OptimizerNode().execute({"model": model}, {"type": "Adam", "lr": lr})["optimizer"]
    p = {"device": "cpu"}
    p.update(params)
    return RLTrainingLoopNode().execute(
        {"model": model, "env": env, "optimizer": optimizer}, p,
        progress, context=context)


# ── metadata ─────────────────────────────────────────────────────────────


def test_node_metadata():
    assert RLTrainingLoopNode.NODE_NAME == "RLTrainingLoop"
    assert RLTrainingLoopNode.CATEGORY == "RL"
    assert RLTrainingLoopNode.cacheable is False
    assert [p.name for p in RLTrainingLoopNode.define_inputs()] == ["model", "env", "optimizer"]
    assert [p.name for p in RLTrainingLoopNode.define_outputs()] == [
        "model", "episode_returns", "losses", "metrics", "report"]


def test_algorithm_specific_params_are_shown_only_for_their_algorithm():
    params = {p.name: p for p in RLTrainingLoopNode.define_params()}
    assert params["algorithm"].options == ["reinforce", "ppo", "dqn"]
    assert params["clip_eps"].visible_when == {"algorithm": ["ppo"]}
    assert params["replay_size"].visible_when == {"algorithm": ["dqn"]}
    assert params["baseline"].visible_when == {"algorithm": ["reinforce"]}
    for shared in ("episodes", "gamma", "target_return", "seed", "max_steps_per_episode"):
        assert params[shared].visible_when is None


# ── GridWorld: each algorithm learns the 4x4 grid ──────────────────────


def test_reinforce_learns_gridworld():
    out = _train(_build("linear", 16, 4, 0), _grid(), 0.05, algorithm="reinforce",
                 episodes=800, episodes_per_update=8, normalize_returns=False, seed=0)
    assert out["metrics"]["mean_return_last_20"] >= 0.8


def test_dqn_learns_gridworld():
    out = _train(_build("dqn", 16, 4, 32), _grid(), 1e-3, algorithm="dqn",
                 episodes=300, learning_starts=200, target_update_every=100,
                 epsilon_decay_episodes=100, seed=0)
    assert out["metrics"]["mean_return_last_20"] >= 0.8


def test_ppo_learns_gridworld():
    out = _train(_build("ppo", 16, 4, 32), _grid(), 3e-3, algorithm="ppo",
                 episodes=300, steps_per_update=256, seed=0)
    assert out["metrics"]["mean_return_last_20"] >= 0.8


# ── CartPole-v1: the three papers ───────────────────────────────────────


def test_reinforce_balances_cartpole():
    out = _train(_build("dqn", 4, 2, 128), _cartpole(), 2e-3, algorithm="reinforce",
                 episodes=1000, target_return=475, seed=0)
    assert out["metrics"]["mean_return_last_20"] >= 150


def test_dqn_balances_cartpole():
    out = _train(_build("dqn", 4, 2, 128), _cartpole(), 5e-4, algorithm="dqn",
                 episodes=700, target_return=200, seed=0)
    assert out["metrics"]["mean_return_last_20"] >= 150


def test_ppo_balances_cartpole():
    out = _train(_build("ppo", 4, 2, 64), _cartpole(), 1e-3, algorithm="ppo",
                 episodes=1500, target_return=475, steps_per_update=512, ppo_epochs=10,
                 minibatch_size=64, value_coef=0.1, grad_clip_norm=0.5, seed=0)
    assert out["metrics"]["mean_return_last_20"] >= 150


# ── determinism, bookkeeping, stopping ──────────────────────────────────


@pytest.mark.parametrize("algorithm,kind", [("reinforce", "dqn"), ("dqn", "dqn"), ("ppo", "ppo")])
def test_same_seed_gives_identical_returns(algorithm, kind):
    model = _build(kind, 4, 2, 32)
    params = dict(algorithm=algorithm, episodes=25, seed=3, learning_starts=50,
                  steps_per_update=64)
    first = _train(copy.deepcopy(model), _cartpole(), 1e-3, **params)
    second = _train(copy.deepcopy(model), _cartpole(), 1e-3, **params)
    assert torch.equal(first["episode_returns"], second["episode_returns"])
    assert torch.equal(first["losses"], second["losses"])
    other = _train(copy.deepcopy(model), _cartpole(), 1e-3, **dict(params, seed=4))
    assert not torch.equal(first["episode_returns"], other["episode_returns"])


def test_training_does_not_touch_the_global_rng():
    before = torch.random.get_rng_state()
    _train(_build("dqn", 4, 2, 16), _cartpole(), 1e-3, algorithm="dqn",
           episodes=5, learning_starts=10, seed=0)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("algorithm,kind", [("reinforce", "dqn"), ("dqn", "dqn"), ("ppo", "ppo")])
def test_one_return_per_episode_and_metrics_agree(algorithm, kind):
    out = _train(_build(kind, 16, 4, 16), _grid(), 1e-3, algorithm=algorithm,
                 episodes=17, learning_starts=20, steps_per_update=50)
    metrics = out["metrics"]
    assert out["episode_returns"].shape == (17,)
    assert metrics["episodes"] == 17
    assert len(out["losses"]) == metrics["updates"] > 0
    assert metrics["solved_at_episode"] is None
    assert metrics["mean_return_last_20"] == pytest.approx(float(out["episode_returns"].mean()))
    assert "17 episodes" in out["report"]


def test_target_return_stops_the_run_and_is_recorded():
    ctx = _Ctx()
    out = _train(_build("dqn", 4, 2, 128), _cartpole(), 2e-3, context=ctx,
                 algorithm="reinforce", episodes=1000, target_return=100, seed=0)
    solved = out["metrics"]["solved_at_episode"]
    assert solved is not None and solved == out["metrics"]["episodes"] < 1000
    assert out["metrics"]["mean_return_last_20"] >= 100
    assert ("solved_at_episode", float(solved), solved) in ctx.metrics
    names = {name for name, _, _ in ctx.metrics}
    assert {"episode_return", "mean_return"} <= names
    last_mean = [v for name, v, _ in ctx.metrics if name == "mean_return"][-1]
    assert last_mean == pytest.approx(out["metrics"]["mean_return_last_20"])


def test_progress_frames_have_the_training_loop_shape():
    frames = []
    _train(_build("linear", 16, 4, 0), _grid(), 0.05, progress=frames.append,
           algorithm="reinforce", episodes=12)
    assert frames, "no progress frame"
    last = frames[-1]
    assert last["event"] == "epoch"
    assert last["epoch"] == 12 and last["total_epochs"] == 12
    assert isinstance(last["loss"], float)
    assert len(last["losses"]) == 12
    assert "mean_return" in last and "episode_return" in last


@pytest.mark.parametrize("algorithm,kind", [("reinforce", "dqn"), ("dqn", "dqn"), ("ppo", "ppo")])
def test_stop_ends_the_run_within_one_episode(algorithm, kind):
    ctx = _Ctx()

    def progress(frame):
        # The first frame always goes out (later ones are rate-limited).
        ctx.stop = True

    out = _train(_build(kind, 16, 4, 16), _grid(), 1e-3, progress=progress, context=ctx,
                 algorithm=algorithm, episodes=500, log_interval=1, learning_starts=10,
                 steps_per_update=40)
    assert out["metrics"]["interrupted"] is True
    assert out["metrics"]["episodes"] <= 2
    assert out["__interrupted__"]["epoch"] == out["metrics"]["episodes"]


# ── refusals ─────────────────────────────────────────────────────────────


def test_continuous_action_env_is_refused_by_name():
    env = EnvWrapperNode().execute({}, {"env_name": "Pendulum-v1"})["env"]
    with pytest.raises(ValueError, match=r"Pendulum-v1.*discrete action spaces only"):
        _train(_build("dqn", 3, 2, 8), env, 1e-3, algorithm="reinforce", episodes=1)


def test_ppo_refuses_a_model_without_a_value_head():
    with pytest.raises(ValueError, match=r"'ppo' needs an actor-critic"):
        _train(_build("dqn", 4, 2, 8), _cartpole(), 1e-3, algorithm="ppo", episodes=1)


def test_dqn_refuses_an_actor_critic():
    with pytest.raises(ValueError, match=r"'dqn' needs a model that returns one Q-value"):
        _train(_build("ppo", 4, 2, 8), _cartpole(), 1e-3, algorithm="dqn", episodes=1)


def test_a_state_dim_mismatch_names_both_sizes():
    with pytest.raises(ValueError, match=r"16 features but the model expects 4"):
        _train(_build("dqn", 4, 4, 8), _grid(), 1e-3, algorithm="reinforce", episodes=1)


def test_an_action_dim_mismatch_names_both_sizes():
    with pytest.raises(ValueError, match=r"has 2 actions but the model outputs 3"):
        _train(_build("dqn", 4, 3, 8), _cartpole(), 1e-3, algorithm="dqn", episodes=1)


# ── through the engine and through the Python export ────────────────────


def _graph(algorithm: str = "dqn") -> tuple[list[dict], list[dict]]:
    def node(node_id, node_type, **params):
        return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
                "data": {"params": params}}

    def edge(source, source_handle, target, target_handle):
        return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
                "source": source, "target": target, "sourceHandle": source_handle,
                "targetHandle": target_handle, "type": "data"}

    nodes = [
        node("start", "Start"),
        node("env", "EnvWrapper", env_name="CartPole-v1"),
        node("net", "DQN", state_dim=4, action_dim=2, hidden_dim=16),
        node("opt", "Optimizer", type="Adam", lr=0.001),
        node("train", "RLTrainingLoop", algorithm=algorithm, episodes=4,
             learning_starts=20, seed=0, device="cpu"),
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "env", "sourceHandle": "trigger",
         "targetHandle": "", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "net", "sourceHandle": "trigger",
         "targetHandle": "", "type": "trigger"},
        edge("net", "model", "opt", "model"),
        edge("net", "model", "train", "model"),
        edge("opt", "optimizer", "train", "optimizer"),
        edge("env", "env", "train", "env"),
    ]
    return nodes, edges


async def test_a_graph_trains_through_execute_graph():
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    nodes, edges = _graph()
    first = await execute_graph(nodes, edges, context=ExecutionContext(device="cpu", seed=1))
    second = await execute_graph(nodes, edges, context=ExecutionContext(device="cpu", seed=1))
    assert first["train"]["episode_returns"].shape == (4,)
    assert first["train"]["metrics"]["episodes"] == 4
    assert torch.equal(first["train"]["episode_returns"], second["train"]["episode_returns"])


def test_the_same_graph_runs_from_the_python_export():
    from app.core.codegen import generate_python
    from app.core.execution_context import ExecutionContext

    nodes, edges = _graph("reinforce")
    script = generate_python(nodes, edges, name="rl-loop", seed=1)
    module: dict = {"__name__": "generated_export"}
    exec(compile(script, "<generated>", "exec"), module)  # noqa: S102
    module["_RT"] = module["_load_runtime"](None)
    module["_RT"].initialize_runtime()
    results = module["run_graph"](ExecutionContext(seed=1, device="cpu"), {})
    assert results["train"]["episode_returns"].shape == (4,)
    assert "reinforce on CartPole-v1" in results["train"]["report"]
