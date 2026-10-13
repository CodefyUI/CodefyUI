"""The four reinforcement-learning paper reproductions reach what their notes say.

``examples/Papers/{REINFORCE,DQN,PPO}-CartPole`` each claim to solve
CartPole-v1 (mean return of the last 20 episodes >= 475) with the stored
run seed on CPU, and ``examples/Papers/QLearning-GridWorld`` claims the
trained policy walks to the goal on the shortest path. These graphs are on
``_SLOW_NODE_TYPES`` in ``test_builtin_examples.py`` (the DQN run alone takes
about half a minute), so this file is where they are executed.

``RLTrainingLoop`` draws every random number from a local generator and the
model's initial weights come from the run seed, so on CPU the episode counts
quoted in the notes should repeat on every platform. The assertions hold the
claim with margin anyway: a platform whose floats send training down a
slightly different path still passes as long as it learns.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
from pathlib import Path

import pytest

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"
_BACKEND_DIR = _REPO_ROOT / "backend"

_CARTPOLE = {
    "Papers/REINFORCE-CartPole": "reinforce",
    "Papers/DQN-CartPole": "dqn",
    "Papers/PPO-CartPole": "ppo",
}
_GRID = "Papers/QLearning-GridWorld"

#: CartPole-v1 counts as solved at this mean return of the last 20 episodes.
_SOLVED = 475.0


def _graph(key: str) -> dict:
    return json.loads((_EXAMPLES_ROOT / key / "graph.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _run(key: str) -> dict:
    """Execute the graph under the run seed its file stores, on CPU."""
    graph = _graph(key)
    prev_cwd = Path.cwd()
    os.chdir(_BACKEND_DIR)
    try:
        return asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast",
            context=ExecutionContext(device="cpu", seed=graph["settings"]["seed"])))
    finally:
        os.chdir(prev_cwd)


def _loop_params(key: str) -> dict:
    graph = _graph(key)
    (loop,) = [n for n in graph["nodes"] if n["type"] == "RLTrainingLoop"]
    return loop["data"]["params"]


def _overview(key: str) -> str:
    graph = _graph(key)
    return next(n for n in graph["nodes"] if n["id"] == "note-overview")["data"]["noteContent"]


@pytest.mark.parametrize("key", [*_CARTPOLE, _GRID])
def test_the_loop_runs_on_cpu_with_the_stored_seed(key: str):
    """The notes quote CPU numbers, so the graph pins the device."""
    params = _loop_params(key)
    assert params["device"] == "cpu"
    assert _graph(key)["settings"] == {"seed": 0}
    assert params["seed"] == 0


@pytest.mark.parametrize("key,algorithm", list(_CARTPOLE.items()))
def test_each_cartpole_example_solves_cartpole(key: str, algorithm: str):
    params = _loop_params(key)
    assert params["algorithm"] == algorithm
    assert params["target_return"] == _SOLVED

    results = _run(key)["loop"]
    metrics = results["metrics"]
    returns = results["episode_returns"]

    assert metrics["solved_at_episode"] is not None, metrics
    assert len(returns) == metrics["episodes"] == metrics["solved_at_episode"]
    assert metrics["episodes"] < params["episodes"], "solved before the budget ran out"
    assert float(returns[-20:].mean()) >= _SOLVED, returns[-20:].tolist()
    # Generous: 2 s, 27 s and 8 s on an M3 MacBook Air.
    assert metrics["wall_time"] < 600, metrics
    assert "475" in _overview(key)


@pytest.mark.parametrize("key", list(_CARTPOLE))
def test_the_cartpole_overview_quotes_an_episode_inside_the_budget(key: str):
    """The quoted episode is this machine's measurement. The trajectory follows
    each platform's float rounding (REINFORCE solves at 280 on an Apple M3 and
    on Linux, 932 on the Windows runner), so the run is held to the claim in
    the test above and the note only to quoting a solve inside the budget."""
    budget = _loop_params(key)["episodes"]
    note = _overview(key)
    quoted = int(note.split("solves CartPole-v1 at episode ")[1].split()[0])
    assert 0 < quoted < budget, (key, quoted, budget)


def test_the_dqn_example_clips_its_gradient():
    """The note explains why: CartPole returns grow to 500 and so do the targets."""
    params = _loop_params("Papers/DQN-CartPole")
    assert params["grad_clip_norm"] == 10
    assert params["double_dqn"] is True


def test_the_gridworld_policy_walks_the_shortest_path_to_the_goal():
    results = _run(_GRID)
    rollout = results["rollout"]

    assert rollout["success_rate"] >= 0.9, rollout["report"]
    lengths = rollout["episode_lengths"].tolist()
    # From (0,0) to (3,3) on a 4x4 grid is 6 moves; nothing shorter exists.
    assert min(lengths) == 6
    assert sorted(lengths)[len(lengths) // 2] == 6, lengths

    graph = _graph(_GRID)
    env = next(n for n in graph["nodes"] if n["type"] == "GridWorldEnv")
    assert env["data"]["params"]["size"] == 4
    assert results["env"]["layout"].count("X") == 2, results["env"]["layout"]
    # The rollout plays the trained model, after training.
    assert any(e["source"] == "loop" and e["sourceHandle"] == "model"
               and e["target"] == "rollout" and e["targetHandle"] == "model"
               for e in graph["edges"])
