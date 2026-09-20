"""Tests for RewardModelNode."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.core.graph_engine import execute_graph
from app.nodes.rl.reward_model_node import RewardModelNode, _RewardHead

#: The pack example whose note quotes what two seed-42 heads score.
_PACK_GRAPH = (Path(__file__).resolve().parents[2] / "plugins" / "rl"
               / "examples" / "C5-3" / "RLHF-Reward-Model" / "graph.json")


def _run(hidden_states=None, **params):
    p = {"input_dim": 8, "hidden_dim": 16, "seed": 42}
    p.update(params)
    inputs = {}
    if hidden_states is not None:
        inputs["hidden_states"] = hidden_states
    return RewardModelNode().execute(inputs, p)


def test_node_metadata():
    assert RewardModelNode.NODE_NAME == "RewardModel"
    assert RewardModelNode.CATEGORY == "RL"
    out_names = [p.name for p in RewardModelNode.define_outputs()]
    assert "model" in out_names
    assert "rewards" in out_names


def test_returns_module_when_no_input():
    res = _run()
    assert isinstance(res["model"], nn.Module)


def test_scores_2d_hidden_states():
    """[B, H] → scalar reward per item, shape [B]."""
    h = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    res = _run(h)
    assert res["rewards"].shape == (4,)


def test_uses_last_token_for_3d_input():
    """[B, T, H] → uses the last token to score the sequence, [B]."""
    h = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    res = _run(h)
    assert res["rewards"].shape == (2,)


def test_seed_makes_init_reproducible():
    h = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
    a = _run(h, seed=42)
    b = _run(h, seed=42)
    assert torch.allclose(a["rewards"], b["rewards"])
    c = _run(h, seed=99)
    assert not torch.allclose(a["rewards"], c["rewards"])


def test_dim_mismatch_raises():
    h = torch.randn(2, 16)
    with pytest.raises(RuntimeError):
        _run(h, input_dim=8)


# ── the seed alone decides the weights ────────────────────────────────────


def _draw_beside_every_linear(monkeypatch) -> list[torch.Tensor]:
    """Make another node draw from the global RNG in the middle of a build.

    A deterministic stand-in for the engine running a level's nodes on
    worker threads: every ``nn.Linear`` the build constructs is preceded by
    one draw from the process-global generator, which is what a neighbouring
    node's draw landing mid-construction amounts to. Returns what that
    neighbour drew, in order.
    """
    drawn: list[torch.Tensor] = []
    original = nn.Linear.reset_parameters

    def interleaved(self):
        drawn.append(torch.rand(1))
        return original(self)

    monkeypatch.setattr(nn.Linear, "reset_parameters", interleaved)
    return drawn


def _differing(a: dict, b: dict) -> list[str]:
    assert list(a) == list(b)
    return [key for key in a if not torch.equal(a[key], b[key])]


def test_same_seed_gives_the_same_weights_while_another_node_draws(monkeypatch):
    """The CI failure on the RLHF pack example, made deterministic.

    Two heads built from seed 42 on one level of an unseeded run came out
    with different weights, because each seeded the process-global
    generator and the other drew from it in between.
    """
    params = {"input_dim": 128, "hidden_dim": 64, "seed": 42}
    alone = _run(**params)["model"].state_dict()

    _draw_beside_every_linear(monkeypatch)
    beside = _run(**params)["model"].state_dict()

    assert not _differing(alone, beside), (
        f"{_differing(alone, beside)} moved because another node drew from "
        f"the global RNG while the head was built")


@pytest.mark.parametrize(("input_dim", "hidden_dim", "seed"), [
    (128, 64, 42),   # plugins/rl/examples/C5-3/RLHF-Reward-Model
    (16, 32, 42),    # examples/RL/RLHF-Reward-and-KL
    (8, 16, 0),
    (8, 16, -7),     # the seed param has no minimum
    (8, 16, 2 ** 40),
])
def test_the_weights_are_the_ones_the_seeded_constructor_gave(
        input_dim, hidden_dim, seed):
    """Bit for bit what ``torch.manual_seed(seed)`` + the constructor gave.

    That is how the node built its head before it stopped touching the
    global RNG, and every number an example note quotes off a RewardModel
    was read off a head built that way.
    """
    torch.manual_seed(seed)
    expected = _RewardHead(input_dim, hidden_dim).state_dict()

    got = _run(input_dim=input_dim, hidden_dim=hidden_dim,
               seed=seed)["model"].state_dict()

    assert not _differing(expected, got)


def test_building_neither_reseeds_nor_rewinds_the_global_rng(monkeypatch):
    """The process-global generator belongs to the run, not to this node.

    Reseeding it hands a node drawing beside this one the seed-42 stream;
    restoring it afterwards (``fork_rng``, or get/set_rng_state) rewinds
    the stream under that node, so the same numbers are handed out twice.
    The constructor's own draws are consumption, which any unseeded node
    does, so they are allowed.
    """
    torch.manual_seed(1234)
    first = torch.rand(1)
    torch.manual_seed(1234)
    drawn = _draw_beside_every_linear(monkeypatch)

    _run()
    after = torch.rand(1)

    assert torch.equal(drawn[0], first), (
        "the node reseeded the global RNG: a draw made beside it came from "
        "another stream than the run's")
    assert not any(torch.equal(after, value) for value in drawn), (
        "the node rewound the global RNG: a value drawn beside it came out "
        "again after it finished")


def test_the_pack_example_scores_the_same_on_every_parallel_run():
    """A SMOKE test; the deterministic guarantee is the test above it.

    Runs the RLHF pack example the way the canvas does without a run seed:
    both heads sit on one level, so the engine builds them on worker
    threads at the same time. That is how its note came out wrong on a CI
    runner (0.086 and -0.236 against the 0.041 and 0.177 it quotes), and
    whether a given run overlaps the two builds is up to the scheduler --
    so passing here proves little, while
    ``test_same_seed_gives_the_same_weights_while_another_node_draws``
    forces the overlap every time.
    """
    graph = json.loads(_PACK_GRAPH.read_text(encoding="utf-8"))
    note = next(n for n in graph["nodes"]
                if n["id"] == "note-overview")["data"]["noteContent"]
    quoted = {"rm_chosen": "0.041", "rm_reject": "0.177"}
    assert all(value in note for value in quoted.values()), (
        "the note no longer quotes these; update the pin with it")

    runs = [asyncio.run(execute_graph(graph["nodes"], graph["edges"],
                                      error_mode="fail_fast"))
            for _ in range(20)]

    for i, results in enumerate(runs):
        chosen = results["rm_chosen"]["model"].state_dict()
        rejected = results["rm_reject"]["model"].state_dict()
        assert not _differing(chosen, rejected), (
            f"run {i}: two heads built from seed 42 hold different weights")
        for node_id, value in quoted.items():
            rewards = results[node_id]["rewards"]
            assert torch.equal(rewards, runs[0][node_id]["rewards"]), (
                f"run {i}: {node_id} scored {rewards.tolist()}, run 0 "
                f"scored {runs[0][node_id]['rewards'].tolist()}")
            assert f"{float(rewards[0]):.3f}" == value, (
                f"run {i}: {node_id} scored {float(rewards[0]):.3f}, the "
                f"note says {value}")
