"""A seeded run's cache key carries the seed the node ran under.

On a seeded run the engine seeds the global RNGs from
``derive_seed(run seed, node id)`` right before every node, so what a random
node puts out depends on the run seed AND on its own id. The cache key held
neither. Measured on two ``TensorCreate(fill="randn")`` nodes sharing one
cache, the way one canvas socket shares it:

* after a seed change both nodes kept the old seed's tensors, where a fresh
  run at the new seed draws different ones;
* the two nodes hashed to one key, so from the second run on both were
  served the same tensor.

The engine now passes the node's derived seed to ``compute_key``. An unseeded
run passes ``None`` and caches exactly as it always did.

Core nodes only; the edu pack's half is ``test_edu_pack_reproducible.py``.
"""

from __future__ import annotations

import copy

import pytest

from app.core.cache import ExecutionCache
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph

_KEY_ARGS = ("TensorCreate", {"shape": "3", "fill": "randn"}, [])


def _randn(node_id: str, shape: str = "3") -> dict:
    return {"id": node_id, "type": "TensorCreate",
            "data": {"params": {"shape": shape, "fill": "randn"}}}


def _graph(*nodes: dict) -> tuple[list[dict], list[dict]]:
    """Start triggering every node given."""
    start = {"id": "start", "type": "Start", "data": {"params": {}}}
    edges = [{"id": f"t_{node['id']}", "source": "start", "target": node["id"],
              "sourceHandle": "trigger", "targetHandle": "__trigger",
              "type": "trigger"}
             for node in nodes]
    return [start, *nodes], edges


async def _run(graph, seed, cache=None, statuses=None) -> dict[str, list[float]]:
    """One run; returns each TensorCreate's tensor as a list."""
    nodes, edges = graph

    async def track(node_id, status, data):
        if (statuses is not None and node_id != "start"
                and status in ("completed", "cached")):
            statuses[node_id] = status

    outputs = await execute_graph(
        copy.deepcopy(nodes), copy.deepcopy(edges),
        context=ExecutionContext(seed=seed), cache=cache, on_progress=track)
    return {node_id: out["tensor"].tolist()
            for node_id, out in outputs.items()
            if isinstance(out, dict) and "tensor" in out}


# ── compute_key ──────────────────────────────────────────────────────────

def test_the_seed_is_part_of_the_key():
    one = ExecutionCache.compute_key(*_KEY_ARGS, seed=1)
    assert one == ExecutionCache.compute_key(*_KEY_ARGS, seed=1)
    assert one != ExecutionCache.compute_key(*_KEY_ARGS, seed=2)


def test_no_seed_equals_leaving_the_argument_out_and_differs_from_seed_zero():
    unseeded = ExecutionCache.compute_key(*_KEY_ARGS)
    assert ExecutionCache.compute_key(*_KEY_ARGS, seed=None) == unseeded
    # 0 is a seed like any other; a seeded run must never be handed what an
    # unseeded run drew.
    assert ExecutionCache.compute_key(*_KEY_ARGS, seed=0) != unseeded


# ── through the engine ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_seed_change_reruns_the_cached_random_nodes():
    graph = _graph(_randn("a"), _randn("b"))
    cache = ExecutionCache()
    await _run(graph, 0, cache)

    after_change = await _run(graph, 1, cache)

    assert after_change == await _run(graph, 1), (
        "a run at seed 1 on a cache primed at seed 0 must give what a fresh "
        "seed-1 run gives; it was served the seed-0 tensors instead")


@pytest.mark.asyncio
async def test_two_identical_random_nodes_keep_their_own_results():
    graph = _graph(_randn("a"), _randn("b"))
    cache = ExecutionCache()
    for attempt in range(2):
        tensors = await _run(graph, 0, cache)
        assert tensors["a"] != tensors["b"], (
            f"run {attempt + 1}: two TensorCreate nodes with the same params "
            "returned one tensor. Their derived seeds differ, so their cache "
            "entries must too")


@pytest.mark.asyncio
async def test_a_seeded_rerun_is_still_served_from_cache():
    """Every assertion above would also hold if nothing were cached."""
    graph = _graph(_randn("a"), _randn("b"))
    cache = ExecutionCache()
    first: dict[str, str] = {}
    second: dict[str, str] = {}
    tensors = await _run(graph, 0, cache, first)

    assert await _run(graph, 0, cache, second) == tensors
    assert first == {"a": "completed", "b": "completed"}
    assert second == {"a": "cached", "b": "cached"}


@pytest.mark.asyncio
async def test_an_unseeded_rerun_is_still_served_from_cache():
    graph = _graph(_randn("a"))
    cache = ExecutionCache()
    statuses: dict[str, str] = {}
    tensors = await _run(graph, None, cache)

    assert await _run(graph, None, cache, statuses) == tensors
    assert statuses == {"a": "cached"}
