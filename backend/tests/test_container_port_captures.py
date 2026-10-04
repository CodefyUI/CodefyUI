"""#553 -- a container's ports read the data their inner ports captured.

A preset card or a block instance never runs. Expansion replaces it with
inner nodes named ``<preset>__<inner>`` and ``<instance>/<inner>``, and the
engine captures what THOSE produce. The Inspector, and every other reader,
asks for the port the canvas draws -- the container's own -- and got a 404,
which the frontend reports as "Run data expired" with a hint to re-run that
changes nothing.

Every test runs a real graph with Record outputs on and reads the captures
back through ``/api/execution/outputs`` the way the frontend asks for them:
a container's own output row is (container, port), and the input row of a
node it feeds is the edge's (source, sourceHandle) -- see
``resolveInputSources`` in ``InspectorPanel/portCaptures.ts``.
"""

from __future__ import annotations

import builtins

import pytest

from app.api import routes_execution_outputs
from app.core.cache import ExecutionCache
from app.core.graph_engine import (
    CaptureAlias,
    build_preset_fallback,
    execute_graph,
)
from app.core.memory_budget import value_bytes
from app.core.port_stats import PortStatsCache
from app.core.run_output_store import RunOutputStore
from app.main import app

BASE = "/api/execution/outputs"


@pytest.fixture(autouse=True)
def _fresh_stores():
    """A store and a stats LRU per test (the test transport has no lifespan)."""
    app.state.run_output_store = RunOutputStore(max_runs=5)
    app.state.port_stats_cache = PortStatsCache(max_bytes=1024 * 1024)
    yield


# -- graph building -----------------------------------------------------------


def _node(node_id: str, node_type: str, **params) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": params},
    }


def _wire(source: str, source_handle: str, target: str,
          target_handle: str) -> dict:
    return {
        "id": f"{source}.{source_handle}->{target}.{target_handle}",
        "source": source,
        "sourceHandle": source_handle,
        "target": target,
        "targetHandle": target_handle,
        "type": "data",
    }


def _source(value: float = 2.0) -> tuple[list[dict], list[dict]]:
    """Start, triggering one 2x2 tensor filled with ``value`` (node ``a``)."""
    nodes = [
        _node("start", "Start"),
        _node("a", "TensorCreate", shape="2,2", fill="full", value=value),
    ]
    edges = [{
        "id": "t", "source": "start", "target": "a",
        "sourceHandle": "trigger", "targetHandle": "__trigger",
        "type": "trigger",
    }]
    return nodes, edges


def _times(scalar: float) -> dict:
    return {"scalar": scalar}


#: A graph-owned preset whose one inner node multiplies by 3. Its input and
#: its output share a name, as ``model`` does on the gallery's Training
#: Pipeline, so a read of (card, port) has to answer with the OUTPUT.
TRIPLE = {
    "preset_name": "CaptureProbeTriple",
    "nodes": [
        {"id": "mul", "type": "ScalarMultiply", "params": _times(3.0)},
    ],
    "edges": [],
    "exposed_inputs": [
        {"name": "x", "internal_node": "mul", "internal_port": "tensor",
         "data_type": "TENSOR"},
    ],
    "exposed_outputs": [
        {"name": "x", "internal_node": "mul", "internal_port": "tensor",
         "data_type": "TENSOR"},
    ],
}


def _block(sid: str, nodes: list[dict], edges: list[dict], *,
           inputs: list[tuple[str, str, str]],
           outputs: list[tuple[str, str, str]]) -> dict:
    """A subgraph definition; ports are ``(port, innerNode, innerPort)``."""
    return {
        "id": sid,
        "name": sid,
        "nodes": nodes,
        "edges": edges,
        "interface": {
            "inputs": [
                {"port": p, "innerNode": n, "innerPort": h}
                for p, n, h in inputs
            ],
            "outputs": [
                {"port": p, "innerNode": n, "innerPort": h}
                for p, n, h in outputs
            ],
            "triggerTargets": [],
        },
    }


#: One inner node multiplying by 2.
DOUBLE = _block(
    "double",
    [_node("mul", "ScalarMultiply", **_times(2.0))],
    [],
    inputs=[("in", "mul", "tensor")],
    outputs=[("out", "mul", "tensor")],
)


async def _run(nodes, edges, *, run_id, presets=(), subgraphs=(),
               record=True, cache=None) -> None:
    await execute_graph(
        nodes,
        edges,
        run_id=run_id,
        output_store=app.state.run_output_store,
        record_outputs=record,
        preset_fallback=build_preset_fallback(list(presets)),
        subgraphs=list(subgraphs),
        cache=cache,
    )


def _input_row(edges: list[dict], target: str, handle: str) -> tuple[str, str]:
    """The (node, port) the Inspector reads for ``target``'s ``handle`` input."""
    edge = next(
        e for e in edges
        if e["target"] == target and e.get("targetHandle") == handle
    )
    return edge["source"], edge["sourceHandle"]


async def _values(client, run_id: str, node_id: str, port: str):
    resp = await client.get(f"{BASE}/{run_id}/{node_id}/{port}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The answer names the port that was ASKED for, which is the one the
    # client keys its result by -- not the inner port the value came from.
    assert (body["node_id"], body["port"]) == (node_id, port)
    return body["values"]


def _filled(value: float) -> list[list[float]]:
    return [[value, value], [value, value]]


# -- presets --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_preset_cards_output_and_its_consumers_input_read_the_inner_port(
    test_client,
):
    nodes, edges = _source(2.0)
    nodes += [
        _node("p", "preset:CaptureProbeTriple"),
        _node("down", "ScalarMultiply", **_times(10.0)),
    ]
    edges += [_wire("a", "tensor", "p", "x"), _wire("p", "x", "down", "tensor")]

    await _run(nodes, edges, run_id="r-preset", presets=[TRIPLE])

    # The card's own output row. 6, not 2: the input of the same name is
    # the value that went IN, and a capture is what a port put out.
    assert await _values(test_client, "r-preset", "p", "x") == _filled(6.0)
    # The input row of the node it feeds asks for the very same pair.
    row = _input_row(edges, "down", "tensor")
    assert row == ("p", "x")
    assert await _values(test_client, "r-preset", *row) == _filled(6.0)
    # And the consumer really received it.
    assert await _values(test_client, "r-preset", "down", "tensor") == _filled(60.0)


@pytest.mark.asyncio
async def test_an_installed_presets_card_reads_its_inner_port(
    test_client, monkeypatch,
):
    """The issue's own case: a card for a preset that ships with CodefyUI."""

    # The convolutions' outputs track gradients, and float() on one makes
    # torch warn -- once per process, so a warning filter here would pass
    # whenever an earlier test had used that one warning up. Checked at the
    # call instead.
    def untracked_float(value):
        assert not getattr(value, "requires_grad", False), (
            "float() on a tensor that tracks gradients")
        return builtins.float(value)

    monkeypatch.setattr(
        routes_execution_outputs, "float", untracked_float, raising=False)
    nodes = [
        _node("start", "Start"),
        _node("img", "TensorCreate", shape="1,1,8,8", fill="ones"),
        _node("cnn", "preset:Simple CNN Classifier"),
        _node("down", "ScalarMultiply", **_times(1.0)),
    ]
    edges = [
        {"id": "t", "source": "start", "target": "img",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        _wire("img", "tensor", "cnn", "tensor"),
        _wire("cnn", "tensor", "down", "tensor"),
    ]

    await _run(nodes, edges, run_id="r-installed")

    inner = await _values(test_client, "r-installed", "cnn__pool2", "tensor")
    assert await _values(test_client, "r-installed", "cnn", "tensor") == inner
    row = _input_row(edges, "down", "tensor")
    assert await _values(test_client, "r-installed", *row) == inner
    resp = await test_client.get(f"{BASE}/r-installed/cnn/tensor")
    assert resp.json()["full_shape"] == [1, 64, 2, 2]


@pytest.mark.asyncio
async def test_a_preset_cards_output_has_stats(test_client):
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))

    await _run(nodes, edges, run_id="r-stats", presets=[TRIPLE])

    resp = await test_client.get(f"{BASE}/r-stats/p/x/stats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["node_id"], body["port"]) == ("p", "x")
    assert body["kind"] == "tensor"
    assert body["mean"] == pytest.approx(6.0)


# -- blocks ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_block_instances_output_and_its_consumers_input_read_the_inner_port(
    test_client,
):
    nodes, edges = _source(2.0)
    nodes += [
        _node("blk", "subgraph:double"),
        _node("down", "ScalarMultiply", **_times(10.0)),
    ]
    edges += [
        _wire("a", "tensor", "blk", "in"),
        _wire("blk", "out", "down", "tensor"),
    ]

    await _run(nodes, edges, run_id="r-block", subgraphs=[DOUBLE])

    assert await _values(test_client, "r-block", "blk", "out") == _filled(4.0)
    row = _input_row(edges, "down", "tensor")
    assert await _values(test_client, "r-block", *row) == _filled(4.0)


# -- containers feeding containers, and nesting ---------------------------------


@pytest.mark.asyncio
async def test_a_container_fed_by_a_container_reads_its_input(test_client):
    """A container's input row is its upstream's output -- here, a card's."""
    nodes, edges = _source(2.0)
    nodes += [
        _node("p", "preset:CaptureProbeTriple"),
        _node("blk", "subgraph:double"),
        _node("down", "ScalarMultiply", **_times(10.0)),
    ]
    edges += [
        _wire("a", "tensor", "p", "x"),
        _wire("p", "x", "blk", "in"),
        _wire("blk", "out", "down", "tensor"),
    ]

    await _run(nodes, edges, run_id="r-chain", presets=[TRIPLE],
               subgraphs=[DOUBLE])

    row = _input_row(edges, "blk", "in")
    assert row == ("p", "x")
    assert await _values(test_client, "r-chain", *row) == _filled(6.0)
    assert await _values(test_client, "r-chain", "blk", "out") == _filled(12.0)
    row = _input_row(edges, "down", "tensor")
    assert await _values(test_client, "r-chain", *row) == _filled(12.0)


@pytest.mark.asyncio
async def test_a_block_holding_a_block_and_a_card_reaches_the_innermost_node(
    test_client,
):
    """Flattening gives ``w/inner/mul`` and ``w/pp__mul``.

    The block's port names the card inside it, and the card's port names
    the node inside THAT, so the answer is two expansions away.
    """
    wrap = _block(
        "wrap",
        [_node("inner", "subgraph:double"),
         _node("pp", "preset:CaptureProbeTriple")],
        [_wire("inner", "out", "pp", "x")],
        inputs=[("in", "inner", "in")],
        outputs=[("out", "pp", "x")],
    )
    nodes, edges = _source(2.0)
    nodes.append(_node("w", "subgraph:wrap"))
    edges.append(_wire("a", "tensor", "w", "in"))

    await _run(nodes, edges, run_id="r-nest", presets=[TRIPLE],
               subgraphs=[DOUBLE, wrap])

    assert await _values(test_client, "r-nest", "w", "out") == _filled(12.0)
    # Every container on the way in is aliased too, straight to the node
    # that ran. No route can name these ids yet -- they hold a "/" -- but a
    # view of the open block, #553's other half, would ask for exactly them.
    store = app.state.run_output_store
    assert await store.get("r-nest", "w/pp", "x") == CaptureAlias(
        "w/pp__mul", "tensor")
    assert await store.get("r-nest", "w/inner", "out") == CaptureAlias(
        "w/inner/mul", "tensor")


@pytest.mark.asyncio
async def test_a_card_whose_port_is_another_cards_port_reaches_the_inner_node(
    test_client,
):
    """A preset nesting a preset: ``p`` -> ``p__q`` -> ``p__q__mul``.

    ``mid`` is exposed straight off the nested card, so it resolves through
    a container id that existed only between two expansion passes.
    """
    outer = {
        "preset_name": "CaptureProbeOuter",
        "nodes": [
            {"id": "q", "type": "preset:CaptureProbeTriple"},
            {"id": "m", "type": "ScalarMultiply", "params": _times(2.0)},
        ],
        "edges": [
            {"source": "q", "sourceHandle": "x",
             "target": "m", "targetHandle": "tensor"},
        ],
        "exposed_inputs": [
            {"name": "in", "internal_node": "q", "internal_port": "x"},
        ],
        "exposed_outputs": [
            {"name": "mid", "internal_node": "q", "internal_port": "x"},
            {"name": "out", "internal_node": "m", "internal_port": "tensor"},
        ],
    }
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeOuter"))
    edges.append(_wire("a", "tensor", "p", "in"))

    await _run(nodes, edges, run_id="r-deep", presets=[TRIPLE, outer])

    assert await _values(test_client, "r-deep", "p", "mid") == _filled(6.0)
    assert await _values(test_client, "r-deep", "p", "out") == _filled(12.0)


# -- bypass ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_block_whose_output_node_is_bypassed_reads_the_forwarded_value(
    test_client,
):
    """The exposed inner node never runs; its output is its input's source."""
    muted = _block(
        "muted",
        [_node("m1", "ScalarMultiply", **_times(2.0)),
         {**_node("m2", "ScalarMultiply", **_times(5.0)),
          "data": {"params": _times(5.0), "bypassed": True}}],
        [_wire("m1", "tensor", "m2", "tensor")],
        inputs=[("in", "m1", "tensor")],
        outputs=[("out", "m2", "tensor")],
    )
    nodes, edges = _source(2.0)
    nodes += [
        _node("mb", "subgraph:muted"),
        _node("down", "ScalarMultiply", **_times(10.0)),
    ]
    edges += [
        _wire("a", "tensor", "mb", "in"),
        _wire("mb", "out", "down", "tensor"),
    ]

    await _run(nodes, edges, run_id="r-muted", subgraphs=[muted])

    assert await _values(test_client, "r-muted", "down", "tensor") == _filled(40.0)
    assert await _values(test_client, "r-muted", "mb", "out") == _filled(4.0)


@pytest.mark.asyncio
async def test_a_bypassed_nodes_consumer_reads_what_it_received(test_client):
    nodes, edges = _source(2.0)
    nodes += [
        {**_node("b", "ScalarMultiply", **_times(5.0)),
         "data": {"params": _times(5.0), "bypassed": True}},
        _node("down", "ScalarMultiply", **_times(10.0)),
    ]
    edges += [
        _wire("a", "tensor", "b", "tensor"),
        _wire("b", "tensor", "down", "tensor"),
    ]

    await _run(nodes, edges, run_id="r-bypass")

    row = _input_row(edges, "down", "tensor")
    assert row == ("b", "tensor")
    assert await _values(test_client, "r-bypass", *row) == _filled(2.0)
    assert await _values(test_client, "r-bypass", "down", "tensor") == _filled(20.0)


# -- every way a capture is written, and every way it is listed -----------------


@pytest.mark.asyncio
async def test_a_cache_hit_inside_a_card_still_answers_for_the_card(test_client):
    """The second run serves the inner node from the cache, not by running it."""
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))
    cache = ExecutionCache()

    await _run(nodes, edges, run_id="r-prime", presets=[TRIPLE], record=False,
               cache=cache)
    await _run(nodes, edges, run_id="r-hit", presets=[TRIPLE], cache=cache)

    assert await _values(test_client, "r-hit", "p", "x") == _filled(6.0)


@pytest.mark.asyncio
async def test_a_cards_port_costs_no_bytes_of_its_own():
    """The store counts every slot against its byte budget.

    A copy of the value under the card's port would count the tensor twice
    and evict other runs early, so the run's bytes must be exactly what the
    nodes that ran produced.
    """
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))

    await _run(nodes, edges, run_id="r-bytes", presets=[TRIPLE])

    store = app.state.run_output_store
    ran = [("a", "tensor"), ("p__mul", "tensor")]
    assert sorted(await store.list_ports("r-bytes")) == sorted(ran + [("p", "x")])
    produced = 0
    for node_id, port in ran:
        produced += value_bytes(await store.get("r-bytes", node_id, port))
    assert produced > 0
    assert (await store.stats())["bytes"] == produced
    assert store._slot_bytes[("r-bytes", "p", "x")] == 0
    assert await store.get("r-bytes", "p", "x") == CaptureAlias(
        "p__mul", "tensor")


@pytest.mark.asyncio
async def test_the_run_listing_names_a_cards_port(test_client):
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))

    await _run(nodes, edges, run_id="r-list", presets=[TRIPLE])

    resp = await test_client.get(f"{BASE}/r-list")
    assert resp.status_code == 200
    listed = {(item["node_id"], item["port"]): item for item in resp.json()}
    assert listed[("p", "x")]["type"] == "tensor"
    assert listed[("p", "x")]["full_shape"] == [2, 2]
    assert listed[("p__mul", "tensor")]["full_shape"] == [2, 2]


# -- what is really not there still says so -------------------------------------


@pytest.mark.asyncio
async def test_a_card_in_a_run_that_recorded_nothing_is_404(test_client):
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))

    await _run(nodes, edges, run_id="r-off", presets=[TRIPLE], record=False)

    assert (await test_client.get(f"{BASE}/r-off/p/x")).status_code == 404
    assert (await test_client.get(f"{BASE}/r-off/p/x/stats")).status_code == 404


@pytest.mark.asyncio
async def test_a_port_the_card_does_not_expose_is_404(test_client):
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))

    await _run(nodes, edges, run_id="r-unknown", presets=[TRIPLE])

    assert (await test_client.get(f"{BASE}/r-unknown/p/nope")).status_code == 404
    assert (await test_client.get(f"{BASE}/r-unknown/ghost/x")).status_code == 404


@pytest.mark.asyncio
async def test_a_deleted_run_is_gone_for_its_cards_too(test_client):
    nodes, edges = _source(2.0)
    nodes.append(_node("p", "preset:CaptureProbeTriple"))
    edges.append(_wire("a", "tensor", "p", "x"))
    await _run(nodes, edges, run_id="r-gone", presets=[TRIPLE])
    assert (await test_client.get(f"{BASE}/r-gone/p/x")).status_code == 200

    assert (await test_client.delete(f"{BASE}/r-gone")).status_code == 200

    assert (await test_client.get(f"{BASE}/r-gone/p/x")).status_code == 404
    assert (await test_client.get(f"{BASE}/r-gone/p/x/stats")).status_code == 404
