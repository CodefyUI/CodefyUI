"""#621 -- the captures of a node inside a block, read by the id the run gave it.

The engine flattens a block before it runs: an inner node runs, and its ports
are captured, as ``<instance>/<inner>`` -- ``<outer>/<inner instance>/<node>``
one level further in. The path routes cannot carry that id. The server decodes
``%2F`` before routing, so ``/run/blk%2Fmul/tensor`` is routed as
``/run/blk/mul/tensor`` and matches nothing: a routing 404, which the
Inspector reports as "Run data expired". ``{node_id:path}`` is no way out --
``/run/blk/norm/stats`` would be both "stats of (blk, norm)" and "the value of
(blk/norm, stats)", and Normalize has a real output called ``stats``.

So each read also has a query form, ``/{run}/value|stats|steps|grads`` with
``?node_id=&port=``, answering exactly what the path form answers. Every test
here runs a real graph with Record outputs on (or seeds the store the way the
engine does) and reads it back the way the frontend asks.
"""

from __future__ import annotations

import pytest
import torch

from app.core.graph_engine import build_preset_fallback, execute_graph
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


# -- graph building (as in test_container_port_captures.py) ---------------------


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


#: A graph-owned preset whose one inner node multiplies by 3.
TRIPLE = {
    "preset_name": "CaptureProbeTriple",
    "nodes": [
        {"id": "mul", "type": "ScalarMultiply", "params": {"scalar": 3.0}},
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

#: x5 -- the block nested inside the one below.
INNER = _block(
    "inner",
    [_node("mul2", "ScalarMultiply", scalar=5.0)],
    [],
    inputs=[("in", "mul2", "tensor")],
    outputs=[("out", "mul2", "tensor")],
)

#: x2, then the nested block (x5), then a preset card (x3), then Normalize.
OUTER = _block(
    "outer",
    [
        _node("mul", "ScalarMultiply", scalar=2.0),
        _node("nest", "subgraph:inner"),
        _node("p", "preset:CaptureProbeTriple"),
        _node("norm", "Normalize"),
    ],
    [
        _wire("mul", "tensor", "nest", "in"),
        _wire("nest", "out", "p", "x"),
        _wire("p", "x", "norm", "tensor"),
    ],
    inputs=[("in", "mul", "tensor")],
    outputs=[("out", "norm", "tensor")],
)


async def _run_blocks(run_id: str) -> None:
    """Start -> a (2.0) -> blk, and b (3.0) -> blk2: two copies of one block."""
    nodes = [
        _node("start", "Start"),
        _node("a", "TensorCreate", shape="2,2", fill="full", value=2.0),
        _node("b", "TensorCreate", shape="2,2", fill="full", value=3.0),
        _node("blk", "subgraph:outer"),
        _node("blk2", "subgraph:outer"),
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "a",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "t2", "source": "start", "target": "b",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        _wire("a", "tensor", "blk", "in"),
        _wire("b", "tensor", "blk2", "in"),
    ]
    await execute_graph(
        nodes,
        edges,
        run_id=run_id,
        output_store=app.state.run_output_store,
        record_outputs=True,
        preset_fallback=build_preset_fallback([TRIPLE]),
        subgraphs=[OUTER, INNER],
    )


def _filled(value: float) -> list[list[float]]:
    return [[value, value], [value, value]]


async def _value(client, run_id: str, node_id: str, port: str) -> dict:
    resp = await client.get(
        f"{BASE}/{run_id}/value", params={"node_id": node_id, "port": port},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The answer names what was ASKED for, which is what the client keys by.
    assert (body["node_id"], body["port"]) == (node_id, port)
    return body


# -- the value of an inner port ----------------------------------------------------


@pytest.mark.asyncio
async def test_an_inner_nodes_port_is_read_by_its_flattened_id(test_client):
    await _run_blocks("r")

    body = await _value(test_client, "r", "blk/mul", "tensor")
    assert body["type"] == "tensor"
    assert body["values"] == _filled(4.0)


@pytest.mark.asyncio
async def test_the_path_form_cannot_carry_that_id(test_client):
    """Why the query form exists: the encoded slash is decoded before routing."""
    await _run_blocks("r")

    resp = await test_client.get(f"{BASE}/r/blk%2Fmul/tensor")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


@pytest.mark.asyncio
async def test_two_copies_of_one_block_answer_for_themselves(test_client):
    await _run_blocks("r")

    assert (await _value(test_client, "r", "blk/mul", "tensor"))["values"] == _filled(4.0)
    assert (await _value(test_client, "r", "blk2/mul", "tensor"))["values"] == _filled(6.0)


@pytest.mark.asyncio
async def test_a_block_inside_a_block_is_read_two_levels_down(test_client):
    await _run_blocks("r")

    inner = await _value(test_client, "r", "blk/nest/mul2", "tensor")
    assert inner["values"] == _filled(20.0)
    # The nested instance's own output port is an alias of the inner port.
    assert (await _value(test_client, "r", "blk/nest", "out"))["values"] == _filled(20.0)


@pytest.mark.asyncio
async def test_a_preset_card_inside_a_block_reads_its_inner_port(test_client):
    await _run_blocks("r")

    assert (await _value(test_client, "r", "blk/p", "x"))["values"] == _filled(60.0)


@pytest.mark.asyncio
async def test_a_port_named_stats_is_read_as_a_value_not_summarised(test_client):
    """The ambiguity that rules out ``{node_id:path}``: Normalize's ``stats``."""
    await _run_blocks("r")

    body = await _value(test_client, "r", "blk/norm", "stats")
    # Normalize puts out a dict of the statistics it used: the VALUE route's
    # answer for it, not the stats route's summary (which carries ``kind``).
    assert body["type"] == "dict"
    assert "mean" in body["repr"]
    assert "kind" not in body


@pytest.mark.asyncio
async def test_slice_and_size_limits_apply_to_the_query_form(test_client):
    await _run_blocks("r")

    resp = await test_client.get(
        f"{BASE}/r/value",
        params={"node_id": "blk/mul", "port": "tensor", "slice": "0,:"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["values"] == [4.0, 4.0]

    bad = await test_client.get(
        f"{BASE}/r/value",
        params={"node_id": "blk/mul", "port": "tensor", "slice": "x"},
    )
    assert bad.status_code == 400

    big = await test_client.get(
        f"{BASE}/r/value",
        params={"node_id": "blk/mul", "port": "tensor", "max_elements": 1},
    )
    assert big.status_code == 413


@pytest.mark.asyncio
async def test_nothing_recorded_is_still_404_and_no_value_still_204(test_client):
    await _run_blocks("r")

    missing = await test_client.get(
        f"{BASE}/r/value", params={"node_id": "blk/ghost", "port": "tensor"},
    )
    assert missing.status_code == 404
    assert "not found" in missing.json()["detail"]

    gone = await test_client.get(
        f"{BASE}/no-such-run/value", params={"node_id": "blk/mul", "port": "tensor"},
    )
    assert gone.status_code == 404
    assert gone.json()["detail"] == "run 'no-such-run' not found"

    await app.state.run_output_store.put("r", "blk/train", "state", None)
    empty = await test_client.get(
        f"{BASE}/r/value", params={"node_id": "blk/train", "port": "state"},
    )
    assert empty.status_code == 204
    assert empty.content == b""


@pytest.mark.asyncio
async def test_the_query_form_needs_a_node_and_a_port(test_client):
    await _run_blocks("r")

    no_port = await test_client.get(f"{BASE}/r/value", params={"node_id": "blk/mul"})
    assert no_port.status_code == 422
    no_node = await test_client.get(f"{BASE}/r/stats", params={"port": "tensor"})
    assert no_node.status_code == 422


@pytest.mark.asyncio
async def test_a_top_level_id_reads_the_same_either_way(test_client):
    await _run_blocks("r")

    by_path = await test_client.get(f"{BASE}/r/a/tensor")
    assert by_path.status_code == 200, by_path.text
    assert by_path.json()["values"] == _filled(2.0)
    by_query = await _value(test_client, "r", "a", "tensor")
    assert by_query == by_path.json()


# -- stats, steps, grads --------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_inner_port_has_stats(test_client):
    await _run_blocks("r")

    resp = await test_client.get(
        f"{BASE}/r/stats", params={"node_id": "blk/mul", "port": "tensor"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["node_id"], body["port"]) == ("blk/mul", "tensor")
    assert body["kind"] == "tensor"
    assert body["mean"] == pytest.approx(4.0)

    # Cached per flattened id: the other copy of the block is its own entry.
    other = await test_client.get(
        f"{BASE}/r/stats", params={"node_id": "blk2/mul", "port": "tensor"},
    )
    assert other.json()["mean"] == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_stats_of_nothing_recorded_keep_their_answers(test_client):
    await _run_blocks("r")

    missing = await test_client.get(
        f"{BASE}/r/stats", params={"node_id": "blk/ghost", "port": "tensor"},
    )
    assert missing.status_code == 404
    assert "Record outputs" in missing.json()["detail"]

    await app.state.run_output_store.put("r", "blk/train", "state", None)
    empty = await test_client.get(
        f"{BASE}/r/stats", params={"node_id": "blk/train", "port": "state"},
    )
    assert empty.status_code == 204


@pytest.mark.asyncio
async def test_steps_and_grads_of_an_inner_node_are_lists_not_a_routing_404(
    test_client,
):
    await _run_blocks("r")

    steps = await test_client.get(f"{BASE}/r/steps", params={"node_id": "blk/mul"})
    assert steps.status_code == 200, steps.text
    assert steps.json() == []
    grads = await test_client.get(f"{BASE}/r/grads", params={"node_id": "blk/mul"})
    assert grads.status_code == 200, grads.text
    assert grads.json() == []


@pytest.mark.asyncio
async def test_steps_and_grads_recorded_for_an_inner_node_are_listed(test_client):
    """Seeded the way the engine writes them, under the flattened id."""
    store = app.state.run_output_store
    meta = {"name": "Softmax", "description": "", "scalars": {}, "tensor_keys": ["p"]}
    await store.put("r", "blk/att", "__step__0__meta", meta)
    await store.put("r", "blk/att", "out__grad", torch.ones(2))
    await store.put("r", "blk/att", "__weight_grad__weight", torch.ones(2))
    # The same canvas id one block over must not leak into the answer.
    await store.put("r", "blk2/att", "__step__1__meta", {**meta, "name": "Other"})

    steps = await test_client.get(f"{BASE}/r/steps", params={"node_id": "blk/att"})
    assert steps.status_code == 200, steps.text
    assert steps.json() == [{"index": 0, **meta}]

    grads = await test_client.get(f"{BASE}/r/grads", params={"node_id": "blk/att"})
    assert grads.status_code == 200, grads.text
    assert {(g["kind"], g["port"]) for g in grads.json()} == {
        ("port", "out"), ("weight", "weight"),
    }


@pytest.mark.asyncio
async def test_steps_and_grads_of_an_unknown_run_are_404(test_client):
    # The handler's own 404, naming the run -- not the router's "Not Found".
    steps = await test_client.get(f"{BASE}/gone/steps", params={"node_id": "blk/mul"})
    assert steps.status_code == 404
    assert steps.json()["detail"] == "run 'gone' not found"
    grads = await test_client.get(f"{BASE}/gone/grads", params={"node_id": "blk/mul"})
    assert grads.status_code == 404
    assert grads.json()["detail"] == "run 'gone' not found"
