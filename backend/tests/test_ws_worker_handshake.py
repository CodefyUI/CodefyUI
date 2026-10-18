"""Handshake validation on ``/ws/worker`` (#781).

Version (identical or refused, naming both), node catalog (a difference is
``degraded``; ``any`` skips it, a named run must use only types it has),
fingerprint (recorded on the worker and on each run it executes), the
remembered rejection per name, and the capability lists. A fake agent
speaks the protocol by hand, as in ``test_ws_worker.py``; the real agent
is ``test_worker_e2e.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from httpx import AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.core import worker_handshake as handshake
from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, TOKEN_QUERY_PARAM, session_token
from app.core.db import Database
from app.core.env_fingerprint import fingerprint_digest
from app.core.node_base import BaseNode
from app.core.node_registry import NodeRegistry, node_catalog
from app.core.resources import DeviceFact
from app.core.run_service import RunService
from app.core.run_store import RunStore
from app.core.version import get_version
from app.core.workers import WorkerRegistry
from app.main import app

BASE = "http://127.0.0.1:8000"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
PLUGIN_TYPE = "foundations:Edu-KNN"
DOCUMENT = {"schema_version": 1, "codefyui": {"version": "x"}}


def _path() -> str:
    return f"{wp.WORKER_WS_PATH}?{TOKEN_QUERY_PARAM}={session_token()}"


def _graph(*types: str) -> dict[str, Any]:
    """Start, then each type in a trigger chain (so none is pruned)."""
    nodes = [{"id": "start", "type": "Start", "data": {"params": {}}}]
    edges = []
    for i, node_type in enumerate(types):
        nodes.append({"id": f"n{i}", "type": node_type,
                      "data": {"params": {}}})
        edges.append({"id": f"t{i}", "source": "start", "target": f"n{i}",
                      "sourceHandle": "trigger", "type": "trigger"})
    return {"nodes": nodes, "edges": edges}


def _plain() -> dict[str, Any]:
    return {"nodes": [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "x", "type": "TensorCreate",
         "data": {"params": {"shape": "2", "fill": "ones"}}}],
        "edges": [{"id": "t", "source": "start", "target": "x",
                   "sourceHandle": "trigger", "type": "trigger"}]}


def _catalog(*, without: tuple[str, ...] = (),
             extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """``hello`` catalog fields: this server's, or a variation of it."""
    catalog = node_catalog()
    if not without and not extra:
        return {"node_catalog_hash": catalog.hash,
                "node_types": list(catalog.node_types)}
    types = sorted((set(catalog.node_types) - set(without)) | set(extra))
    return {"node_catalog_hash": "f" * 64, "node_types": types}


@pytest.fixture
async def env(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    registry = WorkerRegistry(heartbeat_s=10, node_catalog=node_catalog,
                              local_devices=lambda: [
                                  DeviceFact("cpu", "cpu", None, "", None,
                                             None, "none")])
    service = RunService(RunStore(database), workers=registry,
                         shutdown_grace_s=5.0)
    app.state.db = database
    app.state.run_service = service
    app.state.worker_registry = registry
    try:
        yield service, registry
    finally:
        await service.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "run_service", "worker_registry"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE,
                       headers={TOKEN_HEADER: session_token()})


class FakeAgent:
    def __init__(self, ws: Any) -> None:
        self.ws = ws

    async def send(self, frame: Any) -> None:
        await self.ws.send_text(wp.encode_frame(frame))

    async def recv(self, timeout: float = 10.0) -> Any:
        raw = await asyncio.wait_for(self.ws.receive_text(), timeout)
        return wp.decode_control_frame(raw)

    async def hello(self, name: str = "box", **kw: Any) -> Any:
        kw.setdefault("version", get_version())
        await self.send(wp.Hello(name=name, devices=[
            wp.DeviceReport(value="cpu", kind="cpu")], **kw))
        return await self.recv()


async def _terminal(service: RunService, run_id: str) -> Any:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = await service.store.get_run(run_id)
        if record is not None and record.status in TERMINAL \
                and run_id not in service.active_run_ids():
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


async def _submit(client: AsyncClient, graph: dict[str, Any],
                  worker: str) -> Any:
    return await client.post("/api/runs", json={
        "graph": graph, "options": {"target": {"worker": worker}}})


# ── version ──────────────────────────────────────────────────────────────


async def test_a_hello_this_build_cannot_parse_still_hears_the_versions(env):
    """Another release may change ``hello``'s shape: the version pair is
    read off the raw JSON before validation, and the refusal recorded."""
    _, registry = env
    raw = ('{"type": "hello", "name": "new-box", "version": "9.9.9", '
           '"protocol_version": 9, "devices": "not a list"}')
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            await ws.send_text(raw)
            reply = await FakeAgent(ws).recv()
        assert isinstance(reply, wp.Reject)
        assert reply.reason.startswith(
            f"version mismatch: worker 9.9.9, control plane {get_version()}")
        [rejection] = (await client.get("/api/workers")).json()["rejections"]
        assert rejection["name"] == "new-box"
        assert rejection["peer"] == "127.0.0.1"
    # Same version, another protocol: named too.
    assert handshake.raw_refusal(
        '{"type": "hello", "name": "b", "version": "1", '
        '"protocol_version": 1}', control_version="1") == (
        "b", handshake.protocol_mismatch(1))
    assert handshake.raw_refusal("[]", control_version="1") is None
    assert handshake.raw_refusal("not json", control_version="1") is None


async def test_a_refused_attempt_on_a_live_workers_name_is_not_its_own(env):
    """An enrolled agent holds the name; a session-token agent trying to
    take it is refused, and that refusal is marked ``live_owner``."""
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(**_catalog())
        registry.get("box").enrolled = True
        async with aconnect_ws(_path(), client) as other:
            reply = await FakeAgent(other).hello(**_catalog())
        assert isinstance(reply, wp.Reject)
        [rejection] = (await client.get("/api/workers")).json()["rejections"]
        assert rejection["live_owner"] is True
        assert registry.get("box").online


async def test_another_version_is_rejected_naming_both(env):
    _, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            reply = await FakeAgent(ws).hello(version="0.0.1-other",
                                              **_catalog())
        assert isinstance(reply, wp.Reject)
        assert reply.reason.startswith(
            f"version mismatch: worker 0.0.1-other, control plane "
            f"{get_version()}")
        assert registry.workers() == []
        body = (await client.get("/api/workers")).json()
        assert body["workers"] == []
        [rejection] = body["rejections"]
        assert rejection["name"] == "box"
        assert rejection["reason"] == reply.reason
        assert isinstance(rejection["at"], float)

        # Fixed and reconnected: the remembered refusal is gone.
        async with aconnect_ws(_path(), client) as ws:
            assert isinstance(await FakeAgent(ws).hello(**_catalog()),
                              wp.Welcome)
            body = (await client.get("/api/workers")).json()
            assert body["rejections"] == []


def test_refusal_rules_in_order():
    version = get_version()
    assert handshake.refusal(wp.Heartbeat(), control_version=version) \
        == "expected hello, got heartbeat"
    stale = wp.Hello(name="box", version="1", protocol_version=99)
    assert "protocol version 99" in handshake.refusal(
        stale, control_version=version)
    assert handshake.refusal(wp.Hello(name="box", version="1"),
                             control_version="2") == \
        handshake.version_mismatch("1", "2")
    assert "reserved" in handshake.refusal(
        wp.Hello(name="any", version=version), control_version=version)
    assert "lowercase" in handshake.refusal(
        wp.Hello(name="Box", version=version), control_version=version)
    assert handshake.refusal(wp.Hello(name="box", version=version),
                             control_version=version) is None


def test_rejections_are_bounded_and_newest_first():
    registry = WorkerRegistry()
    for i in range(105):
        registry.record_rejection(f"w{i}", "no")
    registry.record_rejection("w50", "again")
    names = [r["name"] for r in registry.rejections()]
    assert len(names) == 100 and names[0] == "w50" and "w0" not in names


# ── welcome capabilities and additive fields ─────────────────────────────


async def test_welcome_lists_this_servers_capabilities(env):
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        welcome = await FakeAgent(ws).hello(capabilities=["ack", "future"],
                                            **_catalog())
        assert welcome.capabilities == list(wp.CAPABILITIES) == [
            "ack", "mounts", "transfer", "memory_budget", "one_shot",
            "s3", "record_outputs"]
        assert registry.get("box").online
        # The registry keeps what the agent listed (#785 reads it).
        assert registry.get("box").capabilities == ("ack", "future")


async def test_drops_missed_while_away_are_sent_on_connecting(env):
    """Recorded outputs this server stopped retaining while the worker
    was away (#789) are dropped there once it connects."""
    _, registry = env
    registry.outputs.drop("r-old", "box")
    assert registry.outputs.pending_drops("box") == ["r-old"]
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(capabilities=list(wp.CAPABILITIES), **_catalog())
        frame = await agent.recv()
        assert frame == wp.DropOutputs(run_id="r-old")
        assert registry.outputs.pending_drops("box") == []


def test_unknown_fields_are_ignored_not_refused():
    """The additive rule: a newer peer's extra field parses here."""
    frame = wp.decode_agent_frame(
        '{"type": "heartbeat", "running": [], "spool_bytes": 12}')
    assert isinstance(frame, wp.Heartbeat)
    welcome = wp.decode_control_frame(
        '{"type": "welcome", "worker_id": "w", "storage": {"x": 1}}')
    assert welcome.worker_id == "w" and welcome.capabilities == []
    assert wp.shared_capabilities(["b", "a"], ["a", "b", "c"]) == ["a", "b"]


# ── node catalog ─────────────────────────────────────────────────────────


async def test_a_matching_catalog_is_online(env):
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(**_catalog())
        [worker] = (await client.get("/api/workers")).json()["workers"]
        assert worker["status"] == "online"
        assert worker["degraded_reasons"] == []
        assert worker["node_catalog"] == {
            "status": "match", "hash": node_catalog().hash,
            "missing": [], "extra": []}


async def test_a_different_catalog_is_degraded_with_the_difference(env):
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(**_catalog(without=("Print",),
                                             extra=("mine:Only",)))
        [worker] = (await client.get("/api/workers")).json()["workers"]
        assert worker["status"] == "degraded"
        assert worker["degraded_reasons"] == ["node_catalog"]
        assert worker["node_catalog"]["status"] == "mismatch"
        assert worker["node_catalog"]["missing"] == ["Print"]
        assert worker["node_catalog"]["extra"] == ["mine:Only"]
        # Draining still reads draining: it decides placement first.
        await client.post("/api/workers/box/drain")
        [worker] = (await client.get("/api/workers")).json()["workers"]
        assert worker["status"] == "draining"


async def test_no_catalog_reported_is_unknown_and_not_degraded(env, caplog):
    _, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello()
        worker = registry.get("box")
        assert worker.catalog.status == handshake.CATALOG_UNKNOWN
        assert worker.api_status() == "online"
        assert "reported no node catalog" in caplog.text


async def test_a_named_degraded_worker_takes_only_graphs_it_can_run(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(**_catalog(without=(PLUGIN_TYPE,)))
        refused = await _submit(client, _graph(PLUGIN_TYPE), "box")
        assert refused.status_code == 400
        assert PLUGIN_TYPE in refused.json()["detail"]
        assert "box" in refused.json()["detail"]

        accepted = await _submit(client, _plain(), "box")
        assert accepted.status_code == 200, accepted.text
        assign = await agent.recv()
        assert isinstance(assign, wp.Assign)
        await agent.send(wp.RunFinished(run_id=assign.run_id, seq=1,
                                        status="succeeded"))
        record = await _terminal(service, assign.run_id)
        assert record.status == "succeeded"


async def test_any_keeps_a_worker_that_only_has_more_types(env):
    """A superset worker stays eligible for ``any``; its run is still
    checked, and it has every type of this graph."""
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(**_catalog(extra=("mine:Only",)))
        assert not registry.get("box").skipped_by_any
        response = await client.post("/api/runs", json={
            "graph": _plain(), "options": {"target": {"worker": "any"}}})
        run_id = response.json()["run_id"]
        # Workers come before this server for ``any``.
        assign = await agent.recv()
        await agent.send(wp.RunFinished(run_id=assign.run_id, seq=1,
                                        status="succeeded"))
        assert (await _terminal(service, run_id)).status == "succeeded"


async def test_any_skips_a_worker_without_a_catalog(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello()
        assert registry.get("box").skipped_by_any
        response = await client.post("/api/runs", json={
            "graph": _plain(), "options": {"target": {"worker": "any"}}})
        record = await _terminal(service, response.json()["run_id"])
        assert record.worker_id == "local"


async def test_a_map_over_an_installed_preset_needs_its_types(env):
    """The types a Map runs come from the installed preset it names."""
    from app.core.preset_registry import preset_registry
    from app.schemas.models import PresetDefinition

    preset_registry.add(PresetDefinition(
        preset_name="e781-knn", nodes=[{"id": "k", "type": PLUGIN_TYPE}],
        edges=[], exposed_inputs=[{"name": "x", "internal_node": "k",
                                   "internal_port": "x"}],
        exposed_outputs=[{"name": "y", "internal_node": "k",
                          "internal_port": "y"}]))
    graph = {"nodes": [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "map", "type": "Map",
         "data": {"params": {"subgraph": "e781-knn"}}}],
        "edges": [{"id": "t", "source": "start", "target": "map",
                   "sourceHandle": "trigger", "type": "trigger"}]}
    try:
        assert PLUGIN_TYPE in handshake.graph_node_types(graph)
        _, registry = env
        async with _client() as client, aconnect_ws(_path(), client) as ws:
            await FakeAgent(ws).hello(**_catalog(without=(PLUGIN_TYPE,)))
            refused = await _submit(client, graph, "box")
            assert refused.status_code == 400
            assert PLUGIN_TYPE in refused.json()["detail"]
    finally:
        preset_registry._presets.pop("e781-knn", None)


def test_preset_types_resolve_nested_presets_and_maps(monkeypatch):
    from app.core import preset_registry as registry_module
    from app.schemas.models import PresetDefinition

    presets = {
        "outer": PresetDefinition(preset_name="outer", edges=[], nodes=[
            {"id": "a", "type": "preset:inner"},
            {"id": "m", "type": "Map", "params": {"subgraph": "leaf"}}]),
        "inner": PresetDefinition(preset_name="inner", edges=[], nodes=[
            {"id": "b", "type": "Print"},
            {"id": "c", "type": "preset:outer"}]),
        "leaf": PresetDefinition(preset_name="leaf", edges=[], nodes=[
            {"id": "d", "type": "x:Leaf"}]),
    }
    monkeypatch.setattr(registry_module.preset_registry, "_presets", presets)
    assert handshake.preset_node_types("outer") == {"Print", "Map", "x:Leaf"}
    assert handshake.preset_node_types("nobody") == set()
    # The graph's own definition of a name wins over the installed one.
    own = {"presets": [{"preset_name": "leaf", "nodes": [
        {"id": "e", "type": "y:Own"}], "edges": []}]}
    assert handshake.preset_node_types("leaf", own) == {"y:Own"}


async def test_any_skips_a_degraded_worker(env):
    service, registry = env
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        await FakeAgent(ws).hello(**_catalog(without=("Print",)))
        response = await client.post("/api/runs", json={
            "graph": _plain(), "options": {"target": {"worker": "any"}}})
        record = await _terminal(service, response.json()["run_id"])
        # Run here, not on the degraded worker.
        assert record.worker_id == "local"


async def test_a_worker_that_lost_a_type_while_a_run_waited_fails_it(env):
    """The pre-assignment check: submitted while the worker had every type,
    started after it reconnected without one. Nothing is assigned."""
    service, registry = env
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            await FakeAgent(ws).hello(**_catalog())
            await client.post("/api/workers/box/drain")
            response = await _submit(client, _graph(PLUGIN_TYPE), "box")
            assert response.status_code == 200, response.text
            run_id = response.json()["run_id"]
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello(**_catalog(without=(PLUGIN_TYPE,)))
            await client.post("/api/workers/box/undrain")
            record = await _terminal(service, run_id)
            assert record.status == "failed"
            assert PLUGIN_TYPE in record.error
            with pytest.raises(asyncio.TimeoutError):
                await agent.recv(timeout=0.3)


async def test_the_comparison_follows_this_servers_catalog():
    """A plugin installed here later makes a matching worker degraded on
    the next lookup, and removing it again matches."""
    class _Extra(BaseNode):
        NODE_NAME = "Extra"
        CATEGORY = "Test"

        @classmethod
        def define_inputs(cls):
            return []

        @classmethod
        def define_outputs(cls):
            return []

        @classmethod
        def define_params(cls):
            return []

        def execute(self, inputs, params):
            return {}

    nodes = NodeRegistry()
    registry = WorkerRegistry(node_catalog=lambda: node_catalog(nodes))
    first = node_catalog(nodes)
    worker, _ = registry.connect(
        wp.Hello(name="box", version="t", node_catalog_hash=first.hash,
                 node_types=list(first.node_types)),
        send=_noop, close=_noop)
    assert worker.catalog.status == handshake.CATALOG_MATCH
    nodes.register(_Extra, plugin_id="new")
    assert registry.get("box").catalog.degraded
    assert worker.catalog.missing == ("new:Extra",)
    nodes.clear()
    assert registry.workers()[0].catalog.status == handshake.CATALOG_MATCH
    await registry.shutdown()


async def _noop(*_: Any) -> None:
    return None


# ── catalog hash and graph types ─────────────────────────────────────────


def test_graph_node_types_expand_and_fall_back():
    assert handshake.graph_node_types(_plain()) == {"Start", "TensorCreate"}
    # A type this server cannot validate: read conservatively.
    odd = _graph("nobody:Unknown")
    odd["subgraphs"] = [{"id": "s", "nodes": [{"type": "Inner"}]}]
    assert handshake.graph_node_types(odd) == {"Start", "nobody:Unknown",
                                               "Inner"}
    # A bare type matches the worker's namespaced one.
    assert handshake.missing_node_types({"Start", "Edu-KNN"},
                                        ["Start", PLUGIN_TYPE]) == []


def test_check_catalog_lists_the_difference():
    catalog = node_catalog()
    assert handshake.check_catalog(catalog, None, []).status == "unknown"
    match = handshake.check_catalog(catalog, catalog.hash, catalog.node_types)
    assert (match.status, match.missing, match.extra) == ("match", (), ())
    other = handshake.check_catalog(
        catalog, "x", [t for t in catalog.node_types if t != "Start"] + ["z"])
    assert (other.status, other.missing, other.extra) == (
        "mismatch", ("Start",), ("z",))


# ── fingerprint ──────────────────────────────────────────────────────────


async def test_the_workers_fingerprint_is_recorded_and_runs_carry_it(env):
    service, registry = env
    digest = fingerprint_digest(DOCUMENT)
    async with _client() as client, aconnect_ws(_path(), client) as ws:
        agent = FakeAgent(ws)
        await agent.hello(fingerprint=wp.FingerprintReport(
            document=DOCUMENT, digest="claimed-wrong"), **_catalog())
        [worker] = (await client.get("/api/workers")).json()["workers"]
        # The digest is this side's, computed from the document.
        assert worker["fingerprint_digest"] == digest
        resolved = await client.get(f"/api/system/fingerprints/{digest}")
        assert resolved.status_code == 200
        assert resolved.json()["fingerprint"] == DOCUMENT

        response = await _submit(client, _plain(), "box")
        assign = await agent.recv()
        await agent.send(wp.RunFinished(run_id=assign.run_id, seq=1,
                                        status="succeeded"))
        record = await _terminal(service, response.json()["run_id"])
        assert record.fingerprint_digest == digest
        assert record.execution_facts["worker_id"] == "box"
        assert await service.store.get_fingerprint(digest) == DOCUMENT


def test_an_uncanonical_fingerprint_is_dropped(caplog):
    report = wp.FingerprintReport(document={"x": float("nan")}, digest="d")
    assert handshake.verified_fingerprint(report, worker="box") == (None,
                                                                    None)
    assert "cannot be canonicalised" in caplog.text
    assert handshake.verified_fingerprint(None) == (None, None)
