"""The environment fingerprint and what a run records of it (#344).

Four layers, in order:

1. ``core.env_fingerprint``: the document is complete, canonical, stable
   within a process, sensitive to every recorded fact, and free of anything
   that identifies the host or the person.
2. ``RunStore``: MIGRATION_006 on an old database, the content-addressed
   table, ``mark_running`` writing the digest and the document together.
3. ``RunService``: a run records the environment that STARTS it -- a queued
   run that waited through an environment change records the new one --
   while its submission provenance stays as it was.
4. The HTTP surface: ``GET /api/system/fingerprint``, the run detail's
   ``fingerprint``, and resolving a digest after the fact.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import env_fingerprint as fpmod
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import (
    STATUS_SUCCEEDED,
    RunProvenance,
    RunStore,
)
from app.main import app

TOP_LEVEL_KEYS = {
    "schema_version", "codefyui", "python", "platform", "libraries",
    "torch_build", "accelerators", "devices", "plugins", "custom_nodes",
    "optional_packages", "env",
}


# ── 1. the document ───────────────────────────────────────────────────────


def test_every_documented_key_is_present_with_the_schema_version():
    fingerprint = fpmod.collect_fingerprint()
    assert set(fingerprint) == TOP_LEVEL_KEYS
    assert fingerprint["schema_version"] == fpmod.SCHEMA_VERSION == 2
    assert set(fingerprint["codefyui"]) == {
        "version", "commit", "dirty", "commit_source"}
    assert set(fingerprint["libraries"]) == {
        "torch", "torchvision", "numpy", "mlx"}
    assert set(fingerprint["env"]) == set(fpmod.NUMERICS_ENV_VARS)
    # An unknown fact is an explicit null, never a missing key.
    for name in fpmod.NUMERICS_ENV_VARS:
        assert name in fingerprint["env"]


def test_the_digest_is_stable_across_two_calls_in_one_process():
    first, digest_a = fpmod.collect_with_digest()
    second, digest_b = fpmod.collect_with_digest()
    assert first == second
    assert digest_a == digest_b
    assert len(digest_a) == 64 and int(digest_a, 16) >= 0


def test_the_canonical_form_ignores_key_order_and_is_compact_utf8():
    a = {"b": 1, "a": {"y": "中", "x": None}}
    b = {"a": {"x": None, "y": "中"}, "b": 1}
    assert fpmod.canonical_json(a) == fpmod.canonical_json(b)
    assert fpmod.canonical_json(a) == '{"a":{"x":null,"y":"中"},"b":1}'.encode()
    assert fpmod.fingerprint_digest(a) == fpmod.fingerprint_digest(b)


def test_a_different_torch_version_changes_the_digest(monkeypatch):
    before = fpmod.fingerprint_digest(fpmod.collect_fingerprint())
    real = fpmod._dist_version
    monkeypatch.setattr(
        fpmod, "_dist_version",
        lambda name: "9.9.9" if name == "torch" else real(name))
    changed = fpmod.collect_fingerprint()
    assert changed["libraries"]["torch"] == "9.9.9"
    assert fpmod.fingerprint_digest(changed) != before


def _lockfile(sha: str) -> dict[str, Any]:
    return {"schema": 1, "plugins": {
        "remote-pack": {"source_kind": "github_url", "source": "o/remote",
                        "ref": "v1", "sha": sha, "enabled": True,
                        "manifest": {"version": "1.0.0"}},
        "deep": {"source_kind": "builtin", "source": "deep", "enabled": True,
                 "manifest": {"version": "0.3.0"}},
        "linked": {"source_kind": "local",
                   "source": str(Path.home() / "src" / "linked"),
                   "path": str(Path.home() / "src" / "linked"),
                   "enabled": True, "manifest": {"version": "0.0.1"}},
        "off": {"source_kind": "github_url", "source": "o/off", "sha": "f" * 40,
                "enabled": False},
    }}


def test_a_different_plugin_pin_changes_the_digest(monkeypatch):
    from app.core import plugin_loader

    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        lambda: _lockfile("a" * 40))
    pinned_a = fpmod.collect_fingerprint()
    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        lambda: _lockfile("b" * 40))
    pinned_b = fpmod.collect_fingerprint()
    assert pinned_a["plugins"]["remote-pack"]["revision"] == "a" * 40
    assert fpmod.fingerprint_digest(pinned_a) \
        != fpmod.fingerprint_digest(pinned_b)


def test_plugin_identities_are_honest_about_what_has_a_revision(monkeypatch):
    from app.core import plugin_loader

    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        lambda: _lockfile("a" * 40))
    fingerprint = fpmod.collect_fingerprint()
    plugins = fingerprint["plugins"]
    assert "off" not in plugins                      # disabled: not loaded
    assert plugins["remote-pack"] == {
        "source_kind": "github_url", "source": "o/remote", "ref": "v1",
        "version": "1.0.0", "revision": "a" * 40, "revision_kind": "git-sha",
        # Not installed in this test, so its files cannot be read.
        "immutable": True, "content_sha256": None}
    # A builtin pack ships inside the checkout: its revision IS the
    # checkout's commit, and its source a repo-relative path.
    deep = plugins["deep"]
    assert deep["source"] == "plugins/deep"
    assert deep["revision"] == fingerprint["codefyui"]["commit"]
    assert deep["revision_kind"] == (
        "codefyui-commit" if deep["revision"] else None)
    assert deep["immutable"] is (bool(deep["revision"])
                                 and fingerprint["codefyui"]["dirty"] is False)
    # A linked pack has no immutable revision, and its path is withheld.
    # Its directory does not exist here, so nothing identifies its code.
    assert plugins["linked"]["source"] is None
    assert plugins["linked"]["revision"] is None
    assert plugins["linked"]["revision_kind"] is None
    assert plugins["linked"]["immutable"] is False
    assert plugins["linked"]["content_sha256"] is None
    assert str(Path.home()) not in json.dumps(fingerprint)


def test_nothing_identifies_the_host_or_the_person(monkeypatch):
    from app.core import plugin_loader

    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        lambda: _lockfile("a" * 40))
    monkeypatch.setenv("CODEFYUI_API_KEY_FOR_TEST", "sk-not-in-the-fingerprint")
    text = json.dumps(fpmod.collect_fingerprint())
    assert str(Path.home()) not in text
    assert "sk-not-in-the-fingerprint" not in text
    hostname = socket.gethostname()
    if len(hostname) >= 4:
        assert hostname not in text
    user = getpass.getuser()
    if len(user) >= 4:
        assert user not in text


def test_the_code_identity_falls_back_to_build_info_then_to_null(tmp_path):
    # No .git: never ask git (an install inside an unrelated repository must
    # not report that repository's commit).
    assert fpmod._code_identity(tmp_path) == {
        "version": fpmod.get_version(), "commit": None, "dirty": None,
        "commit_source": None}
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "build-info.json").write_text(
        json.dumps({"tag": "v9", "commit": "c0ffee", "built_at": "x",
                    "source": "release"}), encoding="utf-8")
    identity = fpmod._code_identity(tmp_path)
    assert identity["commit"] == "c0ffee"
    assert identity["commit_source"] == "build-info"
    assert identity["dirty"] is None
    (dist / "build-info.json").write_text("not json", encoding="utf-8")
    assert fpmod._code_identity(tmp_path)["commit"] is None


def test_a_custom_node_edit_changes_the_digest(monkeypatch, tmp_path):
    from app.config import settings

    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    node = tmp_path / "my_node.py"
    node.write_text("A = 1\n", encoding="utf-8")
    monkeypatch.setattr(settings, "CUSTOM_NODES_DIR", tmp_path)
    before = fpmod.collect_fingerprint()
    assert list(before["custom_nodes"]) == ["my_node.py"]
    node.write_text("A = 2\n", encoding="utf-8")
    assert fpmod.fingerprint_digest(fpmod.collect_fingerprint()) \
        != fpmod.fingerprint_digest(before)


def test_execution_facts_keep_the_request_next_to_the_resolution():
    graph = {"nodes": [], "settings": {"device": "auto", "seed": 7}}
    facts = fpmod.execution_facts(
        options={"device": "auto", "seed": 7, "deterministic": True},
        graph=graph, resolved_device="cpu")
    assert facts["schema_version"] == fpmod.EXECUTION_FACTS_VERSION
    assert facts["requested"] == {
        "device": "auto", "seed": 7, "deterministic": True,
        "graph_settings_device": "auto", "graph_settings_seed": 7}
    assert facts["resolved"]["device"] == "cpu"
    assert facts["resolved"]["seed"] == 7
    assert facts["resolved"]["deterministic"] is True
    assert set(facts["resolved"]["numerics"]) == {
        "default_dtype", "float32_matmul_precision", "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32", "cudnn_benchmark", "num_threads"}
    # A graph that is not a dict reads as "no settings", not as a crash.
    assert fpmod.execution_facts(options={}, graph=None, resolved_device=None)[
        "requested"]["graph_settings_seed"] is None


# ── 2. the store ──────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


async def _new_run(store: RunStore):
    return await store.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={"device": "cpu"},
        provenance=RunProvenance())


async def test_mark_running_stores_the_document_and_its_digest(store, db):
    doc = {"schema_version": 1, "marker": "A"}
    first = await _new_run(store)
    second = await _new_run(store)
    facts = {"schema_version": 1, "resolved": {"device": "cpu"}}
    assert await store.mark_running(first.id, queue_key="cpu",
                                    fingerprint=doc, execution_facts=facts)
    assert await store.mark_running(second.id, queue_key="cpu",
                                    fingerprint=dict(doc))

    a, b = await store.get_run(first.id), await store.get_run(second.id)
    assert a.fingerprint_digest == fpmod.fingerprint_digest(doc)
    assert b.fingerprint_digest == a.fingerprint_digest
    assert a.execution_facts == facts
    assert b.execution_facts is None
    assert await store.get_fingerprint(a.fingerprint_digest) == doc
    # Content-addressed: one environment, one row.
    assert db._conn.execute(
        "SELECT COUNT(*) FROM exec_run_fingerprints").fetchone()[0] == 1
    assert await store.get_fingerprint("0" * 64) is None


async def test_a_run_never_started_has_null_fingerprint_fields(store):
    record = await _new_run(store)
    assert record.fingerprint_digest is None
    assert record.execution_facts is None
    assert (await store.get_run(record.id)).fingerprint_digest is None


async def test_the_fingerprint_outlives_its_runs(store):
    doc = {"schema_version": 1, "marker": "gone"}
    record = await _new_run(store)
    await store.mark_running(record.id, fingerprint=doc)
    await store.mark_finished(record.id, STATUS_SUCCEEDED)
    digest = (await store.get_run(record.id)).fingerprint_digest
    assert await store.delete_run(record.id)
    assert await store.prune(keep_last=0) == 0
    assert await store.get_fingerprint(digest) == doc


def test_migration_006_upgrades_a_v5_database(tmp_path, monkeypatch):
    """Old rows survive the ALTERs and read back with both new fields NULL."""
    from app.core import db as dbmod
    from app.core.migrations import MIGRATIONS

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS[:5]))
    old = Database(tmp_path / "v5.db")
    old.connect()
    assert old._conn.execute("PRAGMA user_version").fetchone()[0] == 5
    old._conn.execute(
        "INSERT INTO exec_runs (id, name, graph_snapshot, options, status, "
        "created_at, git_commit, plugin_pins) VALUES ('legacy', 'old', '{}', "
        "'{\"seed\": 3}', 'succeeded', '2026-01-01T00:00:00.000000Z', "
        "'abc', '{\"p\": {\"sha\": \"1\"}}')")
    old.close()

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS))
    new = Database(tmp_path / "v5.db")
    new.connect()
    try:
        # Every later migration runs too (007, #343, at the time of
        # writing), so this pins the full count rather than 6.
        assert new._conn.execute("PRAGMA user_version").fetchone()[0] \
            == len(MIGRATIONS)
        assert len(MIGRATIONS) >= 6
        record = asyncio.run(RunStore(new).get_run("legacy"))
        assert record.fingerprint_digest is None
        assert record.execution_facts is None
        # The submission provenance a v5 row carried is untouched.
        assert record.git_commit == "abc"
        assert record.plugin_pins == {"p": {"sha": "1"}}
        assert record.options == {"seed": 3}
    finally:
        new.close()


# ── 3. the run service ────────────────────────────────────────────────────


_GATE = threading.Event()


class _FpGateNode(BaseNode):
    """Holds the run on its worker thread until the test opens the gate."""

    NODE_NAME = "_FpGate"
    CATEGORY = "Test"
    DESCRIPTION = "Waits for the test"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        _GATE.wait(timeout=10)
        return {"value": inputs.get("value")}


@pytest.fixture(autouse=True)
def _register_gate():
    registry._nodes["_FpGate"] = _FpGateNode
    _GATE.clear()
    yield
    _GATE.set()
    registry._nodes.pop("_FpGate", None)


def _graph(gated: bool = False, settings: dict | None = None) -> dict:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": "_TestSource", "data": {"params": {"val": "hi"}}},
    ]
    edges = [{"id": "et", "source": "start", "target": "src",
              "sourceHandle": "trigger", "type": "trigger"}]
    if gated:
        nodes.append({"id": "gate", "type": "_FpGate",
                      "data": {"params": {}}})
        edges.append({"id": "eg", "source": "src", "target": "gate",
                      "sourceHandle": "value", "targetHandle": "value"})
    graph: dict[str, Any] = {"nodes": nodes, "edges": edges}
    if settings is not None:
        graph["settings"] = settings
    return graph


async def _await_terminal(store: RunStore, run_id: str, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        if record is not None and record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


async def _await_status(store: RunStore, run_id: str, status: str,
                        timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        if record is not None and record.status == status:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached {status}")


async def test_a_run_records_the_real_fingerprint_and_its_facts(store):
    service = RunService(store, shutdown_grace_s=2.0)
    try:
        submitted = await service.submit(
            _graph(settings={"seed": 11}),
            options={"device": "cpu", "deterministic": True},
            provenance=RunProvenance())
        record = await _await_terminal(store, submitted.run_id)
    finally:
        await service.shutdown()
    assert record.status == STATUS_SUCCEEDED
    stored = await store.get_fingerprint(record.fingerprint_digest)
    assert set(stored) == TOP_LEVEL_KEYS
    assert fpmod.fingerprint_digest(stored) == record.fingerprint_digest
    facts = record.execution_facts
    assert facts["requested"]["device"] == "cpu"
    assert facts["requested"]["graph_settings_seed"] == 11
    assert facts["requested"]["seed"] == 11          # filled from settings
    assert facts["resolved"]["device"] == record.queue_key == "cpu"
    assert facts["resolved"]["seed"] == 11
    assert facts["resolved"]["deterministic"] is True


async def test_a_queued_run_records_the_environment_that_starts_it(
        store, monkeypatch):
    """Submitted under environment A, started under B: the run says B, and
    its submission provenance still says what it said at submit."""
    current = {"doc": {"schema_version": 1, "marker": "A"}}
    monkeypatch.setattr("app.core.run_service.collect_fingerprint",
                        lambda: dict(current["doc"]))
    service = RunService(store, shutdown_grace_s=2.0,
                         limits=QueueLimits(cpu=1, gpu=1, interactive=1))
    pins = {"pack": {"source_kind": "github_url", "sha": "1" * 40}}
    try:
        first = await service.submit(_graph(gated=True),
                                     options={"device": "cpu"},
                                     provenance=RunProvenance())
        await _await_status(store, first.run_id, "running")
        second = await service.submit(
            _graph(), options={"device": "cpu"},
            provenance=RunProvenance(git_commit="c" * 40, git_dirty=False,
                                     plugin_pins=pins))
        assert second.status == "queued"
        current["doc"] = {"schema_version": 1, "marker": "B"}
        _GATE.set()
        a = await _await_terminal(store, first.run_id)
        b = await _await_terminal(store, second.run_id)
    finally:
        _GATE.set()
        await service.shutdown()

    assert a.fingerprint_digest == fpmod.fingerprint_digest(
        {"schema_version": 1, "marker": "A"})
    assert b.fingerprint_digest == fpmod.fingerprint_digest(
        {"schema_version": 1, "marker": "B"})
    assert (await store.get_fingerprint(b.fingerprint_digest))["marker"] == "B"
    # Submission metadata is still there to compare against.
    assert b.git_commit == "c" * 40
    assert b.git_dirty is False
    assert b.plugin_pins == pins


async def test_a_failing_collection_never_fails_the_run(store, monkeypatch):
    def _boom():
        raise RuntimeError("no fingerprint today")

    monkeypatch.setattr("app.core.run_service.collect_fingerprint", _boom)
    service = RunService(store, shutdown_grace_s=2.0)
    try:
        submitted = await service.submit(_graph(), options={"device": "cpu"},
                                         provenance=RunProvenance())
        record = await _await_terminal(store, submitted.run_id)
    finally:
        await service.shutdown()
    assert record.status == STATUS_SUCCEEDED
    assert record.fingerprint_digest is None
    assert record.execution_facts is None


# ── 4. HTTP ───────────────────────────────────────────────────────────────


BASE_URL = f"http://127.0.0.1:{settings.PORT}"


@pytest.fixture
async def client(tmp_path):
    database = Database(tmp_path / "api.db")
    database.connect()
    service = RunService(RunStore(database), shutdown_grace_s=5.0)
    app.state.db = database
    app.state.run_service = service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL,
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http
    finally:
        await service.shutdown()
        database.close()
        for attribute in ("db", "run_service"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


async def test_the_fingerprint_endpoint_answers_with_its_schema(client):
    response = await client.get("/api/system/fingerprint")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"fingerprint", "digest"}
    assert body["fingerprint"]["schema_version"] == fpmod.SCHEMA_VERSION
    assert set(body["fingerprint"]) == TOP_LEVEL_KEYS
    assert body["digest"] == fpmod.fingerprint_digest(body["fingerprint"])


async def test_the_run_detail_carries_the_fingerprint_and_the_list_does_not(
        client):
    response = await client.post("/api/runs", json={
        "graph": _graph(), "options": {"device": "auto"}})
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]
    deadline = time.monotonic() + 15
    while True:
        detail = (await client.get(f"/api/runs/{run_id}")).json()
        if detail["finished_at"] is not None:
            break
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)

    system = (await client.get("/api/system/fingerprint")).json()
    assert detail["fingerprint_digest"] == system["digest"]
    assert detail["fingerprint"] == system["fingerprint"]
    assert detail["execution_facts"]["requested"]["device"] == "auto"
    assert detail["execution_facts"]["resolved"]["device"] \
        == detail["queue_key"]
    # Run options, project git and plugin pins keep their shape.
    for key in ("options", "git_commit", "git_dirty", "plugin_pins"):
        assert key in detail

    rows = (await client.get("/api/runs")).json()["runs"]
    row = next(r for r in rows if r["id"] == run_id)
    assert row["fingerprint_digest"] == system["digest"]
    assert "execution_facts" in row
    assert "fingerprint" not in row

    recorded = await client.get(
        f"/api/system/fingerprints/{system['digest']}")
    assert recorded.status_code == 200
    assert recorded.json() == {"fingerprint": system["fingerprint"],
                               "digest": system["digest"]}


@pytest.mark.parametrize("digest", ["0" * 64, "not-a-digest", "A" * 64])
async def test_an_unknown_digest_is_a_404(client, digest):
    response = await client.get(f"/api/system/fingerprints/{digest}")
    assert response.status_code == 404


# ── the node catalog hash (#781) ─────────────────────────────────────────


class _CatalogNode(BaseNode):
    NODE_NAME = "CatalogProbe"
    CATEGORY = "Test"

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="out", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls):
        from app.core.node_base import ParamDefinition, ParamType

        return [ParamDefinition(name="device", param_type=ParamType.SELECT,
                                default="cuda",
                                options=["auto", "cpu", "cuda", "mps"])]

    def execute(self, inputs, params):
        return {"out": None}


def test_the_catalog_hash_is_stable_and_cached_per_generation():
    from app.core.node_registry import NodeRegistry, node_catalog

    nodes = NodeRegistry()
    nodes.register(_CatalogNode)
    first = node_catalog(nodes)
    assert node_catalog(nodes) is first
    assert first.node_types == ("CatalogProbe",)
    # Registered again (a reload): same definitions, same hash.
    again = NodeRegistry()
    again.register(_CatalogNode)
    assert node_catalog(again).hash == first.hash
    # Another type changes it, and so does the same node under a plugin id.
    again.register(_FpGateNode)
    assert node_catalog(again).hash != first.hash
    prefixed = NodeRegistry()
    prefixed.register(_CatalogNode, plugin_id="p")
    assert node_catalog(prefixed).hash != first.hash


def test_the_catalog_hash_does_not_depend_on_this_machines_devices(
        monkeypatch):
    """The node API narrows a ``device`` param to the devices this machine
    has; the catalog hashes the param as declared, so a CPU box and a CUDA
    box with the same plugins agree."""
    from app.core import device_utils
    from app.core.node_registry import (
        NodeRegistry,
        node_catalog,
        node_definition,
    )

    nodes = NodeRegistry()
    nodes.register(_CatalogNode)
    monkeypatch.setattr(device_utils, "device_options",
                        lambda name, options: ["cpu"])
    cpu_box = node_catalog(nodes).hash
    served = node_definition("CatalogProbe", _CatalogNode)
    assert served.params[0].options == ["cpu"]
    assert served.params[0].default == "cpu"
    nodes.register(_CatalogNode)  # a new generation: recomputed
    monkeypatch.setattr(device_utils, "device_options",
                        lambda name, options: ["cpu", "cuda:0", "cuda:1"])
    assert node_catalog(nodes).hash == cpu_box


def test_the_catalog_cache_never_answers_for_a_dead_registry():
    """#863: the cache was keyed by ``id(registry)``, and a registry made
    right after another was freed gets the same address and generation
    (CPython reuses the freed block at once), so it was handed the dead
    one's catalog. Keyed on the object now. No ``gc.collect()`` here: it
    changes the allocator's pattern and hides the reuse this reproduces."""
    from app.core.node_registry import NodeRegistry, node_catalog

    class Other(BaseNode):
        NODE_NAME = "CatalogOther"
        CATEGORY = "Test"

        @classmethod
        def define_inputs(cls):
            return []

        @classmethod
        def define_outputs(cls):
            return []

        def execute(self, inputs, params):
            return {}

    reused = 0
    for _ in range(30):
        first = NodeRegistry()
        first.register(_CatalogNode)
        first_hash = node_catalog(first).hash
        address = id(first)
        first = None                     # freed now: refcount, not gc
        second = NodeRegistry()
        second.register(Other)           # the same generation `first` had
        reused += id(second) == address
        assert node_catalog(second).hash != first_hash
        assert node_catalog(second).node_types == ("CatalogOther",)
        second = None
    # The old key collided whenever the address was reused, which the
    # allocator does nearly every time; the test is meaningful only when
    # it happened at least once.
    assert reused > 0


def test_a_broken_definition_hashes_instead_of_failing():
    from app.core.node_registry import NodeRegistry, node_catalog

    class _Broken(_CatalogNode):
        NODE_NAME = "Broken"

        @classmethod
        def define_params(cls):
            raise RuntimeError("boom")

    nodes = NodeRegistry()
    nodes.register(_Broken)
    assert node_catalog(nodes).node_types == ("Broken",)
