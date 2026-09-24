"""SECRET params must never reach ``exec_runs.graph_snapshot`` (#251).

Every OTHER path that persists a graph already scrubs: save, export, publish
pre-flight, preset creation, Python codegen. The run path was the exception --
it wrote the graph exactly as submitted, so an API key typed into an
``LLMChat`` node landed verbatim in the shared SQLite file and stayed there
for the last ``RUN_RETENTION_KEEP_LAST`` (200) runs.

The fix is scrub-on-write plus re-inject-at-promotion from an in-memory
vault, and the argument that it is SAFE rests on one fact this file pins
directly (``test_a_queued_run_does_not_survive_a_restart_so_neither_must_its_secret``):
a queued run does not survive a restart. Both indexes of the scheduler are
process memory, and every ``queued`` row a dead process left behind is
retired to ``interrupted`` on the next boot. So the vault's lifetime -- the
process -- is exactly the lifetime over which a snapshot can still be read.
Nothing needs the secret after a restart because nothing runs after one.

The central assertion is deliberately made against BYTES ON DISK rather than
against the decoded row: the leak was at rest, so the test is at rest too.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from app.core.db import Database
from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry
from app.core.run_service import LANE_INTERACTIVE, RunService
from app.core.run_store import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_QUEUED,
    STATUS_SUCCEEDED,
    RunProvenance,
    RunStore,
)
from app.core.secret_params import (
    iter_secret_slots,
    restore_graph_secrets,
    split_graph_secrets,
)
from app.schemas.models import InternalNodeSchema, PresetDefinition

#: The value that must never appear in the database. Distinctive enough that
#: a substring scan of the raw file is meaningful.
LIVE_KEY = "sk-live-DO-NOT-PERSIST-251"


# ── a node that declares a SECRET param and records what it actually got ──


class _SecretEchoNode(BaseNode):
    """Declares a SECRET param and records the value execution SAW.

    The recording is what proves the other half of the fix: scrubbing the
    stored copy is only correct if the run still executes with the real key.
    A test that checked the database alone would pass just as happily if the
    key had been dropped on the floor.
    """

    NODE_NAME = "_SecretEcho"
    CATEGORY = "Test"
    DESCRIPTION = "Echoes a SECRET param so tests can see what ran"

    #: Appended to by every execution in the process.
    seen: list[str] = []

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="api_key", param_type=ParamType.SECRET,
                            default=""),
            ParamDefinition(name="label", param_type=ParamType.STRING,
                            default="x"),
        ]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        type(self).seen.append(str(params.get("api_key", "")))
        return {"value": inputs.get("value")}


class _SecretSourceNode(BaseNode):
    """No inputs, so it can sit behind Start as the value producer."""

    NODE_NAME = "_SecretSource"
    CATEGORY = "Test"
    DESCRIPTION = "Constant source"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="val", param_type=ParamType.STRING,
                                default="hi")]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": params.get("val", "hi")}


class _SecretSlowNode(BaseNode):
    """Occupies the CPU queue so the next submit genuinely waits in line."""

    NODE_NAME = "_SecretSlow"
    CATEGORY = "Test"
    DESCRIPTION = "Blocks the worker thread"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="seconds", param_type=ParamType.FLOAT,
                                default=0.5)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 0.5)))
        return {"value": inputs.get("value")}


_TEST_NODES = {
    "_SecretEcho": _SecretEchoNode,
    "_SecretSource": _SecretSourceNode,
    "_SecretSlow": _SecretSlowNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    _SecretEchoNode.seen = []
    yield
    _SecretEchoNode.seen = []
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "codefyui.db"


@pytest.fixture
def db(db_path):
    database = Database(db_path)
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


@pytest.fixture
async def service(store):
    svc = RunService(store, shutdown_grace_s=2.0)
    try:
        yield svc
    finally:
        await svc.shutdown()


def _secret_graph(key: str = LIVE_KEY, *,
                  middle: str = "_SecretEcho") -> dict[str, Any]:
    """Start -> source -> _SecretEcho(api_key=key) -> Print."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_SecretSource",
             "data": {"params": {"val": "hi"}}},
            {"id": "mid", "type": middle,
             "data": {"params": {"api_key": key, "label": "keep-me"}}},
            {"id": "print", "type": "Print",
             "data": {"params": {"label": "out"}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "mid",
             "sourceHandle": "value", "targetHandle": "value"},
            {"id": "e2", "source": "mid", "target": "print",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
    }


async def _await_terminal(store: RunStore, run_id: str, *,
                          timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        assert record is not None, f"run {run_id} vanished"
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def _db_bytes(db_path) -> bytes:
    """Everything SQLite has written for this database, WAL sidecar included.

    ``journal_mode=WAL`` means a freshly written row may live only in
    ``-wal`` until a checkpoint, so reading the main file alone could report
    "clean" for a secret that is very much on disk.
    """
    blob = b""
    for suffix in ("", "-wal", "-shm"):
        candidate = db_path.parent / (db_path.name + suffix)
        if candidate.exists():
            blob += candidate.read_bytes()
    return blob


# ── the central assertion ─────────────────────────────────────────────────


async def test_a_submitted_secret_is_not_written_to_the_run_database(
        store, service, db_path):
    """THE regression test for #251: submit a key, find it nowhere on disk.

    Fails without the fix -- ``create_run(graph_snapshot=normalized_graph)``
    stored the submitted graph verbatim.
    """
    submitted = await service.submit(_secret_graph())
    record = await _await_terminal(store, submitted.run_id)
    assert record.status == STATUS_SUCCEEDED

    snapshot = await store.get_graph_snapshot(submitted.run_id)
    stored = {n["id"]: n for n in snapshot["nodes"]}
    assert stored["mid"]["data"]["params"]["api_key"] == ""
    # The non-secret sibling param is untouched -- this scrubs a value, it
    # does not redact the graph.
    assert stored["mid"]["data"]["params"]["label"] == "keep-me"

    assert LIVE_KEY.encode() not in _db_bytes(db_path), (
        "the submitted API key is on disk in the run database")


async def test_the_run_still_executes_with_the_real_key(store, service):
    """The other half: scrubbing the stored copy must not blind the run.

    This is what makes option 3 (scrub on write, re-inject at promotion) a
    fix rather than a behaviour change. A queued-lane run is promoted out of
    the FIFO with ``graph=None`` and re-reads the SCRUBBED row, so without
    the vault it would execute with an empty key.
    """
    submitted = await service.submit(_secret_graph())
    await _await_terminal(store, submitted.run_id)
    assert _SecretEchoNode.seen == [LIVE_KEY]


async def test_a_run_that_waited_in_the_queue_still_gets_its_key(
        store, service, db_path):
    """The same, but genuinely promoted off a non-empty queue.

    The default CPU lane admits 2 at once, so three slow runs are enough to
    make the last one wait for a slot rather than start on the spot.
    """
    blockers = [
        await service.submit({
            "nodes": [
                {"id": "start", "type": "Start", "data": {"params": {}}},
                {"id": "src", "type": "_SecretSource",
                 "data": {"params": {"val": "hi"}}},
                {"id": "slow", "type": "_SecretSlow",
                 "data": {"params": {"seconds": 0.6}}},
            ],
            "edges": [
                {"id": "et", "source": "start", "target": "src",
                 "sourceHandle": "trigger", "type": "trigger"},
                {"id": "e1", "source": "src", "target": "slow",
                 "sourceHandle": "value", "targetHandle": "value"},
            ],
        })
        for _ in range(2)
    ]
    waiting = await service.submit(_secret_graph())
    # It really is in line, not running.
    assert waiting.status == STATUS_QUEUED
    assert waiting.run_id in {
        run_id for ids in service.queue_snapshot().values() for run_id in ids
    }

    for blocker in blockers:
        await _await_terminal(store, blocker.run_id)
    await _await_terminal(store, waiting.run_id)

    assert _SecretEchoNode.seen == [LIVE_KEY]
    assert LIVE_KEY.encode() not in _db_bytes(db_path)


async def test_the_interactive_lane_scrubs_its_snapshot_too(
        store, service, db_path):
    """The canvas lane persists a row it never reads back -- scrub it anyway.

    ``_submit_interactive`` hands the LIVE graph straight to ``_start``, so
    its snapshot is write-only. That makes the stored copy pure liability:
    nothing needs the key to be in it, and it is retained exactly as long as
    a queued run's is.
    """
    submitted = await service.submit(
        _secret_graph(), options={"lane": LANE_INTERACTIVE})
    await _await_terminal(store, submitted.run_id)

    snapshot = await store.get_graph_snapshot(submitted.run_id)
    stored = {n["id"]: n for n in snapshot["nodes"]}
    assert stored["mid"]["data"]["params"]["api_key"] == ""
    assert _SecretEchoNode.seen == [LIVE_KEY]
    assert LIVE_KEY.encode() not in _db_bytes(db_path)


async def test_the_interactive_lane_scrubs_a_key_inside_a_preset_node(
        store, service, db_path):
    """A preset node's ``internalParams`` is the third place a key can sit.

    The canvas sends the keys the user typed with every run, and a key typed
    into an old preset that still exposes one lands here, not in ``params``.
    The run must still get it; the stored row must not.
    """
    graph = {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_SecretSource",
             "data": {"params": {"val": "hi"}}},
            {"id": "p", "type": "preset:_KeyedEcho",
             "data": {"params": {},
                      "internalParams": {"echo": {"api_key": LIVE_KEY}}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "p",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
        # Portable, as the canvas sends it, so no preset has to be installed.
        "presets": [{
            "preset_name": "_KeyedEcho", "category": "Test", "description": "",
            "nodes": [{"id": "echo", "type": "_SecretEcho",
                       "params": {"api_key": "", "label": "x"}}],
            "edges": [],
            "exposed_inputs": [{"name": "value", "internal_node": "echo",
                                "internal_port": "value"}],
            "exposed_outputs": [{"name": "value", "internal_node": "echo",
                                 "internal_port": "value"}],
            "exposed_params": [],
        }],
    }
    submitted = await service.submit(graph, options={"lane": LANE_INTERACTIVE})
    record = await _await_terminal(store, submitted.run_id)
    assert record.status == STATUS_SUCCEEDED

    snapshot = await store.get_graph_snapshot(submitted.run_id)
    stored = {n["id"]: n for n in snapshot["nodes"]}
    assert stored["p"]["data"]["internalParams"]["echo"]["api_key"] == ""
    assert _SecretEchoNode.seen == [LIVE_KEY]
    assert LIVE_KEY.encode() not in _db_bytes(db_path)


async def test_a_graph_without_secrets_is_stored_verbatim(store, service):
    """No secret, no vault entry, no copying -- the common case is untouched."""
    submitted = await service.submit(_secret_graph(key=""))
    await _await_terminal(store, submitted.run_id)

    snapshot = await store.get_graph_snapshot(submitted.run_id)
    stored = {n["id"]: n for n in snapshot["nodes"]}
    assert stored["mid"]["data"]["params"] == {"api_key": "", "label": "keep-me"}
    assert service.pending_secret_count() == 0


# ── the vault's lifetime ──────────────────────────────────────────────────


async def test_the_vault_is_emptied_when_the_run_finishes(store, service):
    """An in-memory store that only grows is a leak of a different kind."""
    submitted = await service.submit(_secret_graph())
    await _await_terminal(store, submitted.run_id)
    # _drive's `finally` runs after the row is written; give it the loop.
    for _ in range(50):
        if service.pending_secret_count() == 0:
            break
        await asyncio.sleep(0.02)
    assert service.pending_secret_count() == 0


async def test_cancelling_a_waiting_run_drops_its_secret(store, service):
    """A run retired out of the queue never reaches ``_drive``'s cleanup.

    ``_retire_waiting`` is the one path both cancel and shutdown take, which
    is why the vault is cleared there rather than in each caller.
    """
    blockers = [
        await service.submit({
            "nodes": [
                {"id": "start", "type": "Start", "data": {"params": {}}},
                {"id": "src", "type": "_SecretSource",
                 "data": {"params": {"val": "hi"}}},
                {"id": "slow", "type": "_SecretSlow",
                 "data": {"params": {"seconds": 0.5}}},
            ],
            "edges": [
                {"id": "et", "source": "start", "target": "src",
                 "sourceHandle": "trigger", "type": "trigger"},
                {"id": "e1", "source": "src", "target": "slow",
                 "sourceHandle": "value", "targetHandle": "value"},
            ],
        })
        for _ in range(2)
    ]
    waiting = await service.submit(_secret_graph())
    assert waiting.status == STATUS_QUEUED
    assert service.pending_secret_count() == 1

    outcome = await service.cancel(waiting.run_id)
    assert outcome.cancelled is True
    assert service.pending_secret_count() == 0
    # The cancelled run never ran, so it never saw the key.
    assert _SecretEchoNode.seen == []

    for blocker in blockers:
        await _await_terminal(store, blocker.run_id)


async def test_shutdown_clears_every_held_secret(store):
    """Shutdown retires the queue; nothing may keep a key past that.

    Two things are checked, and the second is why ``shutdown`` clears the
    whole map instead of trusting the per-run drops. The waiting run below is
    covered by ``_retire_waiting`` and would pass either way; the STRAY entry
    is not reachable through any current path, and stands in for a future one
    that forgets to clean up. The sweep is what makes that a bounded mistake
    rather than a process that ends still holding a credential.
    """
    svc = RunService(store, shutdown_grace_s=2.0)
    for _ in range(2):
        await svc.submit({
            "nodes": [
                {"id": "start", "type": "Start", "data": {"params": {}}},
                {"id": "src", "type": "_SecretSource",
                 "data": {"params": {"val": "hi"}}},
                {"id": "slow", "type": "_SecretSlow",
                 "data": {"params": {"seconds": 0.4}}},
            ],
            "edges": [
                {"id": "et", "source": "start", "target": "src",
                 "sourceHandle": "trigger", "type": "trigger"},
                {"id": "e1", "source": "src", "target": "slow",
                 "sourceHandle": "value", "targetHandle": "value"},
            ],
        })
    await svc.submit(_secret_graph())
    assert svc.pending_secret_count() == 1
    # An entry no lifecycle path owns -- see the docstring.
    svc._run_secrets["orphaned-by-a-future-bug"] = {("nodes", 0): LIVE_KEY}
    assert svc.pending_secret_count() == 2

    await svc.shutdown()
    assert svc.pending_secret_count() == 0


async def test_a_queued_run_does_not_survive_a_restart_so_neither_must_its_secret(
        store):
    """The premise the whole design rests on, pinned so it cannot drift.

    If a future change ever made queued runs resume across a restart, an
    in-memory vault would silently become the wrong answer -- the run would
    come back and its key would not. This test fails first if that happens.
    """
    record = await store.create_run(
        graph_snapshot=_secret_graph(), status=STATUS_QUEUED,
        queue_key="cpu", provenance=RunProvenance(),
    )
    # A fresh service is a fresh process: empty scheduler, empty vault.
    fresh = RunService(store, shutdown_grace_s=2.0)
    assert fresh.pending_secret_count() == 0
    assert await fresh.recover_interrupted() == 1

    row = await store.get_run(record.id)
    assert row.status == STATUS_INTERRUPTED
    assert row.started_at is None
    await fresh.shutdown()


# ── the backfill for rows written before the fix ──────────────────────────


async def test_startup_scrubs_secrets_left_by_an_older_version(
        store, service, db_path):
    """The fix protects future writes; the old rows are the exposure today.

    Retention is count-based only, so a key written by an older build sits
    there until 200 further runs push it out -- unbounded in time.
    """
    record = await store.create_run(
        graph_snapshot=_secret_graph(), status=STATUS_QUEUED,
        queue_key="cpu", provenance=RunProvenance(),
    )
    await store.mark_finished(record.id, STATUS_SUCCEEDED,
                              expected=(STATUS_QUEUED,))
    # Precondition: it really is on disk (this is the pre-fix state).
    leaked = await store.get_graph_snapshot(record.id)
    assert leaked["nodes"][2]["data"]["params"]["api_key"] == LIVE_KEY

    assert await service.scrub_stored_secrets() == 1

    cleaned = await store.get_graph_snapshot(record.id)
    assert cleaned["nodes"][2]["data"]["params"]["api_key"] == ""
    assert cleaned["nodes"][2]["data"]["params"]["label"] == "keep-me"
    # Idempotent: a second pass finds nothing left to do.
    assert await service.scrub_stored_secrets() == 0


async def test_freed_pages_are_zeroed_so_pruning_leaves_no_residue(
        store, service, db, db_path):
    """Retention DELETEs a row; its bytes must not stay in a freed page.

    This pins the DELETE half of what `PRAGMA secure_delete=ON` buys, because
    it is the one retention exercises continuously against exactly the rows
    that used to hold unscrubbed keys. Measured on SQLite 3.50.4 with the
    real schema, explicit checkpoints on both sides:

      DELETE                OFF -> residue in main file    ON -> none
      UPDATE 12KB -> 0KB    OFF -> residue in main file    ON -> none
      UPDATE 12KB -> 12KB   OFF -> none                    ON -> none

    The pragma is NOT only about DELETE, and the middle row is the one worth
    remembering: a shrinking rewrite releases overflow pages, so the #251
    backfill sweep needs it too. The bottom row -- a same-size rewrite
    leaving nothing because the chain is rewritten in place -- is incidental
    and flips as soon as the new size crosses a page boundary.

    The checkpoints are what make this deterministic rather than incidental:
    without the first one both writes live in the same WAL generation and the
    main file never sees the secret at all, so the probe would pass for a
    reason that has nothing to do with the pragma.
    """
    async def checkpoint() -> None:
        await db.run(lambda c: c.execute("PRAGMA wal_checkpoint(TRUNCATE)"))

    record = await store.create_run(
        graph_snapshot=_secret_graph(), status=STATUS_QUEUED,
        queue_key="cpu", provenance=RunProvenance(),
    )
    await store.mark_finished(record.id, STATUS_SUCCEEDED,
                              expected=(STATUS_QUEUED,))
    await checkpoint()
    # Precondition: the secret really is in the main file, the way it would
    # be on an install that has been running for days.
    assert LIVE_KEY.encode() in db_path.read_bytes()

    assert await store.prune(keep_last=0) == 1
    await checkpoint()

    assert LIVE_KEY.encode() not in db_path.read_bytes(), (
        "a pruned run's snapshot is still readable in a freed page")


async def test_the_connection_sets_secure_delete(db):
    """It is PER-CONNECTION, not stored in the file.

    A reopened connection reports 0 again, so this cannot be established once
    at create time -- every place that opens a connection has to set it, and
    right now `Database.connect` is the only one. This test is what fails if
    a second connection site appears without it.
    """
    assert await db.run(
        lambda c: c.execute("PRAGMA secure_delete").fetchone()[0]) == 1


async def test_the_backfill_leaves_an_active_row_alone(store, service):
    """A ``queued`` row may belong to THIS process's scheduler.

    Rewriting one would blank the key a promotion is about to read back --
    the vault only covers runs this process submitted, and a row created
    directly is exactly the orphan case it does not.
    """
    record = await store.create_run(
        graph_snapshot=_secret_graph(), status=STATUS_QUEUED,
        queue_key="cpu", provenance=RunProvenance(),
    )
    assert await service.scrub_stored_secrets() == 0
    still = await store.get_graph_snapshot(record.id)
    assert still["nodes"][2]["data"]["params"]["api_key"] == LIVE_KEY


# ── the split/restore primitives ──────────────────────────────────────────


def test_split_returns_the_same_object_when_there_is_no_secret():
    graph = {"nodes": [{"id": "a", "type": "Print",
                        "data": {"params": {"label": "x"}}}],
             "edges": [], "presets": [], "subgraphs": []}
    scrubbed, vault = split_graph_secrets(graph)
    assert vault == {}
    assert scrubbed is graph


def test_split_does_not_mutate_the_caller_s_graph():
    """The submitter still needs the real values to execute with."""
    graph = _secret_graph()
    scrubbed, vault = split_graph_secrets(graph)
    assert graph["nodes"][2]["data"]["params"]["api_key"] == LIVE_KEY
    assert scrubbed["nodes"][2]["data"]["params"]["api_key"] == ""
    assert list(vault.values()) == [LIVE_KEY]


def test_split_and_restore_round_trip_through_json():
    """Addresses must outlive the one round trip the snapshot column makes."""
    import json

    graph = _secret_graph()
    scrubbed, vault = split_graph_secrets(graph)
    reloaded = json.loads(json.dumps(scrubbed))
    assert restore_graph_secrets(reloaded, vault) == 1
    assert reloaded["nodes"][2]["data"]["params"]["api_key"] == LIVE_KEY


def test_restore_is_a_no_op_without_a_vault():
    graph = _secret_graph(key="")
    assert restore_graph_secrets(graph, {}) == 0
    assert restore_graph_secrets(graph, None) == 0


def test_slots_are_found_in_subgraph_definitions():
    """A node inside a collapsed block holds an ordinary param."""
    graph = {
        "nodes": [{"id": "blk", "type": "subgraph:d1", "data": {"params": {}}}],
        "edges": [],
        "presets": [],
        "subgraphs": [{
            "id": "d1",
            "nodes": [{"id": "inner", "type": "_SecretEcho",
                       "data": {"params": {"api_key": LIVE_KEY}}}],
            "edges": [],
        }],
    }
    scrubbed, vault = split_graph_secrets(graph)
    assert list(vault.values()) == [LIVE_KEY]
    assert (scrubbed["subgraphs"][0]["nodes"][0]["data"]["params"]["api_key"]
            == "")
    assert restore_graph_secrets(scrubbed, vault) == 1


def test_slots_are_found_in_portable_preset_definitions():
    """``presets[].nodes[].params`` is a flatter shape than a graph node."""
    graph = {
        "nodes": [{"id": "a", "type": "Print",
                   "data": {"params": {"label": "x"}}}],
        "edges": [],
        "presets": [{
            "preset_name": "P", "category": "Test", "description": "",
            "nodes": [{"id": "chat", "type": "LLMChat",
                       "params": {"openai_api_key": LIVE_KEY}}],
            "edges": [], "exposed_inputs": [], "exposed_outputs": [],
            "exposed_params": [],
        }],
        "subgraphs": [],
    }
    scrubbed, vault = split_graph_secrets(graph)
    assert list(vault.values()) == [LIVE_KEY]
    assert scrubbed["presets"][0]["nodes"][0]["params"]["openai_api_key"] == ""


def test_addresses_are_positional_so_duplicate_ids_cannot_collide():
    """Two nodes sharing an id must still restore their OWN values."""
    graph = {
        "nodes": [
            {"id": "dup", "type": "_SecretEcho",
             "data": {"params": {"api_key": "first"}}},
            {"id": "dup", "type": "_SecretEcho",
             "data": {"params": {"api_key": "second"}}},
        ],
        "edges": [], "presets": [], "subgraphs": [],
    }
    scrubbed, vault = split_graph_secrets(graph)
    assert len(vault) == 2
    assert restore_graph_secrets(scrubbed, vault) == 2
    assert [n["data"]["params"]["api_key"] for n in scrubbed["nodes"]] == [
        "first", "second"]


def test_slots_tolerate_a_malformed_graph():
    """A hand-edited file must not crash the submit path."""
    assert list(iter_secret_slots({})) == []
    assert list(iter_secret_slots({"nodes": "not-a-list"})) == []
    assert list(iter_secret_slots(
        {"nodes": [None, 3], "subgraphs": [None], "presets": [None]})) == []
    scrubbed, vault = split_graph_secrets({"nodes": None})
    assert vault == {}


# ── a node type this server does not know (#537) ──────────────────────────
#
# ``secret_param_names`` names the SECRET params of a type the registry
# resolves, and nothing else. A node of a type it cannot resolve -- a custom
# node disabled from another browser, a plugin removed while a tab stayed
# open, a hand-edited graph posted to ``POST /api/runs`` -- contributed no
# slots at all, and the row is written before the engine gets to refuse the
# type. So on submit, every value such a node carries is kept out of the row
# the way a SECRET value is.

#: No test registers this; the engine refuses it by name.
UNKNOWN_TYPE = "_NoSuchNodeType"


class _LateEchoNode(BaseNode):
    """Registered only part-way through a test, while its run waits.

    Records EVERY param it ran with: a restore that put back the SECRET value
    and left the rest blank would pass a check on the key alone.
    """

    NODE_NAME = "_LateEcho"
    CATEGORY = "Test"
    DESCRIPTION = "Records the params it ran with"

    seen: list[dict[str, Any]] = []

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="api_key", param_type=ParamType.SECRET,
                            default=""),
            ParamDefinition(name="label", param_type=ParamType.STRING,
                            default="x"),
        ]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        type(self).seen.append(dict(params))
        return {"value": inputs.get("value")}


def _slow_graph(seconds: float = 0.6) -> dict[str, Any]:
    """Start -> source -> _SecretSlow: holds one CPU slot for *seconds*."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_SecretSource",
             "data": {"params": {"val": "hi"}}},
            {"id": "slow", "type": "_SecretSlow",
             "data": {"params": {"seconds": seconds}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "slow",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
    }


def _portable_preset(name: str,
                     nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``presets[]`` entry complete enough for ``build_preset_fallback``."""
    return {"preset_name": name, "category": "Test", "description": "",
            "nodes": nodes, "edges": [], "exposed_inputs": [],
            "exposed_outputs": [], "exposed_params": []}


def _restored_through_json(
        scrubbed: Any, vault: Any) -> tuple[dict[str, Any], int]:
    """What promotion sees: the stored copy read back, then restored."""
    reloaded = json.loads(json.dumps(scrubbed))
    return reloaded, restore_graph_secrets(reloaded, vault)


async def test_the_interactive_lane_keeps_an_unknown_node_s_values_out_of_the_row(
        store, service, db_path):
    """Every value, not only the one that looks like a key.

    Nothing on the server can say which of an unknown node's params is
    secret. This lane runs the live graph, so the blanks in the stored copy
    cost the run nothing, and it still ends in the engine's own error.
    """
    submitted = await service.submit(
        _secret_graph(middle=UNKNOWN_TYPE), options={"lane": LANE_INTERACTIVE})
    record = await _await_terminal(store, submitted.run_id)
    assert record.status == STATUS_FAILED
    assert f"Unknown node type: {UNKNOWN_TYPE}" in (record.error or "")

    snapshot = await store.get_graph_snapshot(submitted.run_id)
    stored = {n["id"]: n for n in snapshot["nodes"]}
    assert stored["mid"]["data"]["params"] == {"api_key": "", "label": ""}
    # Its registered neighbours are stored as sent.
    assert stored["src"]["data"]["params"] == {"val": "hi"}
    assert stored["print"]["data"]["params"] == {"label": "out"}
    assert LIVE_KEY.encode() not in _db_bytes(db_path)


async def test_the_queued_lane_holds_an_unknown_node_s_values_in_the_vault(
        store, service, db_path):
    """Blank on the row, held in memory, put back at promotion.

    Promotion re-reads the row, so the vault is what gives the run back the
    values it was sent. Here the type is still unknown when the run starts,
    so it fails with the engine's error, as it did before.
    """
    blockers = [await service.submit(_slow_graph()) for _ in range(2)]
    waiting = await service.submit(_secret_graph(middle=UNKNOWN_TYPE))
    assert waiting.status == STATUS_QUEUED
    # Read before anything awaits, so the run cannot have started yet.
    assert service._run_secrets[waiting.run_id] == {
        ("nodes", 2, "params", "api_key"): LIVE_KEY,
        ("nodes", 2, "params", "label"): "keep-me",
    }

    snapshot = await store.get_graph_snapshot(waiting.run_id)
    assert snapshot["nodes"][2]["data"]["params"] == {"api_key": "",
                                                      "label": ""}
    for blocker in blockers:
        await _await_terminal(store, blocker.run_id)
    record = await _await_terminal(store, waiting.run_id)
    assert record.status == STATUS_FAILED
    assert f"Unknown node type: {UNKNOWN_TYPE}" in (record.error or "")
    assert LIVE_KEY.encode() not in _db_bytes(db_path)


async def test_a_type_registered_while_its_run_waits_runs_with_every_value(
        store, service, db_path):
    """Restore puts back every vaulted value, whatever the registry says NOW.

    A queued run can wait for hours, and the custom node can be re-enabled in
    that time. A restore that asked the registry which slots to fill would
    find only the SECRET one, put back the key, and leave every other value
    at the ``""`` the row holds: a run that succeeds with settings nobody
    chose.
    """
    _LateEchoNode.seen = []
    blockers = [await service.submit(_slow_graph()) for _ in range(2)]
    waiting = await service.submit(_secret_graph(middle="_LateEcho"))
    assert waiting.status == STATUS_QUEUED
    # Registered before anything awaits, so the run cannot have been promoted
    # yet: unknown at submit, known at promotion.
    registry._nodes["_LateEcho"] = _LateEchoNode
    try:
        snapshot = await store.get_graph_snapshot(waiting.run_id)
        for blocker in blockers:
            await _await_terminal(store, blocker.run_id)
        record = await _await_terminal(store, waiting.run_id)
    finally:
        registry._nodes.pop("_LateEcho", None)

    # The row, written while the type was unknown, holds none of its values.
    assert snapshot["nodes"][2]["data"]["params"] == {"api_key": "",
                                                      "label": ""}
    assert record.status == STATUS_SUCCEEDED
    assert _LateEchoNode.seen == [{"api_key": LIVE_KEY, "label": "keep-me"}]
    assert LIVE_KEY.encode() not in _db_bytes(db_path)


async def test_the_startup_sweep_leaves_an_unknown_node_s_values_alone(
        store, service):
    """The sweep does NOT get the unknown-type rule, on purpose.

    It rewrites finished rows at every boot, and "unknown" is a fact about
    one boot. A plugin that failed to load once would have every value of
    every one of its nodes blanked in every past run, for good: the sweep
    keeps no vault to bring them back from.
    """
    record = await store.create_run(
        graph_snapshot=_secret_graph(middle=UNKNOWN_TYPE),
        status=STATUS_QUEUED, queue_key="cpu", provenance=RunProvenance(),
    )
    await store.mark_finished(record.id, STATUS_FAILED,
                              expected=(STATUS_QUEUED,))
    assert await service.scrub_stored_secrets() == 0
    kept = await store.get_graph_snapshot(record.id)
    assert kept["nodes"][2]["data"]["params"] == {"api_key": LIVE_KEY,
                                                  "label": "keep-me"}


def test_a_known_preset_withholds_the_entries_it_cannot_place():
    """Per ``internalParams`` entry, the inner node's type decides.

    A registered inner type keeps the SECRET rule: only its SECRET params
    go. An inner type nobody registered, or an id no definition has, loses
    the whole entry: the server cannot say what is in it.
    """
    graph = {
        "nodes": [{"id": "p", "type": "preset:_Mixed", "data": {
            "params": {},
            "internalParams": {
                "echo": {"api_key": LIVE_KEY, "label": "keep-me"},
                "ghost": {"token": "sk-ghost", "label": "g"},
                "stray": {"token": "sk-stray"},
            }}}],
        "edges": [], "subgraphs": [],
        "presets": [_portable_preset("_Mixed", [
            {"id": "echo", "type": "_SecretEcho", "params": {}},
            {"id": "ghost", "type": UNKNOWN_TYPE, "params": {}},
        ])],
    }
    scrubbed, vault = split_graph_secrets(graph, unknown_types_as_secret=True)
    assert scrubbed["nodes"][0]["data"]["internalParams"] == {
        "echo": {"api_key": "", "label": "keep-me"},
        "ghost": {"token": "", "label": ""},
        "stray": {"token": ""},
    }
    assert vault == {
        ("nodes", 0, "internalParams", "echo", "api_key"): LIVE_KEY,
        ("nodes", 0, "internalParams", "ghost", "token"): "sk-ghost",
        ("nodes", 0, "internalParams", "ghost", "label"): "g",
        ("nodes", 0, "internalParams", "stray", "token"): "sk-stray",
    }
    reloaded, restored = _restored_through_json(scrubbed, vault)
    assert restored == 4
    assert reloaded == graph


def test_a_preset_entry_is_withheld_when_either_definition_cannot_place_it():
    """The installed and the portable definition are UNIONED.

    The same union ``_preset_secret_param_map`` takes for SECRET params: the
    run uses the installed definition, but the portable one travels with the
    graph and is the one used wherever the preset is not installed. An id
    counts as placed only when every type it has across the two is known.
    """
    preset_registry._presets["_Split"] = PresetDefinition(
        preset_name="_Split", category="Test", description="",
        nodes=[InternalNodeSchema(id="a", type="Print", params={})],
        edges=[], exposed_inputs=[], exposed_outputs=[], exposed_params=[])
    try:
        graph = {
            "nodes": [{"id": "p", "type": "preset:_Split", "data": {
                "params": {}, "internalParams": {"a": {"label": "x"}}}}],
            "edges": [], "subgraphs": [],
            "presets": [_portable_preset("_Split", [
                {"id": "a", "type": UNKNOWN_TYPE, "params": {}}])],
        }
        scrubbed, vault = split_graph_secrets(
            graph, unknown_types_as_secret=True)
    finally:
        preset_registry._presets.pop("_Split", None)
    assert scrubbed["nodes"][0]["data"]["internalParams"] == {
        "a": {"label": ""}}
    assert vault == {("nodes", 0, "internalParams", "a", "label"): "x"}


def test_an_unknown_preset_withholds_every_value():
    """Neither the preset registry nor the graph's own ``presets[]`` has it,
    so nothing says what the node is or what any of its inner nodes are."""
    graph = {
        "nodes": [{"id": "p", "type": "preset:_Nobody", "data": {
            "params": {"shown": "v"},
            "internalParams": {
                "chat": {"openai_api_key": LIVE_KEY, "model": "m"}}}}],
        "edges": [], "presets": [], "subgraphs": [],
    }
    scrubbed, vault = split_graph_secrets(graph, unknown_types_as_secret=True)
    data = scrubbed["nodes"][0]["data"]
    assert data["params"] == {"shown": ""}
    assert data["internalParams"] == {
        "chat": {"openai_api_key": "", "model": ""}}
    assert vault == {
        ("nodes", 0, "params", "shown"): "v",
        ("nodes", 0, "internalParams", "chat", "openai_api_key"): LIVE_KEY,
        ("nodes", 0, "internalParams", "chat", "model"): "m",
    }
    reloaded, restored = _restored_through_json(scrubbed, vault)
    assert restored == 3
    assert reloaded == graph


def test_a_block_withholds_the_values_of_an_unknown_node_inside_it():
    """A block's definition holds ordinary nodes, presets included.

    Also pins what counts as a value, by the same test every walk here uses:
    ``""``, ``None``, ``0`` and ``False`` are left as they are.
    """
    graph = {
        "nodes": [{"id": "blk", "type": "subgraph:d1",
                   "data": {"params": {}}}],
        "edges": [], "presets": [],
        "subgraphs": [{
            "id": "d1",
            "nodes": [
                {"id": "ghost", "type": UNKNOWN_TYPE, "data": {"params": {
                    "api_key": LIVE_KEY, "label": "keep-me",
                    "blank": "", "unset": None, "count": 0,
                    "enabled": False}}},
                {"id": "src", "type": "_SecretSource",
                 "data": {"params": {"val": "hi"}}},
                {"id": "p", "type": "preset:_Nobody", "data": {
                    "params": {},
                    "internalParams": {"chat": {"openai_api_key": "sk-in"}}}},
            ],
            "edges": [],
        }],
    }
    scrubbed, vault = split_graph_secrets(graph, unknown_types_as_secret=True)
    inner = scrubbed["subgraphs"][0]["nodes"]
    assert inner[0]["data"]["params"] == {
        "api_key": "", "label": "", "blank": "", "unset": None, "count": 0,
        "enabled": False}
    assert inner[1]["data"]["params"] == {"val": "hi"}
    assert inner[2]["data"]["internalParams"] == {
        "chat": {"openai_api_key": ""}}
    assert vault == {
        ("subgraphs", 0, "nodes", 0, "params", "api_key"): LIVE_KEY,
        ("subgraphs", 0, "nodes", 0, "params", "label"): "keep-me",
        ("subgraphs", 0, "nodes", 2, "internalParams", "chat",
         "openai_api_key"): "sk-in",
    }
    reloaded, restored = _restored_through_json(scrubbed, vault)
    assert restored == 3
    assert reloaded == graph


def test_a_portable_preset_definition_withholds_an_unknown_node_s_defaults():
    """``presets[].nodes[].params``: the third place, in its flatter shape."""
    graph = {
        "nodes": [{"id": "a", "type": "Print",
                   "data": {"params": {"label": "x"}}}],
        "edges": [], "subgraphs": [],
        "presets": [_portable_preset("P", [
            {"id": "ghost", "type": UNKNOWN_TYPE,
             "params": {"token": LIVE_KEY, "label": "g"}},
            {"id": "out", "type": "Print", "params": {"label": "out"}},
        ])],
    }
    scrubbed, vault = split_graph_secrets(graph, unknown_types_as_secret=True)
    definition = scrubbed["presets"][0]["nodes"]
    assert definition[0]["params"] == {"token": "", "label": ""}
    assert definition[1]["params"] == {"label": "out"}
    assert scrubbed["nodes"][0]["data"]["params"] == {"label": "x"}
    assert vault == {
        ("presets", 0, "nodes", 0, "params", "token"): LIVE_KEY,
        ("presets", 0, "nodes", 0, "params", "label"): "g",
    }
    reloaded, restored = _restored_through_json(scrubbed, vault)
    assert restored == 2
    assert reloaded == graph


def test_what_the_server_can_place_is_stored_as_sent():
    """The rule reaches only the types the engine itself refuses.

    A registered type keeps its non-secret values. A bare plugin name counts
    as registered when ``registry.get`` resolves it by suffix. A note is an
    annotation and a ``subgraph:`` instance a reference into ``subgraphs[]``,
    and the engine refuses neither as an unknown type. The canvas writes no
    ``params`` on a note and ``{}`` on an instance; both carry a value here
    because only a value would show a rule that wrongly caught them.
    """
    registry._nodes["zz9:_BareSource"] = _SecretSourceNode
    try:
        graph = {
            "nodes": [
                {"id": "echo", "type": "_SecretEcho",
                 "data": {"params": {"api_key": "", "label": "keep-me"}}},
                {"id": "bare", "type": "_BareSource",
                 "data": {"params": {"val": "hi"}}},
                {"id": "note", "type": "note", "data": {
                    "noteKind": "text", "noteContent": "rotate the key",
                    "params": {"text": "hand-edited"}}},
                {"id": "blk", "type": "subgraph:d1",
                 "data": {"params": {"hint": "x"}}},
                {"id": "p", "type": "preset:_Known", "data": {
                    "params": {}, "internalParams": {"out": {"label": "y"}}}},
            ],
            "edges": [],
            "subgraphs": [{"id": "d1", "nodes": [
                {"id": "in", "type": "Print",
                 "data": {"params": {"label": "inner"}}}], "edges": []}],
            "presets": [_portable_preset("_Known", [
                {"id": "out", "type": "Print", "params": {"label": "z"}}])],
        }
        scrubbed, vault = split_graph_secrets(
            graph, unknown_types_as_secret=True)
    finally:
        registry._nodes.pop("zz9:_BareSource", None)
    assert vault == {}
    assert scrubbed is graph
