"""Tests for /api/sweeps — the Sweep v1 REST surface (#140).

#140's acceptance criteria at the level a user actually meets them (HTTP),
on top of the compiler- and store-level versions in test_sweeps.py. Two test
names deliberately appear in both modules —
`test_a_failed_variant_that_logged_the_objective_is_still_ranked` and
`test_a_diverged_variant_is_unranked` — because the property is worth
proving at both levels; test_sweeps.py's are unit tests over `rank_variants`
and these run a real graph through the queue.

The probe node and the gate machinery are DUPLICATED from test_run_queue.py
on purpose, the way test_api_runs.py duplicates its own: test modules stay
independent.
"""

from __future__ import annotations

import asyncio
import csv
import io
import sqlite3
import threading
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes_runs import _CSV_BOM
from app.config import settings
from app.core import run_service as run_service_module
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.run_service import QueueLimits, RunService, RunServiceUnavailable
from app.core.run_store import TERMINAL_STATUSES, RunProvenance, RunStore
from app.core.sweep_store import SweepStore
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"


# ── the probe node (a trimmed copy of test_run_queue.py's) ────────────────


class _Probe:
    """Start order and per-variant gates. Mutated from ENGINE WORKER
    THREADS, hence the lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started: list[str] = []
        self.gates: dict[str, tuple[threading.Event, threading.Event]] = {}

    def enter(self, label: str) -> None:
        with self.lock:
            self.started.append(label)

    def hold(self, label: str) -> threading.Event:
        """Register *label* to block on a gate instead of returning.

        A gate, never a sleep: the test controls exactly when a run may
        finish, deterministically, rather than sizing a sleep against
        wall-clock timing on a loaded runner. Must be called BEFORE the run
        is submitted -- the node looks its label up on entry.
        """
        entered, release = threading.Event(), threading.Event()
        self.gates[label] = (entered, release)
        return release

    async def wait_entered(self, label: str, timeout: float = 5.0) -> None:
        """Block until *label*'s node has actually reached its hold point.

        Runs the wait in a THREAD: this is awaited from the test's own
        coroutine, which runs on the event loop, and the node's dispatch
        chain needs that same loop to keep turning to reach the gate at all.
        A direct Event.wait would deadlock against the thing it waits for.
        """
        entered, _release = self.gates[label]
        if not await asyncio.to_thread(entered.wait, timeout):
            raise AssertionError(
                f"{label!r} never reached its hold point within {timeout}s")


_probe = _Probe()


class _SweepProbeNode(BaseNode):
    """Logs ``val_loss = lr * 10 + weight_decay`` — a KNOWN function of the
    swept params, which is what lets a ranking test assert the exact winner
    instead of "some ordering happened".

    The shipped examples cannot stand in for this (spec 6.5): only two wire
    a val_dataloader at all and both are heavyweight.
    """

    NODE_NAME = "_SweepProbe"
    CATEGORY = "Test"
    DESCRIPTION = "Logs val_loss derived from its swept params"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="lr", param_type=ParamType.FLOAT,
                            default=0.001, min_value=0.0, max_value=1.0),
            ParamDefinition(name="weight_decay", param_type=ParamType.FLOAT,
                            default=0.0),
            ParamDefinition(name="silent", param_type=ParamType.BOOL,
                            default=False),
            ParamDefinition(name="diverge", param_type=ParamType.BOOL,
                            default=False),
            ParamDefinition(name="fail", param_type=ParamType.BOOL,
                            default=False),
            # A free-text param, so the CSV test can sweep a value that
            # LOOKS LIKE A FORMULA. An explicitly enumerated values list on
            # a string param is a legitimate domain, not a search over
            # strings (spec 3.4.1).
            ParamDefinition(name="note", param_type=ParamType.STRING,
                            default=""),
            ParamDefinition(name="api_key", param_type=ParamType.SECRET,
                            default=""),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *,
                context: Any = None) -> dict[str, Any]:
        # The engine injects the ExecutionContext when and only when the
        # parameter is NAMED `context` (graph_engine.py:1576-1577).
        lr = float(params.get("lr", 0.001))
        weight_decay = float(params.get("weight_decay", 0.0))
        label = _label(lr, weight_decay)
        _probe.enter(label)
        gate = _probe.gates.get(label)
        if gate is not None:
            entered, release = gate
            entered.set()
            # Bounded as a safety net only: a test that forgets to release
            # should fail loudly, not hang the suite.
            if not release.wait(timeout=10.0):
                raise TimeoutError(f"probe hold for {label!r} never released")
        if context is not None and not params.get("silent"):
            value = (float("nan") if params.get("diverge")
                     else lr * 10 + weight_decay)
            context.log_metric("val_loss", value, step=0)
            context.log_metric("train_loss", value * 2, step=0)
        if params.get("fail"):
            # AFTER the log_metric above: a failed variant that DID log the
            # objective must still be rankable (AC 3).
            raise RuntimeError("probe failed on purpose")
        return {"value": inputs.get("value")}


def _label(lr: float, weight_decay: float = 0.0) -> str:
    """The gate key. Distinct per variant because a sweep's whole point is
    that its variants differ in exactly these numbers."""
    return f"lr={lr},wd={weight_decay}"


_TEST_NODES = {"_SweepProbe": _SweepProbeNode}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


@pytest.fixture(autouse=True)
def probe():
    """A clean probe per test. Returned so a test can read it directly."""
    global _probe
    _probe = _Probe()
    return _probe


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    """Every device string resolves to itself.

    resolve_device degrades cuda/mps to cpu on a machine without them, which
    on CI collapses every queue into one and makes the FIFO assertions
    vacuous. Patching is honest here: the scheduler only ever handles the
    result as an opaque string.
    """
    monkeypatch.setattr(
        run_service_module, "resolve_device",
        lambda requested: (requested or "cpu").strip().lower() or "cpu")
    monkeypatch.setattr(run_service_module, "_current_cuda_index",
                        lambda: 0)


# ── fixtures ──────────────────────────────────────────────────────────────


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


@pytest.fixture
def sweep_store(db):
    return SweepStore(db)


@pytest.fixture
async def make_service(store, db, sweep_store):
    """Build services with explicit limits, wire app.state, drain them all.

    Limits are passed in rather than read from settings so a test states the
    policy it is testing. The lifespan does not run under ASGITransport, so
    app.state is set here (main.py:326-328 is the precedent) and torn down
    in the REAL order: drain the service, then close the database.
    """
    built: list[RunService] = []

    def _make(*, cpu: int = 2, gpu: int = 1, interactive: int = 2,
              shutdown_grace_s: float = 5.0, **kwargs: Any) -> RunService:
        service = RunService(
            store, shutdown_grace_s=shutdown_grace_s,
            limits=QueueLimits(cpu=cpu, gpu=gpu, interactive=interactive,
                               overrides=()),
            **kwargs)
        built.append(service)
        app.state.db = db
        app.state.run_service = service
        app.state.sweep_store = sweep_store
        return service

    try:
        yield _make
    finally:
        for service in built:
            await service.shutdown()
        for attribute in ("db", "run_service", "sweep_store"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


@pytest.fixture
async def service(make_service):
    return make_service()


@pytest.fixture
async def client(service):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


# ── helpers ───────────────────────────────────────────────────────────────


def _graph(**params: Any) -> dict[str, Any]:
    """Start --trigger--> _TestSource --value--> _SweepProbe."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_TestSource",
             "data": {"params": {"val": "hi"}}},
            {"id": "probe", "type": "_SweepProbe", "data": {"params": params}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "probe",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
    }


def _values(param: str, values: list[Any]) -> dict[str, Any]:
    return {"node_id": "probe", "param": param, "values": values}


def _body(*params: dict[str, Any], graph: dict[str, Any] | None = None,
          **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "base_graph": graph if graph is not None else _graph(),
        "sweep_spec": {"method": "grid", "params": list(params)},
        "objective": {"metric": "val_loss", "direction": "minimize"},
        "options": {"device": "cpu"},
        "name": "lr sweep",
    }
    body.update(overrides)
    return body


async def _await_sweep_state(sweep_store: SweepStore, sweep_id: str,
                             state: str, *, timeout: float = 5.0):
    """Wait for a DETACHED settle to land.

    `_mark_failed_on_the_way_out` fires its write as its own task, precisely
    so a cancelled request cannot take the write with it -- which means the
    row settles a tick or two after the handler has already unwound. A
    bounded wait on an observed fact, never a sleep.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await sweep_store.get_sweep(sweep_id)
        if record is not None and record.state == state:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"sweep {sweep_id} never reached state {state!r} within {timeout}s")


async def _await_terminal(store: RunStore, run_id: str, *,
                          timeout: float = 15.0):
    """Poll the STORE (never the in-process registry) until the row is
    done."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        assert record is not None, f"run {run_id} vanished"
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# ── auth / openapi surface (spec 9.6) ─────────────────────────────────────


def test_sweeps_routes_are_not_under_an_auth_exempt_prefix():
    """/api/sweeps relies on auth_guard, so it must NOT be prefix-exempt —
    otherwise its two POSTs would sail through with no auth at all. The
    exemption exists for /api/apps and /api/keys, which carry per-route
    dependencies instead."""
    from app.main import _prefix_exempt

    for path in ("/api/sweeps", "/api/sweeps/abc", "/api/sweeps/abc/cancel"):
        assert not _prefix_exempt(path), path


def test_sweeps_routes_appear_in_the_openapi_document():
    # `/api/sweeps/{sweep_id}/cancel` is asserted by the task that adds it;
    # a route this module does not serve yet cannot be pinned here without
    # leaving the tree red.
    paths = app.openapi()["paths"]
    assert "/api/sweeps" in paths
    assert set(paths["/api/sweeps"]) == {"post"}
    assert "/api/sweeps/{sweep_id}" in paths
    assert set(paths["/api/sweeps/{sweep_id}"]) == {"get"}


async def test_a_mutating_sweep_route_without_a_token_is_403(service):
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as http:
        created = await http.post("/api/sweeps", json=_body(
            _values("lr", [0.1])))
        assert created.status_code == 403
        assert TOKEN_HEADER in created.json()["detail"]
        # auth_guard runs as middleware, ahead of routing, so every mutating
        # /api/sweeps path is covered by the same rule whether or not it has
        # a handler yet.
        cancelled = await http.post("/api/sweeps/abc/cancel")
        assert cancelled.status_code == 403
        # The GET is open, like every other GET in the app. The DETAIL is
        # asserted, not just the code: an unauthenticated GET must reach the
        # handler and be told the sweep does not exist, where a routing 404
        # would look identical from the outside.
        unknown = await http.get("/api/sweeps/abc")
        assert unknown.status_code == 404
        assert unknown.json()["detail"] == "sweep 'abc' not found"


# ── submit (spec 5.2, 9.7) ────────────────────────────────────────────────


async def test_a_grid_sweep_queues_every_variant_in_fifo_order(
        make_service, store, probe):
    """3x2 -> 6 rows, every one carrying the sweep's id and its own variant
    index, started in submission order. #140 acceptance criterion 1."""
    service = make_service(cpu=1)          # one slot: start order IS FIFO
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        response = await http.post("/api/sweeps", json=_body(
            _values("lr", [0.001, 0.002, 0.003]),
            _values("weight_decay", [0.0, 0.5])))
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["total_combinations"] == 6
        assert len(body["variants"]) == 6
        assert [v["index"] for v in body["variants"]] == [0, 1, 2, 3, 4, 5]

        children = await store.list_runs_by_sweep(body["sweep_id"])
        assert [c.sweep_variant for c in children] == [0, 1, 2, 3, 4, 5]
        assert {c.sweep_id for c in children} == {body["sweep_id"]}

        for child in children:
            await _await_terminal(store, child.id)
    # The LAST-declared param varies fastest, so start order is the
    # compiled order.
    assert probe.started == [
        _label(0.001, 0.0), _label(0.001, 0.5),
        _label(0.002, 0.0), _label(0.002, 0.5),
        _label(0.003, 0.0), _label(0.003, 0.5),
    ]
    assert service is not None


async def test_the_route_captures_provenance_once_for_the_whole_sweep(
        client, store, monkeypatch):
    """ONE capture for all N variants (spec 4.5).

    The counter sees BOTH call sites, which is what makes this a real
    guard: the route's own `asyncio.to_thread(RunProvenance.capture)` and
    `RunStore.create_run`'s default, which captures per run when nothing is
    passed. So a route that forgot to capture, or captured inside the loop,
    reads 6 here rather than 1 -- and the sentinel on every child row is
    what stops "captured once" from also meaning "captured once and then
    dropped on the floor".
    """
    captures: list[int] = []
    sentinel = RunProvenance(git_commit="sweep-sentinel", git_dirty=False,
                             plugin_pins={})

    def _capture(*_args: Any, **_kwargs: Any) -> RunProvenance:
        captures.append(1)
        return sentinel

    monkeypatch.setattr(RunProvenance, "capture", _capture)
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.001, 0.002, 0.003]),
        _values("weight_decay", [0.0, 0.5])))
    assert response.status_code == 201, response.text
    assert len(captures) == 1, f"expected one capture, got {len(captures)}"
    children = await store.list_runs_by_sweep(response.json()["sweep_id"])
    assert len(children) == 6
    assert {c.git_commit for c in children} == {"sweep-sentinel"}


def _raise_on_nth(real: Any, *, n: int,
                  error: BaseException | None = None) -> Any:
    """Wrap an async method so its *n*th call raises instead of running.

    Calls before *n* run the REAL method, so they land genuine rows -- what
    both callers are about is the work that already succeeded.
    """
    calls = 0

    async def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == n:
            raise (error if error is not None else
                   RunServiceUnavailable("run service is shutting down"))
        return await real(self, *args, **kwargs)

    return _wrapped


async def test_the_submit_loop_failing_part_way_leaves_a_failed_sweep_and_intact_children(
        make_service, store, sweep_store, db, probe, monkeypatch):
    """Spec 9.7.2, in full.

    A sweep leaves `running` only if every variant got a run_id or the
    route marked it failed (spec 4.3, 5.2). A variant with run_id null is
    NOT terminal, so a loop that breaks and does neither would leave the row
    at `running` forever, with no seam able to settle it. Nothing else fails
    if an implementer forgets the row patch or the `k of n` suffix, both of
    which are pure design.

    The queue is saturated first, or the assertion that the two created
    children are still `queued` could never pass: the default cpu cap is 2
    and `submit` pumps immediately, so both would already be running.
    """
    service = make_service(cpu=1)             # one slot on the cpu queue
    release_filler = probe.hold(_label(0.9))
    await service.submit(_graph(lr=0.9), options={"device": "cpu"})
    await probe.wait_entered(_label(0.9))     # the slot is provably occupied
    # Only NOW install the failure, so the filler's own submit is not
    # counted: calls 1 and 2 are variants 0 and 1, call 3 raises.
    monkeypatch.setattr(RunService, "submit",
                        _raise_on_nth(RunService.submit, n=3))
    try:
        async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL,
                headers={TOKEN_HEADER: session_token()}) as http:
            response = await http.post("/api/sweeps", json=_body(
                _values("lr", [0.001, 0.002, 0.003]),
                _values("weight_decay", [0.0, 0.5])))

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert "2 of 6 variants" in detail

        rows = db._conn.execute("SELECT id FROM sweeps").fetchall()
        assert len(rows) == 1
        sweep_id = rows[0][0]
        # The id is in the body, so a partial sweep is never unfindable.
        assert sweep_id in detail

        record = await sweep_store.get_sweep(sweep_id)
        assert record is not None
        assert record.state == "failed"      # NOT still "running"
        assert record.error
        assert record.finished_at is not None
        assert [v.run_id is not None for v in record.variants] == [
            True, True, False, False, False, False]

        # Nothing unwinds or deletes the children that were created.
        children = await store.list_runs_by_sweep(sweep_id)
        assert [c.sweep_variant for c in children] == [0, 1]
        assert [c.status for c in children] == ["queued", "queued"]

        # §9.7.2's last assertion: the partial sweep is READABLE, and the
        # four variants that never got a run report `missing` rather than
        # vanishing from the table.
        async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL,
                headers={TOKEN_HEADER: session_token()}) as http:
            readback = await http.get(f"/api/sweeps/{sweep_id}")
        assert readback.status_code == 200, readback.text
        body = readback.json()
        assert len(body["variants"]) == 6
        assert body["counts"]["missing"] == 4
        assert body["state"] == "failed"      # never overwritten by a harvest
    finally:
        # Held gates make the fixture's drain wait out shutdown_grace_s.
        release_filler.set()


async def test_a_store_failure_while_patching_a_run_id_still_fails_the_sweep(
        client, store, sweep_store, db, monkeypatch):
    """The patch is the loop's THIRD exit, and it needs the same duty.

    `set_variant_run` runs AFTER its variant's child was created, so a store
    fault there used to leave the loop having neither filled in every run id
    nor marked the sweep failed -- and a run_id-None variant is not terminal
    (spec 4.3), so the row sat at `running` with no seam able to settle it.
    A transient fault is real here: one write can fail and the next succeed.
    """
    monkeypatch.setattr(
        SweepStore, "set_variant_run",
        _raise_on_nth(SweepStore.set_variant_run, n=2,
                      error=sqlite3.OperationalError("database is locked")))
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.001, 0.002, 0.003])))
    assert response.status_code == 500, response.text   # not RunService-shaped
    detail = response.json()["detail"]
    assert "1 of 3 variants" in detail

    rows = db._conn.execute("SELECT id FROM sweeps").fetchall()
    assert len(rows) == 1
    sweep_id = rows[0][0]
    assert sweep_id in detail

    record = await sweep_store.get_sweep(sweep_id)
    assert record is not None
    assert record.state == "failed"          # NOT still "running"
    assert record.error
    assert record.finished_at is not None
    # Variant 1's PATCH is what failed, so the row under-reports it ...
    assert [v.run_id is not None for v in record.variants] == [
        True, False, False]
    # ... while its child still exists and stays findable by sweep_id.
    children = await store.list_runs_by_sweep(sweep_id)
    assert [c.sweep_variant for c in children] == [0, 1]


async def test_seed_variants_off_leaves_every_variant_unseeded(client, store):
    """RULING 1: an unseeded run takes the SHARED exclusion and overlaps up
    to the device limit. A seeded one takes a process-wide exclusive lock."""
    response = await client.post("/api/sweeps",
                                 json=_body(_values("lr", [0.1, 0.2])))
    assert response.status_code == 201, response.text
    body = response.json()
    assert all(v["seed"] is None for v in body["variants"])
    for child in await store.list_runs_by_sweep(body["sweep_id"]):
        assert child.options["seed"] is None


async def test_seed_variants_on_derives_seed_plus_index_mod_2_32(
        client, store):
    """The modulo is mandatory: normalize_options rejects a seed above
    MAX_SEED, so a base seed near the top of the range would make late
    variants fail submission for a reason the caller never wrote."""
    # sweep_spec is passed as an override rather than positionally, because
    # it needs a `seed` and _body's positional form does not carry one.
    response = await client.post("/api/sweeps", json=_body(
        seed_variants=True,
        sweep_spec={"method": "grid", "seed": 2 ** 32 - 3,
                    "params": [_values("lr", [0.1, 0.2, 0.3, 0.4, 0.5])]}))
    assert response.status_code == 201, response.text
    body = response.json()
    assert [v["seed"] for v in body["variants"]] == [
        4294967293, 4294967294, 4294967295, 0, 1]
    children = await store.list_runs_by_sweep(body["sweep_id"])
    assert [c.options["seed"] for c in children] == [
        4294967293, 4294967294, 4294967295, 0, 1]


async def test_options_seed_is_refused(client, db):
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.1]), options={"device": "cpu", "seed": 7}))
    assert response.status_code == 400
    assert "seed_variants" in response.json()["detail"]
    assert db._conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0


async def test_the_interactive_lane_is_refused(client, db):
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.1]),
        options={"device": "cpu", "lane": "interactive"}))
    assert response.status_code == 400
    assert "interactive lane" in response.json()["detail"]
    assert db._conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0


async def test_a_padded_interactive_lane_is_refused_before_any_row(client, db):
    """normalize_options STRIPS `lane`, so comparing the RAW string lets
    " interactive " walk past this rule, normalise INTO the interactive lane
    and be refused only by submit's own guard inside the loop -- a 500 plus a
    permanent `failed` sweeps row, where the rule promises a 400 and no rows
    at all. Nothing in v1 deletes a sweeps row, so that one is forever.
    """
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.1]),
        options={"device": "cpu", "lane": " interactive "}))
    assert response.status_code == 400, response.text
    assert "interactive lane" in response.json()["detail"]
    assert db._conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0
    assert db._conn.execute(
        "SELECT COUNT(*) FROM exec_runs").fetchone()[0] == 0


async def test_an_unrecognised_lane_label_queues_instead_of_being_refused(
        client, store):
    """The rule strips but must NOT lowercase, because normalize_options does
    not lowercase either: an unrecognised lane label QUEUES rather than
    bypassing (run_service.py:242-244), so "Interactive" is an ordinary
    queued sweep and refusing it would refuse a legal one. Here to stop the
    strip from being "fixed" into a casefold.
    """
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.1]),
        options={"device": "cpu", "lane": "Interactive"}))
    assert response.status_code == 201, response.text
    children = await store.list_runs_by_sweep(response.json()["sweep_id"])
    assert [c.options["lane"] for c in children] == ["Interactive"]


async def test_seed_variants_without_a_seed_is_refused(client, db):
    """The route's rule, not the compiler's: a grid never needs a planner
    seed, so nothing below this line would refuse it."""
    response = await client.post("/api/sweeps", json=_body(
        _values("lr", [0.1, 0.2]), seed_variants=True))
    assert response.status_code == 400
    assert "sweep_spec.seed is required" in response.json()["detail"]
    assert db._conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0


async def test_record_outputs_over_the_output_store_bound_is_refused(
        client, db):
    from app.core.run_output_store import RunOutputStore

    # Set and removed by hand rather than with monkeypatch, the way
    # `make_service` does it: `app.state` is a starlette State, whose
    # __delattr__ raises KeyError (not AttributeError) on a missing key, so
    # monkeypatch's undo of a raising=False setattr dies in teardown.
    app.state.run_output_store = RunOutputStore(max_runs=4, max_bytes=1024)
    try:
        response = await client.post("/api/sweeps", json=_body(
            _values("lr", [0.1, 0.2, 0.3, 0.4, 0.5]),
            options={"device": "cpu", "record_outputs": True}))
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "5 variants" in detail and "newest 4 runs" in detail
        assert db._conn.execute(
            "SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0
    finally:
        if hasattr(app.state, "run_output_store"):
            delattr(app.state, "run_output_store")


async def test_non_finite_values_are_refused(client, db):
    """json.loads("NaN") is nan and json.loads("1e999") is inf, and both
    pass every other check before corrupting a stored row. The body is sent
    as raw text so the literals survive to the parser."""
    body = _body(_values("lr", [0.1]))
    raw = (
        '{"base_graph": %s, "objective": {"metric": "val_loss", '
        '"direction": "minimize"}, "options": {"device": "cpu"}, '
        '"sweep_spec": {"method": "grid", "params": [{"node_id": "probe", '
        '"param": "lr", "values": [0.1, NaN]}]}}'
        % __import__("json").dumps(body["base_graph"]))
    response = await client.post(
        "/api/sweeps", content=raw.encode(),
        headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "finite" in response.json()["detail"]
    assert db._conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0] == 0


async def test_a_cancelled_submit_loop_still_settles_the_sweep(
        make_service, store, sweep_store, db, probe, monkeypatch):
    """A browser tab closing mid-POST cancels the request task, landing a
    CancelledError between two submits. `except Exception` cannot see a
    BaseException, so without the loop's `finally` the row keeps `running`
    forever: a variant holding a null run id is not terminal, so no harvest
    seam can ever settle it.

    `failed` is the spec's own end state for exactly this (4.3: "the submit
    loop broke part-way and some variants have no run at all"), and the
    children already submitted keep running and stay inspectable (RULING 2).
    """
    service = make_service(cpu=1)
    release_filler = probe.hold(_label(0.9))
    await service.submit(_graph(lr=0.9), options={"device": "cpu"})
    await probe.wait_entered(_label(0.9))

    reached_third = asyncio.Event()
    let_go = asyncio.Event()
    real_submit = RunService.submit
    calls = 0

    async def _hang_on_the_third(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            # A gate, not a sleep: the cancel lands provably INSIDE the loop,
            # with variants 0 and 1 already submitted and patched.
            reached_third.set()
            await let_go.wait()
        return await real_submit(self, *args, **kwargs)

    monkeypatch.setattr(RunService, "submit", _hang_on_the_third)
    try:
        async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL,
                headers={TOKEN_HEADER: session_token()}) as http:
            posting = asyncio.ensure_future(http.post(
                "/api/sweeps", json=_body(
                    _values("lr", [0.001, 0.002, 0.003]),
                    _values("weight_decay", [0.0, 0.5]))))
            await asyncio.wait_for(reached_third.wait(), timeout=5.0)
            posting.cancel()
            # The cancellation must still propagate: the settle is a side
            # effect on the way out, not a recovery.
            with pytest.raises(asyncio.CancelledError):
                await posting

        rows = db._conn.execute("SELECT id FROM sweeps").fetchall()
        assert len(rows) == 1
        sweep_id = rows[0][0]
        record = await _await_sweep_state(sweep_store, sweep_id, "failed")
        assert record.error and "CancelledError" in record.error
        assert record.finished_at is not None
        assert [v.run_id is not None for v in record.variants] == [
            True, True, False, False, False, False]

        # The two children already submitted are untouched and still theirs.
        children = await store.list_runs_by_sweep(sweep_id)
        assert [c.sweep_variant for c in children] == [0, 1]
        assert all(child.status in {"queued", "running"}
                   for child in children), [c.status for c in children]
    finally:
        let_go.set()
        release_filler.set()


async def test_a_cancel_while_the_sweep_row_is_written_still_settles_it(
        client, sweep_store, db, monkeypatch):
    """The one window outside the loop, closed by the same guard.

    `Database.run` finishes its worker before re-raising a cancellation
    (db.py:225-243), so a cancel delivered at `create_sweep`'s own await
    leaves the row WRITTEN while this handler unwinds straight past it,
    never holding the record. The route therefore mints the sweep id itself,
    so the `finally` knows which row to settle even then.
    """
    real_create = SweepStore.create_sweep
    written = asyncio.Event()
    let_go = asyncio.Event()

    async def _write_then_hang(self: Any, **kwargs: Any) -> Any:
        record = await real_create(self, **kwargs)
        written.set()
        await let_go.wait()      # the cancel lands here, row already in
        return record

    monkeypatch.setattr(SweepStore, "create_sweep", _write_then_hang)
    try:
        posting = asyncio.ensure_future(client.post(
            "/api/sweeps", json=_body(_values("lr", [0.1, 0.2]))))
        await asyncio.wait_for(written.wait(), timeout=5.0)
        posting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await posting

        rows = db._conn.execute("SELECT id FROM sweeps").fetchall()
        assert len(rows) == 1
        record = await _await_sweep_state(sweep_store, rows[0][0], "failed")
        assert record.finished_at is not None
        assert [v.run_id for v in record.variants] == [None, None]
        # Nothing was submitted, so nothing is left running either.
        assert db._conn.execute(
            "SELECT COUNT(*) FROM exec_runs").fetchone()[0] == 0
    finally:
        let_go.set()


# ── the ranked table (spec 5.3, 9.7) ──────────────────────────────────────


async def _run_sweep(http: AsyncClient, store: RunStore,
                     *params: dict[str, Any], **overrides: Any) -> str:
    """POST a sweep and wait for every child to reach a terminal status."""
    response = await http.post("/api/sweeps", json=_body(*params,
                                                         **overrides))
    assert response.status_code == 201, response.text
    sweep_id = response.json()["sweep_id"]
    for child in await store.list_runs_by_sweep(sweep_id):
        await _await_terminal(store, child.id)
    return sweep_id


async def test_every_variant_completes_unattended_and_the_table_ranks_them(
        client, store):
    """#140 acceptance criterion 1: 3x2 completes unattended and the table
    ranks by final val_loss. val_loss = lr * 10 + weight_decay, so the
    winner is knowable in advance rather than merely "some ordering"."""
    sweep_id = await _run_sweep(client, store,
                                _values("lr", [0.003, 0.001, 0.002]),
                                _values("weight_decay", [0.5, 0.0]))
    body = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert body["state"] == "finished"
    assert body["counts"]["succeeded"] == 6
    assert [v["rank"] for v in body["variants"]] == [1, 2, 3, 4, 5, 6]
    assert body["variants"][0]["objective"] == pytest.approx(0.01)
    assert body["variants"][0]["params"] == [
        {"node_id": "probe", "param": "lr", "value": 0.001},
        {"node_id": "probe", "param": "weight_decay", "value": 0.0}]
    assert body["best"]["index"] == body["variants"][0]["index"]
    assert body["variants"][0]["final_metrics"]["train_loss"] == \
        pytest.approx(0.02)
    assert all(v["run_exists"] for v in body["variants"])
    assert "objective_warning" not in body
    # The pre-rank identity survives the sort, so a client can put the table
    # back into submission order (spec 5.3).
    assert sorted(v["index"] for v in body["variants"]) == [0, 1, 2, 3, 4, 5]


async def test_the_ranked_order_is_stable_across_polls(client, store):
    """A polled table must not reshuffle between reads.

    Two variants are given the SAME objective on purpose -- val_loss is
    lr * 10 + weight_decay, so (0.1, 0.0) and (0.05, 0.5) both land on 1.0 --
    because a tie is the only case where a sort with no total order can
    legally return either. Ties break on `index`, so the two reads below are
    identical rather than merely sorted.
    """
    sweep_id = await _run_sweep(client, store,
                                _values("lr", [0.1, 0.05]),
                                _values("weight_decay", [0.0, 0.5]))
    first = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    second = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert [v["index"] for v in first["variants"]] == \
        [v["index"] for v in second["variants"]]
    tied = [v["index"] for v in first["variants"]
            if v["objective"] == pytest.approx(1.0)]
    assert tied == sorted(tied) and len(tied) == 2


async def test_a_failed_variant_that_logged_the_objective_is_still_ranked(
        client, store):
    """The HTTP sibling of test_sweeps.py's unit version. AC 3: hiding a
    real number because the run ended badly is the silent disappearance
    that criterion forbids."""
    sweep_id = await _run_sweep(
        client, store, _values("lr", [0.1]),
        graph=_graph(fail=True))
    body = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    variant = body["variants"][0]
    assert variant["status"] == "failed"
    assert variant["objective"] == pytest.approx(1.0)
    assert variant["rank"] == 1
    assert body["counts"]["failed"] == 1
    assert body["state"] == "finished"      # a container, not a run


async def test_a_variant_with_no_objective_is_unranked_but_present(
        client, store):
    sweep_id = await _run_sweep(
        client, store, _values("lr", [0.1]), graph=_graph(silent=True))
    body = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert body["variants"][0]["objective"] is None
    assert body["variants"][0]["rank"] is None
    assert body["best"] is None
    csv_body = (await client.get(
        f"/api/sweeps/{sweep_id}?format=csv")).text
    rows = list(csv.reader(io.StringIO(csv_body.lstrip(_CSV_BOM))))
    assert len(rows) == 2                    # header + the unranked row


async def test_a_diverged_variant_is_unranked(client, store):
    """A non-finite last point is stored as SQL NULL and omitted from the
    series map: a diverged loss must not render as a suspiciously good
    one. The HTTP sibling of test_sweeps.py's unit version."""
    sweep_id = await _run_sweep(
        client, store, _values("lr", [0.1]), graph=_graph(diverge=True))
    body = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert body["variants"][0]["status"] == "succeeded"
    assert body["variants"][0]["objective"] is None
    assert body["variants"][0]["rank"] is None


async def test_no_variant_produced_the_objective(client, store):
    """The single most likely user error -- asking for val_loss from a
    graph that never logs it -- becomes a message naming the fix instead of
    a table of empty cells."""
    sweep_id = await _run_sweep(
        client, store, _values("lr", [0.1, 0.2]),
        objective={"metric": "accuracy", "direction": "maximize"})
    body = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert body["best"] is None
    assert all(v["rank"] is None for v in body["variants"])
    warning = body["objective_warning"]
    assert "accuracy" in warning
    assert "train_loss" in warning and "val_loss" in warning


async def test_the_objective_survives_its_child_being_pruned(client, store):
    """RULING 4: the run id stays as a link that may be dead, and the
    numbers stay."""
    sweep_id = await _run_sweep(client, store, _values("lr", [0.1, 0.2]))
    before = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert before["variants"][0]["objective"] == pytest.approx(1.0)

    assert await store.prune(keep_last=0) == 2
    after = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert after["variants"][0]["objective"] == pytest.approx(1.0)
    assert after["variants"][0]["status"] == "succeeded"   # the harvested one
    assert after["variants"][0]["run_exists"] is False
    assert "final_metrics" not in after["variants"][0]
    assert after["variants"][0]["run_id"] is not None      # a dead link


async def test_a_child_deleted_before_any_read_is_missing_not_a_wedged_sweep(
        client, store, sweep_store):
    """Spec 6.3's accepted hole, reported honestly rather than hidden.

    `DELETE /api/runs/{id}` goes through `delete_run`, not `prune`, so seam
    B's inside-the-transaction guarantee does not cover it: a child deleted
    before any GET harvested it loses its objective permanently. What must
    NOT happen is the sweep wedging at `running` forever with no seam able
    to settle it -- which is exactly what the "row is gone" clause of
    `variant_is_terminal` prevents, and this is the only place it fires.
    """
    sweep_id = await _run_sweep(client, store, _values("lr", [0.1]))
    child = (await store.list_runs_by_sweep(sweep_id))[0]
    # No GET yet, so nothing has been harvested.
    assert (await sweep_store.get_sweep(sweep_id)).variants[0].status is None
    assert await store.delete_run(child.id) is True

    body = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    variant = body["variants"][0]
    assert variant["status"] == "missing"
    assert variant["objective"] is None
    assert variant["rank"] is None
    assert variant["run_exists"] is False
    assert "final_metrics" not in variant
    assert body["counts"]["missing"] == 1
    assert body["state"] == "finished"      # NOT wedged at "running"
    assert (await sweep_store.get_sweep(sweep_id)).state == "finished"


async def test_the_sweep_finishes_through_seam_a(client, store, sweep_store):
    sweep_id = await _run_sweep(client, store, _values("lr", [0.1, 0.2]))
    first = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert first["state"] == "finished"
    assert first["finished_at"] is not None
    # The STORED row, not just the response: read repair means nothing else
    # would ever write it.
    stored = await sweep_store.get_sweep(sweep_id)
    assert stored.state == "finished"
    assert stored.finished_at == first["finished_at"]

    second = (await client.get(f"/api/sweeps/{sweep_id}")).json()
    assert second["finished_at"] == first["finished_at"]   # stamped once


async def test_a_restart_leaves_variants_interrupted_and_the_sweep_inspectable(
        make_service, store, probe):
    """RULING 2. `recover_interrupted` is a STARTUP call and raises when
    runs are in flight, so this uses spec 9.7.1 recipe (a): a graceful
    shutdown, whose phase 0 files every WAITING run `interrupted` -- the
    same status and the same rows a restart would write.
    """
    service = make_service(cpu=1)
    release = probe.hold(_label(0.1))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        response = await http.post("/api/sweeps", json=_body(
            _values("lr", [0.1, 0.2, 0.3, 0.4])))
        sweep_id = response.json()["sweep_id"]
        await probe.wait_entered(_label(0.1))

        # Release the RUNNING child, then shut down with NO await in
        # between. The probe blocks on a threading.Event, not on a
        # context.should_stop() poll, so shutdown's cooperative phase cannot
        # reach a HELD node: it would sit there until phase 2 hard-cancels
        # it after shutdown_grace_s, and shutdown's own docstring says that
        # leaves the row reading `running`.
        #
        # The missing `await` between the two lines is the load-bearing
        # part, not a shortcut. `shutdown` sets `_shutting_down` and empties
        # both pending indexes BEFORE its first await
        # (run_service.py:2637, :2674-2680), and entering a coroutine does
        # not yield -- so the event loop cannot promote a queued child into
        # the slot the released one just freed. Awaiting the released child
        # to terminal first, as the obvious reading of the recipe suggests,
        # hands the loop exactly that chance and makes "three children never
        # started" a race instead of a fact.
        release.set()
        await service.shutdown()

        statuses = {c.sweep_variant: c.status
                    for c in await store.list_runs_by_sweep(sweep_id)}
        # Variant 0 was drained by shutdown's phase 1, so it is terminal --
        # `succeeded` normally, `interrupted` if the stop reason won the
        # race with the node's own return. Either is a real restart outcome;
        # what RULING 2 is about is the three that never started.
        assert statuses[0] in TERMINAL_STATUSES, statuses
        assert all(statuses[i] == "interrupted" for i in (1, 2, 3)), statuses

        body = (await http.get(f"/api/sweeps/{sweep_id}")).json()
        assert len(body["variants"]) == 4
        assert sum(body["counts"].values()) == 4
        assert body["counts"]["interrupted"] >= 3
        assert body["state"] == "finished"     # every variant is terminal
        # The harvest keeps whatever a stopped child had logged: variant 0
        # logged val_loss before the gate released it, and _finalize flushes
        # metrics with force=True whatever the outcome.
        by_index = {v["index"]: v for v in body["variants"]}
        assert by_index[0]["objective"] == pytest.approx(1.0)
        assert all(by_index[i]["objective"] is None for i in (1, 2, 3))


# ── the comparison CSV (spec 5.3, 9.8) ────────────────────────────────────


async def test_the_comparison_csv_has_one_row_per_variant(client, store):
    """#140 acceptance criterion 4. `note` is swept with a value that looks
    like a FORMULA, because Excel, LibreOffice and Sheets all evaluate a
    cell whose first character is '=' and a string param's value is
    caller-supplied."""
    sweep_id = await _run_sweep(
        client, store,
        _values("lr", [0.1, 0.2]),
        _values("note", ["=HYPERLINK(1)", "ok"]),
        graph=_graph(silent=True))          # unranked rows, on purpose
    response = await client.get(f"/api/sweeps/{sweep_id}?format=csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert f"sweep-{sweep_id}-comparison.csv" in disposition

    text = response.text
    # U+FEFF: text/csv; charset=utf-8 is not enough for Excel on Windows,
    # which reads a BOM-less CSV in the ANSI codepage -- in a product that
    # ships zh-TW.
    assert text.startswith(_CSV_BOM)
    rows = list(csv.reader(io.StringIO(text.lstrip(_CSV_BOM))))
    assert rows[0] == ["rank", "variant_index", "domain_index", "run_id",
                       "status", "objective", "probe.lr", "probe.note"]
    assert len(rows) == 5                    # header + 4 variants
    # Every variant is silent, so every rank and objective cell is EMPTY --
    # not "None", which would read as text and poison the column's type --
    # and no row disappears for being unranked.
    assert all(row[0] == "" and row[5] == "" for row in rows[1:])
    assert sorted(row[1] for row in rows[1:]) == ["0", "1", "2", "3"]
    assert all(row[3] and row[4] == "succeeded" for row in rows[1:])
    # The formula guard: a leading apostrophe, the convention every
    # spreadsheet understands as "this is text".
    assert {row[7] for row in rows[1:]} == {"'=HYPERLINK(1)", "ok"}
    # Numeric cells are NOT quoted into text: a chart built on the export
    # would break.
    assert {row[6] for row in rows[1:]} == {"0.1", "0.2"}


async def test_an_unknown_sweep_is_a_404(client):
    response = await client.get("/api/sweeps/nope")
    assert response.status_code == 404
    assert response.json()["detail"] == "sweep 'nope' not found"


async def test_an_unknown_format_is_a_422(client, store):
    sweep_id = await _run_sweep(client, store, _values("lr", [0.1]))
    assert (await client.get(
        f"/api/sweeps/{sweep_id}?format=xlsx")).status_code == 422
