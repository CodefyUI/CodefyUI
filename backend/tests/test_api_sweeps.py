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
import sqlite3
import threading
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

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
from app.core.run_store import RunProvenance, RunStore
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
    # `/api/sweeps/{sweep_id}` and `/api/sweeps/{sweep_id}/cancel` are
    # asserted by the tasks that add them; a route this module does not
    # serve yet cannot be pinned here without leaving the tree red.
    paths = app.openapi()["paths"]
    assert "/api/sweeps" in paths
    assert set(paths["/api/sweeps"]) == {"post"}


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
        # The GET is open, like every other GET in the app.
        assert (await http.get("/api/sweeps/abc")).status_code == 404


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


async def test_the_submit_loop_failing_part_way_leaves_a_failed_sweep(
        make_service, store, sweep_store, db, probe, monkeypatch):
    """A sweep leaves `running` only if every variant got a run_id or the
    route marked it failed (spec 4.3, 5.2). A variant with run_id null is
    NOT terminal, so a loop that breaks and does neither would leave the row
    at `running` forever, with no seam able to settle it.

    §9.7.2's last assertion is a GET, which arrives with that route; the
    rest of it is here, where the failure path is written.

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
