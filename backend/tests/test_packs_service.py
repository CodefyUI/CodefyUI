"""The Package Center's job runner, without a server and without an install.

``PackService`` is the only thing standing between a synchronous, minutes-long
install and an event loop that must stay responsive while it runs. Everything
below is about that seam:

* the flow runs on a worker thread and its events arrive on the loop, stamped
  with a cursor a poller can resume from;
* exactly one terminal event per job, and the four failure shapes of
  ``packs.errors`` map onto four different terminal events -- a cancel is not
  a failure and a restart-required is not either;
* the long poll wakes on an EDGE, so no test here sleeps waiting for one;
* a finished job stays readable, because the last thing a client asks for is
  the tail of a job that has just ended.

Nothing here installs anything: every test injects its own flow through
``PackService(run_flow=...)``, and the injected flow is driven step by step
from the test thread (see :class:`ScriptedFlow`).
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest

from app.core.packs import download, restart, state
from app.core.packs.catalog import Pack, get_pack
from app.core.packs.errors import (
    PackCancelled,
    PackInstallError,
    PackInsufficientDisk,
    PackNeedsRestart,
    RestartRefused,
)
from app.core.packs.flows import InstallOutcome
from app.core.packs.paths import pending_restart_file
from app.core.packs.service import (
    PackBusy,
    PackService,
    RestartUnavailable,
    UnknownJob,
)

SENTENCE = "sentence-embeddings"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """A throwaway cache root, a cold probe cache, and no launch environment.

    The launch variables are DELETED rather than left alone: ``restart``
    reads ``os.environ`` at call time, so a suite run from a shell that had
    itself run ``cdui start`` would find restarts available and take a branch
    no test here asked for.
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    for name in ("CODEFYUI_MANAGED", restart.ENABLE_ENV,
                 restart.LAUNCHER_ENV, restart.RELAUNCH_ARGV_ENV):
        monkeypatch.delenv(name, raising=False)
    state.invalidate()
    yield tmp_path
    state.invalidate()


class ScriptedFlow:
    """A stand-in for ``flows.install_pack_live``, driven from the test thread.

    The real flow blocks its thread for minutes at a time, which is exactly
    the property that makes "while a job is running" hard to test. This one
    blocks until the test tells it what to do next -- so a test can hold a job
    open, release ONE event into a parked long poll, and finish, with no sleep
    anywhere and no wall-clock guesswork.

    It also honours ``cancel_check`` between instructions, the way the real
    flow does between download steps, so Stop is exercised for real rather
    than simulated by setting a status.
    """

    #: How long the worker thread waits for its next instruction before it
    #: gives up. A test that forgets to finish a job fails here with a clear
    #: message instead of hanging the suite.
    STARVED_AFTER_S = 20.0

    def __init__(self) -> None:
        self._steps: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.started = threading.Event()
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    # ── driven from the test thread ──────────────────────────────────────
    def send(self, event: dict) -> None:
        """Make the flow emit *event* next."""
        self._steps.put(("emit", event))

    def fail(self, exc: BaseException) -> None:
        """Make the flow raise *exc* next."""
        self._steps.put(("raise", exc))

    def finish(self) -> None:
        """Make the flow return successfully next."""
        self._steps.put(("return", None))

    def script(self, *events: dict) -> "ScriptedFlow":
        """Queue *events* and a successful return; returns self for chaining."""
        for event in events:
            self.send(event)
        self.finish()
        return self

    # ── runs on the job's worker thread ──────────────────────────────────
    def __call__(self, pack: Pack, item_ids, *, emit, cancel_check):
        self.calls.append((pack.pack_id, tuple(item_ids or ())))
        self.started.set()
        deadline = time.monotonic() + self.STARVED_AFTER_S
        while True:
            if cancel_check():
                raise PackCancelled(f"install of {pack.pack_id} cancelled")
            try:
                kind, payload = self._steps.get(timeout=0.02)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise AssertionError(
                        "scripted flow was never told what to do next")
                continue
            if kind == "emit":
                emit(payload)
            elif kind == "raise":
                raise payload
            else:
                return InstallOutcome(pack_id=pack.pack_id, pip_installed=False,
                                      items_done=tuple(item_ids or ()))


async def drain(service: PackService, job_id: str, *, timeout: float = 20.0
                ) -> tuple[list[dict], str]:
    """Every event of *job_id*, waiting for the job to reach a terminal state.

    Edge-triggered like the route is: each pass parks on the service's own
    wake-up rather than sleeping, so a job that finishes in a millisecond is
    drained in a millisecond.
    """
    limit = 500
    events: list[dict] = []
    cursor, status = 0, "running"
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} never finished (status={status})")
        page, cursor, status = await service.wait_for_events(
            job_id, after_cursor=cursor, limit=limit, wait=1.0)
        events.extend(page)
        # A page cut short by ``limit`` can carry a terminal status without
        # the terminal event, so both have to be true before we stop.
        if status != "running" and len(page) < limit:
            return events, status


async def wait_started(flow: ScriptedFlow, *, timeout: float = 10.0) -> None:
    """Yield to the loop until the flow's worker thread is running.

    NOT ``flow.started.wait(...)``: that blocks the only thread the event
    loop has, so the task that would start the flow never gets to run and
    the wait always times out.
    """
    deadline = time.monotonic() + timeout
    while not flow.started.is_set():
        if time.monotonic() > deadline:
            raise AssertionError("the injected flow never started")
        await asyncio.sleep(0.01)


def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def mark_present(pack: Pack, item_id: str, tmp_path) -> None:
    """Pretend one item of *pack* finished downloading."""
    item = next(i for i in pack.items if i.item_id == item_id)
    target = tmp_path / "downloaded" / item_id
    target.mkdir(parents=True, exist_ok=True)
    state.write_sentinel(pack.pack_id, item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item_id,
        "kind": item.kind, "repo_id": item.repo_id, "revision": item.revision,
        "snapshot_dir": str(target), "bytes": 1, "at": "1970-01-01T00:00:00Z",
    })
    state.invalidate()


# ── the happy path ────────────────────────────────────────────────────────


async def test_job_started_is_the_first_event_and_names_pack_and_items():
    flow = ScriptedFlow().script({"type": "log", "line": "hi"})
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), ["all-MiniLM-L6-v2"],
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    assert events[0]["type"] == "job_started"
    assert events[0]["pack_id"] == SENTENCE
    assert events[0]["items"] == ["all-MiniLM-L6-v2"]
    assert status == "done"


async def test_flow_events_arrive_between_job_started_and_job_done():
    flow = ScriptedFlow().script(
        {"type": "step_started", "step": "pip", "label": "Installing"},
        {"type": "log", "line": "collecting"},
        {"type": "progress", "item": "all-MiniLM-L6-v2", "bytes_done": 1,
         "bytes_total": 2, "percent": 50.0},
        {"type": "step_done", "step": "pip"},
    )
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), None,
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    assert types_of(events) == [
        "job_started", "step_started", "log", "progress", "step_done",
        "job_done"]
    assert status == "done"
    assert job.status == "done"
    assert job.finished_at is not None


async def test_every_event_carries_a_monotonic_cursor_and_a_timestamp():
    flow = ScriptedFlow().script({"type": "log", "line": "a"},
                                 {"type": "log", "line": "b"})
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, _ = await drain(service, job.job_id)

    cursors = [event["cursor"] for event in events]
    assert cursors == sorted(cursors)
    assert len(set(cursors)) == len(cursors)
    assert cursors[0] == 1
    for event in events:
        # ISO-8601 UTC, so a client can order two jobs' logs against
        # each other without knowing the server's timezone.
        assert event["ts"].endswith("+00:00")


async def test_items_none_means_the_items_that_are_not_downloaded_yet(
        isolated_cache):
    pack = get_pack(SENTENCE)
    mark_present(pack, "all-MiniLM-L6-v2", isolated_cache)
    flow = ScriptedFlow().script()
    service = PackService(run_flow=flow)

    job = await service.submit_install(pack, None, mode="live", variant=None)
    await drain(service, job.job_id)

    missing = tuple(item.item_id for item in pack.items
                    if item.item_id != "all-MiniLM-L6-v2")
    assert job.items == missing
    # And what the flow was handed matches what the job advertised, so the
    # disk precheck, the job_started event and the download cannot disagree.
    assert flow.calls == [(SENTENCE, missing)]


async def test_current_step_tracks_step_started_and_step_done():
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), ["bge-small-zh-v1.5"],
                                       mode="live", variant=None)

    flow.send({"type": "step_started", "step": "download:bge-small-zh-v1.5",
               "label": "Downloading"})
    await service.wait_for_events(job.job_id, after_cursor=1, limit=10, wait=5.0)
    assert job.current_step == "download:bge-small-zh-v1.5"

    flow.send({"type": "step_done", "step": "download:bge-small-zh-v1.5"})
    await service.wait_for_events(job.job_id, after_cursor=2, limit=10, wait=5.0)
    assert job.current_step is None

    flow.finish()
    await drain(service, job.job_id)


class CapturingFlow(ScriptedFlow):
    """A ``ScriptedFlow`` that hands the test back the ``emit`` it was given.

    That is the handle a reader thread outliving its job has: ``emit`` is a
    closure over one job's generation, and ``runner._pump`` is a DAEMON
    thread that ``_stop_process`` joins with a TIMEOUT -- so a cancelled
    install's reader can still be holding one after the job is over.
    """

    def __init__(self) -> None:
        super().__init__()
        self.emits: list = []

    def __call__(self, pack: Pack, item_ids, *, emit, cancel_check):
        self.emits.append(emit)
        return super().__call__(pack, item_ids, emit=emit,
                                cancel_check=cancel_check)


async def test_a_late_event_from_a_finished_job_never_touches_the_next_one():
    """A step stamped by whoever emitted it, not by whoever is current.

    ``_store`` used to read ``self._job``, so an event arriving from a
    finished job's reader thread set ``current_step`` on the job that had
    replaced it -- and ``GET /api/packs`` then reported some item of the NEW
    pack as "downloading" on the strength of the old one's log. The window
    is real rather than theoretical: ``emit`` takes the same lock a submit
    takes, so it can be blocked on it while the next job is being installed
    into ``self._job``.

    The event is still BUFFERED. The buffer belongs to whatever job is
    current and a stray line in it is visible, explainable and harmless; a
    wrong item badge is none of those.
    """
    flow = CapturingFlow()
    flow.finish()
    service = PackService(run_flow=flow)

    first = await service.submit_install(get_pack(SENTENCE), [],
                                         mode="live", variant=None)
    await drain(service, first.job_id)
    assert len(flow.emits) == 1, flow.emits
    stale_emit = flow.emits[0]

    second = await service.submit_install(get_pack("word-vectors"), [],
                                          mode="live", variant=None)
    stale_emit({"type": "step_started", "step": "download:ghost"})

    assert service.current_job() is second
    assert second.current_step is None
    assert first.current_step is None

    flow.finish()
    await drain(service, second.job_id)


# ── one job at a time, and the last one stays readable ────────────────────


async def test_a_second_submit_while_running_raises_pack_busy():
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await wait_started(flow)

    with pytest.raises(PackBusy) as excinfo:
        await service.submit_install(get_pack("word-vectors"), [],
                                     mode="live", variant=None)
    assert excinfo.value.job_id == job.job_id

    flow.finish()
    await drain(service, job.job_id)


async def test_a_finished_job_stays_readable_until_the_next_submit():
    first = ScriptedFlow().script({"type": "log", "line": "one"})
    service = PackService(run_flow=first)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, _ = await drain(service, job.job_id)

    # Still there after it ended: this is what the client polls for the tail.
    replay, cursor, status = await service.wait_for_events(
        job.job_id, after_cursor=0, limit=500, wait=0.0)
    assert types_of(replay) == types_of(events)
    assert status == "done"
    assert cursor == events[-1]["cursor"]
    assert service.get_job(job.job_id) is job

    first.script()  # queue a second successful run on the same flow
    later = await service.submit_install(get_pack("word-vectors"), [],
                                         mode="live", variant=None)
    await drain(service, later.job_id)
    with pytest.raises(UnknownJob):
        await service.wait_for_events(job.job_id, after_cursor=0, limit=1,
                                      wait=0.0)


async def test_a_read_for_a_replaced_job_never_returns_the_new_ones_events():
    """The buffer belongs to the CURRENT job, and only to it.

    Reaches into ``_read`` on purpose. The window this closes is a poll that
    parks on a job, is woken by that job finishing, and resumes AFTER a new
    install has already cleared the buffer -- an interleaving the public API
    cannot be made to produce on demand, because every way of ending the
    first job also gives the parked poll a chance to run first.
    """
    flow = ScriptedFlow().script()
    service = PackService(run_flow=flow)
    first = await service.submit_install(get_pack(SENTENCE), [],
                                         mode="live", variant=None)
    await drain(service, first.job_id)

    flow.script()
    second = await service.submit_install(get_pack("word-vectors"), [],
                                          mode="live", variant=None)
    await drain(service, second.job_id)

    with pytest.raises(UnknownJob):
        service._read(first, after_cursor=0, limit=10)


async def test_unknown_job_raises_unknown_job():
    service = PackService(run_flow=ScriptedFlow())
    with pytest.raises(UnknownJob):
        service.get_job("nope")
    with pytest.raises(UnknownJob):
        await service.cancel("nope")
    with pytest.raises(UnknownJob):
        await service.wait_for_events("nope", after_cursor=0, limit=1, wait=0.0)
    assert service.current_job() is None


# ── the four ways a job ends ──────────────────────────────────────────────


async def test_cancel_marks_the_job_cancelled_and_a_second_cancel_is_false():
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await wait_started(flow)

    assert await service.cancel(job.job_id) is True
    events, status = await drain(service, job.job_id)

    assert status == "cancelled"
    assert types_of(events)[-1] == "job_cancelled"
    assert job.error is None
    # Asking twice is not an error, but it did nothing the second time.
    assert await service.cancel(job.job_id) is False


async def test_needs_restart_ends_with_the_command_to_type():
    flow = ScriptedFlow()
    flow.fail(PackNeedsRestart("cannot replace torch in place",
                               command="uv pip install --python /venv/bin/python torch",
                               hint="No solution found"))
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    assert status == "needs_restart"
    assert events[-1]["type"] == "needs_restart"
    assert events[-1]["command"] == "uv pip install --python /venv/bin/python torch"
    assert job.restart_command == "uv pip install --python /venv/bin/python torch"


async def test_install_error_ends_with_its_message_and_hint():
    flow = ScriptedFlow()
    flow.fail(PackInstallError("installing failed (uv exited 1)",
                               hint="No solution found"))
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    assert events[-1]["type"] == "job_failed"
    assert events[-1]["message"] == "installing failed (uv exited 1)"
    assert events[-1]["hint"] == "No solution found"
    assert job.error == {"message": "installing failed (uv exited 1)",
                         "hint": "No solution found"}


async def test_insufficient_disk_raised_by_the_flow_is_a_failure():
    """A disk error can still surface late (the precheck is approximate)."""
    flow = ScriptedFlow()
    flow.fail(PackInsufficientDisk("not enough free disk space",
                                   needed=10, free=1))
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    assert events[-1]["type"] == "job_failed"
    assert "disk" in events[-1]["message"]


async def test_an_unexpected_exception_is_reported_as_a_failure():
    flow = ScriptedFlow()
    flow.fail(ZeroDivisionError("boom"))
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    # repr, not str: "boom" alone would not say what went wrong.
    assert events[-1]["message"] == repr(ZeroDivisionError("boom"))
    assert events[-1]["hint"] is None


class _Abort(BaseException):
    """A BaseException that is NOT KeyboardInterrupt or SystemExit.

    Those two are special-cased by ``asyncio.Task.__step``, which re-raises
    them into the event loop itself -- a test using one would tear down the
    loop rather than exercise the bookkeeping.
    """


async def test_a_base_exception_still_records_a_terminal_state():
    flow = ScriptedFlow()
    flow.fail(_Abort("interrupted"))
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, status = await drain(service, job.job_id)

    # A job stuck on "running" forever would leave the panel offering Stop
    # for a thread that no longer exists, and no submit would be accepted
    # again for the life of the process.
    assert status == "failed"
    assert events[-1]["type"] == "job_failed"
    assert "_Abort" in events[-1]["message"]
    assert job.finished_at is not None
    # And it was re-raised rather than swallowed. Retrieving it here is also
    # what keeps "exception was never retrieved" out of the test output.
    assert isinstance(service._task.exception(), _Abort)


async def test_the_probe_cache_is_dropped_when_a_job_ends(monkeypatch):
    calls = []
    monkeypatch.setattr(state, "invalidate", lambda: calls.append(1))
    flow = ScriptedFlow().script()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await drain(service, job.job_id)
    assert calls, "a finished job must drop the probe cache"


# ── refusals that happen before the job starts ────────────────────────────


async def test_insufficient_disk_is_raised_from_submit(monkeypatch):
    def _no_room(items):
        raise PackInsufficientDisk("not enough free disk space",
                                   needed=700_000_000, free=1_000_000)

    monkeypatch.setattr(download, "check_disk", _no_room)
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)

    with pytest.raises(PackInsufficientDisk) as excinfo:
        await service.submit_install(get_pack(SENTENCE), ["all-MiniLM-L6-v2"],
                                     mode="live", variant=None)
    assert excinfo.value.needed == 700_000_000
    assert excinfo.value.free == 1_000_000
    # Nothing started, so the next submit is not refused as busy.
    assert service.current_job() is None
    assert not flow.started.is_set()


async def test_restart_mode_is_refused_with_the_cli_command():
    service = PackService(run_flow=ScriptedFlow())
    with pytest.raises(RestartUnavailable) as excinfo:
        await service.submit_install(get_pack("gpu-torch"), [],
                                     mode="restart", variant="cu128")
    assert excinfo.value.command == "cdui install --gpu cu128"
    assert service.current_job() is None


async def test_a_restart_mode_pack_is_refused_even_when_asked_for_live():
    """The pack's mode decides. gpu-torch has no pip specs, no probe modules
    and no items, so a live install of it would report success having done
    nothing at all."""
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    with pytest.raises(RestartUnavailable) as excinfo:
        await service.submit_install(get_pack("gpu-torch"), None,
                                     mode="live", variant=None)
    assert excinfo.value.command.startswith("cdui install --gpu ")
    assert service.current_job() is None
    assert not flow.started.is_set()


async def test_an_unknown_item_id_is_a_value_error():
    """The same type, and the same message, ``flows._resolve_items`` uses."""
    service = PackService(run_flow=ScriptedFlow())
    with pytest.raises(ValueError, match="has no item"):
        await service.submit_install(get_pack(SENTENCE), ["not-a-model"],
                                     mode="live", variant=None)
    assert service.current_job() is None


# ── the restart handshake ─────────────────────────────────────────────────


@pytest.fixture
def available(monkeypatch):
    """A server that may restart itself, without the environment for it."""
    monkeypatch.setattr(restart, "restart_available", lambda: True)


async def test_the_handshake_writes_spawns_stores_then_stops_the_server(
        monkeypatch, available):
    """The ORDER is the design, and every step depends on the one before it.

    The claim is written first because ``spawn_helper`` READS the file it is
    given; the job is stored after the spawn because a spawn that fails must
    leave no job behind for the panel to follow; and the shutdown is
    scheduled last, once the caller has a job id -- the 202 goes out before
    the SIGINT lands.
    """
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    steps: list[tuple] = []
    real_write = restart.write_pending

    def _write(pending):
        path = real_write(pending)
        steps.append(("write", pending.job_id, service.current_job()))
        return path

    def _spawn(path):
        steps.append(("spawn", str(path), service.current_job()))
        return 4242

    def _shutdown(loop, delay: float = 0.5):
        job = service.current_job()
        steps.append(("shutdown", job.job_id, job.status))

    monkeypatch.setattr(restart, "write_pending", _write)
    monkeypatch.setattr(restart, "spawn_helper", _spawn)
    monkeypatch.setattr(restart, "schedule_self_shutdown", _shutdown)

    job = await service.submit_install(get_pack("gpu-torch"), None,
                                       mode="restart", variant="cu128")

    assert [step[0] for step in steps] == ["write", "spawn", "shutdown"]
    assert steps[0][1] == job.job_id, "the claim named a different job"
    assert steps[0][2] is None, "a job existed before its claim was written"
    assert steps[1][1] == str(pending_restart_file())
    assert steps[1][2] is None, "the job was stored before the helper started"
    assert steps[2][1:] == (job.job_id, "needs_restart")
    assert not flow.started.is_set(), "the live installer ran anyway"


async def test_a_restart_mode_pip_pack_writes_its_specs_for_the_helper(
        monkeypatch, available):
    """``kind`` is what the helper branches on: a wheel swap and a pip
    install are two different command lines, and it cannot work out which
    from the pack id -- it has no catalog."""
    monkeypatch.setattr(restart, "spawn_helper", lambda path: 4242)
    monkeypatch.setattr(restart, "schedule_self_shutdown",
                        lambda loop, delay=0.5: None)
    pack = get_pack(SENTENCE)
    service = PackService(run_flow=ScriptedFlow())

    job = await service.submit_install(pack, None, mode="restart", variant=None)

    assert job.mode == "restart"
    assert job.status == "needs_restart"
    pending = restart.PendingRestart.from_json(
        pending_restart_file().read_text(encoding="utf-8"))
    assert pending.kind == "pip"
    assert pending.specs == tuple(pack.pip)
    assert pending.packages == ()
    assert pending.index_url is None
    assert job.restart_command == f"cdui packs install {SENTENCE}"


async def test_a_restart_mode_pack_with_nothing_to_install_is_a_value_error(
        available):
    """word-vectors ships data and no packages: there is no pip line a helper
    could run for it, and stopping the server to install nothing is worse
    than refusing."""
    service = PackService(run_flow=ScriptedFlow())

    with pytest.raises(ValueError, match="word-vectors"):
        await service.submit_install(get_pack("word-vectors"), None,
                                     mode="restart", variant=None)

    assert service.current_job() is None
    assert not pending_restart_file().exists(), "a refused install left a claim"


async def test_a_restart_is_refused_while_a_graph_runs(monkeypatch, available):
    """``runs_active`` is INJECTED, so the service never holds the app -- and
    a test never has to build one to answer the only question it asks."""
    monkeypatch.setattr(restart, "spawn_helper",
                        lambda path: pytest.fail("a helper was started"))
    busy = True
    service = PackService(run_flow=ScriptedFlow(), runs_active=lambda: busy)

    with pytest.raises(RestartRefused) as excinfo:
        await service.submit_install(get_pack("gpu-torch"), None,
                                     mode="restart", variant="cu128")
    assert excinfo.value.reason == "a graph is running"
    assert excinfo.value.command == "cdui install --gpu cu128"
    assert isinstance(excinfo.value, PackInstallError), (
        "the route's error mapping is written against PackInstallError")
    assert service.current_job() is None
    assert not pending_restart_file().exists()

    # And the same service goes ahead once the run is over.
    monkeypatch.setattr(restart, "spawn_helper", lambda path: 4242)
    monkeypatch.setattr(restart, "schedule_self_shutdown",
                        lambda loop, delay=0.5: None)
    busy = False
    job = await service.submit_install(get_pack("gpu-torch"), None,
                                       mode="restart", variant="cu128")
    assert job.status == "needs_restart"


async def test_a_live_install_that_lands_during_the_probe_beats_the_restart(
        monkeypatch, available):
    """The busy check AFTER the await, which is the only one that can see it.

    ``install_command_for`` is run on a thread because naming the GPU pack's
    command means asking nvidia-smi, and that is seconds from a cold cache.
    Every suspension point is a chance for another request to land, and the
    one that lands here is a LIVE install: a download that would die
    unfinished with the process this restart is about to stop. So the check
    is repeated on the other side of the await, and it refuses the restart
    rather than the install that is already running.
    """
    probing = threading.Event()
    release = threading.Event()
    real_command = restart.install_command_for

    def _parked(pack, variant=None):
        probing.set()
        assert release.wait(10), "the probe was never released"
        return real_command(pack, variant)

    monkeypatch.setattr(restart, "install_command_for", _parked)
    monkeypatch.setattr(restart, "spawn_helper", lambda path: pytest.fail(
        "a helper was started while a live install was running"))
    monkeypatch.setattr(restart, "schedule_self_shutdown",
                        lambda loop, delay=0.5: pytest.fail(
                            "the server was stopped out from under a live "
                            "install"))
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)

    restarting = asyncio.create_task(service.submit_install(
        get_pack("gpu-torch"), None, mode="restart", variant="cu128"))
    while not probing.is_set():
        await asyncio.sleep(0.01)   # parked on its own thread, loop free

    live = await service.submit_install(get_pack(SENTENCE), [],
                                        mode="live", variant=None)
    await wait_started(flow)
    release.set()

    with pytest.raises(PackBusy) as excinfo:
        await restarting
    assert excinfo.value.job_id == live.job_id, (
        "the refusal has to name the job that is actually running")
    assert not pending_restart_file().exists(), (
        "a claim was written over a live install")
    assert service.current_job() is live, "the live job lost its slot"

    flow.finish()
    events, status = await drain(service, live.job_id)
    assert status == "done", "the install the user was already watching"


async def test_a_spawn_that_fails_withdraws_the_claim_and_starts_no_job(
        monkeypatch, available):
    """The failure is reported by the server that is still running, so it has
    to leave nothing behind -- a pending file nobody will act on refuses
    every later restart with "one is already pending"."""
    monkeypatch.setattr(restart, "spawn_helper", lambda path: (_ for _ in ())
                        .throw(OSError("cannot find the launcher")))
    monkeypatch.setattr(restart, "schedule_self_shutdown",
                        lambda loop, delay=0.5: pytest.fail(
                            "the server was stopped for an install that "
                            "never started"))
    service = PackService(run_flow=ScriptedFlow())

    with pytest.raises(PackInstallError, match="helper"):
        await service.submit_install(get_pack("gpu-torch"), None,
                                     mode="restart", variant="cu128")

    assert service.current_job() is None
    assert not pending_restart_file().exists()


# ── the long poll ─────────────────────────────────────────────────────────


async def test_events_paginate_by_cursor_and_limit():
    flow = ScriptedFlow().script(*[{"type": "log", "line": str(n)}
                                   for n in range(5)])
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await drain(service, job.job_id)

    first, cursor, _ = await service.wait_for_events(
        job.job_id, after_cursor=0, limit=2, wait=0.0)
    assert len(first) == 2
    assert cursor == first[-1]["cursor"]

    second, cursor2, _ = await service.wait_for_events(
        job.job_id, after_cursor=cursor, limit=2, wait=0.0)
    assert [event["cursor"] for event in second] == [cursor + 1, cursor + 2]
    assert cursor2 == second[-1]["cursor"]

    # An empty tail never moves the cursor backwards.
    tail, tail_cursor, _ = await service.wait_for_events(
        job.job_id, after_cursor=10_000, limit=10, wait=0.0)
    assert tail == []
    assert tail_cursor == 10_000


async def test_long_poll_returns_at_once_when_the_job_is_terminal():
    flow = ScriptedFlow().script({"type": "log", "line": "done"})
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    events, _ = await drain(service, job.job_id)

    started = time.monotonic()
    tail, cursor, status = await service.wait_for_events(
        job.job_id, after_cursor=events[-1]["cursor"], limit=10, wait=5.0)
    elapsed = time.monotonic() - started

    assert tail == []
    assert status == "done"
    assert cursor == events[-1]["cursor"]
    assert elapsed < 2.0, "a finished job must not hold the poll open"


async def test_long_poll_wakes_on_the_next_event():
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await wait_started(flow)

    # Park on the tail of what the job has emitted so far (job_started).
    poll = asyncio.create_task(service.wait_for_events(
        job.job_id, after_cursor=1, limit=10, wait=5.0))
    await asyncio.sleep(0.05)
    assert not poll.done(), "the poll must actually park"

    started = time.monotonic()
    flow.send({"type": "log", "line": "woken"})
    page, cursor, status = await asyncio.wait_for(poll, 5.0)
    elapsed = time.monotonic() - started

    assert types_of(page) == ["log"]
    assert cursor == 2
    assert status == "running"
    assert elapsed < 3.0, "the poll waited for its deadline instead of the edge"

    flow.finish()
    await drain(service, job.job_id)


async def test_long_poll_returns_immediately_when_wait_is_zero():
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await wait_started(flow)

    started = time.monotonic()
    page, _, status = await service.wait_for_events(
        job.job_id, after_cursor=1, limit=10, wait=0.0)
    assert page == []
    assert status == "running"
    assert time.monotonic() - started < 1.0

    flow.finish()
    await drain(service, job.job_id)


# ── shutdown ──────────────────────────────────────────────────────────────


async def test_shutdown_cancels_a_running_job():
    flow = ScriptedFlow()
    service = PackService(run_flow=flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)
    await wait_started(flow)

    await service.shutdown()

    assert job.cancel_event.is_set()
    assert job.status == "cancelled"
    assert job.finished_at is not None


async def test_shutdown_with_no_job_is_a_no_op():
    service = PackService(run_flow=ScriptedFlow())
    await service.shutdown()
    assert service.current_job() is None


async def test_shutdown_survives_a_flow_that_raised_a_base_exception():
    """``_run`` records a ``BaseException`` and then RE-RAISES it, so it
    travels out of the shielded task and into the await in ``shutdown`` --
    the one inside the lifespan shutdown hook. A catch that only covered
    ``Exception`` would let it out there and make "the server is coming
    down" a traceback on the way out, for a job that already reached a
    terminal state and told everyone watching.
    """
    class _OutOfBand(BaseException):
        """Private on purpose: nothing else may catch it by accident."""

    def _doomed_flow(pack, item_ids, *, emit, cancel_check):
        raise _OutOfBand("the interpreter is going away")

    service = PackService(run_flow=_doomed_flow)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)

    await service.shutdown()

    assert job.terminal
    assert job.status == "failed"
    assert "_OutOfBand" in job.error["message"]


async def test_shutdown_is_bounded_when_the_flow_ignores_cancellation():
    """Server shutdown must not be held hostage by a stubborn install."""
    release = threading.Event()

    def _deaf_flow(pack, item_ids, *, emit, cancel_check):
        release.wait(20)
        return InstallOutcome(pack_id=pack.pack_id, pip_installed=False,
                              items_done=())

    service = PackService(run_flow=_deaf_flow, shutdown_timeout_s=0.2)
    job = await service.submit_install(get_pack(SENTENCE), [],
                                       mode="live", variant=None)

    started = time.monotonic()
    try:
        await service.shutdown()
        assert time.monotonic() - started < 5.0
        assert job.cancel_event.is_set()
    finally:
        # Let the stubborn flow finish so no task outlives the test loop.
        release.set()
        await drain(service, job.job_id)
