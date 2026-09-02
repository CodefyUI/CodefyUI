"""The generic job runner, with no domain behind it at all.

``JobRunner`` is the single job slot the Package Center and the Plugin Center
share: one job at a time, its events stamped with a cursor a poller can resume
from, a cooperative stop and a bounded shutdown. Everything below is about
that seam and nothing about what a job is FOR -- the domain arrives as two
callbacks (``work`` and ``terminal_for``) and this module supplies the
smallest honest ones it can:

* every event is produced on a worker thread and read from the loop, so the
  lock rules are what is actually under test;
* the terminal event is stored BEFORE the status that explains it flips,
  because the other order loses the last event of every job to whoever polls
  at the wrong moment;
* the long poll wakes on an EDGE, so no test here sleeps waiting for one;
* an exception the domain does not claim reaches a terminal state anyway, and
  then travels on.

``test_packs_service.py`` covers the same runner through the Package Center's
vocabulary; this file is what a second domain can rely on before it is
written. The ``ScriptedWork`` driver below is the ``ScriptedFlow`` from that
file with the pack argument shape taken out.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from dataclasses import dataclass

import pytest

from app.core.jobs import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_NEEDS_RESTART,
    STATUS_RUNNING,
    CancelCheck,
    Emit,
    Job,
    JobBusy,
    JobRunner,
    UnknownJob,
)

# ── the smallest domain that can drive a runner ───────────────────────────


class Cancelled(Exception):
    """The job was asked to stop. NOT a failure -- see :func:`terminal_for`."""


class Failed(Exception):
    """A failure shape this domain designed. ``hint`` is the detail."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class _Abort(BaseException):
    """A BaseException that is NOT KeyboardInterrupt or SystemExit.

    Those two are special-cased by ``asyncio.Task.__step``, which re-raises
    them into the event loop itself -- a test using one would tear down the
    loop rather than exercise the bookkeeping.
    """


def terminal_for(exc: BaseException) -> tuple[str, dict] | None:
    """A domain's whole answer to "what did that exception do to the job?".

    Deliberately the same shape ``PackService._terminal_for`` has, minus the
    packs: a cancel is not a failure, a designed failure keeps its message
    and hint, anything else is a failure reported as its repr -- and a
    ``BaseException`` is not this domain's to claim, which is what ``None``
    says.
    """
    if isinstance(exc, Cancelled):
        return STATUS_CANCELLED, {"type": "job_cancelled"}
    if isinstance(exc, Failed):
        return STATUS_FAILED, {"type": "job_failed",
                               "message": str(exc), "hint": exc.hint}
    if isinstance(exc, Exception):
        return STATUS_FAILED, {"type": "job_failed",
                               "message": repr(exc), "hint": None}
    return None


@dataclass(kw_only=True)
class DemoJob(Job):
    """A ``Job`` with one field of its own, and that field is REQUIRED.

    Which is the whole point of the keyword-only base: a domain adds what it
    needs without having to give it a default just to sit after the base's
    ``status``, ``created_at`` and the rest.
    """

    name: str


class ScriptedWork:
    """A stand-in for a real job's work, driven from the test thread.

    Real work blocks its thread for minutes at a time, which is exactly the
    property that makes "while a job is running" hard to test. This one
    blocks until the test tells it what to do next -- so a test can hold a
    job open, release ONE event into a parked long poll, and finish, with no
    sleep anywhere and no wall-clock guesswork.

    It also honours ``cancel_check`` between instructions, the way real work
    does between its steps, so Stop is exercised for real rather than
    simulated by setting a status.

    ``emits`` keeps every ``emit`` it was handed. That is the handle a reader
    thread outliving its job has -- ``packs.runner._pump`` is a DAEMON thread
    joined with a TIMEOUT -- so a finished job's reader can still be holding
    one.
    """

    #: How long the worker thread waits for its next instruction before it
    #: gives up. A test that forgets to finish a job fails here with a clear
    #: message instead of hanging the suite.
    STARVED_AFTER_S = 20.0

    def __init__(self) -> None:
        self._steps: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.started = threading.Event()
        self.emits: list[Emit] = []

    # ── driven from the test thread ──────────────────────────────────────
    def send(self, event: dict) -> None:
        """Make the work emit *event* next."""
        self._steps.put(("emit", event))

    def fail(self, exc: BaseException) -> None:
        """Make the work raise *exc* next."""
        self._steps.put(("raise", exc))

    def finish(self) -> None:
        """Make the work return successfully next."""
        self._steps.put(("return", None))

    def script(self, *events: dict) -> "ScriptedWork":
        """Queue *events* and a successful return; returns self for chaining."""
        for event in events:
            self.send(event)
        self.finish()
        return self

    # ── runs on the job's worker thread ──────────────────────────────────
    def __call__(self, emit: Emit, cancel_check: CancelCheck) -> None:
        self.emits.append(emit)
        self.started.set()
        deadline = time.monotonic() + self.STARVED_AFTER_S
        while True:
            if cancel_check():
                raise Cancelled("the job was asked to stop")
            try:
                kind, payload = self._steps.get(timeout=0.02)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise AssertionError(
                        "scripted work was never told what to do next")
                continue
            if kind == "emit":
                emit(payload)
            elif kind == "raise":
                raise payload
            else:
                return


def make_runner(**kwargs) -> JobRunner:
    """A runner wired to the domain above. ``label`` shows up in its logs."""
    return JobRunner(terminal_for=terminal_for, label="demo job", **kwargs)


def submit(runner: JobRunner, work, *, name: str = "demo") -> DemoJob:
    """Claim the slot for a fresh job and run *work* in it.

    The two calls a domain makes, in the order it has to make them: nothing
    awaits between them, so the generation ``claim`` installs is the one the
    work's ``emit`` closes over.
    """
    job = DemoJob(job_id=uuid.uuid4().hex, name=name)
    runner.claim(job, {"type": "job_started", "name": name})
    runner.start(job, work)
    return job


async def drain(runner: JobRunner, job_id: str, *, timeout: float = 20.0
                ) -> tuple[list[dict], str]:
    """Every event of *job_id*, waiting for the job to reach a terminal state.

    Edge-triggered like a route would be: each pass parks on the runner's own
    wake-up rather than sleeping, so a job that finishes in a millisecond is
    drained in a millisecond.
    """
    limit = 500
    events: list[dict] = []
    cursor, status = 0, STATUS_RUNNING
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} never finished (status={status})")
        page, cursor, status = await runner.wait_for_events(
            job_id, after_cursor=cursor, limit=limit, wait=1.0)
        events.extend(page)
        # A page cut short by ``limit`` can carry a terminal status without
        # the terminal event, so both have to be true before we stop.
        if status != STATUS_RUNNING and len(page) < limit:
            return events, status


async def wait_started(work: ScriptedWork, *, timeout: float = 10.0) -> None:
    """Yield to the loop until the work's thread is running.

    NOT ``work.started.wait(...)``: that blocks the only thread the event
    loop has, so the task that would start the work never gets to run and the
    wait always times out.
    """
    deadline = time.monotonic() + timeout
    while not work.started.is_set():
        if time.monotonic() > deadline:
            raise AssertionError("the scripted work never started")
        await asyncio.sleep(0.01)


def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


# ── the event stream ──────────────────────────────────────────────────────


async def test_the_first_event_is_cursor_one_and_beats_the_work_to_it():
    """``claim`` buffers it, so a client polling from 0 always sees it first."""
    runner = make_runner()
    work = ScriptedWork().script({"type": "log", "line": "hi"})
    job = submit(runner, work, name="first")
    events, status = await drain(runner, job.job_id)

    assert types_of(events) == ["job_started", "log", "job_done"]
    assert events[0]["cursor"] == 1
    assert events[0]["name"] == "first"
    assert status == STATUS_DONE
    assert job.finished_at is not None


async def test_every_event_carries_a_monotonic_cursor_and_a_timestamp():
    runner = make_runner()
    work = ScriptedWork().script({"type": "log", "line": "a"},
                                 {"type": "log", "line": "b"})
    job = submit(runner, work)
    events, _ = await drain(runner, job.job_id)

    cursors = [event["cursor"] for event in events]
    assert cursors == sorted(cursors)
    assert len(set(cursors)) == len(cursors)
    assert cursors[0] == 1
    for event in events:
        # ISO-8601 UTC, so a client can order two jobs' logs against each
        # other without knowing the server's timezone.
        assert event["ts"].endswith("+00:00")


async def test_the_terminal_event_is_readable_before_the_status_flips():
    """The invariant a poller depends on, checked from both sides.

    From INSIDE the lock hold: ``_finish`` stores the event while the status
    still says running, so the buffer is never behind the status. And from
    outside, on another thread: ``_read`` takes the same lock, so a reader
    that gets in and sees a terminal status has already been handed the event
    explaining it. The other order loses the last event of every job to
    whoever polls at the wrong moment.
    """
    runner = make_runner()
    work = ScriptedWork()
    job = submit(runner, work)
    await wait_started(work)

    ordering: list[tuple[str, str]] = []
    stored = runner._store

    def watched_store(event: dict, owner) -> None:
        stored(event, owner)
        # Still inside _finish's lock hold for a terminal event.
        ordering.append((event.get("type"), job.status))

    runner._store = watched_store

    samples: list[tuple[str, list[str]]] = []

    def poll() -> None:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            page, _, status = runner._read(job, 0, 500)
            samples.append((status, types_of(page)))
            if status != STATUS_RUNNING:
                return
            time.sleep(0.001)

    reader = threading.Thread(target=poll, name="reader", daemon=True)
    reader.start()
    work.finish()
    await drain(runner, job.job_id)
    reader.join(10)
    assert not reader.is_alive(), "the reader thread never saw the job end"

    assert ("job_done", STATUS_RUNNING) in ordering, (
        "the status flipped before the terminal event was buffered")
    terminal = [types for status, types in samples if status != STATUS_RUNNING]
    assert terminal, "the reader never read a terminal status"
    for types in terminal:
        assert types[-1] == "job_done", (
            "a reader was handed a terminal status without the event for it")


async def test_a_late_event_from_a_finished_job_never_touches_the_next_one():
    """A step stamped by whoever emitted it, not by whoever is current.

    ``emit`` is a closure over ONE job's generation, and the thread holding it
    can outlive the job. If ``_store`` read the current job instead, a line
    arriving late from a finished job would set ``current_step`` on the job
    that replaced it -- and a panel would then report a step of the new job
    on the strength of the old one's log.

    The event is still BUFFERED. The buffer belongs to whatever job is
    current and a stray line in it is visible, explainable and harmless; a
    wrong step is none of those.
    """
    runner = make_runner()
    first_work = ScriptedWork()
    first_work.finish()
    first = submit(runner, first_work, name="first")
    await drain(runner, first.job_id)
    assert len(first_work.emits) == 1, first_work.emits
    stale_emit = first_work.emits[0]

    second_work = ScriptedWork()
    second = submit(runner, second_work, name="second")
    stale_emit({"type": "step_started", "step": "ghost"})

    assert runner.current_job() is second
    assert second.current_step is None
    assert first.current_step is None
    page, _, _ = runner._read(second, 0, 500)
    assert "step_started" in types_of(page), "the stray event was dropped"

    second_work.finish()
    await drain(runner, second.job_id)


async def test_the_buffer_drops_its_oldest_events_rather_than_growing():
    """A client four thousand events behind is not reading them anyway."""
    runner = make_runner(max_events=3)
    work = ScriptedWork().script({"type": "log", "line": "a"},
                                 {"type": "log", "line": "b"},
                                 {"type": "log", "line": "c"})
    job = submit(runner, work)
    await drain(runner, job.job_id)

    kept, cursor, status = await runner.wait_for_events(
        job.job_id, after_cursor=0, limit=500, wait=0.0)
    # Five events were produced (job_started, three logs, job_done); the
    # three most recent survive, and their cursors are still the originals.
    assert [event["cursor"] for event in kept] == [3, 4, 5]
    assert types_of(kept) == ["log", "log", "job_done"]
    assert (cursor, status) == (5, STATUS_DONE)


# ── one job at a time, and the last one stays readable ────────────────────


async def test_a_second_claim_while_a_job_runs_raises_job_busy():
    runner = make_runner()
    work = ScriptedWork()
    job = submit(runner, work)
    await wait_started(work)

    with pytest.raises(JobBusy) as excinfo:
        runner.claim(DemoJob(job_id=uuid.uuid4().hex, name="second"),
                     {"type": "job_started", "name": "second"})
    # The id is the point of the refusal: it is what a client follows.
    assert excinfo.value.job_id == job.job_id
    assert runner.current_job() is job

    work.finish()
    await drain(runner, job.job_id)


async def test_a_finished_job_stays_readable_until_the_next_claim():
    runner = make_runner()
    work = ScriptedWork().script({"type": "log", "line": "one"})
    job = submit(runner, work)
    events, _ = await drain(runner, job.job_id)

    # Still there after it ended: this is what a client polls for the tail.
    replay, cursor, status = await runner.wait_for_events(
        job.job_id, after_cursor=0, limit=500, wait=0.0)
    assert types_of(replay) == types_of(events)
    assert status == STATUS_DONE
    assert cursor == events[-1]["cursor"]
    assert runner.get_job(job.job_id) is job

    later = submit(runner, ScriptedWork().script(), name="later")
    await drain(runner, later.job_id)

    # And now it is gone, id and events together.
    with pytest.raises(UnknownJob):
        runner.get_job(job.job_id)
    with pytest.raises(UnknownJob):
        runner._read(job, 0, 10)
    with pytest.raises(UnknownJob):
        await runner.wait_for_events(job.job_id, after_cursor=0, limit=1,
                                     wait=0.0)


async def test_an_id_that_was_never_claimed_is_unknown():
    runner = make_runner()
    assert runner.current_job() is None
    with pytest.raises(UnknownJob):
        runner.get_job("nope")
    with pytest.raises(UnknownJob):
        await runner.cancel("nope")
    with pytest.raises(UnknownJob):
        await runner.wait_for_events("nope", after_cursor=0, limit=1, wait=0.0)


# ── the four ways a job ends ──────────────────────────────────────────────


async def test_cancel_ends_the_job_and_a_second_cancel_is_false():
    runner = make_runner()
    work = ScriptedWork()
    job = submit(runner, work)
    await wait_started(work)

    assert await runner.cancel(job.job_id) is True
    events, status = await drain(runner, job.job_id)

    assert status == STATUS_CANCELLED
    assert types_of(events)[-1] == "job_cancelled"
    # A cancel is not a failure, so it leaves no error behind.
    assert job.error is None
    # Asking twice is not an error, but it did nothing the second time.
    assert await runner.cancel(job.job_id) is False


async def test_a_failure_the_domain_claims_is_recorded_and_not_re_raised():
    runner = make_runner()
    work = ScriptedWork()
    work.fail(Failed("it did not work", hint="the last lines"))
    job = submit(runner, work)
    events, status = await drain(runner, job.job_id)

    assert status == STATUS_FAILED
    assert events[-1]["type"] == "job_failed"
    assert events[-1]["message"] == "it did not work"
    assert events[-1]["hint"] == "the last lines"
    # The job's own summary of it, made from the event so the two cannot
    # disagree about why it failed.
    assert job.error == {"message": "it did not work", "hint": "the last lines"}
    # Swallowed: nobody awaits this task except shutdown, and an escaping
    # exception would be reported long after the job left the UI.
    assert runner._task.exception() is None


async def test_a_base_exception_is_recorded_as_failed_and_then_re_raised():
    runner = make_runner()
    work = ScriptedWork()
    work.fail(_Abort("interrupted"))
    job = submit(runner, work)
    events, status = await drain(runner, job.job_id)

    # A job stuck on "running" forever would leave a panel offering Stop for
    # a thread that no longer exists, and no claim would be accepted again.
    assert status == STATUS_FAILED
    assert events[-1]["type"] == "job_failed"
    assert "_Abort" in events[-1]["message"]
    assert job.error["hint"] is None
    assert job.finished_at is not None
    # And it was re-raised rather than swallowed. Retrieving it here is also
    # what keeps "exception was never retrieved" out of the test output.
    assert isinstance(runner._task.exception(), _Abort)


async def test_a_job_can_be_terminal_on_arrival_with_no_work_at_all():
    """Work that is not this process's to do, which packs' restart mode is.

    The job exists so that a client has the same thing to follow as for any
    other -- one id, one event stream, one terminal event saying what happens
    next -- and there is no task behind it to await. The claim drops the
    PREVIOUS job's task for that reason: ``shutdown`` would otherwise wait on
    something belonging to a job nobody can read any more.
    """
    runner = make_runner()
    earlier = submit(runner, ScriptedWork().script(), name="earlier")
    await drain(runner, earlier.job_id)
    assert runner._task is not None

    job = DemoJob(job_id=uuid.uuid4().hex, name="elsewhere")
    runner.claim(job, {"type": "job_started", "name": job.name})
    runner.finish(job, STATUS_NEEDS_RESTART,
                  {"type": "needs_restart", "command": "cdui install"})

    page, cursor, status = await runner.wait_for_events(
        job.job_id, after_cursor=0, limit=10, wait=5.0)
    assert types_of(page) == ["job_started", "needs_restart"]
    assert (cursor, status) == (2, STATUS_NEEDS_RESTART)
    assert job.terminal and job.finished_at is not None
    assert runner._task is None, "a job with no work must leave no task"
    await runner.shutdown()


async def test_on_settled_runs_however_the_job_ended():
    """Where a domain drops the caches a finished job invalidated.

    Including the path that re-raises: the disk is not what it was whether
    the work returned, failed, or was torn out from under the interpreter.
    """
    runner = make_runner()
    settled: list[str] = []

    work = ScriptedWork().script()
    job = DemoJob(job_id=uuid.uuid4().hex, name="clean")
    runner.claim(job, {"type": "job_started", "name": job.name})
    runner.start(job, work, on_settled=lambda: settled.append("clean"))
    await drain(runner, job.job_id)
    assert settled == ["clean"]

    aborting = ScriptedWork()
    aborting.fail(_Abort("interrupted"))
    second = DemoJob(job_id=uuid.uuid4().hex, name="aborted")
    runner.claim(second, {"type": "job_started", "name": second.name})
    runner.start(second, aborting,
                 on_settled=lambda: settled.append("aborted"))
    await drain(runner, second.job_id)
    assert settled == ["clean", "aborted"]
    assert isinstance(runner._task.exception(), _Abort)


# ── the long poll ─────────────────────────────────────────────────────────


async def test_events_paginate_by_cursor_and_limit():
    runner = make_runner()
    work = ScriptedWork().script(*[{"type": "log", "line": str(n)}
                                   for n in range(5)])
    job = submit(runner, work)
    await drain(runner, job.job_id)

    first, cursor, _ = await runner.wait_for_events(
        job.job_id, after_cursor=0, limit=2, wait=0.0)
    assert len(first) == 2
    assert cursor == first[-1]["cursor"]

    second, cursor2, _ = await runner.wait_for_events(
        job.job_id, after_cursor=cursor, limit=2, wait=0.0)
    assert [event["cursor"] for event in second] == [cursor + 1, cursor + 2]
    assert cursor2 == second[-1]["cursor"]

    # An empty tail never moves the cursor backwards.
    tail, tail_cursor, _ = await runner.wait_for_events(
        job.job_id, after_cursor=10_000, limit=10, wait=0.0)
    assert tail == []
    assert tail_cursor == 10_000


async def test_long_poll_wakes_on_the_next_event():
    runner = make_runner()
    work = ScriptedWork()
    job = submit(runner, work)
    await wait_started(work)

    # Park on the tail of what the job has emitted so far (job_started).
    poll = asyncio.create_task(runner.wait_for_events(
        job.job_id, after_cursor=1, limit=10, wait=5.0))
    await asyncio.sleep(0.05)
    assert not poll.done(), "the poll must actually park"

    started = time.monotonic()
    work.send({"type": "log", "line": "woken"})
    page, cursor, status = await asyncio.wait_for(poll, 5.0)
    elapsed = time.monotonic() - started

    assert types_of(page) == ["log"]
    assert cursor == 2
    assert status == STATUS_RUNNING
    assert elapsed < 3.0, "the poll waited for its deadline instead of the edge"

    work.finish()
    await drain(runner, job.job_id)


async def test_long_poll_returns_at_once_when_the_job_is_terminal():
    runner = make_runner()
    work = ScriptedWork().script({"type": "log", "line": "done"})
    job = submit(runner, work)
    events, _ = await drain(runner, job.job_id)

    started = time.monotonic()
    tail, cursor, status = await runner.wait_for_events(
        job.job_id, after_cursor=events[-1]["cursor"], limit=10, wait=5.0)
    elapsed = time.monotonic() - started

    assert tail == []
    assert status == STATUS_DONE
    assert cursor == events[-1]["cursor"]
    assert elapsed < 2.0, "a finished job must not hold the poll open"


async def test_long_poll_returns_immediately_when_wait_is_zero():
    runner = make_runner()
    work = ScriptedWork()
    job = submit(runner, work)
    await wait_started(work)

    started = time.monotonic()
    page, _, status = await runner.wait_for_events(
        job.job_id, after_cursor=1, limit=10, wait=0.0)
    assert page == []
    assert status == STATUS_RUNNING
    assert time.monotonic() - started < 1.0

    work.finish()
    await drain(runner, job.job_id)


# ── shutdown ──────────────────────────────────────────────────────────────


async def test_shutdown_cancels_a_running_job():
    runner = make_runner()
    work = ScriptedWork()
    job = submit(runner, work)
    await wait_started(work)

    await runner.shutdown()

    assert job.cancel_event.is_set()
    assert job.status == STATUS_CANCELLED
    assert job.finished_at is not None


async def test_shutdown_with_no_job_is_a_no_op():
    runner = make_runner()
    await runner.shutdown()
    assert runner.current_job() is None


async def test_shutdown_survives_work_that_raised_a_base_exception():
    """``_run`` records what ``terminal_for`` does not claim and RE-RAISES it,
    so it travels out of the shielded task and into the await in ``shutdown``
    -- the one inside a lifespan shutdown hook. A catch that only covered
    ``Exception`` would let it out there and make "the server is coming down"
    a traceback on the way out, for a job that already reached a terminal
    state and told everyone watching.
    """
    class _OutOfBand(BaseException):
        """Private on purpose: nothing else may catch it by accident."""

    def doomed(emit: Emit, cancel_check: CancelCheck) -> None:
        raise _OutOfBand("the interpreter is going away")

    runner = make_runner()
    job = DemoJob(job_id=uuid.uuid4().hex, name="doomed")
    runner.claim(job, {"type": "job_started", "name": job.name})
    runner.start(job, doomed)

    await runner.shutdown()

    assert job.terminal
    assert job.status == STATUS_FAILED
    assert "_OutOfBand" in job.error["message"]


async def test_shutdown_is_bounded_when_the_work_ignores_cancellation():
    """Server shutdown must not be held hostage by stubborn work."""
    release = threading.Event()

    def deaf(emit: Emit, cancel_check: CancelCheck) -> None:
        release.wait(20)

    runner = make_runner(shutdown_timeout_s=0.2)
    job = DemoJob(job_id=uuid.uuid4().hex, name="deaf")
    runner.claim(job, {"type": "job_started", "name": job.name})
    runner.start(job, deaf)

    started = time.monotonic()
    try:
        await runner.shutdown()
        assert time.monotonic() - started < 5.0
        assert job.cancel_event.is_set()
    finally:
        # Let the stubborn work finish so no task outlives the test loop.
        release.set()
        await drain(runner, job.job_id)
