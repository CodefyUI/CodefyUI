"""One pack install at a time, off the event loop, readable from anywhere.

``flows.install_pack_live`` is synchronous and blocks its thread for minutes.
This module is the seam between that and an HTTP server that must keep
answering while it runs:

* the flow runs in ``asyncio.to_thread`` inside a background task, so the
  loop is never blocked by pip or by a 470 MB download;
* its ``emit`` callback is called from THAT thread, so every event is
  appended under a ``threading.Lock`` and the loop is woken through
  ``call_soon_threadsafe`` -- the same hand-off ``execution_context`` uses;
* every stored event gets a ``cursor`` and a ``ts``, so a client that
  reconnects (or opens a second tab) replays from where it left off rather
  than from nothing.

ONE JOB AT A TIME, deliberately. Two concurrent ``uv pip install`` runs share
one site-packages and one cache lock, and the honest outcomes are a corrupt
environment or a deadlock. A second submit is refused with the id of the job
already running, which is what the UI needs to show "an install is already
in progress" and offer to follow it.

The finished job STAYS. The last thing a client asks for is the tail of a job
that just ended -- the failure message, the final log lines -- so the job and
its events live until the next submit replaces them.

Terminal events are appended BEFORE the status flips, under the same lock, so
a reader that sees ``status != "running"`` has already been given the event
that says why. The other order loses the last event of every job to whoever
polls at the wrong moment.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import download, flows, restart, state
from .catalog import ModelItem, Pack, get_item
from .errors import PackCancelled, PackInstallError, PackNeedsRestart

log = logging.getLogger(__name__)

#: Events kept for the current job. A pip run over a slow link emits a few
#: hundred lines and a download a few hundred progress frames; four thousand
#: holds a whole noisy install, and dropping the oldest is the right loss --
#: a client that has fallen four thousand events behind is not reading them.
MAX_EVENTS = 4000

#: How long ``shutdown`` waits for a cancelled job to unwind before it stops
#: waiting. The flow checks for cancellation four times a second, so ten
#: seconds is "something is stuck", not "this machine is slow" -- and the
#: server has to come down either way.
SHUTDOWN_TIMEOUT_S = 10.0

#: The four states a job can end in, plus the one it starts in.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_NEEDS_RESTART = "needs_restart"


class PackBusy(Exception):
    """An install is already running. ``job_id`` is the one to follow."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"a pack install is already running (job {job_id})")


class UnknownJob(KeyError):
    """No job with this id. Only the most recent job is kept."""


class RestartUnavailable(Exception):
    """This server cannot restart itself; ``command`` is what to type."""

    def __init__(self, message: str, *, command: str):
        self.command = command
        super().__init__(message)


class _Broadcast:
    """Edge-triggered wake-up for long pollers. No busy waiting anywhere.

    Copied from ``app.core.run_service._Broadcast`` rather than imported: it
    is private to that module, and a fifteen-line copy is cheaper than a
    cross-module dependency on somebody else's internals.

    ``asyncio.Event`` is level-triggered and single-shot; a shared one would
    have to be cleared, and whoever clears it races everyone who has not
    woken yet. Instead each ``notify`` SETS the current event and installs a
    fresh one for the next generation. A waiter captures the event object
    BEFORE it reads the buffer, so a notification that lands during that read
    still wakes it -- the lost-wakeup window is closed by ordering, not by a
    timeout.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def waiter(self) -> asyncio.Event:
        """The generation to wait on. Capture it before reading the buffer."""
        return self._event

    def notify(self) -> None:
        event, self._event = self._event, asyncio.Event()
        event.set()


@dataclass
class PackJob:
    """One install, from submit to terminal event."""

    job_id: str
    pack_id: str
    #: The items this job will fetch, resolved at submit time (see
    #: ``PackService._targets``) -- never the caller's ``None``.
    items: tuple[str, ...]
    mode: str
    status: str = STATUS_RUNNING
    #: The flow's current step name (``"pip"``, ``"download:<item>"``, ...),
    #: which is how ``GET /api/packs`` knows which model is downloading.
    current_step: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    #: Set by ``cancel`` and read by the flow's own ``cancel_check``. A
    #: threading primitive because the flow reads it from its worker thread.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    error: dict | None = None
    restart_command: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status != STATUS_RUNNING


def _now_iso() -> str:
    """An ISO-8601 UTC timestamp, so two clients can order one job's log."""
    return datetime.now(timezone.utc).isoformat()


class PackService:
    """The Package Center's job runner. One instance on ``app.state``."""

    def __init__(
        self,
        *,
        run_flow: Callable[..., object] = flows.install_pack_live,
        shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
    ) -> None:
        # Injectable so tests never run a real install, and so the CLI could
        # reuse the runner with a different flow.
        self._run_flow = run_flow
        self._shutdown_timeout_s = shutdown_timeout_s
        # Guards the buffer, the cursor and every mutation of the current
        # job's status. Held by both the loop and the flow's worker thread,
        # so it is a threading.Lock and never an asyncio one.
        self._lock = threading.Lock()
        self._job: PackJob | None = None
        self._events: "deque[dict]" = deque(maxlen=MAX_EVENTS)
        self._cursor = 0
        self._task: "asyncio.Task | None" = None
        self._broadcast = _Broadcast()

    # ── reading ───────────────────────────────────────────────────────────

    def current_job(self) -> PackJob | None:
        """The most recent job, running or not. None before the first submit."""
        return self._job

    def get_job(self, job_id: str) -> PackJob:
        """The job with this id. Raises :class:`UnknownJob` for any other."""
        job = self._job
        if job is None or job.job_id != job_id:
            raise UnknownJob(job_id)
        return job

    async def wait_for_events(
        self,
        job_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 500,
        wait: float = 0.0,
    ) -> tuple[list[dict], int, str]:
        """Events after *after_cursor*, optionally long-polling for the next.

        Returns ``(events, cursor, status)``. The cursor tracks what was
        actually RETURNED, so an empty page never moves a follower forward
        past events it did not receive.

        Order is load-bearing: the wake-up generation is captured BEFORE the
        buffer is read, so an event that lands during the read still
        satisfies the wait instead of being missed until the next one.

        Returns immediately -- possibly empty -- once the job is terminal. A
        long poll on a finished job must never hang for its full timeout.
        """
        job = self.get_job(job_id)
        waiter = self._broadcast.waiter()
        events, cursor, status = self._read(job, after_cursor, limit)
        if events or status != STATUS_RUNNING or wait <= 0:
            return events, cursor, status

        try:
            await asyncio.wait_for(waiter.wait(), wait)
        except asyncio.TimeoutError:
            pass
        return self._read(job, after_cursor, limit)

    def _read(self, job: PackJob, after_cursor: int, limit: int
              ) -> tuple[list[dict], int, str]:
        """One consistent look at the buffer AND the job's status.

        Both under the same lock hold, which is what makes "terminal status
        implies the terminal event is already readable" true for a caller.

        The identity check closes a narrow window in ``wait_for_events``: a
        poll parked on a job that finishes, followed by a submit that clears
        the buffer, would otherwise resume and hand the NEW job's events back
        under the OLD job's id. There are no events for that job any more, so
        the honest answer is the same one an id we never had gets.
        """
        with self._lock:
            if self._job is not job:
                raise UnknownJob(job.job_id)
            page = [event for event in self._events
                    if event["cursor"] > after_cursor][:limit]
            status = job.status
        return page, (page[-1]["cursor"] if page else after_cursor), status

    # ── submitting ────────────────────────────────────────────────────────

    async def submit_install(
        self,
        pack: Pack,
        item_ids: Sequence[str] | None,
        *,
        mode: str = "live",
        variant: str | None = None,
    ) -> PackJob:
        """Start installing *pack*. Returns the job; raises before starting one.

        :raises PackBusy: an install is already running.
        :raises RestartUnavailable: ``mode="restart"`` on a server that
            cannot restart itself (every server, in this release).
        :raises PackInsufficientDisk: the disk cannot hold what was asked
            for. Checked HERE rather than in the job so the client is told
            by the response to its own request -- a 507 it can act on,
            instead of a job that starts, appears in the panel, and fails.
        :raises KeyError: an item id this pack does not have.
        """
        running = self._job
        if running is not None and not running.terminal:
            raise PackBusy(running.job_id)

        if mode == "restart" and not restart.restart_available():
            # PR 5 replaces this branch with the real restart handshake.
            raise RestartUnavailable(
                f"{pack.title} cannot be installed from inside the running "
                f"server; run the command instead",
                command=restart.install_command_for(pack, variant))

        targets = self._targets(pack, item_ids)
        download.check_disk(targets)

        job = PackJob(job_id=uuid.uuid4().hex, pack_id=pack.pack_id,
                      items=tuple(item.item_id for item in targets), mode=mode)
        loop = asyncio.get_running_loop()
        broadcast = _Broadcast()

        with self._lock:
            self._job = job
            self._events.clear()
            self._cursor = 0
            self._broadcast = broadcast
            self._store({"type": "job_started", "pack_id": pack.pack_id,
                         "items": list(job.items)})
        # job_started is in the buffer before the flow can emit anything, so
        # a client that polls from cursor 0 always sees it first.
        self._task = asyncio.create_task(self._run(job, pack, loop, broadcast))
        return job

    @staticmethod
    def _targets(pack: Pack, item_ids: Sequence[str] | None) -> list[ModelItem]:
        """Which items this job is for -- ``flows._resolve_items``'s rule.

        Resolved ONCE, here, and the result is what the flow is handed: the
        disk precheck, the ``job_started`` event and what actually downloads
        then cannot disagree with each other.

        ``None`` means "the whole pack, minus what is already downloaded":
        a learner adding a second embedding model must not re-fetch the
        first. An unknown id raises ``KeyError`` -- a caller mistake, and the
        route turns it into a 400 before anything is started.
        """
        if item_ids is None:
            return [item for item in pack.items
                    if not state.item_state(pack, item).present]
        return [get_item(pack, item_id) for item_id in item_ids]

    # ── running ───────────────────────────────────────────────────────────

    async def _run(self, job: PackJob, pack: Pack, loop, broadcast: _Broadcast
                   ) -> None:
        """Run the flow on a worker thread and record how it ended.

        Never raises: this is a bare task nobody awaits except ``shutdown``,
        and an escaping exception would be reported as "task exception was
        never retrieved" long after the job disappeared from the UI.
        """
        emit = self._emitter(loop, broadcast)
        try:
            await asyncio.to_thread(
                self._run_flow, pack, list(job.items),
                emit=emit, cancel_check=job.cancel_event.is_set)
        except PackCancelled:
            self._finish(job, STATUS_CANCELLED, {"type": "job_cancelled"})
        except PackNeedsRestart as exc:
            # Checked before PackInstallError: it is a subclass, and "not
            # while the server is running" is not a failure.
            job.restart_command = exc.command
            self._finish(job, STATUS_NEEDS_RESTART,
                         {"type": "needs_restart", "command": exc.command})
        except PackInstallError as exc:
            self._fail(job, str(exc), exc.hint)
        except Exception as exc:
            # Not a failure shape anybody designed, so it goes to the log in
            # full and to the user as its repr -- str() of a bare KeyError is
            # just a quoted key and says nothing about what broke.
            log.exception("pack install job %s raised", job.job_id)
            self._fail(job, repr(exc), None)
        else:
            self._finish(job, STATUS_DONE, {"type": "job_done"})
        finally:
            # Finished, failed or cancelled, the disk is not what it was and
            # the next status poll has to see that. The flow invalidates too;
            # doing it here as well covers an injected flow and costs one
            # dict drop.
            state.invalidate()

    def _emitter(self, loop, broadcast: _Broadcast) -> Callable[[dict], None]:
        """The ``emit`` the flow is handed. Called from the worker thread."""

        def emit(event: dict) -> None:
            with self._lock:
                self._store(event)
            try:
                loop.call_soon_threadsafe(broadcast.notify)
            except RuntimeError:
                # The loop is closed (the server is going down, or a test
                # ended while the thread was still unwinding). Nobody is left
                # to wake; the event is stored either way.
                pass

        return emit

    def _store(self, event: dict) -> None:
        """Stamp *event* and buffer it. CALLER HOLDS ``self._lock``."""
        self._cursor += 1
        self._events.append({**event, "cursor": self._cursor, "ts": _now_iso()})

        job = self._job
        if job is None:
            return
        kind = event.get("type")
        if kind == "step_started":
            job.current_step = event.get("step")
        elif kind == "step_done" and job.current_step == event.get("step"):
            job.current_step = None

    def _finish(self, job: PackJob, status: str, event: dict) -> None:
        """Record the terminal event and the status, atomically.

        The event goes in FIRST and the status flips under the same lock, so
        a poller that sees a terminal status has already been handed the
        event explaining it.

        Called from ``_run``, which runs ON the loop, so the wake-up needs no
        thread hand-off. ``self._broadcast`` is still this job's generation:
        only a submit replaces it, and a submit is refused until the current
        job is terminal -- which is what this method is about to make true.
        """
        with self._lock:
            self._store(event)
            job.status = status
            job.finished_at = time.time()
        self._broadcast.notify()

    def _fail(self, job: PackJob, message: str, hint: str | None) -> None:
        job.error = {"message": message, "hint": hint}
        self._finish(job, STATUS_FAILED,
                     {"type": "job_failed", "message": message, "hint": hint})

    # ── stopping ──────────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> bool:
        """Ask the job to stop. False when it had already finished.

        Cooperative: the flow notices between steps and inside a download,
        and it is the FLOW that ends the job, so the status may still say
        running when this returns. Asking twice is not an error.
        """
        job = self.get_job(job_id)
        if job.terminal:
            return False
        job.cancel_event.set()
        return True

    async def shutdown(self) -> None:
        """Stop a running install and wait for it, bounded.

        Bounded because the server is coming down either way: a flow that
        ignores its cancel check must not be able to hold the process open.
        The task is SHIELDED from the timeout -- cancelling it would not stop
        the worker thread anyway, and would leave the job with no terminal
        event at all.
        """
        job, task = self._job, self._task
        if job is not None and not job.terminal:
            job.cancel_event.set()
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task),
                                   self._shutdown_timeout_s)
        except asyncio.TimeoutError:
            log.warning("pack install job did not stop within %.0fs; "
                        "leaving it to the interpreter",
                        self._shutdown_timeout_s)
        except Exception:  # pragma: no cover - _run swallows its own errors
            log.exception("pack install job failed during shutdown")
