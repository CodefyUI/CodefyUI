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

A RESTART-MODE install has no flow and no thread at all. It cannot run inside
this process -- it would replace packages this process has already imported --
so ``_submit_restart`` writes down what to install, hands it to a detached
helper, records a job that is ALREADY terminal (``needs_restart``) and asks
the server to stop. The job exists so that the client has the same thing to
follow as for a live install: one id, one event stream, one terminal event
saying what happens next. What it follows afterwards is the server coming
back.

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
from .catalog import GPU_TORCH_PACK_ID, ModelItem, Pack, get_item
from .errors import (
    PackCancelled,
    PackInstallError,
    PackNeedsRestart,
    PendingExists,
    RestartRefused,
)

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
        runs_active: Callable[[], bool] | None = None,
        shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
    ) -> None:
        # Injectable so tests never run a real install, and so the CLI could
        # reuse the runner with a different flow.
        self._run_flow = run_flow
        # "Is a graph running?", which vetoes a restart-mode install -- see
        # ``_submit_restart``. INJECTED rather than answered from an
        # application object this class would then have to hold: the question
        # belongs to ``restart.runs_active(app)``, and ``main.py``'s lifespan
        # passes exactly that closure. Left out, the answer is "nothing is
        # running", which is what a service with no application behind it
        # (every test, and any future CLI use) is honestly entitled to say --
        # and it is the same answer ``restart.runs_active`` gives for an app
        # whose run service has not been built yet.
        self._runs_active = runs_active if runs_active is not None else (
            lambda: False)
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

        A RESTART-mode install carries no model items: the helper installs
        Python packages (or swaps the torch wheel) from an interpreter that
        has none of this app's downloader in it. ``item_ids`` is ignored on
        that path, and a pack whose models are also missing is finished with
        an ordinary live install afterwards.

        Only one thing here is awaited, and it is the branch that hands over
        to :meth:`_submit_restart` -- which returns or raises. Everything on
        the live path from the busy check to the ``self._job`` assignment
        therefore runs without a suspension point between them, which is the
        whole of what makes "one install at a time" true: two requests cannot
        both pass the check before either takes the slot.

        :raises PackBusy: an install is already running.
        :raises RestartUnavailable: a restart-mode install on a server that
            cannot restart itself. ``command`` is what to type instead.
        :raises RestartRefused: a restart this server COULD do, but not right
            now -- a graph is running, another restart is already claimed, or
            (for a LIVE install) a restart is already pending and this process
            is seconds from ending. Nothing was written down.
        :raises PackInstallError: the restart helper could not be started.
            The claim is withdrawn first, so a retry is not refused as "one
            is already pending".
        :raises PackInsufficientDisk: the disk cannot hold what was asked
            for. Checked HERE rather than in the job so the client is told
            by the response to its own request -- a 507 it can act on,
            instead of a job that starts, appears in the panel, and fails.
        :raises ValueError: an item id this pack does not have, or a
            restart-mode install with no packages to install.
        """
        running = self._job
        if running is not None and not running.terminal:
            raise PackBusy(running.job_id)

        # The pack's OWN mode decides as much as the request does. A pack
        # marked restart-mode is one whose install replaces something this
        # process has already imported, and running it live does not fail --
        # it succeeds having changed nothing, and the panel then reports a
        # GPU PyTorch install that never happened. So a restart-mode pack
        # takes this path whatever the client asked for.
        if mode == "restart" or pack.install_mode == "restart":
            return await self._submit_restart(pack, variant)

        # A restart-mode job is TERMINAL from the moment it is made -- the
        # check above waves the next install straight through -- but the
        # server it would run in is half a second from raising SIGINT on
        # itself. The flow would appear in the panel, start downloading and
        # die unfinished with the process. Refused rather than queued: there
        # is no queue to survive the restart, and the user can ask again on
        # the server that comes back.
        if (running is not None and running.mode == "restart"
                and running.status == STATUS_NEEDS_RESTART):
            raise RestartRefused(
                f"{pack.title} cannot be installed right now: this server is "
                f"restarting to finish another install",
                reason="a restart is already pending",
                # THIS pack's command, not the pending restart's. Every
                # refusal carries the line that gets the user what they just
                # asked for; ``running.restart_command`` installs the other
                # pack, and handing somebody a command for a package they
                # did not ask for is worse than handing them none.
                command=restart.install_command_for(pack))

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
                         "items": list(job.items)}, job)
        # job_started is in the buffer before the flow can emit anything, so
        # a client that polls from cursor 0 always sees it first.
        self._task = asyncio.create_task(self._run(job, pack, loop, broadcast))
        return job

    async def _submit_restart(self, pack: Pack, variant: str | None) -> PackJob:
        """Write the install down, start the helper, and stop the server.

        The one install that finishes by ending the process that started it.
        There is no flow and no worker thread: this method's whole job is the
        handshake, and the job it returns is already terminal
        (``needs_restart``) -- the SPA reads that status on a restart-mode
        job as "reload when the server answers again".

        PACKAGES ONLY. The helper runs from an interpreter with none of this
        app's downloader in it, so no model item can travel this path; the
        pending file carries pip specs or the torch index and nothing else.
        A pack that needs models as well is finished with an ordinary live
        install once the server is back, which is why the job's ``items`` is
        empty and there is no disk check here.

        The ORDER is not a preference. ``spawn_helper`` READS the pending
        file, so the claim is written first; the job is stored only after the
        helper is running, so a spawn that fails leaves no job in the panel
        for an install that will never happen; and the shutdown is scheduled
        last, delayed, so the 202 carrying the job id is out the door before
        the SIGINT lands.

        Every refusal happens BEFORE the pending file is written, and the one
        failure that can happen after it (the spawn) withdraws the claim on
        its way out. The alternative -- a pending file for a helper that was
        never started -- refuses every later restart with "one is already
        pending" until it goes stale fifteen minutes later.
        """
        # Off the loop, and first. Naming the GPU pack's command means asking
        # which wheel this machine should have, and answering that from a
        # cold cache runs nvidia-smi with a five second timeout. Every branch
        # below wants the answer -- each refusal quotes it and the terminal
        # event carries it -- so it is paid for exactly once here, which also
        # warms the memoised probe ``build_pending`` would otherwise run on
        # the loop thread.
        command = await asyncio.to_thread(
            restart.install_command_for, pack, variant)

        if not restart.restart_available():
            raise RestartUnavailable(
                f"{pack.title} cannot be installed from inside the running "
                f"server; run the command instead",
                command=command)

        kind = self._restart_kind(pack)

        # Nothing from here to the end of the method awaits, so the checks
        # below and the assignment of ``self._job`` cannot be interleaved
        # with another submit -- the same property that makes "one job at a
        # time" hold in ``submit_install``. The busy check is repeated
        # because the probe above IS a suspension point: a live install may
        # have started during it, and stopping the server would kill it.
        running = self._job
        if running is not None and not running.terminal:
            raise PackBusy(running.job_id)

        if self._runs_active():
            raise RestartRefused(
                f"{pack.title} cannot be installed while a graph is running: "
                f"the server has to restart, and the run would go with it",
                reason="a graph is running", command=command)

        loop = asyncio.get_running_loop()
        job = PackJob(job_id=uuid.uuid4().hex, pack_id=pack.pack_id,
                      items=(), mode="restart")
        # No items: the helper installs PACKAGES and nothing else -- it runs
        # from an interpreter with none of this app's downloader in it -- so
        # there is no download to disk-check and nothing to report progress
        # for. A pack whose models are also missing is installed live for
        # those, before or after.
        pending = restart.build_pending(pack, job_id=job.job_id, kind=kind,
                                        variant=variant)
        try:
            pending_path = restart.write_pending(pending)
        except PendingExists as exc:
            # Somebody else's claim, still live. Nothing has been changed and
            # nothing is lost; the next step is to wait for that one.
            raise RestartRefused(
                str(exc), reason="a restart-mode install is already pending",
                command=command, hint=exc.hint) from exc

        try:
            helper_pid = restart.spawn_helper(pending_path)
        except Exception as exc:
            # Withdraw the claim before saying anything: this server is still
            # running and can be retried, and a file nobody will act on would
            # refuse that retry for the next fifteen minutes.
            pending_path.unlink(missing_ok=True)
            log.exception("the restart helper for pack %s did not start",
                          pack.pack_id)
            raise PackInstallError(
                f"the restart helper could not be started, and nothing was "
                f"installed: {exc}") from exc

        log.info("restart-mode install of %s handed to helper pid %s (job %s)",
                 pack.pack_id, helper_pid, job.job_id)
        job.restart_command = command
        # There is no task for this job. Leaving the previous job's finished
        # one in place would have ``shutdown`` await something that belongs
        # to a job nobody can read any more.
        self._task = None

        broadcast = _Broadcast()
        with self._lock:
            self._job = job
            self._events.clear()
            self._cursor = 0
            self._broadcast = broadcast
            self._store({"type": "job_started", "pack_id": pack.pack_id,
                         "items": []}, job)
        # Terminal immediately, and under the lock like every other terminal
        # event: a client that sees ``needs_restart`` has already been handed
        # the event that names the command, the kind and the file.
        self._finish(job, STATUS_NEEDS_RESTART,
                     {"type": "needs_restart", "command": command,
                      "kind": kind, "pending_path": str(pending_path)})

        restart.schedule_self_shutdown(loop)
        return job

    @staticmethod
    def _restart_kind(pack: Pack) -> str:
        """``"torch"`` for the wheel swap, ``"pip"`` for a package install.

        The helper runs the two with different command lines -- one carries
        ``--index-url``, the other a list of PEP 508 specs -- and it has no
        catalog to look the pack up in, so the decision is made here and
        written into the pending file.

        ``catalog.GPU_TORCH_PACK_ID`` rather than a second copy of the
        literal: the catalog is where "which pack is which" is decided, and
        one pack id spelled in three files is two spellings too many.
        ``restart.install_command_for`` and ``state.pack_state`` ask the same
        question off the same constant.

        :raises ValueError: the pack has no packages to install. Stopping the
            server to install nothing is worse than refusing -- and it would
            report success, which is how a panel comes to claim an install
            that never happened. ``restart.build_pending`` refuses the same
            case; this one runs first, before a job id has even been minted,
            so the client is answered by its own request.
        """
        if pack.pack_id == GPU_TORCH_PACK_ID:
            return "torch"
        if not pack.pip:
            raise ValueError(
                f"pack {pack.pack_id!r} has no packages to install, so a "
                f"restart-mode install would stop the server to do nothing")
        return "pip"

    @staticmethod
    def _targets(pack: Pack, item_ids: Sequence[str] | None) -> list[ModelItem]:
        """Which items this job is for -- ``flows._resolve_items``'s rule.

        Resolved ONCE, here, and the result is what the flow is handed: the
        disk precheck, the ``job_started`` event and what actually downloads
        then cannot disagree with each other.

        ``None`` means "the whole pack, minus what is already downloaded":
        a learner adding a second embedding model must not re-fetch the
        first. An unknown id raises ``ValueError`` -- the same type, and the
        same message, ``flows._resolve_items`` raises for it, so a caller
        that handles one handles the other. The route turns it into a 400
        before anything is started.
        """
        if item_ids is None:
            return [item for item in pack.items
                    if not state.item_state(pack, item).present]

        items: list[ModelItem] = []
        for item_id in item_ids:
            try:
                items.append(get_item(pack, item_id))
            except KeyError as exc:
                raise ValueError(
                    f"pack {pack.pack_id!r} has no item {item_id!r}") from exc
        return items

    # ── running ───────────────────────────────────────────────────────────

    async def _run(self, job: PackJob, pack: Pack, loop, broadcast: _Broadcast
                   ) -> None:
        """Run the flow on a worker thread and record how it ended.

        Swallows every ``Exception``: this is a bare task nobody awaits
        except ``shutdown``, and an escaping one would be reported as "task
        exception was never retrieved" long after the job disappeared from
        the UI.

        A ``BaseException`` -- KeyboardInterrupt, SystemExit, CancelledError
        -- is recorded and then RE-RAISED. Those are not ours to swallow, and
        the job reaches a terminal state either way, which is the part that
        matters to anyone still watching it.
        """
        emit = self._emitter(job, loop, broadcast)
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
            event = {"type": "needs_restart", "command": exc.command}
            if restart.restart_available():
                # There is a button for this, not only a command to paste:
                # the same install, submitted with ``mode="restart"``, is
                # exactly what this server would do about it. The key is
                # ABSENT rather than null where there is no such button, so
                # the client's question is "is retry_mode set?" and not "is
                # it set and also not null".
                event["retry_mode"] = "restart"
            self._finish(job, STATUS_NEEDS_RESTART, event)
        except PackInstallError as exc:
            self._fail(job, str(exc), exc.hint)
        except Exception as exc:
            # Not a failure shape anybody designed, so it goes to the log in
            # full and to the user as its repr -- str() of a bare KeyError is
            # just a quoted key and says nothing about what broke.
            log.exception("pack install job %s raised", job.job_id)
            self._fail(job, repr(exc), None)
        except BaseException as exc:
            # KeyboardInterrupt, SystemExit, CancelledError. The job is over
            # either way, and a job left saying "running" forever is worse
            # than one that says why it stopped -- the panel would offer a
            # Stop button for a thread that no longer exists, and no submit
            # would ever be accepted again. Record, then let it travel: these
            # are not ours to swallow.
            log.warning("pack install job %s ended on %s", job.job_id,
                        type(exc).__name__)
            self._fail(job, repr(exc), None)
            raise
        else:
            self._finish(job, STATUS_DONE, {"type": "job_done"})
        finally:
            # Finished, failed or cancelled, the disk is not what it was and
            # the next status poll has to see that. The flow invalidates too;
            # doing it here as well covers an injected flow and costs one
            # dict drop.
            state.invalidate()

    def _emitter(self, job: PackJob, loop, broadcast: _Broadcast
                 ) -> Callable[[dict], None]:
        """The ``emit`` the flow is handed. Called from the worker thread.

        Closes over the JOB as well as its generation's loop and broadcast,
        so an event says which job produced it. A closure is the only thing
        that can: the caller is a reader thread that may outlive the job --
        ``runner._pump`` is a daemon thread joined with a timeout -- and by
        the time its event reaches ``_store`` the service's idea of the
        current job may already be somebody else's.
        """

        def emit(event: dict) -> None:
            with self._lock:
                self._store(event, job)
            try:
                loop.call_soon_threadsafe(broadcast.notify)
            except RuntimeError:
                # The loop is closed (the server is going down, or a test
                # ended while the thread was still unwinding). Nobody is left
                # to wake; the event is stored either way.
                pass

        return emit

    def _store(self, event: dict, job: PackJob | None) -> None:
        """Stamp *event* and buffer it. CALLER HOLDS ``self._lock``.

        *job* is the job that PRODUCED the event, which is not always the
        current one: a finished job's reader thread can still be emitting.
        Buffering is unconditional -- the buffer belongs to whatever job is
        current, and a stray line in it is visible and harmless -- but the
        step bookkeeping is not. ``current_step`` is what
        ``GET /api/packs`` turns into an item's "downloading" badge, so a
        late event stamping it on the next job would put that badge on an
        item of a pack the old job never touched.
        """
        self._cursor += 1
        self._events.append({**event, "cursor": self._cursor, "ts": _now_iso()})

        if job is None or job is not self._job:
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

        The other caller is ``_submit_restart``, where that argument runs the
        other way round and still holds: the generation was replaced by the
        submit itself, moments earlier and under the same lock, and this job
        was terminal on arrival. It has no ``_run`` and no worker thread --
        the install happens in another process, after this one is gone -- so
        this is the only place its status is ever set.
        """
        with self._lock:
            self._store(event, job)
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

        Nothing the awaited task raises escapes. ``_run`` swallows every
        ``Exception`` itself but deliberately RE-RAISES a ``BaseException``
        (KeyboardInterrupt, SystemExit, CancelledError) after recording the
        job, and this is the one place that await happens -- inside the
        lifespan shutdown hook. Letting one through would turn "the server
        is coming down" into a traceback on the way out, of an install that
        has already reached a terminal state and told everyone watching.
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
        except BaseException:
            # BaseException, not Exception: _run re-raises those on purpose
            # (see the docstring), so ``except Exception`` here catches only
            # the cases _run already handled and misses the ones it forwards.
            log.exception("pack install job failed during shutdown")
