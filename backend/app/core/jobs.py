"""One job at a time, off the event loop, readable from anywhere.

Installing a pack and installing a plugin are the same shape of problem: a
synchronous function that blocks its thread for minutes, inside a server that
has to keep answering while it runs. This module is the seam between the two,
and it is deliberately the only one -- a second copy of these lock rules would
drift from this one, and the drift would stay invisible until a client lost
the last event of a job:

* the work runs in ``asyncio.to_thread`` inside a background task, so the
  loop is never blocked by pip or by a 470 MB download;
* its ``emit`` callback is called from THAT thread, so every event is
  appended under a ``threading.Lock`` and the loop is woken through
  ``call_soon_threadsafe`` -- the same hand-off ``execution_context`` uses;
* every stored event gets a ``cursor`` and a ``ts``, so a client that
  reconnects (or opens a second tab) replays from where it left off rather
  than from nothing.

ONE JOB AT A TIME, deliberately. Every caller here mutates one shared thing --
an interpreter's site-packages, a plugins directory -- and two concurrent
writers to it end in a corrupt environment or a deadlock. A second claim is
refused with the id of the job already running, which is what the UI needs to
show "an install is already in progress" and offer to follow it.

The finished job STAYS. The last thing a client asks for is the tail of a job
that just ended -- the failure message, the final log lines -- so the job and
its events live until the next claim replaces them.

Terminal events are appended BEFORE the status flips, under the same lock, so
a reader that sees ``status != "running"`` has already been given the event
that says why. The other order loses the last event of every job to whoever
polls at the wrong moment.

What a job MEANS is not here. Which exception is a cancel and which is a
failure, what an event's payload says, whether there is anything to run at
all -- those belong to the domain, and they arrive as the ``terminal_for``
callback and the ``work`` function. ``app.core.packs.service`` is the first
caller; this module imports none of it, and must not.

``run_service`` keeps its own, older fan-out for graph runs -- many runs at
once, a sqlite store, per-run subscriber sets -- and is deliberately left
alone. This is the single-slot runner; merging the two would be a redesign of
both, and a redesign of the run queue is not what an install needs.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)

#: Events kept for the current job. A pip run over a slow link emits a few
#: hundred lines and a download a few hundred progress frames; four thousand
#: holds a whole noisy install, and dropping the oldest is the right loss --
#: a client that has fallen four thousand events behind is not reading them.
MAX_EVENTS = 4000

#: How long ``shutdown`` waits for a cancelled job to unwind before it stops
#: waiting. The work checks for cancellation four times a second, so ten
#: seconds is "something is stuck", not "this machine is slow" -- and the
#: server has to come down either way.
SHUTDOWN_TIMEOUT_S = 10.0

#: The four states a job can end in, plus the one it starts in.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_NEEDS_RESTART = "needs_restart"

#: What the work is handed. ``emit`` buffers one event and wakes the pollers;
#: ``cancel_check`` is the question the work asks between its own steps.
Emit = Callable[[dict], None]
CancelCheck = Callable[[], bool]

#: The unit of work itself, run on a worker thread. It returns nothing: a job
#: says how it went through its events and its status, not through a value.
Work = Callable[[Emit, CancelCheck], None]

#: The last step of a job that CANNOT run on the worker thread -- a registry
#: the loop reads, an app state -- run on the loop once the work has returned.
#: What it hands back is merged into the terminal ``job_done`` event, so the
#: client learns what that step found out without asking a second question.
AfterWork = Callable[[], Awaitable[dict] | dict]

#: How a domain reads an exception out of its own work: ``(status, event)``
#: for an outcome it recognises, ``None`` for one it does not (see
#: :meth:`JobRunner._run`).
#:
#: TWO shapes, and the runner picks between them ONCE, at construction (see
#: :func:`_wants_the_job`). A translator that finds its job through
#: ``current_job()`` -- true for a domain whose submit is refused until the
#: current job is terminal -- declares just the exception, which is what
#: every caller before the Plugin Center did. One that would rather be told
#: declares the job first. ``Job`` is quoted because it is defined below
#: this block, and ``from __future__ import annotations`` defers ANNOTATIONS,
#: not the right-hand side of an assignment.
TerminalFor = (Callable[[BaseException], tuple[str, dict] | None]
               | Callable[["Job", BaseException], tuple[str, dict] | None])


class JobBusy(Exception):
    """A job is already running. ``job_id`` is the one to follow.

    *message* is there for a subclass that says the same thing in its own
    domain's words ("a pack install is already running"): the id a client
    has to follow is the part that must not vary, and the sentence around it
    is the part that should.
    """

    def __init__(self, job_id: str, message: str | None = None):
        self.job_id = job_id
        super().__init__(message
                         or f"a job is already running (job {job_id})")


class UnknownJob(KeyError):
    """No job with this id. Only the most recent job is kept."""


class Broadcast:
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


@dataclass(kw_only=True)
class Job:
    """One unit of work, from claim to terminal event.

    Keyword-only, and a subclass declares ``kw_only=True`` too: a job is made
    in one place per domain and read everywhere, so the cost of naming the
    fields at that one construction site buys a subclass the freedom to add
    its own -- required ones included -- without having to interleave them
    with the base's defaults.
    """

    job_id: str
    status: str = STATUS_RUNNING
    #: The work's current step name (``"pip"``, ``"download:<item>"``, ...),
    #: which is how ``GET /api/packs`` knows which model is downloading.
    current_step: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    #: Set by ``cancel`` and read by the work's own ``cancel_check``. A
    #: threading primitive because the work reads it from its worker thread.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    error: dict | None = None

    @property
    def terminal(self) -> bool:
        return self.status != STATUS_RUNNING


def _now_iso() -> str:
    """An ISO-8601 UTC timestamp, so two clients can order one job's log."""
    return datetime.now(timezone.utc).isoformat()


def _wants_the_job(terminal_for: TerminalFor) -> bool:
    """Does this translator take ``(job, exc)`` rather than just ``(exc)``?

    Read ONCE, when the runner is built, and never again: the answer cannot
    change, and asking it inside the ``except`` block that is a job's last
    chance to reach a terminal state would put a reflection call on the one
    path that must not fail.

    Only POSITIONAL parameters count, because those are the ones the runner
    passes. A bound method's ``self`` is already gone by the time
    :func:`inspect.signature` answers, so ``PackService._terminal_for(exc)``
    reads as one -- which is what it is from here.

    Anything whose signature cannot be read at all (a builtin, a C callable,
    a ``*args`` catch-all) is treated as the one-argument form: that is the
    shape every caller had before this second one existed, and guessing the
    other way would break them with a ``TypeError`` raised from inside the
    failure path.
    """
    try:
        parameters = inspect.signature(terminal_for).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in parameters
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2


class JobRunner:
    """The single job slot: one at a time, with a replayable event stream.

    Holds the lock discipline and nothing else. What a job is FOR belongs to
    whoever built the runner: it hands in ``terminal_for`` (what an exception
    from its own work means) at construction and a ``work`` function per job.
    """

    def __init__(
        self,
        *,
        terminal_for: TerminalFor,
        label: str = "job",
        max_events: int = MAX_EVENTS,
        shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
    ) -> None:
        # What an exception out of the work did to the job. Injected because
        # the runner must not know one domain's error classes from another's
        # -- see :meth:`_run` for what a ``None`` answer means.
        self._terminal_for = terminal_for
        # Which of the two shapes it declared. Settled here, so the failure
        # path never has to ask.
        self._terminal_for_wants_the_job = _wants_the_job(terminal_for)
        # What to call a job in the log. The runner's own two lines are about
        # a job nobody is watching any more (a shutdown that timed out, a
        # shutdown that caught something), so they have to say WHICH runner
        # they came from -- there is more than one in this process.
        self._label = label
        self._shutdown_timeout_s = shutdown_timeout_s
        # Guards the buffer, the cursor and every mutation of the current
        # job's status. Held by both the loop and the work's worker thread,
        # so it is a threading.Lock and never an asyncio one.
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._events: "deque[dict]" = deque(maxlen=max_events)
        self._cursor = 0
        self._task: "asyncio.Task | None" = None
        self._broadcast = Broadcast()

    # ── reading ───────────────────────────────────────────────────────────

    def current_job(self) -> Job | None:
        """The most recent job, running or not. None before the first claim."""
        return self._job

    def get_job(self, job_id: str) -> Job:
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

    def _read(self, job: Job, after_cursor: int, limit: int
              ) -> tuple[list[dict], int, str]:
        """One consistent look at the buffer AND the job's status.

        Both under the same lock hold, which is what makes "terminal status
        implies the terminal event is already readable" true for a caller.

        The identity check closes a narrow window in ``wait_for_events``: a
        poll parked on a job that finishes, followed by a claim that clears
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

    # ── claiming and running ──────────────────────────────────────────────

    def claim(self, job: Job, first_event: dict) -> None:
        """Take the one job slot for *job*, or refuse with :class:`JobBusy`.

        The check and the assignment happen with NO ``await`` between them --
        they are one synchronous method for exactly that reason -- which is
        the whole of what makes "one job at a time" true: two requests cannot
        both pass the check before either takes the slot. A caller that wants
        to refuse in its own vocabulary (``PackBusy``) checks earlier, before
        it has resolved anything; this is the check that actually hands over
        the slot.

        *first_event* is stored as cursor 1 under the same lock hold, so it
        is in the buffer before the work can emit anything and a client that
        polls from cursor 0 always sees it first.

        The previous job's task is dropped here as well. Leaving a finished
        one in place would have ``shutdown`` await something that belongs to
        a job nobody can read any more -- and a job that is terminal on
        arrival, because its work happens in another process, never gets a
        task of its own at all.
        """
        running = self._job
        if running is not None and not running.terminal:
            raise JobBusy(running.job_id)

        broadcast = Broadcast()
        with self._lock:
            self._job = job
            self._events.clear()
            self._cursor = 0
            self._broadcast = broadcast
            self._task = None
            self._store(first_event, job)

    def start(self, job: Job, work: Work, *,
              after_work: AfterWork | None = None,
              on_settled: Callable[[], None] | None = None) -> asyncio.Task:
        """Run *work* on a worker thread, in a task that records how it ended.

        Call it straight after :meth:`claim`, with no ``await`` in between:
        the generation the emitter closes over is read from ``self`` here,
        and a claim that landed in between would have replaced it -- this
        job's events would then wake the pollers parked on somebody else's.

        Three hooks, in the order they happen, and the order matters:

        1. *work*, on a worker thread, for minutes;
        2. *after_work*, ON THE LOOP, once the work has returned normally --
           the last step of the job for anything the worker thread may not
           touch. Its answer is merged into the terminal event;
        3. *on_settled*, on the loop, once the job is terminal HOWEVER it
           ended -- including the re-raised ``BaseException`` path. A domain
           that has caches to drop after a job (the disk is not what it was)
           drops them there.
        """
        loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(
            self._run(job, work, loop, self._broadcast, on_settled,
                      after_work))
        return self._task

    async def _run(self, job: Job, work: Work,
                   loop: asyncio.AbstractEventLoop, broadcast: Broadcast,
                   on_settled: Callable[[], None] | None,
                   after_work: AfterWork | None = None) -> None:
        """Run the work on a worker thread and record how it ended.

        Swallows every exception ``terminal_for`` recognises: this is a bare
        task nobody awaits except ``shutdown``, and an escaping one would be
        reported as "task exception was never retrieved" long after the job
        disappeared from the UI.

        Anything ``terminal_for`` does NOT recognise -- it answers ``None``,
        which for every domain so far means a ``BaseException``:
        KeyboardInterrupt, SystemExit, CancelledError -- is recorded as
        failed and then RE-RAISED. Those are not ours to swallow, and the job
        reaches a terminal state either way, which is the part that matters
        to anyone still watching it. A job left saying "running" forever is
        worse than one that says why it stopped: the panel would offer a Stop
        button for a thread that no longer exists, and no claim would ever be
        accepted again.

        ``terminal_for`` is called from inside the ``except`` block, so it
        may use ``log.exception`` and it sees the live exception context. It
        is DOMAIN code, though, and domain code has bugs: one that raises
        used to escape from here with the status still on ``running``. It is
        wrapped for that reason, and the job fails on the translator's own
        exception -- that is the bug somebody has to fix -- with the work's
        exception as the hint, since that is what the person watching was
        actually waiting for. The translator's exception is then SWALLOWED:
        the recorded job IS the report, and letting it travel would only
        take down the task ``shutdown`` awaits.

        The whole ordering, in one place, because every part of it is
        load-bearing::

            work (worker thread)
              -> after_work (this loop)
                -> terminal event stored, THEN the status flips (one lock)
                  -> on_settled

        ``after_work`` is the seam for a step that must not run on the worker
        thread -- re-discovering a node registry the loop reads. It runs only
        when the work returned NORMALLY, its dict answer is merged into
        ``job_done``, and if it raises the job is FAILED rather than done: it
        is the domain saying the job did not finish, and reporting success
        there would leave a panel showing something that cannot be loaded.
        """
        emit = self._emitter(job, loop, broadcast)
        try:
            await asyncio.to_thread(work, emit, job.cancel_event.is_set)
        except BaseException as exc:
            try:
                outcome = (self._terminal_for(job, exc)
                           if self._terminal_for_wants_the_job
                           else self._terminal_for(exc))
            except BaseException as translator_exc:
                log.exception("%s: terminal_for raised while recording how "
                              "job %s ended", self._label, job.job_id)
                self._fail(job, repr(translator_exc), repr(exc))
                return
            if outcome is None:
                self._fail(job, repr(exc), None)
                raise
            self._finish(job, *outcome)
        else:
            done = {"type": "job_done"}
            if after_work is not None:
                try:
                    result = after_work()
                    if inspect.isawaitable(result):
                        result = await result
                    # The runner's key wins: a client that has been promised
                    # ``job_done`` on success must get it, whatever a domain
                    # happens to put in the dict it hands back.
                    done = {**(result or {}), "type": "job_done"}
                except BaseException as exc:
                    self._fail(job, str(exc) or repr(exc), None)
                    if not isinstance(exc, Exception):
                        # Not ours to swallow -- the same rule as above, and
                        # the job has already been recorded either way.
                        raise
                    return
            self._finish(job, STATUS_DONE, done)
        finally:
            if on_settled is not None:
                on_settled()

    def _emitter(self, job: Job, loop: asyncio.AbstractEventLoop,
                 broadcast: Broadcast) -> Emit:
        """The ``emit`` the work is handed. Called from the worker thread.

        Closes over the JOB as well as its generation's loop and broadcast,
        so an event says which job produced it. A closure is the only thing
        that can: the caller is a reader thread that may outlive the job --
        ``packs.runner._pump`` is a daemon thread joined with a timeout --
        and by the time its event reaches ``_store`` the runner's idea of the
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

    def _store(self, event: dict, job: Job | None) -> None:
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

    def finish(self, job: Job, status: str, event: dict) -> None:
        """Record the terminal state of a job whose work is not ours to run.

        For an install that happens after this server exits, in a helper it
        just started: there is no worker thread and no task, so nothing will
        ever reach :meth:`_finish` for it on its own. The job is claimed
        anyway, so that a client has the same thing to follow as for any
        other -- one id, one event stream, one terminal event saying what
        happens next -- and it is terminal from the moment it is made.
        """
        self._finish(job, status, event)

    def _finish(self, job: Job, status: str, event: dict) -> None:
        """Record the terminal event and the status, atomically.

        The event goes in FIRST and the status flips under the same lock, so
        a poller that sees a terminal status has already been handed the
        event explaining it.

        Called from ``_run``, which runs ON the loop, so the wake-up needs no
        thread hand-off. ``self._broadcast`` is still this job's generation:
        only a claim replaces it, and a claim is refused until the current
        job is terminal -- which is what this method is about to make true.

        The other caller is :meth:`finish`, where that argument runs the
        other way round and still holds: the generation was replaced by the
        claim itself, moments earlier and under the same lock, and that job
        was terminal on arrival. It has no ``_run`` and no worker thread, so
        this is the only place its status is ever set.

        A FAILED job also gets its ``error``, read out of the event it was
        given. "Why did it fail" is asked of the job (the panel's summary)
        and of the event stream (the log), and two writers for one answer is
        how they come to disagree -- so a terminal event with
        ``STATUS_FAILED`` carries ``message`` and ``hint``, and the job's
        copy is made from it, here, under the lock that publishes both.
        """
        with self._lock:
            self._store(event, job)
            if status == STATUS_FAILED:
                job.error = {"message": event.get("message"),
                             "hint": event.get("hint")}
            job.status = status
            job.finished_at = time.time()
        self._broadcast.notify()

    def _fail(self, job: Job, message: str, hint: str | None) -> None:
        """Finish *job* as failed, with the runner's own ``job_failed`` event.

        The shape every domain's failure uses, so that a client which can
        render one runner's failure can render the other's.
        """
        self._finish(job, STATUS_FAILED,
                     {"type": "job_failed", "message": message, "hint": hint})

    # ── stopping ──────────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> bool:
        """Ask the job to stop. False when it had already finished.

        Cooperative: the work notices between its own steps, and it is the
        WORK that ends the job, so the status may still say running when this
        returns. Asking twice is not an error.
        """
        job = self.get_job(job_id)
        if job.terminal:
            return False
        job.cancel_event.set()
        return True

    async def shutdown(self) -> None:
        """Stop a running job and wait for it, bounded.

        Bounded because the server is coming down either way: work that
        ignores its cancel check must not be able to hold the process open.
        The task is SHIELDED from the timeout -- cancelling it would not stop
        the worker thread anyway, and would leave the job with no terminal
        event at all.

        Nothing the awaited task raises escapes. ``_run`` swallows every
        exception the domain claims but deliberately RE-RAISES the ones it
        does not (KeyboardInterrupt, SystemExit, CancelledError) after
        recording the job, and this is the one place that await happens --
        inside the lifespan shutdown hook. Letting one through would turn
        "the server is coming down" into a traceback on the way out, of a job
        that has already reached a terminal state and told everyone watching.
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
            log.warning("%s did not stop within %.0fs; "
                        "leaving it to the interpreter",
                        self._label, self._shutdown_timeout_s)
        except BaseException:
            # BaseException, not Exception: _run re-raises what terminal_for
            # does not claim on purpose (see the docstring), so
            # ``except Exception`` here catches only the cases _run already
            # handled and misses the ones it forwards.
            log.exception("%s failed during shutdown", self._label)
