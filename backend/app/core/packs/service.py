"""One pack install at a time, off the event loop, readable from anywhere.

``flows.install_pack_live`` is synchronous and blocks its thread for minutes.
``app.core.jobs.JobRunner`` is the seam between that and an HTTP server that
must keep answering while it runs: the worker thread, the lock rules, the
cursor, the ring buffer and the long poll are all its, and the reasoning
behind each of them is in that module.

What is HERE is everything the runner deliberately does not know: which packs
may be installed at all, what the failure shapes of ``packs.errors`` mean for
a job (``_terminal_for``), and the restart handshake below.

ONE INSTALL AT A TIME, deliberately. Two concurrent ``uv pip install`` runs
share one site-packages and one cache lock, and the honest outcomes are a
corrupt environment or a deadlock. A second submit is refused with the id of
the job already running, which is what the UI needs to show "an install is
already in progress" and offer to follow it.

A RESTART-MODE install has no flow and no thread at all. It cannot run inside
this process -- it would replace packages this process has already imported --
so ``_submit_restart`` writes down what to install, hands it to a detached
helper, records a job that is ALREADY terminal (``needs_restart``) and asks
the server to stop. The job exists so that the client has the same thing to
follow as for a live install: one id, one event stream, one terminal event
saying what happens next. What it follows afterwards is the server coming
back.

The finished job STAYS -- the last thing a client asks for is the tail of a
job that has just ended -- and every terminal event is readable before the
status that explains it flips. Both are the runner's doing.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..jobs import (
    MAX_EVENTS,
    SHUTDOWN_TIMEOUT_S,
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
    Work,
)
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

#: This module's public surface, spelled out because half of it is imported
#: from ``app.core.jobs`` and re-exported. ``routes_packs``, the CLI and the
#: tests have always read these names from here, and moving the runner out
#: must not move them: a linter that cannot see a re-export would report the
#: imports above as unused, and deleting them would be an API break.
__all__ = [
    "MAX_EVENTS",
    "SHUTDOWN_TIMEOUT_S",
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_NEEDS_RESTART",
    "STATUS_RUNNING",
    "PLUGIN_INSTALL_RUNNING",
    "PackBusy",
    "PackJob",
    "PackService",
    "RestartUnavailable",
    "UnknownJob",
]

#: The ``reason`` a :class:`PackBusy` carries when the job in the way is the
#: Plugin Center's rather than this one's. A machine-readable string because
#: the SPA branches on it -- the sentence beside it is for the person.
PLUGIN_INSTALL_RUNNING = "plugin_install_running"


class PackBusy(JobBusy):
    """An install is already running. ``job_id`` is the one to follow.

    A :class:`~app.core.jobs.JobBusy` with the installer's wording: the
    runner refuses a second claim in its own generic terms, and this is what
    that refusal is called when the job is a pack install.

    ``reason`` is set only when the job in the way is NOT one of ours -- a
    plugin install, running in the same interpreter. The client is told
    which, because "follow the install you already started" and "wait for
    somebody else's" are different offers to make, and an id alone does not
    say which of the two this is. An ordinary busy refusal leaves it
    ``None``, and the route then leaves the key out altogether: that body
    has been two keys since the panel was written.
    """

    def __init__(self, job_id: str, *, reason: str | None = None,
                 message: str | None = None):
        self.reason = reason
        super().__init__(
            job_id,
            message or f"a pack install is already running (job {job_id})")


class RestartUnavailable(Exception):
    """This server cannot restart itself; ``command`` is what to type."""

    def __init__(self, message: str, *, command: str):
        self.command = command
        super().__init__(message)


@dataclass(kw_only=True)
class PackJob(Job):
    """One install, from submit to terminal event."""

    pack_id: str
    #: The items this job will fetch, resolved at submit time (see
    #: ``PackService._targets``) -- never the caller's ``None``.
    items: tuple[str, ...]
    mode: str
    restart_command: str | None = None


class PackService:
    """The Package Center's job runner. One instance on ``app.state``."""

    def __init__(
        self,
        *,
        run_flow: Callable[..., object] = flows.install_pack_live,
        runs_active: Callable[[], bool] | None = None,
        busy_elsewhere: Callable[[], str | None] | None = None,
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
        # "Is the OTHER installer running?", answered with its job id. The
        # Plugin Center installs Python packages into the same site-packages
        # under the same constraints freeze, so two of them at once end the
        # way two pack installs would -- a corrupt environment or a deadlock
        # -- and the one-job-at-a-time rule has to span both runners.
        # INJECTED for the reason ``runs_active`` is: the answer belongs to
        # that service, and this class must not hold it to ask. Left out, the
        # answer is "nothing is running elsewhere", which is what a service
        # with no second installer behind it -- every test here, and every
        # server built before there was one -- is honestly entitled to say.
        self._busy_elsewhere = (busy_elsewhere if busy_elsewhere is not None
                                else (lambda: None))
        # The single job slot: the lock, the buffer, the cursor, the long
        # poll and the worker task. ``_terminal_for`` is the seam back --
        # the runner asks it what an exception out of the flow did to the
        # job, because knowing that is this module's business and not its.
        self._runner = JobRunner(terminal_for=self._terminal_for,
                                 label="pack install job",
                                 shutdown_timeout_s=shutdown_timeout_s)

    @property
    def _task(self) -> "asyncio.Task | None":
        """The worker task of the current job, or None when it has none.

        The restart-mode path never starts one, and a claim drops the
        previous job's. Exposed because ``shutdown`` used to own it here and
        the service's tests still read it.
        """
        return self._runner._task

    # ── reading ───────────────────────────────────────────────────────────

    def current_job(self) -> PackJob | None:
        """The most recent job, running or not. None before the first submit."""
        # Every job the runner has ever held was made by this class, so
        # every one of them is a PackJob.
        return self._runner.current_job()

    def get_job(self, job_id: str) -> PackJob:
        """The job with this id. Raises :class:`UnknownJob` for any other."""
        return self._runner.get_job(job_id)

    async def wait_for_events(
        self,
        job_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 500,
        wait: float = 0.0,
    ) -> tuple[list[dict], int, str]:
        """Events after *after_cursor*, optionally long-polling for the next.

        Returns ``(events, cursor, status)``. The cursor rules and the
        lost-wakeup ordering that make the long poll safe belong to
        :meth:`app.core.jobs.JobRunner.wait_for_events`.
        """
        return await self._runner.wait_for_events(
            job_id, after_cursor=after_cursor, limit=limit, wait=wait)

    def _read(self, job: PackJob, after_cursor: int, limit: int
              ) -> tuple[list[dict], int, str]:
        """One consistent look at the buffer AND the job's status.

        A pass-through to the runner, kept here because it is what a test
        reaches for to reproduce the replaced-job window: an interleaving the
        public API cannot be made to produce on demand.
        """
        return self._runner._read(job, after_cursor, limit)

    # ── submitting ────────────────────────────────────────────────────────

    def _refuse_if_busy_elsewhere(self) -> None:
        """Refuse while the OTHER installer holds this interpreter.

        Asked HERE and not inside ``JobRunner.claim``: the runner owns one
        slot and knows nothing about a second one, and teaching it about the
        other service would make it exactly the domain-aware thing its own
        module docstring says it must not become. So each service asks the
        other, in its own pre-claim section, and the refusal is spelled in
        its own vocabulary.

        Re-asked after every suspension point, for the reason the local busy
        check is: the answer can change while this coroutine is parked, and
        the only check that decides anything is the last one before the
        claim.

        :raises PackBusy: with the OTHER job's id, and ``reason`` saying it
            is not one of ours -- the client is about to offer "follow it",
            and following somebody else's install is a different offer.
        """
        other = self._busy_elsewhere()
        if other is not None:
            raise PackBusy(
                other, reason=PLUGIN_INSTALL_RUNNING,
                message=f"a plugin install is already running (job {other}); "
                        f"it installs into the same interpreter, so this has "
                        f"to wait for it")

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
        the live path from the busy check to the runner's ``claim``
        therefore runs without a suspension point between them, which is the
        whole of what makes "one install at a time" true: two requests cannot
        both pass the check before either takes the slot.

        :raises PackBusy: an install is already running -- this service's
            own, or the Plugin Center's in the same interpreter, which
            carries ``reason`` so the client knows whose job it is naming.
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
        # The runner refuses a second claim too, in its own words. This check
        # is the one that answers the CLIENT: it happens before anything is
        # resolved or disk-checked, and it raises the refusal the routes map
        # to a 409 naming the install to follow.
        running = self.current_job()
        if running is not None and not running.terminal:
            raise PackBusy(running.job_id)
        # ...and the same question of the OTHER installer, which shares this
        # interpreter with us and has a slot of its own.
        self._refuse_if_busy_elsewhere()

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
        # job_started is in the buffer before the flow can emit anything, so
        # a client that polls from cursor 0 always sees it first.
        self._runner.claim(job, {"type": "job_started",
                                 "pack_id": pack.pack_id,
                                 "items": list(job.items)})
        self._runner.start(job, self._flow_work(job, pack),
                           on_settled=self._settled)
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
        # below and the runner's claim cannot be interleaved with another
        # submit -- the same property that makes "one job at a time" hold in
        # ``submit_install``. Both busy checks are repeated because the probe
        # above IS a suspension point: a live install -- ours or the Plugin
        # Center's -- may have started during it, and stopping the server
        # would kill it.
        running = self.current_job()
        if running is not None and not running.terminal:
            raise PackBusy(running.job_id)
        self._refuse_if_busy_elsewhere()

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

        self._runner.claim(job, {"type": "job_started",
                                 "pack_id": pack.pack_id, "items": []})
        # Terminal immediately, and under the lock like every other terminal
        # event: a client that sees ``needs_restart`` has already been handed
        # the event that names the command, the kind and the file.
        self._runner.finish(job, STATUS_NEEDS_RESTART,
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

    def _flow_work(self, job: PackJob, pack: Pack) -> Work:
        """The unit of work the runner hands to ``asyncio.to_thread``.

        A closure rather than a partial: ``run_flow`` is injectable and the
        runner knows nothing about packs, so this is the one place the flow's
        own argument shape -- ``(pack, item_ids, emit=, cancel_check=)`` --
        is spelled out.

        ``list(job.items)`` and not the tuple, because that is the type the
        flow has always been handed and what a scripted test flow records.
        """

        def work(emit: Emit, cancel_check: CancelCheck) -> None:
            self._run_flow(pack, list(job.items),
                           emit=emit, cancel_check=cancel_check)

        return work

    def _terminal_for(self, exc: BaseException) -> tuple[str, dict] | None:
        """What an exception out of the flow did to the job, and what to say.

        Handed to the runner at construction. The runner owns the lock
        discipline, the buffer and the cursor; this owns the failure shapes
        of ``packs.errors`` and what each of them tells the person watching
        -- which is the whole reason the runner takes a callback instead of
        importing this module.

        The job it is about is :meth:`current_job`. The runner only asks
        while the CURRENT job's worker thread is unwinding, and a submit is
        refused until that job is terminal -- which is what the answer to
        this call is about to make it -- so the job that raised is still the
        one in the slot.

        ``None`` means "not one of ours": KeyboardInterrupt, SystemExit,
        CancelledError. The runner records the job as failed and re-raises
        it, which is what those deserve; the warning is logged here rather
        than in the runner, because this is where a job has a name.
        """
        job = self.current_job()
        if isinstance(exc, PackCancelled):
            return STATUS_CANCELLED, {"type": "job_cancelled"}
        if isinstance(exc, PackNeedsRestart):
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
            return STATUS_NEEDS_RESTART, event
        if isinstance(exc, PackInstallError):
            return STATUS_FAILED, {"type": "job_failed",
                                   "message": str(exc), "hint": exc.hint}
        if isinstance(exc, Exception):
            # Not a failure shape anybody designed, so it goes to the log in
            # full and to the user as its repr -- str() of a bare KeyError is
            # just a quoted key and says nothing about what broke.
            log.exception("pack install job %s raised", job.job_id)
            return STATUS_FAILED, {"type": "job_failed",
                                   "message": repr(exc), "hint": None}
        # KeyboardInterrupt, SystemExit, CancelledError. The job is over
        # either way, and a job left saying "running" forever is worse than
        # one that says why it stopped -- the panel would offer a Stop button
        # for a thread that no longer exists, and no submit would ever be
        # accepted again. The runner records it, then lets it travel: these
        # are not ours to swallow.
        log.warning("pack install job %s ended on %s", job.job_id,
                    type(exc).__name__)
        return None

    @staticmethod
    def _settled() -> None:
        """Drop the probe cache once the job is terminal, however it ended.

        Finished, failed or cancelled, the disk is not what it was and the
        next status poll has to see that. The flow invalidates too; doing it
        here as well covers an injected flow and costs one dict drop.
        """
        state.invalidate()

    # ── stopping ──────────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> bool:
        """Ask the job to stop. False when it had already finished.

        Cooperative: the flow notices between steps and inside a download,
        and it is the FLOW that ends the job, so the status may still say
        running when this returns. Asking twice is not an error.
        """
        return await self._runner.cancel(job_id)

    async def shutdown(self) -> None:
        """Stop a running install and wait for it, bounded.

        Bounded because the server is coming down either way: a flow that
        ignores its cancel check must not be able to hold the process open.
        See :meth:`app.core.jobs.JobRunner.shutdown` for what the bound
        protects and why nothing the job raised escapes here.
        """
        await self._runner.shutdown()
