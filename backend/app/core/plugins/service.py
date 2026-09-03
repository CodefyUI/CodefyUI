"""One plugin install at a time, off the event loop, readable from anywhere.

``flows.install_plugin_live`` is synchronous and blocks its thread for
minutes. ``app.core.jobs.JobRunner`` is the seam between that and an HTTP
server that must keep answering while it runs: the worker thread, the lock
rules, the cursor, the ring buffer and the long poll are all its, and the
reasoning behind each of them is in that module. ``app.core.packs.service``
is the same class over the other installer, and the two are deliberately
shaped alike -- one panel draws both.

What is HERE is everything the runner does not know:

* **The inspections.** Consent is a conversation with two turns -- "this is
  what the plugin asks for" and "yes, install it" -- and the second turn
  must install the FIRST turn's manifest. So the answer to an inspect is
  kept under an id and the install refers to that id, rather than the
  client sending a manifest back for the server to trust. They expire
  (:data:`INSPECTION_TTL_S`) because a consent screen left open overnight
  describes a branch that has moved on, and there are at most
  :data:`MAX_INSPECTIONS` of them because each one holds a whole manifest
  and nothing ever asks the server to forget one.
* **Updates, as one call.** :meth:`PluginService.update` is those two turns
  collapsed for a plugin that is already here: it reads the repository the
  lockfile recorded and answers with one of three things -- there is nothing
  to fetch, a job the grants from last time already cover, or the consent
  screen for what this version asks for BEYOND them. Only the last one is a
  question, which is what keeps an ordinary update one click.
* **Consent enforcement, BEFORE a job exists.** A refusal that arrives as a
  failed job is a refusal the user has to go and read events about; a
  refusal raised here is the answer to their own request, and nothing has
  been downloaded, run or written when it happens.
* **ONE INSTALL AT A TIME, across both installers.** Two ``uv pip install``
  runs share one site-packages and one cache lock, and the honest outcomes
  are a corrupt environment or a deadlock -- and the Package Center is
  installing into the same interpreter. So each service asks the other
  through ``busy_elsewhere`` in its own pre-claim section, and refuses in
  its own vocabulary.
* **The reload, on the loop.** Re-discovering the registry replaces a dict
  the request handlers read, and the flow runs on a worker thread, so it
  cannot be a step of the flow. It is the runner's ``after_work`` instead:
  the last step of the job, emitted like any other step, run on the loop
  after the thread has returned and before ``job_done`` -- which is also
  what makes ``job_done`` able to say which nodes arrived.

The finished job STAYS -- the last thing a client asks for is the tail of a
job that has just ended -- and every terminal event is readable before the
status that explains it flips. Both are the runner's doing.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from app.core import plugin_loader
from app.core.node_registry import registry

from ..jobs import (
    MAX_EVENTS,
    SHUTDOWN_TIMEOUT_S,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_NEEDS_RESTART,
    CancelCheck,
    Emit,
    Job,
    JobRunner,
    UnknownJob,
    Work,
)
from . import flows
from . import inspect as inspect_module
from .errors import (
    AlreadyInstalled,
    ConsentRequired,
    GitHubError,
    InspectBusy,
    InspectionExpired,
    PluginBusy,
    PluginCancelled,
    PluginInstallError,
    PluginNeedsRestart,
    TrustAuthorRequired,
)
from .inspect import Inspection
from .listing import nodes_for_plugin
from .reload import rediscover_now

log = logging.getLogger(__name__)

#: This module's public surface. ``UnknownJob`` is imported from
#: ``app.core.jobs`` and re-exported: the routes catch it around this
#: service's own methods, and a linter that cannot see a re-export would
#: report the import above as unused.
__all__ = [
    "INSPECTION_TTL_S",
    "MAX_INSPECTIONS",
    "PACK_INSTALL_RUNNING",
    "PluginJob",
    "PluginService",
    "StoredInspection",
    "UnknownJob",
    "UpdateOutcome",
]

#: How long an inspection is worth installing from. Fifteen minutes is
#: "somebody read the consent screen and thought about it" rather than "this
#: tab has been open since yesterday": the manifest, the sha and the
#: capability list in a stored inspection are a snapshot of a branch, and
#: installing from a stale one is installing a commit nobody looked at.
INSPECTION_TTL_S = 900.0

#: How many inspections are kept at once. Each one holds a whole manifest,
#: nothing ever asks the server to forget one, and a panel that inspects
#: every row of a catalog would otherwise leave them all here until they
#: time out. The OLDEST goes, which is the one furthest from the consent
#: screen the user is actually looking at.
MAX_INSPECTIONS = 32

#: The ``reason`` a :class:`~.errors.PluginBusy` carries when the job in the
#: way belongs to the Package Center. A machine-readable string because the
#: SPA branches on it -- the sentence beside it is for the person. Its twin
#: is ``packs.service.PLUGIN_INSTALL_RUNNING``.
PACK_INSTALL_RUNNING = "pack_install_running"

#: The clock the inspection deadlines are measured against. A module
#: attribute rather than a direct call so a test can hand this service a
#: clock it moves itself -- a TTL is a rule with a deadline inside it, and
#: the only honest way to test one is to control the time it is measured
#: against. Monotonic and not the wall clock: an NTP correction or a
#: daylight-saving jump must not expire every open consent screen at once.
monotonic = time.monotonic


@dataclass(frozen=True)
class StoredInspection:
    """One inspection, kept until it is installed, evicted or expires.

    Two expiry fields on purpose. ``deadline`` is monotonic and is what this
    service ENFORCES; ``expires_at`` is an ISO-8601 wall-clock instant and is
    what the client is TOLD, because "expires_at: 1731.9" means nothing in a
    browser. They are computed from the same call, and only one of them can
    be affected by the clock being set.
    """

    inspection_id: str
    inspection: Inspection
    expires_at: str
    deadline: float
    #: Whether installing from THIS record replaces what is on disk without
    #: being asked again. Only :meth:`PluginService.update` sets it, and it
    #: is the offer the user already accepted by pressing Update. It lives on
    #: the stored record rather than on the inspection because
    #: ``mode == "update"`` is true of every inspection of an installed
    #: plugin -- a plain ``/inspect`` of one included, and that one must
    #: still be answered with "you already have this".
    force: bool = False


@dataclass(kw_only=True)
class PluginJob(Job):
    """One install, from submit to terminal event.

    ``kind`` is where the files come from (``builtin``/``github``) and
    ``mode`` is what this install is doing to the plugin
    (``install``/``update``) -- the same two words the inspection the job was
    made from uses, so a job and the screen it was started from cannot
    disagree about either.

    ``sha`` is what is being installed: the commit the user consented to at
    submit time, replaced by what the flow reports it actually wrote. They
    agree today (the flow never re-resolves a ref), and reading it back off
    the outcome is what keeps that an observation rather than an assumption.
    """

    plugin_id: str
    kind: str
    mode: str
    source: str
    sha: str | None = None
    restart_command: str | None = None


@dataclass(frozen=True)
class UpdateOutcome:
    """What asking to update one plugin came to. Three shapes, one value.

    ``kind`` says which, and exactly one of the fields beside it is
    populated: ``up_to_date`` carries the ``sha`` that is both installed and
    current, ``needs_consent`` carries the stored ``inspection`` the caller
    must show and can then install by id, and ``job`` carries the install
    that is already running.

    One value rather than three methods because the caller cannot know which
    it will get -- that is the answer, not the question -- and a union it
    branches on once is a route handler that cannot forget a case.
    """

    kind: Literal["up_to_date", "needs_consent", "job"]
    sha: str | None = None
    inspection: StoredInspection | None = None
    job: PluginJob | None = None


def _expires_at(ttl_s: float) -> str:
    """The wall-clock instant *ttl_s* from now, ISO-8601 UTC."""
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_s)).isoformat()


def _inspect_busy() -> InspectBusy:
    """The refusal when a source is already being read.

    A factory rather than the sentence written twice: :meth:`inspect` and
    :meth:`update` both take the same slot, and a caller that told them apart
    by their prose would be reading the wrong difference.
    """
    return InspectBusy("Another source is being read right now.",
                       hint="Try again in a moment.")


def _pack_install_busy(job_id: str) -> PluginBusy:
    """The refusal when the Package Center's install is the one in the way.

    Outside the class and returning the exception rather than raising it, so
    that :meth:`submit_install` can raise it inside the stretch that must not
    ``await`` -- see the comment there -- while :meth:`update` raises the same
    refusal a network round trip earlier.
    """
    return PluginBusy(
        job_id, reason=PACK_INSTALL_RUNNING,
        message=f"a pack install is already running (job {job_id}); it "
                f"installs into the same interpreter, so this has to wait "
                f"for it")


class PluginService:
    """The Plugin Center's job runner. One instance on ``app.state``."""

    def __init__(
        self,
        *,
        run_flow: Callable[..., flows.InstallOutcome] = flows.install_plugin_live,
        reload: Callable[[], object] = rediscover_now,
        busy_elsewhere: Callable[[], str | None] | None = None,
        inspection_ttl_s: float = INSPECTION_TTL_S,
        max_inspections: int = MAX_INSPECTIONS,
        max_events: int = MAX_EVENTS,
        shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
    ) -> None:
        # Injectable so tests never install anything, and so a future caller
        # could drive the same runner with a different flow.
        self._run_flow = run_flow
        # The re-discovery this service runs ON THE LOOP once the flow has
        # returned. Injected for the same reason the flow is: a test has to
        # be able to prove WHERE it ran and what happens when it fails,
        # neither of which is answerable while it is really rebuilding a
        # process-wide registry.
        self._reload = reload
        # "Is the OTHER installer running?", answered with its job id. The
        # Package Center installs Python packages into the same site-packages
        # under the same constraints freeze, so two of them at once end the
        # way two plugin installs would. INJECTED because the answer belongs
        # to that service and this class must not hold it to ask; left out,
        # the answer is "nothing is running elsewhere", which is what a
        # service with no second installer behind it -- every test here -- is
        # honestly entitled to say.
        self._busy_elsewhere = (busy_elsewhere if busy_elsewhere is not None
                                else (lambda: None))
        self._inspection_ttl_s = inspection_ttl_s
        # At least one, whatever was asked for. The store has to hold the
        # inspection it has just made -- an install refers to it by id
        # moments later -- so "keep none" is not a state this can be in, and
        # a zero would leave the eviction loop popping an empty store.
        self._max_inspections = max(1, max_inspections)
        # Insertion-ordered, so the oldest entry is the first one -- which is
        # the eviction rule -- and ``move_to_end`` on every read makes that
        # "least recently used" rather than "oldest read".
        self._inspections: "OrderedDict[str, StoredInspection]" = OrderedDict()
        # One inspection at a time. Not a queue: a caller that waited would
        # get its answer long after the person had typed something else.
        self._inspecting = asyncio.Semaphore(1)
        # The single job slot: the lock, the buffer, the cursor, the long
        # poll and the worker task. ``_terminal_for`` is the seam back -- the
        # runner asks it what an exception out of the flow did to the job,
        # because knowing that is this module's business and not its.
        self._runner = JobRunner(terminal_for=self._terminal_for,
                                 label="plugin install job",
                                 max_events=max_events,
                                 shutdown_timeout_s=shutdown_timeout_s)

    # ── reading ───────────────────────────────────────────────────────────

    def current_job(self) -> PluginJob | None:
        """The most recent job, running or not. None before the first submit."""
        # Every job the runner has ever held was made by this class, so every
        # one of them is a PluginJob.
        return self._runner.current_job()

    def current_job_id(self) -> str | None:
        """The id of the job running NOW, or None when nothing is.

        What the Package Center is handed as its ``busy_elsewhere``. A
        FINISHED job is not in anybody's way -- it stays readable here, but
        reporting it would refuse every pack install for as long as nobody
        started another plugin one.
        """
        job = self._runner.current_job()
        return job.job_id if job is not None and not job.terminal else None

    def active_job_payload(self) -> dict[str, Any] | None:
        """The running install, in the shape ``GET /api/plugins/catalog``
        publishes it -- or None when nothing is running.

        ``kind`` here is the job's MODE, which is the one wart in this
        module: ``catalog_listing`` echoes this dict verbatim and the panel
        reads ``active_job.kind`` as install-or-update (it is what decides
        whether the toast says "installed" or "updated"), while a job's own
        ``kind`` says where the files come from. Spelled out here, once, so
        that no route has to remember it.

        Only a NON-TERMINAL job is reported, and that is load-bearing:
        ``listing._status`` renders any row carrying a job as ``installing``
        without looking at its status, so a finished job passed through here
        would leave that row spinning until the next install.
        """
        job = self._runner.current_job()
        if job is None or job.terminal:
            return None
        return {"job_id": job.job_id, "plugin_id": job.plugin_id,
                "kind": job.mode, "status": job.status,
                "current_step": job.current_step}

    def get_job(self, job_id: str) -> PluginJob:
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

    # ── inspecting ────────────────────────────────────────────────────────

    async def inspect(self, source: str) -> StoredInspection:
        """Read *source* and remember what it said, under a fresh id.

        ONE at a time, and refused rather than queued: reading a source is a
        network round trip (or a manifest off the disk), and a panel that
        inspects per keystroke would otherwise open a socket per keystroke.
        The refusal is immediate, so the caller can ask again when the person
        stops typing.

        The lockfile is read INSIDE the same worker thread as the inspection
        itself. Both are blocking file I/O, ``inspect_source`` requires the
        lockfile it should compare against (there is no default -- an
        inspection with the wrong lockfile would report an update as a fresh
        install), and reading it on the loop for the sake of the keyword
        would put a file read on the thread that is answering everybody else.

        Cancelling the awaiting task -- a client that hangs up mid-read --
        releases the semaphore while the worker thread is still inside
        ``inspect_source``, so the next caller can start a second one over
        the top of it. Not reachable from the shipped client (nothing
        cancels an inspection), and bounded anyway: the threads come from the
        default executor's pool, and each one only reads a manifest.

        :raises InspectBusy: another inspection is already running.
        :raises SourceError: what the user typed names nothing installable.
        :raises ManifestError: the manifest at that source is not one this
            build installs.
        :raises GitHubError: the repository could not be read.
        :raises PluginInstallError: the source names something that cannot be
            installed under that id -- a reserved plugin id, or a catalog row
            that validated away.
        """
        if self._inspecting.locked():
            raise _inspect_busy()
        async with self._inspecting:
            inspection = await asyncio.to_thread(_read_source, source)
        return self._remember(inspection)

    def get_inspection(self, inspection_id: str) -> StoredInspection:
        """The stored inspection with this id, and move it to the front.

        :raises InspectionExpired: it was installed, evicted or timed out.
            One answer for the three, because the caller's next move is the
            same for all of them: inspect the source again.
        """
        self._drop_expired()
        stored = self._inspections.get(inspection_id)
        if stored is None:
            raise InspectionExpired(inspection_id)
        # A consent screen somebody is still reading is not the one to evict
        # when the next inspection arrives.
        self._inspections.move_to_end(inspection_id)
        return stored

    def _remember(self, inspection: Inspection, *,
                  force: bool = False) -> StoredInspection:
        """Store *inspection* under a fresh id, evicting what has to go.

        *force* marks a record an install may replace what is on disk from --
        see :class:`StoredInspection`. It is a keyword this class sets and
        never a value that arrives from outside.
        """
        self._drop_expired()
        while len(self._inspections) >= self._max_inspections:
            self._inspections.popitem(last=False)
        # 32 bytes of entropy, because an inspection id is what authorises an
        # install of the manifest behind it: guessing one must not be a way
        # to install somebody else's pending consent screen.
        stored = StoredInspection(
            inspection_id=secrets.token_hex(16),
            inspection=inspection,
            expires_at=_expires_at(self._inspection_ttl_s),
            deadline=monotonic() + self._inspection_ttl_s,
            force=force)
        self._inspections[stored.inspection_id] = stored
        return stored

    def _drop_expired(self) -> None:
        """Forget every inspection past its deadline.

        On access rather than on a timer: there is no thread here to run one,
        and an expired inspection costs nothing but the manifest it holds
        until the next caller arrives.
        """
        now = monotonic()
        for inspection_id in [key for key, stored in self._inspections.items()
                              if stored.deadline <= now]:
            del self._inspections[inspection_id]

    # ── submitting ────────────────────────────────────────────────────────

    async def submit_install(
        self,
        inspection_id: str,
        *,
        accept_capabilities: Iterable[str] | None = None,
        trust_author: bool = False,
        force: bool = False,
    ) -> PluginJob:
        """Install what *inspection_id* described. Returns the job.

        Every refusal happens BEFORE a job exists, which is the point: the
        client is answered by the response to its own request rather than by
        a job that appears in the panel and immediately fails.

        Nothing here awaits -- not the consent check, not the busy checks,
        not the claim -- so the whole method runs in one turn of the loop and
        two requests cannot both pass the busy check before either takes the
        slot. That is the whole of what makes "one install at a time" true,
        and it is why the checks are in this order rather than in a helper
        that might one day grow an ``await``.

        :raises InspectionExpired: no such inspection, or it has timed out.
        :raises AlreadyInstalled: the plugin is here and *force* was not
            given. Both kinds, and updates too: replacing what is on disk is
            an offer for the user to accept, not a default -- an inspection
            stored by :meth:`update` carries that acceptance with it.
        :raises ConsentRequired: a capability the manifest asks for is not in
            *accept_capabilities* (nor already granted by a previous install).
        :raises TrustAuthorRequired: the manifest declares ``allowed_modules``
            and *trust_author* was not given.
        :raises PluginBusy: an install is already running -- this service's
            own, or the Package Center's in the same interpreter, which
            carries ``reason`` so the client knows whose job it is naming.
        """
        stored = self.get_inspection(inspection_id)
        inspection = stored.inspection
        # An update brings its own force. :meth:`update` stores the
        # inspection it read, and pressing Update IS the offer to replace
        # what is on disk -- so the client finishes that conversation with
        # the id and its consent answers, and is not refused for a decision
        # it has already made. Read off the STORED record and never off the
        # inspection: ``mode == "update"`` is true of every inspection of an
        # installed plugin, a plain ``/inspect`` of one included, and that
        # one must still be answered with "you already have this".
        force = force or stored.force

        # Asked HERE and not only in the flow, which cannot answer it in
        # time: by then a job exists, and for a repository plugin the flow's
        # own copy never fires at all (an inspection of something already
        # installed is a ``mode == "update"``, which the flow lets through).
        # "You already have this" is an OFFER -- Reinstall, --force -- and an
        # offer belongs in the response to the request, not in a job's event
        # log.
        if inspection.installed is not None and not force:
            raise AlreadyInstalled(
                f"Plugin {inspection.plugin_id!r} is already installed.",
                plugin_id=inspection.plugin_id,
                hint="Install it again with force to replace the copy that "
                     "is there.")

        try:
            plan = flows.plan_from_inspection(
                inspection,
                accept_capabilities=accept_capabilities or (),
                trust_author=trust_author,
                force=force)
        except TrustAuthorRequired:
            # Already the narrow class (``consent.check_trust`` raises it).
            raise
        except ConsentRequired as exc:
            # Belt and braces: this service promises the two halves of
            # consent as two classes, whoever inside decided to raise which.
            if exc.allowed_modules and not exc.missing_capabilities:
                raise TrustAuthorRequired(
                    str(exc), allowed_modules=exc.allowed_modules,
                    hint=exc.hint) from exc
            raise

        # ── no ``await`` from here to the claim ───────────────────────────
        other = self._busy_elsewhere()
        if other is not None:
            raise _pack_install_busy(other)
        running = self.current_job()
        if running is not None and not running.terminal:
            # The runner refuses a second claim too, in its own words. This
            # check is the one that answers the CLIENT, in this domain's
            # vocabulary and with the id of the install to follow.
            raise PluginBusy(running.job_id)

        job = PluginJob(job_id=uuid.uuid4().hex, plugin_id=plan.plugin_id,
                        kind=plan.kind, mode=plan.mode,
                        source=inspection.source, sha=plan.sha)
        # job_started is in the buffer before the flow can emit anything, so
        # a client that polls from cursor 0 always sees it first.
        self._runner.claim(job, {"type": "job_started",
                                 "plugin_id": job.plugin_id,
                                 "kind": job.kind,
                                 "source": job.source,
                                 "mode": job.mode})
        # Consumed only now. A refusal above leaves the inspection where it
        # was, so the user can accept the capability they missed, or wait for
        # the other install, and press the button again without re-reading
        # the repository; a successful claim spends it, so the same consent
        # screen cannot start two installs.
        self._inspections.pop(inspection_id, None)
        self._runner.start(job, self._flow_work(job, plan),
                           after_work=self._reload_step)
        return job

    async def update(self, plugin_id: str) -> UpdateOutcome:
        """Fetch what *plugin_id*'s own repository has now. Three answers.

        The two turns of an install collapsed into one call, because for a
        plugin that is already here the server knows both halves: WHICH
        repository (the lockfile recorded it) and WHAT was agreed to last
        time. So the ordinary update -- a new commit asking for nothing new
        -- is a job started from one click, and the consent screen comes back
        only for what this version asks for BEYOND the previous grant. That
        comparison is the whole reason an update is worth inspecting:
        capability creep between the version somebody consented to and the
        one about to replace it is the supply-chain shape a plugin manager
        can actually catch.

        The order of the checks is what the answers are made of:

        1. the lockfile, on the loop -- "you do not have that" and "a
           built-in pack updates with CodefyUI" are true without a network
           and must not have to queue behind somebody else's read;
        2. the install slot, still before the network -- this is a request to
           START a job, and the service runs one at a time across both
           installers, so reading GitHub for a job that cannot be claimed
           spends a round trip to reach the same refusal;
        3. the inspection itself, one at a time and off the loop, exactly as
           :meth:`inspect` does it;
        4. consent, which is :meth:`submit_install`'s to enforce -- asked
           through it rather than re-implemented here, so an update and an
           install cannot disagree about what counts as already granted.

        A refusal at (4) leaves the inspection STORED, which is what makes
        the consent screen it hands back completable: the client answers with
        that id, and the replacement it already agreed to by pressing Update
        travels on the stored record rather than on the wire.

        :raises NotInstalled: no lockfile entry under that id.
        :raises NotUpdatable: it is installed, from something with no
            repository to re-fetch (a built-in pack, a linked directory).
        :raises PluginBusy: an install is already running, here or in the
            Package Center.
        :raises InspectBusy: another source is being read right now.
        :raises GitHubError: the repository could not be read.
        :raises PluginInstallError: what the repository now declares cannot
            be installed under that id -- a reserved id, a manifest this
            build will not take.
        """
        lockfile = plugin_loader.load_lockfile()
        # Raises before anything else happens. The entry itself is not needed
        # here -- ``inspect_installed`` reads it again on the thread, from the
        # same lockfile -- but its VERDICT is, and it is free.
        inspect_module.updatable_entry(plugin_id, lockfile=lockfile)

        running = self.current_job_id()
        if running is not None:
            # The one refusal that is deliberately wider than the lifecycle
            # routes': a delete or a toggle only waits for THIS plugin's job,
            # because two plugins are two directories -- but only one install
            # can run at a time, so an update of anything waits for all of
            # them. ``submit_install`` asks again at the claim, which is
            # where "one at a time" is actually made true.
            raise PluginBusy(running)
        other = self._busy_elsewhere()
        if other is not None:
            raise _pack_install_busy(other)

        if self._inspecting.locked():
            raise _inspect_busy()
        async with self._inspecting:
            inspection = await asyncio.to_thread(
                _read_installed, plugin_id, lockfile)

        if inspection.up_to_date:
            # The commit on disk is the commit the ref still points at. There
            # is nothing to consent to and nothing to fetch, so no inspection
            # is stored: an id nobody can install from is an id nobody needs.
            return UpdateOutcome(kind="up_to_date", sha=inspection.sha)

        stored = self._remember(inspection, force=True)
        try:
            job = await self.submit_install(stored.inspection_id)
        except ConsentRequired:
            # Capability creep, or a module list that grew: the one thing an
            # update has to ask about. ``TrustAuthorRequired`` is a
            # ``ConsentRequired`` and arrives here too -- which half of
            # consent is outstanding is a question for whoever draws the
            # screen, and both are answered by showing this inspection.
            return UpdateOutcome(kind="needs_consent", inspection=stored)
        return UpdateOutcome(kind="job", job=job)

    # ── running ───────────────────────────────────────────────────────────

    def _flow_work(self, job: PluginJob, plan: flows.InstallPlan) -> Work:
        """The unit of work the runner hands to ``asyncio.to_thread``.

        A closure rather than a partial: ``run_flow`` is injectable and the
        runner knows nothing about plugins, so this is the one place the
        flow's own argument shape -- ``(plan, emit=, cancel_check=)`` -- is
        spelled out.
        """

        def work(emit: Emit, cancel_check: CancelCheck) -> None:
            outcome = self._run_flow(plan, emit=emit, cancel_check=cancel_check)
            # What was actually written, which is what ``job_done`` reports.
            job.sha = outcome.sha

        return work

    def _reload_step(self, emit: Emit) -> dict:
        """The last step of the job, on the loop: re-discover, then report.

        Handed to the runner as ``after_work``, so it runs once the flow's
        thread has returned normally and before the terminal event. It has to
        be here rather than in the flow because ``rediscover_now`` replaces
        the registry dict that every ``GET /api/nodes`` reads on this thread;
        a worker thread rebuilding it under them is a request seeing half a
        palette.

        It EMITS, like any other step, because a re-discovery of a large
        install takes a second and a panel with no current step for a second
        looks stopped. Nothing else is emitted: this buffer is shared with
        the flow's own log, and a step that logs its way through a registry
        rebuild would push the pip output a user is reading out of the ring.

        The job is :meth:`current_job` for the same reason the packs'
        translator finds its own that way, only stronger: this runs on the
        loop, between the work returning and the terminal event, and a claim
        is refused until a job is terminal -- so no other job can possibly be
        in the slot while this is running.

        What it returns is merged into ``job_done``: the panel learns the
        nodes it may now drag and the generation it may compare against
        without asking a second question. Both are read AFTER the
        re-discovery, which is the whole point of reading them here.
        """
        job = self.current_job()
        emit({"type": "step_started", "step": "reload",
              "label": f"Loading {job.plugin_id} into the editor"})
        self._reload()
        emit({"type": "step_done", "step": "reload"})
        return {"plugin_id": job.plugin_id,
                "sha": job.sha,
                "nodes": nodes_for_plugin(job.plugin_id, registry),
                "generation": plugin_loader.reload_generation()}

    def _terminal_for(self, job: PluginJob, exc: BaseException
                      ) -> tuple[str, dict] | None:
        """What an exception out of the flow did to the job, and what to say.

        Handed to the runner at construction. The runner owns the lock
        discipline, the buffer and the cursor; this owns the failure shapes
        of ``plugins.errors`` and what each of them tells the person watching
        -- which is the whole reason the runner takes a callback instead of
        importing this module.

        It takes the JOB as well as the exception (the runner reads which
        shape it declared, once, at construction): the two-argument form is
        there so a translator never has to go looking for the job it is
        about, which is the kind of lookup that is right until the day it
        is not.

        ``None`` means "not one of ours": KeyboardInterrupt, SystemExit,
        CancelledError. The runner records the job as failed and re-raises
        it, which is what those deserve; the warning is logged here rather
        than in the runner, because this is where a job has a name.
        """
        if isinstance(exc, PluginCancelled):
            # Not a failure: the system did as it was told.
            return STATUS_CANCELLED, {"type": "job_cancelled"}
        if isinstance(exc, PluginNeedsRestart):
            # Checked before PluginInstallError: it is a subclass, and "the
            # packages cannot be installed into a running server" is not a
            # failure of the plugin. The command is on the job as well as in
            # the event, because the panel's summary reads the job.
            job.restart_command = exc.command
            return STATUS_NEEDS_RESTART, {"type": "needs_restart",
                                          "command": exc.command,
                                          "hint": exc.hint}
        if isinstance(exc, PluginInstallError):
            # Every designed failure, AlreadyInstalled and ConsentRequired
            # included -- the flow keeps its own copies of both checks as
            # defence, and a job that hits one of them is reporting a bug
            # here, not a new shape for the client to learn.
            return STATUS_FAILED, {"type": "job_failed",
                                   "message": str(exc), "hint": exc.hint}
        if isinstance(exc, GitHubError):
            # Not a PluginInstallError (it is about a server, not about an
            # install), and it can still escape the download step. The status
            # is the user's next move -- a 404 is a typo in the repo name, no
            # status at all is the network -- so it is the hint.
            return STATUS_FAILED, {
                "type": "job_failed", "message": str(exc),
                "hint": (f"GitHub answered {exc.status}"
                         if exc.status is not None else
                         "The request never reached GitHub (network, DNS "
                         "or TLS).")}
        if isinstance(exc, Exception):
            # Not a failure shape anybody designed -- a raw OSError from a
            # copy, a ValueError from a plan this build cannot install -- so
            # it goes to the log in full and to the user as its message.
            # ``repr`` is the fallback because ``str`` of some exceptions is
            # empty, and an empty failure message is worse than a class name.
            log.exception("plugin install job %s raised", job.job_id)
            return STATUS_FAILED, {"type": "job_failed",
                                   "message": str(exc) or repr(exc),
                                   "hint": None}
        # KeyboardInterrupt, SystemExit, CancelledError. The job is over
        # either way, and a job left saying "running" forever is worse than
        # one that says why it stopped -- the panel would offer a Stop button
        # for a thread that no longer exists, and no submit would ever be
        # accepted again. The runner records it, then lets it travel: these
        # are not ours to swallow.
        log.warning("plugin install job %s ended on %s", job.job_id,
                    type(exc).__name__)
        return None

    # ── stopping ──────────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> bool:
        """Ask the job to stop. False when it had already finished.

        Cooperative: the flow notices between steps and inside a download or
        a pip run, and it is the FLOW that ends the job, so the status may
        still say running when this returns. Asking twice is not an error.
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


def _read_source(source: str) -> Inspection:
    """Read *source*, with the lockfile to compare it against. BLOCKING.

    A module-level function so the two blocking calls are one thread hop,
    and so that both are reached through their own module objects -- which is
    what a test replaces when it wants an inspection without a network.
    """
    return inspect_module.inspect_source(
        source, lockfile=plugin_loader.load_lockfile())


def _read_installed(plugin_id: str, lockfile: dict[str, Any]) -> Inspection:
    """Read what *plugin_id*'s recorded repository has now. BLOCKING.

    Beside :func:`_read_source` and reached the same way, so a test can
    answer an update without a network. The lockfile is passed IN rather than
    loaded here: :meth:`PluginService.update` has already read it to decide
    whether this plugin has a repository at all, and reading it twice would
    let the two halves of one answer be about two different files.
    """
    return inspect_module.inspect_installed(plugin_id, lockfile=lockfile)
