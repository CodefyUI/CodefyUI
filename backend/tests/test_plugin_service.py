"""The Plugin Center's job runner, without a server and without an install.

``PluginService`` is what stands between a synchronous, minutes-long install
and an event loop that must stay responsive while it runs, and it owns three
things the runner underneath it deliberately does not:

* **consent**, enforced before a job exists -- a capability nobody granted, a
  module list nobody trusted and a plugin that is already there are all
  answers to the REQUEST, not failures of an install somebody then has to go
  and read events about;
* **the inspections** the consent screen was drawn from, which expire and are
  evicted, because the second half of a two-turn conversation must install
  the manifest the first half showed;
* **one install at a time across BOTH installers**, and the registry reload
  on the loop -- after the flow's thread, before ``job_done``.

Nothing here installs anything, reaches the network, or rebuilds the node
registry: every service is built through :func:`a_service`, which injects a
flow driven step by step from the test thread (:class:`ScriptedFlow`) and a
fake reload, over a source table monkeypatched onto ``inspect.inspect_source``.
The four tests that are ABOUT the reload pass their own ``reload=`` and say
so at the construction site; :func:`no_real_rediscovery` is what keeps that a
rule rather than a habit.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any

import pytest

from app.core import plugin_loader
from app.core.plugins import inspect as inspect_module
from app.core.plugins import service as service_module
from app.core.plugins.errors import (
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
from app.core.plugins.flows import InstallOutcome, InstallPlan
from app.core.plugins.inspect import Inspection
from app.core.plugins.service import PluginService

SOURCE = "alice/extras"
SHA = "a" * 40


@pytest.fixture(autouse=True)
def isolated_user_root(tmp_path, monkeypatch):
    """A throwaway plugins root, so no test reads a developer's lockfile.

    ``plugins_user_root`` reads the variable at CALL time, and the service
    loads the lockfile inside its inspection thread, so redirecting the
    environment is enough -- and it redirects every other reader of the root
    in the same process for the duration of the test.
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    return tmp_path / "plugins"


@pytest.fixture(autouse=True)
def no_real_rediscovery(monkeypatch):
    """Nothing here may rebuild the process-wide node registry.

    ``PluginService``'s default ``reload`` is the real ``rediscover_now``,
    which CLEARS the registry every other test in the session shares (the
    suite survives that only because conftest repairs it before each test)
    and bumps the generation the editor polls. Every service here injects a
    fake instead; this is what turns "we remembered to" into a rule.

    The hit is RECORDED and asserted after the test rather than raised at the
    call site alone. ``reload`` runs inside the runner's ``after_work``, which
    catches what its callback raises and reports it as a failed job -- so an
    ``AssertionError`` thrown in there was swallowed, the guard said nothing,
    and the test went on to assert about a job that failed for the reason the
    guard was trying to announce.
    """
    hits: list[str] = []

    def refuse(*args, **kwargs):
        hits.append("rediscover_all")
        raise AssertionError(
            "this test ran a real registry re-discovery -- build the service "
            "through a_service(), or pass your own reload=")

    monkeypatch.setattr(plugin_loader, "rediscover_all", refuse)
    yield
    assert hits == [], (
        "this test ran a real registry re-discovery -- build the service "
        "through a_service(), or pass your own reload=")


def a_service(**kwargs: Any) -> PluginService:
    """A service whose reload is a stub, for every test not about reloading.

    The four that ARE about it pass their own ``reload=`` at the
    construction site, where a reader of that test can see it.
    """
    kwargs.setdefault("reload", lambda: {"total": 0})
    return PluginService(**kwargs)


def an_inspection(**overrides: Any) -> Inspection:
    """A github inspection of ``alice/extras``, with fields overridden."""
    fields: dict[str, Any] = dict(
        kind="github", mode="install", plugin_id="extras", catalog_id=None,
        official=False, source=SOURCE, owner="alice", repo="extras",
        url="https://github.com/alice/extras", ref="main", sha=SHA,
        name="Extras", version="1.0.0", description="", homepage="",
        manifest={"plugin": {"id": "extras", "version": "1.0.0"}},
        capabilities=(), allowed_modules=(), python_deps={},
        has_frontend=False, chapters=(), lessons=(), consent_required=True,
        installed=None, up_to_date=False, capabilities_added=(),
        allowed_modules_added=(), warnings=(),
    )
    fields.update(overrides)
    return Inspection(**fields)


def a_builtin_inspection(**overrides: Any) -> Inspection:
    """A pack that shipped with this release: nobody to be asked about."""
    fields: dict[str, Any] = dict(
        kind="builtin", plugin_id="stats", source="stats", owner=None,
        repo=None, url=None, ref=None, sha=None, consent_required=False,
        official=True, manifest={"plugin": {"id": "stats", "version": "1.0"}},
    )
    fields.update(overrides)
    return an_inspection(**fields)


def installed_record(capabilities=(), trusted_modules=(), sha=None) -> dict:
    """The lockfile facts an inspection carries about a previous install."""
    return {"sha": sha, "version": "0.9.0", "capabilities": list(capabilities),
            "trusted_modules": list(trusted_modules), "enabled": True,
            "source_kind": "github_url"}


class Sources:
    """What ``inspect_source`` answers, per spec, with no network anywhere."""

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {}
        self.calls: list[str] = []
        #: Set to hold the inspection thread inside its call, which is how a
        #: test gets a SECOND inspect to arrive while the first is running.
        self.blocked = threading.Event()
        self.entered = threading.Event()

    def answer(self, spec: str, value: Any) -> None:
        self.answers[spec] = value

    def __call__(self, spec: str, *, lockfile: dict) -> Inspection:
        self.calls.append(spec)
        self.entered.set()
        if self.blocked.is_set():
            # Waits for the test to clear it; bounded so a forgotten clear
            # fails the test instead of hanging the suite.
            deadline = time.monotonic() + 20.0
            while self.blocked.is_set():
                if time.monotonic() > deadline:
                    raise AssertionError("the blocked inspection was never let go")
                time.sleep(0.01)
        answer = self.answers[spec]
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def sources(monkeypatch) -> Sources:
    table = Sources()
    monkeypatch.setattr(inspect_module, "inspect_source", table)
    return table


class ScriptedFlow:
    """A stand-in for ``flows.install_plugin_live``, driven from the test.

    The real flow blocks its thread for minutes at a time, which is exactly
    the property that makes "while a job is running" hard to test. This one
    blocks until the test tells it what to do next -- so a test can hold a
    job open, release ONE event into a parked long poll, and finish, with no
    sleep anywhere and no wall-clock guesswork. It honours ``cancel_check``
    between instructions, the way the real flow does between steps.
    """

    #: How long the worker thread waits for its next instruction before it
    #: gives up. A test that forgets to finish a job fails here with a clear
    #: message instead of hanging the suite.
    STARVED_AFTER_S = 20.0

    def __init__(self) -> None:
        self._steps: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.started = threading.Event()
        #: Every plan this flow was handed -- what consent decided, as the
        #: install itself sees it.
        self.plans: list[InstallPlan] = []

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
    def __call__(self, plan: InstallPlan, *, emit, cancel_check) -> InstallOutcome:
        self.plans.append(plan)
        self.started.set()
        deadline = time.monotonic() + self.STARVED_AFTER_S
        while True:
            if cancel_check():
                raise PluginCancelled(f"install of {plan.plugin_id} cancelled")
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
                return InstallOutcome(
                    plugin_id=plan.plugin_id, sha=plan.sha, deps_installed=(),
                    tombstone_cleared=False, replaced=plan.prior is not None,
                    plugin_dir=None)


async def drain(service: PluginService, job_id: str, *, timeout: float = 20.0
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
    loop has, so the task that would start the flow never gets to run and the
    wait always times out.
    """
    deadline = time.monotonic() + timeout
    while not flow.started.is_set():
        if time.monotonic() > deadline:
            raise AssertionError("the injected flow never started")
        await asyncio.sleep(0.01)


def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


async def remembered(service: PluginService, sources: Sources,
                     inspection: Inspection) -> str:
    """Store *inspection* the way a POST /inspect would, and return its id."""
    sources.answer(inspection.source, inspection)
    stored = await service.inspect(inspection.source)
    return stored.inspection_id


# ── inspecting ────────────────────────────────────────────────────────────


async def test_an_inspection_is_stored_under_an_id_with_an_expiry(sources):
    service = a_service(run_flow=ScriptedFlow())
    sources.answer(SOURCE, an_inspection())

    stored = await service.inspect(SOURCE)

    assert stored.inspection is sources.answers[SOURCE]
    assert len(stored.inspection_id) == 32       # secrets.token_hex(16)
    # ISO-8601 UTC, because "expires_at: 1731.9" means nothing in a browser.
    assert stored.expires_at.endswith("+00:00")
    assert service.get_inspection(stored.inspection_id) is stored
    # The lockfile is the service's job to load, not the caller's: every
    # ``inspect_*`` requires it and an inspection compared against the wrong
    # one reports an update as a fresh install.
    assert sources.calls == [SOURCE]


async def test_a_second_inspection_while_one_is_running_is_refused(sources):
    """Refused, not queued: an answer that arrives after the person has
    typed something else is an answer to a question nobody is asking."""
    service = a_service(run_flow=ScriptedFlow())
    sources.answer(SOURCE, an_inspection())
    sources.answer("bob/other", an_inspection(source="bob/other"))
    sources.blocked.set()

    first = asyncio.create_task(service.inspect(SOURCE))
    while not sources.entered.is_set():
        await asyncio.sleep(0.01)

    with pytest.raises(InspectBusy):
        await service.inspect("bob/other")

    sources.blocked.clear()
    stored = await first
    assert sources.calls == [SOURCE]
    # ...and the gate is released afterwards, not held by the job.
    sources.answer("bob/other", an_inspection(source="bob/other"))
    second = await service.inspect("bob/other")
    assert second.inspection_id != stored.inspection_id


async def test_a_source_that_cannot_be_read_travels_out_untranslated(sources):
    """The route maps these; the service must not turn one into a job."""
    service = a_service(run_flow=ScriptedFlow())
    sources.answer(SOURCE, GitHubError("no such repository", status=404))

    with pytest.raises(GitHubError) as excinfo:
        await service.inspect(SOURCE)
    assert excinfo.value.status == 404
    assert service.current_job() is None


async def test_an_inspection_that_timed_out_is_gone(sources, monkeypatch):
    """A consent screen left open overnight describes a branch that moved."""
    clock = [1000.0]
    monkeypatch.setattr(service_module, "monotonic", lambda: clock[0])
    service = a_service(run_flow=ScriptedFlow(), inspection_ttl_s=900.0)
    inspection_id = await remembered(service, sources, an_inspection())

    clock[0] += 899.0
    assert service.get_inspection(inspection_id).inspection_id == inspection_id

    clock[0] += 2.0
    with pytest.raises(InspectionExpired) as excinfo:
        service.get_inspection(inspection_id)
    assert excinfo.value.inspection_id == inspection_id


async def test_the_oldest_inspection_goes_when_the_store_is_full(sources):
    service = a_service(run_flow=ScriptedFlow(), max_inspections=2)
    first = await remembered(service, sources, an_inspection(source="one"))
    second = await remembered(service, sources, an_inspection(source="two"))
    third = await remembered(service, sources, an_inspection(source="three"))

    with pytest.raises(InspectionExpired):
        service.get_inspection(first)
    assert service.get_inspection(second).inspection_id == second
    assert service.get_inspection(third).inspection_id == third


async def test_reading_an_inspection_keeps_it_alive(sources):
    """LRU and not FIFO: the consent screen somebody is still looking at is
    the last one that should be thrown away."""
    service = a_service(run_flow=ScriptedFlow(), max_inspections=2)
    first = await remembered(service, sources, an_inspection(source="one"))
    second = await remembered(service, sources, an_inspection(source="two"))

    service.get_inspection(first)          # touched, so no longer the oldest
    third = await remembered(service, sources, an_inspection(source="three"))

    assert service.get_inspection(first).inspection_id == first
    assert service.get_inspection(third).inspection_id == third
    with pytest.raises(InspectionExpired):
        service.get_inspection(second)


async def test_an_id_nobody_ever_issued_reads_the_same_as_an_expired_one():
    service = a_service(run_flow=ScriptedFlow())
    with pytest.raises(InspectionExpired):
        service.get_inspection("not-an-id")


# ── consent, before a job exists ──────────────────────────────────────────


async def test_a_capability_nobody_granted_refuses_before_any_job(sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(
        service, sources,
        an_inspection(capabilities=("network", "filesystem")))

    with pytest.raises(ConsentRequired) as excinfo:
        await service.submit_install(inspection_id,
                                     accept_capabilities=["network"],
                                     trust_author=False, force=False)

    # The list is on the exception because the caller's next move is to show
    # it and ask again -- re-deriving it from the sentence is how a
    # translated string turns into a broken dialog.
    assert excinfo.value.missing_capabilities == ("filesystem",)
    assert service.current_job() is None
    assert not flow.started.is_set()
    # ...and the inspection is still there, so saying yes is one more click
    # rather than another round trip to the repository.
    granted = await service.submit_install(
        inspection_id, accept_capabilities=["network", "filesystem"],
        trust_author=False, force=False)
    flow.finish()
    await drain(service, granted.job_id)


async def test_a_module_list_needs_the_author_trusted(sources):
    """The other half of consent, and it has a name of its own: the two are
    answered with different controls."""
    service = a_service(run_flow=ScriptedFlow())
    inspection_id = await remembered(
        service, sources, an_inspection(allowed_modules=("subprocess", "os")))

    with pytest.raises(TrustAuthorRequired) as excinfo:
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=False)

    assert excinfo.value.allowed_modules == ("subprocess", "os")
    assert excinfo.value.missing_capabilities == ()
    # Still a ConsentRequired, so the CLI's own except clause is untouched.
    assert isinstance(excinfo.value, ConsentRequired)
    assert service.current_job() is None


async def test_what_a_previous_install_granted_is_not_asked_again(sources):
    """An update the user already answered for is not a second decision."""
    flow = ScriptedFlow().script()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection(
        mode="update",
        capabilities=("network",), allowed_modules=("subprocess",),
        installed=installed_record(capabilities=["network"],
                                   trusted_modules=["subprocess"])))

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=True)
    _, status = await drain(service, job.job_id)

    assert status == "done"
    assert flow.plans[0].granted_capabilities == ("network",)
    assert flow.plans[0].trust_author is True


async def test_a_capability_that_grew_since_the_install_is_a_new_decision(
        sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection(
        mode="update", capabilities=("network", "filesystem"),
        capabilities_added=("filesystem",),
        installed=installed_record(capabilities=["network"])))

    with pytest.raises(ConsentRequired) as excinfo:
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=True)

    assert excinfo.value.missing_capabilities == ("filesystem",)
    assert not flow.started.is_set()


async def test_a_builtin_pack_is_not_asked_about_capabilities(sources):
    """It arrived through a pull request in this repository: there is no
    third party here for the user to be asked about. Its capabilities are
    still RECORDED -- ``cdui plugin list`` answers "which of my plugins
    reaches the network" for every pack, wherever it came from."""
    flow = ScriptedFlow().script()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(
        service, sources,
        a_builtin_inspection(capabilities=("filesystem",),
                             allowed_modules=("subprocess",)))

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    _, status = await drain(service, job.job_id)

    assert status == "done"
    assert job.kind == "builtin"
    assert flow.plans[0].granted_capabilities == ("filesystem",)


async def test_a_plugin_that_is_already_here_is_an_offer_not_a_job(sources):
    """The one install failure that is not a failure of the plugin, and the
    flow cannot answer it in time: for a repository plugin an inspection of
    something installed is an UPDATE, which the flow's own copy lets through.
    """
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection(
        mode="update", installed=installed_record(sha="b" * 40)))

    with pytest.raises(AlreadyInstalled) as excinfo:
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=False)

    assert excinfo.value.plugin_id == "extras"
    assert service.current_job() is None
    assert not flow.started.is_set()


async def test_a_builtin_that_is_already_here_is_refused_the_same_way(sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, a_builtin_inspection(
        mode="update", installed=installed_record()))

    with pytest.raises(AlreadyInstalled):
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=False)

    assert service.current_job() is None
    assert not flow.started.is_set()


async def test_force_replaces_what_is_there(sources):
    flow = ScriptedFlow().script()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection(
        mode="update", installed=installed_record(sha="b" * 40)))

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=True)
    _, status = await drain(service, job.job_id)

    assert status == "done"
    assert job.mode == "update"
    assert flow.plans[0].force is True


async def test_an_expired_inspection_cannot_be_installed(sources, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(service_module, "monotonic", lambda: clock[0])
    service = a_service(run_flow=ScriptedFlow(), inspection_ttl_s=10.0)
    inspection_id = await remembered(service, sources, an_inspection())

    clock[0] += 11.0
    with pytest.raises(InspectionExpired):
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=False)
    assert service.current_job() is None


async def test_an_inspection_is_spent_by_the_install_it_started(sources):
    """One consent screen, one install. The refusals above deliberately do
    not spend it; a successful claim does."""
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    flow.finish()
    await drain(service, job.job_id)

    with pytest.raises(InspectionExpired):
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=False)


# ── one install at a time, across both installers ─────────────────────────


async def test_an_install_running_in_the_other_center_is_refused(sources):
    """One interpreter, two installers, and only one of them may write.

    A pack install and a plugin install both run ``uv pip install`` into the
    same site-packages under the same constraints freeze; two at once end in
    a corrupt environment or a deadlock. The refusal is this domain's own
    (``PluginBusy``, so the route maps it without learning a new type), and
    ``reason`` is what tells a panel it is about to offer Follow on somebody
    else's job.
    """
    flow = ScriptedFlow()
    service = a_service(run_flow=flow,
                            busy_elsewhere=lambda: "pack-job-1")
    inspection_id = await remembered(service, sources, an_inspection())

    with pytest.raises(PluginBusy) as excinfo:
        await service.submit_install(inspection_id, accept_capabilities=None,
                                     trust_author=False, force=False)

    assert excinfo.value.job_id == "pack-job-1"
    assert excinfo.value.reason == "pack_install_running"
    assert "pack install" in str(excinfo.value), (
        "the sentence a user reads must name the install that is actually "
        "running, not this service's own kind")
    # Refused before anything was claimed: no job, no events, nothing to
    # clean up -- and the inspection is still spendable once it is over.
    assert service.current_job() is None
    assert not flow.started.is_set()
    assert service.get_inspection(inspection_id).inspection_id == inspection_id


async def test_a_second_install_is_refused_with_the_id_to_follow(sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    first_id = await remembered(service, sources, an_inspection())
    second_id = await remembered(service, sources,
                                 an_inspection(source="bob/other"))

    running = await service.submit_install(first_id, accept_capabilities=None,
                                           trust_author=False, force=False)
    await wait_started(flow)

    with pytest.raises(PluginBusy) as excinfo:
        await service.submit_install(second_id, accept_capabilities=None,
                                     trust_author=False, force=False)
    assert excinfo.value.job_id == running.job_id
    # No reason: the job in the way is our own, which the id already implies.
    assert excinfo.value.reason is None

    flow.finish()
    await drain(service, running.job_id)


async def test_the_other_installer_is_told_about_a_running_job_and_no_other(
        sources):
    """What ``PackService(busy_elsewhere=...)`` is handed. A FINISHED job is
    in nobody's way -- it stays readable here, but reporting it would refuse
    every pack install until somebody started another plugin one."""
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    assert service.current_job_id() is None

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)
    assert service.current_job_id() == job.job_id

    flow.finish()
    await drain(service, job.job_id)
    assert service.current_job_id() is None
    assert service.current_job() is job


# ── the job itself ────────────────────────────────────────────────────────


async def test_job_started_names_the_plugin_the_kind_the_source_and_the_mode(
        sources):
    flow = ScriptedFlow().script()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    events, status = await drain(service, job.job_id)

    assert events[0]["type"] == "job_started"
    assert events[0]["plugin_id"] == "extras"
    assert events[0]["kind"] == "github"
    assert events[0]["source"] == SOURCE
    assert events[0]["mode"] == "install"
    assert status == "done"


async def test_the_reload_runs_on_the_loop_after_the_flow_and_before_done(
        sources, monkeypatch):
    """The registry dict is read by request handlers on this thread, so
    rebuilding it from the flow's worker thread would hand a request half a
    palette. It is a STEP as well, because a re-discovery takes a second and
    a panel with no current step for a second looks stopped."""
    loop = asyncio.get_running_loop()
    seen: dict[str, Any] = {}

    def fake_reload() -> dict:
        seen["loop"] = asyncio.get_running_loop()
        seen["thread"] = threading.get_ident()
        seen["status"] = service.current_job().status
        return {"total": 1}

    monkeypatch.setattr(service_module, "nodes_for_plugin",
                        lambda plugin_id, registry: [f"{plugin_id}.Node"])
    flow = ScriptedFlow().script({"type": "log", "line": "installing"})
    service = PluginService(run_flow=flow, reload=fake_reload)
    inspection_id = await remembered(service, sources, an_inspection())

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    events, status = await drain(service, job.job_id)

    assert seen["loop"] is loop
    assert seen["thread"] == threading.get_ident()
    # It ran before the terminal event: the job was still running, and the
    # flow's own event was already in the buffer.
    assert seen["status"] == "running"
    assert types_of(events) == ["job_started", "log", "step_started",
                                "step_done", "job_done"]
    assert events[2]["step"] == "reload"
    assert events[3]["step"] == "reload"
    assert status == "done"
    assert job.current_step is None


async def test_job_done_carries_the_nodes_and_the_generation_after_reloading(
        sources, monkeypatch):
    """Both read AFTER the re-discovery, which is the whole point of reading
    them there: a generation from before it is the number the panel already
    had, and nodes from before it are the ones this install did not add."""
    generation = [7]
    monkeypatch.setattr(plugin_loader, "reload_generation",
                        lambda: generation[0])
    monkeypatch.setattr(service_module, "nodes_for_plugin",
                        lambda plugin_id, registry: ["Extras-Thing"])

    def fake_reload() -> dict:
        generation[0] += 1
        return {"total": 3}

    flow = ScriptedFlow().script()
    service = PluginService(run_flow=flow, reload=fake_reload)
    inspection_id = await remembered(service, sources, an_inspection())

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    events, status = await drain(service, job.job_id)

    assert status == "done"
    assert events[-1] == {
        "type": "job_done", "plugin_id": "extras", "sha": SHA,
        "nodes": ["Extras-Thing"], "generation": 8,
        "cursor": events[-1]["cursor"], "ts": events[-1]["ts"]}


async def test_a_reload_that_fails_fails_the_job(sources):
    """The install landed on disk, but nothing can load it: reporting that
    as done would leave a panel showing a plugin the editor has not got."""
    def fake_reload() -> dict:
        raise RuntimeError("a node module blew up on import")

    flow = ScriptedFlow().script()
    service = PluginService(run_flow=flow, reload=fake_reload)
    inspection_id = await remembered(service, sources, an_inspection())

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    assert job.status == "failed"
    assert "blew up on import" in events[-1]["message"]
    # The step never closed, because it never finished.
    assert types_of(events) == ["job_started", "step_started", "job_failed"]


async def test_the_active_job_is_the_running_one_and_nothing_else(sources):
    """``catalog_listing`` renders ANY row carrying a job as ``installing``
    without reading its status, so a finished job here would leave that row
    spinning until the next install."""
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    assert service.active_job_payload() is None

    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    assert service.active_job_payload() == {
        "job_id": job.job_id, "plugin_id": "extras",
        # The panel reads this as install-or-update: it is what decides
        # whether the toast says "installed" or "updated".
        "kind": "install", "status": "running", "current_step": None}

    flow.finish()
    await drain(service, job.job_id)
    assert service.active_job_payload() is None


# ── how a job ends ────────────────────────────────────────────────────────


async def test_a_cancel_ends_the_job_as_cancelled_not_as_failed(sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    assert await service.cancel(job.job_id) is True
    events, status = await drain(service, job.job_id)

    assert status == "cancelled"
    assert events[-1]["type"] == "job_cancelled"
    # Asking again is not an error, and the answer is no: it is over.
    assert await service.cancel(job.job_id) is False


async def test_packages_that_cannot_be_installed_here_ask_for_a_restart(
        sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    flow.fail(PluginNeedsRestart(
        "torch cannot be replaced while it is imported",
        command="uv pip install torch==2.4.0", hint="resolver conflict"))
    events, status = await drain(service, job.job_id)

    assert status == "needs_restart"
    assert events[-1] == {
        "type": "needs_restart", "command": "uv pip install torch==2.4.0",
        "hint": "resolver conflict", "cursor": events[-1]["cursor"],
        "ts": events[-1]["ts"]}
    # On the job as well as in the event: the panel's summary reads the job.
    assert job.restart_command == "uv pip install torch==2.4.0"


async def test_an_install_failure_carries_its_message_and_its_hint(sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    flow.fail(PluginInstallError("extras was refused by the security scan.",
                                 hint="nodes/evil.py:12 imports subprocess"))
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    assert events[-1]["message"] == "extras was refused by the security scan."
    assert events[-1]["hint"] == "nodes/evil.py:12 imports subprocess"
    assert job.error == {"message": events[-1]["message"],
                         "hint": events[-1]["hint"]}


async def test_a_github_failure_says_what_github_answered(sources):
    """It escapes the download step untranslated, and the status is the
    user's next move: a 404 is a typo in the repo name."""
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    flow.fail(GitHubError("could not download the tarball", status=502))
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    assert events[-1]["message"] == "could not download the tarball"
    assert events[-1]["hint"] == "GitHub answered 502"


async def test_an_exception_nobody_designed_still_ends_the_job(sources):
    """A raw OSError out of a copy, a ValueError for a kind this build
    cannot install: not a shape anybody wrote a message for, and still not
    allowed to leave a job saying running forever."""
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    flow.fail(OSError("[WinError 32] the file is in use"))
    events, status = await drain(service, job.job_id)

    assert status == "failed"
    assert "the file is in use" in events[-1]["message"]
    assert events[-1]["hint"] is None


async def test_a_finished_job_stays_readable(sources):
    """The last thing a client asks for is the tail of a job that just
    ended, so the job and its events live until the next claim."""
    flow = ScriptedFlow().script({"type": "log", "line": "done"})
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await drain(service, job.job_id)

    events, cursor, status = await service.wait_for_events(
        job.job_id, after_cursor=0, limit=500, wait=0.0)
    assert status == "done"
    assert types_of(events)[-1] == "job_done"
    assert cursor == events[-1]["cursor"]
    assert service.get_job(job.job_id) is job


# ── shutdown ──────────────────────────────────────────────────────────────


async def test_shutdown_cancels_a_running_install(sources):
    flow = ScriptedFlow()
    service = a_service(run_flow=flow)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)
    await wait_started(flow)

    await service.shutdown()

    assert job.cancel_event.is_set()
    assert job.status == "cancelled"


async def test_shutdown_with_no_job_is_a_no_op():
    service = a_service(run_flow=ScriptedFlow())
    await service.shutdown()
    assert service.current_job() is None


async def test_shutdown_is_bounded_when_the_flow_ignores_cancellation(sources):
    """Server shutdown must not be held hostage by a stubborn install."""
    release = threading.Event()

    def deaf_flow(plan, *, emit, cancel_check) -> InstallOutcome:
        release.wait(20)
        return InstallOutcome(plugin_id=plan.plugin_id, sha=plan.sha,
                              deps_installed=(), tombstone_cleared=False,
                              replaced=False, plugin_dir=None)

    service = PluginService(run_flow=deaf_flow, reload=lambda: {},
                            shutdown_timeout_s=0.2)
    inspection_id = await remembered(service, sources, an_inspection())
    job = await service.submit_install(inspection_id, accept_capabilities=None,
                                       trust_author=False, force=False)

    started = time.monotonic()
    try:
        await service.shutdown()
        assert time.monotonic() - started < 5.0
        assert job.cancel_event.is_set()
    finally:
        # Let the stubborn flow finish so no task outlives the test loop.
        release.set()
        await drain(service, job.job_id)
