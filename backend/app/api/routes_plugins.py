"""API routes for the Plugin Center: what is installed, and what could be.

    GET  /api/plugins             every INSTALLED plugin, enabled or not
    GET  /api/plugins/catalog     the same rows merged with what this build
                                  can install by name -- what the panel draws
    GET  /api/plugins/generation  the reload counter the editor polls
    POST /api/plugins/reload      re-discover nodes, presets and packs
    POST /api/plugins/inspect     read a source and say what installing it
                                  would cost -- installs nothing
    POST /api/plugins/install     install what an inspection described,
                                  as a job -> 202 {job_id}
    GET  /api/plugins/jobs/{id}/events   that job's log, replayed after a
                                  cursor, with an optional long poll
    POST /api/plugins/jobs/{id}/cancel   ask it to stop
    GET  /api/plugins/{id}        one plugin's manifest, nodes and README
    DELETE /api/plugins/{id}      uninstall it, and say what that left behind
    POST /api/plugins/{id}/enable|disable

Every fixed path is declared before ``/{plugin_id}``; see the comment above
that route for why the order is load-bearing rather than tidy.

Installing is a conversation with two turns. ``/inspect`` reads the source
and answers with what it found and what it would cost; ``/install`` acts on
THAT answer, by its id. So the user agrees to the manifest they were shown
rather than to whatever the branch holds a minute later, and the server never
takes a manifest, a sha or a capability list from a request body. Built-in
packs go through both turns too, so the panel has one flow rather than two.

Reads are open GETs, like every other read the editor polls -- including a
job's events, which is what a second tab that opened mid-install follows.
Everything else here takes the session token (the global ``auth_guard``).
The routes that change what code is on this machine -- inspect, install,
cancel, delete -- are additionally refused unless the server is bound to
loopback (``_require_local_plugin_install``): installing a plugin puts
third-party code where this process will import it, inspecting reaches out
to GitHub on the caller's word, and deleting takes somebody's plugin away
(cancel is in that set because it stops the install they started). None of
them is a thing a stranger on the LAN gets to do.
``cdui plugin install`` still does the same job from a terminal, through the
same flow. ``/reload`` and ``/{id}/enable|disable`` stay token-only on
purpose: they act on code this machine already has and the user already
agreed to.

Refusals from the install path carry a CODE rather than a sentence --
``HTTPException(status, detail={"code": ...})`` -- because the panel draws a
different control for each of them (fix the name, tick the capability,
inspect again), and choosing one by matching on prose breaks the day the
prose is translated.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, field_validator

from ..config import settings
from ..core.auth import bound_to_loopback
from ..core.node_registry import registry
from ..core import plugin_loader
from ..core.plugin_loader import (
    is_enabled,
    iter_plugin_dirs,
    load_lockfile,
)
from ..core.jobs import JobBusy
from ..core.plugins import lifecycle
from ..core.plugins.catalog import catalog_entries
from ..core.plugins.errors import (
    AlreadyInstalled,
    ConsentRequired,
    GitHubError,
    InspectBusy,
    InspectionExpired,
    ManifestError,
    PluginBusy,
    PluginInstallError,
    ReservedPluginId,
    SourceError,
    TrustAuthorRequired,
    UnknownCatalogName,
)
from ..core.plugins.listing import (
    catalog_listing,
    frontend_entry_url,
    installed_facts,
    nodes_for_plugin,
)
from ..core.plugins.reload import rediscover_now
from ..core.plugins.service import PluginService, StoredInspection, UnknownJob

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport -- same API.

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins", tags=["plugins"])

_REMOTE_REFUSAL = (
    "Installing plugins is only allowed from the computer that runs the "
    "server. Set CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1 to override.")

#: GitHub statuses that mean "ask again later" rather than "that is not
#: there". 403 is how the API answers an exhausted rate limit -- its JSON
#: body says so while the reason phrase only says "Forbidden" -- and 429 is
#: the same fact from whatever sits in front of it.
_RATE_LIMITED = frozenset({403, 429})


class PluginInspectRequest(BaseModel):
    """What to read. One field, and ``extra="forbid"`` keeps it one.

    A source is a catalog name, ``owner/repo`` or a GitHub URL -- never a
    manifest, a sha or a capability list. Everything an install later acts
    on is read by the SERVER from that source and kept under an inspection
    id, so there is no field here for a client to smuggle a decision through.

    Named for its domain, like its sibling below: see
    :class:`PluginInstallRequest` for why that is not decoration.
    """

    model_config = ConfigDict(extra="forbid")

    source: str

    @field_validator("source")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Refuse whitespace; hand the rest on trimmed.

        A box submitted empty is a 422 rather than a round trip for
        ``parse_source`` to call it unparseable, and the trim is here
        because a pasted URL usually arrives with one on the end -- the
        cheapest place to lose it is before anything looks it up.
        """
        source = value.strip()
        if not source:
            raise ValueError("a source to inspect, not an empty string")
        return source


class PluginInstallRequest(BaseModel):
    """An inspection, and the answers to what it asked. ``extra="forbid"``.

    Prefixed rather than called ``InstallRequest``, which would read better
    in this file and cost something outside it: ``routes_packs`` already has
    a model of that name, and FastAPI resolves a collision by falling back to
    the module-qualified component name for BOTH -- silently renaming the
    Package Center's schema in ``/docs`` and in anything generated from
    ``/openapi.json``. It is also the name the frontend's own type carries
    (``PluginInstallRequest`` in ``api/rest.ts``).

    Everything an install acts on -- the repository, the commit, the manifest
    -- comes from the inspection named here, never from this body. What IS
    here is what only the user can say.

    ``accept_capabilities`` is a LIST of capability ids and never a boolean.
    "Yes to everything" has to be the client enumerating what it is saying
    yes to, because that is the only form still meaningful when the manifest
    asks for one capability more than the dialog was drawn from -- a schema
    that took ``true`` would turn a stale dialog into a blanket grant.
    """

    model_config = ConfigDict(extra="forbid")

    inspection_id: str
    accept_capabilities: list[str] | None = None
    #: The other half of consent: agreeing to the modules the manifest asks
    #: to import, which is a decision about the AUTHOR rather than the code.
    trust_author: bool = False
    #: Replace what is already installed. What it lifts is an OFFER
    #: (Reinstall, ``--force``), not a failure.
    force: bool = False


def _coded(status_code: int, code: str, **fields: Any) -> HTTPException:
    """A refusal the panel can act on: ``detail`` is a dict with a ``code``.

    ``/api/packs`` answers with flat prose, and it is right to: one panel,
    one sentence, one button. Here the client has to choose between several
    different controls -- fix the name, tick the capability, trust the
    author, follow the job that is in the way, inspect again -- and picking
    one by matching on a sentence breaks the day the sentence is translated.
    The fields beside the code (``known``, ``missing_capabilities``,
    ``allowed_modules``, ``job_id``, ``id``) are what those controls are
    drawn from. Deliberately no ``message``: prose about a coded refusal
    belongs to whoever is talking to the user, in their language.
    """
    return HTTPException(status_code=status_code,
                         detail={"code": code, **fields})


def _installer(request: Request) -> PluginService | None:
    """The service the lifespan built, or ``None``.

    Optional on ``app.state`` for the same reason every other store is: the
    lifespan does not run under httpx's ASGITransport, so a test reaches
    these routes with nothing there unless it installs one.
    """
    return getattr(request.app.state, "plugin_service", None)


def _service(request: Request) -> PluginService:
    """The service, or 503 -- for every route that needs one to answer.

    ``GET /catalog`` deliberately does NOT go through here: it is a read,
    and a server whose installer failed to start can still say what is
    installed.
    """
    service = _installer(request)
    if service is None:
        raise _coded(503, "unavailable")
    return service


def _github_refusal(exc: GitHubError) -> HTTPException:
    """One GitHub failure, split three ways for three different next steps.

    A 404 is the caller's -- a typo in the owner, the repository or the ref
    -- so it travels as a 404 they can act on. Everything else happened
    between this server and GitHub, which is a 502 whether it was a rate
    limit, a 500 or a name that never resolved; the code says which, because
    "wait" and "check the network" are different waits.
    """
    if exc.status == 404:
        return _coded(404, "not_found")
    if exc.status in _RATE_LIMITED:
        return _coded(502, "github_rate_limited")
    return _coded(502, "github_unreachable")


def _inspect_refusal(exc: PluginInstallError) -> HTTPException:
    """The ``PluginInstallError``s an INSPECTION can raise, coded.

    Neither is a failed install -- nothing was installed -- so both are 400s
    about the source. ``inspect_github`` refuses an id this build owns (a
    route under ``/api/plugins/``, a pack that ships here, another
    repository's catalog row) and ``inspect_source`` refuses a catalog row
    too malformed to install from.

    Told apart by CLASS, and the id comes off the exception. It used to be a
    regular expression over the message, because the id was not carried
    anywhere else -- which quietly made the wording of an English sentence
    part of this wire contract: rephrasing that refusal, or translating it,
    would have turned every reserved id into ``invalid_manifest``.
    """
    if isinstance(exc, ReservedPluginId):
        return _coded(400, "reserved_id", id=exc.plugin_id)
    return _coded(400, "invalid_manifest")


def _job_not_found(job_id: str) -> HTTPException:
    """Only the most recent job is kept, so "gone" and "never existed" are
    the same answer -- and the client's next move is the same either way."""
    return _coded(404, "unknown_job", job_id=job_id)


def _refuse_while_busy(request: Request, plugin_id: str) -> None:
    """409 while THIS plugin's own install is running. Silent otherwise.

    An install writes ``plugins/<id>/`` and the lockfile entry beside it; a
    delete removes both, and enable/disable rewrites the entry. Doing either
    while the flow is halfway through is how a plugin ends up on disk with no
    entry, or with an entry pointing at a directory the delete took away --
    and the flow runs on a worker thread, so nothing else would notice.

    Only THAT plugin's job, deliberately. The service runs one install at a
    time across the whole process, so refusing on any running job would make
    a long install of one pack block every lifecycle action on all the
    others, for a conflict that cannot exist: two different plugins are two
    different directories and two different lockfile keys.

    The service is ASKED FOR rather than required (``_installer``, not
    ``_service``): removing a plugin or flipping its flag is a lockfile edit
    and a re-discovery, and a server whose installer never started can still
    do both. No installer means no job, which is the honest answer to "is
    this plugin busy" -- a 503 here would refuse to uninstall a plugin
    because the thing that installs plugins is not running.
    """
    service = _installer(request)
    if service is None:
        return
    job = service.current_job()
    # A FINISHED job is not in anybody's way: it stays readable so a panel can
    # fetch its tail, and nothing is writing the plugin's files any more.
    if job is None or job.terminal or job.plugin_id != plugin_id:
        return
    raise _coded(409, "busy", job_id=job.job_id)


def _inspection_payload(stored: StoredInspection) -> dict[str, Any]:
    """One inspection, in the fields the Plugin Center is written against.

    Selected field by field rather than dumped: an ``Inspection`` also
    carries ``owner`` and ``repo``, which the install path needs and the
    wire contract does not have, and a response built by serialising the
    dataclass would publish whatever the next field added to it happens to
    be. The order is the contract's, so a diff of this list is a diff of the
    TypeScript type beside it.
    """
    found = stored.inspection
    return {
        "inspection_id": stored.inspection_id,
        "expires_at": stored.expires_at,
        "kind": found.kind,
        "mode": found.mode,
        "plugin_id": found.plugin_id,
        "catalog_id": found.catalog_id,
        "official": found.official,
        "source": found.source,
        "url": found.url,
        "ref": found.ref,
        "sha": found.sha,
        "name": found.name,
        "version": found.version,
        "description": found.description,
        "homepage": found.homepage,
        "manifest": found.manifest,
        "capabilities": list(found.capabilities),
        "allowed_modules": list(found.allowed_modules),
        "python_deps": dict(found.python_deps),
        "has_frontend": found.has_frontend,
        "chapters": list(found.chapters),
        "lessons": list(found.lessons),
        "consent_required": found.consent_required,
        "installed": found.installed,
        "up_to_date": found.up_to_date,
        "capabilities_added": list(found.capabilities_added),
        "allowed_modules_added": list(found.allowed_modules_added),
        "warnings": list(found.warnings),
    }


def remote_plugin_install_allowed() -> bool:
    """May a request install a plugin at all, given how the server is bound?

    Installing a plugin puts third-party code where this process will import
    it, so the audience for that is "whoever is at this machine" rather than
    "whoever can reach the port" -- the same rule ``/api/packs`` applies to
    starting a package install, asked through the same helper so the two
    cannot drift apart.
    """
    return bound_to_loopback() or bool(settings.ALLOW_REMOTE_PLUGIN_INSTALL)


async def _require_local_plugin_install() -> None:
    """Dependency for every route that installs, inspects or removes.

    Inspecting is guarded too, and deliberately: it is a request this server
    makes to GitHub on the caller's word, and the answer it stores is what
    an install is later authorised by. A read of what is already here needs
    no such gate, which is why ``GET /catalog`` does not carry one.
    """
    if not remote_plugin_install_allowed():
        raise HTTPException(status_code=403, detail=_REMOTE_REFUSAL)


@router.get("")
async def list_plugins() -> list[dict[str, Any]]:
    """List every installed plugin (enabled + disabled) with metadata.

    ``include_disabled=True`` so the frontend can render disabled rows
    greyed-out without an extra round-trip. Each entry carries an
    explicit ``enabled`` field; nodes list is empty for disabled plugins
    because they are not in the registry.
    """
    lockfile = load_lockfile()
    catalog = catalog_entries()
    out: list[dict[str, Any]] = []
    for plugin_id, plugin_dir in iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(),
        plugin_loader.plugins_user_root(),
        lockfile,
        include_disabled=True,
    ):
        entry = lockfile["plugins"][plugin_id]
        manifest = plugin_loader.read_manifest_safe(plugin_dir)
        plugin_meta = manifest.get("plugin", {})
        lessons_meta = manifest.get("lessons", {})
        enabled = is_enabled(entry)
        out.append({
            "id": plugin_id,
            "name": plugin_meta.get("name", plugin_id),
            "version": plugin_meta.get("version", ""),
            "description": plugin_meta.get("description", ""),
            "source_kind": entry.get("source_kind", ""),
            "source": entry.get("source", plugin_id),
            "sha": entry.get("sha", ""),
            "ref": entry.get("ref", ""),
            "installed_at": entry.get("installed_at", ""),
            "enabled": enabled,
            "homepage": plugin_meta.get("homepage", ""),
            "chapters": lessons_meta.get("chapters", []),
            "lessons": lessons_meta.get("lessons", []),
            "nodes": nodes_for_plugin(plugin_id, registry),
            "frontend_entry": frontend_entry_url(
                plugin_id, plugin_dir, manifest, enabled=enabled
            ),
            # Additive (the six fields the Plugin Center needs on a row it
            # can act on), computed by the same rules /catalog uses.
            **installed_facts(plugin_id, entry, manifest, catalog),
        })
    return out


@router.get("/catalog")
async def plugin_catalog(request: Request) -> dict[str, Any]:
    """Every plugin this build can install, and everything installed.

    The one route the Plugin Center polls, and the reason it is a GET: like
    ``GET /api/packs`` it is a read the editor draws a panel from, so it
    carries no session token. Declared before ``/{plugin_id}`` -- ``catalog``
    is a reserved plugin id precisely because a pack with that name would
    otherwise sit where this route lives, and the router, not the pack, would
    decide which one wins.

    ``active_job`` is the install running right now, which is how a panel
    that has just been opened -- a second tab, a reload mid-install -- finds
    the job it should be following. It is ``None`` when nothing is running
    AND when there is no service at all: this is a READ, and a server whose
    installer failed to start can still say what is installed. The routes
    that would actually install something answer 503 instead.
    """
    service = _installer(request)
    return catalog_listing(
        load_lockfile(),
        registry=registry,
        active_job=None if service is None else service.active_job_payload(),
        remote_install_allowed=remote_plugin_install_allowed(),
        generation=plugin_loader.reload_generation(),
    )


@router.get("/generation")
async def plugins_generation() -> dict[str, int]:
    """Monotonic counter bumped on every reload (plugin/node/enable-disable).

    The editor polls this in dev mode (when a linked plugin is present) to learn
    when to re-activate plugin frontends without a manual refresh. Declared
    before ``/{plugin_id}`` so it isn't swallowed by the dynamic route; a GET, so
    it needs no session token.
    """
    return {"generation": plugin_loader.reload_generation()}


@router.post("/reload")
async def reload_plugins() -> dict[str, int]:
    """Clear and re-discover everything (builtin + custom + plugins + presets)."""
    return rediscover_now()


@router.post("/inspect", dependencies=[Depends(_require_local_plugin_install)])
async def inspect_plugin_source(
    body: PluginInspectRequest, request: Request
) -> dict[str, Any]:
    """Read a source and report what installing it would cost. Installs
    nothing, downloads no tarball, and runs no code from it.

    The first of the two turns every install goes through -- built-in packs
    included, so the panel has one flow rather than two. The answer is kept
    server-side under an ``inspection_id``; the second turn (``/install``)
    names that id, which is what makes the user's "yes" an answer about the
    manifest they were SHOWN rather than about whatever the branch holds a
    minute later.

    Every refusal here is about the source and nothing has been written, so
    all of them are 4xx or 502 with a code the panel can draw a control for:
    a name the catalog does not have arrives with the names it does
    (``known``), an id this build owns arrives with the id (``id``), and a
    GitHub failure is split by what it means (see :func:`_github_refusal`).
    """
    service = _service(request)
    try:
        stored = await service.inspect(body.source)
    except InspectBusy:
        # Before ``PluginInstallError``: it is a subclass, and it is not a
        # refusal about the source at all -- asking again works.
        raise _coded(409, "inspect_busy") from None
    except UnknownCatalogName as exc:
        # The one source failure with something to offer: the user typed a
        # NAME, and what they cannot see is which names this build has.
        raise _coded(400, "unknown_catalog_name",
                     known=list(exc.known)) from None
    except SourceError:
        # ``UnparseableSource`` and anything else ``parse_source`` grows:
        # the string names nothing installable, and there is nothing true to
        # add to that beyond what the client already sent.
        raise _coded(400, "unparseable_source") from None
    except (ManifestError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        # A manifest this build will not install, the broken TOML that never
        # got as far as being validated, and a file that is not text at all --
        # ``fetch_manifest_text`` lets ``UnicodeDecodeError`` out on purpose,
        # and it is neither a ``ManifestError`` nor a ``SourceError``, so it
        # used to leave here as a 500. One code for the three: they all mean
        # "the thing at that source is not a plugin this can install", and
        # none of them is anything the caller can fix from here.
        raise _coded(400, "invalid_manifest") from None
    except OSError:
        # The DISK, and only the disk: a builtin whose manifest file is gone,
        # where ``read_manifest`` raises ``FileNotFoundError`` and nothing
        # above catches it. Every network ``OSError`` -- including one raised
        # while the response body is being read -- is turned into a
        # ``GitHubError`` by the GitHub client and answered below, which is
        # what keeps this clause from reporting a dropped connection as a
        # manifest that is not a manifest.
        raise _coded(400, "invalid_manifest") from None
    except GitHubError as exc:
        raise _github_refusal(exc) from None
    except PluginInstallError as exc:
        raise _inspect_refusal(exc) from None
    return _inspection_payload(stored)


@router.post("/install", status_code=202,
             dependencies=[Depends(_require_local_plugin_install)])
async def install_plugin(
    body: PluginInstallRequest, request: Request
) -> dict[str, str]:
    """Install what an inspection described. 202, and a ``job_id`` to follow.

    The second turn of the conversation ``/inspect`` began: the plugin, the
    commit and the manifest all come from the stored inspection, and this
    body carries only what the user had to decide.

    Everything that can be known before the install starts is refused HERE
    rather than reported as a failed job half a minute later -- an inspection
    that has expired, a plugin that is already installed, a capability nobody
    ticked, a module list nobody trusted, and an install already running in
    either center (they share one interpreter). A 202 means the job exists
    and its first event is already in the buffer.

    The order of the clauses below is load-bearing, not tidy:
    ``TrustAuthorRequired`` is a ``ConsentRequired`` and would otherwise be
    answered as one, which asks the user to tick a capability list that has
    nothing wrong with it -- and ``PluginBusy`` is a ``JobBusy``, so the one
    that can say WHOSE job is in the way has to be caught first.
    """
    service = _service(request)
    try:
        job = await service.submit_install(
            body.inspection_id,
            accept_capabilities=body.accept_capabilities,
            trust_author=body.trust_author,
            force=body.force)
    except InspectionExpired as exc:
        # Built from the attribute: this one is a KeyError, whose ``str`` is
        # the id in quotes rather than a sentence.
        raise _coded(404, "inspection_expired",
                     inspection_id=exc.inspection_id) from None
    except AlreadyInstalled as exc:
        # Not a failure of anything: an offer, which the client accepts by
        # asking again with ``force``.
        raise _coded(409, "already_installed",
                     plugin_id=exc.plugin_id) from None
    except TrustAuthorRequired as exc:
        raise _coded(400, "trust_author_required",
                     allowed_modules=list(exc.allowed_modules)) from None
    except ConsentRequired as exc:
        raise _coded(400, "consent_required",
                     missing_capabilities=list(
                         exc.missing_capabilities)) from None
    except PluginBusy as exc:
        # ``reason`` says whose job is in the way: ours (``None`` -> "busy",
        # and the id is one this client can follow) or the Package Center's
        # (``pack_install_running``, which is somebody else's to wait for).
        raise _coded(409, exc.reason or "busy", job_id=exc.job_id) from None
    except JobBusy as exc:
        # Unreachable today: the service pre-checks both installers and
        # raises ``PluginBusy`` above, so the runner's own claim never loses
        # the race. Here because the runner is what actually owns "one at a
        # time" -- if that pre-check is ever dropped or raced, this is a 409
        # the panel already knows how to draw rather than a 500.
        raise _coded(409, "busy", job_id=exc.job_id) from None
    return {"job_id": job.job_id}


@router.get("/jobs/{job_id}/events")
async def plugin_job_events(
    job_id: str,
    request: Request,
    cursor: int = Query(default=0, ge=0),
    wait: float = Query(default=0.0, ge=0.0, le=60.0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    """Events strictly after *cursor*, oldest first.

    The Package Center's route, over the other installer, down to the bounds
    -- one panel draws both, and a long poll that behaved differently on the
    two would be a bug found only in the half nobody was watching.

    ``wait`` seconds of long polling when the tail is empty: the request
    parks on an in-process wake-up (never a retry loop) and returns the
    moment an event lands, the job ends, or the deadline passes. A finished
    job answers immediately regardless of ``wait``, and its events stay
    readable until the next install replaces them -- which is what makes the
    last page of a failed install fetchable at all.

    An open GET, like every other read the editor polls: a second tab that
    opened mid-install follows the job it found in ``/catalog``.

    The returned ``cursor`` is where to resume, and never moves backwards.
    """
    service = _service(request)
    try:
        events, next_cursor, status = await service.wait_for_events(
            job_id, after_cursor=cursor, limit=limit, wait=wait)
    except UnknownJob:
        # Also reachable AFTER the park: a job that ends while a poll is
        # waiting on it, followed by a new install, takes its events with it.
        raise _job_not_found(job_id) from None
    return {"job_id": job_id, "status": status, "events": events,
            "cursor": next_cursor}


@router.post("/jobs/{job_id}/cancel",
             dependencies=[Depends(_require_local_plugin_install)])
async def cancel_plugin_job(job_id: str, request: Request) -> dict[str, Any]:
    """Ask the running install to stop.

    ``cancelled`` reports whether the request did anything: False for a job
    that had already finished. Both are 200 -- asking twice is not an error,
    and a cancel that raced the last step is normal. Cooperative, so the
    status may still say ``running`` when this returns: it is the FLOW that
    ends the job, between steps or inside a download.
    """
    service = _service(request)
    try:
        cancelled = await service.cancel(job_id)
    except UnknownJob:
        raise _job_not_found(job_id) from None
    return {"job_id": job_id, "cancelled": cancelled}


# Everything above this line has a FIXED path; everything below takes a
# ``{plugin_id}``. Keeping the split in that order is not a style choice --
# Starlette matches in registration order, so a fixed path declared after
# ``/{plugin_id}`` is reachable only because no pack is allowed to be called
# ``reload`` (see RESERVED_PLUGIN_IDS). A structural test pins the order so
# the next route added here cannot quietly depend on that.


@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str) -> dict[str, Any]:
    lockfile = load_lockfile()
    if plugin_id not in lockfile.get("plugins", {}):
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not installed")

    for pid, plugin_dir in iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), lockfile
    ):
        if pid != plugin_id:
            continue
        manifest = plugin_loader.read_manifest_safe(plugin_dir)
        readme_path = plugin_dir / "README.md"
        readme = ""
        if readme_path.exists():
            try:
                readme = readme_path.read_text(encoding="utf-8")
            except OSError:
                pass
        return {
            "id": plugin_id,
            "manifest": manifest,
            "lockfile_entry": lockfile["plugins"][plugin_id],
            "nodes": nodes_for_plugin(plugin_id, registry),
            "readme": readme,
        }

    raise HTTPException(
        status_code=404,
        detail=f"Plugin '{plugin_id}' is in the lockfile but its files are missing",
    )


@router.delete("/{plugin_id}",
               dependencies=[Depends(_require_local_plugin_install)])
async def uninstall_plugin(plugin_id: str, request: Request) -> dict[str, Any]:
    """Remove an installed plugin, and say what removing it left behind.

    Behind the same two gates as an install (the session token and a loopback
    bind): deleting somebody's plugin is not a thing a stranger on the LAN
    gets to do either.

    What is removed is decided by ``lifecycle.uninstall_plugin`` -- the same
    call ``cdui plugin uninstall`` makes, so the terminal and the panel cannot
    disagree about which files go: a downloaded pack's directory is deleted,
    a built-in one loses its lockfile entry and gains a ``removed`` record
    (#175) with the repo's own files untouched, and a linked development
    directory is left exactly where its author put it.

    What is NOT removed is the plugin's Python dependencies. Uninstalling
    packages from inside the process that imported them is how you get a
    half-loaded interpreter serving requests, so they are reported instead:
    ``python_deps_left`` and the ``uninstall_command`` to run by hand with the
    server stopped. They are in the answer because whatever draws this has to
    SAY so -- an uninstall that silently leaves packages behind is the half of
    the story the user finds out about from a disk that never got smaller.

    Then this process forgets the plugin, in this order: its modules leave
    ``sys.modules`` and only THEN is the registry re-discovered. Nothing else
    ever drops those modules -- a re-discovery only rebuilds the namespaces of
    plugins the lockfile still lists -- and the re-discovery is what bumps the
    generation the editor polls. Purging afterwards would publish "the palette
    changed" while the deleted pack was still importable in this interpreter,
    which is the one moment a client is guaranteed to come back and read.
    """
    _refuse_while_busy(request, plugin_id)

    outcome = lifecycle.uninstall_plugin(plugin_id)
    if outcome is None:
        raise _coded(404, "not_installed")
    if not outcome.removed:
        # The files could not be deleted -- Windows holding one open is the
        # ordinary cause -- so the lockfile entry was left alone and the
        # plugin is still installed. Reported as a conflict rather than a
        # 500: nothing is broken, something is in the way. ``error`` is the
        # operating system's own sentence and ``hint`` is what to do about
        # it -- the half a client can put in front of the user.
        raise _coded(
            409, "files_locked",
            error=outcome.error,
            hint=(f"{outcome.directory} is still there. Close whatever is "
                  "using those files -- or stop the server -- and remove "
                  "the plugin again."))

    plugin_loader.purge_plugin_modules(plugin_id)
    rediscover_now()

    return {
        "id": plugin_id,
        # Read off the outcome rather than written as ``True``: it is true by
        # construction here (the other exit is the 409 above), and reading it
        # keeps that an observation of what the lifecycle did.
        "removed": outcome.removed,
        "tombstoned": outcome.tombstoned,
        "files_removed": outcome.files_removed,
        "python_deps_left": list(outcome.python_deps_left),
        "uninstall_command": outcome.uninstall_command,
        "reinstall_hint": outcome.reinstall_hint,
    }


def _set_plugin_enabled(plugin_id: str, enabled: bool) -> dict[str, Any]:
    """Shared implementation behind the two toggle endpoints.

    Returns the new state on success; raises HTTPException 404 when the
    plugin is not installed. Hot-reloads the registry so the change is
    immediately visible without restarting the server -- including when the
    flag was already in the requested state, because a client that asks
    twice is usually a client whose registry disagrees with the lockfile.
    """
    if lifecycle.set_enabled(plugin_id, enabled) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Plugin '{plugin_id}' is not installed",
        )

    rediscover_now()
    return {"id": plugin_id, "enabled": enabled}


@router.post("/{plugin_id}/enable")
async def enable_plugin(plugin_id: str, request: Request) -> dict[str, Any]:
    """Activate a previously-installed plugin without re-downloading.

    The lockfile entry stays put; only the ``enabled`` flag flips. After
    the call the plugin's nodes are in the registry, its examples appear
    in ``GET /api/examples/list``, and any ``assets/`` route is mounted.

    Token-only, with no loopback gate: this activates code the user already
    consented to and this machine already has, which is a different question
    from putting new code here. It IS refused while this plugin's own install
    is running (:func:`_refuse_while_busy`) -- the flag lives in the entry the
    flow is about to rewrite.
    """
    _refuse_while_busy(request, plugin_id)
    return _set_plugin_enabled(plugin_id, True)


@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str, request: Request) -> dict[str, Any]:
    """Deactivate a plugin without uninstalling — its files stay on disk.

    The plugin's nodes are dropped from the registry, examples and assets
    are hidden, but a follow-up ``/enable`` re-activates instantly with no
    re-download (useful for large third-party packs).

    Token-only and refused while that plugin installs, for the reasons on
    :func:`enable_plugin`.
    """
    _refuse_while_busy(request, plugin_id)
    return _set_plugin_enabled(plugin_id, False)
