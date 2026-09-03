"""API routes for the Plugin Center: what is installed, and what could be.

    GET  /api/plugins             every INSTALLED plugin, enabled or not
    GET  /api/plugins/catalog     the same rows merged with what this build
                                  can install by name -- what the panel draws
    GET  /api/plugins/generation  the reload counter the editor polls
    POST /api/plugins/reload      re-discover nodes, presets and packs
    POST /api/plugins/inspect     read a source and say what installing it
                                  would cost -- installs nothing
    GET  /api/plugins/{id}        one plugin's manifest, nodes and README
    POST /api/plugins/{id}/enable|disable

Every fixed path is declared before ``/{plugin_id}``; see the comment above
that route for why the order is load-bearing rather than tidy.

Reads are open GETs, like every other read the editor polls. Reading a
SOURCE is not one of them: ``/inspect`` reaches out to GitHub on the
caller's word, so it takes the session token like any mutating route and is
additionally refused unless the server is bound to loopback --
``_require_local_plugin_install``, the same gate installing hangs off. That
gate exists because installing a plugin puts third-party code where this
process will import it, and that is not a thing a stranger on the LAN gets
to start. Installing itself still happens in the ``cdui plugin`` CLI, which
writes the lockfile and the files and then POSTs to ``/api/plugins/reload``.

Refusals from the install path carry a CODE rather than a sentence --
``HTTPException(status, detail={"code": ...})`` -- because the panel draws a
different control for each of them (fix the name, tick the capability,
inspect again), and choosing one by matching on prose breaks the day the
prose is translated.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
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
from ..core.plugins import lifecycle
from ..core.plugins.catalog import catalog_entries
from ..core.plugins.errors import (
    GitHubError,
    InspectBusy,
    ManifestError,
    PluginInstallError,
    SourceError,
    UnknownCatalogName,
)
from ..core.plugins.listing import (
    catalog_listing,
    frontend_entry_url,
    installed_facts,
    nodes_for_plugin,
)
from ..core.plugins.reload import rediscover_now
from ..core.plugins.service import PluginService, StoredInspection

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

#: The message ``inspect.inspect_github`` refuses a reserved id with. Matched
#: rather than caught by class because the id is not on the exception and the
#: panel has to name it; see :func:`_inspect_refusal`.
_RESERVED_ID = re.compile(r"^Plugin id '(?P<id>[^']+)' is reserved")


class InspectRequest(BaseModel):
    """What to read. One field, and ``extra="forbid"`` keeps it one.

    A source is a catalog name, ``owner/repo`` or a GitHub URL -- never a
    manifest, a sha or a capability list. Everything an install later acts
    on is read by the SERVER from that source and kept under an inspection
    id, so there is no field here for a client to smuggle a decision through.
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
    too malformed to install from. They are told apart by the message
    because the id is not on the exception and the panel has to say WHICH id
    clashed; both couplings are pinned by tests that drive the real
    ``inspect`` into them.
    """
    match = _RESERVED_ID.match(str(exc))
    if match is None:
        return _coded(400, "invalid_manifest")
    return _coded(400, "reserved_id", id=match.group("id"))


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
    body: InspectRequest, request: Request
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
    except (ManifestError, tomllib.TOMLDecodeError):
        # A manifest this build will not install, and the broken TOML that
        # never got as far as being validated. One code: both mean "the
        # thing at that source is not a plugin this can install", and
        # neither is anything the caller can fix from here.
        raise _coded(400, "invalid_manifest") from None
    except GitHubError as exc:
        raise _github_refusal(exc) from None
    except PluginInstallError as exc:
        raise _inspect_refusal(exc) from None
    return _inspection_payload(stored)


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
async def enable_plugin(plugin_id: str) -> dict[str, Any]:
    """Activate a previously-installed plugin without re-downloading.

    The lockfile entry stays put; only the ``enabled`` flag flips. After
    the call the plugin's nodes are in the registry, its examples appear
    in ``GET /api/examples/list``, and any ``assets/`` route is mounted.
    """
    return _set_plugin_enabled(plugin_id, True)


@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str) -> dict[str, Any]:
    """Deactivate a plugin without uninstalling — its files stay on disk.

    The plugin's nodes are dropped from the registry, examples and assets
    are hidden, but a follow-up ``/enable`` re-activates instantly with no
    re-download (useful for large third-party packs).
    """
    return _set_plugin_enabled(plugin_id, False)
