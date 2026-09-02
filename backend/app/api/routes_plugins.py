"""API routes for the Plugin Center: what is installed, and what could be.

    GET  /api/plugins             every INSTALLED plugin, enabled or not
    GET  /api/plugins/catalog     the same rows merged with what this build
                                  can install by name -- what the panel draws
    GET  /api/plugins/generation  the reload counter the editor polls
    POST /api/plugins/reload      re-discover nodes, presets and packs
    GET  /api/plugins/{id}        one plugin's manifest, nodes and README
    POST /api/plugins/{id}/enable|disable

Every fixed path is declared before ``/{plugin_id}``; see the comment above
that route for why the order is load-bearing rather than tidy.

Reads are open GETs, like every other read the editor polls. Installing is
not here yet: it still happens in the ``cdui plugin`` CLI, which writes the
lockfile and the files and then POSTs to ``/api/plugins/reload``. What IS
here is the gate the install routes will hang off --
``_require_local_plugin_install`` -- because installing a plugin puts
third-party code where this process will import it, and that is not a thing
a stranger on the LAN gets to start.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

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
from ..core.plugins.listing import (
    catalog_listing,
    frontend_entry_url,
    installed_facts,
    nodes_for_plugin,
)
from ..core.plugins.reload import rediscover_now

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins", tags=["plugins"])

_REMOTE_REFUSAL = (
    "Installing plugins is only allowed from the computer that runs the "
    "server. Set CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1 to override.")


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
    """Dependency for every route that installs or removes a plugin.

    Not attached to anything yet: this PR adds the read half of the Plugin
    Center. It is defined and tested here so the install routes cannot land
    without it -- a gate written in the same PR as the route it guards is a
    gate a reviewer reads as boilerplate.
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
async def plugin_catalog() -> dict[str, Any]:
    """Every plugin this build can install, and everything installed.

    The one route the Plugin Center polls, and the reason it is a GET: like
    ``GET /api/packs`` it is a read the editor draws a panel from, so it
    carries no session token. Declared before ``/{plugin_id}`` -- ``catalog``
    is a reserved plugin id precisely because a pack with that name would
    otherwise sit where this route lives, and the router, not the pack, would
    decide which one wins.

    ``active_job`` is ``None`` until the install routes land; the field is
    here now so the panel is written against the final shape.
    """
    return catalog_listing(
        load_lockfile(),
        registry=registry,
        active_job=None,
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
