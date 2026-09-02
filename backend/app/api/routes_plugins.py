"""API routes for inspecting installed CodefyUI plugins.

Read-only listing endpoints plus a hot-reload trigger that mirrors
``POST /api/nodes/reload``. Actual install/uninstall happens in the
``cdui plugin`` CLI (writes the lockfile + files on disk, then POSTs
to ``/api/plugins/reload``).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ..core.node_registry import registry
from ..core import plugin_loader
from ..core.plugin_loader import (
    frontend_entry_rel,
    is_enabled,
    iter_plugin_dirs,
    load_lockfile,
)
from ..core.plugins import lifecycle
from ..core.plugins.reload import rediscover_now

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


def _provider_token(plugin_id: str) -> str:
    """The ``cdui_plugins.<X>`` slot, kebab → snake."""
    return plugin_id.replace("-", "_")


def _nodes_for_plugin(plugin_id: str) -> list[str]:
    token = _provider_token(plugin_id)
    prefix = f"cdui_plugins.{token}."
    return sorted(
        cls.NODE_NAME
        for cls in registry.nodes.values()
        if (cls.__module__ or "").startswith(prefix)
    )


@router.get("")
async def list_plugins() -> list[dict[str, Any]]:
    """List every installed plugin (enabled + disabled) with metadata.

    ``include_disabled=True`` so the frontend can render disabled rows
    greyed-out without an extra round-trip. Each entry carries an
    explicit ``enabled`` field; nodes list is empty for disabled plugins
    because they are not in the registry.
    """
    lockfile = load_lockfile()
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
        entry_rel = frontend_entry_rel(manifest)
        frontend_entry = None
        if enabled and entry_rel and (plugin_dir / entry_rel).is_file():
            frontend_entry = f"/plugins/{plugin_id}/{entry_rel}"
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
            "nodes": _nodes_for_plugin(plugin_id),
            "frontend_entry": frontend_entry,
        })
    return out


@router.get("/generation")
async def plugins_generation() -> dict[str, int]:
    """Monotonic counter bumped on every reload (plugin/node/enable-disable).

    The editor polls this in dev mode (when a linked plugin is present) to learn
    when to re-activate plugin frontends without a manual refresh. Declared
    before ``/{plugin_id}`` so it isn't swallowed by the dynamic route; a GET, so
    it needs no session token.
    """
    return {"generation": plugin_loader.reload_generation()}


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
            "nodes": _nodes_for_plugin(plugin_id),
            "readme": readme,
        }

    raise HTTPException(
        status_code=404,
        detail=f"Plugin '{plugin_id}' is in the lockfile but its files are missing",
    )


@router.post("/reload")
async def reload_plugins() -> dict[str, int]:
    """Clear and re-discover everything (builtin + custom + plugins + presets)."""
    return rediscover_now()


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
