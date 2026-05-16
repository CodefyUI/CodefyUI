"""API routes for inspecting installed CodefyUI plugins.

Read-only listing endpoints plus a hot-reload trigger that mirrors
``POST /api/nodes/reload``. Actual install/uninstall happens in the
``cdui plugin`` CLI (writes the lockfile + files on disk, then POSTs
to ``/api/plugins/reload``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport — same API.

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..core.node_registry import registry
from ..core.plugin_loader import (
    MANIFEST_FILENAME,
    iter_plugin_dirs,
    load_lockfile,
    rediscover_all,
)
from ..core.preset_registry import preset_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


def _read_manifest(plugin_dir: Path) -> dict[str, Any]:
    try:
        return tomllib.loads((plugin_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


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
    """List every installed plugin with active node names + lockfile metadata."""
    lockfile = load_lockfile()
    out: list[dict[str, Any]] = []
    for plugin_id, plugin_dir in iter_plugin_dirs(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, lockfile
    ):
        entry = lockfile["plugins"][plugin_id]
        manifest = _read_manifest(plugin_dir)
        plugin_meta = manifest.get("plugin", {})
        lessons_meta = manifest.get("lessons", {})
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
            "homepage": plugin_meta.get("homepage", ""),
            "chapters": lessons_meta.get("chapters", []),
            "lessons": lessons_meta.get("lessons", []),
            "nodes": _nodes_for_plugin(plugin_id),
        })
    return out


@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str) -> dict[str, Any]:
    lockfile = load_lockfile()
    if plugin_id not in lockfile.get("plugins", {}):
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not installed")

    for pid, plugin_dir in iter_plugin_dirs(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, lockfile
    ):
        if pid != plugin_id:
            continue
        manifest = _read_manifest(plugin_dir)
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
    return rediscover_all(
        registry,
        preset_registry,
        nodes_dir=settings.NODES_DIR,
        custom_nodes_dir=settings.CUSTOM_NODES_DIR,
        presets_dir=settings.PRESETS_DIR,
        builtin_root=settings.PLUGINS_BUILTIN_DIR,
        user_root=settings.PLUGINS_USER_DIR,
    )
