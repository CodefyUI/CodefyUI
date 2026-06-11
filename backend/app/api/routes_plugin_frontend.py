"""Serve plugin-shipped frontend bundles.

Unlike the ``assets/`` StaticFiles mounts (created once at startup), this
route resolves the plugin directory on every request from the lockfile, so
a plugin installed while the server is running is servable immediately
after ``POST /api/plugins/reload`` -- no restart, no remount.

Windows note: ``mimetypes`` can map ``.js`` to ``text/plain`` depending on
registry state, which makes browsers reject ESM imports. Media types for
the handful of bundle extensions are pinned explicitly.
"""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core import plugin_loader
from ..core.plugin_loader import (
    MANIFEST_FILENAME,
    frontend_entry_rel,
    iter_plugin_dirs,
    load_lockfile,
)

router = APIRouter(prefix="/plugins", tags=["plugin-frontend"])

_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
}


def _read_manifest(plugin_dir) -> dict[str, Any]:
    try:
        return tomllib.loads(
            (plugin_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, tomllib.TOMLDecodeError):
        return {}


@router.get("/{plugin_id}/frontend/{resource_path:path}")
async def serve_plugin_frontend(plugin_id: str, resource_path: str) -> FileResponse:
    lockfile = load_lockfile()
    for pid, plugin_dir in iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), lockfile
    ):
        if pid != plugin_id:
            continue
        if frontend_entry_rel(_read_manifest(plugin_dir)) is None:
            break
        base = (plugin_dir / "frontend").resolve()
        target = (base / resource_path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            break
        return FileResponse(
            target, media_type=_MEDIA_TYPES.get(target.suffix.lower())
        )
    raise HTTPException(status_code=404, detail="Plugin frontend resource not found")
