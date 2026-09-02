"""Serve the files a plugin ships: its frontend bundle and its ``assets/``.

Both routes resolve the plugin directory from the lockfile on EVERY request,
and that is their whole reason for existing. ``assets/`` used to be one
``StaticFiles`` mount per plugin, built once in the startup lifespan, so a
plugin installed while the server was running served 404 for every CSV and
image it shipped until somebody restarted -- which is precisely the thing a
Plugin Center in the browser exists to avoid. A route reads the lockfile
that ``POST /api/plugins/reload`` has just rewritten, so a pack is servable
the moment it is installed, and stops being servable the moment it is
uninstalled.

Enabled plugins only, on both routes: ``iter_plugin_dirs`` skips disabled
entries by default, which is what makes ``POST /api/plugins/{id}/disable``
mean what its docstring promises -- the pack's files stop being served, not
just its nodes stop being registered.

Traversal is the same discipline in both, and it is ``resolve``-then-compare
rather than a check on the string that arrived: ``..`` has several spellings
over the wire, and a symlink inside a pack is not one of them at all. Only a
real file that is still inside the directory is served -- never a directory
itself.

Windows note: ``mimetypes`` can map ``.js`` to ``text/plain`` depending on
registry state, which makes browsers reject ESM imports. Media types for the
handful of bundle extensions are pinned explicitly here; assets are anything
a pack cares to ship, so they answer from ``mimetypes``, whose Windows
entries :mod:`app.main` corrects at import.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core import plugin_loader
from ..core.plugin_loader import (
    frontend_entry_rel,
    iter_plugin_dirs,
    load_lockfile,
    read_manifest_safe,
)

router = APIRouter(prefix="/plugins", tags=["plugin-frontend"])

_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
}

# Plugin files ship under fixed names (frontend/index.js, assets/data.csv)
# and change on `cdui plugin update`, so they must be revalidated -- without
# this header browsers heuristically cache (ETag/Last-Modified only) and keep
# serving stale plugin code and stale data after an update. "no-cache" still
# allows caching but forces revalidation; FileResponse answers conditional
# requests with 304 when the file is unchanged.
_REVALIDATE = {"Cache-Control": "no-cache"}


def _enabled_plugin_dir(plugin_id: str) -> Path | None:
    """Where *plugin_id* is installed, if it is installed AND enabled.

    The lockfile is read here rather than cached anywhere, because the answer
    is allowed to change between two requests: an install writes an entry, an
    uninstall removes one, and ``disable`` flips a flag. One helper for both
    routes so they cannot come to disagree about which plugins are servable.
    """
    lockfile = load_lockfile()
    for pid, plugin_dir in iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), lockfile
    ):
        if pid == plugin_id:
            return plugin_dir
    return None


def _file_under(directory: Path, resource_path: str) -> Path | None:
    """The file *resource_path* names inside *directory*, or ``None``.

    ``None`` covers every way this can fail to be a file the caller may have
    -- the directory does not exist, the path escapes it, the file is not
    there, it is a directory -- because from outside they are all one answer:
    404. Distinguishing them in the response would tell a caller which paths
    exist above the plugin's own directory.
    """
    base = directory.resolve()
    target = (base / resource_path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        return None
    return target


@router.get("/{plugin_id}/frontend/{resource_path:path}")
async def serve_plugin_frontend(plugin_id: str, resource_path: str) -> FileResponse:
    """A file out of an enabled plugin's ``frontend/`` directory.

    Gated on the manifest declaring a ``[frontend]`` entry: a pack with no
    browser code has no bundle to serve, and answering out of a directory it
    happens to have called ``frontend`` would serve files the manifest never
    offered.
    """
    plugin_dir = _enabled_plugin_dir(plugin_id)
    if plugin_dir is not None and (
        frontend_entry_rel(read_manifest_safe(plugin_dir)) is not None
    ):
        target = _file_under(plugin_dir / "frontend", resource_path)
        if target is not None:
            return FileResponse(
                target,
                media_type=_MEDIA_TYPES.get(target.suffix.lower()),
                headers=_REVALIDATE,
            )
    raise HTTPException(status_code=404, detail="Plugin frontend resource not found")


@router.get("/{plugin_id}/assets/{resource_path:path}")
async def serve_plugin_asset(plugin_id: str, resource_path: str) -> FileResponse:
    """A file out of an enabled plugin's ``assets/`` directory.

    No manifest gate: ``assets/`` is a directory a pack either has or has
    not, and a node that ships a CSV reads it through this URL without
    declaring anything. The media type is whatever ``mimetypes`` says, and
    ``application/octet-stream`` when it says nothing -- a pack may ship a
    ``.npz`` or a ``.pt`` as readily as a ``.png``, and a browser downloading
    an unknown type is right where guessing would not be.
    """
    plugin_dir = _enabled_plugin_dir(plugin_id)
    if plugin_dir is not None:
        target = _file_under(plugin_dir / "assets", resource_path)
        if target is not None:
            return FileResponse(
                target,
                media_type=(
                    mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                ),
                headers=_REVALIDATE,
            )
    raise HTTPException(status_code=404, detail="Plugin asset not found")
