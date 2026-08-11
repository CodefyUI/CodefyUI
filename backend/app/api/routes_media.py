"""Serve run-produced media inline (#310).

The download routes (``routes_images``, ``routes_models``) force
``application/octet-stream`` because their job is handing the user a file.
This route exists for the opposite job: letting the editor PLAY what a run
produced — a ``<video>`` or ``<img>`` element pointed at ``/api/media/...``
needs a real ``Content-Type``, and (for mp4 seeking) the Range support
``FileResponse`` provides.

Read-only by design: files appear here exclusively through nodes writing
under ``settings.MEDIA_DIR`` (VideoWrite), so there is no upload and no
delete — the auth model stays "GETs are unauthenticated reads", same as
every other download route.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])

#: Extension -> Content-Type. An allowlist rather than ``mimetypes.guess``:
#: only kinds a browser can render inline belong here, and a file written
#: with any other suffix stays unreachable through this route.
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _safe_path(base_dir: Path, filename: str) -> Path:
    """Resolve *filename* under *base_dir* and ensure it stays within it."""
    resolved = (base_dir / filename).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return resolved


@router.get("")
async def list_media_files():
    """List playable files under the media directory (recursive)."""
    media_dir = settings.MEDIA_DIR
    media_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(media_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in _MEDIA_TYPES:
            files.append({
                "filename": f.relative_to(media_dir).as_posix(),
                "size": f.stat().st_size,
            })
    return files


@router.get("/{filename:path}")
async def serve_media_file(filename: str):
    """Serve one media file inline, with its real Content-Type."""
    media_dir = settings.MEDIA_DIR
    filepath = _safe_path(media_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    media_type = _MEDIA_TYPES.get(filepath.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=400, detail="Not a playable media file")

    return FileResponse(path=filepath, media_type=media_type)
