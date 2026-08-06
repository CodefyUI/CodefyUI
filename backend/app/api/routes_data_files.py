"""API routes for managing tabular / plain-data files (list, upload, download, delete).

Mirrors ``routes_images.py`` — kept as a separate module so each file kind
can evolve its own extension whitelist without branching inside one handler.
Backs the ``DATA_FILE`` param type used by CSVReader, so a learner picks a
file from a dropdown instead of typing a filesystem path.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".json"}


def _safe_path(base_dir: Path, filename: str) -> Path:
    """Resolve *filename* under *base_dir* and ensure it stays within it."""
    resolved = (base_dir / filename).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return resolved


@router.get("")
async def list_data_files():
    """List all data files in the data-files directory."""
    files_dir = settings.DATA_FILES_DIR
    files_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(files_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
            })
    return files


@router.post("/upload")
async def upload_data_file(file: UploadFile):
    """Upload a data file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    files_dir = settings.DATA_FILES_DIR
    files_dir.mkdir(parents=True, exist_ok=True)
    dest = _safe_path(files_dir, safe_name)

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    dest.write_bytes(content)

    logger.info("Uploaded data file: %s (%d bytes)", safe_name, len(content))
    return {"filename": safe_name, "size": len(content)}


@router.get("/download/{filename:path}")
async def download_data_file(filename: str):
    """Download a data file as an attachment."""
    files_dir = settings.DATA_FILES_DIR
    filepath = _safe_path(files_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a data file")

    logger.info("Downloading data file: %s (%d bytes)", filename, filepath.stat().st_size)
    return FileResponse(
        path=filepath,
        filename=filepath.name,
        media_type="application/octet-stream",
    )


@router.delete("/{filename}")
async def delete_data_file(filename: str):
    """Delete a data file."""
    files_dir = settings.DATA_FILES_DIR
    filepath = _safe_path(files_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a data file")

    filepath.unlink()
    logger.info("Deleted data file: %s", filename)
    return {"message": f"Deleted {filename}"}
