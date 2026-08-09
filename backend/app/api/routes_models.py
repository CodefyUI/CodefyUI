"""API routes for managing model weight files (list, upload, download, delete)."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

ALLOWED_EXTENSIONS = {".pt", ".pth", ".safetensors", ".ckpt", ".bin"}


def _safe_path(base_dir: Path, filename: str) -> Path:
    """Resolve *filename* under *base_dir* and ensure it stays within it."""
    resolved = (base_dir / filename).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return resolved


@router.get("")
async def list_model_files():
    """List all model files in the models directory."""
    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(models_dir.iterdir()):
        if f.is_file() and f.suffix in ALLOWED_EXTENSIONS:
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
            })
    return files


@router.post("/upload")
async def upload_model_file(file: UploadFile):
    """Upload a model weight file (.pt, .pth, .safetensors, .ckpt, .bin)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Use only the basename to prevent path traversal via filename
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = _safe_path(models_dir, safe_name)

    # Bounded before it is read: core.body_limit caps the multipart BODY as it
    # arrives, so this can no longer buffer an unbounded upload and then refuse
    # it (core#242). The check below measures the FILE rather than the body and
    # enforces the documented number exactly.
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    dest.write_bytes(content)

    logger.info("Uploaded model file: %s (%d bytes)", safe_name, len(content))
    return {"filename": safe_name, "size": len(content)}


@router.get("/download/{filename:path}")
async def download_model_file(filename: str):
    """Download a model weight file as an attachment.

    Supports nested paths (e.g. ``runs/exp1/model.pt``) so weights saved to
    sub-directories by ``ModelSaverNode`` can be retrieved — useful when the
    backend runs inside a container and files are otherwise hard to reach.
    """
    models_dir = settings.MODELS_DIR
    filepath = _safe_path(models_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    if filepath.suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a model file")

    logger.info("Downloading model file: %s (%d bytes)", filename, filepath.stat().st_size)
    return FileResponse(
        path=filepath,
        filename=filepath.name,
        media_type="application/octet-stream",
    )


@router.delete("/{filename}")
async def delete_model_file(filename: str):
    """Delete a model weight file."""
    models_dir = settings.MODELS_DIR
    filepath = _safe_path(models_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    if filepath.suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a model file")

    filepath.unlink()
    logger.info("Deleted model file: %s", filename)
    return {"message": f"Deleted {filename}"}
