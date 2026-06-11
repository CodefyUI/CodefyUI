"""API routes for managing custom nodes (list, enable/disable, upload, delete)."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ..config import settings
from ..core.node_base import BaseNode
from ..core.node_registry import registry
from ..core import plugin_loader
from ..core.plugin_loader import rediscover_all
from ..core.plugin_validator import PluginValidationError, validate_python_source
from ..core.preset_registry import preset_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/custom-nodes", tags=["custom-nodes"])


def _safe_path(base_dir: Path, filename: str) -> Path:
    """Resolve *filename* under *base_dir* and ensure it stays within it."""
    resolved = (base_dir / filename).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return resolved


def _scan_file(filepath: Path) -> list[str]:
    """Try to detect BaseNode subclass names in a .py file without importing."""
    node_names: list[str] = []
    try:
        content = filepath.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("NODE_NAME") and "=" in stripped:
                # Extract the string value
                val = stripped.split("=", 1)[1].strip().strip("\"'")
                if val:
                    node_names.append(val)
    except Exception:
        pass
    return node_names


@router.get("")
async def list_custom_nodes():
    """List all custom node files with their status."""
    custom_dir = settings.CUSTOM_NODES_DIR
    custom_dir.mkdir(parents=True, exist_ok=True)

    result = []
    for f in sorted(custom_dir.iterdir()):
        if f.name.startswith("__"):
            continue
        if f.suffix == ".py":
            result.append({
                "filename": f.name,
                "enabled": True,
                "nodes": _scan_file(f),
            })
        elif f.name.endswith(".py.disabled"):
            result.append({
                "filename": f.name,
                "enabled": False,
                "nodes": _scan_file(f),
            })
    return result


@router.post("/toggle")
async def toggle_custom_node(data: dict):
    """Enable or disable a custom node file by renaming .py <-> .py.disabled."""
    filename = data.get("filename", "")
    custom_dir = settings.CUSTOM_NODES_DIR
    filepath = _safe_path(custom_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if filename.endswith(".py.disabled"):
        # Enable: rename .py.disabled -> .py
        new_name = filename.replace(".py.disabled", ".py")
        new_path = _safe_path(custom_dir, new_name)
        filepath.rename(new_path)
        _reload_all()
        return {"filename": new_name, "enabled": True}
    elif filename.endswith(".py"):
        # Disable: rename .py -> .py.disabled
        new_name = filename + ".disabled"
        new_path = _safe_path(custom_dir, new_name)
        filepath.rename(new_path)
        _reload_all()
        return {"filename": new_name, "enabled": False}
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")


@router.post("/upload")
async def upload_custom_node(file: UploadFile):
    """Upload a .py file to the custom_nodes directory."""
    if not file.filename or not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py files are accepted")

    custom_dir = settings.CUSTOM_NODES_DIR
    custom_dir.mkdir(parents=True, exist_ok=True)
    dest = _safe_path(custom_dir, Path(file.filename).name)

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        validate_python_source(content, file.filename)
    except PluginValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dest.write_bytes(content)
    _reload_all()
    return {"filename": dest.name, "message": "Uploaded successfully"}


@router.delete("/{filename}")
async def delete_custom_node(filename: str):
    """Delete a custom node file."""
    custom_dir = settings.CUSTOM_NODES_DIR
    filepath = _safe_path(custom_dir, filename)

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if filepath.name.startswith("__"):
        raise HTTPException(status_code=400, detail="Cannot delete system files")

    filepath.unlink()
    _reload_all()
    return {"message": f"Deleted {filename}"}


def _reload_all():
    """Re-discover all nodes and presets after a custom node change.

    Delegates to :func:`rediscover_all` so plugins stay registered when a
    custom node is added/toggled (otherwise ``registry.clear()`` would
    silently drop them from the palette).
    """
    rediscover_all(
        registry,
        preset_registry,
        nodes_dir=settings.NODES_DIR,
        custom_nodes_dir=settings.CUSTOM_NODES_DIR,
        presets_dir=settings.PRESETS_DIR,
        builtin_root=plugin_loader.plugins_builtin_root(),
        user_root=plugin_loader.plugins_user_root(),
    )
