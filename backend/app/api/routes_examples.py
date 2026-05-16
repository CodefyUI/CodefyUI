import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..core.plugin_loader import iter_plugin_dirs, load_lockfile

router = APIRouter(prefix="/api/examples", tags=["examples"])


def _scan_examples(base: Path, source: str, path_prefix: str = "") -> list[dict]:
    """Walk *base* for ``graph.json`` files. Each entry is tagged with *source*.

    ``path_prefix`` is prepended to the returned ``path`` so the loader can
    distinguish ``Classical/Foo`` (built-in) from ``plugin:c2/Classical/Foo``
    (plugin-shipped).
    """
    out: list[dict] = []
    if not base.exists():
        return out
    for graph_file in sorted(base.rglob("graph.json")):
        try:
            data = json.loads(graph_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel = graph_file.parent.relative_to(base)
        parts = rel.parts
        category = parts[0] if parts else "Other"
        path = rel.as_posix()
        if path_prefix:
            path = f"{path_prefix}/{path}"
        out.append({
            "name": data.get("name", rel.name),
            "description": data.get("description", ""),
            "category": category,
            "path": path,
            "source": source,
            "node_count": len(data.get("nodes", [])),
            "edge_count": len(data.get("edges", [])),
        })
    return out


@router.get("/list")
async def list_examples():
    results = _scan_examples(settings.EXAMPLES_DIR, source="builtin")
    for plugin_id, plugin_dir in iter_plugin_dirs(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, load_lockfile()
    ):
        results.extend(
            _scan_examples(
                plugin_dir / "examples",
                source=f"plugin:{plugin_id}",
                path_prefix=f"plugin:{plugin_id}",
            )
        )
    return results


@router.get("/load")
async def load_example(path: str = Query(..., description="Relative path to the example directory")):
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")

    # plugin:<id>/<rest> resolves under the plugin's examples/ root.
    if path.startswith("plugin:"):
        head, _, rest = path.partition("/")
        plugin_id = head.split(":", 1)[1]
        for pid, plugin_dir in iter_plugin_dirs(
            settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, load_lockfile()
        ):
            if pid == plugin_id:
                examples_root = (plugin_dir / "examples").resolve()
                resolved = (examples_root / rest / "graph.json").resolve()
                if not str(resolved).startswith(str(examples_root)):
                    raise HTTPException(status_code=400, detail="Invalid path")
                if not resolved.exists():
                    raise HTTPException(status_code=404, detail=f"Example not found: {path}")
                return json.loads(resolved.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not installed")

    resolved = (settings.EXAMPLES_DIR / path / "graph.json").resolve()
    if not str(resolved).startswith(str(settings.EXAMPLES_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Example not found: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))
