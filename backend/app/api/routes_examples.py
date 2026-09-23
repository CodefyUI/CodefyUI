import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..core import plugin_loader
from ..core.data_paths import resolve_under
from ..core.graph_engine import is_note_node
from ..core.plugin_loader import iter_plugin_dirs, load_lockfile

router = APIRouter(prefix="/api/examples", tags=["examples"])

#: The sections the gallery is grouped into, in the order the frontend
#: renders them. A ``gallery.section`` outside this set is served as
#: ``null``, which puts the example in "other" rather than inventing a
#: heading no locale has a title for.
GALLERY_SECTIONS = frozenset(
    {"quickstart", "training", "llm", "concepts", "architectures"})


def _as_list(value) -> list:
    """*value* if it is a list, an empty list otherwise.

    ``nodes`` and ``edges`` are the two keys the list route reads for their
    length rather than their contents, and a file saying ``"nodes": null``
    parses fine and then raises on iteration. Same reasoning as
    :func:`_gallery_metadata`: the cost of an unreadable field is that
    field, not the whole gallery.
    """
    return value if isinstance(value, list) else []


def _gallery_metadata(data: dict) -> dict:
    """The three gallery keys, read defensively out of *data*.

    Always the same three keys, so the frontend groups on values rather
    than on whether a key exists, and each one is read on its own: the list
    walks every installed pack's examples directory, and one pack's typo
    must cost that pack's card its heading, not everybody's gallery.
    """
    block = data.get("gallery")
    if not isinstance(block, dict):
        return {"section": None, "family": None, "order": None}
    section = block.get("section")
    family = block.get("family")
    order = block.get("order")
    return {
        # ``isinstance`` first: ``in`` on a set raises TypeError for an
        # unhashable value, and ``"section": []`` is a file somebody can write.
        "section": section if isinstance(section, str)
        and section in GALLERY_SECTIONS else None,
        "family": family if isinstance(family, str) else None,
        # ``bool`` is an ``int`` in Python and would sort as 0 or 1; a file
        # saying ``"order": true`` means nothing, so it says nothing.
        "order": order if isinstance(order, int)
        and not isinstance(order, bool) else None,
    }


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
        # Valid JSON, but not an object: a file with no ``name`` and no
        # ``nodes`` to read is not an example, and listing it would put a
        # card in the gallery that opens onto nothing.
        if not isinstance(data, dict):
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
            # Notes are annotations the canvas keeps in the node list, not
            # graph -- counting them would make the best-explained example
            # look like the most complicated one.
            "node_count": sum(
                1 for node in _as_list(data.get("nodes"))
                if isinstance(node, dict) and not is_note_node(node)
            ),
            "edge_count": len(_as_list(data.get("edges"))),
            **_gallery_metadata(data),
        })
    return out


@router.get("/list")
async def list_examples():
    results = _scan_examples(settings.EXAMPLES_DIR, source="builtin")
    for plugin_id, plugin_dir in iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), load_lockfile()
    ):
        results.extend(
            _scan_examples(
                plugin_dir / "examples",
                source=f"plugin:{plugin_id}",
                path_prefix=f"plugin:{plugin_id}",
            )
        )
    return results


def _safe_resolve_under(base: Path, rel: str) -> Path:
    """*rel* resolved under *base*, or a 400 "Invalid path".

    The rule is :func:`app.core.data_paths.resolve_under` (#483).
    """
    candidate = resolve_under(base, rel)
    if candidate is None:
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


@router.get("/load")
async def load_example(path: str = Query(..., description="Relative path to the example directory")):
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid path")

    # plugin:<id>/<rest> resolves under the plugin's examples/ root.
    if path.startswith("plugin:"):
        head, _, rest = path.partition("/")
        plugin_id = head.split(":", 1)[1]
        for pid, plugin_dir in iter_plugin_dirs(
            plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), load_lockfile()
        ):
            if pid == plugin_id:
                examples_root = plugin_dir / "examples"
                resolved = _safe_resolve_under(examples_root, f"{rest}/graph.json")
                if not resolved.exists():
                    raise HTTPException(status_code=404, detail=f"Example not found: {path}")
                return json.loads(resolved.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not installed")

    resolved = _safe_resolve_under(settings.EXAMPLES_DIR, f"{path}/graph.json")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Example not found: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))
