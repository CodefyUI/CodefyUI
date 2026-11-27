"""Import the flat-mode graphs into a project (phase 4A, R9).

A server that starts in flat mode because ``backend/data/graphs`` holds
graphs offers to put them into a project. This is that import: the graphs
(``adopt_graphs``, the same copy ``cdui project init --adopt`` makes) and,
unless declined, the files they reference under the flat data folder's
``models/``, ``images/`` and ``media/``. A flat graph's ``ModelLoader``
with ``weights.pt`` resolved against ``backend/data/models``; in the
project it resolves against ``<project>/assets/models``, so without the
file it would be missing.

Everything is copied, nothing is moved or rewritten: the flat folder is
left exactly as it was (acceptance 3), and a file the project already has
is never replaced. Uploads (``files/``) stay install-wide, as documented,
and are not copied.

A string anywhere in a graph counts as a reference when it names an
existing file or folder strictly inside one of those three folders, either
as an absolute path or relative to the folder a node resolves it against
(the kind folder itself, or the data folder).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .project import AdoptResult, adopt_graphs, collect_graph_files

#: The data folders whose referenced files are offered for copying.
ASSET_KINDS = ("models", "images", "media")
#: Longest string considered as a path.
_MAX_PATH_TEXT = 1024


def flat_data_root() -> Path:
    """``backend/data``: where flat mode keeps graphs and assets (the
    parent of :func:`projects.flat_graphs_dir`, the class defaults)."""
    from . import projects

    return projects.flat_graphs_dir().parent


def _flat_graphs() -> Path:
    from . import projects

    return projects.flat_graphs_dir()


def _kind_dirs() -> dict[str, Path]:
    root = flat_data_root()
    return {kind: root / kind for kind in ASSET_KINDS}


@dataclass
class AssetRef:
    #: Path relative to the flat data folder (``models/w.pt``), POSIX form.
    rel: str
    kind: str
    size: int
    files: int
    is_dir: bool

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.rel, "kind": self.kind, "size": self.size,
                "files": self.files, "is_dir": self.is_dir}


@dataclass
class ImportPlan:
    graphs: list[dict[str, Any]] = field(default_factory=list)
    assets: list[AssetRef] = field(default_factory=list)

    @property
    def assets_bytes(self) -> int:
        return sum(a.size for a in self.assets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": str(flat_data_root()),
            "graphs": self.graphs,
            "graphs_bytes": sum(g["size"] for g in self.graphs),
            "assets": [a.as_dict() for a in self.assets],
            "assets_bytes": self.assets_bytes,
        }


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if 0 < len(value) <= _MAX_PATH_TEXT and "\n" not in value:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _tree_size(path: Path) -> tuple[int, int]:
    total = count = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
                count += 1
            except OSError:
                pass
    return total, count


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _reference(text: str, root: Path, kinds: dict[str, Path]) -> tuple[str, Path] | None:
    """``(kind, resolved path)`` when *text* names an existing file or folder
    strictly inside one of the kind folders, else None."""
    raw = Path(text)
    candidates = ([raw] if raw.is_absolute()
                  else [*(d / raw for d in kinds.values()), root / raw])
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        for kind, folder in kinds.items():
            if _inside(resolved, folder.resolve()):
                return kind, resolved
    return None


def plan_import(graphs_dir: Path | None = None) -> ImportPlan:
    """What an import would copy, with sizes. Reads only."""
    import json

    source = graphs_dir if graphs_dir is not None \
        else _flat_graphs()
    root = flat_data_root().resolve()
    kinds = _kind_dirs()
    plan = ImportPlan()
    seen: dict[str, AssetRef] = {}
    for base, path in collect_graph_files(source):
        try:
            size = path.stat().st_size
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            plan.graphs.append({"name": base, "file": path.name, "size": 0,
                                "readable": False})
            continue
        plan.graphs.append({"name": base, "file": path.name, "size": size,
                            "readable": True})
        for text in _strings(payload):
            found = _reference(text, root, kinds)
            if found is None:
                continue
            kind, resolved = found
            rel = resolved.relative_to(root).as_posix()
            if rel in seen:
                continue
            if resolved.is_dir():
                bytes_, files = _tree_size(resolved)
                ref = AssetRef(rel, kind, bytes_, files, True)
            else:
                ref = AssetRef(rel, kind, resolved.stat().st_size, 1, False)
            seen[rel] = ref
    # A folder already covers the files inside it.
    folders = [r.rel + "/" for r in seen.values() if r.is_dir]
    plan.assets = sorted(
        (r for r in seen.values()
         if not any(r.rel.startswith(f) and r.rel + "/" != f for f in folders)),
        key=lambda r: r.rel)
    return plan


@dataclass
class ImportResult:
    graphs: AdoptResult
    copied: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


def _copy_file(src: Path, dest: Path, result: ImportResult, rel: str) -> None:
    if src.is_symlink():
        # Inside a referenced folder, a link may point anywhere; only what
        # is really in the flat data folder is copied.
        return
    if dest.exists() or dest.is_symlink():
        result.kept.append(rel)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    result.copied.append(rel)


def run_import(target: Path, *, include_assets: bool,
               keep_existing: bool, graphs_dir: Path | None = None) -> ImportResult:
    """Copy the flat graphs (and their files) into the project *target*."""
    source = graphs_dir if graphs_dir is not None \
        else _flat_graphs()
    plan = plan_import(source) if include_assets else ImportPlan()
    result = ImportResult(graphs=adopt_graphs(source, target,
                                              keep_existing=keep_existing))
    root = flat_data_root().resolve()
    assets = target / "assets"
    for ref in plan.assets:
        src = root / ref.rel
        if ref.is_dir:
            for dirpath, _dirs, files in os.walk(src):
                for name in sorted(files):
                    file = Path(dirpath) / name
                    rel = file.relative_to(root).as_posix()
                    _copy_file(file, assets / rel, result, rel)
        else:
            _copy_file(src, assets / ref.rel, result, ref.rel)
    return result
