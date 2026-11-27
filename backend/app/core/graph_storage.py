"""Where a saved graph lives on disk, and the one way to read or write it.

Every caller that used to build a ``graphs/*.graph.json`` path and
``json.loads`` it itself goes through a :class:`GraphStore` instead: the
editor's save, load, list, rename and delete (``routes_graph``), running a
saved graph and its contract (``routes_graph_run``), the app publish snapshot
(``routes_apps``), the agent context, ``cdui project validate``, ``cdui run``
and ``run_graph.py``. Phase 4B adds a second on-disk form behind it.

Two readers, on purpose:

``read_logic(stem)``
    What a v1 logic file holds: nodes, edges, presets, subgraphs, settings,
    without positions in project mode. Running, the contract, publishing and
    validating use this. In flat mode (no project) and for a legacy
    single-file ``<stem>.json`` it is the whole file, as it always was.
``read(stem)``
    What the editor loads: the logic merged with the layout file.

Flat mode (``layout_dir`` is None) is byte-for-byte the pre-project layout:
one ``<stem>.json`` per graph, written verbatim.
"""

from __future__ import annotations

import hashlib as _hashlib
import json
import os as _os
import re as _re
import shutil as _shutil
import unicodedata as _unicodedata
from bisect import bisect_left as _bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project import (
    GraphAmbiguityError,
    _atomic_write,
    collect_graph_files,
    merge_graph,
    resolve_graph_file,
    split_graph,
    write_graph_pair,
)

__all__ = [
    "GraphAmbiguityError",
    "GraphExists",
    "GraphNotFound",
    "GraphStorageError",
    "GraphStore",
    "GraphUnreadable",
    "LayoutOrphanExists",
    "graph_folder_of",
    "load_graph_document",
    "sanitize_graph_name",
    "snapshot_text",
]


def manifest_graph_format(project_dir: Path) -> str:
    """The manifest's ``[storage] graph_format``: ``"dir"`` or ``"file"``.

    It decides only the form of a NEW graph. A project without the key
    (every project made before phase 4B) is ``"file"``.
    """
    from .project import read_project_manifest

    storage = read_project_manifest(project_dir).get("storage")
    value = storage.get("graph_format") if isinstance(storage, dict) else None
    return "dir" if value == "dir" else "file"


def sanitize_graph_name(name: str) -> str:
    """Filesystem-safe graph name: every char outside [alnum, '-', '_'] -> '_'."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


class GraphStorageError(Exception):
    """Base of the storage layer's refusals. ``stem`` names the graph."""

    def __init__(self, stem: str, message: str) -> None:
        self.stem = stem
        super().__init__(message)


class GraphNotFound(GraphStorageError):
    def __init__(self, stem: str, display_name: str | None = None) -> None:
        super().__init__(stem, f"Graph '{display_name or stem}' not found")


class GraphUnreadable(GraphStorageError):
    """The graph is on disk but cannot be parsed into a graph document."""


class GraphExists(GraphStorageError):
    """A rename target that another graph already occupies."""


class LayoutOrphanExists(GraphStorageError):
    """A rename target whose layout file outlived its graph."""


@dataclass
class SaveResult:
    """What a save did: the graph's file or folder, the files written and
    deleted (folders only), and what had to be dropped to store it."""

    path: Path
    written: list[str] | None = None
    deleted: list[str] | None = None
    dropped: list[str] | None = None


@dataclass(frozen=True)
class GraphStore:
    """The saved graphs of one folder: a project's ``graphs/`` (with its
    ``layout/``), or the flat graphs directory (``layout_dir`` None)."""

    graphs_dir: Path
    layout_dir: Path | None = None
    #: The form a NEW graph is saved in (``"file"`` or ``"dir"``, from the
    #: manifest's ``[storage] graph_format``). An existing graph keeps its own.
    new_graph_format: str = "file"

    @classmethod
    def current(cls) -> "GraphStore":
        """The store the running server reads and writes: the open project's,
        or the flat directory."""
        from ..config import settings

        if settings.PROJECT_DIR is None:
            return cls(settings.GRAPHS_DIR, None)
        return cls(settings.GRAPHS_DIR, settings.LAYOUT_DIR,
                   new_graph_format=manifest_graph_format(settings.PROJECT_DIR))

    @classmethod
    def for_project(cls, project_dir: Path) -> "GraphStore":
        """A project's store, for code that runs without the server (the CLI)."""
        return cls(project_dir / "graphs", project_dir / "layout",
                   new_graph_format=manifest_graph_format(project_dir))

    @property
    def project(self) -> bool:
        return self.layout_dir is not None

    # ── Paths ─────────────────────────────────────────────────────────────

    def logic_path(self, stem: str) -> Path:
        """Where a save writes the graph's logic file."""
        if not self.project:
            return self.graphs_dir / f"{stem}.json"
        return self.graphs_dir / f"{stem}.graph.json"

    def layout_path(self, stem: str) -> Path | None:
        """The layout file of a project graph, else None."""
        if self.layout_dir is None:
            return None
        return self.layout_dir / f"{stem}.layout.json"

    def resolve(self, stem: str, display_name: str | None = None) -> Path:
        """The file a READ of *stem* opens, existing or not.

        Project mode applies the canonical-vs-legacy rule
        (``resolve_graph_file``) and raises GraphAmbiguityError when both
        forms are present.
        """
        if not self.project:
            return self.graphs_dir / f"{stem}.json"
        folder = self._folder_if_any(stem, display_name)
        if folder is not None:
            return folder
        return resolve_graph_file(self.graphs_dir, stem, display_name=display_name)

    def folder(self, stem: str) -> Path:
        """Where graph *stem* lives in the directory form."""
        return self.graphs_dir / stem

    def _folder_if_any(self, stem: str, display_name: str | None = None) -> Path | None:
        """The folder of *stem* when it is stored as one, else None. Raises
        FolderAmbiguityError when a single-file form of it exists as well."""
        if not self.project:
            return None
        folder = self.folder(stem)
        if not folder.is_dir():
            return None
        for single in (self.graphs_dir / f"{stem}.graph.json",
                       self.graphs_dir / f"{stem}.json"):
            if single.exists():
                raise FolderAmbiguityError(display_name or stem, folder, single)
        return folder

    def storage_of(self, stem: str) -> str | None:
        """``"dir"``, ``"file"`` or None (no such graph).

        Raises FolderAmbiguityError for a folder beside a single file. Both
        single-file forms at once is still ``"file"``: the pair save removes
        the legacy half, which is what resolves that ambiguity."""
        if self._folder_if_any(stem) is not None:
            return "dir"
        return "file" if self.existing(stem) is not None else None

    def read_folder(self, stem: str, display_name: str | None = None
                    ) -> tuple[dict, list[Repair]]:
        """The logic document of a graph stored as a folder, with repairs."""
        files = read_tree(self.folder(stem))
        try:
            return implode(files)
        except LookupError as exc:
            raise NewerStorageVersion(stem, exc.args[0] if exc.args else None) from None
        except ValueError as exc:
            raise GraphUnreadable(stem, f"Graph '{display_name or stem}' cannot "
                                  f"be read: {exc}") from None

    def existing(self, stem: str) -> Path | None:
        """The graph already occupying *stem*, or None. Never raises: when
        both project forms exist the answer is still "yes" (the canonical one),
        which is all the save guard asks."""
        if self.project and self.folder(stem).is_dir():
            return self.folder(stem)
        canonical = self.logic_path(stem)
        if canonical.exists():
            return canonical
        legacy = self.graphs_dir / f"{stem}.json"
        if self.project and legacy.exists():
            return legacy
        return None

    def exists(self, stem: str, display_name: str | None = None) -> bool:
        return self.resolve(stem, display_name).exists()

    # ── Reading ───────────────────────────────────────────────────────────

    def _read_json(self, path: Path, stem: str, display_name: str | None) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise GraphNotFound(stem, display_name) from None
        except (ValueError, OSError) as exc:
            raise GraphUnreadable(
                stem, f"Graph '{display_name or stem}' exists but is not valid "
                f"JSON: {exc}") from None
        if not isinstance(data, dict):
            raise GraphUnreadable(
                stem, f"Graph '{display_name or stem}' is not a graph document")
        return data

    def read_logic(self, stem: str, display_name: str | None = None) -> dict:
        """The graph's logic document (see the module docstring)."""
        path = self.resolve(stem, display_name)
        if not path.exists():
            raise GraphNotFound(stem, display_name)
        if path.is_dir():
            return self.read_folder(stem, display_name)[0]
        return self._read_json(path, stem, display_name)

    def read(self, stem: str, display_name: str | None = None) -> dict:
        """The graph as the editor loads it: logic merged with its layout.

        Only the canonical project pair is merged; a legacy single file (and
        every flat-mode file) carries its own positions and loads verbatim.
        """
        path = self.resolve(stem, display_name)
        if not path.exists():
            raise GraphNotFound(stem, display_name)
        repairs: list[Repair] = []
        if path.is_dir():
            data, repairs = self.read_folder(stem, display_name)
        else:
            data = self._read_json(path, stem, display_name)
        if self.project and (path.is_dir() or path.name.endswith(".graph.json")):
            layout = None
            layout_path = self.layout_path(stem)
            if layout_path is not None and layout_path.exists():
                try:
                    layout = json.loads(layout_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    layout = None
            data, _missing = merge_graph(data, layout)
        if repairs:
            # Said once by the editor ("repaired N things"), which also marks
            # the tab changed: a repair reaches disk only when the user saves.
            data["storage_repairs"] = [r.as_dict() for r in repairs]
        return data

    def read_logic_with_snapshot(self, stem: str, display_name: str | None = None
                                 ) -> tuple[dict, str]:
        """The logic document and the text an app version stores for it.

        A graph stored as one file publishes that file's exact bytes
        (Decision B of the apps design: the snapshot is what was saved). A
        graph with no single file publishes ``snapshot_text`` of its logic;
        both parse to the same document.
        """
        path = self.resolve(stem, display_name)
        if not path.exists():
            raise GraphNotFound(stem, display_name)
        if path.is_dir():
            logic = self.read_folder(stem, display_name)[0]
            return logic, snapshot_text(logic)
        try:
            text = path.read_text()
        except OSError as exc:
            raise GraphUnreadable(stem, f"Graph '{display_name or stem}' "
                                  f"cannot be read: {exc}") from None
        try:
            data = json.loads(text)
        except ValueError:
            raise GraphUnreadable(stem, f"Graph '{display_name or stem}' exists "
                                  "but is not valid JSON") from None
        if not isinstance(data, dict):
            raise GraphUnreadable(stem, f"Graph '{display_name or stem}' is not "
                                  "a graph document")
        return data, text

    def title(self, stem: str, fallback: str) -> str:
        """The name a saved graph calls itself, or *fallback* when it has
        none or cannot be read (used to name the graph a save would replace)."""
        path = self.existing(stem)
        if path is None:
            return fallback
        if path.is_dir():
            path = path / INDEX
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return fallback
        if not isinstance(data, dict):
            return fallback
        name = data.get("name")
        return name if isinstance(name, str) and name else fallback

    def stems(self, *, strict: bool = True) -> list[str]:
        """Every graph's stem, sorted. In project mode a stem present in two
        forms raises GraphAmbiguityError, unless *strict* is False: then it is
        listed once (opening it reports the conflict)."""
        if not self.graphs_dir.is_dir():
            return []
        if not self.project:
            return sorted(f.stem for f in self.graphs_dir.glob("*.json"))
        folders = self._folders()
        try:
            singles = [base for base, _path in collect_graph_files(self.graphs_dir)]
            if strict:
                for base in singles:
                    if base in folders:
                        self._folder_if_any(base)  # raises FolderAmbiguityError
        except GraphAmbiguityError:
            if strict:
                raise
            singles = [f.name[:-len(".graph.json")] if f.name.endswith(".graph.json")
                       else f.stem
                       for f in self.graphs_dir.glob("*.json")
                       if not f.name.endswith(".layout.json")]
        return sorted(set(singles) | set(folders))

    def _folders(self) -> list[str]:
        """Stems stored as folders (a sub-folder with an ``index.json``)."""
        if not self.project or not self.graphs_dir.is_dir():
            return []
        return sorted(p.name for p in self.graphs_dir.iterdir()
                      if p.is_dir() and (p / INDEX).is_file())

    def list(self) -> list[dict[str, Any]]:
        """``[{name, file, modified}]`` for the Graphs panel. A file that
        cannot be read is left out (it cannot show a name). Raises
        GraphAmbiguityError in project mode."""
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        if not self.project:
            for f in self.graphs_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    entries.append({
                        "name": data.get("name", f.stem),
                        "file": f.stem,
                        # Raw st_mtime (float epoch seconds), not an int: the
                        # Graphs panel's default order is most-recent-first,
                        # and truncating would shuffle two graphs saved within
                        # the same second into an arbitrary order.
                        "modified": f.stat().st_mtime,
                    })
                except Exception:  # noqa: BLE001 -- unreadable: not listed
                    continue
            return entries
        for stem in self._folders():
            folder = self.folder(stem)
            for single in (self.graphs_dir / f"{stem}.graph.json",
                           self.graphs_dir / f"{stem}.json"):
                if single.exists():
                    raise FolderAmbiguityError(stem, folder, single)
            try:
                index = json.loads((folder / INDEX).read_text(encoding="utf-8"))
                if not isinstance(index, dict):
                    continue
                files = [p for p in folder.rglob("*")
                         if p.is_file() and _managed(p.relative_to(folder).as_posix())]
                entry: dict[str, Any] = {
                    "name": index.get("name", stem),
                    "file": stem,
                    # The newest file in the folder: an edited node does not
                    # touch the index.
                    "modified": max(p.stat().st_mtime for p in files),
                }
                if index.get("storage_version") != STORAGE_VERSION:
                    # Listed so it is not silently missing, but marked: this
                    # build cannot open or save it.
                    entry["newer"] = True
                entries.append(entry)
            except Exception:  # noqa: BLE001 -- unreadable: not listed
                continue
        for base, f in collect_graph_files(self.graphs_dir):
            try:
                data = json.loads(f.read_text())
                entries.append({
                    "name": data.get("name", base),
                    "file": base,
                    # The LOGIC file's mtime. The layout half is rewritten by
                    # every drag, so sorting on it would rank "somebody nudged
                    # a node" above "somebody rewired the graph".
                    "modified": f.stat().st_mtime,
                })
            except Exception:  # noqa: BLE001 -- unreadable: not listed
                continue
        return entries

    # ── Writing ───────────────────────────────────────────────────────────

    def write(self, stem: str, payload: dict, *, requires: dict | None = None
              ) -> "SaveResult":
        """Save the merged graph *payload* under *stem*.

        A graph keeps the form it has on disk; a new one takes
        ``new_graph_format``. Project mode writes the logic/layout pair (and
        removes a legacy single file of the same stem) or the folder; flat
        mode writes the payload verbatim. *requires* is the folder index's
        ``requires`` block (ignored for a single file).
        """
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        if self.project:
            kind = self.storage_of(stem) or self.new_graph_format
            if kind == "dir":
                return self._write_folder(stem, payload, requires)
            logic_path = self.logic_path(stem)
            write_graph_pair(logic_path, self.layout_path(stem), payload,
                             legacy_path=self.graphs_dir / f"{stem}.json")
            return SaveResult(logic_path)
        path = self.resolve(stem)
        # `_atomic_write`, not `path.write_text`: a bare write OPENS an existing
        # directory entry, so on a case-insensitive filesystem saving "my graph"
        # over `My_Graph.json` kept the old spelling. `json.dumps` is
        # ASCII-only by default, so the bytes are what `write_text` wrote.
        _atomic_write(path, json.dumps(payload, indent=2))
        return SaveResult(path)

    def _write_folder(self, stem: str, payload: dict, requires: dict | None
                      ) -> "SaveResult":
        folder = self.folder(stem)
        logic, layout = split_graph(payload)
        logic, dropped = prepare_for_folder(logic)
        previous = read_tree(folder)
        if INDEX in previous:
            try:
                version = _parse_json(previous[INDEX]).get("storage_version")
            except (ValueError, UnicodeDecodeError, AttributeError):
                version = STORAGE_VERSION  # unreadable: this save replaces it
            if version != STORAGE_VERSION:
                raise NewerStorageVersion(stem, version)
        if requires is None:
            requires = derive_requires(logic, index_requires_safe(previous))
        try:
            files = explode(logic, previous, dir_len=len(f"graphs/{stem}/"),
                            requires=requires)
            back, repairs = implode(files)
        except (ValueError, LookupError) as exc:
            raise StorageCheckFailed(stem, f"Graph '{stem}' cannot be stored as "
                                     f"a folder: {exc}; nothing was written") from None
        # The last line of defence against losing data: the folder about to be
        # written must read back as exactly the graph being saved.
        if back != logic or repairs:
            raise StorageCheckFailed(
                stem, f"Graph '{stem}' did not read back as saved "
                f"({len(repairs)} repair(s)); nothing was written")
        written: list[str] = []
        layout_path = self.layout_path(stem)
        layout_bytes = (json.dumps(layout, indent=2) + "\n").encode("utf-8")
        try:
            on_disk = layout_path.read_bytes()
        except OSError:
            on_disk = None
        if on_disk != layout_bytes:
            _atomic_write_bytes(layout_path, layout_bytes)
            written.append(str(layout_path))
        changed, deleted = sync_tree(folder, files, previous)
        written += [str(folder / rel) for rel in changed]
        return SaveResult(folder, written=written,
                          deleted=[str(folder / rel) for rel in deleted],
                          dropped=dropped)

    def rename(self, src_stem: str, dst_stem: str, *, new_name: str,
               src_display: str | None = None,
               dst_display: str | None = None) -> None:
        """Move a graph (and its layout) to *dst_stem* and set its ``name``.

        Raises GraphAmbiguityError, GraphNotFound, GraphExists,
        LayoutOrphanExists or GraphUnreadable before anything moves.
        """
        src = self.resolve(src_stem, src_display)
        # Resolved rather than just probing the write target: this catches
        # renaming a canonical pair onto a stem that exists only in the
        # legacy form (and the reverse), which would leave both forms of one
        # stem on disk.
        dst_existing = self.resolve(dst_stem, dst_display)
        if not src.exists():
            raise GraphNotFound(src_stem, src_display)
        if dst_existing.exists() and dst_existing != src:
            raise GraphExists(dst_stem, f"Graph '{dst_display or dst_stem}' already exists")
        # A layout can outlive its graph (a checkout of `graphs/` alone, a
        # manual delete), and `Path.replace` overwrites silently, so refuse
        # here, before anything has moved.
        layout_src = self.layout_path(src_stem)
        layout_dst = self.layout_path(dst_stem)
        if layout_dst is not None and layout_dst != layout_src and layout_dst.exists():
            raise LayoutOrphanExists(
                dst_stem,
                f"A layout file for '{dst_display or dst_stem}' already exists "
                "without its graph; remove or restore it before renaming onto "
                "that name.")
        if src.is_dir():
            self._rename_folder(src_stem, dst_stem, new_name, src_display)
            if (layout_src is not None and layout_dst is not None
                    and layout_dst != layout_src and layout_src.exists()):
                layout_dst.parent.mkdir(parents=True, exist_ok=True)
                layout_src.replace(layout_dst)
            return
        raw = src.read_text()
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise GraphUnreadable(
                src_stem,
                f"Graph '{src_display or src_stem}' is not a readable graph file")
        data["name"] = new_name
        # Rename into the form the file is ALREADY in: a legacy single-file
        # graph stays legacy and upgrades to the pair on its next save.
        keeps_pair = self.project and src.name.endswith(".graph.json")
        dst = self.graphs_dir / (
            f"{dst_stem}.graph.json" if keeps_pair else f"{dst_stem}.json")
        # Move first, rewrite second: a failed move leaves the graph whole
        # under its old name; a failed rewrite leaves it whole under the new
        # one, only still titled the old one. The trailing newline is kept as
        # found so a rename is not also a last-line change in git.
        if dst != src:
            src.replace(dst)
        _atomic_write(
            dst, json.dumps(data, indent=2) + ("\n" if raw.endswith("\n") else ""))
        if (layout_src is not None and layout_dst is not None
                and layout_dst != layout_src and layout_src.exists()):
            layout_dst.parent.mkdir(parents=True, exist_ok=True)
            layout_src.replace(layout_dst)

    def _rename_folder(self, src_stem: str, dst_stem: str, new_name: str,
                       src_display: str | None) -> None:
        src = self.folder(src_stem)
        index_path = src / INDEX
        try:
            raw = index_path.read_text(encoding="utf-8")
            index = json.loads(raw)
            if not isinstance(index, dict):
                raise ValueError("not an object")
        except (OSError, ValueError):
            raise GraphUnreadable(
                src_stem,
                f"Graph '{src_display or src_stem}' is not a readable graph file") from None
        if index.get("storage_version") != STORAGE_VERSION:
            raise NewerStorageVersion(src_stem, index.get("storage_version"))
        dst = self.folder(dst_stem)
        if dst != src:
            src.replace(dst)
        index["name"] = new_name
        first = [(k, index.pop(k)) for k in ("storage_version", "format_version",
                                             "name", "description") if k in index]
        _atomic_write_bytes(dst / INDEX, _doc_bytes(first, index))

    def delete(self, stem: str, display_name: str | None = None) -> list[Path]:
        """Remove the graph and its layout file; returns what was removed."""
        path = self.resolve(stem, display_name)
        if not path.exists():
            raise GraphNotFound(stem, display_name)
        removed = [path]
        if path.is_dir():
            _shutil.rmtree(path)
        else:
            path.unlink()
        layout_path = self.layout_path(stem)
        if layout_path is not None and layout_path.exists():
            layout_path.unlink()
            removed.append(layout_path)
        return removed


def graph_folder_of(path: Path) -> Path | None:
    """The graph folder *path* names, or None: the folder itself, or its
    ``index.json`` (so pointing a command at either works)."""
    path = Path(path)
    if path.is_dir() and (path / INDEX).is_file():
        return path
    if path.name == INDEX and path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if isinstance(doc, dict) and "storage_version" in doc:
            return path.parent
    return None


def load_graph_document(path: Path, repairs: list | None = None) -> dict:
    """A graph named by a path on the command line (``cdui run``,
    ``run_graph.py``): a graph folder (or its ``index.json``) read as its
    logic document, else the JSON document in that file. What a folder read
    had to repair is appended to *repairs* when given."""
    folder = graph_folder_of(path)
    if folder is not None:
        try:
            logic, found = implode(read_tree(folder))
        except LookupError as exc:
            raise ValueError(f"{folder} was saved by a newer CodefyUI (storage "
                             f"version {exc.args[0] if exc.args else '?'})") from None
        if repairs is not None:
            repairs.extend(found)
        return logic
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a graph document")
    return data


def snapshot_text(logic: dict) -> str:
    """The text an app version stores for a published graph.

    Serialized from the logic document rather than copied from a file, so a
    graph publishes to the same snapshot whichever way it is stored; keys
    sorted so the bytes do not depend on the order a reader produced them in.
    """
    return json.dumps(logic, indent=2, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════
# The v2 directory form (phase 4B, reviews/phase-4b-design.md)
#
#   graphs/<stem>/
#     index.json                 {storage_version: 2, format_version: 1, name,
#                                 description, settings?, requires?, ...}
#     nodes/<file>.json          {id, type, o, data}   one per node
#     nodes/<file>.<param>.<ext> a long text param, extracted
#     edges.jsonl                one edge per line, with "o"; sorted by
#                                (source, sourceHandle, target, targetHandle)
#     subgraphs/<file>/          a block definition: index.json (+ interface,
#                                o), nodes/, edges.jsonl
#     presets/<file>.json        one embedded preset each, with "o"
#
# A node exists iff its file exists (no node list to conflict on); order is
# the integer "o" on each node file, edge line, block and preset, so two
# branches that each add something touch different files or different lines.
# ``explode`` / ``implode`` are pure (document <-> {relative path: bytes});
# ``sync_tree`` makes a folder equal to such a map.
# ═══════════════════════════════════════════════════════════════════════════

#: What ``index.json`` says about how the folder is laid out.
STORAGE_VERSION = 2
INDEX = "index.json"
EDGES = "edges.jsonl"
#: The gap between consecutive "o" values a fresh graph gets.
O_STEP = 1024
#: A string param longer than this (with a newline) is stored in a file.
EXTRACT_MIN_CHARS = 400
#: Paths (relative to the project) longer than this use a hash file name.
PATH_LIMIT = 240

_BODY_KEYS = ("nodes", "edges", "presets", "subgraphs")
_INDEX_RESERVED = ("storage_version", "requires")
_SAFE_CHARS = _re.compile(r"[^A-Za-z0-9._-]")
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)})
_CODE_PARAMS = ("code", "script")


@dataclass(frozen=True)
class Repair:
    """Something ``implode`` could not take as it was, and what it did.

    ``kind`` is one of: ``edge_missing_node`` (dropped), ``duplicate_edge``
    (later copy dropped), ``duplicate_id`` (later file dropped),
    ``missing_text_file`` (value became ""), ``unreadable_file`` (skipped).
    """

    kind: str
    path: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path, "detail": self.detail}


class NewerStorageVersion(GraphUnreadable):
    """The folder was written by a newer CodefyUI (``storage_version`` > 2):
    listed, but neither opened nor written, since it cannot be read right."""

    def __init__(self, stem: str, version: object) -> None:
        self.version = version
        super().__init__(
            stem, f"Graph '{stem}' was saved by a newer CodefyUI (storage "
            f"version {version}); update CodefyUI to open or save it")


class StorageCheckFailed(GraphStorageError):
    """The folder a save would write does not read back as the graph saved.
    Nothing was written."""


class FolderAmbiguityError(GraphAmbiguityError):
    """``<stem>/`` and a single-file form of the same graph both exist."""

    def __init__(self, name: str, folder: Path, single: Path) -> None:
        Exception.__init__(
            self,
            f"Graph '{name}' is ambiguous: both the folder {folder.name}/ and "
            f"the file {single.name} exist in {folder.parent}. This happens "
            "when an older CodefyUI (which does not know graph folders) saves "
            "the graph again, or when a conversion was interrupted. Keep the "
            "one you want and remove the other; `cdui project migrate` "
            "converts between the two forms.")
        self.name = name
        self.canonical = folder
        self.legacy = single


# ── Names ─────────────────────────────────────────────────────────────────

def _fold(name: str) -> str:
    return _unicodedata.normalize("NFC", name).casefold()


def _short_hash(key: str, size: int = 8) -> str:
    return _hashlib.sha256(key.encode("utf-8")).hexdigest()[:size]


def _unsafe_name(base: str) -> bool:
    return (not base or base.startswith(".") or base.endswith((".", " "))
            or base.split(".")[0].upper() in _WINDOWS_RESERVED)


def file_name_for(key: str, taken: set[str], *, dir_len: int = 0,
                  suffix_len: int = 5) -> str:
    """A file (or folder) name for the entry *key* (a node id, block id or
    preset name), unique in its folder.

    *taken* holds the casefolded names already used there (updated here).
    The name is *key* when that is a safe, portable name; otherwise the
    cleaned key plus a short hash of it. Past ``PATH_LIMIT`` (with *dir_len*
    the length of the folder's path and *suffix_len* the longest suffix the
    caller will add) it is a hash alone. The name is only a hint: readers go
    by the ``id`` inside the file.
    """
    base = _SAFE_CHARS.sub("_", key)
    if base != key or _unsafe_name(base) or _fold(base) in taken:
        # Dots go too: Windows reserves "aux.txt-1a2b" as it does "aux".
        stem = base.replace(".", "_").strip("_ ")[:60]
        base = f"{stem}-{_short_hash(key)}" if stem else _short_hash(key)
    if dir_len + len(base) + suffix_len > PATH_LIMIT:
        base = _short_hash(key, 16)
    size = 8
    while _fold(base) in taken:
        size += 4
        stem = _SAFE_CHARS.sub("_", key).replace(".", "_").strip("_ ")[:40]
        base = f"{stem}-{_short_hash(key, size)}" if stem else _short_hash(key, size)
    taken.add(_fold(base))
    return base


def _param_file_name(node_base: str, param: str) -> str:
    safe = _SAFE_CHARS.sub("_", param) or "param"
    if safe != param:
        safe = f"{safe}-{_short_hash(param)}"
    ext = "py" if (param in _CODE_PARAMS or param.endswith("_code")) else "txt"
    return f"{node_base}.{safe}.{ext}"


# ── Order ─────────────────────────────────────────────────────────────────

def _longest_increasing(values: list[tuple[int, int]]) -> set[int]:
    """Indices (the first item of each pair) of a longest strictly
    increasing run of the second items, in O(n log n)."""
    tails: list[int] = []       # smallest tail value of a run of each length
    tail_at: list[int] = []     # position in *values* of that tail
    parent: list[int] = [-1] * len(values)
    for pos, (_index, value) in enumerate(values):
        k = _bisect_left(tails, value)
        if k == len(tails):
            tails.append(value)
            tail_at.append(pos)
        else:
            tails[k] = value
            tail_at[k] = pos
        parent[pos] = tail_at[k - 1] if k > 0 else -1
    keep: set[int] = set()
    pos = tail_at[-1] if tail_at else -1
    while pos != -1:
        keep.add(values[pos][0])
        pos = parent[pos]
    return keep


def assign_orders(keys: list, previous: dict) -> list[int]:
    """The "o" of each item, in memory order (*keys*), given the values the
    items had on disk (*previous*, by key).

    Items on a longest increasing run of their previous values keep them;
    the others (new or moved) take evenly spaced values between their kept
    neighbours, ``O_STEP`` past the last one at the end. Only when a gap has
    no room left is everything renumbered (``O_STEP`` apart).
    """
    count = len(keys)
    old = [previous.get(key) for key in keys]
    known = [(i, o) for i, o in enumerate(old) if isinstance(o, int) and not isinstance(o, bool)]
    kept = _longest_increasing(known)
    out: list[int | None] = [old[i] if i in kept else None for i in range(count)]
    i = 0
    while i < count:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < count and out[j] is None:
            j += 1
        low = out[i - 1] if i > 0 else None
        high = out[j] if j < count else None
        run = j - i
        if low is None and high is None:
            values = [O_STEP * (k + 1) for k in range(run)]
        elif high is None:
            values = [low + O_STEP * (k + 1) for k in range(run)]
        elif low is None:
            values = [high - O_STEP * (run - k) for k in range(run)]
        else:
            if high - low - 1 < run:
                return [O_STEP * (k + 1) for k in range(count)]
            values = [low + (high - low) * (k + 1) // (run + 1) for k in range(run)]
        out[i:j] = values
        i = j
    return out  # type: ignore[return-value]


# ── Serialization ─────────────────────────────────────────────────────────

def _sorted_deep(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sorted_deep(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_deep(v) for v in value]
    return value


def _doc_bytes(first: list[tuple[str, Any]], rest: dict) -> bytes:
    """A JSON document: the *first* keys in that order, then *rest* sorted;
    nested objects sorted; 2-space indent, LF, trailing newline."""
    ordered = {key: _sorted_deep(value) for key, value in first}
    for key in sorted(rest):
        ordered[key] = _sorted_deep(rest[key])
    return (json.dumps(ordered, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _edge_key(edge: dict) -> tuple[str, str, str, str]:
    return tuple(str(edge.get(k) or "") for k in
                 ("source", "sourceHandle", "target", "targetHandle"))  # type: ignore[return-value]


_EDGE_FIRST = ("source", "sourceHandle", "target", "targetHandle", "type", "id")


def _edge_line(edge: dict, order: int) -> str:
    ordered = {k: _sorted_deep(edge[k]) for k in _EDGE_FIRST if k in edge}
    ordered["o"] = order
    for key in sorted(k for k in edge if k not in _EDGE_FIRST):
        ordered[key] = _sorted_deep(edge[key])
    return json.dumps(ordered, ensure_ascii=False)


def _parse_json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


# ── explode ───────────────────────────────────────────────────────────────

def _previous_entries(previous: dict[str, bytes], prefix: str, key_field: str
                      ) -> dict[str, tuple[str, int | None]]:
    """``{key: (file stem, o)}`` of the JSON files directly in *prefix*."""
    found: dict[str, tuple[str, int | None]] = {}
    for rel in sorted(previous):
        if not rel.startswith(prefix) or not rel.endswith(".json"):
            continue
        name = rel[len(prefix):]
        if "/" in name:
            continue
        try:
            doc = _parse_json(previous[rel])
        except (ValueError, UnicodeDecodeError):
            continue
        key = doc.get(key_field) if isinstance(doc, dict) else None
        if isinstance(key, str) and key not in found:
            found[key] = (name[:-len(".json")], doc.get("o"))
    return found


def _previous_edge_orders(raw: bytes | None) -> dict[tuple, int]:
    orders: dict[tuple, int] = {}
    for line in (raw or b"").decode("utf-8", "replace").splitlines():
        try:
            edge = json.loads(line)
        except ValueError:
            continue
        if isinstance(edge, dict):
            orders.setdefault(_edge_key(edge), edge.get("o"))
    return orders


def _reuse_names(keys: list[str], prior: dict[str, tuple[str, int | None]],
                 dir_len: int, suffix_len: int) -> dict[str, str]:
    """A file name per key: its previous one when still free, else a new one
    (assigned in key order, so the result does not depend on memory order)."""
    taken: set[str] = set()
    names: dict[str, str] = {}
    for key in keys:
        old = prior.get(key)
        if old is not None and _fold(old[0]) not in taken and not _unsafe_name(old[0]):
            taken.add(_fold(old[0]))
            names[key] = old[0]
    for key in sorted(k for k in keys if k not in names):
        names[key] = file_name_for(key, taken, dir_len=dir_len, suffix_len=suffix_len)
    return names


def _extract_params(node: dict, base: str, prefix: str,
                    files: dict[str, bytes]) -> dict:
    """*node* with each long text param replaced by ``{"$file": name}`` (the
    text written to *files*), and each param value that would read as a
    marker wrapped in ``{"$literal": ...}``."""
    data = node.get("data")
    params = data.get("params") if isinstance(data, dict) else None
    if not isinstance(params, dict):
        return node
    out_params = {}
    for name, value in params.items():
        if isinstance(value, str) and "\n" in value and "\r" not in value \
                and len(value) > EXTRACT_MIN_CHARS:
            file_name = _param_file_name(base, str(name))
            files[f"{prefix}nodes/{file_name}"] = value.encode("utf-8")
            out_params[name] = {"$file": file_name}
        elif isinstance(value, dict) and ("$file" in value or "$literal" in value):
            out_params[name] = {"$literal": value}
        else:
            out_params[name] = value
    return {**node, "data": {**data, "params": out_params}}


def _explode_body(files: dict[str, bytes], prefix: str, nodes: list, edges: list,
                  previous: dict[str, bytes], dir_len: int) -> None:
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{prefix or 'graph'}: two nodes share an id")
    prior = _previous_entries(previous, f"{prefix}nodes/", "id")
    names = _reuse_names(ids, prior, dir_len + len(prefix) + len("nodes/"),
                         suffix_len=len(".json") + 40)
    orders = assign_orders(ids, {k: v[1] for k, v in prior.items()})
    for node, order in zip(nodes, orders):
        if "o" in node:
            raise ValueError(f"node {node['id']!r} has a key named 'o'")
        base = names[node["id"]]
        stored = _extract_params(node, base, prefix, files)
        rest = {k: v for k, v in stored.items() if k not in ("id", "type", "data")}
        files[f"{prefix}nodes/{base}.json"] = _doc_bytes(
            [("id", stored["id"]), ("type", stored.get("type")), ("o", order),
             ("data", stored.get("data", {}))], rest)
    keys = [_edge_key(edge) for edge in edges]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{prefix or 'graph'}: two edges join the same ports")
    for edge in edges:
        if "o" in edge:
            raise ValueError(f"edge {edge.get('id')!r} has a key named 'o'")
    edge_orders = assign_orders(keys, _previous_edge_orders(previous.get(f"{prefix}{EDGES}")))
    lines = sorted(zip(keys, edge_orders, edges), key=lambda item: item[0])
    files[f"{prefix}{EDGES}"] = "".join(
        _edge_line(edge, order) + "\n" for _key, order, edge in lines).encode("utf-8")


def explode(logic: dict, previous: dict[str, bytes] | None = None, *,
            dir_len: int = 0, requires: dict | None = None) -> dict[str, bytes]:
    """The folder for the logic document *logic*: ``{relative path: bytes}``.

    *previous* is the folder as it is on disk: existing "o" values, file
    names and ``requires`` entries are kept from it. *dir_len* is the length
    of the folder's path relative to the project (for ``PATH_LIMIT``).
    *requires* replaces the index's ``requires`` when given. Raises
    ValueError for a document the format cannot hold (two nodes with one id,
    two edges joining the same ports, an entry with a key named ``o``).
    """
    previous = previous or {}
    files: dict[str, bytes] = {}
    meta = {k: v for k, v in logic.items() if k not in _BODY_KEYS}
    for key in _INDEX_RESERVED:
        if key in meta:
            raise ValueError(f"a graph document cannot have a key named {key!r}")
    if requires is None:
        try:
            old_index = _parse_json(previous[INDEX]) if INDEX in previous else {}
        except (ValueError, UnicodeDecodeError):
            old_index = {}
        requires = old_index.get("requires") if isinstance(old_index, dict) else None
    first = [("storage_version", STORAGE_VERSION),
             ("format_version", meta.pop("format_version", 1))]
    for key in ("name", "description"):
        if key in meta:
            first.append((key, meta.pop(key)))
    if requires:
        meta["requires"] = requires
    files[INDEX] = _doc_bytes(first, meta)
    _explode_body(files, "", logic.get("nodes", []), logic.get("edges", []),
                  previous, dir_len)

    presets = logic.get("presets", [])
    names_in = [str(p.get("preset_name", "")) for p in presets]
    if len(set(names_in)) != len(names_in):
        raise ValueError("two embedded presets share a name")
    prior = _previous_entries(previous, "presets/", "preset_name")
    names = _reuse_names(names_in, prior, dir_len + len("presets/"), len(".json"))
    orders = assign_orders(names_in, {k: v[1] for k, v in prior.items()})
    for preset, key, order in zip(presets, names_in, orders):
        if "o" in preset:
            raise ValueError(f"preset {key!r} has a key named 'o'")
        files[f"presets/{names[key]}.json"] = _doc_bytes(
            [("preset_name", preset.get("preset_name")), ("o", order)],
            {k: v for k, v in preset.items() if k != "preset_name"})

    subgraphs = logic.get("subgraphs", [])
    ids = [str(d.get("id", "")) for d in subgraphs]
    if len(set(ids)) != len(ids):
        raise ValueError("two block definitions share an id")
    prior = {}
    for rel in sorted(previous):
        parts = rel.split("/")
        if len(parts) == 3 and parts[0] == "subgraphs" and parts[2] == INDEX:
            try:
                doc = _parse_json(previous[rel])
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(doc, dict) and isinstance(doc.get("id"), str):
                prior.setdefault(doc["id"], (parts[1], doc.get("o")))
    names = _reuse_names(ids, prior, dir_len + len("subgraphs/"), 60)
    orders = assign_orders(ids, {k: v[1] for k, v in prior.items()})
    for definition, key, order in zip(subgraphs, ids, orders):
        if "o" in definition:
            raise ValueError(f"block {key!r} has a key named 'o'")
        prefix = f"subgraphs/{names[key]}/"
        head = {k: v for k, v in definition.items() if k not in ("nodes", "edges")}
        files[f"{prefix}{INDEX}"] = _doc_bytes(
            [("id", definition.get("id")), ("o", order)],
            {k: v for k, v in head.items() if k != "id"})
        _explode_body(files, prefix, definition.get("nodes", []),
                      definition.get("edges", []), previous, dir_len)
    return files


# ── implode ───────────────────────────────────────────────────────────────

def _order_key(o: Any) -> tuple[int, float]:
    if isinstance(o, int) and not isinstance(o, bool):
        return (0, o)
    return (1, 0)  # no "o" (a hand-written file): after every ordered one


def _restore_params(node: dict, prefix: str, files: dict[str, bytes],
                    repairs: list[Repair], path: str) -> dict:
    data = node.get("data")
    params = data.get("params") if isinstance(data, dict) else None
    if not isinstance(params, dict):
        return node
    out = {}
    for name, value in params.items():
        if isinstance(value, dict) and len(value) == 1 and "$file" in value:
            rel = f"{prefix}nodes/{value['$file']}"
            raw = files.get(rel)
            if raw is None:
                repairs.append(Repair("missing_text_file", rel,
                                      f"param {name!r} of node {node.get('id')!r}: "
                                      "its text file is missing; the value is now empty"))
                out[name] = ""
            else:
                out[name] = raw.decode("utf-8", "replace")
        elif isinstance(value, dict) and len(value) == 1 and "$literal" in value:
            out[name] = value["$literal"]
        else:
            out[name] = value
    return {**node, "data": {**data, "params": out}}


def _implode_body(files: dict[str, bytes], prefix: str, repairs: list[Repair]
                  ) -> tuple[list[dict], list[dict]]:
    found: list[tuple[tuple, str, dict]] = []
    seen: dict[str, str] = {}
    node_dir = f"{prefix}nodes/"
    for rel in sorted(files):
        if not rel.startswith(node_dir) or not rel.endswith(".json") \
                or "/" in rel[len(node_dir):]:
            continue
        try:
            doc = _parse_json(files[rel])
            if not isinstance(doc, dict) or not isinstance(doc.get("id"), str):
                raise ValueError("not a node file")
        except (ValueError, UnicodeDecodeError) as exc:
            repairs.append(Repair("unreadable_file", rel, f"skipped: {exc}"))
            continue
        if doc["id"] in seen:
            repairs.append(Repair("duplicate_id", rel,
                                  f"node {doc['id']!r} is also in {seen[doc['id']]}; "
                                  "this copy was left out"))
            continue
        seen[doc["id"]] = rel
        order = doc.pop("o", None)
        found.append(((_order_key(order), doc["id"]), rel, doc))
    found.sort(key=lambda item: item[0])
    nodes = [_restore_params(doc, prefix, files, repairs, rel) for _k, rel, doc in found]
    ids = {node["id"] for node in nodes}

    edges_found: list[tuple[tuple, dict]] = []
    keys: set[tuple] = set()
    rel = f"{prefix}{EDGES}"
    for number, line in enumerate(files.get(rel, b"").decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            edge = json.loads(line)
            if not isinstance(edge, dict):
                raise ValueError("not an edge")
        except ValueError as exc:
            repairs.append(Repair("unreadable_file", f"{rel}:{number}", f"skipped: {exc}"))
            continue
        key = _edge_key(edge)
        if key in keys:
            repairs.append(Repair("duplicate_edge", f"{rel}:{number}",
                                  f"a second edge from {key[0]}.{key[1]} to "
                                  f"{key[2]}.{key[3]} was left out"))
            continue
        if edge.get("source") not in ids or edge.get("target") not in ids:
            repairs.append(Repair("edge_missing_node", f"{rel}:{number}",
                                  f"the edge from {edge.get('source')!r} to "
                                  f"{edge.get('target')!r} names a node that is "
                                  "not there; it was left out"))
            continue
        keys.add(key)
        order = edge.pop("o", None)
        edges_found.append(((_order_key(order), key), edge))
    edges_found.sort(key=lambda item: item[0])
    return nodes, [edge for _k, edge in edges_found]


def _read_index(files: dict[str, bytes], rel: str) -> dict:
    try:
        doc = _parse_json(files[rel])
    except KeyError:
        raise ValueError(f"{rel} is missing") from None
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{rel} cannot be read: {exc}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{rel} is not a JSON object")
    return doc


def implode(files: dict[str, bytes]) -> tuple[dict, list[Repair]]:
    """The logic document a folder holds, and what had to be repaired.

    Raises ValueError when ``index.json`` is missing or unreadable (the graph
    cannot be read: an error, not a repair) and NewerStorageVersion-like
    ``LookupError`` when ``storage_version`` is not one this build knows.
    """
    index = _read_index(files, INDEX)
    version = index.get("storage_version")
    if version != STORAGE_VERSION:
        raise LookupError(version)
    repairs: list[Repair] = []
    logic: dict[str, Any] = {"format_version": index.get("format_version", 1)}
    for key in ("name", "description"):
        if key in index:
            logic[key] = index[key]
    nodes, edges = _implode_body(files, "", repairs)
    logic["nodes"] = nodes
    logic["edges"] = edges

    presets: list[tuple[tuple, dict]] = []
    seen: set[str] = set()
    for rel in sorted(files):
        parts = rel.split("/")
        if len(parts) != 2 or parts[0] != "presets" or not rel.endswith(".json"):
            continue
        try:
            doc = _parse_json(files[rel])
            if not isinstance(doc, dict):
                raise ValueError("not a preset")
        except (ValueError, UnicodeDecodeError) as exc:
            repairs.append(Repair("unreadable_file", rel, f"skipped: {exc}"))
            continue
        name = str(doc.get("preset_name", ""))
        if name in seen:
            repairs.append(Repair("duplicate_id", rel,
                                  f"preset {name!r} appears twice; this copy was left out"))
            continue
        seen.add(name)
        order = doc.pop("o", None)
        presets.append(((_order_key(order), name), doc))
    presets.sort(key=lambda item: item[0])
    logic["presets"] = [doc for _k, doc in presets]

    blocks: list[tuple[tuple, dict]] = []
    seen = set()
    folders = sorted({rel.split("/")[1] for rel in files
                      if rel.startswith("subgraphs/") and rel.count("/") >= 2})
    for folder in folders:
        prefix = f"subgraphs/{folder}/"
        try:
            head = _read_index(files, f"{prefix}{INDEX}")
            if not isinstance(head.get("id"), str):
                raise ValueError("the block has no id")
        except ValueError as exc:
            repairs.append(Repair("unreadable_file", f"{prefix}{INDEX}", f"block skipped: {exc}"))
            continue
        if head["id"] in seen:
            repairs.append(Repair("duplicate_id", prefix,
                                  f"block {head['id']!r} appears twice; this copy was left out"))
            continue
        seen.add(head["id"])
        order = head.pop("o", None)
        block_nodes, block_edges = _implode_body(files, prefix, repairs)
        definition = {"id": head.pop("id")}
        definition.update({k: v for k, v in head.items()})
        definition["nodes"] = block_nodes
        definition["edges"] = block_edges
        blocks.append(((_order_key(order), definition["id"]), definition))
    blocks.sort(key=lambda item: item[0])
    logic["subgraphs"] = [d for _k, d in blocks]

    for key, value in index.items():
        if key not in logic and key not in _INDEX_RESERVED:
            logic[key] = value
    return logic, repairs


def index_requires(files: dict[str, bytes]) -> dict | None:
    """The ``requires`` block of a folder's index, or None."""
    try:
        value = _read_index(files, INDEX).get("requires")
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


# ── The folder on disk ────────────────────────────────────────────────────

def _managed(rel: str) -> bool:
    """Whether *rel* is a file the format owns (and a save may delete)."""
    parts = rel.split("/")
    if parts[0] in (INDEX, EDGES) and len(parts) == 1:
        return True
    if parts[0] in ("nodes", "presets") and len(parts) == 2:
        return True
    if parts[0] == "subgraphs" and len(parts) >= 3:
        inner = parts[2:]
        return (len(inner) == 1 and inner[0] in (INDEX, EDGES)) or \
            (len(inner) == 2 and inner[0] == "nodes")
    return False


def read_tree(folder: Path) -> dict[str, bytes]:
    """The format's files under *folder*: ``{relative posix path: bytes}``."""
    files: dict[str, bytes] = {}
    if not folder.is_dir():
        return files
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or ".tmp-" in path.name:
            continue
        rel = path.relative_to(folder).as_posix()
        if _managed(rel):
            files[rel] = path.read_bytes()
    return files


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{_os.getpid()}")
    try:
        tmp.write_bytes(data)
        _os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def sync_tree(folder: Path, files: dict[str, bytes],
              previous: dict[str, bytes]) -> tuple[list[str], list[str]]:
    """Make *folder* hold exactly *files* (of the files the format owns).

    Order: new and changed files first, then each ``edges.jsonl``, then each
    ``index.json`` (the graph's own last), and only then the files that left
    the graph are deleted. Returns ``(written, deleted)``.
    """
    changed = [rel for rel, data in files.items() if previous.get(rel) != data]

    def rank(rel: str) -> tuple[int, str]:
        name = rel.rsplit("/", 1)[-1]
        if rel == INDEX:
            return (3, rel)
        if name == INDEX:
            return (2, rel)
        if name == EDGES:
            return (1, rel)
        return (0, rel)

    changed.sort(key=rank)
    for rel in changed:
        _atomic_write_bytes(folder / rel, files[rel])
    kept = {_fold(rel) for rel in files}
    deleted = []
    for rel in sorted(previous):
        # On a case-insensitive disk a file renamed only in case is the same
        # entry as the one just written: deleting the old spelling would
        # delete the new file.
        if rel in files or _fold(rel) in kept or not _managed(rel):
            continue
        try:
            (folder / rel).unlink()
            deleted.append(rel)
        except FileNotFoundError:
            pass
    for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and path != folder and not any(path.iterdir()):
            path.rmdir()
    return changed, deleted


def prepare_for_folder(logic: dict) -> tuple[dict, list[str]]:
    """*logic* made storable as a folder, and what had to go.

    A folder cannot hold two nodes with one id or two edges joining the same
    ports, and an edge to a node that is not there would be dropped on the
    next read. Such leftovers are removed here, before the round-trip check,
    so a graph that has one can still be saved; the caller reports them.
    """
    notes: list[str] = []

    def clean(scope: str, nodes: list, edges: list) -> tuple[list, list]:
        kept_nodes, ids = [], set()
        for node in nodes:
            if node.get("id") in ids:
                notes.append(f"{scope}: a second node {node.get('id')!r} was dropped")
                continue
            ids.add(node.get("id"))
            kept_nodes.append(node)
        kept_edges, keys = [], set()
        for edge in edges:
            key = _edge_key(edge)
            if edge.get("source") not in ids or edge.get("target") not in ids:
                notes.append(f"{scope}: an edge from {edge.get('source')!r} to "
                             f"{edge.get('target')!r} named a missing node and was dropped")
                continue
            if key in keys:
                notes.append(f"{scope}: a second edge from {key[0]}.{key[1]} to "
                             f"{key[2]}.{key[3]} was dropped")
                continue
            keys.add(key)
            kept_edges.append(edge)
        return kept_nodes, kept_edges

    out = dict(logic)
    out["nodes"], out["edges"] = clean("graph", logic.get("nodes", []), logic.get("edges", []))
    blocks = []
    for definition in logic.get("subgraphs", []):
        d = dict(definition)
        d["nodes"], d["edges"] = clean(f"block {d.get('id')!r}",
                                       d.get("nodes", []), d.get("edges", []))
        blocks.append(d)
    out["subgraphs"] = blocks
    return out, notes


# ── requires ──────────────────────────────────────────────────────────────

#: The first CodefyUI release that reads storage version 2.
REQUIRES_CODEFYUI = ">=3.4"


def _node_types(logic: dict) -> set[str]:
    """Every node type the graph uses: top level, inside blocks, in presets."""
    found: set[str] = set()

    def walk(nodes) -> None:
        for node in nodes or ():
            if isinstance(node, dict) and isinstance(node.get("type"), str):
                found.add(node["type"])

    walk(logic.get("nodes"))
    for group in ("subgraphs", "presets"):
        for entry in logic.get(group) or ():
            if isinstance(entry, dict):
                walk(entry.get("nodes"))
    return found


def _registry_provider(node_type: str) -> str | None:
    from .node_registry import provider_for, registry

    cls = registry.get(node_type)
    return None if cls is None else provider_for(node_type, cls)


def index_requires_safe(files: dict[str, bytes]) -> dict | None:
    try:
        return index_requires(files)
    except (ValueError, LookupError):
        return None


def derive_requires(logic: dict, previous: dict | None,
                    provider_of=_registry_provider) -> dict:
    """The index's ``requires``: which plugin and custom ("mods") node types
    the graph uses, with no versions or hashes (pinning is the manifest's).

    A type this machine does not know keeps the entry the previous save
    recorded: a missing plugin must not erase the requirement for it.
    """
    previous = previous if isinstance(previous, dict) else {}
    old_owner: dict[str, tuple[str, str | None]] = {}
    for plugin_id, entry in (previous.get("plugins") or {}).items():
        for t in (entry or {}).get("nodes") or ():
            old_owner[t] = ("plugin", plugin_id)
    for t in (previous.get("mods") or {}).get("nodes") or ():
        old_owner[t] = ("mods", None)
    plugins: dict[str, set[str]] = {}
    mods: set[str] = set()
    for node_type in _node_types(logic):
        provider = provider_of(node_type)
        if provider is None:
            owner = old_owner.get(node_type)
            if owner is None:
                continue
            kind, plugin_id = owner
        elif provider.startswith("plugin:"):
            kind, plugin_id = "plugin", provider.split(":", 1)[1]
        elif provider == "custom":
            kind, plugin_id = "mods", None
        else:
            continue
        if kind == "plugin":
            plugins.setdefault(plugin_id, set()).add(node_type)
        else:
            mods.add(node_type)
    out: dict = {"codefyui": REQUIRES_CODEFYUI}
    if plugins:
        out["plugins"] = {pid: {"nodes": sorted(types)}
                          for pid, types in sorted(plugins.items())}
    if mods:
        out["mods"] = {"nodes": sorted(mods)}
    return out
