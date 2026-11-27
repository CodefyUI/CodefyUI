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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project import (
    GraphAmbiguityError,
    _atomic_write,
    collect_graph_files,
    merge_graph,
    resolve_graph_file,
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
    "load_graph_document",
    "sanitize_graph_name",
    "snapshot_text",
]


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


@dataclass(frozen=True)
class GraphStore:
    """The saved graphs of one folder: a project's ``graphs/`` (with its
    ``layout/``), or the flat graphs directory (``layout_dir`` None)."""

    graphs_dir: Path
    layout_dir: Path | None = None

    @classmethod
    def current(cls) -> "GraphStore":
        """The store the running server reads and writes: the open project's,
        or the flat directory."""
        from ..config import settings

        if settings.PROJECT_DIR is None:
            return cls(settings.GRAPHS_DIR, None)
        return cls(settings.GRAPHS_DIR, settings.LAYOUT_DIR)

    @classmethod
    def for_project(cls, project_dir: Path) -> "GraphStore":
        """A project's store, for code that runs without the server (the CLI)."""
        return cls(project_dir / "graphs", project_dir / "layout")

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
        return resolve_graph_file(self.graphs_dir, stem, display_name=display_name)

    def existing(self, stem: str) -> Path | None:
        """The graph already occupying *stem*, or None. Never raises: when
        both project forms exist the answer is still "yes" (the canonical one),
        which is all the save guard asks."""
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
        return self._read_json(path, stem, display_name)

    def read(self, stem: str, display_name: str | None = None) -> dict:
        """The graph as the editor loads it: logic merged with its layout.

        Only the canonical project pair is merged; a legacy single file (and
        every flat-mode file) carries its own positions and loads verbatim.
        """
        path = self.resolve(stem, display_name)
        if not path.exists():
            raise GraphNotFound(stem, display_name)
        data = self._read_json(path, stem, display_name)
        if self.project and path.name.endswith(".graph.json"):
            layout = None
            layout_path = self.layout_path(stem)
            if layout_path is not None and layout_path.exists():
                try:
                    layout = json.loads(layout_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    layout = None
            data, _missing = merge_graph(data, layout)
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
        try:
            return [base for base, _path in collect_graph_files(self.graphs_dir)]
        except GraphAmbiguityError:
            if strict:
                raise
        names = {f.name[:-len(".graph.json")] if f.name.endswith(".graph.json")
                 else f.stem
                 for f in self.graphs_dir.glob("*.json")
                 if not f.name.endswith(".layout.json")}
        return sorted(names)

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

    def write(self, stem: str, payload: dict) -> Path:
        """Save the merged graph *payload* under *stem*; returns the logic
        file. Project mode writes the logic/layout pair (and removes a legacy
        single file of the same stem); flat mode writes the payload verbatim.
        """
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        if self.project:
            logic_path = self.logic_path(stem)
            write_graph_pair(logic_path, self.layout_path(stem), payload,
                             legacy_path=self.graphs_dir / f"{stem}.json")
            return logic_path
        path = self.resolve(stem)
        # `_atomic_write`, not `path.write_text`: a bare write OPENS an existing
        # directory entry, so on a case-insensitive filesystem saving "my graph"
        # over `My_Graph.json` kept the old spelling. `json.dumps` is
        # ASCII-only by default, so the bytes are what `write_text` wrote.
        _atomic_write(path, json.dumps(payload, indent=2))
        return path

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

    def delete(self, stem: str, display_name: str | None = None) -> list[Path]:
        """Remove the graph and its layout file; returns what was removed."""
        path = self.resolve(stem, display_name)
        if not path.exists():
            raise GraphNotFound(stem, display_name)
        removed = [path]
        path.unlink()
        layout_path = self.layout_path(stem)
        if layout_path is not None and layout_path.exists():
            layout_path.unlink()
            removed.append(layout_path)
        return removed


def load_graph_document(path: Path) -> dict:
    """A graph named by a path on the command line (``cdui run``,
    ``run_graph.py``): the JSON document in that file."""
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
