"""One saved graph, as the logic document, at one ref (phase 4B).

A graph is one file (``graphs/<name>.graph.json``, or the legacy
``graphs/<name>.json``) or a folder (``graphs/<name>/``). The Source Control
tab compares a graph between two refs to say what changed in it ("2 nodes
changed, 1 added, 3 wires"), and for a folder that means reading many files
at a ref without a checkout: ``ls-tree`` (or ``ls-files -s`` for the index)
names the blobs, and ONE ``cat-file --batch`` reads them all.

The answer is the logic document the run and publish paths read
(``GraphStore.read_logic``'s shape), so v1 and v2 compare the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..graph_storage import (
    GraphNotFound,
    GraphStorageError,
    GraphStore,
    implode,
    sanitize_graph_name,
)
from .errors import GitError
from .paths import validate_sha
from .runner import T_READ, run_git

#: The most one graph may weigh at a ref, all of its files together.
MAX_GRAPH_BYTES = 32 * 1024 * 1024


def _stem(name: str) -> str:
    stem = sanitize_graph_name(name)
    if not name or stem != name:
        raise GitError("invalid_path", 400, f"{name!r} is not a saved graph's name",
                       hint="use the name the Graphs panel lists (its file stem)")
    return stem


def _blobs_at(root: Path, stem: str, ref: str) -> dict[str, str]:
    """``path -> blob id`` for the graph's files at *ref* (``index``,
    ``HEAD`` or a sha). Empty when nothing of it is there."""
    specs = [f"graphs/{stem}/", f"graphs/{stem}.graph.json", f"graphs/{stem}.json"]
    if ref == "index":
        result = run_git(["ls-files", "-s", "-z", "--", *specs], cwd=root,
                         timeout=T_READ, read_only=True)
        entries = {}
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            head, _, path = record.partition(b"\t")
            _mode, blob, stage = head.decode().split(" ")
            if stage != "0":
                raise GitError("conflict", 409, f"graphs/{stem} is in a conflict",
                               hint="resolve the conflict first")
            entries[path.decode("utf-8", "surrogateescape")] = blob
        return entries
    commit = "HEAD" if ref == "HEAD" else validate_sha(ref)
    result = run_git(["ls-tree", "-r", "-z", commit, "--", *specs], cwd=root,
                     timeout=T_READ, ok_codes=(0, 128), read_only=True)
    if result.returncode != 0:
        # An unborn HEAD or a commit that is not there: nothing at that ref.
        if ref == "HEAD":
            return {}
        raise GitError("invalid_ref", 404, f"{ref} is not a commit here")
    entries = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        head, _, path = record.partition(b"\t")
        _mode, kind, blob = head.decode().split(" ")
        if kind == "blob":
            entries[path.decode("utf-8", "surrogateescape")] = blob
    return entries


def _read_blobs(root: Path, blobs: dict[str, str]) -> dict[str, bytes]:
    """Every blob in *blobs*, through one ``cat-file --batch``."""
    if not blobs:
        return {}
    order = list(blobs.items())
    sizes = run_git(["cat-file", "--batch-check"], cwd=root, timeout=T_READ,
                    input_bytes="".join(f"{b}\n" for _, b in order).encode(),
                    read_only=True)
    total = 0
    for line in sizes.out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].isdigit():
            total += int(parts[2])
    if total > MAX_GRAPH_BYTES:
        raise GitError("invalid_value", 413,
                       f"the graph is {total} bytes at this ref, over the "
                       f"{MAX_GRAPH_BYTES} this view reads")
    result = run_git(["cat-file", "--batch"], cwd=root, timeout=T_READ,
                     input_bytes="".join(f"{b}\n" for _, b in order).encode(),
                     read_only=True)
    out, pos, files = result.stdout, 0, {}
    for path, _blob in order:
        newline = out.index(b"\n", pos)
        header = out[pos:newline].split()
        if len(header) != 3:
            raise GitError("git_failed", 500, f"cat-file could not read {path}")
        size = int(header[2])
        start = newline + 1
        files[path] = out[start:start + size]
        pos = start + size + 1
    return files


def graph_at_ref(root: Path, name: str, ref: str) -> dict[str, Any]:
    """The logic document of graph *name* at *ref*: ``worktree``, ``index``,
    ``HEAD`` or a commit sha.

    Answers ``{"name", "ref", "storage": "dir"|"file", "logic", "repairs"}``;
    404 when the graph is not there at that ref.
    """
    stem = _stem(name)
    if ref == "worktree":
        store = GraphStore.for_project(root)
        try:
            kind = store.storage_of(stem)
            if kind is None:
                raise GraphNotFound(stem)
            if kind == "dir":
                logic, repairs = store.read_folder(stem)
            else:
                logic, repairs = store.read_logic(stem), []
        except GraphNotFound:
            raise GitError("not_found", 404, f"no graph {stem} in the working tree")
        except GraphStorageError as exc:
            raise GitError("invalid_value", 422, str(exc))
        return {"name": stem, "ref": ref, "storage": kind, "logic": logic,
                "repairs": [r.as_dict() for r in repairs]}

    files = _read_blobs(root, _blobs_at(root, stem, ref))
    prefix = f"graphs/{stem}/"
    folder = {p[len(prefix):]: data for p, data in files.items() if p.startswith(prefix)}
    singles = [p for p in (f"graphs/{stem}.graph.json", f"graphs/{stem}.json")
               if p in files]
    if folder and singles:
        raise GitError("conflict", 409,
                       f"graphs/{stem}/ and {singles[0]} are both at {ref}",
                       hint="keep one form of the graph")
    if folder:
        try:
            logic, repairs = implode(folder)
        except LookupError:
            raise GitError("invalid_value", 422,
                           f"graphs/{stem}/ at {ref} is from a newer CodefyUI")
        except ValueError as exc:
            raise GitError("invalid_value", 422, f"graphs/{stem}/ at {ref}: {exc}")
        return {"name": stem, "ref": ref, "storage": "dir", "logic": logic,
                "repairs": [r.as_dict() for r in repairs]}
    if singles:
        import json

        try:
            logic = json.loads(files[singles[0]].decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise GitError("invalid_value", 422, f"{singles[0]} at {ref}: {exc}")
        return {"name": stem, "ref": ref, "storage": "file", "logic": logic,
                "repairs": []}
    raise GitError("not_found", 404, f"no graph {stem} at {ref}")
