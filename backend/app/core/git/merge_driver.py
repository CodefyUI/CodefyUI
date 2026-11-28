"""A keyed three-way merge for the files of a graph stored as a folder.

git merges text line by line, and the files a folder graph is made of are
not text in that sense: ``edges.jsonl`` is a SET of wires kept in sorted
order, the layout file maps node ids to positions, ``index.json`` is a
document. Two branches that each add a wire whose lines sort next to each
other conflict as text, although nothing about the graph conflicts. This
merges them by key instead:

* ``edges.jsonl`` -- the key is the wire's
  ``(source, sourceHandle, target, targetHandle)``. Keys added on either
  side are kept; a key one side removed and the other left alone is removed;
  one key changed differently on both sides is a real conflict.
* ``layout/<name>.layout.json`` -- ``positions``, ``notes`` and
  ``subgraphPositions`` by node id; every other top-level key three-way.
* ``index.json`` -- objects merged key by key, recursively; a list of
  strings sorted on every side (``requires`` node types) merged as a set.

Anything it does not understand -- not JSON, a key twice, a shape it does
not expect -- and every real conflict falls back to git's own text merge
(``git merge-file``) and exits non-zero, so git reports a conflict with the
usual markers. It never guesses.

Run by git as a merge driver::

    <python> -I <this file> %O %A %B %P

The server builds that command line itself (``merge_driver_config``); the
repository decides only WHICH files use it, through ``.gitattributes``
(``merge=codefyui``). Without the driver defined -- somebody merging in a
terminal -- git falls back to its text merge: more conflicts, never a wrong
merge. Standard library only, and run with ``-I``: it imports nothing from
the repository it runs in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: The ``merge=`` attribute value the project's ``.gitattributes`` names.
DRIVER_NAME = "codefyui"

_EDGE_FIELDS = ("source", "sourceHandle", "target", "targetHandle")
_INDEX_FIRST = ("storage_version", "format_version", "name", "description")
_KEYED_LAYOUT = ("positions", "notes", "subgraphPositions")


class Unmergeable(Exception):
    """A real conflict, or input this driver does not understand."""


_MISSING = object()


# ── The three-way rule ────────────────────────────────────────────────────

def _pick(base: Any, ours: Any, theirs: Any) -> Any:
    """The one-value three-way rule: the side that changed wins; both
    changing the same way is fine; both changing differently is not."""
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    raise Unmergeable("both sides changed the same entry differently")


def _sorted_strings(value: Any) -> bool:
    return (isinstance(value, list) and all(isinstance(v, str) for v in value)
            and value == sorted(set(value)))


def merge_values(base: Any, ours: Any, theirs: Any) -> Any:
    """Three-way merge of JSON values: objects key by key, recursively;
    sorted string lists as sets; anything else by :func:`_pick`."""
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    if all(isinstance(v, dict) or v is _MISSING for v in (base, ours, theirs)) \
            and isinstance(ours, dict) and isinstance(theirs, dict):
        return merge_objects(base if isinstance(base, dict) else {}, ours, theirs)
    if all(_sorted_strings(v) or v is _MISSING for v in (base, ours, theirs)) \
            and ours is not _MISSING and theirs is not _MISSING:
        b = set(base) if base is not _MISSING else set()
        o, t = set(ours), set(theirs)
        return sorted((o & t) | (o - b) | (t - b))
    raise Unmergeable("both sides changed the same entry differently")


def merge_objects(base: dict, ours: dict, theirs: dict) -> dict:
    """Key by key. A key removed on one side and untouched on the other is
    removed. Keys keep OUR order, then theirs' new keys in their order."""
    out: dict = {}
    for key in list(ours) + [k for k in theirs if k not in ours] \
            + [k for k in base if k not in ours and k not in theirs]:
        value = merge_values(base.get(key, _MISSING), ours.get(key, _MISSING),
                             theirs.get(key, _MISSING))
        if value is not _MISSING:
            out[key] = value
    return out


# ── The three file kinds ──────────────────────────────────────────────────

def _edge_key(edge: dict) -> tuple[str, ...]:
    return tuple(str(edge.get(k) or "") for k in _EDGE_FIELDS)


def _read_edges(text: str) -> dict[tuple, tuple[dict, str]]:
    found: dict[tuple, tuple[dict, str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        edge = json.loads(line)
        if not isinstance(edge, dict):
            raise Unmergeable("an edge line is not an object")
        key = _edge_key(edge)
        if key in found:
            raise Unmergeable(f"the edge {key} is there twice")
        found[key] = (edge, line)
    return found


def merge_edges(base: str, ours: str, theirs: str) -> str:
    """``edges.jsonl``: the set of wires, keyed by their four ends, written
    back in the storage's order (sorted by that key)."""
    b, o, t = _read_edges(base), _read_edges(ours), _read_edges(theirs)
    lines: dict[tuple, str] = {}
    for key in set(b) | set(o) | set(t):
        bv = b[key][0] if key in b else _MISSING
        ov = o[key][0] if key in o else _MISSING
        tv = t[key][0] if key in t else _MISSING
        chosen = _pick(bv, ov, tv)
        if chosen is _MISSING:
            continue
        # The chosen side's own line, byte for byte.
        lines[key] = o[key][1] if key in o and o[key][0] == chosen else t[key][1]
    return "".join(lines[key] + "\n" for key in sorted(lines))


def _json_object(text: str) -> dict:
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise Unmergeable("not a JSON object")
    return doc


def _keyed_order(merged: dict, ours: dict) -> dict:
    """Sorted when our side was sorted (a folder graph's layout), else our
    order with new keys after it (a single-file graph's layout keeps its
    node order)."""
    if list(ours) == sorted(ours):
        return {k: merged[k] for k in sorted(merged)}
    return merged


def merge_layout(base: str, ours: str, theirs: str) -> str:
    b, o, t = _json_object(base), _json_object(ours), _json_object(theirs)
    out = merge_objects(b, o, t)
    for key in _KEYED_LAYOUT:
        if isinstance(out.get(key), dict) and isinstance(o.get(key), dict):
            out[key] = _keyed_order(out[key], o[key])
    return json.dumps(out, indent=2) + "\n"


def _sorted_deep(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sorted_deep(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_deep(v) for v in value]
    return value


def merge_index(base: str, ours: str, theirs: str) -> str:
    """``index.json``, written in the storage's canonical form."""
    merged = merge_objects(_json_object(base), _json_object(ours), _json_object(theirs))
    ordered = {k: _sorted_deep(merged[k]) for k in _INDEX_FIRST if k in merged}
    for key in sorted(merged):
        if key not in ordered:
            ordered[key] = _sorted_deep(merged[key])
    return json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"


def merger_for(path: str):
    """The merge for the file at repository path *path*, or None."""
    name = path.replace("\\", "/")
    if name.startswith("graphs/") and name.endswith("/edges.jsonl"):
        return merge_edges
    if name.startswith("graphs/") and name.endswith("/index.json"):
        return merge_index
    if name.startswith("layout/") and name.endswith(".layout.json"):
        return merge_layout
    return None


def merge_texts(path: str, base: str, ours: str, theirs: str) -> str:
    """The merged text, or :class:`Unmergeable`."""
    merger = merger_for(path)
    if merger is None:
        raise Unmergeable(f"no keyed merge for {path}")
    try:
        return merger(base, ours, theirs)
    except (ValueError, TypeError) as exc:
        raise Unmergeable(str(exc)) from None


# ── The git entry point ───────────────────────────────────────────────────

def _text_fallback(base: Path, ours: Path, theirs: Path) -> None:
    """git's own line merge into *ours* (with conflict markers)."""
    subprocess.run(["git", "merge-file", "-L", "ours", "-L", "base", "-L", "theirs",
                    str(ours), str(base), str(theirs)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv: list[str] | None = None) -> int:
    """``%O %A %B %P``: base, ours (the result is written here), theirs,
    and the path in the repository. 0 when merged, 1 for a conflict."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        print("usage: merge_driver.py BASE OURS THEIRS PATH", file=sys.stderr)
        return 2
    base, ours, theirs = (Path(a) for a in args[:3])
    try:
        texts = [p.read_bytes().decode("utf-8") for p in (base, ours, theirs)]
        merged = merge_texts(args[3], *texts)
    except (Unmergeable, OSError, UnicodeDecodeError) as exc:
        print(f"codefyui merge: {args[3]}: {exc}; falling back to a text merge",
              file=sys.stderr)
        _text_fallback(base, ours, theirs)
        return 1
    ours.write_bytes(merged.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
