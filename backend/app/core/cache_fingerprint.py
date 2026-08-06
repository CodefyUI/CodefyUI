"""Content-aware fingerprints folded into ExecutionCache keys (#144, #145).

#116 (PR #142) made every file-reading node ``cacheable = False`` outright:
correct, but it means a Dataset/CSV/model-weights root re-reads from disk on
every run even when nothing on disk changed. The cache key already hashes
``params``, but a ``path`` param only names WHERE to read, not what is
there -- so two runs with an edited file underneath an unchanged path
string produce the same key, and the second run gets the first run's
answer.

The fix is not "cache everything" or "cache nothing" but "put enough of
the *content* into the key that a change on disk changes the key too".
This module is that fingerprint: cheap metadata (size + mtime for one
file; per-file identity -- not just a sum -- for a directory or list of
files) rather than hashing file CONTENT, which stays proportional to what
``os.stat`` costs, not to what READING the file costs -- the whole point
of still being able to skip the read on a hit.

For a directory or a file list, "per-file identity" is load-bearing: an
earlier version summed count / total size / latest mtime across the
files, which a rename or a move between sibling directories leaves
completely unchanged (a plain rename touches neither size nor mtime) --
the exact staleness #144 exists to prevent, reintroduced one level up.
``paths_fingerprint`` and ``directory_fingerprint`` therefore hash the
per-file ``(path, size, mtime)`` tuples, not just their sums; see the
docstring on each for the order guarantee that matters to its caller.

A node opts in by overriding ``BaseNode.cache_fingerprint(params)`` (see
``node_base.py``) to call one of these three helpers on whatever path(s)
its params name, and ``graph_engine`` folds the result into
``ExecutionCache.compute_key``. None of the three ever raises on a missing
or unreadable path -- a fingerprint answers "did the input change", not
"is the input valid"; the node's own ``execute()`` still raises the real,
specific error for that.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

#: Recursive walks stop after this many files and report ``truncated``.
#:
#: This is NOT a safety margin. A file past the cap is invisible to the
#: fingerprint -- if it changes, the key does not, and a cache HIT serves
#: the stale answer. Missing a change is exactly how staleness happens;
#: there is no direction in which truncating is safe. The cap exists only
#: as a backstop against a clearly-wrong path (an accidental scan of a
#: drive root, a symlink loop), not as a normal operating condition, so it
#: is set high enough that a real dataset should never reach it -- CIFAR-10
#: unpacked as an ImageFolder tree is ~60k files; this is more than 3x that.
#: Hitting it in ordinary use is a bug report to raise the number further,
#: not a tolerated trade-off.
_WALK_CAP = 200_000

#: Files at or below this size also get hashed, not just stat()'d. size +
#: mtime alone has a real gap: two writes that land inside the same
#: filesystem timestamp tick (plausible on Windows, whose clock-interrupt
#: granularity can be coarser than a fast back-to-back rewrite) and happen
#: to produce the same byte count are indistinguishable to (size, mtime)
#: even though the content differs. A single-file read+hash is cheap at
#: this size (single-digit milliseconds) and closes the gap exactly for
#: the shape of file this hook exists to protect -- a CSV, a small image,
#: a weights file, GraphInput's canvas image. Above the threshold (a
#: dataset archive, a large checkpoint) reading the whole thing just to
#: fingerprint it would cost close to what the read this is meant to let
#: a cache HIT skip costs, so those stay stat-only.
_HASH_MAX_BYTES = 8 * 1024 * 1024


def path_fingerprint(path: Path | str) -> dict[str, Any]:
    """(size, mtime, and -- for files at or under ``_HASH_MAX_BYTES`` -- a
    content hash) for one file.

    Never raises: a missing or unreadable path fingerprints as
    ``{"exists": False}``, which is itself a valid, stable fingerprint (and
    different from any real file's), so a path that starts/stops existing
    between runs still busts the cache correctly.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {"exists": False}
    fp: dict[str, Any] = {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    if st.st_size <= _HASH_MAX_BYTES:
        digest = _hash_file(p)
        if digest is not None:
            fp["hash"] = digest
    return fp


def _hash_file(path: Path) -> str | None:
    """BLAKE2b of *path*'s bytes, or None if it could not be read.

    Streamed in 1 MiB chunks so the peak memory cost does not scale with
    the (already size-capped) file; ``None`` rather than raising keeps this
    consistent with the rest of the module -- a fingerprint answers "did
    this change", never "is this valid".
    """
    h = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _identity_hash(entries: list[tuple[str, int, float]]) -> str:
    """BLAKE2b of the exact sequence of ``(path, size, mtime)`` triples.

    The sequence, not a sorted/deduplicated view of it -- callers that care
    about order (see :func:`paths_fingerprint`) rely on that.
    """
    payload = json.dumps(entries, separators=(",", ":"))
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def paths_fingerprint(paths: Iterable[Path | str]) -> dict[str, Any]:
    """Fingerprint over an explicit, ORDERED list of files (e.g. a glob
    result). ``count`` / ``total_size`` / ``latest_mtime`` are kept as
    cheap, human-readable summaries, but equality is decided by
    ``identity_hash``: a BLAKE2b of the per-file ``(path, size, mtime)``
    triples IN THE GIVEN ORDER.

    Order matters here on purpose and is NOT normalised (contrast
    :func:`directory_fingerprint`, which sorts): ``ImageBatchReaderNode``
    passes its already-glob-sorted read order, which is also the order it
    stacks images into a batch tensor. A rename that changes sort position
    reorders that tensor even when the file SET and every aggregate sum
    are unchanged, so the fingerprint must be sensitive to position, not
    just membership -- summing alone made a reorder invisible.

    Cheap even for thousands of entries -- pure ``stat()`` metadata, no
    content read. Entries that no longer exist are skipped rather than
    raising, same rationale as :func:`path_fingerprint`.
    """
    count = 0
    total_size = 0
    latest_mtime = 0.0
    entries: list[tuple[str, int, float]] = []
    for raw in paths:
        try:
            st = Path(raw).stat()
        except OSError:
            continue
        count += 1
        total_size += st.st_size
        latest_mtime = max(latest_mtime, st.st_mtime)
        entries.append((str(raw), st.st_size, st.st_mtime))
    return {
        "count": count,
        "total_size": total_size,
        "latest_mtime": latest_mtime,
        "identity_hash": _identity_hash(entries),
    }


def directory_fingerprint(root: Path | str) -> dict[str, Any]:
    """Fingerprint over every file under *root*, recursively.

    Same shape as :func:`paths_fingerprint` -- ``count`` / ``total_size`` /
    ``latest_mtime`` for readability, ``identity_hash`` decides equality --
    plus ``truncated``, set when ``_WALK_CAP`` cut the walk short (see its
    own docstring: this is a backstop, not a safety feature).

    The per-file ``(relative path, size, mtime)`` triples are SORTED before
    hashing, unlike :func:`paths_fingerprint`: ``os.walk``'s traversal
    order is filesystem/OS-dependent and not meaningful to any caller here
    (torchvision's dataset loaders do their own discovery and sorting), so
    leaving it unsorted would make the same tree fingerprint differently
    across machines or filesystems with nothing actually different. Paths
    are relative to *root* so moving the whole tree to a different parent
    directory -- nothing inside it changed -- does not itself look like an
    edit.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return {"exists": False}

    count = 0
    total_size = 0
    latest_mtime = 0.0
    truncated = False
    entries: list[tuple[str, int, float]] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if count >= _WALK_CAP:
                truncated = True
                break
            fp = Path(dirpath) / name
            try:
                st = fp.stat()
            except OSError:
                continue
            count += 1
            total_size += st.st_size
            latest_mtime = max(latest_mtime, st.st_mtime)
            try:
                rel = str(fp.relative_to(root_path))
            except ValueError:
                rel = str(fp)
            entries.append((rel, st.st_size, st.st_mtime))
        if truncated:
            break

    entries.sort()

    return {
        "exists": True,
        "count": count,
        "total_size": total_size,
        "latest_mtime": latest_mtime,
        "identity_hash": _identity_hash(entries),
        "truncated": truncated,
    }
