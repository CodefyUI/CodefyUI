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
This module is that fingerprint: cheap metadata (size + mtime, and for a
directory, count / total size / latest mtime) rather than a content hash --
proportional to what ``os.stat`` costs, not to what READING the file costs,
which is the whole point of still being able to skip the read on a hit.

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
import os
from pathlib import Path
from typing import Any, Iterable

#: Recursive walks stop after this many files and report ``truncated``. A
#: cap keeps one enormous tree from making a cache HIT cost more than the
#: miss it exists to avoid. A truncated fingerprint can only ever miss
#: noticing an edit past the cap -- it never reports a change that did not
#: happen -- so it stays on the safe side of "never serve stale data".
_WALK_CAP = 20_000

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


def paths_fingerprint(paths: Iterable[Path | str]) -> dict[str, Any]:
    """Aggregate fingerprint over an explicit list of files (e.g. a glob
    result): count, total size, latest mtime. Cheap even for thousands of
    entries -- pure ``stat()`` metadata, no content read. Entries that no
    longer exist are skipped rather than raising, same rationale as
    :func:`path_fingerprint`.
    """
    count = 0
    total_size = 0
    latest_mtime = 0.0
    for raw in paths:
        try:
            st = Path(raw).stat()
        except OSError:
            continue
        count += 1
        total_size += st.st_size
        latest_mtime = max(latest_mtime, st.st_mtime)
    return {"count": count, "total_size": total_size, "latest_mtime": latest_mtime}


def directory_fingerprint(root: Path | str) -> dict[str, Any]:
    """Aggregate fingerprint over every file under *root*, recursively.

    Same shape as :func:`paths_fingerprint` plus ``truncated``. Bounded by
    ``_WALK_CAP`` so one enormous tree cannot make a cache hit cost more
    than the miss it is trying to avoid; because a truncated walk can only
    ever UNDER-report change (miss an edit past the cap) and never
    over-report one, it stays on the safe side of "never serve stale data".
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return {"exists": False}

    count = 0
    total_size = 0
    latest_mtime = 0.0
    truncated = False
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if count >= _WALK_CAP:
                truncated = True
                break
            try:
                st = (Path(dirpath) / name).stat()
            except OSError:
                continue
            count += 1
            total_size += st.st_size
            latest_mtime = max(latest_mtime, st.st_mtime)
        if truncated:
            break

    return {
        "exists": True,
        "count": count,
        "total_size": total_size,
        "latest_mtime": latest_mtime,
        "truncated": truncated,
    }
