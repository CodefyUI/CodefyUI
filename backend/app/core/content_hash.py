"""Content identity of files and directories, for run provenance (#344).

Two answers, kept apart on purpose because they promise different things:

- :func:`stat_identity` is the cheap one: each file's relative path, size
  and modification time, hashed together. It never reads a byte of
  content. Two trees with the same stat identity are very probably the same
  files, but an edit that keeps the size and lands within the same
  timestamp tick, or a copy that resets every mtime, makes it lie in both
  directions. Its ``method`` is ``"stat"``.
- :func:`content_identity` is the real one: the sha256 of every file's
  bytes, combined over the sorted relative paths. Its ``method`` is
  ``"sha256"``. Two trees with the same content identity hold the same
  bytes under the same names, whatever their timestamps say.

Hashing a dataset of tens of gigabytes on every run would make the run pay
for the read it is about to do anyway, twice. So content hashing is
**incremental**: each file's sha256 is cached under its absolute path,
size, ``mtime_ns`` and inode, in a small JSON file per tree root (in the
user data directory, never in the dataset). A file whose stat triple is
unchanged reuses its cached digest without being opened; only new or
changed files are read. The cache is an optimisation, not a source of
truth: a missing, unreadable or unwritable cache only means every file is
read again. It is plain files written with an atomic rename, so the worker
child that hashes never touches SQLite.

The same reuse rule has the same blind spot as the stat identity: a file
rewritten in place with the same size, mtime and inode keeps its cached
digest. Rewriting a file within one timestamp tick and keeping its size is
rare enough that the trade is the right one for the large-dataset case
this exists for; ``refresh=True`` re-reads everything.

Every function here returns facts and never raises: a path that does not
exist is ``exists: False``, an unreadable file is counted in
``unreadable`` and leaves ``digest`` ``None`` (a digest over the files that
could be read would claim to identify a tree it did not see).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: A walk stops after this many files and reports ``truncated``; the digest
#: is then ``None`` (it would describe part of the tree only). Same bound as
#: ``cache_fingerprint._WALK_CAP``, for the same reason: a backstop against a
#: drive root picked by mistake, not an operating limit.
WALK_CAP = 200_000

#: Directory names a walk never enters: version control, caches and
#: editor state. They are not content a dataset or a pack is made of, and
#: ``__pycache__`` changes with every import.
SKIPPED_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__",
                          "node_modules", ".mypy_cache", ".pytest_cache",
                          ".ruff_cache", ".ipynb_checkpoints"})

#: File suffixes a walk leaves out for the same reason.
SKIPPED_SUFFIXES = (".pyc", ".pyo")

#: Bytes read per ``read`` call while hashing.
_CHUNK = 1 << 20

#: One lock per process for the cache files: two threads of one process
#: hashing the same root must not interleave their writes. Two processes
#: may still race; the loser's entries are simply recomputed next time.
_CACHE_LOCK = threading.Lock()


def default_cache_dir() -> Path:
    """``<user data>/cache/content-hashes`` (``CODEFYUI_USER_DATA_DIR`` aware)."""
    from ..config import _user_data_root

    return _user_data_root() / "cache" / "content-hashes"


# ── walking ───────────────────────────────────────────────────────────────


def _walk(root: Path) -> tuple[list[tuple[str, Path]], bool]:
    """``([(relative posix path, absolute path)], truncated)`` under *root*.

    A file *root* is its own single entry, named by its file name. Sorted,
    so the order is the same on every filesystem.
    """
    if root.is_file():
        return [(root.name, root)], False
    entries: list[tuple[str, Path]] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIPPED_DIRS)
        for name in filenames:
            if name.endswith(SKIPPED_SUFFIXES):
                continue
            if len(entries) >= WALK_CAP:
                truncated = True
                break
            path = Path(dirpath) / name
            entries.append((path.relative_to(root).as_posix(), path))
        if truncated:
            break
    entries.sort()
    return entries, truncated


def _empty(method: str) -> dict[str, Any]:
    return {"method": method, "exists": False, "digest": None, "files": 0,
            "bytes": 0, "latest_mtime": None, "truncated": False,
            "unreadable": 0}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


# ── the stat identity ─────────────────────────────────────────────────────


def stat_identity(paths: Iterable[Path | str]) -> dict[str, Any]:
    """The ``"stat"`` identity of one or more files or directories.

    ``digest`` is the sha256 of the sorted ``relative path, size,
    mtime_ns`` lines; ``files`` / ``bytes`` / ``latest_mtime`` summarise
    them. No content is read. Several *paths* are combined under their own
    names (``<name>/<relative path>``) so the same files under two roots
    are two identities.
    """
    result = _empty("stat")
    lines: list[str] = []
    latest: float | None = None
    any_exists = False
    for raw in _roots(paths):
        root = Path(raw)
        if not root.exists():
            lines.append(f"{root.name}\0missing")
            continue
        any_exists = True
        entries, truncated = _walk(root)
        result["truncated"] = result["truncated"] or truncated
        for rel, path in entries:
            try:
                st = path.stat()
            except OSError:
                result["unreadable"] += 1
                continue
            result["files"] += 1
            result["bytes"] += st.st_size
            latest = st.st_mtime if latest is None else max(latest, st.st_mtime)
            lines.append(f"{root.name}/{rel}\0{st.st_size}\0{st.st_mtime_ns}")
    result["exists"] = any_exists
    result["latest_mtime"] = _iso(latest)
    if any_exists and not result["truncated"]:
        result["digest"] = hashlib.sha256(
            "\n".join(lines).encode("utf-8")).hexdigest()
    return result


def _roots(paths: Iterable[Path | str]) -> list[Path]:
    seen: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path not in seen:
            seen.append(path)
    return sorted(seen, key=lambda p: (p.name, str(p)))


# ── the content identity ──────────────────────────────────────────────────


class _HashCache:
    """The per-root JSON file of ``{abs path: [size, mtime_ns, ino, sha256]}``."""

    def __init__(self, cache_dir: Path | None, root: Path) -> None:
        self.path: Path | None = None
        self.entries: dict[str, list[Any]] = {}
        self.dirty = False
        if cache_dir is None:
            return
        key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
        self.path = cache_dir / f"{key[:32]}.json"
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            self.entries = data["files"]

    def lookup(self, path: Path, st: os.stat_result) -> str | None:
        entry = self.entries.get(str(path))
        if (isinstance(entry, list) and len(entry) == 4
                and entry[0] == st.st_size and entry[1] == st.st_mtime_ns
                and entry[2] == st.st_ino and isinstance(entry[3], str)):
            return entry[3]
        return None

    def store(self, path: Path, st: os.stat_result, digest: str) -> None:
        self.entries[str(path)] = [st.st_size, st.st_mtime_ns, st.st_ino,
                                   digest]
        self.dirty = True

    def keep_only(self, paths: set[str]) -> None:
        stale = [p for p in self.entries if p not in paths]
        for p in stale:
            del self.entries[p]
        self.dirty = self.dirty or bool(stale)

    def save(self) -> None:
        if self.path is None or not self.dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"schema": 1, "files": self.entries},
                                      separators=(",", ":")),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            logger.debug("could not write the content-hash cache %s",
                         self.path, exc_info=True)


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def content_identity(paths: Iterable[Path | str], *,
                     cache_dir: Path | None | bool = True,
                     refresh: bool = False) -> dict[str, Any]:
    """The ``"sha256"`` identity of one or more files or directories.

    One file alone: ``digest`` is that file's own sha256, the value
    ``sha256sum`` prints. Anything else: the sha256 of the sorted
    ``<name>/<relative path>, sha256`` lines. ``reused`` counts the files
    whose digest came from the cache, ``hashed`` the files read now.

    *cache_dir* ``True`` uses :func:`default_cache_dir`; ``None`` or
    ``False`` disables the cache. *refresh* ignores cached digests (and
    rewrites them).
    """
    result = _empty("sha256")
    result["reused"] = 0
    result["hashed"] = 0
    if cache_dir is True:
        try:
            directory: Path | None = default_cache_dir()
        except Exception:  # noqa: BLE001 -- no user data dir: no cache
            directory = None
    else:
        directory = cache_dir or None
    roots = _roots(paths)
    lines: list[str] = []
    single: str | None = None
    latest: float | None = None
    any_exists = False
    for root in roots:
        if not root.exists():
            lines.append(f"{root.name}\0missing")
            continue
        any_exists = True
        entries, truncated = _walk(root)
        result["truncated"] = result["truncated"] or truncated
        with _CACHE_LOCK:
            cache = _HashCache(directory, root)
            seen: set[str] = set()
            for rel, path in entries:
                try:
                    st = path.stat()
                except OSError:
                    result["unreadable"] += 1
                    continue
                seen.add(str(path))
                digest = None if refresh else cache.lookup(path, st)
                if digest is None:
                    digest = _sha256_file(path)
                    if digest is None:
                        result["unreadable"] += 1
                        continue
                    cache.store(path, st, digest)
                    result["hashed"] += 1
                else:
                    result["reused"] += 1
                result["files"] += 1
                result["bytes"] += st.st_size
                latest = (st.st_mtime if latest is None
                          else max(latest, st.st_mtime))
                lines.append(f"{root.name}/{rel}\0{digest}")
                single = digest
            if not truncated:
                cache.keep_only(seen)
            cache.save()
    result["exists"] = any_exists
    result["latest_mtime"] = _iso(latest)
    if any_exists and not result["truncated"] and not result["unreadable"]:
        if len(roots) == 1 and roots[0].is_file() and single is not None:
            result["digest"] = single
        else:
            result["digest"] = hashlib.sha256(
                "\n".join(lines).encode("utf-8")).hexdigest()
    return result


def identity(paths: Iterable[Path | str], mode: str) -> dict[str, Any]:
    """:func:`content_identity` for *mode* ``"content"``, else
    :func:`stat_identity`."""
    paths = list(paths)
    if mode == "content":
        return content_identity(paths)
    return stat_identity(paths)
