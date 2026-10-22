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
exist is ``exists: False`` (with several paths, a missing one is a note),
an unreadable file or directory is counted in ``unreadable``, and any of
them leaves ``digest`` ``None`` (a digest over the inputs that could be
read would claim to identify inputs it did not see). ``notes`` says what
was missing, unreadable or not followed.

A walk follows directory symlinks (a dataset folder that links its class
folders from another disk is made of those files), never a link back into
its own path, and skips version control, caches, virtual environments,
build output and desktop metadata (:data:`SKIPPED_DIRS`,
:data:`SKIPPED_DIR_SUFFIXES`, :data:`SKIPPED_FILES`,
:data:`SKIPPED_SUFFIXES`).
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

#: Directory names a walk never enters: version control, caches, editor
#: state, virtual environments and build output. They are not content a
#: dataset or a pack is made of, and they change on their own
#: (``__pycache__`` with every import, a ``.venv`` with every
#: ``pip install``); a ``.venv`` inside a linked pack checkout would also
#: be gigabytes read at every run start.
SKIPPED_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__",
                          "node_modules", ".mypy_cache", ".pytest_cache",
                          ".ruff_cache", ".ipynb_checkpoints", ".venv",
                          "venv", "env", ".tox", ".nox", "build", "dist"})

#: Directory name suffixes a walk never enters (``<name>.egg-info``).
SKIPPED_DIR_SUFFIXES = (".egg-info",)

#: File suffixes a walk leaves out for the same reason.
SKIPPED_SUFFIXES = (".pyc", ".pyo")

#: File names a walk leaves out: desktop metadata the file manager writes.
SKIPPED_FILES = frozenset({".DS_Store", "Thumbs.db"})

#: Seconds ``git ls-files`` may take before a pack is walked instead.
GIT_TIMEOUT = 10.0

#: Bytes read per ``read`` call while hashing.
_CHUNK = 1 << 20

#: One lock per cache file, held only while that file is loaded or merged
#: and saved, never while content is read: hashing a large dataset must
#: not stall the hashing of an unrelated root (every run start hashes the
#: enabled packs). ``_LOCKS_GUARD`` guards the dict itself.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _cache_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def default_cache_dir() -> Path:
    """``<user data>/cache/content-hashes`` (``CODEFYUI_USER_DATA_DIR`` aware)."""
    from ..config import _user_data_root

    return _user_data_root() / "cache" / "content-hashes"


# ── walking ───────────────────────────────────────────────────────────────


def _skipped_dir(name: str) -> bool:
    return name in SKIPPED_DIRS or name.endswith(SKIPPED_DIR_SUFFIXES)


def _skipped_file(name: str) -> bool:
    return name in SKIPPED_FILES or name.endswith(SKIPPED_SUFFIXES)


class _Listing:
    """What a walk found: ``entries`` (sorted ``(relative posix path,
    path)``), ``truncated``, ``unreadable`` (directories that could not be
    listed) and ``notes``."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, Path]] = []
        self.truncated = False
        self.unreadable = 0
        self.notes: list[str] = []

    def add(self, rel: str, path: Path) -> bool:
        """Add one file; False (and ``truncated``) once the cap is hit."""
        if len(self.entries) >= WALK_CAP:
            self.truncated = True
            return False
        self.entries.append((rel, path))
        return True


def _dir_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _walk_dir(directory: Path, prefix: str, listing: _Listing,
              ancestors: tuple[tuple[int, int], ...]) -> None:
    """Add the files under *directory*, following directory symlinks.

    A dataset folder often links its class folders from elsewhere
    (``train/cats -> /mnt/x/cats``); those files are the dataset. A link
    back to a directory on the current path (a loop) is not followed, and
    a note says so; a directory that cannot be listed counts as
    ``unreadable``. Names are visited in sorted order, so the listing is
    the same on every run.
    """
    key = _dir_key(directory)
    if key is None or key in ancestors:
        listing.notes.append(
            f"not followed (a symlink loop): {prefix.rstrip('/') or '.'}")
        return
    ancestors = ancestors + (key,)
    try:
        with os.scandir(directory) as scan:
            children = sorted((entry.name, entry) for entry in scan)
    except OSError as exc:
        listing.unreadable += 1
        listing.notes.append(
            f"unreadable directory: {prefix.rstrip('/') or '.'} "
            f"({type(exc).__name__})")
        return
    for name, entry in children:
        try:
            is_dir = entry.is_dir()  # follows a symlink
        except OSError:
            is_dir = False
        if is_dir:
            if _skipped_dir(name):
                continue
            _walk_dir(Path(entry.path), f"{prefix}{name}/", listing,
                      ancestors)
        elif not _skipped_file(name):
            if not listing.add(f"{prefix}{name}", Path(entry.path)):
                return
        if listing.truncated:
            return


def _git_listing(root: Path) -> _Listing | None:
    """The files of *root* as git sees them, or None (not a checkout, no
    git, too slow, or git lists nothing).

    Tracked files plus untracked files that are not ignored
    (``git ls-files --cached --others --exclude-standard``): what a pack
    author's ``.gitignore`` excludes (a virtual environment, build output,
    downloaded weights) is not the pack's code. A listed path that is a
    directory (a submodule, a linked directory) is walked; a tracked file
    deleted from the work tree is left out.
    """
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        return None
    try:
        done = subprocess.run(
            [git, "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, timeout=GIT_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    names = sorted({name for name in
                    done.stdout.decode("utf-8", "surrogateescape").split("\0")
                    if name})
    if not names:
        return None
    listing = _Listing()
    for rel in names:
        parts = rel.split("/")
        if any(_skipped_dir(part) for part in parts[:-1]) \
                or _skipped_file(parts[-1]):
            continue
        path = root / rel
        if path.is_dir():
            if _skipped_dir(parts[-1]):
                continue
            _walk_dir(path, f"{rel}/", listing, (_dir_key(root) or (0, 0),))
        elif path.exists():
            listing.add(rel, path)
        if listing.truncated:
            break
    listing.entries.sort()
    return listing


def _walk(root: Path, *, use_git: bool = False) -> _Listing:
    """The files under *root*: a file *root* is its own single entry, named
    by its file name. With *use_git*, a directory inside a git checkout is
    listed by git (:func:`_git_listing`) when it can be."""
    if root.is_file():
        listing = _Listing()
        listing.add(root.name, root)
        return listing
    if use_git:
        listing = _git_listing(root)
        if listing is not None:
            return listing
    listing = _Listing()
    _walk_dir(root, "", listing, ())
    listing.entries.sort()
    return listing


def _empty(method: str) -> dict[str, Any]:
    return {"method": method, "exists": False, "digest": None, "files": 0,
            "bytes": 0, "latest_mtime": None, "truncated": False,
            "unreadable": 0, "notes": []}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _missing(result: dict[str, Any], root: Path) -> None:
    result["missing"] = True
    result["notes"].append(f"missing: {root.name}")


def _finish(result: dict[str, Any], lines: list[str],
            latest: float | None) -> None:
    """Set ``exists``, ``latest_mtime`` and ``digest``.

    The digest is ``None`` unless every input was seen: an input that is
    missing, a file or directory that could not be read, or a walk that
    stopped at :data:`WALK_CAP` leaves it ``None``, because a digest over
    the part that could be read would claim to identify inputs it did not
    see.
    """
    missing = result.pop("missing", False)
    result["latest_mtime"] = _iso(latest)
    if (result["exists"] and not missing and not result["truncated"]
            and not result["unreadable"]):
        result["digest"] = hashlib.sha256(
            "\n".join(lines).encode("utf-8")).hexdigest()


# ── the stat identity ─────────────────────────────────────────────────────


def stat_identity(paths: Iterable[Path | str], *,
                  use_git: bool = False) -> dict[str, Any]:
    """The ``"stat"`` identity of one or more files or directories.

    ``digest`` is the sha256 of the sorted ``relative path, size,
    mtime_ns`` lines; ``files`` / ``bytes`` / ``latest_mtime`` summarise
    them. No content is read. Several *paths* are combined under their own
    names (``<name>/<relative path>``) so the same files under two roots
    are two identities. *use_git* as for :func:`content_identity`.
    """
    result = _empty("stat")
    lines: list[str] = []
    latest: float | None = None
    for root in _roots(paths):
        if not root.exists():
            _missing(result, root)
            continue
        result["exists"] = True
        listing = _walk(root, use_git=use_git)
        result["truncated"] = result["truncated"] or listing.truncated
        result["unreadable"] += listing.unreadable
        result["notes"].extend(listing.notes)
        for rel, path in listing.entries:
            try:
                st = path.stat()
            except OSError:
                result["unreadable"] += 1
                result["notes"].append(f"unreadable: {root.name}/{rel}")
                continue
            result["files"] += 1
            result["bytes"] += st.st_size
            latest = st.st_mtime if latest is None else max(latest, st.st_mtime)
            lines.append(f"{root.name}/{rel}\0{st.st_size}\0{st.st_mtime_ns}")
    _finish(result, lines, latest)
    return result


def _roots(paths: Iterable[Path | str]) -> list[Path]:
    seen: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path not in seen:
            seen.append(path)
    return sorted(seen, key=lambda p: (p.name, str(p)))


# ── the content identity ──────────────────────────────────────────────────


def _read_cache_file(path: Path) -> dict[str, list[Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("files"), dict):
        return data["files"]
    return {}


class _HashCache:
    """The per-root JSON file of ``{abs path: [size, mtime_ns, ino, sha256]}``.

    Loaded once before the files are read, and merged into the file on
    disk at :meth:`save`: the file is read again under the lock and this
    call's new entries (and removals) are applied to what is there, so a
    concurrent hash of the same root in another thread or process keeps
    the entries it wrote. Neither load nor save holds the lock while a
    dataset file is read.
    """

    def __init__(self, cache_dir: Path | None, root: Path) -> None:
        self.path: Path | None = None
        self.key = ""
        self.entries: dict[str, list[Any]] = {}
        self.fresh: dict[str, list[Any]] = {}
        self.keep: set[str] | None = None
        if cache_dir is None:
            return
        self.key = hashlib.sha256(
            str(root.resolve()).encode("utf-8")).hexdigest()[:32]
        self.path = cache_dir / f"{self.key}.json"
        with _cache_lock(self.key):
            self.entries = _read_cache_file(self.path)

    def lookup(self, path: Path, st: os.stat_result) -> str | None:
        entry = self.entries.get(str(path))
        if (isinstance(entry, list) and len(entry) == 4
                and entry[0] == st.st_size and entry[1] == st.st_mtime_ns
                and entry[2] == st.st_ino and isinstance(entry[3], str)):
            return entry[3]
        return None

    def store(self, path: Path, st: os.stat_result, digest: str) -> None:
        self.fresh[str(path)] = [st.st_size, st.st_mtime_ns, st.st_ino,
                                 digest]

    def keep_only(self, paths: set[str]) -> None:
        """Drop the entries of files this tree no longer has, at save."""
        self.keep = set(paths)

    def save(self) -> None:
        if self.path is None:
            return
        stale = (self.keep is not None
                 and any(p not in self.keep for p in self.entries))
        if not self.fresh and not stale:
            return
        try:
            with _cache_lock(self.key):
                merged = _read_cache_file(self.path)
                merged.update(self.fresh)
                if self.keep is not None:
                    merged = {p: e for p, e in merged.items()
                              if p in self.keep}
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(
                    f"{self.path.name}.{os.getpid()}."
                    f"{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps({"schema": 1, "files": merged},
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
                     refresh: bool = False,
                     use_git: bool = False) -> dict[str, Any]:
    """The ``"sha256"`` identity of one or more files or directories.

    One file alone: ``digest`` is that file's own sha256, the value
    ``sha256sum`` prints. Anything else: the sha256 of the sorted
    ``<name>/<relative path>, sha256`` lines. ``reused`` counts the files
    whose digest came from the cache, ``hashed`` the files read now.

    *cache_dir* ``True`` uses :func:`default_cache_dir`; ``None`` or
    ``False`` disables the cache. *refresh* ignores cached digests (and
    rewrites them). *use_git* lists a directory inside a git checkout by
    ``git ls-files`` (tracked and untracked, not ignored files) when git
    answers within :data:`GIT_TIMEOUT`, and walks it otherwise; it is for
    pack directories, whose ``.gitignore`` says what is not code.
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
    for root in roots:
        if not root.exists():
            _missing(result, root)
            continue
        result["exists"] = True
        listing = _walk(root, use_git=use_git)
        result["truncated"] = result["truncated"] or listing.truncated
        result["unreadable"] += listing.unreadable
        result["notes"].extend(listing.notes)
        cache = _HashCache(directory, root)
        seen: set[str] = set()
        for rel, path in listing.entries:
            try:
                st = path.stat()
            except OSError:
                result["unreadable"] += 1
                result["notes"].append(f"unreadable: {root.name}/{rel}")
                continue
            seen.add(str(path))
            digest = None if refresh else cache.lookup(path, st)
            if digest is None:
                digest = _sha256_file(path)
                if digest is None:
                    result["unreadable"] += 1
                    result["notes"].append(f"unreadable: {root.name}/{rel}")
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
        if not listing.truncated and not listing.unreadable:
            cache.keep_only(seen)
        cache.save()
    _finish(result, lines, latest)
    if (result["digest"] is not None and len(roots) == 1
            and roots[0].is_file() and single is not None):
        result["digest"] = single
    return result


def identity(paths: Iterable[Path | str], mode: str) -> dict[str, Any]:
    """:func:`content_identity` for *mode* ``"content"``, else
    :func:`stat_identity`."""
    paths = list(paths)
    if mode == "content":
        return content_identity(paths)
    return stat_identity(paths)
