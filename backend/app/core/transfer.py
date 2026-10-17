"""The ``transfer`` storage backend: the control plane's half (#785).

Without a shared filesystem the control plane is the artifact repository
(the model is Argo Workflows' artifact repository, keyed per workflow):

- **Inputs.** A run that targets a worker is read at submit
  (``run_needs``). When every file it reads is a stored upload named by a
  typed param (``needs["inputs"]``: a ``DATA_FILE``, ``IMAGE_FILE`` or
  ``MODEL_FILE`` value) and they add up to at most
  ``TRANSFER_MAX_INPUT_BYTES``, the run may go to a worker that does not
  hold the mount (:func:`annotate` writes the verdict into the needs as
  ``needs["transfer"]``). At assignment the inputs are named by sha256
  (:class:`ContentIndex`, kept in ``file_hashes`` and recomputed only when
  a file's size or mtime moved); the agent fetches what its cache lacks
  from the ordinary download endpoints (:data:`DOWNLOAD_ROUTES`, open reads
  like every GET; the agent sends its worker token anyway) and checks the
  hash.
- **Outputs.** The agent uploads what the run produced to
  ``POST /api/runs/{id}/artifacts/upload`` (``routes_runs``, streamed by
  :class:`Receiver`). Two places (``place``):

  * ``run``: a file the run reported as an artifact (a checkpoint, a
    TensorBoard folder file by file), kept at
    ``<data root>/runs/<run_id>/<kind>/<name>`` (:func:`run_key`, the
    name kept whole: :func:`file_part`); the run owns it.
  * ``data``: a file a writer node wrote (``CheckpointSaver``,
    ``ModelSaver``, ``ImageWriter``, ``VideoWrite``), kept where the same
    node writes on a local run: this server's ``MODELS_DIR``, ``MEDIA_DIR``
    or ``<data root>/output``, at the same relative path (:data:`DATA_PLACES`).
    Like a local run's, these are the user's: retention leaves them.

  ``run``: the same name sent again replaces its file (a retry); another
  name that cleans to a name already taken is suffixed (``a-1.pt``), the
  names recorded in the run's :data:`RUN_NAMES_FILE` (#838). ``data``: a
  different file already there is kept and the new one suffixed
  (``saved-1.pt``); the same content is the same file. Either way the worker reports the key the upload answered with in its
  ``artifact`` frame (``backend: "transfer"``), which records the row in
  the run's event order; :func:`key_path` maps a key back to a file and
  refuses one that is neither of those shapes.
- **Retention.** ``RunStore.prune`` and ``DELETE /api/runs/{id}`` remove a
  worker run's own directory (:func:`remove_run_files`, symlink-safe via
  :func:`owned_run_dir`); files placed in the data folders stay.

Placement (``scheduling.WorkerScheduler``): a worker that holds the mount is
preferred; a worker without it takes the run only through this backend; a
run this backend cannot serve keeps #782's refusal, now with the reason.

The agent's half is ``agent.transfer``; it imports this module's light
constants and helpers (nothing here imports anything heavy at module level).
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

#: ``run_warning`` kind for the agent's fetch progress ("inputs 3/5").
WARNING_TRANSFER_PROGRESS = "transfer_progress"
#: ``run_warning`` kind for an output the agent could not upload.
WARNING_TRANSFER_FAILED = "transfer_failed"

#: Where an upload goes (the ``place`` form field).
PLACE_RUN = "run"
PLACE_DATA = "data"
#: ``place=data``: the first part of the name -> where it lands here: a
#: settings attribute, or ``None`` for ``<data root>/output`` (ImageWriter's
#: folder). The same names as the run's own data folder on the worker.
DATA_PLACES: dict[str, str | None] = {"models": "MODELS_DIR",
                                      "media": "MEDIA_DIR", "output": None}

#: The upload route, and its path pattern (``main.auth_guard`` lets POST on
#: it through to the route's own check; ``body_limit`` caps it like an upload).
UPLOAD_ROUTE = "/api/runs/{run_id}/artifacts/upload"
UPLOAD_PATH_PATTERN = re.compile(r"^/api/runs/[^/]+/artifacts/upload$")

#: Input kind -> the setting naming its upload store.
STORE_SETTINGS = {"data_file": "DATA_FILES_DIR", "image": "IMAGES_DIR",
                  "model": "MODELS_DIR"}

#: Input kind -> the GET route that serves it (``{name}`` URL-quoted, ``/``
#: kept).
DOWNLOAD_ROUTES = {"data_file": "/api/files/download/",
                   "image": "/api/images/download/",
                   "model": "/api/models/download/"}

#: The directory under the data root that holds every run's received files.
RUNS_DIRNAME = "runs"

#: Bytes read at a time when hashing or copying.
CHUNK = 1024 * 1024

#: The name prefix of a file an upload is being received into
#: (:class:`Receiver`); listings skip such files.
INCOMING_PREFIX = ".incoming-"
#: Under the data root: one entry per upload being received, naming its
#: temporary file, removed when the upload ends either way. What is left
#: at startup names the temporary files a server that died mid-upload
#: left (:func:`sweep_incoming`), without walking the data folders.
INCOMING_LEDGER_DIRNAME = ".upload-ledger"


def is_incoming(path: Path | str) -> bool:
    """Whether *path* is an upload still being received (or left by one)."""
    return Path(path).name.startswith(INCOMING_PREFIX)


class TransferRefused(ValueError):
    """The run cannot use the transfer backend; the message says why."""


def human_size(value: float) -> str:
    """``12 B``, ``2.0 KiB``, ``3.10 GiB``."""
    if value < 1024:
        return f"{value:.0f} B"
    unit = "KiB"
    for unit in ("KiB", "MiB", "GiB"):
        value /= 1024
        if value < 1024:
            break
    return f"{value:.2f} {unit}" if unit == "GiB" else f"{value:.1f} {unit}"



def _allowed_extensions(kind: str) -> set[str]:
    """What the download route for *kind* serves (it refuses the rest)."""
    if kind == "data_file":
        from ..api.routes_data_files import ALLOWED_EXTENSIONS
    elif kind == "image":
        from ..api.routes_images import ALLOWED_EXTENSIONS
    else:
        from ..api.routes_models import ALLOWED_EXTENSIONS
    return set(ALLOWED_EXTENSIONS)


def store_path(kind: str, name: str) -> Path | None:
    """The stored upload *name* of *kind* on this server, or None.

    The node's own lookup decides what counts: ``CSVReader`` and
    ``ImageReader`` (and every reader sharing their rule) look a BARE file
    name up in their upload store, so a ``data_file`` or ``image`` input
    must be one; ``MODELS_DIR`` is searched by relative path, so a
    ``model`` may be nested (``periodic/x.pt``). The file must also be one
    the download route serves (its extension whitelist).
    """
    from ..config import settings
    from .data_paths import resolve_under

    attr = STORE_SETTINGS.get(kind)
    if attr is None or not name:
        return None
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    if kind != "model" and len(relative.parts) != 1:
        return None
    resolved = resolve_under(Path(getattr(settings, attr)), relative.as_posix())
    if resolved is None:
        return None
    try:
        if not resolved.is_file():
            return None
    except OSError:
        return None
    if resolved.suffix.lower() not in _allowed_extensions(kind):
        return None
    return resolved


def check(needs: Mapping[str, Any]) -> dict[str, Any]:
    """``{"ok", "reason", "bytes"}``: whether a run with *needs* may use
    the transfer backend. Reads the stores (a ``stat`` per input)."""
    from ..config import settings

    if not settings.TRANSFER_ENABLED:
        return {"ok": False, "bytes": 0,
                "reason": "the transfer backend is off on this server "
                          "(CODEFYUI_TRANSFER_ENABLED=0)"}
    paths = list(needs.get("paths") or [])
    if paths:
        shown = ", ".join(f"`{p}`" for p in paths[:3])
        more = f" and {len(paths) - 3} more" if len(paths) > 3 else ""
        return {"ok": False, "bytes": 0,
                "reason": f"it reads {shown}{more} by path, which the "
                          "transfer backend cannot ship (only uploads "
                          "picked in a file param can be)"}
    total = 0
    for entry in needs.get("inputs") or []:
        kind, name = str(entry.get("kind")), str(entry.get("name"))
        path = store_path(kind, name)
        if path is None:
            return {"ok": False, "bytes": 0,
                    "reason": f"`{name}` is not an upload on this server "
                              f"({kind.replace('_', ' ')})"}
        try:
            total += path.stat().st_size
        except OSError:
            return {"ok": False, "bytes": 0,
                    "reason": f"`{name}` could not be read on this server"}
    cap = int(settings.TRANSFER_MAX_INPUT_BYTES)
    if total > cap:
        return {"ok": False, "bytes": total,
                "reason": f"its inputs add up to {human_size(total)}, over the "
                          f"transfer limit of {human_size(cap)} "
                          "(CODEFYUI_TRANSFER_MAX_INPUT_BYTES)"}
    return {"ok": True, "bytes": total, "reason": None}


def annotate(needs: dict[str, Any] | None) -> dict[str, Any] | None:
    """*needs* with ``needs["transfer"]`` (:func:`check`) when it names a
    mount; returned as given otherwise."""
    if isinstance(needs, dict) and needs.get("mount"):
        needs["transfer"] = check(needs)
    return needs


def transferable(needs: Any) -> bool:
    """Whether a run with *needs* may go to a worker without its mount."""
    verdict = needs.get("transfer") if isinstance(needs, Mapping) else None
    return bool(isinstance(verdict, Mapping) and verdict.get("ok"))


def refusal(needs: Any) -> str | None:
    """Why *needs* cannot use the transfer backend, or None."""
    verdict = needs.get("transfer") if isinstance(needs, Mapping) else None
    if isinstance(verdict, Mapping) and not verdict.get("ok"):
        reason = verdict.get("reason")
        return str(reason) if reason else None
    return None


# ── the content index ─────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContentIndex:
    """sha256 per stored upload, refreshed lazily (``file_hashes``).

    :meth:`manifest` is what an assignment carries: the run's inputs with
    their hash and size. A row is trusted while the file's size and
    ``st_mtime_ns`` match it; otherwise the file is hashed again (off the
    event loop) and the row replaced.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    async def _row(self, kind: str, name: str) -> sqlite3.Row | None:
        def _select(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return conn.execute(
                "SELECT sha256, bytes, mtime FROM file_hashes "
                "WHERE kind = ? AND name = ?", (kind, name)).fetchone()

        return await self.db.run(_select)

    async def _store(self, kind: str, name: str, sha: str, size: int,
                     mtime: int) -> None:
        def _upsert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO file_hashes (kind, name, sha256, bytes, mtime) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(kind, name) DO UPDATE "
                "SET sha256 = excluded.sha256, bytes = excluded.bytes, "
                "mtime = excluded.mtime", (kind, name, sha, size, mtime))

        await self.db.run(_upsert)

    async def digest(self, kind: str, name: str) -> tuple[str, int]:
        """``(sha256, bytes)`` of the stored upload; ``TransferRefused``
        when it is not one."""
        path = store_path(kind, name)
        if path is None:
            raise TransferRefused(f"`{name}` is not an upload on this server "
                                  f"({kind.replace('_', ' ')})")
        try:
            stat = path.stat()
        except OSError as exc:
            raise TransferRefused(f"`{name}` could not be read: {exc}") \
                from None
        row = await self._row(kind, name)
        if row is not None and row["bytes"] == stat.st_size \
                and row["mtime"] == stat.st_mtime_ns:
            return row["sha256"], stat.st_size
        sha = await asyncio.to_thread(sha256_file, path)
        await self._store(kind, name, sha, stat.st_size, stat.st_mtime_ns)
        return sha, stat.st_size

    async def manifest(self, inputs: Iterable[Mapping[str, Any]]
                       ) -> list[dict[str, Any]]:
        """``[{"kind", "name", "sha256", "bytes"}]`` for *inputs*, or
        ``TransferRefused`` (a file gone, or the total over the cap since
        the run was submitted)."""
        from ..config import settings

        manifest: list[dict[str, Any]] = []
        total = 0
        for entry in inputs:
            kind, name = str(entry.get("kind")), str(entry.get("name"))
            sha, size = await self.digest(kind, name)
            total += size
            manifest.append({"kind": kind, "name": name, "sha256": sha,
                             "bytes": size})
        cap = int(settings.TRANSFER_MAX_INPUT_BYTES)
        if total > cap:
            raise TransferRefused(
                f"the run's inputs add up to {human_size(total)}, over the "
                f"transfer limit of {human_size(cap)} "
                "(CODEFYUI_TRANSFER_MAX_INPUT_BYTES)")
        return manifest


# ── received files ────────────────────────────────────────────────────────


def _part(value: str) -> str:
    """One id component (a run id, an artifact kind): capped at
    ``checkpoints.MAX_NAME_PART`` like the checkpoint builders' own."""
    from .checkpoints import _safe_part

    return _safe_part(value)


#: Caps on one file-name component of a received file (:func:`file_part`):
#: characters, and UTF-8 bytes (most filesystems allow 255 bytes per name).
#: Long enough for every name a checkpoint builder makes (two 48-character
#: id components plus the ``-e<epoch>b<batch>.pt`` stamp).
MAX_FILE_NAME = 128
MAX_FILE_NAME_BYTES = 240
#: Characters a shortened name keeps from its end (the position stamp and
#: the extension).
KEEP_TAIL = 40
#: The shortest a shortened name may be (:func:`file_part`): below it the
#: name is refused (:class:`NameTooLong`) rather than cut to nothing.
MIN_FILE_NAME = 24
#: The longest absolute path a received file may have, or None for no
#: limit beyond the per-name caps. Windows without long paths enabled
#: refuses a path over 259 characters (MAX_PATH less its NUL), so there the
#: last name is shortened to fit the folder it lands in (#838).
PATH_BUDGET: int | None = 259 if os.name == "nt" else None


class NameTooLong(ValueError):
    """A received file's path cannot fit this server's path limit."""


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def file_part(value: str, limit: int = MAX_FILE_NAME) -> str:
    """One file-name component of a received file: the characters
    ``checkpoints._safe_part`` keeps (``checkpoints._clean_part``), the
    length kept too (#838).

    ``_safe_part`` caps an id component at 48 characters; a whole file name
    (``<run id>-<node id>-e100.pt``) cut there lost its epoch stamp and
    extension, and two checkpoints took one name. A name over *limit*
    characters (at most :data:`MAX_FILE_NAME`) or :data:`MAX_FILE_NAME_BYTES`
    bytes is shortened in the middle: its start, a hash of the whole name,
    and its end (up to :data:`KEEP_TAIL` characters), so distinct names stay
    distinct and the stamp and extension survive. A name within
    ``MAX_NAME_PART`` comes out exactly as ``_safe_part`` made it.
    ``NameTooLong`` when *limit* is under :data:`MIN_FILE_NAME`.
    """
    from .checkpoints import _clean_part

    limit = min(limit, MAX_FILE_NAME)
    cleaned = _clean_part(value)
    if len(cleaned) <= limit and _utf8_len(cleaned) <= MAX_FILE_NAME_BYTES:
        return cleaned
    if limit < MIN_FILE_NAME:
        raise NameTooLong(f"{value!r} does not fit this server's path limit")
    digest = hashlib.sha256(
        value.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    room = limit - len(digest) - 2  # head + tail, around "-<digest>-"
    tail = cleaned[-min(KEEP_TAIL, room // 2):]
    head = cleaned[:room - len(tail)]
    spare = MAX_FILE_NAME_BYTES - _utf8_len(tail) - len(digest) - 2
    if _utf8_len(head) > spare:  # multi-byte characters: trim to the bytes
        used, keep = 0, 0
        for keep, char in enumerate(head):  # noqa: B007
            used += _utf8_len(char)
            if used > spare:
                break
        head = head[:keep]
    return f"{head}-{digest}-{tail}"


def checked_relative(name: str) -> PurePosixPath:
    """*name* as a relative POSIX path that stays where it is put;
    ``ValueError`` for one that is empty, absolute or climbs."""
    relative = PurePosixPath(str(name).replace("\\", "/"))
    if not relative.parts or relative.is_absolute() \
            or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"invalid name {name!r}")
    return relative


def clean_name(name: str) -> PurePosixPath:
    """:func:`checked_relative`, each part made a safe file name
    (:func:`file_part`)."""
    return PurePosixPath(*(file_part(p) for p in checked_relative(name).parts))


def runs_root() -> Path:
    from .data_paths import data_root

    return data_root() / RUNS_DIRNAME


def run_files_dir(run_id: str) -> Path:
    """``<data root>/runs/<run_id>``: what the run received."""
    return runs_root() / _part(run_id)


def owned_run_dir(run_id: str) -> Path | None:
    """*run_id*'s directory, iff it is a real directory exactly there.

    The delete guard, in the spirit of ``tensorboard.owned_logdir``: the
    path is built from the run id (never read from a row), it must not be a
    symlink, and it must resolve to itself under the resolved runs root, so
    a link planted at ``runs`` or ``runs/<run_id>`` cannot redirect a
    delete. None when there is no such directory.
    """
    try:
        literal = runs_root()  # under the (resolved) data root
        if literal.is_symlink() or literal.resolve() != literal:
            return None
        candidate = literal / _part(run_id)
        if candidate.is_symlink() or not candidate.is_dir():
            return None
        if candidate.resolve() != candidate:
            return None
    except OSError:
        return None
    return candidate


def run_key(run_id: str, kind: str, name: str) -> str:
    """``runs/<run_id>/<kind>/<name>``: a ``place=run`` key, relative to the
    data root."""
    return PurePosixPath(RUNS_DIRNAME, _part(run_id), _part(kind),
                         clean_name(name)).as_posix()


def _data_base(first: str) -> Path | None:
    from ..config import settings
    from .data_paths import data_root

    if first not in DATA_PLACES:
        return None
    attr = DATA_PLACES[first]
    return Path(getattr(settings, attr)) if attr else data_root() / "output"


def key_location(key: str) -> Path | None:
    """The file a ``transfer`` key names on this server, or None for a key
    of neither shape (:func:`run_key`, ``<models|media|output>/<path>``)."""
    from .data_paths import data_root

    try:
        relative = checked_relative(key)
    except ValueError:
        return None
    parts = relative.parts
    if parts[0] == RUNS_DIRNAME:
        return data_root().joinpath(*parts) if len(parts) >= 4 else None
    base = _data_base(parts[0])
    if base is None or len(parts) < 2:
        return None
    return base.joinpath(*parts[1:])


def key_path(run_id: str, key: str) -> Path | None:
    """:func:`key_location`, refusing a ``runs/`` key of another run: what
    a worker may record for *run_id*."""
    relative = PurePosixPath(str(key))
    if relative.parts[:1] == (RUNS_DIRNAME,) \
            and relative.parts[1:2] != (_part(run_id),):
        return None
    return key_location(key)


def remove_run_files(run_id: str, *, keep_checkpoints: bool = False) -> bool:
    """Delete what was received for *run_id* (retention, ``DELETE``).
    Blocking. Symlink-safe: only :func:`owned_run_dir`, and nothing below it
    is followed (``rmtree`` unlinks a link, never its target).

    *keep_checkpoints* keeps its real ``checkpoint`` directory: the run was
    ``interrupted``, and ``RunStore.prune`` leaves such a run's resume
    point on disk. True when something was removed.
    """
    directory = owned_run_dir(run_id)
    if directory is None:
        return False
    if not keep_checkpoints:
        shutil.rmtree(directory, ignore_errors=True)
        return True
    removed = False
    for child in directory.iterdir():
        if child.name == "checkpoint" and child.is_dir() \
                and not child.is_symlink():
            continue
        if child.name == RUN_NAMES_FILE:  # who took the kept names
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        removed = True
    return removed


def incoming_ledger() -> Path:
    from .data_paths import data_root

    return data_root() / INCOMING_LEDGER_DIRNAME


def sweep_incoming() -> int:
    """Remove the temporary files of uploads that never finished: the ones
    the ledger (:data:`INCOMING_LEDGER_DIRNAME`) still names. Startup only:
    ``server.lock`` keeps one server per data directory, so no upload is in
    progress then and every entry is stale. Reads the ledger, never walks
    the data folders (they may be large). A named file that is not an
    ``.incoming-*`` regular file is left alone. Blocking. How many
    temporary files were removed."""
    ledger = incoming_ledger()
    if ledger.is_symlink() or not ledger.is_dir():
        return 0
    removed = 0
    for entry in ledger.iterdir():
        try:
            if entry.is_file() and not entry.is_symlink():
                path = Path(entry.read_text(encoding="utf-8").strip())
                if path.is_absolute() and is_incoming(path) \
                        and not path.is_symlink() and path.is_file():
                    path.unlink()
                    removed += 1
            entry.unlink(missing_ok=True)
        except OSError:
            logger.debug("could not sweep %s", entry, exc_info=True)
    if removed:
        logger.info("removed %d unfinished upload(s) (%s*) left by an "
                    "earlier server", removed, INCOMING_PREFIX)
    return removed


#: In a run's directory: which source name took each received name
#: (:func:`place_file`), ``{"<kind>/<name>": "<name as sent>"}``.
RUN_NAMES_FILE = ".names.json"
_run_names_lock = threading.Lock()


def _fit_name(raw: str, folder: Path, n: int = 0, *,
              budget: bool = True) -> str:
    """:func:`file_part` of *raw* for a file in *folder*, suffixed ``-n``
    when *n*, shortened to fit :data:`PATH_BUDGET` when *budget*."""
    tag = f"-{n}" if n else ""
    limit = MAX_FILE_NAME - len(tag)
    if budget and PATH_BUDGET is not None:
        limit = min(limit, PATH_BUDGET - len(str(folder)) - 1 - len(tag))
    name = file_part(raw, limit)
    if not tag:
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{tag}.{ext}" if dot and stem else f"{name}{tag}"


def _check_budget(path: Path) -> None:
    if PATH_BUDGET is not None and len(str(path)) > PATH_BUDGET:
        raise NameTooLong(f"{path} is over this server's path limit of "
                          f"{PATH_BUDGET} characters")


def _too_long(exc: OSError) -> bool:
    return exc.errno == errno.ENAMETOOLONG \
        or getattr(exc, "winerror", None) == 206  # ERROR_FILENAME_EXCED_RANGE


def place_file(tmp: Path, place: str, run_id: str, kind: str, name: str,
               sha256: str) -> str:
    """Move the received *tmp* to its place; the key. Blocking.

    ``run``: under the run's directory (:func:`_place_run`): the same name
    sent again replaces its file atomically (a retry, a file that grew); a
    different name that cleans to a name another one took is suffixed
    ``-1``, ``-2`` ... (#838). ``data``: the writer's path under
    :data:`DATA_PLACES`, never over a different file (:func:`_claim`).
    Either way the last name is shortened to fit :data:`PATH_BUDGET`.
    ``ValueError`` for a name that fits neither, ``NameTooLong`` (a
    ``ValueError`` too) for a path this server cannot hold.
    """
    try:
        if place == PLACE_RUN:
            return _place_run(tmp, run_id, kind, name)
        if place != PLACE_DATA:
            raise ValueError(f"unknown place {place!r}")
        raw = checked_relative(name).parts
        relative = clean_name(name)
        if relative.parts[0] not in DATA_PLACES or len(relative.parts) < 2:
            raise ValueError(f"{name!r} is not under "
                             f"{', '.join(DATA_PLACES)}")
        base = _data_base(relative.parts[0])
        assert base is not None
        folder = base.joinpath(*relative.parts[1:-1])
        folder.mkdir(parents=True, exist_ok=True)
        candidate = _claim(
            tmp, lambda n: folder / _fit_name(raw[-1], folder, n), sha256)
        key_parts = (relative.parts[0], *candidate.relative_to(base).parts)
        return PurePosixPath(*key_parts).as_posix()
    except OSError as exc:
        if _too_long(exc):
            raise NameTooLong(f"{name!r}: {exc.strerror or exc}") from None
        raise


def _read_names(path: Path) -> dict[str, str]:
    try:
        names = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return names if isinstance(names, dict) else {}


def _place_run(tmp: Path, run_id: str, kind: str, name: str) -> str:
    """``place=run``: each part of *name* takes the cleaned name it maps to,
    unless a different source name (recorded in :data:`RUN_NAMES_FILE`) or
    a file nobody recorded (one 3.1.0 received) holds it; then ``-1``,
    ``-2`` ... So a directory artifact (``tb run/...``) claims its folder
    once, and another folder that cleans alike (``tb_run/...``) gets its
    own."""
    raw = checked_relative(name).parts
    run_dir = run_files_dir(run_id)
    if owned_run_dir(run_id) is None:
        raise ValueError("the run's directory is not a plain directory")
    kind_part = _part(kind)
    with _run_names_lock:
        ledger = run_dir / RUN_NAMES_FILE
        names = _read_names(ledger)
        changed = False
        folder = run_dir / kind_part
        taken: list[str] = [kind_part]
        for depth, part in enumerate(raw):
            source = "/".join(raw[:depth + 1])
            last = depth == len(raw) - 1
            n = 0
            while True:
                candidate = _fit_name(part, folder, n, budget=last)
                key = "/".join((*taken, candidate))
                path = folder / candidate
                owner = names.get(key)
                wrong_type = path.is_dir() if last else path.is_file()
                if owner == source and not wrong_type:
                    break
                if owner is None and not os.path.lexists(path):
                    names[key] = source
                    changed = True
                    break
                n += 1
            taken.append(candidate)
            folder = path
        _check_budget(folder)
        folder.parent.mkdir(parents=True, exist_ok=True)
        if changed:
            scratch = ledger.with_name(f"{INCOMING_PREFIX}{uuid.uuid4().hex}")
            scratch.write_text(json.dumps(names), encoding="utf-8")
            os.replace(scratch, ledger)
        os.replace(tmp, folder)
    return PurePosixPath(RUNS_DIRNAME, _part(run_id), *taken).as_posix()


def _claim(tmp: Path, candidate_at: Callable[[int], Path],
           sha256: str) -> Path:
    """Move *tmp* to ``candidate_at(0)``, or ``candidate_at(1)``, ``(2)``
    ... when a different file holds the name; the path it landed at. A file
    with the same content already there (same size, then same sha256) is
    taken as this one (*tmp* is dropped). ``os.link`` refuses an existing
    name, so two uploads cannot take one name."""
    size = tmp.stat().st_size
    n = 0
    while True:
        candidate = candidate_at(n)
        _check_budget(candidate)
        try:
            os.link(tmp, candidate)
        except FileExistsError:
            try:
                same = candidate.is_file() \
                    and candidate.stat().st_size == size \
                    and sha256_file(candidate) == sha256
            except OSError:
                same = False
            if same:
                tmp.unlink(missing_ok=True)
                break
            n += 1
            continue
        except OSError as exc:  # no hard links here: claim the name alone
            if _too_long(exc):
                raise
            try:
                fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                n += 1
                continue
            os.close(fd)
            os.replace(tmp, candidate)
            break
        tmp.unlink(missing_ok=True)
        break
    return candidate


class Receiver:
    """Streams one multipart upload straight to a temporary file (no second
    spool), capped and hashed as it arrives.

    Feed it the body with :meth:`write` (sync: ``python_multipart`` calls
    back), then :meth:`close`. Form fields (``kind``, ``name``, ``place``,
    ``sha256``) must come before the ``file`` part, as the agent sends them;
    the file goes to ``<directory(fields)>/.incoming-<uuid>`` (a name of its
    own per request, beside where it will land, so the final move is a
    rename on one filesystem), recorded in the ledger
    (:func:`incoming_ledger`) until :meth:`discard`.
    """

    FIELDS = ("kind", "name", "place", "sha256")

    def __init__(self, content_type: str,
                 directory: Callable[[dict[str, str]], Path],
                 limit: int) -> None:
        from python_multipart.multipart import MultipartParser, \
            parse_options_header

        _, options = parse_options_header(content_type)
        boundary = options.get(b"boundary")
        if not boundary:
            raise ValueError("expected a multipart body")
        self.directory = directory
        self.limit = limit
        self.fields: dict[str, str] = {}
        self.size = 0
        self.tmp: Path | None = None
        self._ledger_entry: Path | None = None
        self.got_file = False
        self._digest = hashlib.sha256()
        self._out: Any = None
        self._header_field = b""
        self._headers: dict[bytes, bytes] = {}
        self._part: str | None = None
        self._value = bytearray()
        self.error: str | None = None
        callbacks = {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field_cb,
            "on_header_value": self._header_value_cb,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }
        self._parser = MultipartParser(boundary, callbacks)

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def _part_begin(self) -> None:
        self._headers = {}
        self._part = None
        self._value = bytearray()

    def _header_field_cb(self, data: bytes, start: int, end: int) -> None:
        self._header_field = data[start:end].lower()

    def _header_value_cb(self, data: bytes, start: int, end: int) -> None:
        self._headers[self._header_field] = \
            self._headers.get(self._header_field, b"") + data[start:end]

    def _headers_finished(self) -> None:
        from python_multipart.multipart import parse_options_header

        _, options = parse_options_header(
            self._headers.get(b"content-disposition", b""))
        self._part = (options.get(b"name") or b"").decode("utf-8", "replace")
        if self._part in self.FIELDS and self.got_file:
            # Where the file goes depends on them: they come first.
            raise ValueError("the form fields must come before the file")
        if self._part == "file":
            if self.got_file:
                raise ValueError("one file per upload")
            self.got_file = True
            folder = self.directory(self.fields)
            folder.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            self.tmp = folder / f"{INCOMING_PREFIX}{token}"
            ledger = incoming_ledger()
            ledger.mkdir(parents=True, exist_ok=True)
            self._ledger_entry = ledger / token
            self._ledger_entry.write_text(str(self.tmp.resolve()),
                                          encoding="utf-8")
            self._out = open(self.tmp, "wb")  # noqa: SIM115 -- closed in close()

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._part == "file":
            chunk = data[start:end]
            self.size += len(chunk)
            if self.size > self.limit:
                raise OverflowError(f"file is over {self.limit} bytes")
            self._digest.update(chunk)
            self._out.write(chunk)
        elif self._part in self.FIELDS:
            self._value += data[start:end]
            if len(self._value) > 4096:
                raise ValueError(f"field {self._part} is too long")

    def _part_end(self) -> None:
        if self._part == "file":
            self._out.close()
            self._out = None
        elif self._part in self.FIELDS:
            self.fields[self._part] = self._value.decode("utf-8").strip()

    def write(self, chunk: bytes) -> None:
        self._parser.write(chunk)

    def close(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None
        self._parser.finalize()

    def discard(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None
        if self.tmp is not None:
            self.tmp.unlink(missing_ok=True)
        if self._ledger_entry is not None:
            self._ledger_entry.unlink(missing_ok=True)
