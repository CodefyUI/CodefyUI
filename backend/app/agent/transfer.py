"""The ``transfer`` storage backend: the agent's half (#785).

A run assigned with ``storage.backend == "transfer"`` comes from a control
plane this worker shares no directory with. The agent then

1. gives the run a data directory of its own, ``<runs_dir>/<run_id>/data``,
   laid out like a plain data directory (``files/``, ``images/``,
   ``models/``, ``media/`` ...; ``core.mounts.data_root_env`` points the
   child's settings at it), so whatever the graph writes stays inside the
   run and nothing touches this machine's own data directory but its
   ``cache/`` (:data:`SHARED_DIRS`, linked in);
2. fetches each input (``storage.inputs``) its cache lacks from the control
   plane's download route (``GET /api/files/download/{name}`` and its two
   siblings, sending the worker token, over one HTTP connection per run),
   checks its sha256, and keeps it in the per-worker cache
   ``<user-data>/agent-cache/<sha256>`` (:class:`InputCache`). The run's
   directory gets a COPY under the name the graph uses (so ``CSVReader``
   resolves a bare ``train.csv`` against ``DATA_FILES_DIR`` exactly as on
   the control plane): a node that writes over its input cannot reach the
   cache, and a changed input is uploaded back as an output. Progress goes
   out as ``run_warning`` frames of kind ``transfer_progress``
   ("fetching inputs 2/3");
3. uploads what the run produced (:class:`RunTransfer`, on an uploader
   thread of its own so the run subprocess never waits on the network):
   every file an ``artifact`` frame names (a directory file by file) as it
   arrives, into the run's directory on the control plane (``place=run``);
   after the subprocess exited, every other file the run wrote under
   ``models/``, ``media/`` and ``output/`` (``CheckpointSaver``,
   ``ModelSaver``, ``VideoWrite``, ``ImageWriter``), and any input it
   changed, into the control plane's own data folders at the same relative
   path (``place=data``, kind ``output``); and last the interrupt
   checkpoint, held back until then, so a cancelled run's resume point is
   on the control plane before its ``run_finished``. The ``artifact`` frame
   then carries ``backend: "transfer"`` and the key the control plane
   stored the file under. A file that cannot be uploaded (over the control
   plane's ``MAX_UPLOAD_SIZE``, the control plane unreachable, the uploads
   cancelled) stays reported by its path on this worker, with a
   ``transfer_failed`` warning. An unreachable control plane (no
   connection, :data:`OUTAGE_STATUSES`) is retried, backing off, for the
   control plane's resume grace (``welcome.resume_grace_s``, else
   :data:`UPLOAD_RETRY_WINDOW_S`), one window per run's outage, so a run
   that ends during a connection drop still delivers its files; its
   ``run_finished`` goes out after them.

The run's directory goes with the agent's own run retention
(``runner.prune_run_dirs``); the cache is never pruned (delete the folder
to reclaim it; a missing entry is fetched again).

Blocking code, called from the agent's threads. Shares its constants and
helpers with the control plane's half (``core.transfer``), which imports
nothing heavy.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote

from ..core.transfer import (
    CHUNK,
    DATA_PLACES,
    DOWNLOAD_ROUTES,
    PLACE_DATA,
    PLACE_RUN,
    UPLOAD_ROUTE,
    checked_relative,
    human_size,
    sha256_file,
)

logger = logging.getLogger(__name__)

#: The cache folder under the agent's user-data directory.
CACHE_DIRNAME = "agent-cache"
#: The run's own data directory, under ``<runs_dir>/<run_id>/``.
RUN_DATA_DIRNAME = "data"

#: Input kind -> its folder in a plain data directory
#: (``core.mounts.DATA_DIR_LAYOUT``).
KIND_DIRS = {"data_file": "files", "image": "images", "model": "models"}

#: The artifact kind of a file the run wrote without reporting it.
OUTPUT_KIND = "output"
#: Where, under the run's data directory, the run's unreported outputs are
#: looked for: the writer nodes' folders, named as the control plane
#: places them (``core.transfer.DATA_PLACES``).
OUTPUT_DIRS = tuple(DATA_PLACES)

#: Attempts per HTTP transfer, with a pause of ``RETRY_PAUSE_S * attempt``.
ATTEMPTS = 3
RETRY_PAUSE_S = 1.0
#: An upload that failed because the control plane is away (no
#: connection, or :data:`OUTAGE_STATUSES`) keeps being retried for the
#: control plane's resume grace (``CODEFYUI_WORKER_RESUME_GRACE_S``, sent
#: in ``welcome.resume_grace_s``; this default when an older control plane
#: does not send it), the pause doubling up to :data:`UPLOAD_MAX_PAUSE_S`:
#: a connection drop that the run survives must not lose the files it wrote
#: meanwhile, and the run's ``run_finished`` waits for them. Any other 5xx
#: gets the :data:`ATTEMPTS` quick retries only.
UPLOAD_RETRY_WINDOW_S = 300.0
UPLOAD_MAX_PAUSE_S = 30.0
#: Statuses that mean "not reachable right now", not "this upload failed":
#: a proxy with no upstream (502/503/504), a timeout, a rate limit.
OUTAGE_STATUSES = frozenset({408, 429, 502, 503, 504})
TIMEOUT_S = 60.0
#: How close to a cache entry's last write its stat is not trusted
#: (:class:`InputCache`): above every common file clock's tick.
RACY_WINDOW_NS = 3_000_000_000
#: The stamp of an entry verified within :data:`RACY_WINDOW_NS` of its last
#: write: it never matches, so the next lookup hashes the entry again.
RACY_STAMP = "racy"
#: The wall clock the racy window is measured on (the file clock's).
_now_ns = time.time_ns


class TransferError(RuntimeError):
    """An input could not be fetched, or an output not uploaded."""


class TransferCancelled(TransferError):
    """The run was cancelled while its inputs were being fetched."""


def _resolve(value: Any) -> Any:
    return value() if callable(value) else value


def safe_relative(name: str) -> PurePosixPath:
    """``core.transfer.checked_relative``, as a ``TransferError``."""
    try:
        return checked_relative(name)
    except ValueError:
        raise TransferError(f"refusing the input name {name!r}") from None


def walk_files(directory: Path) -> Iterator[Path]:
    """Every regular file under *directory*, sorted, never following a
    symlink (a link to a folder is not entered, a link to a file skipped)."""
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = Path(root) / name
            if not path.is_symlink() and path.is_file():
                yield path


class InputCache:
    """``<root>/<sha256>``: every input this worker fetched, by content.

    Beside each entry, ``<sha256>.ok`` records its stat (size, mtime, file
    id and, off Windows, ctime) when it was verified. An entry whose stat no
    longer matches (someone edited the cache) is hashed again before it is
    used, and fetched again when it no longer hashes to its name.

    A stat taken within :data:`RACY_WINDOW_NS` of the entry's last change
    (its mtime, or its ctime when later, off Windows) proves nothing: the file clock ticks coarsely (about 16 ms on Windows,
    a few ms on Linux, 2 s on FAT), so an edit of the same size in the same
    tick leaves every field as it was. Such an entry gets a stamp that never
    matches, and is hashed again on its next lookup (git's "racily clean"
    rule); the stamp written then, once the window has passed, is trusted.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, sha256: str) -> Path:
        return self.root / sha256

    def _stamp(self, path: Path) -> str:
        stat = path.stat()
        fields = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
        changed = stat.st_mtime_ns
        if os.name != "nt":  # st_ctime is the creation time on Windows
            fields.append(stat.st_ctime_ns)
            changed = max(changed, stat.st_ctime_ns)
        # A last change in the future (a clock that moved) is racy too.
        if _now_ns() - changed < RACY_WINDOW_NS:
            return RACY_STAMP
        return " ".join(str(value) for value in fields)

    def lookup(self, sha256: str, size: int) -> Path | None:
        """The verified cache entry for *sha256*, or None."""
        path = self.path(sha256)
        stamp_file = path.with_name(f"{sha256}.ok")
        try:
            if not path.is_file() or path.stat().st_size != size:
                return None
            stamp = stamp_file.read_text(encoding="utf-8")
            if stamp != RACY_STAMP and stamp == self._stamp(path):
                return path
        except OSError:
            pass
        try:
            if sha256_file(path) != sha256:
                logger.warning("cache entry %s changed on disk; fetching "
                               "it again", sha256)
                path.unlink(missing_ok=True)
                return None
            stamp_file.write_text(self._stamp(path), encoding="utf-8")
        except OSError:
            return None
        return path

    def add(self, sha256: str, chunks: Iterable[bytes],
            cancelled: Callable[[], bool] = lambda: False) -> Path:
        """Write *chunks* as the entry for *sha256*, verifying the hash.
        ``TransferError`` on a mismatch, ``TransferCancelled`` when
        *cancelled* turns true; the partial file is removed either way."""
        import hashlib

        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".{sha256}.{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        try:
            with open(tmp, "wb") as out:
                for chunk in chunks:
                    if cancelled():
                        raise TransferCancelled("cancelled while fetching")
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != sha256:
                raise TransferError(
                    f"the file fetched hashes to {digest.hexdigest()}, not "
                    f"{sha256}: it changed on the control plane after the "
                    "run was assigned; run it again")
            path = self.path(sha256)
            os.replace(tmp, path)
            path.with_name(f"{sha256}.ok").write_text(self._stamp(path),
                                                      encoding="utf-8")
            return path
        finally:
            tmp.unlink(missing_ok=True)


def copy_input(source: Path, target: Path) -> tuple[int, int]:
    """Copy a cached input to the name the run uses; its ``(size,
    mtime_ns)`` there, to tell later whether the run wrote over it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    shutil.copyfile(source, target)
    stat = target.stat()
    return stat.st_size, stat.st_mtime_ns


class TransferClient:
    """HTTP to the control plane for one agent: fetch inputs, upload
    outputs. *base_url* and *token* may be callables, read per request (a
    same-host agent's session token changes with every server start).

    :meth:`session` opens one ``httpx.Client`` (one connection pool) for a
    batch: a run's fetches, or a run's uploads. Without one, a call opens
    its own."""

    #: ``artifact.backend`` of what it uploads.
    backend = "transfer"

    def __init__(self, *, base_url: Any, token: Any, cache: InputCache,
                 timeout_s: float = TIMEOUT_S,
                 ssl_context: Callable[[], Any] | None = None,
                 uds: str | None = None) -> None:
        self._base_url = base_url
        self._token = token
        self.cache = cache
        self.timeout_s = timeout_s
        #: The agent's TLS verification (``agent.trust``, #786), read per
        #: session: an ``ssl.SSLContext`` over https, None over http.
        self._ssl_context = ssl_context
        #: An SSH worker's tunnel (#784): every request goes to this Unix
        #: socket; the base URL still names the control plane.
        self.uds = uds

    def _url(self, path: str) -> str:
        return str(_resolve(self._base_url)).rstrip("/") + path

    def _headers(self) -> dict[str, str]:
        from .connection import TOKEN_HEADER

        token = _resolve(self._token)
        return {TOKEN_HEADER: token} if token else {}

    def _client(self) -> Any:
        import httpx

        context = self._ssl_context() if self._ssl_context else None
        extra: dict[str, Any] = {}
        if self.uds:
            extra["transport"] = httpx.HTTPTransport(uds=self.uds)
        return httpx.Client(timeout=httpx.Timeout(self.timeout_s,
                                                  connect=10.0),
                            follow_redirects=False,
                            verify=context if context is not None else True,
                            **extra)

    def session(self) -> Any:
        """A client for a batch of requests; close it (``with``) after."""
        return self._client()

    def fetch(self, entry: Any, cancelled: Callable[[], bool] = lambda: False,
              *, client: Any = None) -> tuple[Path, bool]:
        """``(cache path, fetched)`` for one input (``wp.TransferInput``):
        from the cache when it holds it, else downloaded and verified."""
        cached = self.cache.lookup(entry.sha256, entry.bytes)
        if cached is not None:
            return cached, False
        route = DOWNLOAD_ROUTES.get(entry.kind)
        if route is None:
            raise TransferError(f"unknown input kind {entry.kind!r}")
        url = self._url(route + quote(safe_relative(entry.name).as_posix(),
                                      safe="/"))
        own = client is None
        http = self._client() if own else client
        try:
            last: Exception | None = None
            for attempt in range(1, ATTEMPTS + 1):
                if cancelled():
                    raise TransferCancelled("cancelled while fetching")
                try:
                    with http.stream("GET", url,
                                     headers=self._headers()) as response:
                        if response.status_code == 200:
                            return self.cache.add(
                                entry.sha256, response.iter_bytes(CHUNK),
                                cancelled), True
                        response.read()
                        error = TransferError(
                            f"fetching {entry.name}: HTTP "
                            f"{response.status_code} {_detail(response)}")
                        if response.status_code < 500:
                            raise error
                        last = error
                except TransferError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- httpx and OS errors
                    last = TransferError(f"fetching {entry.name}: {exc}")
                time.sleep(RETRY_PAUSE_S * attempt)
            raise last or TransferError(f"fetching {entry.name} failed")
        finally:
            if own:
                http.close()

    def upload(self, run_id: str, kind: str, name: str, path: Path, *,
               place: str = PLACE_RUN, client: Any = None,
               cancelled: Callable[[], bool] = lambda: False,
               retry_until: Callable[[], float] | None = None,
               fail_fast: bool = False) -> str:
        """Upload *path* as *run_id*'s *kind*/*name* to *place*; the key the
        control plane stored it under. ``TransferError`` when it was not
        taken: refused (a 4xx but 408/429), a 5xx still failing after
        :data:`ATTEMPTS` quick attempts, an outage (no connection,
        :data:`OUTAGE_STATUSES`) still on after :data:`ATTEMPTS` attempts
        and the retry window (backing off), or *cancelled*.

        The window ends :data:`UPLOAD_RETRY_WINDOW_S` after the first
        outage failure, or at what *retry_until* answers (asked then), so
        a run's uploads share one window. *fail_fast*: that window is
        already spent; one attempt, no pause. Form fields go before the
        file (the control plane streams the body and must know where it
        goes first)."""
        url = self._url(UPLOAD_ROUTE.format(run_id=quote(run_id, safe="")))
        checksum = sha256_file(path)
        own = client is None
        http = self._client() if own else client

        def attempt() -> str:
            try:
                with open(path, "rb") as handle:
                    response = http.post(
                        url, headers=self._headers(),
                        data={"kind": kind, "name": name, "place": place,
                              "sha256": checksum},
                        files={"file": (path.name, handle,
                                        "application/octet-stream")})
            except Exception as exc:  # noqa: BLE001 -- httpx and OS errors
                raise RetryableUpload(TransferError(
                    f"uploading {name}: {exc}"), outage=True) from None
            status = response.status_code
            if status == 200:
                try:
                    return str(response.json()["path"])
                except Exception as exc:  # noqa: BLE001 -- a proxy's page
                    # Not the control plane's answer (#785: retried as an
                    # outage, like any other request that did not reach it).
                    raise RetryableUpload(TransferError(
                        f"uploading {name}: {exc}"), outage=True) from None
            error = TransferError(f"uploading {name}: HTTP {status} "
                                  f"{_detail(response)}")
            if status < 500 and status not in OUTAGE_STATUSES:
                raise error
            raise RetryableUpload(error, outage=status in OUTAGE_STATUSES)

        try:
            return retry_upload(attempt, name, cancelled=cancelled,
                                retry_until=retry_until, fail_fast=fail_fast)
        finally:
            if own:
                http.close()


class RetryableUpload(Exception):
    """One upload attempt that failed in a way worth trying again:
    *outage* (the other end is away: no connection, or
    :data:`OUTAGE_STATUSES`) gets the long window, anything else the
    :data:`ATTEMPTS` quick retries."""

    def __init__(self, error: TransferError, *, outage: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.outage = outage


def retry_upload(attempt: Callable[[], str], name: str, *,
                 cancelled: Callable[[], bool] = lambda: False,
                 retry_until: Callable[[], float] | None = None,
                 fail_fast: bool = False) -> str:
    """Run *attempt* (one upload; the key, ``TransferError`` for a refusal,
    :class:`RetryableUpload` otherwise) until it succeeds, under the retry
    rules ``TransferClient.upload`` documents. Shared by the transfer and
    the S3 backend (#791)."""
    started = time.monotonic()
    deadline: float | None = None
    last: Exception | None = None
    tries = 0
    while True:
        tries += 1
        if cancelled():
            raise TransferError("the uploads were cancelled")
        try:
            return attempt()
        except RetryableUpload as failed:
            last, outage = failed.error, failed.outage
        if fail_fast:
            break
        if not outage:
            if tries >= ATTEMPTS:
                break
            _sleep(RETRY_PAUSE_S * tries, cancelled)
            continue
        if deadline is None:
            deadline = retry_until() if retry_until is not None \
                else started + UPLOAD_RETRY_WINDOW_S
        if tries >= ATTEMPTS and time.monotonic() >= deadline:
            break
        if tries == ATTEMPTS:
            logger.warning("%s; retrying for up to %.0f s more", last,
                           max(0.0, deadline - time.monotonic()))
        pause = min(RETRY_PAUSE_S * 2 ** (tries - 1), UPLOAD_MAX_PAUSE_S)
        _sleep(pause, cancelled)
    raise last or TransferError(f"uploading {name} failed")


def _sleep(seconds: float, cancelled: Callable[[], bool]) -> None:
    """Sleep *seconds*, waking early (within a second) once *cancelled*."""
    end = time.monotonic() + seconds
    while not cancelled():
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 1.0))


def _detail(response: Any) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:  # noqa: BLE001 -- not JSON
        pass
    return (response.text or "")[:200]


#: Folders of the agent's own data directory a transfer run shares instead
#: of starting empty: ``cache/`` holds what nodes rebuild per host
#: (``LMTokenizedDataset``'s token blocks), never a run's output.
SHARED_DIRS = ("cache",)


def run_data_root(run_dir: Path, own_root: Path | None = None) -> Path:
    """The run's own data directory, created with its input folders. With
    *own_root* (this agent's data directory), each of :data:`SHARED_DIRS`
    is a symlink to it, so a cache survives the run; where symlinks are
    not allowed (Windows without the privilege) the run gets its own."""
    root = run_dir / RUN_DATA_DIRNAME
    for sub in ("files", "images", "models", "media"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if own_root is not None:
        for name in SHARED_DIRS:
            link = root / name
            if link.exists() or link.is_symlink():
                continue
            try:
                (own_root / name).mkdir(parents=True, exist_ok=True)
                link.symlink_to(own_root / name, target_is_directory=True)
            except OSError:
                logger.debug("no shared %s/ for %s", name, run_dir.name,
                             exc_info=True)
    return root


def prepare_inputs(client: TransferClient, inputs: list[Any], root: Path,
                   progress: Callable[[str], None],
                   cancelled: Callable[[], bool] = lambda: False
                   ) -> dict[Path, tuple[int, int]]:
    """Fetch (or find in the cache) every input, over one connection, and
    copy it into *root*. *progress* gets one line per input. Returns each
    placed path with its ``(size, mtime_ns)``, which tells the output
    sweep whether the run wrote over it."""
    placed: dict[Path, tuple[int, int]] = {}
    total = len(inputs)
    if not total:
        return placed
    with client.session() as http:
        for index, entry in enumerate(inputs, start=1):
            source, fetched = client.fetch(entry, cancelled, client=http)
            target = root / KIND_DIRS[entry.kind] / safe_relative(entry.name)
            placed[target] = copy_input(source, target)
            how = "fetched" if fetched else "cached"
            progress(f"fetching inputs {index}/{total}: {entry.name} "
                     f"({human_size(entry.bytes)}, {how})")
    return placed


@dataclass
class RunTransfer:
    """One transfer run's outputs, on the agent (its uploader thread's)."""

    run_id: str
    root: Path
    client: TransferClient
    max_upload_bytes: int | None = None
    #: Inputs placed in ``root`` with their ``(size, mtime_ns)`` then: one
    #: whose stat moved was written over by the run, and is an output.
    inputs: dict[Path, tuple[int, int]] = field(default_factory=dict)
    #: Every file uploaded (resolved), so the sweep skips them.
    uploaded: set[Path] = field(default_factory=set)
    #: Names given out, per kind, so two files never share a key.
    names: dict[str, set[str]] = field(default_factory=dict)
    #: Interrupt checkpoints' artifact bodies, uploaded last.
    held: list[dict[str, Any]] = field(default_factory=list)
    #: Set (by a cancel that came after the run ended): upload nothing more.
    stop: Callable[[], bool] = lambda: False
    _http: Any = None
    #: The outage retry window, read when an outage starts: the control
    #: plane's resume grace (``Runner.upload_window_s``, from ``welcome``).
    retry_window_s: Callable[[], float] = lambda: UPLOAD_RETRY_WINDOW_S
    #: When the control plane stopped taking this run's uploads
    #: (monotonic), or None: the run's uploads share one retry window.
    _failing_since: float | None = None
    _window_end: float | None = None

    def _retry_until(self) -> float:
        if self._failing_since is None or self._window_end is None:
            self._failing_since = time.monotonic()
            self._window_end = self._failing_since \
                + max(0.0, float(self.retry_window_s()))
        return self._window_end

    def _window_spent(self) -> bool:
        return self._window_end is not None \
            and time.monotonic() >= self._window_end

    @property
    def backend(self) -> str:
        """``artifact.backend`` of what this run uploads: its client's
        (``s3`` for ``agent.s3.S3Client``, #791), else ``transfer``."""
        return str(getattr(self.client, "backend", "transfer"))

    def _session(self) -> Any:
        if self._http is None:
            self._http = self.client.session()
        return self._http

    def close(self) -> None:
        """Close the run's HTTP client. Idempotent."""
        if self._http is not None:
            try:
                self._http.close()
            finally:
                self._http = None

    def _name(self, kind: str, wanted: str) -> str:
        taken = self.names.setdefault(kind, set())
        name, counter = wanted, 1
        while name in taken:
            counter += 1
            name = f"{counter}-{wanted}"
        taken.add(name)
        return name

    def _upload_file(self, kind: str, name: str, path: Path, *,
                     place: str = PLACE_RUN) -> str:
        if self.stop():
            raise TransferError("the uploads were cancelled")
        size = path.stat().st_size
        if self.max_upload_bytes is not None and size > self.max_upload_bytes:
            raise TransferError(
                f"{path.name} is {human_size(size)}, over the control "
                f"plane's upload limit of "
                f"{human_size(self.max_upload_bytes)} (MAX_UPLOAD_SIZE)")
        key = self.client.upload(self.run_id, kind, name, path, place=place,
                                 client=self._session(), cancelled=self.stop,
                                 retry_until=self._retry_until,
                                 fail_fast=self._window_spent())
        self._failing_since = self._window_end = None
        self.uploaded.add(path.resolve())
        return key

    def upload_artifact(self, body: dict[str, Any]
                        ) -> tuple[dict[str, Any], str | None]:
        """``(artifact body, warning)`` for an ``artifact`` message: the
        body rewritten to the uploaded key, or as it was with a
        ``transfer_failed`` warning's detail."""
        raw = str(body.get("path") or "")
        path = Path(raw)
        kind = str(body.get("kind") or "file")
        try:
            # Only what the run wrote in its own data folder: a row is a
            # claim (``log_artifact`` is plugin API), and a path elsewhere
            # on this machine is not the run's to send.
            if not raw or not self._inside_root(path):
                raise TransferError(f"{raw or '(no path)'} is outside the "
                                    "run's data folder")
            if path.is_dir() and not path.is_symlink():
                files = list(walk_files(path))
                if not files:
                    raise TransferError(f"{path.name}/ is empty")
                base = self._name(kind, path.name)
                key = None
                for file in files:
                    relative = file.relative_to(path).as_posix()
                    uploaded = self._upload_file(kind, f"{base}/{relative}",
                                                 file)
                    depth = len(PurePosixPath(relative).parts)
                    key = PurePosixPath(uploaded).parents[depth - 1] \
                        .as_posix()
            elif path.is_file():
                key = self._upload_file(kind, self._name(kind, path.name),
                                        path)
            else:
                raise TransferError(f"{path} is not on this worker")
        except Exception as exc:  # noqa: BLE001 -- reported, never raised
            logger.warning("run %s: could not upload %s: %s", self.run_id,
                           path, exc)
            return body, (f"{kind} {path.name} stays on worker: {exc}")
        return {**body, "path": key, "backend": self.backend,
                "mount_label": None}, None

    def _inside_root(self, path: Path) -> bool:
        try:
            return path.is_absolute() and path.resolve().is_relative_to(
                self.root.resolve())
        except OSError:
            return False

    def _changed_input(self, path: Path) -> bool:
        before = self.inputs.get(path)
        if before is None:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        return (stat.st_size, stat.st_mtime_ns) != before

    def unreported_outputs(self) -> list[Path]:
        """What the run wrote and nothing has uploaded: every file under
        :data:`OUTPUT_DIRS` that is not an unchanged input, then every input
        (in any folder) the run wrote over."""
        held = {Path(str(body.get("path") or "")).resolve()
                for body in self.held}
        found: list[Path] = []
        for sub in OUTPUT_DIRS:
            base = self.root / sub
            if not base.is_dir() or base.is_symlink():
                continue
            for path in walk_files(base):
                resolved = path.resolve()
                if resolved in self.uploaded or resolved in held:
                    continue
                if path in self.inputs and not self._changed_input(path):
                    continue
                found.append(path)
        for path in sorted(self.inputs):
            if path not in found and self._changed_input(path) \
                    and path.resolve() not in self.uploaded:
                found.append(path)
        return found

    def upload_output(self, path: Path
                      ) -> tuple[dict[str, Any] | None, str | None]:
        """``(artifact body, warning)`` for one unreported output, placed in
        the control plane's own data folders at the same relative path (an
        input written over keeps its folder: ``files/x.csv`` goes to
        ``output/files/x.csv``)."""
        name = path.relative_to(self.root).as_posix()
        placed = name if PurePosixPath(name).parts[0] in OUTPUT_DIRS \
            else f"output/{name}"
        try:
            key = self._upload_file(OUTPUT_KIND, placed, path,
                                    place=PLACE_DATA)
        except Exception as exc:  # noqa: BLE001 -- reported, never raised
            logger.warning("run %s: could not upload %s: %s", self.run_id,
                           path, exc)
            return None, f"output {name} stays on worker: {exc}"
        return {"kind": OUTPUT_KIND, "path": key, "meta": {"name": name},
                "node_id": None, "mount_label": None,
                "backend": self.backend}, None


def is_interrupt_checkpoint(body: dict[str, Any]) -> bool:
    """The interrupt checkpoint a stopped training node reports
    (``loop_control.save_interrupt_checkpoint``): uploaded last."""
    meta = body.get("meta")
    return body.get("kind") == "checkpoint" and isinstance(meta, dict) \
        and meta.get("reason") == "interrupted"

