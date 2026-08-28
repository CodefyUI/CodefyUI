"""Getting a pack's model files onto the disk, visibly and interruptibly.

Two sources, one shape of progress. A Hugging Face item is a snapshot of many
files; an asset item is one file over HTTPS. Both report the same
``{"type": "progress", "item": ..., "bytes_done": ..., "bytes_total": ...,
"percent": ...}`` event, because the person watching does not care which kind
they asked for.

Three things this module is careful about:

* **The total is known before the first byte.** ``model_info(files_metadata=True)``
  gives the size of every file, so the bar starts at 0/470 MB rather than
  counting up from nowhere. It also means the FILTERED list is what the total
  describes -- see :func:`list_hf_files`, which exists because a
  sentence-transformers repo ships the same weights three or four times over
  and downloading all of them costs four times the bytes for one usable model.
* **Stop works mid-file.** ``hf_hub_download`` is one blocking call per file
  and a 400 MB file is minutes of it, so the cancel check lives inside the
  progress hook, where a between-files check would not have been reached
  until the file finished. Getting out from there takes two different
  things on the two transports, and both are needed: raising from
  ``tqdm.update`` unwinds the classic HTTP download, while the hf_xet fast
  path discards anything raised in its callback and has to be aborted
  explicitly -- see :func:`abort_xet_transfer`, and
  ``tests/test_packs_network.py``, which proves it against the real hub.
* **``HF_HOME`` is never touched.** That variable is the whole machine's
  Hugging Face cache, shared with every other tool its owner runs. Packs
  download into CodefyUI's own cache via ``cache_dir=``, so uninstalling a
  pack cannot delete somebody else's models and installing one cannot
  quietly fill their home directory.
"""

from __future__ import annotations

import fnmatch
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from .. import asset_cache
from ..asset_cache import AssetSpec, AssetVerificationError
from . import state
from .catalog import ModelItem, Pack
from .errors import PackCancelled, PackInstallError, PackInsufficientDisk
from .paths import hf_cache_dir

log = logging.getLogger(__name__)

#: Minimum seconds between two progress events for one item. The UI redraws a
#: bar; four frames a second is smooth to a human and is three orders of
#: magnitude fewer events than a 1 MB-chunked download would otherwise send
#: down a WebSocket.
PROGRESS_MIN_INTERVAL_S = 0.25

#: Multiplier over an item's ``approx_bytes`` for the disk precheck. The
#: Hugging Face cache stores each file once as a blob and once more as a
#: snapshot entry -- a real copy on any filesystem without symlinks, which on
#: Windows means most of them -- and the catalog's sizes are approximate.
DISK_HEADROOM = 1.5

#: What a model needs to load: config, tokenizer, weights. Matched against the
#: WHOLE path with ``fnmatch``, whose ``*`` crosses ``/``, so ``*.json`` also
#: keeps ``1_Pooling/config.json``.
_KEEP_PATTERNS = (
    "*.json", "*.txt", "*.safetensors", "*.model", "vocab.*", "merges.txt",
    "*.tiktoken",
    # The other weight formats are kept here and dropped again below when a
    # safetensors copy exists. A repo that never published safetensors still
    # has to be downloadable.
    "*.bin", "*.h5", "*.ot", "*.msgpack", "*.ckpt",
)

#: Exports for runtimes CodefyUI does not use. Whole directories, and each is
#: a full second copy of the weights.
_DROP_PREFIXES = ("onnx/", "openvino/")

#: Weight formats that are redundant once ``*.safetensors`` is present.
_DUPLICATE_WEIGHTS = (".bin", ".h5", ".ot", ".msgpack", ".ckpt")


def _now_iso() -> str:
    """An ISO-8601 UTC timestamp for a sentinel's ``at`` field."""
    return datetime.now(timezone.utc).isoformat()


class _ByteMeter:
    """Bytes moved for one item, turned into throttled ``progress`` events.

    Deliberately NOT ``loop_control.ProgressThrottle``: this needs a FORCED
    emit for the first and last frame (a download that ends between two
    throttle windows would otherwise stop reporting at 97%), and that is the
    one thing ``ProgressThrottle`` has no way to express.

    ``add`` and ``advance_to`` both check for cancellation, which is what
    makes Stop work inside a single multi-hundred-megabyte file: they are
    called from the transfer's own thread, so raising here unwinds the
    download instead of waiting for it to finish.

    ``begin_file`` sets a CEILING, and the ceiling is not decoration: several
    progress bars can be alive for one file at once (huggingface_hub's Xet
    path builds one for reconstruction and one for network transfer), and
    they report the same bytes twice. Whatever they claim, a 400-byte file
    contributes 400 bytes.
    """

    __slots__ = ("_emit", "_item_id", "_cancel_check", "_min_interval",
                 "_last", "_ceiling", "_initial_seen", "total", "done")

    def __init__(
        self,
        *,
        emit: Callable[[dict], None],
        item_id: str,
        total: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
        min_interval_s: float | None = None,
    ) -> None:
        # Read at CONSTRUCTION, not baked into the signature default: a
        # default argument is evaluated at import time, which would make
        # ``PROGRESS_MIN_INTERVAL_S`` unpatchable and every frame this class
        # drops invisible to a test. Same reasoning as
        # ``loop_control.ProgressThrottle``.
        if min_interval_s is None:
            min_interval_s = PROGRESS_MIN_INTERVAL_S
        self._emit = emit
        self._item_id = item_id
        self._cancel_check = cancel_check
        self._min_interval = max(0.0, float(min_interval_s))
        self._last: float | None = None
        self._ceiling: int | None = None
        self._initial_seen = False
        self.total = total if total else None
        self.done = 0

    def begin_file(self, base: int, size: int | None) -> None:
        """Start accounting for one file of *size* bytes, *base* already done.

        With a known size this caps what the file may contribute; with an
        unknown one (the hub does not always report a size) there is no cap,
        because a ceiling of zero would freeze the bar instead of bounding it.

        And never below what has already been counted. A file the hub
        reported as zero bytes ran UNCAPPED and may well have moved the
        meter past the *base* the next file is handed; a ceiling under
        ``done`` would then clamp the bar BACKWARDS, and a progress bar that
        goes down reads as a restart rather than as arithmetic.
        """
        self._ceiling = max(base + size, self.done) if size else None
        self._initial_seen = False

    def note_initial(self, initial: int) -> None:
        """Credit bytes a RESUMED transfer already had on disk.

        tqdm is told about them once, as ``initial=``, and never ``update``s
        them -- so a download that resumes at 300 MB would otherwise report
        300 MB less than it has. Applied once per file: huggingface_hub
        rebuilds the bar on a retry with the same ``initial``.
        """
        if self._initial_seen or initial <= 0:
            return
        self._initial_seen = True
        self.add(initial)

    def _payload(self) -> dict:
        percent = None
        if self.total:
            percent = round(100.0 * self.done / self.total, 1)
        return {"type": "progress", "item": self._item_id,
                "bytes_done": self.done, "bytes_total": self.total,
                "percent": percent}

    def _check_cancelled(self) -> None:
        if self._cancel_check is not None and self._cancel_check():
            raise PackCancelled(f"download of {self._item_id} cancelled")

    def emit_now(self) -> None:
        """Report immediately, whatever the throttle would have said."""
        self._last = time.monotonic()
        self._emit(self._payload())

    def _emit_throttled(self) -> None:
        now = time.monotonic()
        if self._last is not None and now - self._last < self._min_interval:
            return
        self._last = now
        self._emit(self._payload())

    def add(self, n: int) -> None:
        """Account for *n* more bytes (what a tqdm ``update`` reports).

        *n* may be NEGATIVE: huggingface_hub rolls a bar back by
        ``-resume_size`` when a server ignores a Range request and sends the
        file from the start, and treating that as zero would count the
        resumed part twice.
        """
        self._check_cancelled()
        self.done = max(0, self.done + int(n or 0))
        if self._ceiling is not None:
            self.done = min(self.done, self._ceiling)
        self._emit_throttled()

    def advance_to(self, value: int) -> None:
        """Account for an ABSOLUTE byte count, never going backwards.

        Two callers need this. ``asset_cache`` reports a running total rather
        than a delta, and a Hugging Face file that was already cached reports
        nothing at all -- without this the bar would stop at the last byte
        that happened to move over the network.

        Not subject to the per-file ceiling: this is THIS module telling the
        meter what it knows, where ``add`` is a progress bar telling it what
        it claims.
        """
        self._check_cancelled()
        self.done = max(self.done, int(value))
        self._emit_throttled()


#: True once :func:`abort_xet_transfer` has reported that huggingface_hub no
#: longer has the hook it needs. Module state rather than a log filter: the
#: message is worth saying once per process and worth NOT saying on every
#: subsequent cancel.
_warned_missing_xet_abort = False


def abort_xet_transfer() -> None:
    """Cancel whatever the hf_xet runtime is currently downloading.

    Raising from the progress hook is NOT enough on the Xet fast path, and
    this was measured rather than assumed: hf_xet drives that hook from its
    Rust download group, and an exception raised inside it is caught there,
    printed, and DISCARDED. The transfer runs to completion; the
    ``PackCancelled`` is only noticed by the next check in Python, which is
    after the file has finished. Pressing Stop on a 470 MB model would wait
    for all 470 MB -- the exact failure the mid-file check was written to
    prevent -- and neither the exception TYPE nor a ``BaseException`` makes
    any difference: ``KeyboardInterrupt`` is swallowed the same way.

    ``abort_xet_session`` is the escape hatch huggingface_hub uses for its
    own Ctrl-C handling in ``file_download.xet_get``. It is thread-safe and
    it is safe to call from inside the progress callback (both were measured:
    the transfer stops within a few hundredths of a second and leaves no blob
    behind). What comes back OUT of ``hf_hub_download`` afterwards is a Rust
    ``RuntimeError('Operation cancelled: ...')``, which
    :func:`download_hf_item` translates back into ``PackCancelled``.

    The session is process-wide, and that has a cost this deliberately
    accepts rather than avoids. Every hf_xet transfer in this interpreter
    ends here, not only this item's -- and a graph run CAN have one:
    ``TextCorpusDataset`` and ``HuggingFaceDataset`` fetch through
    ``datasets.load_dataset``, and ``Tokenizer`` through
    ``tokenizers.Tokenizer.from_pretrained``. Pressing Stop on a pack
    install while one of those is streaming truncates it too, and that node
    sees the raw ``RuntimeError('Operation cancelled: ...')`` with none of
    the translation :func:`download_hf_item` does for its own call.

    Accepted because the alternative is a Stop button that does nothing to
    the four hundred megabytes it was pressed to stop. The damage is a run
    the user can start again; nothing is corrupted and nothing stays broken,
    because the next call builds a fresh session
    (``tests/test_packs_network.py`` downloads a second model right after
    the cancel to prove a stopped install does not poison the ones after
    it). The abort says so in the log, so a truncated corpus fetch is
    explained rather than mysterious.

    ``abort_xet_session`` is PRIVATE API. Losing it is not fatal -- cancel
    degrades to waiting for the file in flight, which is what it did before
    this existed -- but it is silent, so the loss is logged here once, and
    ``test_installed_huggingface_hub_still_exposes_the_xet_abort_hook``
    fails the build the day a hub upgrade removes it.
    """
    global _warned_missing_xet_abort

    try:
        from huggingface_hub.utils._xet import abort_xet_session
    except Exception as exc:
        if not _warned_missing_xet_abort:
            # Once per process: a server whose user cancels ten downloads
            # should say this once, not ten times.
            _warned_missing_xet_abort = True
            log.warning(
                "huggingface_hub no longer exposes "
                "utils._xet.abort_xet_session (%s), so cancelling a model "
                "download now waits for the file in flight to finish "
                "instead of stopping mid-file. See "
                "app.core.packs.download.abort_xet_transfer.", exc)
        return

    log.warning(
        "Aborting the hf_xet session to stop this download. This is called "
        "on every cancel, including a classic HTTP one with no Xet session "
        "to abort, so it is not a claim that anything was interrupted: what "
        "it means is that any other Hugging Face transfer this process has "
        "in flight -- a graph run streaming a dataset or fetching a "
        "tokenizer -- is interrupted too, if there is one.")
    try:
        abort_xet_session()
    except Exception:
        log.warning("abort_xet_session() failed; the download will run to "
                    "the end of the file in flight", exc_info=True)


def make_tqdm_class(meter: _ByteMeter) -> type:
    """A ``tqdm`` subclass that feeds *meter* and draws nothing.

    ``hf_hub_download`` takes the progress bar CLASS rather than a callback,
    so this is the only hook it offers -- and the hf_xet fast path funnels
    through the same class, which is why the meter sees those transfers too.

    ``disable=True`` is forced in the constructor: huggingface_hub only
    injects its own ``disable`` for subclasses of ITS tqdm, and a bar left
    enabled would write escape sequences into a server log.

    ``update_transfer`` and ``set_transfer_postfix_str`` exist to be FOUND.
    ``XetDownloadProgressReporter`` asks the class it is handed whether it
    can ``update_transfer``; a class that cannot gets TWO bars -- one for
    reconstruction, one for network transfer -- both built from it and both
    updated, which for one shared meter is every byte counted twice and a
    bar that runs to 200%. Answering yes keeps it to a single bar, and the
    network figure it then routes here is dropped: those are the same bytes
    the reconstruction bar has already reported.
    """
    from tqdm.auto import tqdm as _tqdm

    class _MeterTqdm(_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            # huggingface_hub passes ``name=`` to bars it recognises as its
            # own. We are not one, so today it does not -- but vanilla tqdm
            # raises TqdmKeyError on an unknown keyword, and an install that
            # dies on a keyword argument after a hub upgrade would be a very
            # confusing bug report.
            kwargs.pop("name", None)
            super().__init__(*args, **kwargs)
            # huggingface_hub always passes ``initial`` by keyword.
            meter.note_initial(int(kwargs.get("initial") or 0))

        def update(self, n=1):
            # Before ``super()``: a disabled tqdm returns early without
            # touching ``n``, so this is the only place the bytes exist.
            try:
                meter.add(n or 0)
            except PackCancelled:
                # The raise alone stops the classic HTTP path. The Xet path
                # discards it, so the transfer has to be told separately --
                # see :func:`abort_xet_transfer`. Still raised afterwards,
                # because on the path that DOES honour it that is what
                # unwinds the download.
                abort_xet_transfer()
                raise
            return super().update(n)

        def update_transfer(self, n=1, *args, **kwargs):
            """Network bytes for the Xet path. Counted by ``update``
            already, so this only has to EXIST -- see the class docstring."""
            return None

        def set_transfer_postfix_str(self, *args, **kwargs):
            """The postfix that goes with ``update_transfer``. Nothing is
            drawn, but the reporter calls it whenever it calls that."""
            return None

    return _MeterTqdm


def list_hf_files(repo_id: str, revision: str) -> list[tuple[str, int]]:
    """``[(path, size)]`` for the files of *repo_id* worth downloading.

    Sizes come from ``files_metadata=True`` and are 0 when the hub does not
    report one -- "unknown", which the caller turns into a bar with no total
    rather than a wrong one.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)

    kept: list[tuple[str, int]] = []
    for sibling in getattr(info, "siblings", None) or []:
        path = getattr(sibling, "rfilename", None)
        if not path:
            continue
        lowered = path.lower()
        if lowered.startswith(_DROP_PREFIXES):
            continue
        if not any(fnmatch.fnmatch(lowered, pattern)
                   for pattern in _KEEP_PATTERNS):
            continue
        kept.append((path, int(getattr(sibling, "size", 0) or 0)))

    if any(path.lower().endswith(".safetensors") for path, _ in kept):
        kept = [(path, size) for path, size in kept
                if not path.lower().endswith(_DUPLICATE_WEIGHTS)]
    return kept


def _snapshot_root(returned: Path) -> Path:
    """The snapshot directory holding *returned*.

    ``hf_hub_download`` answers with the path of one FILE, which for
    ``1_Pooling/config.json`` is two levels below the directory a model is
    loaded from. The layout is ``<cache>/models--org--name/snapshots/<rev>/...``,
    so the snapshot root is the ancestor whose own parent is ``snapshots``.
    """
    directory = returned.parent
    for ancestor in [directory, *directory.parents]:
        if ancestor.parent.name == "snapshots":
            return ancestor
    return directory


def download_hf_item(
    pack: Pack,
    item: ModelItem,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
) -> Path:
    """Download one Hugging Face item; returns its snapshot directory.

    The sentinel is written LAST, once every file has landed: an interrupted
    snapshot looks exactly like a finished one on disk, so the sentinel is
    the only honest record that the download completed.
    """
    from huggingface_hub import hf_hub_download

    files = list_hf_files(item.repo_id, item.revision)
    if not files:
        raise PackInstallError(
            f"{item.repo_id} has no files this installer knows how to use",
            hint=f"pack={pack.pack_id} item={item.item_id}")

    total = sum(size for _, size in files)
    meter = _ByteMeter(emit=emit, item_id=item.item_id,
                       total=total or item.approx_bytes,
                       cancel_check=cancel_check)
    meter.emit_now()

    cache = hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    tqdm_class = make_tqdm_class(meter)

    snapshot: Path | None = None
    accounted = 0
    for path, size in files:
        if cancel_check():
            raise PackCancelled(f"download of {item.item_id} cancelled")
        # Everything the bars for THIS file report is capped at its size.
        meter.begin_file(accounted, size)
        try:
            returned = hf_hub_download(
                repo_id=item.repo_id,
                filename=path,
                revision=item.revision,
                cache_dir=str(cache),
                tqdm_class=tqdm_class,
            )
        except PackCancelled:
            raise
        except Exception as exc:
            # A cancelled Xet transfer comes back as the Rust runtime's own
            # ``RuntimeError('Operation cancelled: ...')`` -- see
            # :func:`abort_xet_transfer`. Deciding on ``cancel_check()``
            # rather than on the message means no error TEXT has to be
            # matched, and a genuine failure that happens to land while Stop
            # is held is reported as what the user asked for. Only the middle
            # of the hierarchy is caught: a real Ctrl-C stays a
            # ``KeyboardInterrupt``.
            if cancel_check():
                raise PackCancelled(
                    f"download of {item.item_id} cancelled") from exc
            raise
        accounted += size
        # A file already in the cache transfers nothing and so reports
        # nothing; count it here or the bar never reaches the end.
        meter.advance_to(accounted)
        if snapshot is None:
            snapshot = _snapshot_root(Path(returned))

    meter.advance_to(total or meter.done)
    meter.emit_now()

    if snapshot is None:
        # Unreachable while `files` is non-empty, which is checked above --
        # but a sentinel written with no directory behind it is the one
        # failure ``state`` cannot detect, so it is worth a guard.
        raise PackInstallError(
            f"{item.repo_id} downloaded no files",
            hint=f"pack={pack.pack_id} item={item.item_id}")

    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1,
        "pack_id": pack.pack_id,
        "item_id": item.item_id,
        "kind": "hf",
        "repo_id": item.repo_id,
        "revision": item.revision,
        "snapshot_dir": str(snapshot),
        "bytes": total or meter.done,
        "at": _now_iso(),
    })
    return snapshot


def download_asset_item(
    pack: Pack,
    item: ModelItem,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
) -> Path:
    """Download one single-file asset; returns the file.

    An item whose catalog entry has no ``sha256`` is UNVERIFIED, and this
    says so twice: once to ``asset_cache.resolve``, which refuses to fetch
    such a spec without the explicit opt-in, and once in the log, which
    prints the digest that was computed so a maintainer can record it in the
    catalog and make the next install a verified one.
    """
    meter = _ByteMeter(emit=emit, item_id=item.item_id,
                       total=item.approx_bytes or None,
                       cancel_check=cancel_check)
    meter.emit_now()

    def _progress(done: int, total: int | None) -> None:
        if total:
            meter.total = total
        meter.advance_to(done)

    spec = AssetSpec(name=item.filename, url=item.url, sha256=item.sha256)
    try:
        path = asset_cache.resolve(spec, progress_callback=_progress,
                                   allow_unverified=item.sha256 is None)
    except AssetVerificationError as exc:
        raise PackInstallError(
            f"{item.item_id} failed sha256 verification and was discarded",
            hint=str(exc)) from exc

    size = path.stat().st_size
    meter.total = size
    meter.advance_to(size)
    meter.emit_now()

    digest = item.sha256
    if digest is None:
        digest = asset_cache.sha256_of(path)
        emit({"type": "log", "line": (
            f"sha256 {digest} ({item.filename}) -- the catalog records no "
            f"digest for this file, so it was NOT VERIFIED")})

    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1,
        "pack_id": pack.pack_id,
        "item_id": item.item_id,
        "kind": "asset",
        "url": item.url,
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "at": _now_iso(),
    })
    return path


def _measurable_dir(path: Path) -> Path:
    """The nearest ancestor of *path* that exists.

    ``shutil.disk_usage`` raises on a path that is not there, and the pack
    cache does not exist until the first install -- which would make "we
    could not measure" read as "no space" on a machine with plenty.
    """
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return Path.cwd()


def check_disk(items: Iterable[ModelItem]) -> None:
    """Refuse before downloading when the disk cannot hold *items*.

    Raises :class:`PackInsufficientDisk`, which carries the two numbers so
    the UI can say how much short it is. Finding out at 90% of a 470 MB
    download that the disk was always too small wastes the download and
    leaves a half-written cache behind.
    """
    approx = sum(item.approx_bytes for item in items)
    if approx <= 0:
        return

    needed = int(DISK_HEADROOM * approx)
    free = shutil.disk_usage(_measurable_dir(hf_cache_dir())).free
    if free < needed:
        raise PackInsufficientDisk(
            f"not enough free disk space: {needed // 1_000_000} MB needed, "
            f"{free // 1_000_000} MB free",
            needed=needed, free=free)
