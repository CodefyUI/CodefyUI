"""Lazy-download cache for large LLM assets (word vectors, tokenizer files, …).

Cached files live under the user's per-OS cache directory (via ``platformdirs``):

* Windows : ``%LOCALAPPDATA%\\codefyui\\Cache``
* macOS   : ``~/Library/Caches/codefyui``
* Linux   : ``~/.cache/codefyui``

The first request for a missing asset downloads it over HTTPS and verifies the
sha256 before returning. Subsequent requests hit the cache. We deliberately
do NOT bundle large binaries inside the Python package — that would bloat
``pip install`` and the source tree — but each asset can declare a small
in-Python fallback so canonical demos still work air-gapped.

This module only handles the *delivery* of asset bytes; node code is
responsible for parsing whatever it gets back.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

log = logging.getLogger(__name__)

_APP_NAME = "codefyui"


@dataclass(frozen=True)
class AssetSpec:
    """Describes one downloadable asset.

    ``url`` should serve a stable, content-addressed binary (e.g. a GitHub
    Release asset). ``sha256`` lets us detect corruption / tampering in the
    cache before handing the bytes back to a node.

    ``sha256=None`` means the digest has NOT BEEN RECORDED YET -- it never
    means "no check needed". :func:`resolve` refuses such a spec unless the
    caller passes ``allow_unverified=True``, so that accepting unverified
    bytes is a decision somebody wrote down rather than a check that quietly
    did not happen.
    """

    name: str
    url: str
    sha256: str | None


def cache_dir() -> Path:
    """Return (and create) the codefyui asset cache directory.

    Honors ``CODEFYUI_USER_DATA_DIR`` so a dev clone keeps its downloaded
    assets in ``.codefyui_dev/cache/`` rather than the OS-wide cache,
    matching the same dev-mode isolation applied to the plugin lockfile
    and session token.
    """
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    if override:
        p = Path(override) / "cache"
    else:
        p = Path(user_cache_dir(_APP_NAME, appauthor=False))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_of(path: Path) -> str:
    """The sha256 of a file, read in 1 MB chunks.

    Public because the Package Center computes the digest of an asset the
    catalog has no digest for yet, so a maintainer can record it.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AssetMissingError(RuntimeError):
    """Raised when an asset isn't cached and the caller asked us not to fetch."""


class AssetVerificationError(RuntimeError):
    """Raised when a cached or downloaded file fails sha256 verification."""


def _content_length(resp) -> int | None:
    """The declared size of *resp*, or None when the server did not say.

    "Unknown" is a perfectly good thing for a progress bar to be told, and
    every failure mode here — no ``headers``, no ``Content-Length``, a
    non-numeric one, a chunked response that reports 0 — means exactly that.
    """
    try:
        raw = resp.headers.get("Content-Length")
    except AttributeError:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve(
    spec: AssetSpec,
    *,
    allow_fetch: bool = True,
    timeout: float = 30.0,
    progress_callback: Callable[[int, int | None], None] | None = None,
    allow_unverified: bool = False,
) -> Path:
    """Return the local cached path for ``spec``, downloading on first use.

    ``allow_fetch=False`` makes this a strict lookup: if the asset isn't
    already in the cache it raises ``AssetMissingError`` rather than
    contacting the network — useful for air-gapped tests.

    ``progress_callback(bytes_done, bytes_total)`` is called after every 1 MB
    chunk and once more when the last one lands, with ``bytes_total`` taken
    from ``Content-Length`` (``None`` when the server did not declare one).
    The default — no callback — leaves the download path byte-for-byte as it
    was. Raising from the callback aborts the download, which is how a
    cancelled install stops one mid-flight.

    ``allow_unverified=True`` is the caller's explicit acceptance that
    ``spec.sha256`` is None and therefore that NOTHING about these bytes can
    be checked. Without it such a spec is refused before a single byte is
    fetched: an asset whose digest nobody has recorded yet must not sail
    through the same code path as one that was verified.
    """
    if spec.sha256 is None and not allow_unverified:
        raise AssetVerificationError(
            f"asset {spec.name!r} has no sha256 recorded, so it cannot be "
            f"verified; pass allow_unverified=True to accept that"
        )

    target = cache_dir() / spec.name

    if target.exists():
        if spec.sha256 is None:
            # Nothing to compare against, and re-downloading would not make
            # the bytes any more trustworthy than the ones already here.
            return target
        actual = sha256_of(target)
        if actual == spec.sha256:
            return target
        log.warning(
            "Cached asset %s sha256 mismatch (%s != expected %s); refetching",
            spec.name,
            actual,
            spec.sha256,
        )
        target.unlink(missing_ok=True)

    if not allow_fetch:
        raise AssetMissingError(
            f"asset {spec.name!r} not present in cache and allow_fetch=False"
        )

    log.info("Downloading %s from %s", spec.name, spec.url)
    tmp = target.with_suffix(target.suffix + ".part")
    # A download that does not finish takes its own temp file with it. The
    # rename below is the only thing that would ever have named these bytes,
    # so an abandoned .part is pure residue -- nothing resumes it (the open
    # is "wb", which truncates), nothing counts it in the disk precheck, and
    # nothing removes it: the pack remover only knows the item's own
    # filename. For GloVe that is 69 MB a user can never find.
    #
    # BaseException, not Exception: a cancelled install raises PackCancelled
    # out of the progress callback (an Exception) and a Ctrl-C raises
    # KeyboardInterrupt (not one), and both abandon the same file.
    try:
        with urllib.request.urlopen(spec.url, timeout=timeout) as resp:  # noqa: S310 — fixed URL set
            total = _content_length(resp)
            done = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress_callback is not None:
                        progress_callback(done, total)
        if progress_callback is not None:
            progress_callback(done, total)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_error:
            # The cleanup must never become the failure that gets reported.
            # On Windows this unlink raises PermissionError whenever another
            # process (an indexer, a scanner) still holds the .part open, and
            # an unguarded one would travel out of here in place of the
            # PackCancelled it is tidying up after -- turning a Stop the user
            # asked for into a failed job. The leftover gets a log line and
            # the original exception gets the raise.
            log.warning("Could not remove partial download %s: %s",
                        tmp, cleanup_error)
        raise

    if spec.sha256 is not None:
        actual = sha256_of(tmp)
        if actual != spec.sha256:
            tmp.unlink(missing_ok=True)
            raise AssetVerificationError(
                f"sha256 mismatch for {spec.name!r}: got {actual}, expected {spec.sha256}"
            )

    tmp.replace(target)
    return target
