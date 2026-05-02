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
import urllib.request
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
    """

    name: str
    url: str
    sha256: str


def cache_dir() -> Path:
    """Return (and create) the codefyui asset cache directory."""
    p = Path(user_cache_dir(_APP_NAME, appauthor=False))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AssetMissingError(RuntimeError):
    """Raised when an asset isn't cached and the caller asked us not to fetch."""


class AssetVerificationError(RuntimeError):
    """Raised when a cached or downloaded file fails sha256 verification."""


def resolve(spec: AssetSpec, *, allow_fetch: bool = True, timeout: float = 30.0) -> Path:
    """Return the local cached path for ``spec``, downloading on first use.

    ``allow_fetch=False`` makes this a strict lookup: if the asset isn't
    already in the cache it raises ``AssetMissingError`` rather than
    contacting the network — useful for air-gapped tests.
    """
    target = cache_dir() / spec.name

    if target.exists():
        actual = _sha256_of(target)
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
    with urllib.request.urlopen(spec.url, timeout=timeout) as resp:  # noqa: S310 — fixed URL set
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    actual = _sha256_of(tmp)
    if actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise AssetVerificationError(
            f"sha256 mismatch for {spec.name!r}: got {actual}, expected {spec.sha256}"
        )

    tmp.replace(target)
    return target
