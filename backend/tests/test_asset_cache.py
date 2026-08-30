"""The asset cache: bytes in, verified file out -- and now a progress report.

Three properties are pinned here and nothing else in the suite pins them:

* a caller can watch a download byte by byte (the Package Center draws a bar
  from it), and passing no callback leaves the old code path untouched;
* a spec with NO recorded sha256 is UNVERIFIED, not verified. The GloVe table
  ships with ``sha256=None`` until a maintainer records the real digest, and
  the difference between "we could not check" and "we checked" must be an
  explicit decision at the call site rather than a silently skipped ``if``;
* cleaning up the abandoned ``.part`` cannot change what failed. The unlink
  runs inside the handler for the exception it is tidying up after, so an
  unlucky one of its own must not take that exception's place.

Never touches the network: ``urllib.request.urlopen`` is replaced with a fake
that serves bytes from memory.
"""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
from pathlib import Path

import pytest

from app.core.asset_cache import (
    AssetMissingError,
    AssetSpec,
    AssetVerificationError,
    cache_dir,
    resolve,
)

PAYLOAD = b"x" * ((1 << 20) * 2 + 512)  # two full 1 MB chunks and a short one
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _FakeResponse:
    """Just enough of an ``http.client.HTTPResponse`` for ``resolve``."""

    def __init__(self, payload: bytes, *, content_length: bool = True):
        self._buffer = io.BytesIO(payload)
        self.headers = (
            {"Content-Length": str(len(payload))} if content_length else {})

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def cache_root(tmp_path, monkeypatch):
    """A throwaway cache root, so a test never reads the developer's real one."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    return tmp_path / "cache"


@pytest.fixture
def served(monkeypatch):
    """Serve *payload* from memory for every ``urlopen``; records the URLs."""
    calls: list[str] = []

    def _serve(payload: bytes, *, content_length: bool = True) -> list[str]:
        def _urlopen(url, timeout=None):
            calls.append(url)
            return _FakeResponse(payload, content_length=content_length)

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        return calls

    return _serve


def _spec(sha256: str | None = DIGEST) -> AssetSpec:
    return AssetSpec(name="thing.bin", url="https://example.invalid/thing.bin",
                     sha256=sha256)


def test_asset_cache_progress_callback_invoked(served):
    """Every chunk reports, the totals come from Content-Length, and the last
    call says the whole file arrived."""
    served(PAYLOAD)
    seen: list[tuple[int, int | None]] = []

    path = resolve(_spec(), progress_callback=lambda done, total: seen.append(
        (done, total)))

    assert path.read_bytes() == PAYLOAD
    assert seen, "no progress was reported"
    assert [total for _, total in seen] == [len(PAYLOAD)] * len(seen)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert seen[0][0] == 1 << 20
    assert seen[-1][0] == len(PAYLOAD)


def test_asset_cache_progress_total_is_none_without_content_length(served):
    """A server that does not declare a length gives ``None``, not a guess."""
    served(PAYLOAD, content_length=False)
    seen: list[tuple[int, int | None]] = []

    resolve(_spec(), progress_callback=lambda done, total: seen.append(
        (done, total)))

    assert seen
    assert all(total is None for _, total in seen)
    assert seen[-1][0] == len(PAYLOAD)


def test_asset_cache_progress_reports_an_empty_response_once(served):
    """A zero-byte body produces no chunks and so no per-chunk report. The
    final call is the one that says "this finished" -- without it a caller's
    bar sits at "starting" for a download that already ended."""
    empty = b""
    served(empty)
    seen: list[tuple[int, int | None]] = []

    resolve(_spec(hashlib.sha256(empty).hexdigest()),
            progress_callback=lambda done, total: seen.append((done, total)))

    assert seen == [(0, None)]


def test_asset_cache_without_progress_callback_still_downloads(served):
    """The default keeps today's behaviour: no callback, same file."""
    served(PAYLOAD)

    path = resolve(_spec())

    assert path == cache_dir() / "thing.bin"
    assert path.read_bytes() == PAYLOAD


def test_asset_cache_verifies_recorded_digest(served):
    """A recorded digest that does not match the bytes is still a failure."""
    served(b"different bytes")

    with pytest.raises(AssetVerificationError):
        resolve(_spec())

    assert not (cache_dir() / "thing.bin").exists()


def test_asset_cache_unrecorded_digest_needs_an_opt_in(served):
    """``sha256=None`` means UNVERIFIED. Without the opt-in nothing is fetched
    at all -- the caller has to say, in writing, that it accepts that."""
    calls = served(PAYLOAD)

    with pytest.raises(AssetVerificationError, match="no sha256"):
        resolve(_spec(sha256=None))

    assert calls == [], "an unverifiable asset was downloaded anyway"


def test_asset_cache_allow_unverified_downloads_and_skips_the_check(served):
    """With the opt-in the bytes land and nothing pretends to have checked them."""
    served(PAYLOAD)

    calls = served(PAYLOAD)
    path = resolve(_spec(sha256=None), allow_unverified=True)

    assert path.read_bytes() == PAYLOAD
    assert calls == ["https://example.invalid/thing.bin"]

    # And a second call is a cache hit rather than a mismatch-and-refetch:
    # with no digest to compare against there is nothing to re-fetch FOR.
    calls.clear()
    assert resolve(_spec(sha256=None), allow_unverified=True) == path
    assert calls == []


def test_asset_cache_allow_fetch_false_still_refuses_to_download(served):
    """The strict-lookup path is unchanged by either new argument."""
    served(PAYLOAD)

    with pytest.raises(AssetMissingError):
        resolve(_spec(), allow_fetch=False, progress_callback=lambda *_: None)


def test_a_failed_cleanup_unlink_keeps_the_original_exception(served,
                                                              monkeypatch,
                                                              caplog):
    """Deleting the abandoned ``.part`` cannot become the reported failure.

    On Windows an ``unlink`` of a file another process holds open without
    ``FILE_SHARE_DELETE`` -- an indexer, a backup agent, a scanner that
    opened the growing file -- raises ``PermissionError``. That unlink runs
    inside the ``except`` cleaning up after a cancel, so an unguarded one
    propagates INSTEAD OF the ``PackCancelled`` (or ``KeyboardInterrupt``)
    that was travelling: the user presses Stop and the job is reported
    *failed* with ``[WinError 32]``, having also failed to delete anything.
    The leftover is worth a warning naming it and nothing more.
    """
    served(PAYLOAD)
    real_unlink = Path.unlink

    def _unlink(self, *args, **kwargs):
        if self.suffix == ".part":
            raise PermissionError(32, "another process is using this file")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink)

    class _Stopped(Exception):
        """Stands in for the ``PackCancelled`` a real Stop raises."""

    def _stop(done, total):
        raise _Stopped("the user pressed Stop")

    with caplog.at_level(logging.WARNING, logger="app.core.asset_cache"):
        with pytest.raises(_Stopped):
            resolve(_spec(), progress_callback=_stop)

    assert any("thing.bin.part" in record.getMessage()
               for record in caplog.records), caplog.text
