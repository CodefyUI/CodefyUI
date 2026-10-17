"""``app.agent.transfer``: the agent's half of the transfer backend (#785).

The input cache (by sha256, verified, re-checked when its stat moves), the
HTTP client against a mock transport, the run's outputs, and the Runner
driving a whole transfer run with a cheap child: inputs fetched before the
subprocess starts, outputs uploaded as they are reported, the unreported
ones after it exits, the interrupt checkpoint last, every ``seq`` in order.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent import transfer as tr
from app.agent.runner import Runner
from app.core import worker_protocol as wp

from . import _helpers as h


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(name: str, data: bytes, kind: str = "data_file"
           ) -> wp.TransferInput:
    return wp.TransferInput(kind=kind, name=name, sha256=_sha(data),
                            bytes=len(data))


# ── the cache ─────────────────────────────────────────────────────────────


def test_a_cache_entry_is_verified_and_found_again(tmp_path):
    cache = tr.InputCache(tmp_path / "cache")
    data = b"a,b\n1,2\n"
    assert cache.lookup(_sha(data), len(data)) is None
    path = cache.add(_sha(data), [data[:3], data[3:]])
    assert path == tmp_path / "cache" / _sha(data)
    assert path.read_bytes() == data
    assert cache.lookup(_sha(data), len(data)) == path
    # The wrong size is a miss, not a match.
    assert cache.lookup(_sha(data), len(data) + 1) is None


def test_a_mismatched_download_is_refused_and_leaves_nothing(tmp_path):
    cache = tr.InputCache(tmp_path / "cache")
    with pytest.raises(tr.TransferError, match="changed on the control"):
        cache.add(_sha(b"expected"), [b"something else"])
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_cancelled_download_leaves_nothing(tmp_path):
    cache = tr.InputCache(tmp_path / "cache")
    with pytest.raises(tr.TransferCancelled):
        cache.add(_sha(b"x"), [b"x"], cancelled=lambda: True)
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_run_gets_a_copy_that_cannot_reach_the_cache(tmp_path):
    """A node that writes over its input writes over the run's copy; the
    cache keeps the original."""
    cache = tr.InputCache(tmp_path / "cache")
    data = b"original"
    path = cache.add(_sha(data), [data])
    copied = tmp_path / "run" / "files" / "in.csv"
    stat = tr.copy_input(path, copied)
    assert stat == (len(data), copied.stat().st_mtime_ns)
    copied.write_bytes(b"modified")
    assert path.read_bytes() == data
    assert cache.lookup(_sha(data), len(data)) == path


def test_an_entry_edited_in_the_cache_is_rehashed_and_dropped(tmp_path):
    cache = tr.InputCache(tmp_path / "cache")
    data = b"original"
    path = cache.add(_sha(data), [data])
    before = path.stat()
    path.write_bytes(b"modified")
    # Within one tick of a coarse file clock (Windows: ~16 ms) the edit
    # leaves size and mtime as they were; the entry, stamped that close to
    # its write, is hashed again all the same.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cache.lookup(_sha(data), len(data)) is None
    assert not path.exists()
    # Touched but intact: hashed again, kept, re-stamped.
    path = cache.add(_sha(data), [data])
    path.with_name(f"{_sha(data)}.ok").write_text("stale", encoding="utf-8")
    assert cache.lookup(_sha(data), len(data)) == path
    assert cache.lookup(_sha(data), len(data)) == path


def _later(monkeypatch) -> None:
    """The cache's clock, well past the racy window of anything written so
    far (a ctime cannot be dated back, so the clock moves instead)."""
    real = tr._now_ns
    monkeypatch.setattr(tr, "_now_ns",
                        lambda: real() + 10 * tr.RACY_WINDOW_NS)


def test_an_entry_stamped_right_after_its_write_is_hashed_again(
        tmp_path, monkeypatch):
    cache = tr.InputCache(tmp_path / "cache")
    data = b"original"
    path = cache.add(_sha(data), [data])
    stamp = path.with_name(f"{_sha(data)}.ok")
    assert stamp.read_text(encoding="utf-8") == tr.RACY_STAMP
    hashed: list[Path] = []
    real = tr.sha256_file
    monkeypatch.setattr(tr, "sha256_file",
                        lambda p: hashed.append(p) or real(p))
    # Still within the window: hashed on every lookup, never trusted.
    assert cache.lookup(_sha(data), len(data)) == path
    assert stamp.read_text(encoding="utf-8") == tr.RACY_STAMP
    assert hashed == [path]
    _later(monkeypatch)
    assert cache.lookup(_sha(data), len(data)) == path
    assert hashed == [path, path]
    # Re-stamped once settled: trusted from now on, not hashed again.
    assert stamp.read_text(encoding="utf-8") != tr.RACY_STAMP
    assert cache.lookup(_sha(data), len(data)) == path
    assert hashed == [path, path]


def test_a_last_change_in_the_future_is_racy(tmp_path):
    cache = tr.InputCache(tmp_path / "cache")
    data = b"original"
    path = cache.add(_sha(data), [data])
    future = tr._now_ns() + 10 * tr.RACY_WINDOW_NS
    os.utime(path, ns=(future, future))
    assert cache.lookup(_sha(data), len(data)) == path
    assert path.with_name(f"{_sha(data)}.ok").read_text(
        encoding="utf-8") == tr.RACY_STAMP


@pytest.mark.skipif(os.name == "nt", reason="no ctime on Windows")
def test_a_settled_entry_edited_with_its_mtime_put_back_is_dropped(
        tmp_path, monkeypatch):
    cache = tr.InputCache(tmp_path / "cache")
    data = b"original"
    path = cache.add(_sha(data), [data])
    _later(monkeypatch)
    assert cache.lookup(_sha(data), len(data)) == path  # trusted stamp
    assert path.with_name(f"{_sha(data)}.ok").read_text(
        encoding="utf-8") != tr.RACY_STAMP
    before = path.stat()
    # A later edit: past the file clock's tick (Linux: a few ms), as any
    # edit after a trusted stamp is (the stamp is 3 s past the last change).
    time.sleep(0.05)
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cache.lookup(_sha(data), len(data)) is None
    assert not path.exists()


def test_walk_files_never_follows_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"s")
    base = tmp_path / "base"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "a").write_bytes(b"a")
    (base / "b").write_bytes(b"b")
    (base / "link-dir").symlink_to(outside, target_is_directory=True)
    (base / "link-file").symlink_to(outside / "secret")
    assert [p.relative_to(base).as_posix() for p in tr.walk_files(base)] == [
        "b", "sub/a"]


@pytest.mark.parametrize("name", ["", "/etc/passwd", "../x", "a/../../x",
                                  "a/.."])
def test_an_input_name_must_stay_where_it_is_put(name):
    with pytest.raises(tr.TransferError):
        tr.safe_relative(name)


# ── the HTTP client ───────────────────────────────────────────────────────


class _Server:
    """A mock control plane: download routes and the upload route."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[httpx.Request] = []
        self.uploads: dict[str, bytes] = {}
        self.fail_next: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0),
                                  json={"detail": "nope"})
        path = request.url.path
        if request.method == "GET":
            for route in tr.DOWNLOAD_ROUTES.values():
                if path.startswith(route):
                    name = path[len(route):]
                    if name in self.files:
                        return httpx.Response(200, content=self.files[name])
            return httpx.Response(404, json={"detail": "File not found"})
        body = request.read()
        boundary = request.headers["content-type"].split("boundary=")[1]
        fields: dict[str, bytes] = {}
        for part in body.split(b"--" + boundary.encode()):
            if b"\r\n\r\n" not in part:
                continue
            head, _, value = part.partition(b"\r\n\r\n")
            name = head.split(b'name="')[1].split(b'"')[0].decode()
            fields[name] = value.rstrip(b"\r\n")
        run_id = path.split("/")[3]
        assert list(fields)[-1] == "file"  # the fields come first
        if fields["place"] == b"data":
            key = fields["name"].decode()
        else:
            key = f"runs/{run_id}/{fields['kind'].decode()}/" \
                  f"{fields['name'].decode()}"
        assert _sha(fields["file"]) == fields["sha256"].decode()
        self.uploads[key] = fields["file"]
        return httpx.Response(200, json={"path": key,
                                         "bytes": len(fields["file"])})


def _client(tmp_path: Path, server: _Server, monkeypatch) -> tr.TransferClient:
    monkeypatch.setattr(tr, "RETRY_PAUSE_S", 0.0)
    monkeypatch.setattr(tr, "UPLOAD_RETRY_WINDOW_S", 0.0)
    client = tr.TransferClient(base_url=lambda: "http://cp:8000/",
                               token=lambda: "worker-token",
                               cache=tr.InputCache(tmp_path / "cache"))
    monkeypatch.setattr(client, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(server)))
    return client


def test_fetch_downloads_once_with_the_token_then_uses_the_cache(
        tmp_path, monkeypatch):
    data = b"a,b\n1,2\n"
    server = _Server({"train.csv": data})
    client = _client(tmp_path, server, monkeypatch)
    path, fetched = client.fetch(_entry("train.csv", data))
    assert fetched and path.read_bytes() == data
    [request] = server.requests
    assert str(request.url) == "http://cp:8000/api/files/download/train.csv"
    assert request.headers["X-CodefyUI-Token"] == "worker-token"
    path, fetched = client.fetch(_entry("train.csv", data))
    assert not fetched and len(server.requests) == 1


def test_fetch_quotes_a_nested_model_name(tmp_path, monkeypatch):
    data = b"weights"
    server = _Server({"periodic/a b.pt": data})
    client = _client(tmp_path, server, monkeypatch)
    client.fetch(_entry("periodic/a b.pt", data, kind="model"))
    assert server.requests[0].url.raw_path == \
        b"/api/models/download/periodic/a%20b.pt"


def test_fetch_retries_a_server_error_but_not_a_refusal(tmp_path,
                                                        monkeypatch):
    data = b"x"
    server = _Server({"x.csv": data})
    client = _client(tmp_path, server, monkeypatch)
    server.fail_next = [503]
    assert client.fetch(_entry("x.csv", data))[1] is True
    assert len(server.requests) == 2
    with pytest.raises(tr.TransferError, match="HTTP 404 File not found"):
        client.fetch(_entry("gone.csv", b"y"))
    server.fail_next = [500, 500, 500]
    with pytest.raises(tr.TransferError, match="HTTP 500"):
        client.fetch(_entry("again.csv", b"z"))


def test_fetch_reports_a_dead_control_plane(tmp_path, monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(tmp_path, _Server({}), monkeypatch)
    monkeypatch.setattr(client, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(down)))
    with pytest.raises(tr.TransferError, match="refused"):
        client.fetch(_entry("x.csv", b"x"))


def test_fetch_refuses_an_unknown_kind(tmp_path, monkeypatch):
    client = _client(tmp_path, _Server({}), monkeypatch)
    entry = _entry("x.csv", b"x").model_copy(update={"kind": "video"})
    with pytest.raises(tr.TransferError, match="unknown input kind"):
        client.fetch(entry)


def test_upload_answers_the_key_and_refuses_a_413(tmp_path, monkeypatch):
    server = _Server({})
    client = _client(tmp_path, server, monkeypatch)
    source = tmp_path / "ck.pt"
    source.write_bytes(b"weights")
    key = client.upload("run-1", "checkpoint", "ck.pt", source)
    assert key == "runs/run-1/checkpoint/ck.pt"
    assert server.uploads[key] == b"weights"
    assert server.requests[-1].headers["X-CodefyUI-Token"] == "worker-token"
    server.fail_next = [413]
    with pytest.raises(tr.TransferError, match="HTTP 413"):
        client.upload("run-1", "checkpoint", "ck.pt", source)
    server.fail_next = [502, 502, 502]
    with pytest.raises(tr.TransferError, match="HTTP 502"):
        client.upload("run-1", "checkpoint", "ck.pt", source)


class _Clock:
    """``time`` for ``agent.transfer``: sleeping advances a fake clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _outage(tmp_path, monkeypatch, failures: int):
    """A client whose control plane is unreachable for *failures*
    requests, then answers; a fake clock; the backoff of the real one."""
    server = _Server({})
    client = _client(tmp_path, server, monkeypatch)
    monkeypatch.setattr(tr, "RETRY_PAUSE_S", 1.0)
    monkeypatch.setattr(tr, "UPLOAD_RETRY_WINDOW_S", 300.0)
    clock = _Clock()
    monkeypatch.setattr(tr, "time", clock)
    left = [failures]

    def flaky(request: httpx.Request) -> httpx.Response:
        server.attempts = getattr(server, "attempts", 0) + 1
        if left[0] > 0:
            left[0] -= 1
            raise httpx.ConnectError("refused", request=request)
        return server(request)

    monkeypatch.setattr(client, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(flaky)))
    return client, server, clock


def test_an_upload_outlasts_a_connection_drop(tmp_path, monkeypatch):
    """Past the 3 quick attempts it keeps retrying, backing off, inside
    the resume grace: a run ending during a drop keeps its files."""
    client, server, clock = _outage(tmp_path, monkeypatch, failures=8)
    source = tmp_path / "ck.pt"
    source.write_bytes(b"weights")
    key = client.upload("run-1", "checkpoint", "ck.pt", source)
    assert server.uploads[key] == b"weights"
    pauses = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
    assert sum(clock.slept) == sum(pauses)
    assert clock.now < tr.UPLOAD_RETRY_WINDOW_S


def test_an_upload_gives_up_after_the_window_and_on_a_cancel(
        tmp_path, monkeypatch):
    client, _, clock = _outage(tmp_path, monkeypatch, failures=10_000)
    source = tmp_path / "ck.pt"
    source.write_bytes(b"weights")
    with pytest.raises(tr.TransferError, match="refused"):
        client.upload("run-1", "checkpoint", "ck.pt", source)
    assert tr.UPLOAD_RETRY_WINDOW_S <= clock.now \
        < tr.UPLOAD_RETRY_WINDOW_S + tr.UPLOAD_MAX_PAUSE_S
    stop = [False]

    def cancel_soon() -> bool:
        stop[0] = stop[0] or clock.now > 5
        return stop[0]

    clock.now = 0.0
    with pytest.raises(tr.TransferError, match="cancelled"):
        client.upload("run-1", "checkpoint", "ck.pt", source,
                      cancelled=cancel_soon)
    assert clock.now < 10


def test_a_runs_uploads_share_one_retry_window(tmp_path, monkeypatch):
    """Once the window ran out, the next file is tried once, with no pause,
    so a dead control plane holds a run's end for one window in all. The
    window is the run's (the control plane's resume grace)."""
    client, server, clock = _outage(tmp_path, monkeypatch, failures=10_000)
    run = tr.RunTransfer(run_id="r1", root=tr.run_data_root(tmp_path / "r"),
                         client=client, retry_window_s=lambda: 1800.0)
    for name in ("a.pt", "b.pt"):
        (run.root / "models" / name).write_bytes(b"1")
    body = {"kind": "checkpoint", "meta": None, "node_id": None,
            "mount_label": None}
    _, warning = run.upload_artifact(
        {**body, "path": str(run.root / "models" / "a.pt")})
    assert "stays on worker" in warning
    assert 1800 <= clock.now < 1800 + tr.UPLOAD_MAX_PAUSE_S
    first, tries = clock.now, server.attempts
    _, warning = run.upload_artifact(
        {**body, "path": str(run.root / "models" / "b.pt")})
    assert "stays on worker" in warning
    assert clock.now == first  # no pause
    assert server.attempts == tries + 1  # one attempt
    run.close()


def test_only_an_outage_gets_the_long_window(tmp_path, monkeypatch):
    """No connection, 408, 429, 502-504: the control plane is away, so the
    upload waits for it. Any other 5xx fails after the quick attempts."""
    client, server, clock = _outage(tmp_path, monkeypatch, failures=0)
    source = tmp_path / "ck.pt"
    source.write_bytes(b"weights")
    server.fail_next = [500] * 10
    with pytest.raises(tr.TransferError, match="HTTP 500"):
        client.upload("run-1", "checkpoint", "ck.pt", source)
    assert len(server.requests) == tr.ATTEMPTS
    assert clock.now == 1.0 + 2.0
    for status in (408, 429, 502, 503, 504):
        server.fail_next = [status] * 5
        clock.now = 0.0
        key = client.upload("run-1", "checkpoint", "ck.pt", source)
        assert server.uploads[key] == b"weights", status
        assert clock.now == 1.0 + 2.0 + 4.0 + 8.0 + 16.0, status


def test_prepare_inputs_copies_each_input_under_its_kind(tmp_path,
                                                         monkeypatch):
    files = {"t.csv": b"csv", "cat.png": b"png", "sub/m.pt": b"pt"}
    client = _client(tmp_path, _Server(files), monkeypatch)
    root = tr.run_data_root(tmp_path / "run")
    lines: list[str] = []
    entries = [_entry("t.csv", b"csv"), _entry("cat.png", b"png", "image"),
               _entry("sub/m.pt", b"pt", "model")]
    sessions: list[Any] = []
    real = client.session

    def counting() -> Any:
        sessions.append(1)
        return real()

    client.session = counting  # type: ignore[method-assign]
    placed = tr.prepare_inputs(client, entries, root, lines.append)
    assert list(placed) == [root / "files" / "t.csv",
                            root / "images" / "cat.png",
                            root / "models" / "sub" / "m.pt"]
    assert [p.read_bytes() for p in placed] == [b"csv", b"png", b"pt"]
    assert placed[root / "files" / "t.csv"][0] == 3
    assert len(sessions) == 1  # one connection for the batch
    assert tr.prepare_inputs(client, [], root, lines.append) == {}
    assert lines[0] == "fetching inputs 1/3: t.csv (3 B, fetched)"
    assert lines[-1].startswith("fetching inputs 3/3: sub/m.pt")
    lines.clear()
    tr.prepare_inputs(client, entries[:1], root, lines.append)
    assert lines == ["fetching inputs 1/1: t.csv (3 B, cached)"]


def test_a_run_shares_the_agents_cache_folder(tmp_path, monkeypatch):
    """LMTokenizedDataset's token blocks live in ``<data>/cache``: a run's
    own data folder links the agent's, so they survive the run."""
    own = tmp_path / "own-data"
    root = tr.run_data_root(tmp_path / "run", own)
    assert (root / "cache").is_symlink()
    assert (root / "cache").resolve() == (own / "cache").resolve()
    (root / "cache" / "blocks").write_bytes(b"b")
    assert (own / "cache" / "blocks").exists()
    assert tr.run_data_root(tmp_path / "run", own) == root  # idempotent

    def refused(*_: Any, **__: Any) -> None:
        raise OSError("symlinks need a privilege here")

    monkeypatch.setattr(Path, "symlink_to", refused)
    other = tr.run_data_root(tmp_path / "run2", own)
    assert not (other / "cache").exists()


def test_sizes_read_naturally():
    from app.core.transfer import human_size

    assert human_size(12) == "12 B"
    assert human_size(2048) == "2.0 KiB"
    assert human_size(3 * 1024 ** 3) == "3.00 GiB"
    assert human_size(5 * 1024 ** 4) == "5120.00 GiB"


# ── a run's outputs ───────────────────────────────────────────────────────


class _Uploads:
    """A ``TransferClient`` stand-in that records uploads."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, str, str, bytes]] = []
        self.fail = fail or set()

    def session(self) -> Any:
        self.sessions = getattr(self, "sessions", 0) + 1
        return _Closable()

    def upload(self, run_id: str, kind: str, name: str, path: Path, *,
               place: str = "run", client: Any = None, **_: Any) -> str:
        assert client is not None  # the run's one session
        if name in self.fail:
            raise tr.TransferError(f"uploading {name}: HTTP 503")
        self.calls.append((run_id, kind, name, path.read_bytes()))
        return name if place == "data" else f"runs/{run_id}/{kind}/{name}"


class _Closable:
    closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_Closable":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _file(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / f"src-{_sha(data)[:8]}"
    path.write_bytes(data)
    return path


def _run(tmp_path: Path, client: Any, **kw: Any) -> tr.RunTransfer:
    root = tr.run_data_root(tmp_path / "run")
    return tr.RunTransfer(run_id="r1", root=root, client=client, **kw)


def test_a_reported_file_is_uploaded_and_named_once(tmp_path):
    uploads = _Uploads()
    run = _run(tmp_path, uploads)
    ck = run.root / "models" / "ck.pt"
    ck.write_bytes(b"1")
    body = {"kind": "checkpoint", "path": str(ck), "meta": None,
            "node_id": "n", "mount_label": None}
    rewritten, warning = run.upload_artifact(body)
    assert warning is None
    assert rewritten == {**body, "path": "runs/r1/checkpoint/ck.pt",
                         "backend": "transfer"}
    # The same name again gets a key of its own.
    again, _ = run.upload_artifact(body)
    assert again["path"] == "runs/r1/checkpoint/2-ck.pt"
    assert uploads.sessions == 1  # one client for the run
    run.close()
    run.close()
    # Cancelled: nothing more is sent.
    run.stop = lambda: True
    kept, warning = run.upload_artifact(body)
    assert kept == body and "cancelled" in warning


def test_a_reported_directory_is_uploaded_file_by_file(tmp_path):
    uploads = _Uploads()
    run = _run(tmp_path, uploads)
    logdir = run.root / "runs" / "r1" / "tb" / "train"
    (logdir / "sub").mkdir(parents=True)
    (logdir / "events.1").write_bytes(b"e1")
    (logdir / "sub" / "events.2").write_bytes(b"e2")
    body = {"kind": "tensorboard", "path": str(logdir), "meta": None,
            "node_id": None, "mount_label": None}
    rewritten, warning = run.upload_artifact(body)
    assert warning is None
    assert rewritten["path"] == "runs/r1/tensorboard/train"
    assert [c[2] for c in uploads.calls] == ["train/events.1",
                                             "train/sub/events.2"]
    empty = run.root / "runs" / "r1" / "tb" / "empty"
    empty.mkdir()
    kept, warning = run.upload_artifact({**body, "path": str(empty)})
    assert kept["path"] == str(empty) and "is empty" in warning


def test_a_file_that_cannot_go_stays_reported_by_its_path(tmp_path):
    run = _run(tmp_path, _Uploads(fail={"ck.pt"}), max_upload_bytes=4)
    big = run.root / "models" / "big.pt"
    big.write_bytes(b"12345")
    body = {"kind": "checkpoint", "path": str(big), "meta": None,
            "node_id": None, "mount_label": None}
    kept, warning = run.upload_artifact(body)
    assert kept == body
    assert "over the control plane's upload limit" in warning
    ck = run.root / "models" / "ck.pt"
    ck.write_bytes(b"1")
    kept, warning = run.upload_artifact({**body, "path": str(ck)})
    assert "HTTP 503" in warning
    kept, warning = run.upload_artifact({**body, "path": "/nowhere/x.pt"})
    assert "outside the run's data folder" in warning
    # A row is a claim: a file elsewhere on this machine is not sent, nor
    # is an empty path (which would name the working directory).
    secret = tmp_path / "secret.key"
    secret.write_bytes(b"k")
    for path in (str(secret), "", "models/ck.pt"):
        kept, warning = run.upload_artifact({**body, "path": path})
        assert "outside the run's data folder" in warning, path
    gone = run.root / "models" / "gone.pt"
    kept, warning = run.upload_artifact({**body, "path": str(gone)})
    assert "is not on this worker" in warning


def test_unreported_outputs_skip_inputs_and_uploaded_files(tmp_path):
    import os

    uploads = _Uploads(fail={"media/clip.mp4"})
    run = _run(tmp_path, uploads)
    model_input = run.root / "models" / "base.pt"
    run.inputs[model_input] = tr.copy_input(_file(tmp_path, b"in"),
                                            model_input)
    csv_input = run.root / "files" / "t.csv"
    run.inputs[csv_input] = tr.copy_input(_file(tmp_path, b"c"), csv_input)
    reported = run.root / "models" / "periodic" / "p.pt"
    reported.parent.mkdir()
    reported.write_bytes(b"p")
    run.upload_artifact({"kind": "checkpoint", "path": str(reported),
                         "meta": None, "node_id": None, "mount_label": None})
    (run.root / "models" / "saved.pt").write_bytes(b"s")
    (run.root / "output").mkdir()
    (run.root / "output" / "img.png").write_bytes(b"i")
    (run.root / "media" / "clip.mp4").write_bytes(b"v")
    (run.root / "files" / "not-an-output.csv").write_bytes(b"c")
    found = run.unreported_outputs()
    assert [p.relative_to(run.root).as_posix() for p in found] == [
        "models/saved.pt", "media/clip.mp4", "output/img.png"]
    body, warning = run.upload_output(found[0])
    assert warning is None and body == {
        "kind": "output", "path": "models/saved.pt",
        "meta": {"name": "models/saved.pt"}, "node_id": None,
        "mount_label": None, "backend": "transfer"}
    body, warning = run.upload_output(found[1])
    assert body is None and "media/clip.mp4 stays on worker" in warning
    # A node that wrote over an input: that is an output too, and an input
    # outside the writers' folders lands under output/.
    model_input.write_bytes(b"retrained")
    csv_input.write_bytes(b"rewritten")
    stat = csv_input.stat()
    os.utime(csv_input, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    found = run.unreported_outputs()
    assert [p.relative_to(run.root).as_posix() for p in found] == [
        "models/base.pt", "media/clip.mp4", "output/img.png", "files/t.csv"]
    body, _ = run.upload_output(found[-1])
    assert body["path"] == "output/files/t.csv"


def test_only_a_stopped_loops_checkpoint_is_held_back():
    assert tr.is_interrupt_checkpoint(
        {"kind": "checkpoint", "meta": {"reason": "interrupted"}})
    assert not tr.is_interrupt_checkpoint(
        {"kind": "checkpoint", "meta": {"reason": "periodic"}})
    assert not tr.is_interrupt_checkpoint({"kind": "checkpoint",
                                           "meta": None})


# ── the Runner, end to end with a cheap child ─────────────────────────────


class _FakeClient:
    """``Runner.transfer``: fetches from a dict, uploads into a dict."""

    def __init__(self, tmp_path: Path, files: dict[str, bytes]) -> None:
        self.cache = tr.InputCache(tmp_path / "cache")
        self.files = files
        self.uploads: list[tuple[str, str]] = []
        self.gate: threading.Event | None = None
        self.upload_gate: threading.Event | None = None

    def session(self) -> Any:
        return _Closable()

    def fetch(self, entry: Any, cancelled: Any = lambda: False, *,
              client: Any = None) -> tuple[Path, bool]:
        if self.gate is not None:
            while not cancelled():
                self.gate.wait(0.01)
            raise tr.TransferCancelled("cancelled while fetching")
        found = self.cache.lookup(entry.sha256, entry.bytes)
        if found is not None:
            return found, False
        if entry.name not in self.files:
            raise tr.TransferError(f"fetching {entry.name}: HTTP 404")
        return self.cache.add(entry.sha256, [self.files[entry.name]]), True

    def upload(self, run_id: str, kind: str, name: str, path: Path, *,
               place: str = "run", client: Any = None, **_: Any) -> str:
        if self.upload_gate is not None:
            self.upload_gate.wait(10)
        self.uploads.append((kind, name))
        return name if place == "data" else f"runs/{run_id}/{kind}/{name}"


def _transfer_assign(run_id: str, inputs: list[wp.TransferInput]
                     ) -> wp.Assign:
    return wp.Assign(run_id=run_id, graph={"nodes": [], "edges": []},
                     storage=wp.Storage(backend="transfer", mount="data",
                                        relative_artifacts=False,
                                        inputs=inputs,
                                        max_upload_bytes=1 << 20))


async def test_a_transfer_run_fetches_first_and_uploads_its_outputs_last(
        tmp_path):
    data = b"a,b\n1,2\n"
    client = _FakeClient(tmp_path, {"train.csv": data})
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.transfer_child,
                    transfer=client)
    await runner.assign(_transfer_assign("t1", [_entry("train.csv", data)]))
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "t1")
    frames = sink.of("t1")
    assert [f.seq for f in frames] == list(range(1, len(frames) + 1))
    assert [f.type for f in frames] == [
        "warning", "run_started", "event", "artifact", "artifact",
        "artifact", "run_finished"]
    progress = frames[0]
    assert progress.kind == "transfer_progress"
    assert progress.detail == "fetching inputs 1/1: train.csv (8 B, fetched)"
    assert frames[2].payload == {"text": data.decode()}
    periodic, saved, stopped = frames[3:6]
    assert (periodic.kind, periodic.path, periodic.backend) == (
        "checkpoint", "runs/t1/checkpoint/p1.pt", "transfer")
    # A writer's file goes to the control plane's own data folder.
    assert (saved.kind, saved.path) == ("output", "models/saved.pt")
    # The interrupt checkpoint goes last, after the unreported outputs.
    assert (stopped.path, stopped.meta["reason"]) == (
        "runs/t1/checkpoint/i1.pt", "interrupted")
    assert finished.status == "cancelled"
    assert client.uploads == [("checkpoint", "p1.pt"),
                              ("output", "models/saved.pt"),
                              ("checkpoint", "i1.pt")]
    await _until(lambda: "t1" not in runner.runs)
    # The run's directory holds its own data root, with the input copied.
    assert (tmp_path / "agent-runs" / "t1" / "data" / "files" /
            "train.csv").read_bytes() == data

    # A second run finds the input in the cache.
    await runner.assign(_transfer_assign("t2", [_entry("train.csv", data)]))
    await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "t2")
    assert sink.of("t2")[0].detail.endswith("(8 B, cached)")


async def test_a_failed_fetch_fails_the_run_without_a_subprocess(tmp_path):
    client = _FakeClient(tmp_path, {})
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.quick_child, transfer=client)
    await runner.assign(_transfer_assign("f1", [_entry("gone.csv", b"x")]))
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "f1")
    assert finished.status == "failed" and finished.seq == 1
    assert "could not fetch the run's input files" in finished.error
    assert "HTTP 404" in finished.error
    assert not any(isinstance(f, wp.RunStarted) for f in sink.frames)
    await _until(lambda: "f1" not in runner.runs)


async def test_a_cancel_while_fetching_ends_the_run_cancelled(tmp_path):
    client = _FakeClient(tmp_path, {})
    client.gate = threading.Event()
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.quick_child, transfer=client)
    await runner.assign(_transfer_assign("c1", [_entry("big.csv", b"x")]))
    assert runner.running_ids() == ["c1"]  # in heartbeats while fetching
    assert await runner.cancel("c1") is True
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "c1")
    assert finished.status == "cancelled"
    await _until(lambda: "c1" not in runner.runs)
    assert await runner.cancel("c1") is False


async def test_shutdown_stops_a_run_that_is_fetching(tmp_path):
    client = _FakeClient(tmp_path, {})
    client.gate = threading.Event()
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.quick_child, transfer=client)
    await runner.assign(_transfer_assign("s1", [_entry("big.csv", b"x")]))
    await asyncio.wait_for(runner.shutdown(grace_s=1), 30)
    assert sink.finished("s1").status == "cancelled"


async def test_an_agent_without_a_transfer_client_refuses(tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink)
    await runner.assign(_transfer_assign("n1", []))
    finished = sink.finished("n1")
    assert finished.status == "failed"
    assert "cannot fetch input files" in finished.error


def test_the_connection_gives_the_runner_a_client(tmp_path):
    from app.agent.connection import AgentConnection

    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"})
    AgentConnection(runner=runner, name="w", version="0",
                    control_url=lambda: "http://cp:1", token="t",
                    probe_devices=lambda: [])
    assert isinstance(runner.transfer, tr.TransferClient)
    assert runner.transfer.cache.root == tmp_path / "agent-cache"
    assert runner.transfer._url("/x") == "http://cp:1/x"
    assert runner.transfer._headers() == {"X-CodefyUI-Token": "t"}


def test_file_transfers_verify_the_control_plane_like_the_socket(tmp_path):
    """#786: over https the transfer client uses the connection's trust
    context, so a pinned certificate is pinned for files too."""
    import ssl

    from app.agent.connection import AgentConnection

    url = {"value": "http://cp:1"}
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"})
    connection = AgentConnection(runner=runner, name="w", version="0",
                                 control_url=lambda: url["value"], token="t",
                                 probe_devices=lambda: [])
    assert connection._http_tls() is None
    with runner.transfer._client() as client:
        assert client is not None
    url["value"] = "https://cp:1"
    context = connection._http_tls()
    assert isinstance(context, ssl.SSLContext)
    assert connection.trust is not None
    with runner.transfer._client() as client:
        assert client is not None


def test_the_protocol_carries_a_transfer_assignment():
    frame = _transfer_assign("p1", [_entry("a.csv", b"a")])
    decoded = wp.decode_control_frame(wp.encode_frame(frame))
    assert decoded == frame
    assert json.loads(wp.encode_frame(frame))["storage"]["backend"] == \
        "transfer"
    with pytest.raises(wp.ProtocolError):
        wp.decode_control_frame(json.dumps({
            "type": "assign", "run_id": "p1", "graph": {},
            "storage": {"backend": "transfer", "mount": "data", "inputs": [
                {"kind": "data_file", "name": "a", "sha256": "nothex",
                 "bytes": 1}]}}))


async def _until(predicate: Any, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition never held"
        await asyncio.sleep(0.02)


async def test_uploads_never_hold_up_the_subprocess_and_a_late_cancel_stops_them(
        tmp_path):
    """#785 review: the subprocess exits while its first upload is still
    going (the uploader is a thread of its own); the run is not reported
    until the uploads end; a cancel then stops the uploads not yet sent,
    with one warning, and ``run_finished`` comes last. Messages the agent
    cannot use are dropped, not fatal."""
    client = _FakeClient(tmp_path, {})
    client.upload_gate = threading.Event()
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.slow_upload_child,
                    transfer=client)
    await runner.assign(_transfer_assign("u1", []))
    await _until(lambda: "u1" in runner.runs and runner.runs["u1"].reaped)
    handle = runner.runs["u1"]
    assert not handle.exited.is_set()
    assert sink.finished("u1") is None
    assert await runner.cancel("u1") is True
    assert handle.uploads_cancelled
    client.upload_gate.set()
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "u1")
    frames = sink.of("u1")
    assert [f.seq for f in frames] == list(range(1, len(frames) + 1))
    assert [f.type for f in frames] == ["run_started", "artifact", "warning",
                                        "run_finished"]
    assert frames[1].path == "runs/u1/checkpoint/p1.pt"
    assert frames[2].detail.startswith("1 file(s) not uploaded")
    assert finished.status == "succeeded"
    assert client.uploads == [("checkpoint", "p1.pt")]
    await _until(lambda: "u1" not in runner.runs)
