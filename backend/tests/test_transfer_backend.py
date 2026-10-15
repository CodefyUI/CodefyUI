"""The ``transfer`` storage backend, control plane's half (#785).

``core.transfer`` (which runs it may serve, the content index, the run's
received files), the upload and download routes with their auth and body
cap, retention, placement (the mount preferred, transfer otherwise, #782's
refusal only when neither applies) and ``WorkerExecutor``'s assignment.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import transfer
from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.executors import (
    ExecutionRequest,
    TransferredArtifactSignal,
    WorkerExecutor,
    transfer_storage,
    uses_transfer,
)
from app.core.run_store import RunStore
from app.core.scheduling import Placement
from app.main import app
from tests.test_worker_executor import _Agent, _Recorder
from tests.test_worker_scheduler import (
    _Entry,
    _options,
    _registry_with,
    _scheduler,
    _service,
    _target,
    _worker,
)

GIB = 1024 ** 3


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """This server's data directories under ``tmp_path/data``, with a CSV,
    an image and a nested model uploaded."""
    root = tmp_path / "data"
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(settings, "MOUNT_LABEL", "data")
    monkeypatch.setattr(settings, "TRANSFER_ENABLED", True)
    monkeypatch.setattr(settings, "TRANSFER_MAX_INPUT_BYTES", 2 * GIB)
    for name, sub in (("DATA_FILES_DIR", "files"), ("IMAGES_DIR", "images"),
                      ("MODELS_DIR", "models"), ("MEDIA_DIR", "media")):
        (root / sub).mkdir(parents=True)
        monkeypatch.setattr(settings, name, root / sub)
    (root / "files" / "train.csv").write_bytes(b"a,b\n1,2\n")
    (root / "images" / "cat.png").write_bytes(b"png")
    (root / "models" / "periodic").mkdir()
    (root / "models" / "periodic" / "x.pt").write_bytes(b"weights")
    return root.resolve()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


def _needs(*inputs: tuple[str, str], paths: tuple[str, ...] = ()
           ) -> dict[str, Any]:
    return {"mount": "data",
            "files": sorted({n for _, n in inputs} | set(paths)),
            "inputs": [{"kind": k, "name": n} for k, n in inputs],
            "paths": list(paths)}


# ── which runs it serves ──────────────────────────────────────────────────


def test_an_upload_is_found_by_the_rule_its_node_uses(data_root):
    assert transfer.store_path("data_file", "train.csv") == \
        data_root / "files" / "train.csv"
    assert transfer.store_path("image", "cat.png") == \
        data_root / "images" / "cat.png"
    assert transfer.store_path("model", "periodic/x.pt") == \
        data_root / "models" / "periodic" / "x.pt"
    # CSVReader looks a bare name up in its store: nothing else is one.
    for kind, name in (("data_file", "files/train.csv"),
                       ("data_file", "../files/train.csv"),
                       ("data_file", "/etc/passwd"),
                       ("model", "../files/train.csv"),
                       ("data_file", "missing.csv"),
                       ("data_file", ""),
                       ("video", "train.csv")):
        assert transfer.store_path(kind, name) is None, (kind, name)
    # The download route serves only its whitelist.
    (data_root / "files" / "notes.exe").write_bytes(b"x")
    assert transfer.store_path("data_file", "notes.exe") is None
    # A directory is not a file.
    (data_root / "files" / "dir.csv").mkdir()
    assert transfer.store_path("data_file", "dir.csv") is None


def test_check_accepts_uploads_and_says_why_not_otherwise(data_root,
                                                          monkeypatch):
    ok = transfer.check(_needs(("data_file", "train.csv"),
                               ("model", "periodic/x.pt")))
    assert ok == {"ok": True, "bytes": 8 + 7, "reason": None}
    assert transfer.check(_needs())["ok"] is True  # a writer only
    paths = transfer.check(_needs(paths=("a/", "b", "c", "d")))
    assert not paths["ok"]
    assert paths["reason"].startswith(
        "it reads `a/`, `b`, `c` and 1 more by path")
    missing = transfer.check(_needs(("data_file", "gone.csv")))
    assert missing["reason"] == ("`gone.csv` is not an upload on this "
                                 "server (data file)")
    monkeypatch.setattr(settings, "TRANSFER_MAX_INPUT_BYTES", 10)
    over = transfer.check(_needs(("data_file", "train.csv"),
                                 ("model", "periodic/x.pt")))
    assert not over["ok"] and over["bytes"] == 15
    assert "CODEFYUI_TRANSFER_MAX_INPUT_BYTES" in over["reason"]
    monkeypatch.setattr(settings, "TRANSFER_ENABLED", False)
    off = transfer.check(_needs())
    assert "CODEFYUI_TRANSFER_ENABLED=0" in off["reason"]


def test_annotate_only_touches_a_run_that_needs_the_mount(data_root):
    assert transfer.annotate(None) is None
    plain = {"mount": None, "files": []}
    assert transfer.annotate(plain) == {"mount": None, "files": []}
    needs = transfer.annotate(_needs(("data_file", "train.csv")))
    assert transfer.transferable(needs) and transfer.refusal(needs) is None
    bad = transfer.annotate(_needs(("data_file", "gone.csv")))
    assert not transfer.transferable(bad)
    assert "gone.csv" in transfer.refusal(bad)
    assert transfer.refusal({"transfer": {"ok": False}}) is None
    assert not transfer.transferable({"mount": "data"})


# ── the content index ─────────────────────────────────────────────────────


async def test_the_index_hashes_once_and_again_when_the_file_moves(
        data_root, db, monkeypatch):
    index = transfer.ContentIndex(db)
    hashed: list[Path] = []
    real = transfer.sha256_file

    def counting(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(transfer, "sha256_file", counting)
    csv = data_root / "files" / "train.csv"
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert await index.digest("data_file", "train.csv") == (expected, 8)
    assert await index.digest("data_file", "train.csv") == (expected, 8)
    assert len(hashed) == 1
    csv.write_bytes(b"a,b\n3,4\n")  # same size, new mtime
    import os

    stat = csv.stat()
    os.utime(csv, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    sha, _ = await index.digest("data_file", "train.csv")
    assert sha == hashlib.sha256(b"a,b\n3,4\n").hexdigest()
    assert len(hashed) == 2
    with pytest.raises(transfer.TransferRefused, match="not an upload"):
        await index.digest("data_file", "gone.csv")


async def test_the_manifest_names_every_input_and_rechecks_the_cap(
        data_root, db, monkeypatch):
    index = transfer.ContentIndex(db)
    manifest = await index.manifest([
        {"kind": "data_file", "name": "train.csv"},
        {"kind": "image", "name": "cat.png"}])
    assert [(m["kind"], m["name"], m["bytes"]) for m in manifest] == [
        ("data_file", "train.csv", 8), ("image", "cat.png", 3)]
    assert manifest[1]["sha256"] == hashlib.sha256(b"png").hexdigest()
    monkeypatch.setattr(settings, "TRANSFER_MAX_INPUT_BYTES", 10)
    with pytest.raises(transfer.TransferRefused, match="over the transfer"):
        await index.manifest([{"kind": "data_file", "name": "train.csv"},
                              {"kind": "image", "name": "cat.png"}])


async def test_a_file_that_cannot_be_read_is_refused(data_root, db,
                                                     monkeypatch):
    def unreadable(self: Path, *a: Any, **k: Any) -> Any:
        raise OSError("permission denied")

    path = transfer.store_path("data_file", "train.csv")
    monkeypatch.setattr(transfer, "store_path", lambda *_: path)
    monkeypatch.setattr(Path, "stat", unreadable)
    with pytest.raises(transfer.TransferRefused, match="could not be read"):
        await transfer.ContentIndex(db).digest("data_file", "train.csv")
    assert transfer.check(_needs(("data_file", "train.csv")))[
        "reason"] == "`train.csv` could not be read on this server"


# ── received files ────────────────────────────────────────────────────────


def test_keys_name_the_runs_directory_or_a_writers_folder(data_root):
    assert transfer.run_key("r1", "checkpoint", "a.pt") == \
        "runs/r1/checkpoint/a.pt"
    assert transfer.run_key("r1", "tensorboard", "train/ev 1") == \
        "runs/r1/tensorboard/train/ev_1"
    for bad in ("", "/abs", "../up", "a/../../b"):
        with pytest.raises(ValueError):
            transfer.run_key("r1", "checkpoint", bad)
    assert transfer.key_path("r1", "runs/r1/checkpoint/a.pt") == \
        data_root / "runs" / "r1" / "checkpoint" / "a.pt"
    # A writer's file: where the node writes on a local run.
    assert transfer.key_path("r1", "models/a.pt") == \
        data_root / "models" / "a.pt"
    assert transfer.key_path("r1", "media/v.mp4") == \
        data_root / "media" / "v.mp4"
    assert transfer.key_path("r1", "output/i.png") == \
        data_root / "output" / "i.png"
    for bad in ("runs/r2/checkpoint/a.pt", "runs/r1/a.pt", "/runs/r1/c/a",
                "runs/r1/c/../../../x", "files/a.csv", "models", ""):
        assert transfer.key_path("r1", bad) is None, bad


def _tmp(directory: Path, data: bytes) -> Path:
    import uuid

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".incoming-{uuid.uuid4().hex}"
    path.write_bytes(data)
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_writers_file_never_lands_over_a_different_one(data_root):
    models = data_root / "models"
    (models / "saved.pt").write_bytes(b"the user's own")
    key = transfer.place_file(_tmp(models, b"new"), "data", "r1", "output",
                              "models/saved.pt", _sha(b"new"))
    assert key == "models/saved-1.pt"
    assert (models / "saved.pt").read_bytes() == b"the user's own"
    assert (models / "saved-1.pt").read_bytes() == b"new"
    # The same content again is the same file: no third copy.
    again = transfer.place_file(_tmp(models, b"new"), "data", "r1",
                                "output", "models/saved.pt", _sha(b"new"))
    assert again == "models/saved-1.pt"
    newer = transfer.place_file(_tmp(models, b"newer"), "data", "r1",
                                "output", "models/saved.pt", _sha(b"newer"))
    assert newer == "models/saved-2.pt"
    nested = transfer.place_file(_tmp(data_root / "output" / "a", b"i"),
                                 "data", "r1", "output", "output/a/i.png",
                                 _sha(b"i"))
    assert nested == "output/a/i.png"
    assert not [p for p in data_root.rglob(".incoming-*")]
    for place, name in (("data", "files/x.csv"), ("data", "models"),
                        ("elsewhere", "models/x.pt")):
        with pytest.raises(ValueError):
            transfer.place_file(_tmp(models, b"x"), place, "r1", "output",
                                name, _sha(b"x"))


def test_a_writers_file_takes_a_name_without_hard_links(data_root,
                                                        monkeypatch):
    import os

    models = data_root / "models"
    (models / "saved.pt").write_bytes(b"old")

    def no_links(*_: Any) -> None:
        raise OSError("hard links not supported")

    monkeypatch.setattr(transfer.os, "link", no_links)
    key = transfer.place_file(_tmp(models, b"new"), "data", "r1", "output",
                              "models/saved.pt", _sha(b"new"))
    assert key == "models/saved-1.pt"
    assert (models / "saved-1.pt").read_bytes() == b"new"
    assert os.path.exists(models / "saved.pt")


def test_a_run_file_is_replaced_atomically(data_root):
    run_dir = data_root / "runs" / "r1"
    first = transfer.place_file(_tmp(run_dir, b"1"), "run", "r1",
                                "checkpoint", "a.pt", _sha(b"1"))
    second = transfer.place_file(_tmp(run_dir, b"1"), "run", "r1",
                                 "checkpoint", "a.pt", _sha(b"1"))
    assert first == second == "runs/r1/checkpoint/a.pt"
    assert (run_dir / "checkpoint" / "a.pt").read_bytes() == b"1"


def _receiving(directory: Path, data: bytes) -> transfer.Receiver:
    """A Receiver part-way through a file: the server died here."""
    boundary = "b0undary"
    head = "".join(
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f'name="{name}"\r\n\r\n{value}\r\n'
        for name, value in (("kind", "checkpoint"), ("name", "a.pt")))
    head += (f"--{boundary}\r\nContent-Disposition: form-data; "
             'name="file"; filename="a.pt"\r\n'
             "Content-Type: application/octet-stream\r\n\r\n")
    receiver = transfer.Receiver(
        f"multipart/form-data; boundary={boundary}", lambda _f: directory,
        1 << 20)
    receiver.write(head.encode() + data)
    return receiver


def test_unfinished_uploads_are_swept_and_never_listed(data_root, tmp_path):
    """A server that died mid-upload leaves ``.incoming-*`` files beside
    where they would have landed: startup removes every one the upload
    ledger names (no walk of the data folders, no age rule), and the
    models and media listings never show one."""
    from app.api.routes_media import list_media_files
    from app.api.routes_models import list_model_files

    died = [_receiving(data_root / "runs" / "r1", b"part"),
            _receiving(data_root / "models" / "periodic", b"part"),
            _receiving(data_root / "media" / "clips", b"part")]
    for receiver in died:
        receiver._out.flush()
    stale = [receiver.tmp for receiver in died]
    assert all(path.is_file() and transfer.is_incoming(path)
               for path in stale)
    assert len(list(transfer.incoming_ledger().iterdir())) == 3
    # A finished upload leaves no ledger entry.
    done = _receiving(data_root / "models", b"x")
    done.discard()
    assert len(list(transfer.incoming_ledger().iterdir())) == 3
    # Listings skip them, whatever their name looks like.
    (data_root / "models" / ".incoming-c.pt").write_bytes(b"partial")
    (data_root / "media" / ".incoming-d.mp4").write_bytes(b"partial")
    names = {f["filename"] for f in asyncio.run(list_model_files())}
    assert not any(transfer.is_incoming(n) for n in names)
    assert asyncio.run(list_media_files()) == []
    # An entry naming something else is not followed.
    outside = tmp_path / "precious"
    outside.write_bytes(b"keep")
    (transfer.incoming_ledger() / "forged").write_text(str(outside))
    for receiver in died:
        receiver._out.close()

    assert transfer.sweep_incoming() == len(stale)
    assert not any(path.exists() for path in stale)
    assert outside.exists()
    assert list(transfer.incoming_ledger().iterdir()) == []
    assert (data_root / "models" / "periodic" / "x.pt").exists()


def test_the_run_directory_guard_refuses_symlinks(data_root, tmp_path):
    outside = tmp_path / "precious"
    (outside / "checkpoint").mkdir(parents=True)
    (outside / "keep.txt").write_bytes(b"k")
    runs = data_root / "runs"
    runs.mkdir()
    (runs / "linked").symlink_to(outside, target_is_directory=True)
    assert transfer.owned_run_dir("linked") is None
    assert transfer.remove_run_files("linked") is False
    assert (outside / "keep.txt").exists()
    # A real run directory holding a link: the link goes, never its target.
    real = runs / "r9"
    real.mkdir()
    (real / "escape").symlink_to(outside, target_is_directory=True)
    (real / "checkpoint").symlink_to(outside / "checkpoint",
                                     target_is_directory=True)
    assert transfer.remove_run_files("r9", keep_checkpoints=True) is True
    assert sorted(p.name for p in real.iterdir()) == []
    assert (outside / "keep.txt").exists()
    (real / "escape").symlink_to(outside, target_is_directory=True)
    assert transfer.remove_run_files("r9") is True
    assert not real.exists() and (outside / "keep.txt").exists()
    # The runs root itself a link: nothing under it is "owned".
    other = tmp_path / "other-runs"
    (other / "r1").mkdir(parents=True)
    import shutil

    shutil.rmtree(runs)
    runs.symlink_to(other, target_is_directory=True)
    assert transfer.owned_run_dir("r1") is None
    with pytest.raises(ValueError):
        transfer.place_file(_tmp(tmp_path / "t", b"x"), "run", "r1",
                            "checkpoint", "a.pt", _sha(b"x"))


def test_removing_a_runs_files_keeps_an_interrupted_runs_checkpoints(
        data_root):
    def fill(run_id: str) -> Path:
        base = data_root / "runs" / run_id
        for sub in ("checkpoint", "output/models"):
            (base / sub).mkdir(parents=True)
            (base / sub / "f").write_bytes(b"x")
        (base / "loose").write_bytes(b"x")
        return base

    gone = fill("r1")
    assert transfer.remove_run_files("r1") is True
    assert not gone.exists()
    kept = fill("r2")
    assert transfer.remove_run_files("r2", keep_checkpoints=True) is True
    assert sorted(p.name for p in kept.iterdir()) == ["checkpoint"]
    assert transfer.remove_run_files("never") is False


def test_a_transfer_row_is_found_under_this_servers_data_root(data_root):
    from app.core.mounts import translate

    assert translate("transfer", None, "runs/r1/checkpoint/a.pt") == \
        str(data_root / "runs" / "r1" / "checkpoint" / "a.pt")
    assert translate("transfer", None, "/etc/passwd") is None
    assert translate("transfer", None, "runs/../../x") is None
    assert translate("transfer", None, "models/saved-1.pt") == \
        str(data_root / "models" / "saved-1.pt")


async def test_retention_removes_a_worker_runs_received_files(data_root, db):
    store = RunStore(db)

    async def run(status: str, worker: str) -> str:
        record = await store.create_run(graph_snapshot={"nodes": []},
                                        status="running", worker_id=worker)
        key = transfer.run_key(record.id, "checkpoint", "a.pt")
        path = transfer.key_location(key)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        (path.parent.parent / "output").mkdir()
        await store.add_artifact(record.id, "checkpoint", key,
                                 backend="transfer", worker_id=worker)
        # A writer's file placed in this server's data folder.
        saved = data_root / "models" / f"{record.id}.pt"
        saved.write_bytes(b"s")
        await store.add_artifact(record.id, "output",
                                 f"models/{record.id}.pt",
                                 backend="transfer", worker_id=worker)
        await store.mark_finished(record.id, status)
        return record.id

    done = await run("succeeded", "gpu-1")
    stopped = await run("interrupted", "gpu-1")
    here = await store.create_run(graph_snapshot={"nodes": []},
                                  status="running")
    (data_root / "runs" / here.id / "tb").mkdir(parents=True)
    await store.mark_finished(here.id, "succeeded")
    row, _ = await store.list_artifacts(done)
    assert row.backend == "transfer" and row.path == str(
        data_root / "runs" / done / "checkpoint" / "a.pt")
    assert await store.prune(keep_last=0) == 3
    assert not (data_root / "runs" / done).exists()
    # An interrupted run keeps its resume point, as a local one does.
    assert sorted(p.name for p in (data_root / "runs" / stopped).iterdir()) \
        == ["checkpoint"]
    # A run of this process is not a worker run: its directory is the
    # TensorBoard sweep's business, untouched here.
    assert (data_root / "runs" / here.id / "tb").is_dir()
    # Writers' files are the user's, as on a local run.
    assert (data_root / "models" / f"{done}.pt").exists()


def test_an_artifact_event_names_a_transfer_rows_key():
    from app.core.run_events import artifact_payload

    record = SimpleNamespace(id=3, kind="checkpoint", path="/d/runs/r/c/a",
                             meta=None, backend="transfer",
                             stored_path="runs/r/c/a", worker_id="gpu-1")
    assert artifact_payload(record, "n") == {
        "artifact_id": 3, "kind": "checkpoint", "path": "/d/runs/r/c/a",
        "meta": None, "node_id": "n", "backend": "transfer",
        "relative_path": "runs/r/c/a", "worker_id": "gpu-1"}


# ── the routes ────────────────────────────────────────────────────────────


@pytest.fixture
async def http(data_root, db, monkeypatch):
    """The app with a stand-in run service (run ``r1`` on ``gpu-1``) and a
    worker store that knows two worker tokens."""
    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", run_id="r1",
                                    worker_id="gpu-1")
    running = {"r1": "gpu-1"}
    service = SimpleNamespace(store=store,
                              running_on=lambda run_id: running.get(run_id))
    tokens = {"token-gpu-1": "gpu-1", "token-gpu-2": "gpu-2"}

    async def authenticate(token: str | None) -> str | None:
        return tokens.get(token or "")

    monkeypatch.setattr(app.state, "run_service", service, raising=False)
    monkeypatch.setattr(app.state, "worker_store",
                        SimpleNamespace(authenticate=authenticate),
                        raising=False)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://127.0.0.1:8000") as client:
        client.run = record  # type: ignore[attr-defined]
        client.running = running  # type: ignore[attr-defined]
        yield client


def _upload(client: AsyncClient, token: str | None, *, run_id: str = "r1",
            data: bytes = b"weights", name: str = "a.pt",
            sha: str | None = None) -> Any:
    headers = {TOKEN_HEADER: token} if token else {}
    fields = {"kind": "checkpoint", "name": name}
    if sha is not None:
        fields["sha256"] = sha
    return client.post(f"/api/runs/{run_id}/artifacts/upload",
                       headers=headers, data=fields,
                       files={"file": ("a.pt", data,
                                       "application/octet-stream")})


async def test_a_worker_uploads_its_runs_file(http, data_root):
    response = await _upload(http, "token-gpu-1",
                             sha=hashlib.sha256(b"weights").hexdigest())
    assert response.status_code == 200, response.text
    assert response.json() == {
        "path": "runs/r1/checkpoint/a.pt", "bytes": 7,
        "sha256": hashlib.sha256(b"weights").hexdigest()}
    assert (data_root / "runs" / "r1" / "checkpoint" / "a.pt").read_bytes() \
        == b"weights"
    # The session token may upload for any worker run.
    assert (await _upload(http, session_token(), name="b.pt")).status_code \
        == 200


async def test_an_upload_is_refused_unless_the_run_is_that_workers(http):
    assert (await _upload(http, None)).status_code == 403
    assert (await _upload(http, "forged")).status_code == 403
    other = await _upload(http, "token-gpu-2")
    assert other.status_code == 409
    assert "not running on worker gpu-2" in other.json()["detail"]
    http.running.clear()
    assert (await _upload(http, session_token())).status_code == 409


async def test_a_damaged_or_malformed_upload_is_refused(http, data_root,
                                                        monkeypatch):
    bad = await _upload(http, "token-gpu-1", sha="0" * 64)
    assert bad.status_code == 400 and "sha256 mismatch" in bad.text
    assert not (data_root / "runs" / "r1" / "checkpoint" / "a.pt").exists()
    climbing = await _upload(http, "token-gpu-1", name="../../x")
    assert climbing.status_code == 400
    missing = await http.post("/api/runs/r1/artifacts/upload",
                              headers={TOKEN_HEADER: "token-gpu-1"},
                              data={"kind": "checkpoint"})
    assert missing.status_code == 400
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE", 3)
    big = await _upload(http, "token-gpu-1")
    assert big.status_code == 413


def test_the_upload_route_skips_the_session_guard_and_takes_the_file_cap():
    from app.core.body_limit import MULTIPART_ENVELOPE_ALLOWANCE, \
        limit_for_path
    from app.main import _AUTH_EXEMPT_ROUTE_PATTERNS

    [(method, pattern)] = _AUTH_EXEMPT_ROUTE_PATTERNS
    assert method == "POST"
    assert pattern.fullmatch("/api/runs/abc/artifacts/upload")
    assert not pattern.fullmatch("/api/runs/abc/artifacts/upload/x")
    assert not pattern.fullmatch("/api/runs/a/b/artifacts/upload")
    assert limit_for_path("/api/runs/abc/artifacts/upload") == \
        settings.MAX_UPLOAD_SIZE + MULTIPART_ENVELOPE_ALLOWANCE
    assert limit_for_path("/api/runs/abc/cancel") == \
        settings.MAX_RUN_BODY_BYTES


async def test_other_mutations_on_runs_still_need_the_session_token(http):
    response = await http.delete("/api/runs/r1/artifacts/upload",
                                 headers={TOKEN_HEADER: "token-gpu-1"})
    assert response.status_code == 403
    response = await http.post("/api/runs/r1/cancel",
                               headers={TOKEN_HEADER: "token-gpu-1"})
    assert response.status_code == 403


async def test_an_uploaded_file_downloads_and_nothing_else_does(
        http, data_root):
    await _upload(http, "token-gpu-1")
    store = app.state.run_service.store
    row = await store.add_artifact("r1", "checkpoint",
                                   "runs/r1/checkpoint/a.pt",
                                   backend="transfer", worker_id="gpu-1")
    got = await http.get(f"/api/runs/r1/artifacts/{row.id}/download")
    assert got.status_code == 200 and got.content == b"weights"
    local = await store.add_artifact("r1", "checkpoint", "/etc/passwd")
    refused = await http.get(f"/api/runs/r1/artifacts/{local.id}/download")
    assert refused.status_code == 400
    assert (await http.get("/api/runs/r1/artifacts/999/download")
            ).status_code == 404
    assert (await http.get("/api/runs/nope/artifacts/1/download")
            ).status_code == 404
    gone = await store.add_artifact("r1", "checkpoint",
                                    "runs/r1/checkpoint/gone.pt",
                                    backend="transfer")
    assert (await http.get(f"/api/runs/r1/artifacts/{gone.id}/download")
            ).status_code == 404
    (data_root / "runs" / "r1" / "tensorboard" / "t").mkdir(parents=True)
    logdir = await store.add_artifact("r1", "tensorboard",
                                      "runs/r1/tensorboard/t",
                                      backend="transfer")
    assert (await http.get(f"/api/runs/r1/artifacts/{logdir.id}/download")
            ).status_code == 400
    listed = (await http.get("/api/runs/r1/artifacts")).json()["artifacts"]
    assert listed[0]["backend"] == "transfer"
    assert listed[0]["relative_path"] == "runs/r1/checkpoint/a.pt"


# ── placement ─────────────────────────────────────────────────────────────


@dataclass
class _NeedyEntry(_Entry):
    needs: dict[str, Any] | None = None


SHIPPABLE = {"mount": "data", "files": ["t.csv"],
             "inputs": [{"kind": "data_file", "name": "t.csv"}],
             "paths": [], "transfer": {"ok": True, "bytes": 1,
                                       "reason": None}}
NOT_SHIPPABLE = {**SHIPPABLE, "paths": ["imgs/"],
                 "transfer": {"ok": False, "bytes": 0,
                              "reason": "it reads `imgs/` by path"}}


def _speaking(worker: Any) -> Any:
    """*worker* with an agent that lists the ``transfer`` capability."""
    worker.capabilities = tuple(wp.CAPABILITIES)
    return worker


def test_a_worker_without_the_mount_takes_a_shippable_run(data_root):
    bare = _worker("bare", "cpu")
    scheduler = _scheduler(bare)
    entry = _NeedyEntry("r1", "bare/cpu", _target("bare"), SHIPPABLE)
    # Only once its agent says it speaks the transfer backend (#781).
    assert scheduler.place(entry, {}) is None
    _speaking(bare)
    assert scheduler.place(entry, {}).worker_id == "bare"
    assert scheduler.explain(entry) is None
    assert scheduler.reservation(_NeedyEntry(
        "r1", "bare/cpu", _target("bare", "cpu", ["cpu", "cpu"]),
        SHIPPABLE)) is not None
    refused = _NeedyEntry("r2", "bare/cpu", _target("bare"), NOT_SHIPPABLE)
    assert scheduler.place(refused, {}) is None
    assert scheduler.explain(refused) == (
        "worker bare has no `data` mount; add `--mount data=...` to its "
        "agent, or make the run fit the transfer backend: it reads `imgs/` "
        "by path")


def test_any_prefers_the_mount_then_transfer_then_this_process(data_root):
    bare = _speaking(_worker("a-bare", "cpu"))
    nas = _worker("b-nas", "cpu")
    nas.mounts = {"data": "/n"}
    mute = _worker("c-mute", "cpu")  # no transfer capability
    scheduler = _scheduler(bare, nas, mute)
    entry = _NeedyEntry("r1", "any/cpu", _target("any"), SHIPPABLE)
    # a-bare sorts first, but b-nas holds the mount.
    assert scheduler.place(entry, {}).worker_id == "b-nas"
    assert scheduler.place(entry, {("b-nas", "cpu"): 2}).worker_id == "a-bare"
    assert scheduler.place(entry, {("b-nas", "cpu"): 2,
                                   ("a-bare", "cpu"): 2}).worker_id == "local"
    cuda = _NeedyEntry("r2", "any/cuda", _target("any", "cuda"), SHIPPABLE)
    assert scheduler.explain(cuda) == "waiting for a worker that offers cuda"


def test_submit_accepts_a_shippable_run_for_a_worker_without_the_mount(
        data_root):
    service = _service(_registry_with(
        _speaking(_worker("gpu-box-1", "cpu", "cuda:0"))))
    check = service._check_storage
    check(_options(worker="gpu-box-1"), SHIPPABLE)
    check(_options(worker="any", device="cuda"), SHIPPABLE)
    with pytest.raises(Exception, match="make the run fit the transfer"):
        check(_options(worker="gpu-box-1"), NOT_SHIPPABLE)
    with pytest.raises(Exception, match="cannot ship them: it reads"):
        check(_options(worker="any", device="cuda"), NOT_SHIPPABLE)


def test_needs_computed_at_submit_carry_the_verdict(data_root):
    service = _service(_registry_with(_worker("gpu-box-1", "cpu")))
    graph = {"nodes": [
        {"id": "s", "type": "Start", "data": {"params": {}}},
        {"id": "c", "type": "CSVReader",
         "data": {"params": {"path": "train.csv"}}}],
        "edges": [{"id": "t", "source": "s", "target": "c",
                   "sourceHandle": "trigger", "type": "trigger"}]}
    *_, needs = service._prepare(graph, {"target": {"worker": "gpu-box-1"}})
    assert needs["inputs"] == [{"kind": "data_file", "name": "train.csv"}]
    assert needs["transfer"] == {"ok": True, "bytes": 8, "reason": None}


# ── the assignment ────────────────────────────────────────────────────────


def _request(recorder: _Recorder, needs: Any) -> ExecutionRequest:
    from app.core.execution_context import ExecutionContext

    return ExecutionRequest(
        run_id="r1", graph={"nodes": []}, options={},
        context=ExecutionContext(execution_id="r1", device="cuda:2"),
        session=None, output_store=None, record_outputs=False,
        sinks=recorder.sinks(), explicit_metric_nodes=set(),
        placement=Placement("box", ("cuda:2",), (("box", "cuda:2"),)),
        needs=needs)


async def test_a_transfer_assignment_names_its_inputs_by_hash(
        data_root, db):
    from app.core.workers import WorkerRegistry

    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _speaking(agent.worker)
        recorder = _Recorder()
        needs = transfer.annotate(_needs(("data_file", "train.csv")))
        executor = WorkerExecutor(registry,
                                  content_index=transfer.ContentIndex(db))
        task = asyncio.create_task(executor.run(_request(recorder, needs)))
        assign = await agent.wait_for(wp.Assign)
        storage = assign.storage
        assert storage.backend == "transfer" and storage.mount == "data"
        assert storage.relative_artifacts is False
        assert storage.max_upload_bytes == settings.MAX_UPLOAD_SIZE
        assert storage.inputs == [wp.TransferInput(
            kind="data_file", name="train.csv", bytes=8,
            sha256=hashlib.sha256(b"a,b\n1,2\n").hexdigest())]
        for frame in (wp.Artifact(run_id="r1", seq=1, kind="checkpoint",
                                  path="runs/r1/checkpoint/a.pt",
                                  backend="transfer"),
                      wp.RunFinished(run_id="r1", seq=2,
                                     status="succeeded")):
            registry.route(agent.worker, frame)
        outcome = await task
        assert outcome.status == "succeeded"
        signal = recorder.calls[0][1]
        assert isinstance(signal, TransferredArtifactSignal)
        assert signal.path == "runs/r1/checkpoint/a.pt"
    finally:
        await registry.shutdown()


async def test_an_input_gone_by_assignment_fails_the_run_unassigned(
        data_root, db):
    from app.core.workers import WorkerRegistry

    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _speaking(agent.worker)
        needs = transfer.annotate(_needs(("data_file", "train.csv")))
        (data_root / "files" / "train.csv").unlink()
        outcome = await WorkerExecutor(
            registry, content_index=transfer.ContentIndex(db)).run(
                _request(_Recorder(), needs))
        assert outcome.status == "failed"
        assert outcome.error.startswith(
            "cannot send this run's input files to worker box: `train.csv`")
        assert agent.sent == []
        assert registry.run_state("box", "r1") is None
    finally:
        await registry.shutdown()


async def test_transfer_storage_without_inputs_needs_no_index(data_root):
    storage = await transfer_storage("data", _needs(), None)
    assert storage.inputs == [] and storage.backend == "transfer"
    with pytest.raises(transfer.TransferRefused, match="no content index"):
        await transfer_storage("data", _needs(("data_file", "train.csv")),
                               None)


def test_only_a_shippable_run_on_a_worker_without_the_mount_transfers(
        data_root):
    assert not uses_transfer(_worker("mute", "cpu"), SHIPPABLE)
    bare = _speaking(_worker("bare", "cpu"))
    nas = _worker("nas", "cpu")
    nas.mounts = {"data": "/n"}
    assert uses_transfer(bare, SHIPPABLE)
    assert not uses_transfer(nas, SHIPPABLE)
    assert not uses_transfer(bare, NOT_SHIPPABLE)
    assert not uses_transfer(bare, {"mount": None})
    assert not uses_transfer(bare, None)


async def test_a_transferred_row_must_be_one_of_the_runs_own_keys(
        data_root, db):
    from app.core.run_service import RunService

    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", worker_id="gpu-1")
    service = RunService(store)
    active = SimpleNamespace(
        run_id=record.id,
        placement=Placement("gpu-1", ("cpu",), (("gpu-1", "cpu"),)))
    await service._record_artifact(active, TransferredArtifactSignal(
        kind="checkpoint", path=f"runs/{record.id}/checkpoint/a.pt"))
    await service._record_artifact(active, TransferredArtifactSignal(
        kind="checkpoint", path="runs/someone-else/checkpoint/a.pt"))
    [row] = await store.list_artifacts(record.id)
    assert (row.backend, row.worker_id, row.stored_path) == (
        "transfer", "gpu-1", f"runs/{record.id}/checkpoint/a.pt")


async def test_a_writers_file_is_placed_in_this_servers_data_folder(
        http, data_root):
    (data_root / "models" / "saved.pt").write_bytes(b"the user's own")
    headers = {TOKEN_HEADER: "token-gpu-1"}

    async def upload(data: bytes) -> Any:
        return await http.post(
            "/api/runs/r1/artifacts/upload", headers=headers,
            data={"kind": "output", "name": "models/saved.pt",
                  "place": "data"},
            files={"file": ("saved.pt", data, "application/octet-stream")})

    first = await upload(b"trained")
    assert first.status_code == 200, first.text
    assert first.json()["path"] == "models/saved-1.pt"
    assert (data_root / "models" / "saved.pt").read_bytes() == \
        b"the user's own"
    # A retry of the same content: 200, the same file.
    again = await upload(b"trained")
    assert again.status_code == 200 and again.json() == first.json()
    assert sorted(p.name for p in (data_root / "models").iterdir()
                  if p.is_file()) == ["saved-1.pt", "saved.pt"]
    bad = await http.post(
        "/api/runs/r1/artifacts/upload", headers=headers,
        data={"kind": "output", "name": "files/x.csv", "place": "data"},
        files={"file": ("x.csv", b"x", "application/octet-stream")})
    assert bad.status_code == 400
    assert not list(data_root.rglob(".incoming-*"))


async def test_the_fields_must_come_before_the_file(http, data_root):
    boundary = "b0undary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"a.pt\"\r\n\r\nweights\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"kind\"\r\n\r\ncheckpoint\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"name\"\r\n\r\na.pt\r\n--{boundary}--\r\n").encode()
    response = await http.post(
        "/api/runs/r1/artifacts/upload", content=body,
        headers={TOKEN_HEADER: "token-gpu-1",
                 "content-type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 400
    assert not list(data_root.rglob(".incoming-*"))
    plain = await http.post("/api/runs/r1/artifacts/upload", content=b"x",
                            headers={TOKEN_HEADER: "token-gpu-1",
                                     "content-type": "text/plain"})
    assert plain.status_code == 400


async def test_deleting_a_worker_run_removes_its_received_files(
        http, data_root, monkeypatch):
    store = app.state.run_service.store
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", worker_id="gpu-1")
    await store.mark_finished(record.id, "succeeded")
    run_dir = data_root / "runs" / record.id / "checkpoint"
    run_dir.mkdir(parents=True)
    (run_dir / "a.pt").write_bytes(b"x")
    (data_root / "models" / "kept.pt").write_bytes(b"k")
    service = app.state.run_service
    monkeypatch.setattr(service, "is_active", lambda _: False, raising=False)
    monkeypatch.setattr(service, "is_queued", lambda _: False, raising=False)
    response = await http.delete(f"/api/runs/{record.id}",
                                 headers={TOKEN_HEADER: session_token()})
    assert response.status_code == 200, response.text
    assert not (data_root / "runs" / record.id).exists()
    assert (data_root / "models" / "kept.pt").exists()
    # A local run's directory (its TensorBoard logs) is not this rule's.
    here = await store.create_run(graph_snapshot={"nodes": []},
                                  status="running")
    await store.mark_finished(here.id, "succeeded")
    (data_root / "runs" / here.id / "tb").mkdir(parents=True)
    response = await http.delete(f"/api/runs/{here.id}",
                                 headers={TOKEN_HEADER: session_token()})
    assert response.status_code == 200
    assert (data_root / "runs" / here.id / "tb").is_dir()


async def test_through_an_ssh_tunnel_the_transfer_routes_need_a_worker_token(
        http, data_root):
    """#784 + #785: a request through an SSH worker's own listener
    (``TunnelApp``) reaches the transfer backend's routes only with a
    valid worker token; nothing else exists there (404), whatever the
    token or the ``Host``."""
    from httpx import ASGITransport, AsyncClient

    from app.core.tunnel_listener import TunnelApp

    tunnel = {"host": "127.0.0.1:8000"}
    worker = {**tunnel, TOKEN_HEADER: "token-gpu-1"}
    main_http = http
    http = AsyncClient(transport=ASGITransport(app=TunnelApp(app, "gpu-1")),
                       base_url="http://127.0.0.1:47915")
    try:
        for path in ("/api/files/download/train.csv",
                     "/api/images/download/cat.png",
                     "/api/models/download/periodic/x.pt"):
            anonymous = await http.get(path, headers=tunnel)
            assert anonymous.status_code == 403, path
            forged = await http.get(path, headers={**tunnel,
                                                   TOKEN_HEADER: "forged"})
            assert forged.status_code == 403, path
            # Not the session token either: a worker's route.
            session = await http.get(path, headers={
                **tunnel, TOKEN_HEADER: session_token()})
            assert session.status_code == 403, path
            got = await http.get(path, headers=worker)
            assert got.status_code == 200, (path, got.text)
        assert (await http.get("/api/files/download/train.csv",
                               headers=worker)).content == b"a,b\n1,2\n"
        uploaded = await http.post(
            "/api/runs/r1/artifacts/upload", headers=worker,
            data={"kind": "checkpoint", "name": "a.pt"},
            files={"file": ("a.pt", b"w", "application/octet-stream")})
        assert uploaded.status_code == 200, uploaded.text
        refused = await http.post(
            "/api/runs/r1/artifacts/upload", headers=tunnel,
            data={"kind": "checkpoint", "name": "a.pt"},
            files={"file": ("a.pt", b"w", "application/octet-stream")})
        assert refused.status_code == 403
        # Nothing else exists through the tunnel, token or not: the
        # listings, the run API, the artifact download route (the agent
        # never uses it), a path that only starts like a download route,
        # the session bootstrap.
        for method, path in (("GET", "/api/files"), ("GET", "/api/runs"),
                             ("GET", "/api/runs/r1/artifacts/1/download"),
                             ("GET", "/api/files/download/"),
                             ("DELETE", "/api/files/train.csv"),
                             ("POST", "/api/runs/r1/cancel"),
                             ("GET", "/api/auth/bootstrap")):
            response = await http.request(method, path, headers=worker)
            assert response.status_code == 404, (method, path)
    finally:
        await http.aclose()
    # On the main port, a forged tunnel Host is just a disallowed Host.
    assert (await main_http.get("/api/files/download/train.csv",
                                headers={**worker, "host": "127.0.0.1:47915"})
            ).status_code == 421
