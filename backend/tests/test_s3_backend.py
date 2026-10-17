"""The ``s3`` artifact backend, control plane's half (#791).

``core.s3`` (keys, credentials, inputs, retention), which backend a worker
gets, ``WorkerExecutor``'s assignment and credential renewal, the rows and
the download route. The bucket is moto's in-process S3 (no network); the
tests that need it skip when moto is not installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import executors
from app.core import s3
from app.core import transfer
from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, session_token
from app.core.executors import (
    TransferredArtifactSignal,
    WorkerExecutor,
    storage_backend,
)
from app.core.mounts import translate
from app.core.run_store import RunStore
from app.core.scheduling import Placement, no_mount_backend, ships_files
from app.main import app
from tests import test_transfer_backend as _transfer_tests
from tests.test_transfer_backend import SHIPPABLE, _needs, _request
from tests.test_worker_executor import _Agent, _Recorder
from tests.test_worker_scheduler import _worker

CSV = b"a,b\n1,2\n"

#: The transfer backend's fixtures: this server's data folders, a database.
data_root = _transfer_tests.data_root
db = _transfer_tests.db


@pytest.fixture
def s3_on(monkeypatch):
    """S3 configured as for a MinIO bucket ``lab`` (static keys)."""
    for name, value in (("S3_ENABLED", True), ("S3_BUCKET", "lab"),
                        ("S3_ENDPOINT_URL", ""), ("S3_REGION", "us-east-1"),
                        ("S3_PREFIX", ""), ("S3_ACCESS_KEY_ID", "AK"),
                        ("S3_CREDENTIALS", "static"),
                        ("S3_DOWNLOAD", "proxy")):
        monkeypatch.setattr(settings, name, value)
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "S3_SECRET_ACCESS_KEY", SecretStr("SK"))
    s3._clients.clear()
    yield
    s3._clients.clear()


@pytest.fixture
def bucket(s3_on):
    """moto's S3 with the bucket ``lab``; the control plane's client."""
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        s3._clients.clear()
        client = s3.control_client()
        client.create_bucket(Bucket="lab")
        yield client
    s3._clients.clear()


def _keys(client: Any) -> list[str]:
    listed = client.list_objects_v2(Bucket="lab").get("Contents") or []
    return sorted(item["Key"] for item in listed)


def _put(client: Any, key: str, data: bytes = b"x") -> None:
    client.put_object(Bucket="lab", Key=key, Body=data)


# ── settings and keys ─────────────────────────────────────────────────────


def test_it_is_off_unless_asked_for_and_says_why(monkeypatch):
    monkeypatch.setattr(settings, "S3_ENABLED", False)
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    assert not s3.enabled() and s3.problem() is None
    monkeypatch.setattr(settings, "S3_ENABLED", True)
    assert s3.problem() == "CODEFYUI_S3_BUCKET is not set"
    monkeypatch.setattr(settings, "S3_BUCKET", "lab")
    monkeypatch.setattr(s3, "client_available", lambda: False)
    assert "boto3 is not installed" in s3.problem()
    assert not s3.enabled() and not s3.configured()
    monkeypatch.setattr(s3, "client_available", lambda: True)
    assert s3.enabled() and s3.problem() is None


def test_the_secret_key_never_shows_in_the_settings_repr(s3_on):
    assert "SK" not in repr(settings.S3_SECRET_ACCESS_KEY)
    assert "SK" not in str(settings.S3_SECRET_ACCESS_KEY)


def test_keys_sit_under_the_prefix_and_a_run_owns_only_its_own(
        s3_on, monkeypatch):
    assert s3.run_prefix("r1") == "runs/r1/"
    long_id = "a" * 100
    assert s3.run_prefix(long_id) == f"runs/{long_id}/"  # never shortened
    assert s3.owns_key(long_id, f"runs/{long_id}/checkpoint/{'b' * 80}.pt")
    assert s3.key_part("../we ird") == "we_ird"
    assert s3.input_key("ab" * 32) == f"inputs/{'ab' * 32}"
    monkeypatch.setattr(settings, "S3_PREFIX", "/codefyui/lab/")
    assert s3.key_prefix() == "codefyui/lab/"
    assert s3.run_prefix("r1") == "codefyui/lab/runs/r1/"
    assert s3.owns_key("r1", "codefyui/lab/runs/r1/checkpoint/a.pt")
    assert s3.owns_key("r1", "codefyui/lab/runs/r1/tensorboard/t")
    # A row's key carries its own prefix: one from before a prefix change
    # is still the run's; the prefix a run was assigned is checked when its
    # rows are recorded.
    assert s3.owns_key("r1", "runs/r1/checkpoint/a.pt")
    assert not s3.owns_key("r1", "runs/r1/checkpoint/a.pt",
                           "codefyui/lab/runs/r1/")
    assert s3.owns_key("r1", "codefyui/lab/runs/r1/checkpoint/a.pt",
                       "codefyui/lab/runs/r1/")
    for key in ("codefyui/lab/runs/r2/checkpoint/a.pt",
                "codefyui/lab/runs/r10/checkpoint/a.pt",
                "x/../runs/r1/checkpoint/a.pt",
                "codefyui/lab/runs/r1/a.pt",
                "codefyui/lab/runs/r1/checkpoint/../../r2/x",
                "codefyui/lab/runs/r1//a.pt",
                "codefyui/lab/runs/r1/checkpoint/a\\b"):
        assert not s3.owns_key("r1", key), key
    assert s3.uri("codefyui/lab/runs/r1/x/a.pt") == \
        "s3://lab/codefyui/lab/runs/r1/x/a.pt"
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    assert s3.uri("k") is None


def test_an_s3_row_is_named_by_its_uri(s3_on):
    assert translate("s3", None, "runs/r1/checkpoint/a.pt") == \
        "s3://lab/runs/r1/checkpoint/a.pt"


def test_the_session_policy_allows_the_runs_prefix_and_its_inputs():
    policy = json.loads(s3.session_policy("lab", "runs/r1/",
                                          ["inputs/aa", "inputs/bb"]))
    write, read = policy["Statement"]
    assert write["Resource"] == ["arn:aws:s3:::lab/runs/r1/*"]
    assert "s3:PutObject" in write["Action"]
    assert read == {"Effect": "Allow", "Action": ["s3:GetObject"],
                    "Resource": ["arn:aws:s3:::lab/inputs/aa",
                                 "arn:aws:s3:::lab/inputs/bb"]}
    many = json.loads(s3.session_policy(
        "lab", "runs/r1/", [f"inputs/{n:064x}" for n in range(40)]))
    assert many["Statement"][1]["Resource"] == ["arn:aws:s3:::lab/inputs/*"]
    alone = json.loads(s3.session_policy("lab", "runs/r1/", []))
    assert len(alone["Statement"]) == 1


def test_error_codes_are_read_through_a_wrapping_exception():
    from botocore.exceptions import ClientError

    inner = ClientError({"Error": {"Code": "AccessDenied",
                                   "Message": "no"},
                         "ResponseMetadata": {"HTTPStatusCode": 403}},
                        "PutObject")
    try:
        try:
            raise inner
        except ClientError:
            raise RuntimeError("Failed to upload") from None
    except RuntimeError as outer:
        assert s3.error_code(outer) == ("AccessDenied", 403)
    assert s3.describe(inner) == "AccessDenied (HTTP 403): no"
    assert s3.error_code(ValueError("x")) == (None, None)
    assert s3.describe(ValueError("x")) == "ValueError: x"


# ── credentials ───────────────────────────────────────────────────────────


def test_static_credentials_are_this_servers_keys(s3_on):
    access = s3.issue_access("r1", [])
    assert (access.bucket, access.prefix) == ("lab", "runs/r1/")
    assert (access.access_key_id, access.secret_access_key) == ("AK", "SK")
    assert access.session_token is None and access.expires_at is None
    # Never in a repr (a log line), never in a validation error.
    assert "SK" not in repr(access)
    with pytest.raises(Exception) as caught:
        wp.S3Access.model_validate({"bucket": "", "prefix": "p",
                                    "access_key_id": "AK",
                                    "secret_access_key": "TOPSECRET"})
    assert "TOPSECRET" not in str(caught.value)


def test_sts_gives_temporary_credentials(bucket, monkeypatch):
    monkeypatch.setattr(settings, "S3_CREDENTIALS", "sts")
    access = s3.issue_access("r1", ["inputs/aa"])
    assert access.session_token and access.expires_at
    assert access.access_key_id != "AK"


def test_auto_falls_back_to_static_and_sts_alone_refuses(s3_on, monkeypatch):
    class _NoSts:
        def assume_role(self, **_: Any) -> Any:
            raise RuntimeError("not supported")

    asked: list[int] = []

    class _CountingNoSts(_NoSts):
        def assume_role(self, **kw: Any) -> Any:
            asked.append(1)
            return super().assume_role(**kw)

    monkeypatch.setattr(s3, "control_client",
                        lambda service="s3": _CountingNoSts())
    monkeypatch.setattr(s3, "_sts_failed", None)
    monkeypatch.setattr(settings, "S3_CREDENTIALS", "auto")
    access = s3.issue_access("r1")
    assert access.access_key_id == "AK" and access.expires_at is None
    # A refusal is remembered: the next runs do not ask STS again ...
    s3.issue_access("r2")
    assert len(asked) == 1
    # ... until the settings change, or the pause is over.
    monkeypatch.setattr(settings, "S3_STS_ROLE_ARN", "arn:aws:iam::1:role/x")
    s3.issue_access("r3")
    assert len(asked) == 2
    monkeypatch.setattr(s3, "STS_RETRY_AFTER_S", 0.0)
    s3.issue_access("r4")
    assert len(asked) == 3
    monkeypatch.setattr(settings, "S3_CREDENTIALS", "sts")
    with pytest.raises(s3.S3Unavailable, match="STS AssumeRole failed"):
        s3.issue_access("r1")


# ── inputs, reading, retention ────────────────────────────────────────────


def test_an_input_is_uploaded_once_by_content(bucket, data_root,
                                              monkeypatch):
    sha = hashlib.sha256(CSV).hexdigest()
    manifest = [{"kind": "data_file", "name": "train.csv", "sha256": sha,
                 "bytes": len(CSV)}]
    [entry] = s3.ensure_inputs(manifest)
    assert entry["key"] == f"inputs/{sha}"
    got = bucket.get_object(Bucket="lab", Key=entry["key"])
    assert got["Body"].read() == CSV
    assert got["Metadata"] == {"sha256": sha}
    calls: list[str] = []
    real = bucket.upload_file
    monkeypatch.setattr(bucket, "upload_file",
                        lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    s3.ensure_inputs(manifest)
    assert calls == []  # already there
    (data_root / "files" / "train.csv").unlink()
    with pytest.raises(s3.S3Unavailable, match="not an upload"):
        s3.ensure_inputs([{**manifest[0], "sha256": "cd" * 32}])


def test_an_unreachable_bucket_is_reported(s3_on, data_root, monkeypatch):
    from botocore.exceptions import ClientError

    class _Denied:
        def head_object(self, **_: Any) -> Any:
            raise ClientError({"Error": {"Code": "AccessDenied"},
                               "ResponseMetadata": {"HTTPStatusCode": 403}},
                              "HeadObject")

    monkeypatch.setattr(s3, "control_client", lambda service="s3": _Denied())
    with pytest.raises(s3.S3Unavailable, match="AccessDenied"):
        s3.ensure_inputs([{"kind": "data_file", "name": "train.csv",
                           "sha256": "ab" * 32, "bytes": 8}])
    with pytest.raises(s3.S3Unavailable):
        s3.head("runs/r1/x/a")


def test_objects_are_read_and_presigned(bucket):
    _put(bucket, "runs/r1/checkpoint/a.pt", b"weights")
    _put(bucket, "runs/r1/tensorboard/t/events.1", b"e")
    assert s3.head("runs/r1/checkpoint/a.pt")["ContentLength"] == 7
    assert s3.head("runs/r1/checkpoint/gone.pt") is None
    assert s3.is_folder("runs/r1/tensorboard/t")
    assert not s3.is_folder("runs/r1/checkpoint/a.pt")
    assert b"".join(s3.open_stream("runs/r1/checkpoint/a.pt")) == b"weights"
    with pytest.raises(s3.S3Unavailable):
        s3.open_stream("runs/r1/checkpoint/gone.pt")
    url = s3.presigned_url("runs/r1/checkpoint/a.pt", "a.pt")
    assert "runs/r1/checkpoint/a.pt" in url and "Signature" in url
    assert s3.filename("runs/r1/checkpoint/a.pt") == "a.pt"


def test_removing_a_runs_objects_keeps_writers_files_and_a_resume_point(
        bucket):
    for key in ("runs/r1/checkpoint/i.pt", "runs/r1/checkpoint/p.pt",
                "runs/r1/tensorboard/t/e", "runs/r1/output/models/best.pt",
                "runs/r10/checkpoint/x.pt", "inputs/aa"):
        _put(bucket, key)
    bucket.create_multipart_upload(Bucket="lab", Key="runs/r1/checkpoint/big")
    assert s3.remove_run_objects("r1", keep_checkpoints=True,
                                 prefix="runs/r1/") == 1
    assert _keys(bucket) == ["inputs/aa", "runs/r1/checkpoint/i.pt",
                             "runs/r1/checkpoint/p.pt",
                             "runs/r1/output/models/best.pt",
                             "runs/r10/checkpoint/x.pt"]
    assert not bucket.list_multipart_uploads(Bucket="lab").get("Uploads")
    assert s3.remove_runs_objects([("r1", False, "lab", "runs/r1/"),
                                   ("r10", False, "lab", "runs/r10/")]) == \
        (["r1", "r10"], [])
    assert _keys(bucket) == ["inputs/aa", "runs/r1/output/models/best.pt"]


def test_a_bucket_that_cannot_be_reached_never_breaks_retention(
        s3_on, monkeypatch):
    def broken(service: str = "s3") -> Any:
        raise s3.S3Unavailable("down")

    monkeypatch.setattr(s3, "control_client", broken)
    assert s3.remove_runs_objects([("r1", False, "lab", "runs/r1/")]) == \
        ([], ["r1: S3Unavailable: down"])


async def test_retention_removes_the_s3_objects_of_the_runs_it_drops(
        bucket, data_root, db):
    store = RunStore(db)

    async def run(status: str) -> str:
        record = await store.create_run(graph_snapshot={"nodes": []},
                                        status="running", worker_id="gpu-1")
        for kind in ("checkpoint", "output"):
            key = f"runs/{record.id}/{kind}/a.pt"
            _put(bucket, key)
            await store.add_artifact(record.id, kind, key, backend="s3",
                                     worker_id="gpu-1")
        await store.mark_finished(record.id, status)
        return record.id

    done = await run("succeeded")
    stopped = await run("interrupted")
    [row, _] = await store.list_artifacts(done)
    assert row.backend == "s3"
    assert row.path == f"s3://lab/runs/{done}/checkpoint/a.pt"
    assert await store.prune(keep_last=0) == 2
    assert _keys(bucket) == sorted([
        f"runs/{done}/output/a.pt", f"runs/{stopped}/checkpoint/a.pt",
        f"runs/{stopped}/output/a.pt"])


async def test_retention_without_the_bucket_keeps_going(s3_on, data_root, db,
                                                        monkeypatch, caplog):
    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", worker_id="gpu-1")
    await store.add_artifact(record.id, "checkpoint",
                             f"runs/{record.id}/checkpoint/a.pt",
                             backend="s3", worker_id="gpu-1")
    await store.mark_finished(record.id, "succeeded")
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    assert await store.prune(keep_last=0) == 1
    assert "objects stay" in caplog.text


# ── which backend a worker gets ───────────────────────────────────────────


def _caps(worker: Any, *names: str) -> Any:
    worker.capabilities = names
    return worker


def test_s3_goes_first_to_an_agent_that_speaks_it(s3_on, monkeypatch):
    both = _caps(_worker("both", "cpu"), "transfer", "s3")
    old = _caps(_worker("old", "cpu"), "transfer")
    mute = _caps(_worker("mute", "cpu"))
    assert no_mount_backend(both) == "s3"
    assert no_mount_backend(old) == "transfer"
    assert no_mount_backend(mute) is None
    assert storage_backend(both, SHIPPABLE) == "s3"
    assert storage_backend(old, SHIPPABLE) == "transfer"
    assert storage_backend(both, {"mount": None}) is None
    monkeypatch.setattr(settings, "S3_ENABLED", False)
    assert no_mount_backend(both) == "transfer"
    # Transfer off, S3 on: only an agent that speaks S3 takes the run.
    monkeypatch.setattr(settings, "S3_ENABLED", True)
    monkeypatch.setattr(settings, "TRANSFER_ENABLED", False)
    assert ships_files(both) and not ships_files(old)
    assert transfer.check({"mount": "data", "inputs": [], "paths": []})["ok"]
    monkeypatch.setattr(settings, "S3_ENABLED", False)
    assert not transfer.check({"mount": "data", "inputs": [],
                               "paths": []})["ok"]


# ── the assignment ────────────────────────────────────────────────────────


async def test_an_s3_assignment_names_its_inputs_by_object(
        bucket, data_root, db):
    from app.core.workers import WorkerRegistry

    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _caps(agent.worker, *wp.CAPABILITIES)
        recorder = _Recorder()
        needs = transfer.annotate(_needs(("data_file", "train.csv")))
        executor = WorkerExecutor(registry,
                                  content_index=transfer.ContentIndex(db))
        task = asyncio.create_task(executor.run(_request(recorder, needs)))
        assign = await agent.wait_for(wp.Assign)
        storage = assign.storage
        sha = hashlib.sha256(CSV).hexdigest()
        assert storage.backend == "s3" and storage.relative_artifacts is False
        assert storage.inputs == [wp.TransferInput(
            kind="data_file", name="train.csv", bytes=8, sha256=sha,
            key=f"inputs/{sha}")]
        assert storage.s3.bucket == "lab"
        assert storage.s3.prefix == "runs/r1/"
        assert bucket.get_object(Bucket="lab",
                                 Key=f"inputs/{sha}")["Body"].read() == CSV
        for frame in (wp.Artifact(run_id="r1", seq=1, kind="checkpoint",
                                  path="runs/r1/checkpoint/a.pt",
                                  backend="s3"),
                      wp.RunFinished(run_id="r1", seq=2,
                                     status="succeeded")):
            registry.route(agent.worker, frame)
        assert (await task).status == "succeeded"
        signal = [c[1] for c in recorder.calls if c[0] == "artifact"][0]
        assert isinstance(signal, TransferredArtifactSignal)
        assert (signal.backend, signal.path) == ("s3",
                                                 "runs/r1/checkpoint/a.pt")
    finally:
        await registry.shutdown()


async def test_a_bucket_that_refuses_fails_the_run_unassigned(
        s3_on, data_root, db, monkeypatch):
    from app.core.workers import WorkerRegistry

    def refuse(manifest: Any) -> Any:
        raise s3.S3Unavailable("bucket lab: AccessDenied (HTTP 403)")

    monkeypatch.setattr(s3, "ensure_inputs", refuse)
    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _caps(agent.worker, *wp.CAPABILITIES)
        needs = transfer.annotate(_needs(("data_file", "train.csv")))
        outcome = await WorkerExecutor(
            registry, content_index=transfer.ContentIndex(db)).run(
                _request(_Recorder(), needs))
        assert outcome.status == "failed"
        assert outcome.error == ("cannot send this run's input files to the "
                                 "S3 bucket: bucket lab: AccessDenied "
                                 "(HTTP 403)")
        assert agent.sent == []
    finally:
        await registry.shutdown()


async def test_an_agent_without_s3_gets_the_transfer_backend_with_a_warning(
        s3_on, data_root, db):
    from app.core.workers import WorkerRegistry

    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _caps(agent.worker, "ack", "mounts", "transfer")
        recorder = _Recorder()
        needs = transfer.annotate(_needs(("data_file", "train.csv")))
        task = asyncio.create_task(WorkerExecutor(
            registry, content_index=transfer.ContentIndex(db)).run(
                _request(recorder, needs)))
        assign = await agent.wait_for(wp.Assign)
        assert assign.storage.backend == "transfer"
        registry.route(agent.worker, wp.RunFinished(run_id="r1", seq=1,
                                                    status="succeeded"))
        await task
        [warning] = [c for c in recorder.calls if c[0] == "emit"]
        assert warning[2]["kind"] == "storage_fallback"
        assert "no boto3" in warning[2]["detail"]
    finally:
        await registry.shutdown()


async def test_temporary_credentials_are_renewed_while_the_run_lasts(
        s3_on, data_root, db, monkeypatch):
    from app.core.workers import WorkerRegistry

    issued: list[float] = []

    def issue(run_id: str, input_keys: Any = ()) -> Any:
        import time

        issued.append(time.time())
        return wp.S3Access(bucket="lab", prefix=f"runs/{run_id}/",
                           access_key_id=f"tmp{len(issued)}",
                           secret_access_key="s", session_token="t",
                           expires_at=time.time() + 0.3)

    monkeypatch.setattr(s3, "issue_access", issue)
    monkeypatch.setattr(executors, "REFRESH_MIN_S", 0.01)
    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _caps(agent.worker, *wp.CAPABILITIES)
        needs = transfer.annotate(_needs())
        task = asyncio.create_task(WorkerExecutor(registry).run(
            _request(_Recorder(), needs)))
        assign = await agent.wait_for(wp.Assign)
        assert assign.storage.s3.access_key_id == "tmp1"
        renewed = await agent.wait_for(wp.Credentials)
        assert renewed.run_id == "r1"
        assert renewed.s3.access_key_id == "tmp2"
        registry.route(agent.worker, wp.RunFinished(run_id="r1", seq=1,
                                                    status="succeeded"))
        await task
        count = len(issued)
        await asyncio.sleep(0.5)
        assert len(issued) == count  # stopped with the run
    finally:
        await registry.shutdown()


async def test_a_failed_renewal_is_retried(s3_on, monkeypatch):
    import time

    calls: list[int] = []

    def issue(run_id: str, input_keys: Any = ()) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise s3.S3Unavailable("sts down")
        return wp.S3Access(bucket="lab", prefix="runs/r1/",
                           access_key_id="new", secret_access_key="s")

    sent: list[Any] = []

    async def send(worker_id: str, frame: Any) -> bool:
        sent.append(frame)
        return True

    monkeypatch.setattr(s3, "issue_access", issue)
    monkeypatch.setattr(executors, "REFRESH_MIN_S", 0.0)
    monkeypatch.setattr(executors, "REFRESH_RETRY_S", 0.01)
    executor = WorkerExecutor(SimpleNamespace(get=lambda _: None))
    monkeypatch.setattr(executor, "_send", send)
    old = wp.S3Access(bucket="lab", prefix="runs/r1/", access_key_id="o",
                      secret_access_key="s", expires_at=time.time() + 900)
    executor._s3_runs["r1"] = executors._S3Run("lab", "runs/r1/", old)
    monkeypatch.setattr(executors, "REFRESH_AT", 0.0)
    request = SimpleNamespace(run_id="r1")
    started = time.monotonic()
    await asyncio.wait_for(executor._refresh_credentials(request, "box", []),
                           5)
    # Retried after REFRESH_RETRY_S, not after 2/3 of what was left.
    assert time.monotonic() - started < 2
    assert len(calls) == 2
    [frame] = sent
    assert frame.s3.access_key_id == "new"  # static now: no more renewals


# ── rows and the routes ───────────────────────────────────────────────────


async def test_an_s3_row_must_be_one_of_the_runs_own_keys(s3_on, db):
    from app.core.run_service import RunService

    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", worker_id="gpu-1")
    service = RunService(store)
    active = SimpleNamespace(
        run_id=record.id,
        placement=Placement("gpu-1", ("cpu",), (("gpu-1", "cpu"),)))
    await service._record_artifact(active, TransferredArtifactSignal(
        kind="checkpoint", path=f"runs/{record.id}/checkpoint/a.pt",
        backend="s3"))
    await service._record_artifact(active, TransferredArtifactSignal(
        kind="checkpoint", path="runs/someone-else/checkpoint/a.pt",
        backend="s3"))
    [row] = await store.list_artifacts(record.id)
    assert (row.backend, row.stored_path) == (
        "s3", f"runs/{record.id}/checkpoint/a.pt")
    from app.core.run_events import artifact_payload

    payload = artifact_payload(row, None)
    assert payload["backend"] == "s3"
    assert payload["relative_path"] == f"runs/{record.id}/checkpoint/a.pt"
    assert payload["path"] == f"s3://lab/runs/{record.id}/checkpoint/a.pt"


@pytest.fixture
async def http(data_root, db, monkeypatch):
    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", run_id="r1",
                                    worker_id="gpu-1")
    await store.mark_finished(record.id, "succeeded")
    service = SimpleNamespace(store=store, running_on=lambda _: None,
                              is_active=lambda _: False,
                              is_queued=lambda _: False)
    monkeypatch.setattr(app.state, "run_service", service, raising=False)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://127.0.0.1:8000") as client:
        client.store = store  # type: ignore[attr-defined]
        yield client


async def test_an_s3_file_downloads_through_this_server_or_a_presigned_url(
        bucket, http, monkeypatch):
    store = http.store
    _put(bucket, "runs/r1/output/models/best.pt", b"trained weights")
    _put(bucket, "runs/r1/tensorboard/t/events", b"e")
    row = await store.add_artifact("r1", "output",
                                   "runs/r1/output/models/best.pt",
                                   backend="s3", worker_id="gpu-1")
    logdir = await store.add_artifact("r1", "tensorboard",
                                      "runs/r1/tensorboard/t",
                                      backend="s3", worker_id="gpu-1")
    gone = await store.add_artifact("r1", "checkpoint",
                                    "runs/r1/checkpoint/gone.pt",
                                    backend="s3", worker_id="gpu-1")
    stray = await store.add_artifact("r1", "checkpoint",
                                     "runs/r2/checkpoint/a.pt",
                                     backend="s3", worker_id="gpu-1")
    response = await http.get(f"/api/runs/r1/artifacts/{row.id}/download")
    assert response.status_code == 200
    assert response.content == b"trained weights"
    assert response.headers["content-disposition"] == \
        'attachment; filename="best.pt"'
    assert response.headers["content-length"] == "15"
    assert (await http.get(f"/api/runs/r1/artifacts/{logdir.id}/download")
            ).status_code == 400
    assert (await http.get(f"/api/runs/r1/artifacts/{gone.id}/download")
            ).status_code == 404
    assert (await http.get(f"/api/runs/r1/artifacts/{stray.id}/download")
            ).status_code == 404
    monkeypatch.setattr(settings, "S3_DOWNLOAD", "presign")
    redirect = await http.get(f"/api/runs/r1/artifacts/{row.id}/download")
    assert redirect.status_code == 307
    assert "runs/r1/output/models/best.pt" in redirect.headers["location"]
    listed = (await http.get("/api/runs/r1/artifacts")).json()["artifacts"]
    assert listed[0]["backend"] == "s3"
    assert listed[0]["path"] == "s3://lab/runs/r1/output/models/best.pt"
    assert listed[0]["relative_path"] == "runs/r1/output/models/best.pt"


async def test_a_download_says_when_the_bucket_cannot_be_reached(
        s3_on, http, monkeypatch):
    row = await http.store.add_artifact("r1", "checkpoint",
                                        "runs/r1/checkpoint/a.pt",
                                        backend="s3", worker_id="gpu-1")
    url = f"/api/runs/r1/artifacts/{row.id}/download"

    def down(key: str, bucket: Any = None) -> Any:
        raise s3.S3Unavailable("EndpointConnectionError")

    monkeypatch.setattr(s3, "head", down)
    response = await http.get(url)
    assert response.status_code == 503
    assert "EndpointConnectionError" in response.json()["detail"]
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    response = await http.get(url)
    assert response.status_code == 503
    assert "CODEFYUI_S3_BUCKET is not set" in response.json()["detail"]


async def test_a_non_ascii_name_downloads_with_an_encoded_filename(
        bucket, http):
    _put(bucket, "runs/r1/output/models/模型.pt", b"w")
    row = await http.store.add_artifact("r1", "output",
                                        "runs/r1/output/models/模型.pt",
                                        backend="s3", worker_id="gpu-1")
    response = await http.get(f"/api/runs/r1/artifacts/{row.id}/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith(
        "attachment; filename*=utf-8''")


async def test_deleting_a_run_removes_its_s3_objects_but_writers_files(
        bucket, http):
    store = http.store
    for kind in ("checkpoint", "output"):
        key = f"runs/r1/{kind}/a.pt"
        _put(bucket, key)
        await store.add_artifact("r1", kind, key, backend="s3",
                                 worker_id="gpu-1")
    response = await http.delete("/api/runs/r1",
                                 headers={TOKEN_HEADER: session_token()})
    assert response.status_code == 200, response.text
    assert _keys(bucket) == ["runs/r1/output/a.pt"]


# ── review fixes (#836) ───────────────────────────────────────────────────


async def test_renewal_survives_a_failed_send_and_a_reconnect(
        s3_on, monkeypatch):
    """A send that raises (the socket closed after the online check) does
    not end the renewals, and a reconnect re-sends the latest credentials."""
    import time

    issued: list[str] = []

    def issue(run_id: str, input_keys: Any = ()) -> Any:
        issued.append(f"k{len(issued) + 1}")
        return wp.S3Access(bucket="lab", prefix="runs/r1/",
                           access_key_id=issued[-1], secret_access_key="s",
                           expires_at=time.time() + 0.05)

    attempts: list[str] = []

    async def send(worker_id: str, frame: Any) -> bool:
        attempts.append(frame.s3.access_key_id)
        if len(attempts) == 1:
            raise ConnectionError("socket closed")
        return True

    monkeypatch.setattr(s3, "issue_access", issue)
    monkeypatch.setattr(executors, "REFRESH_MIN_S", 0.0)
    monkeypatch.setattr(executors, "CANCEL_POLL_S", 0.0)
    executor = WorkerExecutor(SimpleNamespace(get=lambda _: None))
    monkeypatch.setattr(executor, "_send", send)
    executor._s3_runs["r1"] = executors._S3Run(
        "lab", "runs/r1/", wp.S3Access(bucket="lab", prefix="runs/r1/",
                                       access_key_id="k0",
                                       secret_access_key="s",
                                       expires_at=time.time()))
    task = asyncio.create_task(executor._refresh_credentials(
        SimpleNamespace(run_id="r1"), "box", []))
    while len(issued) < 3:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert attempts[:2] == ["k1", "k1"]  # the failed send went again
    assert "k2" in attempts
    # After a reconnect the latest goes out once more.
    sent: list[Any] = []

    async def record(worker_id: str, frame: Any) -> bool:
        sent.append(frame)
        return True

    monkeypatch.setattr(executor, "_send", record)
    request = SimpleNamespace(
        run_id="r1", context=SimpleNamespace(should_stop=lambda: False),
        sinks=_Recorder().sinks())

    class _NoAcks:
        async def resend(self) -> None:
            return None

    await executor._reconnected(request, "box", _NoAcks())
    [frame] = [f for f in sent if isinstance(f, wp.Credentials)]
    assert frame.s3.access_key_id == executor._s3_runs["r1"].access \
        .access_key_id


async def test_rows_keep_their_bucket_and_prefix_after_a_settings_change(
        bucket, http, monkeypatch):
    """A row records its bucket and its key carries its prefix: listing,
    download and Delete keep finding the object after both settings
    change."""
    _put(bucket, "old/runs/r1/output/a.pt", b"kept")
    _put(bucket, "old/runs/r1/checkpoint/c.pt", b"gone")
    bucket.create_bucket(Bucket="lab2")
    store = http.store
    for kind, key in (("output", "old/runs/r1/output/a.pt"),
                      ("checkpoint", "old/runs/r1/checkpoint/c.pt")):
        await store.add_artifact("r1", kind, key, backend="s3",
                                 bucket="lab", worker_id="gpu-1")
    monkeypatch.setattr(settings, "S3_BUCKET", "lab2")
    monkeypatch.setattr(settings, "S3_PREFIX", "new/")
    listed = (await http.get("/api/runs/r1/artifacts")).json()["artifacts"]
    assert listed[0]["path"] == "s3://lab/old/runs/r1/output/a.pt"
    response = await http.get(f"/api/runs/r1/artifacts/{listed[0]['id']}"
                              "/download")
    assert response.status_code == 200 and response.content == b"kept"
    deleted = await http.delete("/api/runs/r1",
                                headers={TOKEN_HEADER: session_token()})
    assert deleted.json() == {"run_id": "r1", "deleted": True}
    assert _keys(bucket) == ["old/runs/r1/output/a.pt"]


async def test_retention_uses_the_rows_bucket_and_prefix(bucket, data_root,
                                                        db, monkeypatch):
    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", worker_id="gpu-1")
    key = f"old/runs/{record.id}/checkpoint/a.pt"
    _put(bucket, key)
    await store.add_artifact(record.id, "checkpoint", key, backend="s3",
                             bucket="lab", worker_id="gpu-1")
    await store.mark_finished(record.id, "succeeded")
    monkeypatch.setattr(settings, "S3_BUCKET", "elsewhere")
    monkeypatch.setattr(settings, "S3_PREFIX", "new/")
    assert await store.prune(keep_last=0) == 1
    assert _keys(bucket) == []


async def test_a_row_is_recorded_with_its_bucket_and_assigned_prefix(
        s3_on, db):
    from app.core.run_service import RunService

    store = RunStore(db)
    record = await store.create_run(graph_snapshot={"nodes": []},
                                    status="running", worker_id="gpu-1")
    service = RunService(store)
    active = SimpleNamespace(
        run_id=record.id,
        placement=Placement("gpu-1", ("cpu",), (("gpu-1", "cpu"),)))
    prefix = f"p/runs/{record.id}/"
    for path in (f"{prefix}checkpoint/a.pt",
                 f"runs/{record.id}/checkpoint/b.pt"):  # not its prefix
        await service._record_artifact(active, TransferredArtifactSignal(
            kind="checkpoint", path=path, backend="s3", bucket="lab9",
            prefix=prefix))
    [row] = await store.list_artifacts(record.id)
    assert (row.bucket, row.mount_label, row.stored_path) == (
        "lab9", None, f"{prefix}checkpoint/a.pt")
    assert row.path == f"s3://lab9/{prefix}checkpoint/a.pt"


async def test_delete_says_when_the_objects_stay(s3_on, http, monkeypatch,
                                                 caplog):
    await http.store.add_artifact("r1", "checkpoint",
                                  "runs/r1/checkpoint/a.pt", backend="s3",
                                  bucket="lab", worker_id="gpu-1")
    monkeypatch.setattr(s3, "client_available", lambda: False)
    response = await http.delete("/api/runs/r1",
                                 headers={TOKEN_HEADER: session_token()})
    assert response.status_code == 200
    assert "stay" in response.json()["warning"]
    assert "boto3" in response.json()["warning"]
    assert "S3 objects stay" in caplog.text


async def test_a_delete_reports_a_bucket_that_refuses(s3_on, http,
                                                      monkeypatch):
    await http.store.add_artifact("r1", "checkpoint",
                                  "runs/r1/checkpoint/a.pt", backend="s3",
                                  bucket="lab", worker_id="gpu-1")

    def broken(service: str = "s3") -> Any:
        raise s3.S3Unavailable("down")

    monkeypatch.setattr(s3, "control_client", broken)
    response = await http.delete("/api/runs/r1",
                                 headers={TOKEN_HEADER: session_token()})
    assert response.status_code == 200
    assert "r1: S3Unavailable: down" in response.json()["warning"]


def test_a_presigned_download_encodes_a_non_ascii_name(bucket):
    from urllib.parse import parse_qs, urlparse

    url = s3.presigned_url("runs/r1/checkpoint/模型-e10.pt", "模型-e10.pt")
    [disposition] = parse_qs(urlparse(url).query)[
        "response-content-disposition"]
    assert disposition == ("attachment; filename*=utf-8''"
                           "%E6%A8%A1%E5%9E%8B-e10.pt")
    assert disposition.isascii()
    assert s3.content_disposition("a.pt") == 'attachment; filename="a.pt"'
    assert s3.content_disposition('a"b.pt').startswith(
        "attachment; filename*=utf-8''")


def test_locations_read_each_rows_bucket_and_prefix(s3_on):
    rows = [("r1", "lab", "old/runs/r1/checkpoint/a.pt"),
            ("r1", "lab", "old/runs/r1/output/b.pt"),
            ("r1", None, "runs/r1/checkpoint/c.pt"),
            ("r1", "lab", "not-a-run-key")]
    assert s3.locations(rows) == [("r1", "lab", "old/runs/r1/"),
                                  ("r1", "lab", "runs/r1/")]


def test_migration_014_adds_a_nullable_bucket_column(tmp_path):
    """014 stands alone: applied to a bare artifacts table (whatever came
    before it), it adds ``bucket`` and backfills nothing."""
    import sqlite3

    from app.core.migrations import MIGRATION_014, MIGRATIONS, iter_statements

    assert MIGRATIONS[-1] is MIGRATION_014
    conn = sqlite3.connect(tmp_path / "bare.db")
    conn.execute("CREATE TABLE exec_run_artifacts (id INTEGER PRIMARY KEY, "
                 "path TEXT, mount_label TEXT)")
    conn.execute("INSERT INTO exec_run_artifacts (path, mount_label) "
                 "VALUES ('a', 'lab')")
    for statement in iter_statements(MIGRATION_014):
        conn.execute(statement)
    assert conn.execute("SELECT path, mount_label, bucket FROM "
                        "exec_run_artifacts").fetchall() == [("a", "lab",
                                                              None)]
    conn.close()


async def test_the_bucket_is_its_own_column(db):
    """An ``s3`` row keeps its bucket in ``bucket``, never ``mount_label``;
    every other row has no bucket."""
    store = RunStore(db)
    run_id = (await store.create_run(graph_snapshot={"nodes": []},
                                     status="running",
                                     worker_id="gpu-1")).id
    s3_row = await store.add_artifact(
        run_id, "checkpoint", f"runs/{run_id}/checkpoint/a.pt", backend="s3",
        bucket="lab", mount_label="lab", worker_id="gpu-1")
    mount_row = await store.add_artifact(
        run_id, "checkpoint", "c/a.pt", backend="mount", mount_label="shared",
        bucket="lab", worker_id="gpu-1")
    assert (s3_row.bucket, s3_row.mount_label) == ("lab", None)
    assert (mount_row.bucket, mount_row.mount_label) == (None, "shared")
    listed = {row.backend: row
              for row in await store.list_artifacts(run_id)}
    assert (listed["s3"].bucket, listed["s3"].mount_label) == ("lab", None)
    key = f"runs/{run_id}/checkpoint/a.pt"
    assert listed["s3"].path == f"s3://lab/{key}"
    assert listed["mount"].bucket is None
    assert await store.s3_objects(run_id) == [(run_id, "lab", key)]


async def test_the_listing_reports_the_bucket(bucket, http):
    await http.store.add_artifact("r1", "output", "runs/r1/output/a.pt",
                                  backend="s3", bucket="lab",
                                  worker_id="gpu-1")
    [row] = (await http.get("/api/runs/r1/artifacts")).json()["artifacts"]
    assert (row["backend"], row["bucket"], row["mount_label"],
            row["relative_path"]) == ("s3", "lab", None,
                                      "runs/r1/output/a.pt")
