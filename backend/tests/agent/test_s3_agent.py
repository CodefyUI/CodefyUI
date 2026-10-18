"""``app.agent.s3``: the agent's half of the S3 artifact backend (#791).

``S3Client`` against moto's in-process S3 (no network; skipped without
moto): inputs fetched by object key into the per-worker cache, outputs put
under the run's prefix, the retry rules, renewed credentials. Then the
Runner driving a whole ``s3`` run with a cheap child, the same child the
transfer backend's test uses, and the agent's capability and dispatch.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from app.agent import s3 as agent_s3
from app.agent import transfer as tr
from app.agent.runner import Runner
from app.core import worker_protocol as wp

from . import _helpers as h

CSV = b"a,b\n1,2\n"
SHA = hashlib.sha256(CSV).hexdigest()


def _access(**changes: Any) -> wp.S3Access:
    fields = dict(bucket="lab", prefix="runs/r1/", access_key_id="AK",
                  secret_access_key="SK", region="us-east-1")
    fields.update(changes)
    return wp.S3Access(**fields)


@pytest.fixture
def bucket():
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        from app.core.s3 import make_client

        client = make_client(endpoint_url=None, region="us-east-1",
                             addressing_style="auto", access_key_id="AK",
                             secret_access_key="SK")
        client.create_bucket(Bucket="lab")
        client.put_object(Bucket="lab", Key=f"inputs/{SHA}", Body=CSV)
        yield client


def _entry(key: str | None = f"inputs/{SHA}") -> wp.TransferInput:
    return wp.TransferInput(kind="data_file", name="train.csv",
                            sha256=SHA, bytes=len(CSV), key=key)


def _client(tmp_path: Path, **kw: Any) -> agent_s3.S3Client:
    return agent_s3.S3Client(_access(), tr.InputCache(tmp_path / "cache"),
                             **kw)


# ── fetch and upload ──────────────────────────────────────────────────────


def test_an_input_is_fetched_by_its_key_once(bucket, tmp_path):
    client = _client(tmp_path)
    path, fetched = client.fetch(_entry())
    assert fetched and path.read_bytes() == CSV
    assert client.fetch(_entry()) == (path, False)  # cached
    with client.session() as session:
        assert session is not None
    with pytest.raises(tr.TransferError, match="names no S3 object"):
        _client(tmp_path / "other").fetch(_entry(key=None))


def test_a_missing_or_changed_object_fails_the_fetch(bucket, tmp_path):
    with pytest.raises(tr.TransferError, match="NoSuchKey"):
        _client(tmp_path).fetch(_entry(key="inputs/gone"))
    bucket.put_object(Bucket="lab", Key=f"inputs/{SHA}", Body=b"changed")
    with pytest.raises(tr.TransferError, match="hashes to"):
        _client(tmp_path).fetch(_entry())


def test_outputs_go_under_the_runs_prefix_with_their_hash(bucket, tmp_path):
    client = _client(tmp_path)
    source = tmp_path / "ck.pt"
    source.write_bytes(b"weights")
    key = client.upload("r1", "checkpoint", "ck.pt", source)
    assert key == "runs/r1/checkpoint/ck.pt"
    got = bucket.get_object(Bucket="lab", Key=key)
    assert got["Body"].read() == b"weights"
    assert got["Metadata"] == {
        "sha256": hashlib.sha256(b"weights").hexdigest()}
    saved = client.upload("r1", "output", "models/best.pt", source,
                          place=tr.PLACE_DATA)
    assert saved == "runs/r1/output/models/best.pt"
    assert client.key_for("../up", "b.pt").startswith("runs/r1/")
    # Long names keep every character (#838): no two checkpoints share a key.
    run_id = "93ead97cebd444ddafe453ea56d34e53"
    names = [f"{run_id}-preset1__train_loop-e{n}.pt" for n in (100, 200)]
    keys = [client.key_for("checkpoint", n) for n in names]
    assert keys == [f"runs/r1/checkpoint/{n}" for n in names]
    assert client.key_for("checkpoint", "dir/模型 v2.pt") == \
        "runs/r1/checkpoint/dir/模型 v2.pt"
    for climbing in ("../x", "a/../b.pt", "/abs"):
        with pytest.raises(tr.TransferError, match="refusing"):
            client.key_for("checkpoint", climbing)
    with pytest.raises(tr.TransferError, match="unknown place"):
        client.key_for("checkpoint", "x", "elsewhere")


def test_a_refusal_fails_at_once_and_an_outage_is_retried(
        bucket, tmp_path, monkeypatch):
    from botocore.exceptions import ClientError, EndpointConnectionError

    monkeypatch.setattr(tr, "RETRY_PAUSE_S", 0.0)
    source = tmp_path / "ck.pt"
    source.write_bytes(b"w")

    def failing(*errors: BaseException) -> Any:
        queue = list(errors)

        class _Client:
            calls = 0

            def upload_file(self, *a: Any, **k: Any) -> None:
                _Client.calls += 1
                if queue:
                    raise queue.pop(0)

        return _Client

    def denied() -> ClientError:
        return ClientError({"Error": {"Code": "AccessDenied"},
                            "ResponseMetadata": {"HTTPStatusCode": 403}},
                           "PutObject")

    def expired() -> ClientError:
        return ClientError({"Error": {"Code": "ExpiredToken"},
                            "ResponseMetadata": {"HTTPStatusCode": 400}},
                           "PutObject")

    def server() -> ClientError:
        return ClientError({"Error": {"Code": "InternalError"},
                            "ResponseMetadata": {"HTTPStatusCode": 500}},
                           "PutObject")

    fake = failing(denied())
    client = _client(tmp_path, factory=lambda **_: fake())
    with pytest.raises(tr.TransferError, match="AccessDenied"):
        client.upload("r1", "checkpoint", "ck.pt", source)
    assert fake.calls == 1
    # Expired temporary credentials wait for the renewal, like an outage;
    # so does an endpoint that does not answer.
    fake = failing(server(), expired(),
                   EndpointConnectionError(endpoint_url="x"), expired())
    client = _client(tmp_path, factory=lambda **_: fake())
    assert client.upload("r1", "checkpoint", "ck.pt", source) == \
        "runs/r1/checkpoint/ck.pt"
    assert fake.calls == 5
    # A local failure (the file went away mid-upload) is not retried.
    fake = failing(FileNotFoundError("gone"))
    client = _client(tmp_path, factory=lambda **_: fake())
    with pytest.raises(tr.TransferError, match="FileNotFoundError"):
        client.upload("r1", "checkpoint", "ck.pt", source)
    assert fake.calls == 1


def test_a_fetch_retries_a_server_error(tmp_path, monkeypatch):
    from botocore.exceptions import ClientError

    monkeypatch.setattr(agent_s3, "RETRY_PAUSE_S", 0.0)
    errors = [ClientError({"Error": {"Code": "InternalError"},
                           "ResponseMetadata": {"HTTPStatusCode": 503}},
                          "GetObject")]

    class _Body:
        def iter_chunks(self, size: int) -> Any:
            return iter([CSV])

        def close(self) -> None:
            pass

    class _Client:
        def get_object(self, **_: Any) -> Any:
            if errors:
                raise errors.pop(0)
            return {"Body": _Body()}

    client = _client(tmp_path, factory=lambda **_: _Client())
    path, fetched = client.fetch(_entry())
    assert fetched and path.read_bytes() == CSV
    stop = _client(tmp_path / "c", factory=lambda **_: _Client())
    with pytest.raises(tr.TransferCancelled):
        stop.fetch(_entry(), cancelled=lambda: True)


def test_renewed_credentials_build_a_new_client(tmp_path):
    built: list[str] = []

    def factory(**kw: Any) -> Any:
        built.append(kw["access_key_id"])
        return object()

    client = _client(tmp_path, factory=factory)
    client._s3()
    client._s3()
    client.update_access(_access(access_key_id="NEW", session_token="t"))
    client._s3()
    assert built == ["AK", "NEW"]
    assert client.access.access_key_id == "NEW"


# ── the Runner ────────────────────────────────────────────────────────────


def _s3_assign(run_id: str) -> wp.Assign:
    return wp.Assign(
        run_id=run_id, graph={"nodes": [], "edges": []},
        storage=wp.Storage(backend="s3", mount="data",
                           relative_artifacts=False, inputs=[_entry()],
                           s3=_access(prefix=f"runs/{run_id}/")))


async def _until(predicate: Any, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition never held"
        await asyncio.sleep(0.02)


async def test_an_s3_run_reads_and_writes_the_bucket(bucket, tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.transfer_child)
    assert runner.speaks_s3()
    await runner.assign(_s3_assign("t1"))
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "t1")
    frames = sink.of("t1")
    assert [f.seq for f in frames] == list(range(1, len(frames) + 1))
    assert frames[0].detail == \
        "fetching inputs 1/1: train.csv (8 B, fetched)"
    artifacts = [f for f in frames if isinstance(f, wp.Artifact)]
    assert [(a.kind, a.path, a.backend) for a in artifacts] == [
        ("checkpoint", "runs/t1/checkpoint/p1.pt", "s3"),
        ("output", "runs/t1/output/models/saved.pt", "s3"),
        ("checkpoint", "runs/t1/checkpoint/i1.pt", "s3")]
    assert finished.status == "cancelled"
    for key, data in (("runs/t1/checkpoint/p1.pt", b"periodic"),
                      ("runs/t1/output/models/saved.pt", b"saved"),
                      ("runs/t1/checkpoint/i1.pt", b"interrupted")):
        assert bucket.get_object(Bucket="lab", Key=key)["Body"].read() \
            == data
    await _until(lambda: "t1" not in runner.runs)
    # The cache is the agent's, shared with the transfer backend's.
    assert (tmp_path / "agent-cache" / SHA).read_bytes() == CSV


async def test_credentials_reach_a_running_s3_run(tmp_path):
    clients: list[Any] = []

    class _Fake:
        backend = "s3"

        def __init__(self, access: Any) -> None:
            self.cache = tr.InputCache(tmp_path / "cache")
            self.access = access
            self.gate = asyncio.Event()
            clients.append(self)

        def update_access(self, access: Any) -> None:
            self.access = access

        def session(self) -> Any:
            return agent_s3._NoSession()

        def fetch(self, entry: Any, cancelled: Any = lambda: False, *,
                  client: Any = None) -> Any:
            import time

            while not cancelled():
                time.sleep(0.01)
            raise tr.TransferCancelled("cancelled while fetching")

    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.quick_child, s3_client=_Fake)
    await runner.assign(_s3_assign("c1"))
    await _until(lambda: clients and runner.runs["c1"].s3 is clients[0])
    assert runner.renew("c1", _access(access_key_id="NEW"))
    assert clients[0].access.access_key_id == "NEW"
    assert not runner.renew("nope", _access())
    await runner.cancel("c1")
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished) and f.run_id == "c1")
    assert finished.status == "cancelled"
    await _until(lambda: "c1" not in runner.runs)
    assert not runner.renew("c1", _access())


async def test_an_agent_without_boto3_refuses_an_s3_run(tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(agent_s3, "client_available", lambda: False)
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink)
    assert not runner.speaks_s3()
    await runner.assign(_s3_assign("n1"))
    finished = sink.finished("n1")
    assert finished.status == "failed"
    assert "boto3 is not installed" in finished.error


def test_hello_lists_s3_only_with_a_client():
    assert "s3" in wp.agent_capabilities(True)
    assert wp.agent_capabilities(False) == ["ack", "mounts", "transfer",
                                           "memory_budget", "one_shot",
                                           "record_outputs", "instances"]


async def test_the_connection_hands_credentials_to_the_runner(tmp_path):
    from app.agent.connection import AgentConnection

    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"})
    seen: list[tuple[str, Any]] = []
    runner.renew = lambda run_id, access: seen.append((run_id, access)) \
        or False  # type: ignore[method-assign]
    connection = AgentConnection(runner=runner, name="w", version="0",
                                 control_url=lambda: "http://cp:1",
                                 token="t", probe_devices=lambda: [])
    frame = wp.Credentials(run_id="r1", s3=_access())
    await connection._dispatch(wp.decode_control_frame(wp.encode_frame(frame)))
    assert seen == [("r1", _access())]


def test_the_protocol_carries_an_s3_assignment_without_showing_secrets():
    frame = _s3_assign("p1")
    assert wp.decode_control_frame(wp.encode_frame(frame)) == frame
    assert "SK" not in repr(frame)



def test_unknown_keys_wait_for_renewal_only_when_temporary_ones_expired(
        tmp_path, monkeypatch):
    """MinIO answers ``InvalidAccessKeyId`` once STS credentials expire
    (it forgets them): retried like an outage then, until the renewal
    arrives; with static keys, or temporary ones still valid, a refusal."""
    import time

    from botocore.exceptions import ClientError

    monkeypatch.setattr(tr, "RETRY_PAUSE_S", 0.0)

    def unknown() -> ClientError:
        return ClientError({"Error": {"Code": "InvalidAccessKeyId"},
                            "ResponseMetadata": {"HTTPStatusCode": 403}},
                           "PutObject")

    source = tmp_path / "ck.pt"
    source.write_bytes(b"w")
    calls: list[str] = []

    def factory(**kw: Any) -> Any:
        class _Client:
            def upload_file(self, *a: Any, **k: Any) -> None:
                calls.append(kw["access_key_id"])
                if kw["access_key_id"] == "OLD":
                    raise unknown()

        return _Client()

    expired = _access(access_key_id="OLD", session_token="t",
                      expires_at=time.time() - 1)
    client = agent_s3.S3Client(expired, tr.InputCache(tmp_path / "c"),
                               factory=factory)
    real_sleep = tr._sleep

    def renew_while_waiting(seconds: float, cancelled: Any) -> None:
        client.update_access(_access(access_key_id="NEW", session_token="t",
                                     expires_at=time.time() + 900))
        real_sleep(0, cancelled)

    monkeypatch.setattr(tr, "_sleep", renew_while_waiting)
    assert client.upload("r1", "checkpoint", "ck.pt", source) == \
        "runs/r1/checkpoint/ck.pt"
    assert calls == ["OLD", "NEW"]
    for access in (_access(access_key_id="OLD"),  # static
                   _access(access_key_id="OLD", session_token="t",
                           expires_at=time.time() + 900)):  # still valid
        calls.clear()
        refused = agent_s3.S3Client(access, tr.InputCache(tmp_path / "c"),
                                    factory=factory)
        with pytest.raises(tr.TransferError, match="InvalidAccessKeyId"):
            refused.upload("r1", "checkpoint", "ck.pt", source)
        assert calls == ["OLD"]
