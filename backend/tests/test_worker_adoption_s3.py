"""Adoption of ``s3`` worker runs after a control-plane restart (#793 x #791).

The credentials an ``s3`` run's worker holds were issued by the previous
process, and the task that renews them lived in its memory. An adopted
run gets new ones (``s3.issue_access`` with the run's recorded input
keys) as soon as its worker is back, then renewal as for any run; a
replayed ``s3`` artifact row the previous process wrote is recorded once.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from httpx_ws import aconnect_ws

from app.core import executors, s3
from app.core import worker_protocol as wp
from app.core.executors import WORKER_S3_KEY, TransferredArtifactSignal
from app.core.run_service import WORKER_RESUME_KEY

from .test_s3_backend import data_root as s3_data_root  # noqa: F401
from .test_s3_backend import db as s3_db  # noqa: F401
from .test_s3_backend import bucket as s3_bucket  # noqa: F401
from .test_s3_backend import s3_on  # noqa: F401
from .test_worker_adoption import db  # noqa: F401
from .test_worker_adoption_review import _worker_row
from .test_ws_worker import FakeAgent, _client, _path, _terminal

PREFIX = "runs/box-run/"
FACTS = {WORKER_RESUME_KEY: {"last_seq": 3, "slots": [["box", "cpu"]]},
         WORKER_S3_KEY: {"bucket": "lab", "prefix": PREFIX,
                         "input_keys": ["inputs/aa"]}}


async def test_an_adopted_s3_run_gets_new_credentials_when_its_worker_is_back(
        db, s3_on, monkeypatch):  # noqa: F811
    issued: list[tuple[str, list[str]]] = []

    def issue(run_id: str, input_keys: Any = ()) -> Any:
        issued.append((run_id, list(input_keys)))
        return wp.S3Access(bucket="lab", prefix=PREFIX,
                           access_key_id=f"fresh{len(issued)}",
                           secret_access_key="s", session_token="t",
                           expires_at=time.time() + 0.3)

    monkeypatch.setattr(s3, "issue_access", issue)
    monkeypatch.setattr(executors, "REFRESH_MIN_S", 0.01)
    first = db()
    await _worker_row(first.service.store, facts=FACTS)
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    async with _client() as client:
        async with aconnect_ws(_path(), client) as ws:
            agent = FakeAgent(ws)
            await agent.hello()
            await agent.send(wp.RunAlive(runs=[wp.RunAliveEntry(
                run_id="box-run", last_seq=3)]))
            frame = await agent.recv()
            while not isinstance(frame, wp.Credentials):
                frame = await agent.recv()
            assert frame.run_id == "box-run"
            assert frame.s3.access_key_id.startswith("fresh")
            # ... and renewed after that, as for a run started here.
            frame = await agent.recv()
            while not isinstance(frame, wp.Credentials) \
                    or frame.s3.access_key_id == "fresh1":
                frame = await agent.recv()
            await agent.send(wp.RunFinished(run_id="box-run", seq=4,
                                            status="succeeded"))
            record = await _terminal(second.service, "box-run")
    assert record.status == "succeeded"
    assert issued[0] == ("box-run", ["inputs/aa"])
    count = len(issued)
    await asyncio.sleep(0.5)
    assert len(issued) == count  # renewal stopped with the run


async def test_an_s3_artifact_the_previous_process_wrote_is_one_row(
        db, s3_on):  # noqa: F811
    first = db()
    store = first.service.store
    await _worker_row(store, facts=FACTS)
    key = PREFIX + "checkpoint/e1.pt"
    await store.add_artifact("box-run", "checkpoint", key, meta={"epoch": 1},
                             backend="s3", worker_id="box", bucket="lab")
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    active = second.service._runs["box-run"]
    assert active.resume_s3 == FACTS[WORKER_S3_KEY]

    def signal(epoch: int) -> Any:
        return TransferredArtifactSignal(
            kind="checkpoint", path=key, meta={"epoch": epoch},
            backend="s3", bucket="lab", prefix=PREFIX)
    await second.service._record_artifact(active, signal(1))  # replay
    await second.service._record_artifact(active, signal(2))  # a rewrite
    rows = await store.list_artifacts("box-run")
    assert [(r.backend, r.meta["epoch"]) for r in rows] == [("s3", 1),
                                                           ("s3", 2)]


async def test_an_s3_assignment_records_what_adoption_needs(
        s3_bucket, s3_data_root, s3_db):  # noqa: F811
    """Before the ``assign`` goes out: the bucket, prefix and input keys."""
    from app.core import transfer
    from app.core.executors import WorkerExecutor
    from app.core.workers import WorkerRegistry

    from .test_s3_backend import _caps
    from .test_transfer_backend import _needs, _request
    from .test_worker_executor import _Agent, _Recorder

    registry = WorkerRegistry()
    try:
        agent = _Agent(registry)
        _caps(agent.worker, *wp.CAPABILITIES)
        recorder = _Recorder()
        needs = transfer.annotate(_needs(("data_file", "train.csv")))
        executor = WorkerExecutor(
            registry, content_index=transfer.ContentIndex(s3_db))
        task = asyncio.create_task(executor.run(_request(recorder, needs)))
        assign = await agent.wait_for(wp.Assign)
        assert assign.storage.backend == "s3"
        keys = [i.key for i in assign.storage.inputs if i.key]
        assert keys
        facts = [c[1] for c in recorder.calls if c[0] == "facts"]
        assert {WORKER_S3_KEY: {"bucket": "lab", "prefix": "runs/r1/",
                                "input_keys": keys}} in facts
        registry.route(agent.worker, wp.RunFinished(run_id="r1", seq=1,
                                                    status="succeeded"))
        await task
    finally:
        await registry.shutdown()
