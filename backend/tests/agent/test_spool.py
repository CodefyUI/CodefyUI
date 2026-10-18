"""The agent's per-run spool and its replay after a reconnect (#783)."""

from __future__ import annotations

import os
from typing import Any

from app.agent.runner import FINISHED_MARKER, prune_run_dirs
from app.agent.spool import SPOOL_FILE, Spool, SpoolStore
from app.core import worker_protocol as wp

from . import _helpers as h
from .test_connection import Agent, FakeControlPlane


def _line(seq: int) -> str:
    return wp.encode_frame(wp.Event(run_id="r1", seq=seq, event_type="e",
                                    payload={"n": seq}))


# ── Spool ────────────────────────────────────────────────────────────────


def _seqs(spool: Spool, after: int = 0, limit: int = 1000) -> list[int]:
    return [seq for seq, _ in spool.read_after(after, limit)]


def test_acks_move_an_offset_and_the_rest_replays_in_order(tmp_path):
    spool = Spool(tmp_path / "r1", "r1")
    for seq in range(1, 11):
        spool.append(seq, _line(seq))
    assert spool.pending == 10 and spool.last_seq == 10
    assert _seqs(spool) == list(range(1, 11))
    # Bounded batches, after any seq.
    assert _seqs(spool, after=0, limit=3) == [1, 2, 3]
    assert _seqs(spool, after=3, limit=3) == [4, 5, 6]
    size = spool.path.stat().st_size

    # Under half the file: the ack moves an offset and writes nothing.
    assert spool.ack(4) is False
    assert spool.pending == 6 and spool.path.stat().st_size == size
    assert _seqs(spool) == [5, 6, 7, 8, 9, 10]
    assert [wp.decode_agent_frame(line).payload["n"]
            for _, line in spool.read_after(0, 2)] == [5, 6]
    spool.ack(2)  # an older ack changes nothing
    assert _seqs(spool) == [5, 6, 7, 8, 9, 10]

    # Past half the file: the acknowledged prefix is cut off.
    spool.ack(7)
    assert spool.path.read_text().splitlines() == [_line(8), _line(9),
                                                   _line(10)]
    assert _seqs(spool, after=8) == [9, 10]
    spool.append(11, _line(11))
    assert _seqs(spool) == [8, 9, 10, 11]
    spool.ack(11)
    assert spool.pending == 0 and spool.path.read_text() == ""
    spool.append(12, _line(12))
    assert spool.read_after(0, 10) == [(12, _line(12))]
    spool.close(delete=True)
    assert not spool.path.exists()


def test_a_large_acknowledged_prefix_is_compacted(tmp_path, monkeypatch):
    from app.agent import spool as spool_module

    monkeypatch.setattr(spool_module, "COMPACT_BYTES", 2 * len(_line(1)))
    spool = Spool(tmp_path / "r1", "r1")
    for seq in range(1, 11):
        spool.append(seq, _line(seq))
    spool.ack(1)
    assert spool.path.read_text().splitlines()[0] == _line(1)  # kept
    spool.ack(2)  # two lines acknowledged: the size threshold
    assert spool.path.read_text().splitlines()[0] == _line(3)
    assert _seqs(spool) == list(range(3, 11))


def test_the_spool_is_complete_once_run_finished_is_acked(tmp_path):
    store = SpoolStore(tmp_path)
    spool = store.spool("r1")
    spool.append(1, _line(1))
    finished = wp.encode_frame(wp.RunFinished(run_id="r1", seq=2,
                                              status="succeeded"))
    spool.append(2, finished, final=True)
    assert store.ack("r1", 1) is False and spool.path.exists()
    assert store.ack("r1", 2) is True
    assert not (tmp_path / "r1" / SPOOL_FILE).exists()
    assert store.spools() == []
    assert store.ack("r1", 2) is False  # nothing left to acknowledge


def test_spools_a_previous_agent_left_are_discarded(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / SPOOL_FILE).write_text(_line(1) + "\n")
    (tmp_path / "plain").mkdir()
    store = SpoolStore(tmp_path)
    assert not (tmp_path / "old" / SPOOL_FILE).exists()
    assert store.spools() == []


def test_retention_keeps_a_directory_that_still_spools(tmp_path):
    for index, name in enumerate(("a", "b", "c")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        marker = run_dir / FINISHED_MARKER
        marker.touch()
        os.utime(marker, (1000 + index, 1000 + index))
    (tmp_path / "a" / SPOOL_FILE).write_text("")
    prune_run_dirs(tmp_path, keep_last=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "c"]


# ── replay through a real connection ─────────────────────────────────────


def _spool(agent: Agent, run_id: str) -> Spool | None:
    return next((s for s in agent.connection.spools.spools()
                 if s.run_id == run_id), None)


def _run_frames(cp: FakeControlPlane, run_id: str,
                connection: int | None = None) -> list[Any]:
    return [f for n, f in cp.frames if getattr(f, "run_id", None) == run_id
            and getattr(f, "seq", None) is not None
            and (connection is None or n == connection)]


async def test_frames_made_while_disconnected_are_replayed(tmp_path):
    """The network drops mid-run: the child keeps going, its frames are
    spooled, and the next connection gets ``run_alive`` and then every
    unacknowledged frame, in order, ending in ``run_finished``."""
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url, child_entry=h.ticking_child) as agent:
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="tick1", graph={"nodes": []}))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event)
                          and f.seq >= 4)
        await cp.send(wp.Ack(run_id="tick1", seq=2))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event)
                          and f.seq >= 6)
        spool = _spool(agent, "tick1")
        assert spool.acked == 2
        await cp.sockets[0].close()  # the network drops

        alive = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunAlive), timeout=30)
        assert [e.run_id for e in alive.runs] == ["tick1"]
        finished = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunFinished),
            timeout=60)
        assert finished.status == "succeeded"
        replayed = [f.seq for f in _run_frames(cp, "tick1", 1)]
        # Everything after the ack, in order, without a gap.
        assert replayed == list(range(3, finished.seq + 1))
        # Nothing was lost across the two connections.
        seen = {f.seq for f in _run_frames(cp, "tick1")}
        assert seen == set(range(1, finished.seq + 1))
        # run_alive came before the replay on the new connection.
        order = [type(f).__name__ for n, f in cp.frames if n == 1]
        assert order.index("RunAlive") < order.index("Event")

        await cp.send(wp.Ack(run_id="tick1", seq=finished.seq))
        await cp.wait_for(lambda n, f: _spool(agent, "tick1") is None, timeout=10)
        assert not (tmp_path / "agent-runs" / "tick1" / SPOOL_FILE).exists()


async def test_a_finished_run_still_spooled_is_alive_after_a_reconnect(
        tmp_path):
    """A run that ended while the network was down is still reported in
    ``run_alive``, and its ``run_finished`` is replayed."""
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url, child_entry=h.quick_child) as agent:
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="q1", graph={"nodes": []}))
        first = await cp.wait_for(lambda n, f: isinstance(f, wp.RunFinished))
        await cp.wait_for(lambda n, f: "q1" not in agent.runner.runs,
                          timeout=10)
        await cp.sockets[0].close()
        alive = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunAlive), timeout=30)
        assert [(e.run_id, e.last_seq) for e in alive.runs] == [
            ("q1", first.seq)]
        again = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunFinished))
        assert again.seq == first.seq


async def test_a_refused_run_is_spooled_until_acked(tmp_path):
    """A refusal has no run directory of its own until it spools; the ack
    of its ``run_finished`` marks that directory for retention."""
    async with FakeControlPlane() as cp, Agent(tmp_path, cp.url) as agent:
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="no1", graph={"nodes": []},
                                device="cuda:7"))
        refused = await cp.wait_for(lambda n, f: isinstance(f, wp.RunFinished))
        assert refused.status == "failed"
        run_dir = tmp_path / "agent-runs" / "no1"
        assert (run_dir / SPOOL_FILE).exists()
        await cp.send(wp.Ack(run_id="no1", seq=refused.seq))
        await cp.wait_for(lambda n, f: (run_dir / FINISHED_MARKER).exists(),
                          timeout=10)
        assert _spool(agent, "no1") is None


async def test_the_last_ack_leaves_a_retired_runs_marker_alone(tmp_path):
    """Review of #783: only a refused assignment's directory is marked on
    its final ack; a run that started was marked when it retired, and
    retention orders by that time."""
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url, child_entry=h.quick_child) as agent:
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="q2", graph={"nodes": []}))
        finished = await cp.wait_for(
            lambda n, f: isinstance(f, wp.RunFinished))
        marker = tmp_path / "agent-runs" / "q2" / FINISHED_MARKER
        await cp.wait_for(lambda n, f: marker.exists(), timeout=10)
        os.utime(marker, (1000, 1000))
        await cp.send(wp.Ack(run_id="q2", seq=finished.seq))
        await cp.wait_for(lambda n, f: _spool(agent, "q2") is None,
                          timeout=10)
        assert marker.stat().st_mtime == 1000


async def test_heartbeats_go_out_during_a_long_replay(tmp_path, monkeypatch):
    """Review of #783: a replay must not starve the heartbeat."""
    import time

    from app.agent import connection as connection_module
    from app.agent.spool import Spool as SpoolClass

    monkeypatch.setattr(connection_module, "REPLAY_BATCH", 1)
    original = SpoolClass.read_after

    def slow_read_after(self, seq, limit):
        time.sleep(0.01)
        return original(self, seq, limit)

    monkeypatch.setattr(SpoolClass, "read_after", slow_read_after)
    async with FakeControlPlane(heartbeat_s=0.1) as cp:
        agent = Agent(tmp_path, cp.url)
        spool = agent.connection.spools.spool("old1")
        for seq in range(1, 101):
            spool.append(seq, wp.encode_frame(wp.Event(
                run_id="old1", seq=seq, event_type="e", payload={})))
        async with agent:
            last = await cp.wait_for(
                lambda n, f: isinstance(f, wp.Event) and f.seq == 100,
                timeout=30)
            assert last.run_id == "old1"
            kinds = [type(f).__name__ for n, f in cp.frames if n == 0]
            first_event = kinds.index("Event")
            last_event = len(kinds) - 1 - kinds[::-1].index("Event")
            assert kinds[0] == "Hello" and kinds[1] == "RunAlive"
            assert "Heartbeat" in kinds[first_event:last_event]
            seqs = [f.seq for n, f in cp.frames
                    if n == 0 and isinstance(f, wp.Event)]
            assert seqs == list(range(1, 101))
