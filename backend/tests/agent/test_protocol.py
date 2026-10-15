"""``app.core.worker_protocol``: the frames both ends share (#771/#772)."""

from __future__ import annotations

import json
import math

import pytest

from app.core import worker_protocol as wp


def _round_trip_agent(frame):
    return wp.decode_agent_frame(wp.encode_frame(frame))


def test_every_frame_type_round_trips():
    agent_frames = [
        wp.Hello(name="box", version="3.1.0", devices=[
            wp.DeviceReport(value="cuda:0", kind="cuda", index=0)],
            mounts={"data": "/mnt/data"}, platform={"system": "Linux"}),
        wp.Heartbeat(devices=[], running=["r1"]),
        wp.RunStarted(run_id="r1", seq=1, pid=42, execution_facts={
            "visible_devices": "2,0",
            "device_aliases": {"cuda:2": "cuda:0", "cuda:0": "cuda:1"}}),
        wp.Event(run_id="r1", seq=2, event_type="node_status",
                 payload={"node_id": "a", "status": "running"}),
        wp.Metrics(run_id="r1", seq=3, points=[
            wp.MetricPointReport(name="loss", value=0.1, step=1)]),
        wp.Artifact(run_id="r1", seq=4, kind="checkpoint", path="/c.pt"),
        wp.RunWarning(run_id="r1", seq=5, kind="k", detail="d"),
        wp.RunFinished(run_id="r1", seq=6, status="succeeded"),
        wp.RunAlive(runs=[wp.RunAliveEntry(run_id="r1", last_seq=6)]),
    ]
    assert {f.type for f in agent_frames} == set(wp.AGENT_FRAME_TYPES)
    for frame in agent_frames:
        assert _round_trip_agent(frame) == frame
    control_frames = [
        wp.Welcome(worker_id="w1"),
        wp.Assign(run_id="r1", graph={"nodes": []}, devices=["cuda:1"]),
        wp.Cancel(run_id="r1", grace_s=3),
        wp.Drain(),
        wp.Reject(reason="no"),
        wp.Ack(run_id="r1", seq=4),
    ]
    assert {f.type for f in control_frames} == set(wp.CONTROL_FRAME_TYPES)
    for frame in control_frames:
        assert wp.decode_control_frame(wp.encode_frame(frame)) == frame


def test_frames_are_flat_json_objects():
    raw = wp.encode_frame(wp.Cancel(run_id="r1"))
    assert json.loads(raw) == {"type": "cancel", "run_id": "r1",
                               "grace_s": None}


def test_nan_crosses_the_wire_unchanged():
    """The control plane sanitises payloads the way it does locally."""
    frame = _round_trip_agent(wp.Event(run_id="r1", seq=1, event_type="x",
                                       payload={"loss": float("nan")}))
    assert math.isnan(frame.payload["loss"])


@pytest.mark.parametrize("raw", [
    "not json", "[1, 2]", '{"type": "nope"}',
    '{"type": "event", "run_id": "r1", "seq": 0, "event_type": "x"}',
    '{"type": "event", "run_id": "../etc", "seq": 1, "event_type": "x"}',
    '{"type": "metrics", "run_id": "r1", "seq": 1, "points": []}',
])
def test_invalid_agent_frames_are_refused(raw):
    with pytest.raises(wp.ProtocolError):
        wp.decode_agent_frame(raw)


def test_directions_do_not_mix():
    with pytest.raises(wp.ProtocolError):
        wp.decode_control_frame(wp.encode_frame(wp.Heartbeat()))
    with pytest.raises(wp.ProtocolError):
        wp.decode_agent_frame(wp.encode_frame(wp.Drain()))
