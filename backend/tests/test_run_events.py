"""Tests for app.core.run_events: the pure event-payload builders (#766).

The builders were moved out of ``RunService``'s three engine bridges without
changing a byte of what they produce. Two kinds of test pin that:

1. Unit tests per builder, fed the payloads the bridge fixtures of
   ``test_run_service.py`` produce (``_RunImage``'s declared image port,
   ``_RunMetrics``' epoch frames, ``_RunLogMetrics``' explicit metrics and
   checkpoint artifact) and the error frame shape ``test_inner_node_status``
   drives. Key ORDER is asserted where it is part of the stored JSON.
2. A round trip: the raw engine callbacks of a graph, rebuilt with these
   functions, equal the ``node_status`` payloads ``RunService`` stores for
   the same graph, compared as serialized JSON.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.core import run_events, run_service
from app.core.execution_context import (
    ArtifactSignal,
    DroppedSignal,
    ExecutionContext,
    MetricSignal,
    WarningSignal,
)
from app.core.graph_engine import execute_graph
from app.core.node_registry import registry
from app.core.output_entries import (
    OUTPUT_KIND_TEXT,
    build_node_output_entries,
    declared_media_ports,
)
from app.core.run_events import (
    artifact_meta,
    artifact_payload,
    dropped_signals_payload,
    error_fields,
    inner_node_status_payload,
    metric_point,
    metric_points_from_progress,
    node_status_payload,
    warning_payload,
)
from app.core.run_service import RunService
from app.core.run_store import ArtifactRecord, MetricPoint, RunStore
from tests.test_run_service import (
    _TEST_NODES,
    _await_terminal,
    _graph,
    _RunImageNode,
)

#: ``_RunMetrics``' epoch frame (epoch 2 of 3), verbatim.
EPOCH_FRAME = {
    "event": "epoch", "epoch": 2, "total_epochs": 3, "loss": 0.5,
    "accuracy": 0.52, "label": "not-a-metric", "converged": False,
}

#: The engine's error frame for ``_InnerStatusBoom`` (a ``KeyError``).
ERROR_FRAME = {"error": "'inner node failed on purpose'",
               "error_type": "KeyError"}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


def _image_media_ports() -> dict[str, dict[str, list[str]]]:
    return declared_media_ports(
        [{"id": "mid", "type": "_RunImage", "data": {"params": {}}}])


def _image_result() -> dict[str, Any]:
    return _RunImageNode().execute({"value": "hi"}, {"size": 8})


# ── re-exports ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "EVENT_RUN_STARTED", "EVENT_NODE_STATUS", "EVENT_RUN_COMPLETED",
    "EVENT_RUN_FAILED", "EVENT_RUN_STOPPED", "EVENT_METRIC", "EVENT_ARTIFACT",
    "EVENT_WARNING", "EVENT_INNER_NODE_STATUS", "NON_METRIC_KEYS",
    "NON_METRIC_EVENTS", "scalar_metrics", "metric_step",
])
def test_run_service_still_publishes_the_moved_names(name):
    assert getattr(run_service, name) is getattr(run_events, name)


# ── error_fields ──────────────────────────────────────────────────────────


def test_error_fields_only_on_an_error_frame():
    assert error_fields("error", ERROR_FRAME) == ERROR_FRAME
    assert error_fields("completed", ERROR_FRAME) == {}
    assert error_fields("error", None) == {}


def test_error_fields_omit_a_missing_or_empty_type():
    assert error_fields("error", {"error": "boom"}) == {"error": "boom"}
    assert error_fields("error", {"error": "boom", "error_type": ""}) == {
        "error": "boom"}
    assert error_fields("error", {}) == {}
    assert error_fields("error", {"error_type": "KeyError"}) == {
        "error": "", "error_type": "KeyError"}


# ── node_status_payload ───────────────────────────────────────────────────


def test_node_status_carries_the_declared_image_and_port_summary():
    media_ports = _image_media_ports()
    result = _image_result()
    payload = node_status_payload("mid", "completed", result, media_ports)
    assert list(payload) == ["node_id", "status", "outputs"]
    assert payload["outputs"] == build_node_output_entries(
        "completed", result, media_ports["mid"])
    assert payload["outputs"][0]["output_kind"] == "image"


def test_node_status_has_no_outputs_key_when_there_are_none():
    assert node_status_payload("src", "running", None, {}) == {
        "node_id": "src", "status": "running"}


def test_node_status_error_frame_key_order():
    payload = node_status_payload("blk", "error", ERROR_FRAME, {})
    assert list(payload) == ["node_id", "status", "error", "error_type"]
    assert payload["error_type"] == "KeyError"


def test_node_status_progress_frame_names_the_inner_node_first():
    payload = node_status_payload("blk", "progress", EPOCH_FRAME, {},
                                  inner_node_id="blk/train")
    assert list(payload) == ["node_id", "status", "inner_node_id", "outputs"]
    assert payload["inner_node_id"] == "blk/train"
    assert payload["outputs"] == [{"output_kind": "progress",
                                   "progress": EPOCH_FRAME}]


def test_node_status_reads_media_ports_at_build_time():
    """``on_expanded`` adds container nodes to the dict AFTER the bridge is
    built; the builder must look the node up per call, not capture it."""
    media_ports: dict[str, dict[str, list[str]]] = {}
    result = _image_result()
    before = node_status_payload("mid", "completed", result, media_ports)
    media_ports.update(_image_media_ports())
    after = node_status_payload("mid", "completed", result, media_ports)
    assert [e["output_kind"] for e in before["outputs"]] == ["tensor_summary"]
    assert [e["output_kind"] for e in after["outputs"]] == [
        "image", "tensor_summary"]


# ── inner_node_status_payload ─────────────────────────────────────────────


def test_inner_node_status_shape_and_media():
    media_ports = {"blk/mid": _image_media_ports()["mid"]}
    payload = inner_node_status_payload(
        "blk/mid", "blk", "completed", _image_result(), media_ports)
    assert list(payload) == ["node_id", "container_id", "status", "outputs"]
    assert payload["container_id"] == "blk"
    assert payload["outputs"][0]["output_kind"] == "image"


def test_inner_node_status_drops_text_the_card_already_relays():
    payload = inner_node_status_payload(
        "blk/p", "blk", "running", {"__log__": "hello"}, {})
    assert "outputs" not in payload
    full = build_node_output_entries("running", {"__log__": "hello"}, None)
    assert [e["output_kind"] for e in full] == [OUTPUT_KIND_TEXT]


def test_inner_node_status_error_frame():
    payload = inner_node_status_payload(
        "blk/boom", "blk", "error", ERROR_FRAME, {})
    assert payload == {"node_id": "blk/boom", "container_id": "blk",
                       "status": "error", **ERROR_FRAME}
    assert list(payload) == ["node_id", "container_id", "status", "error",
                             "error_type"]


# ── metrics ───────────────────────────────────────────────────────────────


def test_metric_points_from_an_epoch_frame():
    points, step = metric_points_from_progress(
        "mid", EPOCH_FRAME, explicit_metric_nodes=set(), last_step=0)
    assert step == 2
    assert points == [
        MetricPoint(name="loss", value=0.5, step=2, node_id="mid"),
        MetricPoint(name="accuracy", value=0.52, step=2, node_id="mid"),
    ]


def test_metric_points_stand_down_for_a_node_that_logs_its_own():
    points, step = metric_points_from_progress(
        "mid", EPOCH_FRAME, explicit_metric_nodes={"mid"}, last_step=7)
    assert (points, step) == ([], 7)


def test_metric_points_fall_back_to_the_running_counter():
    points, step = metric_points_from_progress(
        "mid", {"loss": 0.25}, explicit_metric_nodes=set(), last_step=4)
    assert step == 5
    assert points == [MetricPoint(name="loss", value=0.25, step=5,
                                  node_id="mid")]


def test_a_batch_frame_advances_the_step_but_yields_no_points():
    frame = {"event": "batch", "epoch": 1, "batch": 3, "total_batches": 4,
             "loss": 0.3}
    points, step = metric_points_from_progress(
        "mid", frame, explicit_metric_nodes=set(), last_step=0)
    assert (points, step) == ([], 1)


def test_metric_point_from_an_explicit_signal():
    signal = MetricSignal(name="val_loss", value=2.0, step=1, node_id="mid")
    assert metric_point(signal) == MetricPoint(
        name="val_loss", value=2.0, step=1, node_id="mid")


# ── signals ───────────────────────────────────────────────────────────────


def test_artifact_meta_carries_the_producing_node():
    signal = ArtifactSignal(kind="checkpoint", path="interrupted/fake.pt",
                            meta={"reason": "interrupted", "epoch": 1},
                            node_id="mid")
    assert artifact_meta(signal) == {"reason": "interrupted", "epoch": 1,
                                     "node_id": "mid"}
    assert signal.meta == {"reason": "interrupted", "epoch": 1}, (
        "the signal's own meta must not be mutated")


def test_artifact_meta_without_a_node_is_the_signal_meta():
    meta = {"reason": "x"}
    assert artifact_meta(ArtifactSignal(kind="k", path="p", meta=meta)) is meta
    assert artifact_meta(ArtifactSignal(kind="k", path="p")) is None
    assert artifact_meta(
        ArtifactSignal(kind="k", path="p", node_id="n")) == {"node_id": "n"}


def test_artifact_payload_describes_the_stored_row():
    record = ArtifactRecord(id=7, run_id="r", kind="checkpoint",
                            path="interrupted/fake.pt",
                            meta={"epoch": 1, "node_id": "mid"},
                            created_at="2026-10-15T00:00:00Z")
    payload = artifact_payload(record, "mid")
    assert payload == {"artifact_id": 7, "kind": "checkpoint",
                       "path": "interrupted/fake.pt",
                       "meta": {"epoch": 1, "node_id": "mid"},
                       "node_id": "mid"}
    assert list(payload) == ["artifact_id", "kind", "path", "meta", "node_id"]


def test_dropped_signals_payload():
    payload = dropped_signals_payload(DroppedSignal(count=12))
    assert list(payload) == ["kind", "count", "detail"]
    assert payload["kind"] == "dropped_signals"
    assert payload["count"] == 12
    assert payload["detail"] == (
        "progress/metric signals were dropped to keep the run's outbox "
        "bounded; the series may have gaps")


def test_warning_payload():
    signal = WarningSignal(kind="node_state_evicted", detail="weights gone",
                           node_id="mid")
    assert warning_payload(signal) == {"kind": "node_state_evicted",
                                       "detail": "weights gone",
                                       "node_id": "mid"}
    assert list(warning_payload(signal)) == ["kind", "detail", "node_id"]


# ── round trip against what RunService stores ─────────────────────────────


@pytest.fixture
def store(tmp_path):
    from app.core.db import Database

    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield RunStore(database)
    finally:
        database.close()


@pytest.mark.parametrize("middle,params", [
    ("_RunImage", {"size": 8}),
    ("_RunMetrics", {"epochs": 2}),
    ("_RunBoom", {}),
])
async def test_rebuilt_callbacks_equal_the_stored_node_status_events(
        store, middle, params):
    graph = _graph(middle, params=params)

    calls: list[tuple[Any, ...]] = []

    async def record(node_id, status, result, inner_node_id=None):
        calls.append((node_id, status, result, inner_node_id))

    try:
        await execute_graph(graph["nodes"], graph["edges"],
                            on_progress=record,
                            context=ExecutionContext())
    except Exception:
        pass  # _RunBoom fails the graph; its frames are what we compare
    media_ports = declared_media_ports(graph["nodes"])
    rebuilt = [node_status_payload(n, s, r, media_ports, inner_node_id=i)
               for n, s, r, i in calls]

    service = RunService(store, shutdown_grace_s=2.0)
    try:
        submitted = await service.submit(graph)
        await _await_terminal(store, submitted.run_id)
    finally:
        await service.shutdown()
    stored = [e.payload for e in await store.get_events(submitted.run_id)
              if e.type == run_events.EVENT_NODE_STATUS]

    assert rebuilt, "the engine reported nothing"
    assert [json.dumps(p, sort_keys=False) for p in stored] == [
        json.dumps(json.loads(json.dumps(p)), sort_keys=False)
        for p in rebuilt]
