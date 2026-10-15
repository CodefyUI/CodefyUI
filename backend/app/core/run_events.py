"""Run event payloads, built from engine callbacks: pure functions (#766).

``RunService`` used to build every event payload inside the three engine
bridges it handed ``execute_graph`` (``_progress_bridge``,
``_inner_status_bridge``, ``_signal_bridge``), in the same breath as it
persisted them. A run that executes somewhere without a ``RunService`` (the
worker subprocess of #771) must produce the SAME payloads, byte for byte, or
a run's log would depend on where it ran. So the two halves are split:

- **this module** turns engine callback arguments into payloads. No I/O, no
  database, no event loop, no ``RunService``: every function here is a plain
  function of its arguments, which is what lets a worker import it.
  One caveat for that worker: the output entries come from
  ``core.output_entries``, which reads ``settings`` and the node registry
  (media detection, path rules). A worker builds the same bytes only if it
  runs with the same settings and the same registered node types as the
  control plane.
- **persistence** (``cap_event_payload``, the event append, the metric
  batcher, the artifact row) stays in ``run_service``, reached through the
  ``RunSinks`` an executor is given (``core.executors``).

The event TYPES live here too, because a payload without its type is half
an event. ``run_service`` re-exports every name it published before #766, so
existing imports keep working.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .execution_context import (
    ArtifactSignal,
    DroppedSignal,
    MetricSignal,
    WarningSignal,
)
from .output_entries import OUTPUT_KIND_TEXT, build_node_output_entries
from .run_store import ArtifactRecord, MetricPoint

#: ``graph_engine.execute_graph``'s error policy when a run does not name
#: one. Defined here (#766) because two modules need it: ``run_service``
#: validates and stores options against it, and the executor passes it to
#: the engine. ``run_service`` imports ``executors``, so neither of those can
#: own it without a cycle.
DEFAULT_ERROR_MODE = "fail_fast"

#: ``declared_media_ports`` of a run's nodes: node id -> media kind -> port
#: names. Read when a frame is built, because the run adds the nodes inside
#: its containers to it (``on_expanded``) before anything runs.
MediaPorts = Mapping[str, dict[str, list[str]]]

# ── event vocabulary ──────────────────────────────────────────────────────
#
# Deliberately the WebSocket message types, verbatim. The durable event log
# exists to be replayed onto that wire (#121: attach to a running run and
# catch up), so a 1:1 mapping means the bridge is
# ``{"type": event.type, **(event.payload or {})}`` and there is no second
# vocabulary to keep in sync. The ``type`` key itself is NOT stored in the
# payload — the column already carries it.
EVENT_RUN_STARTED = "execution_start"
EVENT_NODE_STATUS = "node_status"
EVENT_RUN_COMPLETED = "execution_complete"
EVENT_RUN_FAILED = "execution_error"
EVENT_RUN_STOPPED = "execution_stopped"

#: #122's additions. New TYPES rather than new node_status payloads, because
#: neither is about a node changing state: a client that only knows the five
#: types above ignores them (the WS bridge is
#: ``{"type": event.type, **payload}`` for every type), and a client that
#: wants live charts subscribes to ``metric`` instead of re-deriving series
#: from progress frames.
#:
#: ``metric`` carries a whole FLUSH, not one point: the batcher already
#: groups points by size or by 0.5 s, and reusing that cadence is what keeps
#: a per-batch series from writing one event row per batch. Payload is
#: ``{"points": [{"name", "value", "step", "node_id"}, ...]}``, emitted only
#: after the points are durable.
EVENT_METRIC = "metric"
#: One row appended to ``exec_run_artifacts``: ``{"kind", "path", "meta",
#: "node_id"}``. The interrupt checkpoint is the first producer.
EVENT_ARTIFACT = "artifact"
#: Something was lost or degraded but the run continues:
#: ``{"kind": <what>, ...}``. Today the only kind is ``dropped_signals``.
EVENT_WARNING = "run_warning"
#: A node the canvas draws INSIDE a block or preset card (#559), which
#: ``node_status`` rolls up into the outermost card: ``{"node_id",
#: "container_id", "status"}`` plus ``error``/``error_type`` on an error
#: frame, and ``outputs`` when the node produced something: the entries a
#: ``node_status`` would carry for it (its media, by the ports the node
#: declares, and its port summary) minus its text, which the card's own
#: frames relay already. ``node_id`` is the id the run gives the node --
#: ``blk/nest/mul``, the entered instances in front of the canvas id -- so
#: two copies of one block never share a status or a value; ``container_id``
#: is the outermost card. The card's own ``node_status`` frames keep their
#: shape; its terminal one now carries the values on its drawn output ports
#: as its port summary. A container nested inside another reports here too,
#: rolled up the same way. A new TYPE so a client
#: that only knows ``node_status`` keeps reading one status per card. A
#: progress frame from inside a card stays a ``node_status`` for the card and
#: names the inner node in ``inner_node_id``: one frame, both cards.
EVENT_INNER_NODE_STATUS = "inner_node_status"

#: Progress-payload keys that describe the LOOP, not a measurement. Turning
#: ``epoch`` into a series named "epoch" whose value equals its own step is
#: noise; ``total_epochs`` is a constant repeated once per point. #122 added
#: the per-batch pair for exactly the same reason.
NON_METRIC_KEYS = frozenset({"event", "step", "epoch", "total_epochs",
                             "start_epoch", "batch", "total_batches"})

#: Progress ``event`` values whose payload must NOT be mined for metrics.
#: A per-batch frame (#122) is throttled by WALL CLOCK, so several of them
#: legitimately share one step and their count depends on how fast the
#: machine is -- turning that into rows would make the series a measure of
#: the hardware. The node logs the authoritative per-batch series itself,
#: through ``context.log_metric`` under its ``batch_metrics`` option.
#:
#: Mirrors ``core.loop_control.EVENT_BATCH``; not imported from there
#: because this module must not depend on the node-side toolkit, and the
#: string is the wire contract either way.
NON_METRIC_EVENTS = frozenset({"batch"})


# ── payload helpers ───────────────────────────────────────────────────────


def error_fields(
    status: str, result: dict[str, Any] | None,
) -> dict[str, str]:
    """``error`` and, when the engine named the class, ``error_type``.

    Only on an error frame. ``error_type`` is copied out rather than left in
    ``result`` because ``str(exc)`` never contains the class name --
    ``str(KeyError('x'))`` is ``"'x'"`` -- so a client that wants to say
    something kinder than the raw message for one exception KIND has nothing
    else to key on. Absent when the engine did not supply one, so a client
    can tell "no type" from a type it does not recognise.
    """
    if not result or status != "error":
        return {}
    fields = {"error": result.get("error", "")}
    error_type = result.get("error_type")
    if isinstance(error_type, str) and error_type:
        fields["error_type"] = error_type
    return fields



def scalar_metrics(
    payload: Any, *, node_id: str | None, step: int,
) -> list[MetricPoint]:
    """Extract chartable scalars from one progress payload.

    Every finite top-level number that is not a loop descriptor becomes a
    point of a series named after its key. Booleans are excluded (a flag is
    not a measurement), as are strings, lists and nested objects — a
    ``losses`` array is per-batch detail that belongs in the event payload,
    not as one row per element in the metrics table.

    This is INFERENCE, and it exists for nodes that report progress but
    know nothing about metrics. A node that calls ``context.log_metric``
    says what it means; see :func:`metric_points_from_progress`, which
    stops inferring from a node the moment that node speaks for itself.
    """
    if not isinstance(payload, dict):
        return []
    if payload.get("event") in NON_METRIC_EVENTS:
        return []
    points: list[MetricPoint] = []
    for key, value in payload.items():
        if key in NON_METRIC_KEYS or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        points.append(MetricPoint(name=key, value=float(value), step=step,
                                  node_id=node_id))
    return points


def metric_step(payload: Any, *, fallback: int) -> int:
    """The step a progress payload belongs to.

    ``step`` wins over ``epoch``; both beat the caller's running counter, so
    a producer that reports nothing still gets a monotonic x-axis instead of
    every point piling up on step 0.
    """
    if isinstance(payload, dict):
        for key in ("step", "epoch"):
            value = payload.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
    return fallback



# ── node status ───────────────────────────────────────────────────────────


def node_status_payload(
    node_id: str, status: str, result: dict[str, Any] | None,
    media_ports: MediaPorts, *, inner_node_id: str | None = None,
) -> dict[str, Any]:
    """The ``node_status`` payload for one ``on_progress`` callback.

    The message body is byte-for-byte what ``ws_execution`` puts on the
    wire minus its ``type`` key; see the event-vocabulary block above.
    A ``node_status`` payload is ``{"node_id", "status"}`` plus, on an
    error frame only, ``error`` and (when the engine named the class)
    ``error_type`` (see :func:`error_fields`); plus ``outputs`` whenever
    the node produced any; plus ``inner_node_id`` on a progress frame a
    node inside the card sent (#559).

    ``media_ports`` is :func:`declared_media_ports` of the graph, read
    when a frame is built: the run adds the nodes inside its containers
    to it before anything runs.
    """
    message: dict[str, Any] = {"node_id": node_id, "status": status}
    if inner_node_id is not None:
        message["inner_node_id"] = inner_node_id
    message.update(error_fields(status, result))
    entries = build_node_output_entries(
        status, result, media_ports.get(node_id))
    if entries:
        message["outputs"] = entries
    return message


def inner_node_status_payload(
    node_id: str, container_id: str, status: str,
    result: dict[str, Any] | None, media_ports: MediaPorts,
) -> dict[str, Any]:
    """The ``inner_node_status`` payload for one non-progress status (#559).

    See :data:`EVENT_INNER_NODE_STATUS`: the status, on an error what
    failed, and what the node produced (a picture or chart it drew and
    its port summary) under its own id, so an image made inside a block
    reaches the run's log like one made at the top level. Its text is
    left out: the card relays every line already (#601). A PROGRESS frame
    from inside a card is not built here; it is the card's
    :func:`node_status_payload`, naming the inner node.
    """
    message: dict[str, Any] = {
        "node_id": node_id, "container_id": container_id,
        "status": status,
    }
    message.update(error_fields(status, result))
    entries = [
        entry for entry in build_node_output_entries(
            status, result, media_ports.get(node_id))
        if entry.get("output_kind") != OUTPUT_KIND_TEXT
    ]
    if entries:
        message["outputs"] = entries
    return message


# ── metrics ───────────────────────────────────────────────────────────────


def metric_points_from_progress(
    node_id: str, payload: Any, *, explicit_metric_nodes: Iterable[str],
    last_step: int,
) -> tuple[list[MetricPoint], int]:
    """Infer series from a progress payload, unless the node speaks.

    Returns ``(points, last_step)``: the points to record and the step
    counter to carry into the next call (unchanged when the node is
    skipped). The counter belongs to the caller, one per run, from 0.

    Explicit beats inferred (#122). ``TrainingLoop`` logs ``train_loss``
    and ``val_loss`` through ``context.log_metric`` AND keeps
    ``loss``/``val_loss`` in its epoch payload for the live chart that
    has always read them; mining both would file every validation loss
    twice. The first explicit point from a node retires the inference
    for that node, which also leaves every node that has no metric API
    (a plugin, a chart pack) working exactly as it did.
    """
    if node_id in explicit_metric_nodes:
        return [], last_step
    step = metric_step(payload, fallback=last_step + 1)
    return scalar_metrics(payload, node_id=node_id, step=step), step


def metric_point(signal: MetricSignal) -> MetricPoint:
    """The point an explicitly-logged ``MetricSignal`` records (#122)."""
    return MetricPoint(name=signal.name, value=signal.value, step=signal.step,
                       node_id=signal.node_id)


# ── signals (#122) ────────────────────────────────────────────────────────


def artifact_meta(signal: ArtifactSignal) -> Any:
    """The ``meta`` an ``ArtifactSignal``'s row is stored with.

    The producing node's id rides in it, so the row says where the file
    came from without a column of its own.
    """
    meta = signal.meta
    if signal.node_id:
        meta = {**(meta or {}), "node_id": signal.node_id}
    return meta


def artifact_payload(
    record: ArtifactRecord, node_id: str | None,
) -> dict[str, Any]:
    """The ``artifact`` event payload for a row that has been stored.

    Built from the STORED record: an ``artifact`` event must describe the
    row a client will find.
    """
    return {
        "artifact_id": record.id, "kind": record.kind,
        "path": record.path, "meta": record.meta,
        "node_id": node_id,
    }


def dropped_signals_payload(signal: DroppedSignal) -> dict[str, Any]:
    """The ``run_warning`` payload for signals the outbox discarded."""
    return {
        "kind": "dropped_signals", "count": signal.count,
        "detail": ("progress/metric signals were dropped to "
                   "keep the run's outbox bounded; the "
                   "series may have gaps"),
    }


def warning_payload(signal: WarningSignal) -> dict[str, Any]:
    """The ``run_warning`` payload for a ``WarningSignal`` a node sent."""
    return {
        "kind": signal.kind, "detail": signal.detail,
        "node_id": signal.node_id,
    }
