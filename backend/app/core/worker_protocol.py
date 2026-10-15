"""The ``/ws/worker`` protocol: frames between a worker agent and the control plane.

One module, imported by both ends (#771 the agent, #772 the control plane),
so a frame has exactly one definition. Pure: pydantic and the standard
library only. In particular nothing here imports torch, because the agent
process never may (only its run subprocesses initialise CUDA).

Wire format
-----------
Every frame is ONE JSON text message: an object whose ``type`` key names
the frame and whose other keys are that frame's fields, flat (no
``data`` envelope). :func:`encode_frame` writes one, :func:`decode_agent_frame`
and :func:`decode_control_frame` read one. JSON is written with
``allow_nan=True``: an event payload crosses this wire as the Python value
the engine produced, NaN included, and the control plane sanitises it with
the same ``json_safe`` / ``cap_event_payload`` an in-process run goes
through. Converting NaN here would make a worker run's log differ from a
local one's.

Direction and order
-------------------
The agent dials out (``ws(s)://<control>/ws/worker?token=...``); the control
plane never connects to a worker.

1. agent -> ``hello``; control plane -> ``welcome`` (or ``reject`` and close).
2. agent -> ``run_alive`` right after ``welcome`` when it has runs in
   flight from a previous connection.
3. agent -> ``heartbeat`` every ``welcome.heartbeat_s`` seconds.
4. control plane -> ``assign``; the agent answers with ``run_started``, then
   any number of ``event`` / ``metrics`` / ``artifact`` / ``warning``, then
   exactly one ``run_finished``. A refused assignment is a ``run_finished``
   with status ``failed`` and no ``run_started``.
5. control plane -> ``cancel`` / ``drain`` at any time.

Per-run frames carry ``seq``: 1 for the first frame the agent sends about a
run, +1 for every frame after it, ``run_finished`` included. #783 acks and
replays by it; until then it lets a reader prove a stream has no gaps.

What the frames carry
---------------------
- ``event``: one run event, ``event_type`` + ``payload`` exactly as
  ``core.run_events`` built them in the run subprocess. The consumer hands
  them to ``RunSinks.emit``. Only the node-level vocabulary travels
  (``node_status``, ``inner_node_status``, ``run_warning`` for dropped
  signals); ``execution_start`` and the terminal event stay the
  control plane's, as they are for an in-process run.
- ``metrics``: points for ``RunSinks.record_metrics``. Never empty: the
  agent does not forward the empty batches the engine bridge produces to
  poke the batcher, so the consumer should poke its own batcher (call
  ``record_metrics([])``) on every ``event`` frame.
- ``artifact`` / ``warning``: an ``ArtifactSignal`` / ``WarningSignal``'s
  fields, for ``RunSinks.record_artifact`` / ``RunSinks.warn``.
- ``run_finished.status``: ``succeeded``, ``failed`` or ``cancelled`` as an
  ``ExecutionOutcome`` reports them, or ``interrupted``: the run did not
  stop within ``grace_s`` of a ``cancel`` and the agent terminated it
  (``last_node`` names the node that was running). A subprocess that died
  without reporting is ``failed`` with ``exit_code`` and ``stderr_tail``.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

#: Bumped on any change a peer of the previous version would misread. Sent
#: in ``hello`` and ``welcome``; #781 refuses a mismatch.
PROTOCOL_VERSION = 1

#: Where the control plane serves the protocol.
WORKER_WS_PATH = "/ws/worker"

# ── frame types: agent -> control plane ──────────────────────────────────
HELLO = "hello"
HEARTBEAT = "heartbeat"
RUN_STARTED = "run_started"
EVENT = "event"
METRICS = "metrics"
ARTIFACT = "artifact"
WARNING = "warning"
RUN_FINISHED = "run_finished"
RUN_ALIVE = "run_alive"

# ── frame types: control plane -> agent ──────────────────────────────────
WELCOME = "welcome"
ASSIGN = "assign"
CANCEL = "cancel"
DRAIN = "drain"
REJECT = "reject"

AGENT_FRAME_TYPES = (HELLO, HEARTBEAT, RUN_STARTED, EVENT, METRICS, ARTIFACT,
                     WARNING, RUN_FINISHED, RUN_ALIVE)
CONTROL_FRAME_TYPES = (WELCOME, ASSIGN, CANCEL, DRAIN, REJECT)

#: Seconds between heartbeats unless ``welcome`` says otherwise.
DEFAULT_HEARTBEAT_S = 10.0
#: Seconds a cancelled run gets to stop on its own before it is terminated.
DEFAULT_GRACE_S = 30.0

#: ``run_finished.status`` values. The first three are ``ExecutionOutcome``'s.
FINISHED_SUCCEEDED = "succeeded"
FINISHED_FAILED = "failed"
FINISHED_CANCELLED = "cancelled"
FINISHED_INTERRUPTED = "interrupted"
FinishedStatus = Literal["succeeded", "failed", "cancelled", "interrupted"]

#: A run id is used as a directory name on the worker, so it is held to a
#: path-safe alphabet (the control plane's ids are uuid4 hex).
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")

RunId = Annotated[str, Field(pattern=f"^{RUN_ID_PATTERN.pattern}$")]
Seq = Annotated[int, Field(ge=1)]


class _Frame(BaseModel):
    # Both ends run the same version (#764), so an unknown field is a bug,
    # not a newer peer: refuse it rather than drop it silently.
    model_config = ConfigDict(extra="forbid")


class _RunFrame(_Frame):
    run_id: RunId
    seq: Seq


# ── shared shapes ─────────────────────────────────────────────────────────


class DeviceReport(_Frame):
    """One device a worker offers: ``core.resources.DeviceFact``'s fields."""

    value: str
    kind: str
    index: int | None = None
    name: str = ""
    total_bytes: int | None = None
    free_bytes: int | None = None
    source: str = "none"


class MetricPointReport(_Frame):
    """One ``run_store.MetricPoint`` (without ``ts``, which the store stamps)."""

    name: str
    value: float
    step: int
    node_id: str | None = None


class RunAliveEntry(_Frame):
    run_id: RunId
    #: The last ``seq`` the agent sent for this run (#783 replays after it).
    last_seq: int = Field(ge=0)


# ── agent -> control plane ────────────────────────────────────────────────


class Hello(_Frame):
    type: Literal["hello"] = HELLO
    protocol_version: int = PROTOCOL_VERSION
    name: str
    version: str
    #: ``env_fingerprint`` digest. ``None`` in this wave: computing it
    #: imports torch, which the agent process may not; #781 fills it.
    fingerprint: str | None = None
    #: Same reason and owner as ``fingerprint``.
    node_catalog_hash: str | None = None
    devices: list[DeviceReport] = Field(default_factory=list)
    #: Storage label -> path on this worker (#782).
    mounts: dict[str, str] = Field(default_factory=dict)
    #: ``{"system", "release", "machine", "python"}``.
    platform: dict[str, str] = Field(default_factory=dict)


class Heartbeat(_Frame):
    type: Literal["heartbeat"] = HEARTBEAT
    devices: list[DeviceReport] = Field(default_factory=list)
    running: list[str] = Field(default_factory=list)


class RunStarted(_RunFrame):
    type: Literal["run_started"] = RUN_STARTED
    #: The run subprocess's pid, for the worker's own logs and the UI.
    pid: int | None = None


class Event(_RunFrame):
    type: Literal["event"] = EVENT
    event_type: str
    payload: dict[str, Any] | None = None


class Metrics(_RunFrame):
    type: Literal["metrics"] = METRICS
    points: list[MetricPointReport] = Field(min_length=1)


class Artifact(_RunFrame):
    type: Literal["artifact"] = ARTIFACT
    kind: str
    path: str
    meta: dict[str, Any] | None = None
    node_id: str | None = None


class RunWarning(_RunFrame):
    type: Literal["warning"] = WARNING
    kind: str
    detail: str
    node_id: str | None = None


class RunFinished(_RunFrame):
    type: Literal["run_finished"] = RUN_FINISHED
    status: FinishedStatus
    error: str | None = None
    #: The subprocess's exit code when it did not report its own outcome.
    exit_code: int | None = None
    #: The last 4 KB of the subprocess's output, on a crash.
    stderr_tail: str | None = None
    #: The node that was running when the run was terminated.
    last_node: str | None = None
    #: ``ExecutionOutcome.peak_memory`` when the run reported one (#770).
    peak_memory: dict[str, Any] | None = None


class RunAlive(_Frame):
    type: Literal["run_alive"] = RUN_ALIVE
    runs: list[RunAliveEntry] = Field(default_factory=list)


# ── control plane -> agent ────────────────────────────────────────────────


class Welcome(_Frame):
    type: Literal["welcome"] = WELCOME
    protocol_version: int = PROTOCOL_VERSION
    worker_id: str
    heartbeat_s: float = Field(default=DEFAULT_HEARTBEAT_S, gt=0)
    grace_s: float = Field(default=DEFAULT_GRACE_S, ge=0)


class Assign(_Frame):
    type: Literal["assign"] = ASSIGN
    run_id: RunId
    #: ``{"nodes", "edges", "presets", "subgraphs", ...}`` with its secret
    #: params restored: the worker has no vault.
    graph: dict[str, Any]
    #: The run's normalised submit options.
    options: dict[str, Any] = Field(default_factory=dict)
    #: The device the run's ``ExecutionContext`` names, as the control plane
    #: resolved it (its queue key): ``cpu``, ``cuda:2``, ``mps``.
    device: str = "cpu"
    #: The cards the run may use, as this worker numbers them. A CUDA set
    #: becomes the subprocess's ``CUDA_VISIBLE_DEVICES``.
    devices: list[str] = Field(default_factory=list)
    #: Overrides ``welcome.grace_s`` for this run's cancel.
    grace_s: float | None = Field(default=None, ge=0)


class Cancel(_Frame):
    type: Literal["cancel"] = CANCEL
    run_id: RunId
    grace_s: float = Field(default=DEFAULT_GRACE_S, ge=0)


class Drain(_Frame):
    type: Literal["drain"] = DRAIN


class Reject(_Frame):
    type: Literal["reject"] = REJECT
    reason: str


AgentFrame = Annotated[
    Union[Hello, Heartbeat, RunStarted, Event, Metrics, Artifact, RunWarning,
          RunFinished, RunAlive],
    Field(discriminator="type"),
]
ControlFrame = Annotated[
    Union[Welcome, Assign, Cancel, Drain, Reject],
    Field(discriminator="type"),
]

_AGENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(AgentFrame)
_CONTROL_ADAPTER: TypeAdapter[Any] = TypeAdapter(ControlFrame)


class ProtocolError(ValueError):
    """A message that is not a valid frame for the direction it came from."""


def encode_frame(frame: _Frame) -> str:
    """One frame as the JSON text message that goes on the wire."""
    return json.dumps(frame.model_dump(mode="python"), allow_nan=True,
                      separators=(",", ":"), ensure_ascii=False)


def _decode(adapter: TypeAdapter[Any], raw: str | bytes) -> Any:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"frame is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("frame is not a JSON object")
    try:
        return adapter.validate_python(data)
    except ValueError as exc:
        raise ProtocolError(f"invalid {data.get('type')!r} frame: {exc}") from exc


def decode_agent_frame(raw: str | bytes) -> Any:
    """Parse a message the control plane received. Raises ``ProtocolError``."""
    return _decode(_AGENT_ADAPTER, raw)


def decode_control_frame(raw: str | bytes) -> Any:
    """Parse a message the agent received. Raises ``ProtocolError``."""
    return _decode(_CONTROL_ADAPTER, raw)
