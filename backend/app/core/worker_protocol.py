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
and :func:`decode_control_frame` read one. Event payloads arrive
already passed through ``json_safe`` and ``cap_event_payload`` (the run
subprocess applies the control plane's own policy), so they are small and
NaN-free; the control plane applies both again, which changes nothing.

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
6. control plane -> ``ack`` for a run's frames once they are recorded.

Versions and additive features (#781)
-------------------------------------
The policy (#781). Both ends run the SAME CodefyUI version:
``hello.version`` must equal the control plane's, or the agent gets
``reject`` naming both (#764); the control plane reads the version off the
raw JSON first, so even a ``hello`` it cannot parse is refused that way.
Within that rule the protocol grows by addition, and from version 3 on an
addition never bumps :data:`PROTOCOL_VERSION`:

- every frame model IGNORES a field it does not know (``extra="ignore"``),
  so a frame carrying a new optional field still parses on a reader that
  predates it;
- ``hello`` and ``welcome`` each list the optional features their sender
  speaks (:data:`CAPABILITIES`: ``"ack"``, ``"mounts"``,
  ``"transfer"``). A new feature declares its name there, and is used only
  when both lists name it (:func:`shared_capabilities`).

:data:`PROTOCOL_VERSION` is bumped only for a change a peer would misread
(a field changing meaning or type, a frame becoming mandatory). It is 3:
2 added ``ack`` (#783) and 3 the mount fields (#782), before this rule.

Per-run frames carry ``seq``: 1 for the first frame the agent sends about a
run, +1 for every frame after it, ``run_finished`` included. The agent
writes every per-run frame to the run's spool before it sends it
(``agent.spool``, #783); ``ack{run_id, seq}`` says every frame of the run
up to ``seq`` is durable on the control plane, so the agent drops them.
After a reconnect the agent lists its runs in ``run_alive`` (``last_seq``:
the last frame it produced) and replays every frame not yet acknowledged,
in order; the control plane drops a frame whose ``seq`` it has already
seen, so a replay that overlaps what arrived before the drop is harmless.

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
  fields, for ``RunSinks.record_artifact`` / ``RunSinks.warn``. An
  ``artifact`` under the run's mount carries ``mount_label`` and a path
  relative to that mount (#782); ``assign.storage`` names the mount. An
  ``artifact`` the agent uploaded to the control plane (the ``transfer``
  backend, #785) carries ``backend: "transfer"`` and the key the upload
  answered with (``runs/<run_id>/<kind>/<name>``) as its ``path``.
- ``assign.storage.backend == "transfer"`` (#785): the worker holds no
  directory for the control plane's mount. ``storage.inputs`` lists the
  uploads the run reads, by content hash; the agent fetches the ones its
  cache lacks over HTTP before it starts the run subprocess, reporting
  ``warning`` frames of kind ``transfer_progress`` (these may come before
  ``run_started``), and uploads what the run writes before ``run_finished``.
- ``assign.storage.backend == "s3"`` (#791): as ``transfer``, but the
  bytes go between the worker and an S3 / MinIO bucket. ``storage.s3``
  carries the bucket, the run's key prefix and the credentials the agent
  uses (short-lived when the control plane got them from STS); each
  ``storage.inputs`` entry names its object ``key``. The artifact frames
  carry ``backend: "s3"`` and the object key. ``credentials`` replaces a
  running run's short-lived credentials before they expire.
- ``run_finished.status``: ``succeeded``, ``failed`` or ``cancelled`` as an
  ``ExecutionOutcome`` reports them, or ``interrupted``: after a ``cancel``
  the subprocess ended without reporting, because the agent terminated it
  after ``grace_s`` or it exited cleanly while stopping (``last_node``
  names the node that was running). A subprocess that died without
  reporting otherwise is ``failed`` with ``exit_code``. ``stderr_tail``
  rides on every unreported end.
- A frame the agent cannot encode is replaced by a ``warning`` with the
  same ``seq`` (kind ``unserialisable_frame``), never dropped silently.

Recorded outputs (#789, capability ``record_outputs``)
------------------------------------------------------
A run assigned with ``options.record_outputs`` keeps its captures on the
worker: in the run subprocess's ``RunOutputStore`` while it runs, then in
files under the run's directory there. The control plane reads them on
demand, one read of ``/api/execution/outputs/*`` at a time:

- control plane -> ``fetch_output{request_id, run_id, op, params,
  timeout_s}``: ``op`` is the read (``list``, ``value``, ``stats``,
  ``steps``, ``grads``, ``delete``), ``params`` its query (``node_id``,
  ``port``, ``slice``, ``max_elements``, ``preview``, ``inner``), and
  ``timeout_s`` how long the control plane waits for the answer (the
  agent gives up a little sooner, so its own limits never outlast it);
- control plane -> ``cancel_fetch{request_id}`` when it stopped waiting
  for a read (it timed out, or the browser went away): the agent stops
  answering it and frees its place;
- agent -> ``output_answer{request_id, status, part, parts, body, final}``:
  the HTTP status and JSON body text the read answers with, the body cut
  into ``parts`` frames (``part`` 0 first) so no frame outgrows a socket's
  message limit; ``final`` says the run has ended there (its captures
  cannot change, so the control plane may cache what it derived);
- control plane -> ``drop_outputs{run_id}`` when its own retention evicts
  the run: the agent deletes the captures.

These frames carry no ``seq`` and are never spooled: a read whose
connection drops is answered by the control plane itself ("worker
unreachable"), and the browser asks again.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Iterable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

#: Bumped on any change a peer of the previous version would misread. Sent
#: in ``hello`` and ``welcome``; a mismatch is refused (#781). 2: ``ack`` and
#: the spool replay (#783). 3: storage mounts (#782): ``assign.storage``,
#: ``artifact.mount_label`` and ``heartbeat.mounts``. From 3 on, frames
#: ignore unknown fields and an additive feature declares a capability
#: instead of a bump (see the module docstring).
PROTOCOL_VERSION = 3

#: The optional protocol features this build speaks, sent in ``hello`` (the
#: agent's) and ``welcome`` (the control plane's). ``ack`` (#783) and
#: ``mounts`` (#782) came with PROTOCOL_VERSION 2 and 3, before the
#: capability rule, and are listed for the record: every peer of version 3
#: speaks both. Every additive feature from now on (#784's tunnel, #785's
#: ``transfer`` storage) adds its name here instead of bumping the version.
#: ``transfer`` (#785): ``assign.storage.backend == "transfer"`` with
#: ``inputs`` and ``max_upload_bytes``, and ``artifact.backend``; the control
#: plane sends a run through the transfer backend only to a worker that
#: lists it. ``memory_budget`` (#788): ``assign.memory_budgets``; the agent
#: caps each listed card with ``set_per_process_memory_fraction`` in the run
#: subprocess and sends ``run_started`` only once the cap is in place. A
#: worker that does not list it is never given a ``shared`` card's runs.
#: ``one_shot`` (#790): ``hello.one_shot`` (with
#: ``one_shot_run_id`` and ``instance_id``) is understood: the control plane
#: keeps such a worker out of ``any`` and gives it exactly one run; a
#: one-shot agent refuses a control plane whose ``welcome`` lacks it.
#: ``s3`` (#791): ``assign.storage.backend == "s3"`` with ``storage.s3`` and
#: ``inputs[].key``, ``artifact.backend == "s3"``, and the ``credentials``
#: frame. An agent lists it only when it can load an S3 client (boto3):
#: :func:`agent_capabilities`.
#: ``record_outputs`` (#789): ``options.record_outputs`` is honoured, and
#: ``fetch_output`` / ``drop_outputs`` / ``output_answer`` are understood.
#: The control plane assigns a recorded run only to a worker listing it.
CAPABILITIES: tuple[str, ...] = ("ack", "mounts", "transfer",
                                 "memory_budget", "one_shot", "s3",
                                 "record_outputs")

#: The capability name of the transfer storage backend (#785).
CAPABILITY_TRANSFER = "transfer"
#: The capability name of enforced memory budgets (#788).
CAPABILITY_MEMORY_BUDGET = "memory_budget"
#: The capability name of one-shot workers (#790).
CAPABILITY_ONE_SHOT = "one_shot"
#: ``target.pool`` value (#790): any connected one-shot worker. The only
#: pool there is.
POOL_ONE_SHOT = "one-shot"
#: The capability name of the S3 artifact backend (#791).
CAPABILITY_S3 = "s3"
#: The capability name of recorded outputs on workers (#789).
CAPABILITY_RECORD_OUTPUTS = "record_outputs"


def agent_capabilities(s3_client: bool) -> list[str]:
    """What an agent lists in ``hello``: :data:`CAPABILITIES`, without
    ``s3`` when it has no S3 client to speak it with (*s3_client*)."""
    return [c for c in CAPABILITIES if s3_client or c != CAPABILITY_S3]

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
OUTPUT_ANSWER = "output_answer"

# ── frame types: control plane -> agent ──────────────────────────────────
WELCOME = "welcome"
ASSIGN = "assign"
CANCEL = "cancel"
DRAIN = "drain"
REJECT = "reject"
ACK = "ack"
CREDENTIALS = "credentials"
FETCH_OUTPUT = "fetch_output"
DROP_OUTPUTS = "drop_outputs"
CANCEL_FETCH = "cancel_fetch"

AGENT_FRAME_TYPES = (HELLO, HEARTBEAT, RUN_STARTED, EVENT, METRICS, ARTIFACT,
                     WARNING, RUN_FINISHED, RUN_ALIVE, OUTPUT_ANSWER)
CONTROL_FRAME_TYPES = (WELCOME, ASSIGN, CANCEL, DRAIN, REJECT, ACK,
                       CREDENTIALS, FETCH_OUTPUT, DROP_OUTPUTS,
                       CANCEL_FETCH)

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

#: ``target.worker`` values with a meaning of their own (#769): this
#: control-plane process, and "wherever there is room". Neither can be a
#: worker's name. The one definition every module imports (#772).
TARGET_LOCAL = "local"
TARGET_ANY = "any"
#: ``enroll`` and ``enrollments`` are taken by the REST surface
#: (``/api/workers/enroll``, ``/api/workers/enrollments``, #780): a worker
#: under either name would collide with them in ``/api/workers/{id}``.
RESERVED_WORKER_NAMES = frozenset({TARGET_LOCAL, TARGET_ANY, "enroll",
                                   "enrollments"})
#: Worker names, matched with ``fullmatch`` (the ``target.worker`` alphabet).
WORKER_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")

_DEVICE_SYNTAX = re.compile(r"^(cpu|cuda|mps)(?::(\d+))?$")


def canonical_device(device: str) -> str:
    """One spelling per device, for comparing sets on both ends.

    Trimmed and lower-cased; ``cuda:01`` -> ``cuda:1``; ``mps`` -> ``mps:0``
    (there is one); ``cpu`` -> ``cpu``. A bare ``cuda`` stays bare: which
    card it means is the executing process's to decide. ``ValueError`` for
    anything that is not ``cpu``, ``cuda[:N]`` or ``mps[:N]``.
    """
    match = _DEVICE_SYNTAX.match(str(device).strip().lower())
    if match is None:
        raise ValueError(f"unknown device {device!r}")
    kind, index = match.groups()
    if kind == "mps" and index is None:
        index = "0"
    return kind if index is None else f"{kind}:{int(index)}"


#: A run id is used as a directory name on the worker, so it is held to a
#: path-safe alphabet (the control plane's ids are uuid4 hex).
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")

RunId = Annotated[str, Field(pattern=f"^{RUN_ID_PATTERN.pattern}$")]
Seq = Annotated[int, Field(ge=1)]


def shared_capabilities(mine: Iterable[str], theirs: Iterable[str]) -> list[str]:
    """The features both peers listed, sorted: the ones a connection uses."""
    return sorted(set(mine) & set(theirs))


class _Frame(BaseModel):
    # Additive growth (#781): a field this build does not know is dropped,
    # not refused, so a newer optional field never breaks an older reader.
    # A field whose meaning changes bumps PROTOCOL_VERSION instead.
    # No frame's input in a validation error (#791): an ``assign`` carries
    # S3 credentials and secret params, and ``ProtocolError`` is logged.
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)


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
    #: A CUDA card's physical identity (NVML UUID, lower case, no ``GPU-``
    #: prefix), or None. A same-host worker's card shares this server's
    #: slot only when the UUIDs match (#773).
    uuid: str | None = None


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


class FingerprintReport(_Frame):
    """A worker's environment fingerprint (``core.env_fingerprint``)."""

    #: ``collect_fingerprint()``'s document, as the worker computed it.
    document: dict[str, Any]
    #: Its ``fingerprint_digest``. The control plane recomputes the digest
    #: from *document* and keeps its own (they agree unless the document
    #: was altered on the way).
    digest: str


class Hello(_Frame):
    type: Literal["hello"] = HELLO
    protocol_version: int = PROTOCOL_VERSION
    name: str
    #: ``core.version.get_version()``; must equal the control plane's.
    version: str
    #: The agent's environment (#781), collected once at agent start by a
    #: short-lived helper subprocess (the agent itself never imports
    #: torch). None when that helper failed.
    fingerprint: FingerprintReport | None = None
    #: ``node_registry.node_catalog().hash`` on the worker (#781): which
    #: node types it can execute, and how each is defined. None with
    #: ``fingerprint``.
    node_catalog_hash: str | None = None
    #: The node types behind ``node_catalog_hash``, sorted, so the control
    #: plane can say which differ.
    node_types: list[str] = Field(default_factory=list)
    #: Optional features this agent speaks (:data:`CAPABILITIES`).
    capabilities: list[str] = Field(default_factory=list)
    devices: list[DeviceReport] = Field(default_factory=list)
    #: Storage label -> path on this worker (#782).
    mounts: dict[str, str] = Field(default_factory=dict)
    #: ``{"system", "release", "machine", "python"}``.
    platform: dict[str, str] = Field(default_factory=dict)
    #: ``core.machine_identity.machine_identity()`` (a digest, never the raw
    #: OS id), or None. Equal to the control plane's own: the worker runs on
    #: the same machine and shares its devices (#773).
    machine_id: str | None = None
    #: ``cdui agent start --one-shot`` (#790): this agent takes exactly one
    #: run, reports it and exits. The control plane keeps it out of ``any``
    #: placement; only a run naming it (``target.worker``) or the one-shot
    #: pool (``target.pool``) goes there.
    one_shot: bool = False
    #: ``--run-id``: the one run this one-shot agent takes, or None (the
    #: first one placed on it).
    one_shot_run_id: RunId | None = None
    #: Random per agent process, so the control plane can tell this agent
    #: reconnecting from another agent taking the same name (#790). None
    #: from an agent that does not say.
    instance_id: str | None = Field(default=None, max_length=64)
    #: The run this one-shot agent already took, on a reconnect (#790):
    #: a control plane that restarted since has no other record of it, and
    #: keeps it from placing a second run there.
    one_shot_taken: RunId | None = None


class Heartbeat(_Frame):
    type: Literal["heartbeat"] = HEARTBEAT
    devices: list[DeviceReport] = Field(default_factory=list)
    running: list[str] = Field(default_factory=list)
    #: The labels the agent can serve now (#782): a configured mount whose
    #: directory is missing (a share not mounted yet) is left out until it
    #: appears. None: unchanged since ``hello``.
    mounts: dict[str, str] | None = None


class RunStarted(_RunFrame):
    type: Literal["run_started"] = RUN_STARTED
    #: The run subprocess's pid, for the worker's own logs and the UI.
    pid: int | None = None
    #: A patch for the run's ``execution_facts`` (#774), merged key by key
    #: the way ``peak_memory`` is (``RunStore.update_execution_facts``):
    #: ``{"visible_devices": "2,0", "device_aliases": {"cuda:2": "cuda:0",
    #: "cuda:0": "cuda:1"}}``. ``visible_devices`` is the subprocess's
    #: ``CUDA_VISIBLE_DEVICES`` as set (``""``: no card; ``None``: left as
    #: the agent inherited it); ``device_aliases`` maps each physical card
    #: of the set to its in-run name, which is its position in the set.
    execution_facts: dict[str, Any] | None = None


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
    #: Absolute on the worker, or, with ``mount_label``, relative to that
    #: mount (#782): the control plane then records a ``mount`` row.
    path: str
    meta: dict[str, Any] | None = None
    node_id: str | None = None
    mount_label: str | None = None
    #: ``transfer`` (#785): the file was uploaded to the control plane and
    #: ``path`` is the key it is stored under there. ``s3`` (#791): the file
    #: was uploaded to the run's S3 bucket and ``path`` is its object key.
    #: None: a path on the worker (absolute, or relative to ``mount_label``).
    backend: Literal["transfer", "s3"] | None = None


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


#: A ``fetch_output`` request id: the control plane's, echoed back.
RequestId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
#: The reads ``fetch_output`` names (#789).
OutputOp = Literal["list", "value", "stats", "steps", "grads", "delete"]
#: Characters of body text per ``output_answer`` frame (#789): well under
#: any socket's message limit (uvicorn's default is 16 MiB).
OUTPUT_ANSWER_PART_CHARS = 1024 * 1024


class OutputAnswer(_Frame):
    """One part of the answer to a ``fetch_output`` (#789)."""

    type: Literal["output_answer"] = OUTPUT_ANSWER
    request_id: RequestId
    #: The HTTP status the read answers with (200, 204, 404, 413, ...).
    status: int = Field(ge=100, le=599)
    part: int = Field(default=0, ge=0)
    parts: int = Field(default=1, ge=1)
    #: This part of the JSON body text ("" for a 204).
    body: str = ""
    #: The run has ended on the worker: its captures cannot change.
    final: bool = False


# ── control plane -> agent ────────────────────────────────────────────────


class Welcome(_Frame):
    type: Literal["welcome"] = WELCOME
    protocol_version: int = PROTOCOL_VERSION
    worker_id: str
    #: Optional features the control plane speaks (:data:`CAPABILITIES`).
    capabilities: list[str] = Field(default_factory=list)
    heartbeat_s: float = Field(default=DEFAULT_HEARTBEAT_S, gt=0)
    grace_s: float = Field(default=DEFAULT_GRACE_S, ge=0)
    #: How long this control plane keeps a disconnected worker's runs
    #: (``CODEFYUI_WORKER_RESUME_GRACE_S``, #783): the agent retries a
    #: transfer upload through an outage that long (#785). Optional and
    #: additive (no capability): None from a control plane that predates
    #: it, and the agent then uses its default.
    resume_grace_s: float | None = Field(default=None, ge=0)


#: ``TransferInput.kind`` -> the run's data setting its file goes under.
TRANSFER_KINDS = ("data_file", "image", "model")


class TransferInput(_Frame):
    """One upload a ``transfer`` run reads (#785): where it goes in the
    run's data directories (``kind`` + ``name``, the name the graph uses)
    and what it must hash to."""

    kind: Literal["data_file", "image", "model"]
    name: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    #: ``s3`` only (#791): the object the agent downloads it from
    #: (``<prefix>inputs/<sha256>``, uploaded by the control plane).
    key: str | None = Field(default=None, max_length=1024)


class S3Access(_Frame):
    """Where an ``s3`` run's objects go, and the credentials to reach them
    (#791). Secret fields stay out of ``repr`` and validation errors: the
    agent holds them in memory only (#251) and never logs them."""

    #: ``CODEFYUI_S3_ENDPOINT_URL``; None: AWS S3 for ``region``.
    endpoint_url: str | None = None
    region: str | None = None
    bucket: str = Field(min_length=1)
    #: The run's key prefix, ``[<CODEFYUI_S3_PREFIX>]runs/<run_id>/``: every
    #: object the run writes goes under it.
    prefix: str = Field(min_length=1)
    addressing_style: Literal["auto", "path", "virtual"] = "auto"
    access_key_id: str = Field(repr=False)
    secret_access_key: str = Field(repr=False)
    #: Set for temporary (STS) credentials, with :attr:`expires_at`.
    session_token: str | None = Field(default=None, repr=False)
    #: When temporary credentials expire (Unix time); None: static ones.
    #: The control plane sends ``credentials`` before then.
    expires_at: float | None = None


class Storage(_Frame):
    """Where an assigned run's files are (#782, ``assign.storage``).

    ``mount`` is the label the control plane gives its data directory: the
    run subprocess derives its data directories from the worker's own
    directory for that label (``--mount``).

    ``dirs`` is set only when the control plane LENDS its directories to a
    same-host worker (#773) that declared none for the label
    (``core.mounts.lends_mount``): ``CODEFYUI_<NAME>`` -> path for
    ``DATA_FILES_DIR``, ``IMAGES_DIR``, ``MODELS_DIR``, ``MEDIA_DIR``,
    exactly as the control plane has them configured, which the run
    subprocess sets as they are.

    ``relative_artifacts``: report a file under the mount relative to it
    (with ``artifact.mount_label``). Off for a same-host worker: its files
    are on the control plane's machine, recorded as ``local`` rows with
    their absolute path.

    ``backend == "transfer"`` (#785): the worker holds no directory for
    ``mount``, so nothing is shared. The run gets a fresh data directory of
    its own on the worker, holding ``inputs`` (fetched by content hash from
    the control plane's download endpoints, cached per worker), and every
    file it reports or writes there is uploaded to the control plane. See
    ``core.transfer`` (the control plane's half) and ``agent.transfer``.
    """

    backend: Literal["mount", "transfer", "s3"] = "mount"
    mount: str
    dirs: dict[str, str] | None = None
    relative_artifacts: bool = True
    #: ``transfer`` only (#785): the uploads the run reads, which the agent
    #: fetches into its cache by ``sha256`` and links into the run's own
    #: data directory under ``name``.
    inputs: list[TransferInput] | None = None
    #: ``transfer`` only: the largest file the control plane accepts on
    #: ``POST /api/runs/{id}/artifacts/upload`` (its ``MAX_UPLOAD_SIZE``),
    #: so the agent does not send a file that would be refused.
    max_upload_bytes: int | None = Field(default=None, ge=0)
    #: ``s3`` only (#791): the bucket, the run's prefix and credentials.
    s3: S3Access | None = None


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
    #: The control plane's mount label (and, for a same-host worker without
    #: one, its path); None from a peer before #782.
    storage: Storage | None = None
    #: Device (as this worker numbers it, one of ``devices``) -> the run's
    #: memory budget on it in bytes (#788): the run was admitted to a
    #: ``shared`` card by memory. The agent caps the run subprocess's
    #: allocator to it before the run starts, and ``run_started`` comes
    #: after the cap is set. Empty: no budget, the card is the run's alone.
    memory_budgets: dict[str, Annotated[int, Field(gt=0)]] = Field(
        default_factory=dict)


class Cancel(_Frame):
    type: Literal["cancel"] = CANCEL
    run_id: RunId
    #: Seconds before the run is terminated. ``None``: the assignment's
    #: ``grace_s``, else ``welcome.grace_s``, else :data:`DEFAULT_GRACE_S`.
    grace_s: float | None = Field(default=None, ge=0)


class Drain(_Frame):
    type: Literal["drain"] = DRAIN


class Credentials(_Frame):
    """New credentials for a running ``s3`` run (#791), sent before the
    previous temporary ones expire; only to an agent that lists ``s3``."""

    type: Literal["credentials"] = CREDENTIALS
    run_id: RunId
    s3: S3Access


class Reject(_Frame):
    type: Literal["reject"] = REJECT
    reason: str


class FetchOutput(_Frame):
    """A read of a worker run's recorded outputs (#789); answered with
    ``output_answer`` frames carrying the same ``request_id``. Sent only
    to an agent listing ``record_outputs``."""

    type: Literal["fetch_output"] = FETCH_OUTPUT
    request_id: RequestId
    run_id: RunId
    op: OutputOp
    #: The read's query: ``node_id``, ``port``, ``slice``,
    #: ``max_elements``, ``preview``, ``inner``, as the route got them.
    params: dict[str, Any] = Field(default_factory=dict)
    #: Seconds the control plane waits for the answer; None: unsaid (the
    #: agent assumes the default, 60).
    timeout_s: float | None = Field(default=None, gt=0)


class CancelFetch(_Frame):
    """The control plane no longer waits for the answer to *request_id*
    (#789): the agent stops working on it. Sent only to an agent listing
    ``record_outputs``; one for a read already answered is ignored."""

    type: Literal["cancel_fetch"] = CANCEL_FETCH
    request_id: RequestId


class DropOutputs(_Frame):
    """The control plane no longer retains *run_id*'s captures (#789): the
    agent deletes them."""

    type: Literal["drop_outputs"] = DROP_OUTPUTS
    run_id: RunId


class Ack(_Frame):
    """Every frame of *run_id* up to and including *seq* is recorded (#783).

    Cumulative: an ``ack`` covers every earlier one. The agent truncates the
    run's spool to the frames after *seq*, and deletes it once the frame
    acknowledged is the run's ``run_finished``.
    """

    type: Literal["ack"] = ACK
    run_id: RunId
    seq: Seq


AgentFrame = Annotated[
    Union[Hello, Heartbeat, RunStarted, Event, Metrics, Artifact, RunWarning,
          RunFinished, RunAlive, OutputAnswer],
    Field(discriminator="type"),
]
ControlFrame = Annotated[
    Union[Welcome, Assign, Cancel, Drain, Reject, Ack, Credentials,
          FetchOutput, DropOutputs, CancelFetch],
    Field(discriminator="type"),
]

_AGENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(
    AgentFrame, config=ConfigDict(hide_input_in_errors=True))
_CONTROL_ADAPTER: TypeAdapter[Any] = TypeAdapter(
    ControlFrame, config=ConfigDict(hide_input_in_errors=True))


class ProtocolError(ValueError):
    """A message that is not a valid frame for the direction it came from."""


def encode_frame(frame: _Frame) -> str:
    """One frame as the JSON text message that goes on the wire.

    ``default=str`` is the control plane's rule for measuring an event
    payload (``run_events.json_size``): a value JSON has no type for is
    sent as its ``str`` rather than failing the frame. What can still
    raise (a circular structure) is the sender's to handle.
    """
    return json.dumps(frame.model_dump(mode="python"), allow_nan=True,
                      separators=(",", ":"), ensure_ascii=False, default=str)


def _decode(adapter: TypeAdapter[Any], raw: str | bytes) -> Any:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"frame is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("frame is not a JSON object")
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        # Locations and messages only, never the input (#791: an assign
        # carries credentials), whatever the models' config says.
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False,
                                    include_context=False))
        raise ProtocolError(
            f"invalid {data.get('type')!r} frame: {detail}") from None
    except ValueError as exc:
        raise ProtocolError(f"invalid {data.get('type')!r} frame: {exc}") from exc


def decode_agent_frame(raw: str | bytes) -> Any:
    """Parse a message the control plane received. Raises ``ProtocolError``."""
    return _decode(_AGENT_ADAPTER, raw)


def decode_control_frame(raw: str | bytes) -> Any:
    """Parse a message the agent received. Raises ``ProtocolError``."""
    return _decode(_CONTROL_ADAPTER, raw)
