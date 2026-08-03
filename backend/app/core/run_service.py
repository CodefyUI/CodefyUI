"""Server-owned graph runs — the Run Service (#120).

A run used to live inside the WebSocket handler: ``ws_execution.py`` created
the ``asyncio.Task``, and it cancelled that task on ``WebSocketDisconnect``.
Closing the browser tab killed a training job. Submitting a second graph on
the same socket killed the first. Both are properties of *where the task was
owned*, not of the engine — ``graph_engine.execute_graph`` is already a
perfectly reusable coroutine.

This module owns the task instead. The service is a plain object (no
FastAPI, no HTTP — the ``db.py``/``api_contract.py`` precedent), constructed
once in the lifespan and reachable at ``app.state.run_service``. Its
responsibilities, and nothing else:

- **submit** — persist the run (``RunStore``), then SCHEDULE it. Since #123
  those are genuinely two steps: a queued-lane run joins the FIFO for its
  device and starts when that device has a free slot.
- **observe** — every engine callback is appended to ``exec_run_events`` and
  fanned out to live subscribers; scalar progress values are batched into
  ``exec_run_metrics``. Since #122 the same path also carries what nodes
  report DIRECTLY from their worker threads — ``context.log_metric`` and
  ``context.log_artifact`` — through ``_signal_bridge``.
- **cancel** — cooperative, via the ``ExecutionContext`` flag (never
  ``Task.cancel``; see ``cancel``).
- **survive** — startup recovery retires rows a dead process left
  ``running``; shutdown drains in-flight tasks BEFORE the database closes.

Layering
--------
``core`` never imports ``api``. The one exception #120 shipped — a
function-local import of the #117 output-entry builders out of
``api/ws_execution.py`` — is gone: #121 moved them to
``core.output_entries``, where they belong now that the WebSocket is a
*view* over this service rather than the thing that produces node_status.

Lanes
-----
``options["lane"]`` labels where a run came from. ``queued`` (the default)
is a server-owned, fully isolated run — the shape the FIFO queue schedules.
``interactive`` is the canvas: a live editor session that expects its
``nn.Module`` weights, its execution cache and its captured outputs to
carry across Run clicks. Those things are process-local by nature and
cannot be reconstructed from a stored row, so they do NOT ride in
``options``; the submitter hands them over as an :class:`InteractiveSession`
and the service refuses one on any other lane. Isolation is therefore the
DEFAULT and statefulness is a capability you must pass in, rather than a
flag anyone can set.

Scheduling (#123)
-----------------
Two admission policies, because the two lanes fail in opposite directions.

``queued`` runs join a **per-device FIFO**. The queue key is the resolved
device in canonical form (``canonical_queue_key(resolve_device(...))``:
``cpu``, ``cuda:0``, ``mps:0`` — the canonicalisation is what stops
``cuda`` and ``cuda:0`` being two queues over one card), so a six-hour
CUDA job and a CPU preprocessing run never wait on each
other, and two CUDA jobs on one card never fight over its VRAM. Each key
admits :data:`QueueLimits` runs at once — 1 for an accelerator, 2 for
``cpu`` — and everything else waits in submit order. Waiting is CHEAP: a
pending run holds a run id, its options and a wake-up channel, and nothing
else; its graph is re-read from ``exec_runs.graph_snapshot`` at promotion,
so a queue two hundred deep costs no more memory than the rows already do.

``interactive`` runs BYPASS the FIFO entirely — a classroom demo must not
sit behind a training job — and are bounded two other ways instead:

- a hard cap (``RUN_INTERACTIVE_MAX_CONCURRENT``, default 2) on how many
  may be in flight at once, and
- **one run per editor session**, identified by the socket's
  ``ExecutionCache`` (see ``RunService._session_key`` — the session WRAPPER
  is rebuilt per message, the cache is not). Two concurrent runs behind one
  cache would serve each other half-built tensors under a matching key —
  the hazard #121 disclosed and left to the frontend's disabled Run button.
  It is refused here, structurally, so no client can provoke it.

Both refusals raise :class:`RunServiceUnavailable` (REST 503, and an
``execution_error`` frame on the canvas) rather than parking the run: a
live user can retry, and a click that silently joined an invisible queue
is indistinguishable from a hang.

Note what interactive bypass does NOT mean: an interactive run still uses
its device, so it can push a card past that device's queued-lane cap. That
is the deliberate trade — the FIFO exists so unattended work is orderly,
not so a device is exclusive.


Ids
---
ONE id per run: ``RunStore``'s ``uuid4().hex``. It is the ``exec_runs.id``,
the ``ExecutionContext.execution_id``, and the ``run_id`` handed to
``execute_graph`` (so ``RunOutputStore`` entries key off the same string).
``ExecutionContext``'s own ``str(uuid4())`` default is never used here —
two id shapes for one run is how outputs stop being findable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import weakref
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, NamedTuple

from ..config import settings
from .cache import ExecutionCache
from .db import utc_now_iso
#: ``_current_cuda_index`` is imported rather than re-implemented here:
#: "which card does a bare ``cuda`` mean in this process" is one policy
#: decision, and #135 gave ``device_utils`` a second copy of it for the
#: index-validation fallback. Two copies of a policy drift.
from .device_utils import _current_cuda_index, resolve_device
from .execution_context import (
    ArtifactSignal,
    CancellationError,
    DroppedSignal,
    ExecutionContext,
    MetricSignal,
)
from .graph_engine import GraphValidationError, build_preset_fallback, execute_graph
from .node_state_store import NodeStateStore
from .output_entries import build_node_output_entries, declared_media_ports
from .run_output_store import RunOutputStore
from .run_store import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    EventRecord,
    MetricPoint,
    RunStore,
    json_safe,
)
from .seeding import MAX_SEED as SEEDING_MAX_SEED
from .seeding import seed_rngs

logger = logging.getLogger(__name__)


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

#: ``execution_stopped`` payload discriminator. The WS protocol has one
#: "stopped" frame; the run ROW distinguishes cancelled from interrupted,
#: and this key carries the same fact into the replay log.
STOP_REASON_CANCELLED = "cancelled"
STOP_REASON_INTERRUPTED = "interrupted"

# ── submit options ────────────────────────────────────────────────────────

#: The complete option vocabulary. Unknown keys are REJECTED rather than
#: ignored: ``{"devcie": "cuda"}`` silently running on CPU for forty minutes
#: is the failure mode this closes. Extending the set is a one-line additive
#: change — #121 added the canvas flags below and #123 (queue priority) will
#: do the same, deliberately.
#:
#: Everything here is JSON and lands verbatim in ``exec_runs.options``, so a
#: stored run fully describes its own configuration. Process-local
#: collaborators (an execution cache, the module store) are NOT options; see
#: :class:`InteractiveSession`.
OPTION_KEYS = frozenset({
    "device", "seed", "record_outputs", "lane",
    # Reproducibility (#134). ``deterministic`` rides with ``seed`` because
    # it is the other half of the same request: a seed fixes what is drawn,
    # this asks torch for kernels that combine those draws the same way
    # twice.
    "deterministic",
    # Canvas/teaching flags (#121). They mirror the fields the WebSocket
    # execute message has carried since the A1-A3 educational features and
    # map 1:1 onto ``ExecutionContext``.
    "verbose", "graph_id", "weights_persistent", "backward_mode",
    "auto_backward",
    # Engine error policy, likewise straight off the execute message.
    "error_mode", "max_retries",
})

#: Default lane, and the one the per-device FIFO schedules. Every lane that
#: is not :data:`LANE_INTERACTIVE` is scheduled this way — an unrecognised
#: lane label queues rather than bypassing, because bypass is a capability
#: and a typo must not grant one.
DEFAULT_LANE = "queued"

#: The canvas lane. Three things follow from it and nothing else does: it
#: is the only lane that may carry an :class:`InteractiveSession`, it
#: bypasses the FIFO (so a six-hour training job cannot block a classroom
#: demo), and it is capped instead — see the Scheduling section above.
LANE_INTERACTIVE = "interactive"

MAX_LANE_LENGTH = 64
MAX_NAME_LENGTH = 64
MAX_GRAPH_ID_LENGTH = 128
#: Re-exported from :mod:`app.core.seeding`, which owns the seed space now
#: that seeds are DERIVED rather than just applied. Kept as a module
#: attribute because it is part of this module's published surface.
MAX_SEED = SEEDING_MAX_SEED

#: ``graph_engine.execute_graph``'s error policies, validated as a value for
#: the same reason ``device`` is: a typo'd ``"fail-fast"`` would silently
#: become "continue past every failure" and the run would report success
#: over a graph half of which never executed.
ERROR_MODES = frozenset({"fail_fast", "continue", "retry"})
DEFAULT_ERROR_MODE = "fail_fast"

#: Retry ceiling. ``retry`` mode re-runs a failing node ``max_retries``
#: times; unbounded, one client typo parks a worker thread forever.
MAX_RETRIES_LIMIT = 10

#: The device vocabulary ``resolve_device`` actually understands. Validated
#: as a VALUE, not just a key: strict option keys exist to stop a typo from
#: silently running forty minutes on the wrong device, and ``{"device":
#: "cudda"}`` defeats that entirely if only the key is checked — it sails
#: through, hits ``resolve_device``'s unknown-value branch, and degrades to
#: CPU with nothing but a log line. ``auto`` is accepted because the canvas
#: device selector emits it; note that ``resolve_device`` currently maps it
#: to CPU (pre-existing behaviour, deliberately not changed here).
DEVICE_PATTERN = re.compile(r"^(cpu|auto|cuda(:\d+)?|mps(:\d+)?)$")

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

#: Flush thresholds for the metric batcher (see ``_flush_metrics``).
METRIC_FLUSH_MAX_POINTS = 64
METRIC_FLUSH_INTERVAL_S = 0.5

#: How long ``shutdown`` lets a run stop cooperatively before hard-cancelling.
DEFAULT_SHUTDOWN_GRACE_S = 5.0

#: Refusal for a run whose row was written while the service was stopping.
#: One string, two raise sites (``submit`` after ``create_run``, ``_start``
#: before it registers a task) because it is one situation: the row is
#: durable, nothing will schedule it, and startup recovery retires it.
SHUTDOWN_AFTER_PERSIST = (
    "run service began shutting down while the run was being persisted; "
    "it stays queued and will be retired at startup")

#: Per-subscriber queue depth. Overflow drops (see ``RunSubscription``).
DEFAULT_SUBSCRIBER_QUEUE_SIZE = 256

#: Queue keys that are NOT an accelerator. Everything else — ``cuda:1``,
#: ``mps:0`` — is treated as one, so a backend added to ``resolve_device``
#: tomorrow inherits the conservative limit instead of silently getting the
#: CPU one.
CPU_QUEUE_KEY = "cpu"


def canonical_queue_key(device: str) -> str:
    """Collapse the aliases of one physical device onto ONE queue key.

    ``resolve_device`` hands back what it was asked for: ``cuda`` and
    ``cuda:0`` both survive verbatim, and on a single-GPU box they are the
    same card. As QUEUE KEYS that would be two independent FIFOs over one
    piece of hardware, each admitting its own run — so ``--device cuda`` and
    ``--device cuda:0`` would execute CONCURRENTLY and fight over the same
    VRAM, which is the exact failure the per-device cap exists to prevent.
    Same for ``mps`` versus ``mps:0``.

    Indexing is the canonical form (``cuda`` -> ``cuda:N``, ``mps`` ->
    ``mps:0``) rather than the other direction, because collapsing
    ``cuda:0`` -> ``cuda`` would merge two genuinely different cards on a
    multi-GPU box the moment someone asked for ``cuda:1``.

    Applied once in ``submit`` and used for BOTH the row's ``queue_key`` and
    the ``ExecutionContext.device``, so the queue a run waits in and the
    device it runs on are the same string by construction.
    """
    if device == "cuda":
        return f"cuda:{_current_cuda_index()}"
    if device == "mps":
        return "mps:0"
    return device


@dataclass(frozen=True)
class QueueLimits:
    """How many runs each queue key admits at once.

    A value object rather than four ``settings`` reads scattered through the
    scheduler: the limits are consulted on every promotion, tests need to
    set them without touching global state, and "what is the cap for
    ``cuda:0``" should have exactly one answer computed in one place.

    ``overrides`` beats the per-class default, which is the point of it:
    ``cuda:0=2`` on a 48 GB card is a legitimate thing to want without
    changing what ``cuda:1`` does.
    """

    cpu: int = 2
    gpu: int = 1
    interactive: int = 2
    overrides: tuple[tuple[str, int], ...] = ()

    def for_key(self, queue_key: str) -> int:
        """The concurrency cap for one resolved device string. Always >= 1.

        Never zero: a cap of 0 would accept runs into a queue that can never
        drain, which is a hang dressed up as a configuration. A user who
        wants that should not submit the run.
        """
        for key, value in self.overrides:
            if key == queue_key:
                return max(1, value)
        return max(1, self.cpu if queue_key == CPU_QUEUE_KEY else self.gpu)

    @classmethod
    def from_settings(cls) -> "QueueLimits":
        """Read the caps out of ``config.settings``.

        Called per service construction rather than captured in a module
        constant, for the reason spelled out on
        ``RunService._event_payload_cap_bytes``: a module constant is bound
        at import time, before anything can override the setting.
        """
        return cls(
            cpu=int(settings.RUN_QUEUE_MAX_CONCURRENT_CPU),
            gpu=int(settings.RUN_QUEUE_MAX_CONCURRENT_GPU),
            interactive=int(settings.RUN_INTERACTIVE_MAX_CONCURRENT),
            overrides=parse_queue_overrides(settings.RUN_QUEUE_MAX_CONCURRENT),
        )


def parse_queue_overrides(raw: str) -> tuple[tuple[str, int], ...]:
    """``"cuda:0=2,cpu=8"`` -> ``(("cuda:0", 2), ("cpu", 8))``.

    Split on the LAST ``=`` so a key may contain one; a key legitimately
    contains a colon (``cuda:0``), which is why this is not a simple
    ``dict(pair.split("=") for ...)``.

    A malformed entry is skipped with a warning and never raises: this is
    parsed during startup, and one typo in an env var must not stop the
    server from booting — it degrades to the default cap for that key,
    which is the safe direction.
    """
    out: list[tuple[str, int]] = []
    for chunk in (raw or "").split(","):
        entry = chunk.strip()
        if not entry:
            continue
        key, separator, value = entry.rpartition("=")
        key = key.strip().lower()
        if not separator or not key:
            logger.warning("ignoring run queue override %r: expected key=count",
                           entry)
            continue
        try:
            count = int(value.strip())
        except ValueError:
            logger.warning("ignoring run queue override %r: %r is not a number",
                           entry, value.strip())
            continue
        if count < 1:
            logger.warning("ignoring run queue override %r: count must be >= 1",
                           entry)
            continue
        out.append((key, count))
    return tuple(out)


class RunSubmitError(ValueError):
    """Malformed submit envelope or options. REST maps this to 400."""


class RunServiceUnavailable(RuntimeError):
    """The service cannot take work right now. REST maps this to 503.

    Distinct from a bare ``RuntimeError`` on purpose: "the server is
    shutting down, retry" and "something inside broke" are different answers
    and must not share a status code.
    """


class SubmitResult(NamedTuple):
    """What ``submit`` hands back: the id, and where the run actually is.

    ``status`` is ``running`` when the run started immediately and ``queued``
    when it joined a FIFO that was already full (#123). Read it — do not
    assume: "submitted" and "started" have been different events since the
    queue landed, and the difference is the whole point of the queue.
    """

    run_id: str
    status: str


class CancelOutcome(NamedTuple):
    """Result of a cancel request. ``cancelled`` is False for a no-op."""

    run_id: str
    status: str
    cancelled: bool


@dataclass(frozen=True)
class InteractiveSession:
    """One canvas connection's process-local state, lent to a run (#121).

    Everything here is a live Python object that cannot be written to
    ``exec_runs`` and cannot be rebuilt from it, which is precisely why it
    is a separate argument rather than three more ``options`` keys: a stored
    run must describe itself completely, and "shared an ``nn.Module`` with
    whatever else that browser tab ran" is not a describable configuration.

    Held by the WebSocket handler for the lifetime of a SOCKET, not of a
    run. That is what makes the teaching loop work the way it always has —
    click Run twice and the second execution reuses the first's weights and
    its cached upstream nodes — while the run itself now outlives the
    socket. A reconnect legitimately starts a fresh cache, exactly as a
    page reload always did.

    Accepted only on :data:`LANE_INTERACTIVE` (``submit`` rejects it
    otherwise), so "which runs are stateful" stays answerable from the
    stored row.

    Fields:
      ``cache`` — cross-run ``ExecutionCache``; ``changed_nodes`` is
      meaningless without it, since forcing a re-run only matters where a
      cache hit would otherwise have been served.
      ``changed_nodes`` — the canvas's dirty-node hint for partial
      re-execution.
      ``node_state_store`` — the process-global persistent-module store.
      Reaches ``StatefulModuleMixin`` only when ``weights_persistent`` is
      also set.
    """

    cache: ExecutionCache | None = None
    changed_nodes: tuple[str, ...] = ()
    node_state_store: NodeStateStore | None = None


def normalize_options(raw: Any) -> dict[str, Any]:
    """Validate the submit options and fill in defaults.

    Strict on purpose (see ``OPTION_KEYS``). Returns a NEW dict with every
    key present, so ``exec_runs.options`` always records the full effective
    configuration rather than "whatever the client happened to send" — a
    stored run has to be readable years later without knowing which defaults
    were in force at submit time.

    ``device`` is normalised (trimmed, lowercased) but NOT availability-
    checked here: that resolution happens at start time via
    ``resolve_device`` and lands in ``exec_runs.queue_key``, so the row keeps
    both what was asked for and what was used.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RunSubmitError(
            f"options must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - OPTION_KEYS)
    if unknown:
        raise RunSubmitError(
            f"unknown option(s) {unknown}; supported: {sorted(OPTION_KEYS)}")

    device = raw.get("device", "cpu")
    if not isinstance(device, str) or not device.strip():
        raise RunSubmitError("device must be a non-empty string")
    device = device.strip().lower()
    if DEVICE_PATTERN.match(device) is None:
        raise RunSubmitError(
            f"unknown device {device!r}; expected cpu, auto, cuda, cuda:N, "
            "mps or mps:N")

    seed = raw.get("seed")
    if seed is not None:
        # bool is an int subclass; True is not a seed.
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RunSubmitError("seed must be an integer or null")
        if not 0 <= seed <= MAX_SEED:
            raise RunSubmitError(f"seed must be between 0 and {MAX_SEED}")

    lane = raw.get("lane", DEFAULT_LANE)
    if not isinstance(lane, str) or not lane.strip():
        raise RunSubmitError("lane must be a non-empty string")
    if len(lane) > MAX_LANE_LENGTH:
        raise RunSubmitError(f"lane must be at most {MAX_LANE_LENGTH} chars")

    graph_id = raw.get("graph_id", "")
    if not isinstance(graph_id, str):
        raise RunSubmitError(
            f"graph_id must be a string, got {type(graph_id).__name__}")
    if len(graph_id) > MAX_GRAPH_ID_LENGTH:
        raise RunSubmitError(
            f"graph_id must be at most {MAX_GRAPH_ID_LENGTH} chars")

    error_mode = raw.get("error_mode", DEFAULT_ERROR_MODE)
    if error_mode not in ERROR_MODES:
        raise RunSubmitError(
            f"unknown error_mode {error_mode!r}; expected one of "
            f"{sorted(ERROR_MODES)}")

    max_retries = raw.get("max_retries", 0)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise RunSubmitError("max_retries must be an integer")
    if not 0 <= max_retries <= MAX_RETRIES_LIMIT:
        raise RunSubmitError(
            f"max_retries must be between 0 and {MAX_RETRIES_LIMIT}")

    # ``weights_persistent`` defaults to FALSE here while
    # ``ExecutionContext``'s own default is True, and that inversion is
    # deliberate: a server-owned run is isolated unless someone asks
    # otherwise, whereas the dataclass predates run isolation. Asking is
    # also not enough on its own — without an InteractiveSession there is no
    # module store to persist into, so the flag simply has no effect.
    flags = {}
    for key in ("record_outputs", "verbose", "weights_persistent",
                "backward_mode", "auto_backward", "deterministic"):
        value = raw.get(key, False)
        if not isinstance(value, bool):
            raise RunSubmitError(f"{key} must be true or false")
        flags[key] = value

    return {
        "device": device,
        "seed": seed,
        "lane": lane.strip(),
        "graph_id": graph_id,
        "error_mode": error_mode,
        "max_retries": max_retries,
        **flags,
    }


def normalize_name(raw: Any) -> str | None:
    """Validate the optional run label. Bounded like ``lane``.

    A run name is display text on a list row, not a document: an unbounded
    one would be stored on every row and re-sent on every poll of the Runs
    panel. Blank (or whitespace-only) means unnamed, stored as SQL NULL
    rather than an empty string nobody can tell apart from "not set".
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RunSubmitError(f"name must be a string, got {type(raw).__name__}")
    name = raw.strip()
    if not name:
        return None
    if len(name) > MAX_NAME_LENGTH:
        raise RunSubmitError(f"name must be at most {MAX_NAME_LENGTH} chars")
    return name


def normalize_graph(raw: Any) -> dict[str, Any]:
    """Validate the submit ENVELOPE only — never the graph's semantics.

    Structural validation (unknown node types, missing entry points, cycles)
    stays where it already lives: ``prepare_executable_graph``, inside the
    run. Duplicating it here would mean two validators drifting apart, and
    would make ``POST /api/runs`` reject things the engine accepts. A graph
    that cannot execute becomes a ``failed`` run with the engine's own error
    on the row, which is a far more useful artifact than a 400.
    """
    if not isinstance(raw, dict):
        raise RunSubmitError(
            "graph must be an object with 'nodes' and 'edges'")
    nodes = raw.get("nodes")
    # Absent means "none of these"; anything else is checked, never coerced.
    # ``raw.get(k) or []`` would quietly accept a malformed ``"edges": {}``
    # as an empty edge list because an empty dict is falsy.
    edges = [] if raw.get("edges") is None else raw["edges"]
    presets = [] if raw.get("presets") is None else raw["presets"]
    if not isinstance(nodes, list):
        raise RunSubmitError("graph.nodes must be a list")
    if not nodes:
        raise RunSubmitError("graph.nodes is empty")
    if not isinstance(edges, list):
        raise RunSubmitError("graph.edges must be a list")
    if not isinstance(presets, list):
        raise RunSubmitError("graph.presets must be a list")
    return {"nodes": nodes, "edges": edges, "presets": presets}


def json_size(value: Any) -> int:
    """Bytes this value would occupy as a stored JSON column.

    Matches ``run_store._dumps``' encoding (compact separators, the default
    ``ensure_ascii=True``), so the number is the real storage cost and not
    an estimate. ``ensure_ascii`` also means every character encodes to one
    byte, which is why ``len`` of the string is the byte count with no
    encode step. ``default=str`` keeps MEASURING from ever raising on
    something exotic — a sizer that throws would fail the run it was only
    trying to bound.
    """
    try:
        return len(json.dumps(value, separators=(",", ":"), default=str))
    except (TypeError, ValueError, RecursionError):  # pragma: no cover
        return 0


def _elide_entry(entry: Any, budget_bytes: int) -> Any:
    """Replace one over-budget output entry with a placeholder.

    ``output_kind`` and ``port`` survive so a UI can render "image, elided"
    in the right slot rather than a hole. The payload key (``image``,
    ``tensor_summary``, …) is DROPPED rather than stuffed with a marker
    object: a consumer that reads ``entry["text"]`` expecting a string must
    not silently receive a dict instead. ``"elided" in entry`` is the check.
    """
    if not isinstance(entry, dict):
        return entry
    size = json_size(entry)
    if size <= budget_bytes:
        return entry
    elided: dict[str, Any] = {"output_kind": entry.get("output_kind"),
                              "elided": True, "bytes": size,
                              "cap_bytes": budget_bytes}
    if "port" in entry:
        elided["port"] = entry["port"]
    return elided


#: Appended to a string that was cut short. ASCII by construction, so its
#: contribution to the encoded payload is exactly ``len`` bytes.
TRUNCATION_MARKER = "... [truncated]"


def _fit_prefix(value: str, budget_bytes: int) -> str:
    """The LONGEST prefix of *value* whose JSON encoding fits the budget.

    Binary search rather than ``value[:budget_bytes]`` because
    ``ensure_ascii=True`` (``run_store._dumps``' encoding, which
    ``json_size`` mirrors) makes one source character cost anywhere from 1
    to 12 bytes — a CJK traceback sliced by character count would still
    blow the cap and collapse to the marker this exists to avoid. The
    encoded size is monotonic in the prefix length, so the search is exact;
    it costs ~log2(len) measurements and only ever runs on the rare
    over-cap path.
    """
    low, high = 0, len(value)
    while low < high:
        mid = (low + high + 1) // 2
        if json_size(value[:mid]) <= budget_bytes:
            low = mid
        else:
            high = mid - 1
    return value[:low]


def _truncate_strings(payload: dict[str, Any], *, cap_bytes: int) -> dict[str, Any]:
    """Cut over-long top-level strings down to a head + marker.

    The bug this fixes: an ``execution_error`` carrying a 200 KB traceback
    has no ``outputs`` to elide, so it fell straight through to the marker
    and the user got a red banner with NO message — on the one payload
    where the text is the entire point. A first paragraph of a traceback is
    almost always enough to diagnose the failure; nothing is.

    Largest string first, re-measuring after each: the budget for one
    string is the cap minus everything else in the payload AS IT NOW
    STANDS, so a payload with several strings spends its bytes where they
    are actually needed instead of splitting them evenly between a 200 KB
    ``error`` and a 6-character ``status``.

    Two whole-payload measurements per string sounds expensive, and would
    be if the key count were open-ended. It is not: every payload reaching
    here is built by ``_progress_bridge`` or ``_finalize`` and has at most a
    handful of top-level keys (``node_id``, ``status``, ``error``,
    ``outputs``, ``reason``, ``run_id``). Nothing a client sends becomes a
    key here.
    """
    keys = sorted((k for k, v in payload.items() if isinstance(v, str)),
                  key=lambda k: len(payload[k]), reverse=True)
    out = dict(payload)
    for key in keys:
        if json_size(out) <= cap_bytes:
            break
        room = (cap_bytes - json_size({**out, key: ""})
                - len(TRUNCATION_MARKER))
        if room <= 0:
            continue
        kept = _fit_prefix(out[key], room)
        # Only when the result is genuinely SHORTER. With a tiny budget the
        # marker can outweigh what it replaces ("ok" -> "o... [truncated]"),
        # and growing a payload that is already over cap helps nobody — the
        # marker fallback is the right answer for that one.
        if len(kept) + len(TRUNCATION_MARKER) < len(out[key]):
            out[key] = kept + TRUNCATION_MARKER
    return out


def cap_event_payload(payload: Any, *, cap_bytes: int) -> Any:
    """Bound one event payload before it becomes a durable row.

    Retention counts RUNS, not bytes. Nothing else stops a graph whose nodes
    emit base64 PNGs (#117 image entries are the full image) from writing
    hundreds of megabytes of ``exec_run_events`` for a single run — with
    ``record_outputs`` off, because this path is the live status stream, not
    the output capture. Same for a training payload whose per-batch array
    grows every epoch.

    Four steps, cheapest first, each preserving more meaning than the next:

    1. Measure once. Under cap — the overwhelmingly common case, an ordinary
       ``node_status`` is a couple of KB — and the payload is returned
       untouched, having cost exactly one ``json.dumps``.
    2. Over cap, elide the output entries that are pulling more than their
       equal share of the budget. Equal share, rather than the whole cap,
       so a payload made of several medium entries degrades entry by entry
       instead of collapsing wholesale.
    3. Still over, and the weight is in a top-level string (an ``error``
       with a giant traceback): keep its head and mark it truncated.
    4. Still over — nothing left to trim — and the payload collapses to a
       marker that keeps the identifying keys. Losing the body beats
       refusing the event.

    ``cap_bytes <= 0`` disables the whole thing.
    """
    if cap_bytes <= 0 or payload is None:
        return payload
    size = json_size(payload)
    if size <= cap_bytes:
        return payload

    if isinstance(payload, dict):
        entries = payload.get("outputs")
        if isinstance(entries, list) and entries:
            budget = max(1, cap_bytes // len(entries))
            payload = {**payload,
                       "outputs": [_elide_entry(e, budget) for e in entries]}
            size = json_size(payload)
            if size <= cap_bytes:
                return payload

        payload = _truncate_strings(payload, cap_bytes=cap_bytes)
        size = json_size(payload)
        if size <= cap_bytes:
            return payload

    marker: dict[str, Any] = {"elided": True, "bytes": size,
                              "cap_bytes": cap_bytes}
    if isinstance(payload, dict):
        for key in ("node_id", "status", "run_id", "reason"):
            if key in payload:
                marker[key] = payload[key]
    return marker


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
    says what it means; see ``RunService._collect_metrics``, which stops
    inferring from a node the moment that node speaks for itself.
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


def apply_seed(seed: int | None) -> None:
    """Seed the RNGs a run might touch, before the graph starts.

    Thin alias for :func:`app.core.seeding.seed_rngs`, kept because it is
    this module's published name for the operation. It sets the RUN-level
    baseline: anything that draws outside a node's ``execute`` (preset
    expansion, a validator) is covered by it.

    It is no longer what makes a run reproducible. That is the per-node
    derivation in ``graph_engine`` — this call alone leaves the answer
    dependent on the order in which concurrently-scheduled nodes happened to
    draw, which is exactly the bug #134 fixed. Still PROCESS-GLOBAL, so two
    runs with different seeds executing at once do not get independent
    streams; the per-node seeding narrows that window to one node's execute
    but does not close it.
    """
    seed_rngs(seed)


# ── run exclusion ─────────────────────────────────────────────────────────


class _RunExclusion:
    """A seeded run does not overlap another run. Readers-writer, in-process.

    Per-node seeding writes the PROCESS-GLOBAL RNGs, so ``execute_graph``
    already serialises the nodes WITHIN a seeded run. That is only half the
    problem: the cpu queue admits two runs at once by default, and a second
    run drawing from the same global generators between (or during) the
    first run's nodes moves its numbers just as surely. Measured before this
    gate existed: 8/8 node values diverged between a solo and a concurrent
    run at the same seed.

    So the promise the docs make -- same seed, same loss curve -- is made
    TRUE here rather than narrowed in prose: a seeded run waits for the runs
    already going, then runs alone, and the runs behind it wait for it.
    Unseeded runs are unaffected and still overlap each other freely, which
    is what keeps this from being a global serialisation of the server.

    "Alone" means alone in the PROCESS, so every path that executes a graph
    has to take it -- ``RunService._drive`` below and
    ``routes_graph_run``'s headless invoke, which has no RunService at all.
    That is why the gate is reached through :func:`run_exclusion` rather
    than owned by a service instance.

    WRITER-PREFERRING on purpose. A plain readers-writer lock lets a steady
    trickle of unseeded submissions starve a seeded run forever, and "your
    reproducible run never starts" is a worse failure than "your throughput
    run waited". New shared holders therefore queue behind a waiting
    exclusive one.

    The cost is real and worth stating: a seeded canvas run can now wait
    behind a long queued job, even though the interactive lane otherwise
    bypasses the FIFO. Reproducibility is opt-in; a run that asked for it
    would rather be late than wrong.

    In-process only, like every other lock here. Two servers pointed at one
    database were never isolated from each other and are no worse now.
    """

    __slots__ = ("_condition", "_readers", "_writer", "_waiting_writers",
                 "_wakes")

    def __init__(self) -> None:
        # Built lazily-bound: asyncio primitives take the running loop on
        # first await, and a RunService is constructed before the loop in
        # some tests.
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        # Strong references to in-flight wake-ups; asyncio keeps only weak
        # ones, so a task nobody holds can be collected before it runs.
        self._wakes: set[asyncio.Task] = set()

    @property
    def busy(self) -> bool:
        """True when anything holds the gate. For assertions and logging."""
        return self._writer or self._readers > 0

    def for_seed(self, seed: int | None):
        """The right hold for a run: exclusive when seeded, shared otherwise."""
        return self.exclusive() if seed is not None else self.shared()

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and self._waiting_writers == 0)
            self._readers += 1
        try:
            yield
        finally:
            self._readers -= 1
            if self._readers == 0:
                await self._wake()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        async with self._condition:
            # Counted BEFORE waiting, so new readers see a writer pending and
            # queue behind it. Decremented in a ``finally`` because a
            # cancelled wait (shutdown) must not leave the count raised —
            # that would block every reader for the life of the process.
            self._waiting_writers += 1
            try:
                if self._writer or self._readers:
                    logger.info(
                        "seeded run waiting for %d run(s) to finish; a seeded "
                        "run executes alone so its numbers are reproducible",
                        self._readers + (1 if self._writer else 0))
                await self._condition.wait_for(
                    lambda: not self._writer and self._readers == 0)
            finally:
                self._waiting_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            self._writer = False
            await self._wake()

    async def _wake(self) -> None:
        """Let the waiters re-test the gate. NOT skippable by a cancel.

        The counters above are given back with a plain assignment, which has
        no suspension point and therefore no cancellation point: whatever
        happens here, the gate is already free. Only the notification needs
        the condition lock, and awaiting a CONTENDED lock inside a
        ``finally`` IS a cancellation point — a second ``Task.cancel``
        landing there used to escape with ``_writer`` still True, closing
        the gate for every run in the process (#188 re-review, D3). One
        cancel was survivable because ``Lock.acquire`` has a non-awaiting
        fast path; two were not.

        ``shield`` is what fixes it: the notification is its own task, so
        the cancel takes this await and leaves the task running. The
        CancelledError still propagates, as it must.
        """
        try:
            notify = asyncio.ensure_future(self._notify_all())
        except RuntimeError:  # pragma: no cover - loop already gone
            # Nothing can be waiting without a loop to wait on, and the
            # counters are already back. Never let a teardown-time failure
            # here replace the exception on its way out.
            return
        self._wakes.add(notify)
        notify.add_done_callback(self._wakes.discard)
        notify.add_done_callback(_swallow_task_result)
        await asyncio.shield(notify)

    async def _notify_all(self) -> None:
        async with self._condition:
            self._condition.notify_all()


def _swallow_task_result(task: asyncio.Task) -> None:
    """Retrieve a finished task's outcome so asyncio does not log it."""
    if not task.cancelled():
        task.exception()


#: One gate per event loop, which in production means one per server
#: process. NOT one per ``RunService``: the thing it protects is the
#: process-global RNG state, and ``POST /api/graph/run/{name}`` executes
#: graphs without a RunService at all (#188 re-review, D2) — a per-instance
#: gate left that path free to move a seeded run's numbers, measured at 8/8
#: node values diverged. Keyed by loop rather than a module singleton
#: because ``asyncio.Condition`` binds to the first loop that awaits it and
#: every test gets a fresh loop.
_EXCLUSIONS: "weakref.WeakKeyDictionary[Any, _RunExclusion]" = (
    weakref.WeakKeyDictionary())


def run_exclusion() -> _RunExclusion:
    """The reproducibility gate for the running loop. Never None."""
    loop = asyncio.get_running_loop()
    gate = _EXCLUSIONS.get(loop)
    if gate is None:
        gate = _RunExclusion()
        _EXCLUSIONS[loop] = gate
    return gate


# ── in-process fan-out ────────────────────────────────────────────────────


class _Broadcast:
    """Edge-triggered wake-up for long pollers. No busy waiting anywhere.

    ``asyncio.Event`` is level-triggered and single-shot; a shared one would
    have to be cleared, and whoever clears it races everyone who has not
    woken yet. Instead each ``notify`` SETS the current event and installs a
    fresh one for the next generation. A waiter captures the event object
    BEFORE it reads the store, so a notification that lands during that read
    still wakes it — the classic lost-wakeup window is closed by ordering,
    not by a timeout.

    Not an ``asyncio.Condition``: a Condition makes every waiter contend for
    one lock just to be told to go read sqlite, and its ``wait()`` reacquires
    that lock before returning, serialising the wake-ups.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def waiter(self) -> asyncio.Event:
        """The generation to wait on. Capture it before checking the store."""
        return self._event

    def notify(self) -> None:
        event, self._event = self._event, asyncio.Event()
        event.set()


# eq=False: a subscription is a live handle, not a value. Identity equality
# (and therefore identity hashing) is what lets it live in the per-run
# subscriber set; the dataclass default would generate __eq__ and set
# __hash__ to None, making it unhashable.
@dataclass(eq=False)
class RunSubscription:
    """A live feed of one run's events, for in-process push consumers (#121).

    Drop-tolerant BY DESIGN. The queue is bounded and a full queue drops the
    new event rather than applying backpressure — a stalled consumer must
    never be able to slow down or deadlock the run that is feeding it. The
    durable log is the source of truth: a consumer that sees ``take_dropped()
    > 0`` re-reads the tail from ``RunStore.get_events`` starting at its last
    cursor and loses nothing. ``None`` on the queue is the end-of-run
    sentinel; ``closed`` says the same thing for a consumer that never drains.
    """

    run_id: str
    queue: "asyncio.Queue[EventRecord | None]"
    closed: bool = False
    _dropped: int = 0

    def offer(self, event: "EventRecord | None") -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1

    def take_dropped(self) -> int:
        """How many events were dropped since the last call; resets to 0."""
        dropped, self._dropped = self._dropped, 0
        return dropped


@dataclass(eq=False)
class _QueuedRun:
    """A submitted run waiting for its device. The FIFO's element (#123).

    Deliberately tiny. It holds what SCHEDULING needs — which queue to join,
    which options to start with — and not the graph, which is re-read from
    ``exec_runs.graph_snapshot`` at promotion so a deep queue costs rows and
    not RAM.

    ``signal`` and ``subscribers`` are the run's two wake-up channels, and
    both OUTLIVE the wait: the same objects are handed to the ``_ActiveRun``
    at promotion. That is what lets a client attach to a run that has not
    started yet — it subscribes here, waits, and the queue it is holding is
    the same queue the started run pushes into, with no re-subscribe and no
    window between the two. Without the handover, attaching to a queued run
    would return an immediately-closed feed and the client would sit on a
    stream that never delivers anything.
    """

    run_id: str
    queue_key: str
    options: dict[str, Any]
    signal: _Broadcast
    subscribers: set[RunSubscription] = field(default_factory=set)

    def close(self) -> None:
        """End every subscription and wake every poller. Sync — never fails.

        The waiting counterpart of ``_ActiveRun.close``, for a run that is
        retired before it ever starts. There is no ``finished`` flag to set:
        a pending entry is only ever closed on its way out of both indexes,
        so "gone from the queue" is the same fact.
        """
        for subscription in list(self.subscribers):
            subscription.offer(None)
            subscription.closed = True
        self.subscribers.clear()
        self.signal.notify()


@dataclass(eq=False)
class _ActiveRun:
    """Everything the service holds for one in-flight run."""

    run_id: str
    context: ExecutionContext
    signal: _Broadcast
    #: Which FIFO's slot this run occupies (the resolved device), and which
    #: lane it came in on. Held so the run can RELEASE what it took when it
    #: ends — the counters cannot be re-derived from a row that is by then
    #: already terminal.
    queue_key: str = CPU_QUEUE_KEY
    lane: str = DEFAULT_LANE
    session: "InteractiveSession | None" = None
    record_outputs: bool = False
    #: True once this run's admission — its device slot, or its interactive
    #: slot and session key — has been given back. Released EARLY (see
    #: ``_release_admission``), so the flag is what keeps ``_drive``'s
    #: backstop from releasing a second time.
    admission_released: bool = False
    task: "asyncio.Task[None] | None" = None
    subscribers: set[RunSubscription] = field(default_factory=set)
    #: "cancelled" (a user asked) or "interrupted" (the server is going
    #: away). Set BEFORE the context flag so the unwinding task can tell the
    #: two apart — they are different facts about why a run stopped.
    stop_reason: str | None = None
    #: True once the OUTCOME is decided — set on entry to ``_finalize``,
    #: before any of its awaits. A cancel arriving after this cannot change
    #: anything, so it must not claim it did.
    terminating: bool = False
    #: True once the terminal event is durable. Long pollers stop waiting on
    #: this, not on registry removal, which happens a few statements later.
    #: Strictly after ``terminating``: between the two the closing event is
    #: still being written, and a poller should keep waiting for it.
    finished: bool = False
    pending_metrics: list[MetricPoint] = field(default_factory=list)
    last_metric_step: int = 0
    last_flush_monotonic: float = field(default_factory=time.monotonic)
    #: Nodes that have logged an EXPLICIT metric (#122). Progress payloads
    #: from these are no longer mined by ``scalar_metrics`` — otherwise a
    #: training loop that logs ``val_loss`` and also puts ``val_loss`` in its
    #: epoch payload would write the same point twice.
    explicit_metric_nodes: set[str] = field(default_factory=set)

    def close(self) -> None:
        """End every subscription and wake every poller. Sync — never fails."""
        self.finished = True
        for subscription in list(self.subscribers):
            subscription.offer(None)
            subscription.closed = True
        self.subscribers.clear()
        self.signal.notify()


# ── the service ───────────────────────────────────────────────────────────


class RunService:
    """Owns every server-side run: registry, tasks, events, lifecycle.

    One instance per process, held on ``app.state.run_service``. Everything
    it touches concurrently lives on the event loop (plain dict/set mutation
    with no ``await`` in between), so it needs no lock of its own; the only
    shared resource with real contention is the database, and ``Database.run``
    already serialises that.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        output_store: RunOutputStore | None = None,
        retention_keep_last: int | None = None,
        event_payload_cap_bytes: int | None = None,
        shutdown_grace_s: float = DEFAULT_SHUTDOWN_GRACE_S,
        subscriber_queue_size: int = DEFAULT_SUBSCRIBER_QUEUE_SIZE,
        metric_flush_max_points: int = METRIC_FLUSH_MAX_POINTS,
        metric_flush_interval_s: float = METRIC_FLUSH_INTERVAL_S,
        limits: QueueLimits | None = None,
    ) -> None:
        self.store = store
        self._output_store = output_store
        self._retention_keep_last = retention_keep_last
        # ONE source for the ceiling, read HERE rather than defaulted to a
        # module constant: #120 shipped two independent copies of
        # ``128 * 1024`` (this module's and the setting's) that nothing kept
        # in step, so raising ``CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES`` moved
        # only the lifespan-constructed service and left every other caller
        # — tests, scripts, a future CLI — on the stale number. A default
        # argument would not fix it either: it is evaluated at import time,
        # before anything can override the setting.
        self._event_payload_cap_bytes = (
            settings.RUN_EVENT_PAYLOAD_CAP_BYTES
            if event_payload_cap_bytes is None else event_payload_cap_bytes)
        self._shutdown_grace_s = shutdown_grace_s
        self._subscriber_queue_size = subscriber_queue_size
        self._metric_flush_max_points = metric_flush_max_points
        self._metric_flush_interval_s = metric_flush_interval_s
        self._limits = QueueLimits.from_settings() if limits is None else limits
        self._runs: dict[str, _ActiveRun] = {}
        # ── the queue (#123) ─────────────────────────────────────────────
        # One FIFO per resolved device, plus a by-id index so cancel is O(1)
        # instead of a scan of every queue. The two are written together and
        # only ever from the event loop; ``_enqueue``/``_dequeue`` are the
        # only places that touch both, so they cannot drift.
        self._pending: dict[str, deque[_QueuedRun]] = {}
        self._pending_by_id: dict[str, _QueuedRun] = {}
        # Slots taken per queue key. NOT derived from ``_runs`` on demand:
        # counting a dict by predicate on every promotion is both slower and
        # easier to get subtly wrong than incrementing where the slot is
        # actually taken and released.
        self._slots: dict[str, int] = {}
        # No reproducibility gate here on purpose: ``_drive`` reaches it
        # through ``run_exclusion()``. It is orthogonal to the queue's
        # admission (the queue decides how many runs may be IN FLIGHT, the
        # gate decides which of them may be EXECUTING at the same moment)
        # and it guards process-global RNG state, which the headless invoke
        # route touches without ever constructing a RunService.
        # The interactive lane's two bounds. ``_interactive_inflight``
        # includes runs whose row is still being written (submit reserves
        # BEFORE it awaits ``create_run``), which is what stops two
        # simultaneous canvas clicks from both passing a cap of one.
        self._interactive_inflight = 0
        # ``session_key(session) -> run_id``. Keyed by the CACHE, not by the
        # session wrapper; ``_admit_interactive`` explains why. The cache
        # object is the dict key, so holding it here also keeps it alive for
        # as long as the entry exists.
        self._interactive_sessions: dict[Any, str] = {}
        self._shutting_down = False

    # ── introspection ─────────────────────────────────────────────────────

    def active_run_ids(self) -> list[str]:
        """Runs THIS process is driving, submit order."""
        return list(self._runs)

    def is_active(self, run_id: str) -> bool:
        return run_id in self._runs

    def execution_id(self, run_id: str) -> str | None:
        """The ``ExecutionContext.execution_id`` of a live run (== its id)."""
        active = self._runs.get(run_id)
        return None if active is None else active.context.execution_id

    def is_queued(self, run_id: str) -> bool:
        """True while the run is waiting for a device slot (#123)."""
        return run_id in self._pending_by_id

    def queue_position(self, run_id: str) -> int | None:
        """1-based place in this run's device FIFO; ``None`` if not waiting.

        Position 1 is next to start. A run that is running, finished, or
        unknown to this process has no position — the answer is ``None`` for
        all three, because "not waiting" is the fact a caller acts on.
        """
        entry = self._pending_by_id.get(run_id)
        if entry is None:
            return None
        queue = self._pending.get(entry.queue_key)
        if queue is None:  # pragma: no cover - the two indexes move together
            return None
        for index, waiting in enumerate(queue, start=1):
            if waiting.run_id == run_id:
                return index
        return None  # pragma: no cover - as above

    def queue_snapshot(self) -> dict[str, list[str]]:
        """``{queue_key: [run_id, ...]}`` in service order. Waiting runs only.

        The scheduler's own view, for tests and diagnostics. Running runs are
        NOT in it — they no longer occupy a place in line, they occupy a
        slot.
        """
        return {key: [entry.run_id for entry in queue]
                for key, queue in self._pending.items() if queue}

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(
        self,
        graph: Any,
        *,
        options: Any = None,
        name: str | None = None,
        session: InteractiveSession | None = None,
    ) -> SubmitResult:
        """Persist a run, then SCHEDULE it. Raises ``RunSubmitError`` on input.

        The two halves are separate on purpose. ``create_run`` makes the run
        exist and queryable before anything can execute, so a crash between
        the two leaves a row that startup recovery retires honestly instead
        of losing the submission entirely.

        Scheduling is what happens between them, and it depends on the lane
        (see the module's Scheduling section). A ``queued``-lane run joins
        its device's FIFO and starts when that device has room — so the
        returned ``status`` is genuinely ``queued`` when it did not, which is
        why callers were told from #120 to read this field rather than assume
        ``running``. An ``interactive`` run bypasses the FIFO and is admitted
        against its own two limits, or refused.

        The device is resolved ONCE here and handed to both the row
        (``queue_key``) and the run (``ExecutionContext.device``), so the
        queue a run waits in and the device it eventually uses cannot
        disagree.

        *session* is the canvas's process-local state; see
        :class:`InteractiveSession`. Passing one on any other lane is a
        submit error rather than a silent no-op: it would mean a run that
        behaves statefully while its stored row says ``queued``.
        """
        if self._shutting_down:
            raise RunServiceUnavailable("run service is shutting down")
        normalized_graph = normalize_graph(graph)
        normalized_options = normalize_options(options)
        normalized_name = normalize_name(name)
        lane = normalized_options["lane"]
        if session is not None and lane != LANE_INTERACTIVE:
            raise RunSubmitError(
                f"an interactive session requires lane={LANE_INTERACTIVE!r}, "
                f"got {lane!r}")
        queue_key = canonical_queue_key(
            resolve_device(normalized_options.get("device")))

        if lane == LANE_INTERACTIVE:
            return await self._submit_interactive(
                normalized_graph, normalized_options, normalized_name,
                queue_key, session)

        record = await self.store.create_run(
            graph_snapshot=normalized_graph,
            options=normalized_options,
            name=normalized_name,
            status=STATUS_QUEUED,
            queue_key=queue_key,
        )
        if self._shutting_down:
            # The same re-check ``_start`` makes, for the same reason and at
            # the same seam: ``create_run`` awaits (provenance capture can
            # shell out to git), and ``shutdown`` retires the queue at any
            # await. A submit parked in that window would otherwise join a
            # queue that has already been drained and wait for a pump that
            # will never run again.
            raise RunServiceUnavailable(SHUTDOWN_AFTER_PERSIST)
        # Enqueue THEN pump, always — even when the device is idle and the
        # run will start on the very next statement. A fast path that started
        # an empty-queue submit directly would be a second admission rule to
        # keep in step with this one, and the first bug it produced would be
        # a run overtaking a queue that filled between the check and the
        # start.
        entry = _QueuedRun(run_id=record.id, queue_key=queue_key,
                           options=normalized_options, signal=_Broadcast())
        self._enqueue(entry)
        self._pump(queue_key)
        return SubmitResult(
            run_id=record.id,
            status=(STATUS_RUNNING if record.id in self._runs
                    else STATUS_QUEUED))

    async def _submit_interactive(
        self, graph: dict[str, Any], options: dict[str, Any],
        name: str | None, queue_key: str,
        session: InteractiveSession | None,
    ) -> SubmitResult:
        """Admit a canvas run past the FIFO, or refuse it. Never queues.

        Admission is reserved BEFORE ``create_run`` awaits and released on
        every failure path after it. Checking after the write would leave the
        window that makes a cap of one meaningless: two clicks arriving
        together would both pass the check while neither had a row yet.

        A refusal here must not leave a row behind, which is the other reason
        the check comes first — a ``queued`` row nothing will ever schedule
        is exactly the orphan startup recovery exists to clean up, and
        manufacturing them on a busy canvas is not a good use of it.
        """
        self._admit_interactive(session)
        try:
            record = await self.store.create_run(
                graph_snapshot=graph, options=options, name=name,
                status=STATUS_QUEUED, queue_key=queue_key,
            )
            return self._start(record.id, graph, options,
                               queue_key=queue_key, lane=LANE_INTERACTIVE,
                               session=session)
        except BaseException:
            self._release_interactive(session)
            raise

    @staticmethod
    def _session_key(session: InteractiveSession | None) -> Any:
        """What two concurrent runs must not share: the socket's cache.

        NOT the :class:`InteractiveSession` itself, and that distinction is
        the whole correctness of the rule. ``ws_execution.handle_execute``
        builds a FRESH session wrapper on every ``execute`` message (it has
        to — ``changed_nodes`` differs per click) around the ONE
        ``ExecutionCache`` it created when the socket opened. Keying on the
        wrapper would therefore compare two objects that are never the same
        one, and the rule would silently never fire; keying on the cache
        keys on the thing that is genuinely per-socket and genuinely shared.

        ``None`` when there is no cache: there is then no cross-run state to
        corrupt, and only the concurrency cap applies. (The module store is
        deliberately not part of the key — it is process-global, so it would
        serialise every canvas in the process against every other.)
        """
        return None if session is None else session.cache

    def _admit_interactive(self, session: InteractiveSession | None) -> None:
        """Reserve one interactive slot, or raise ``RunServiceUnavailable``.

        Two independent bounds, reported separately because the fixes are
        different: "wait for another tab to finish" versus "this tab already
        has a run going".

        The per-session rule is the structural close of #121's disclosed
        gap. Two concurrent runs on one canvas connection share its
        ``ExecutionCache`` — one would serve the other's half-built tensors
        under a matching key — and, with ``weights_persistent``, mutate the
        same ``nn.Module``. The frontend prevents it by disabling Run; that
        is a convention, and conventions do not hold for a reconnecting tab,
        a second client speaking the protocol, or a bug.
        """
        cap = max(1, self._limits.interactive)
        if self._interactive_inflight >= cap:
            raise RunServiceUnavailable(
                f"{self._interactive_inflight} interactive run(s) already in "
                f"flight (limit {cap}); wait for one to finish or raise "
                "CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT")
        key = self._session_key(session)
        if key is not None and key in self._interactive_sessions:
            busy = self._interactive_sessions[key]
            raise RunServiceUnavailable(
                "this editor session already has a run in flight"
                + (f" ({busy})" if busy else "")
                + "; two runs cannot share one session's cache and weights")
        self._interactive_inflight += 1
        if key is not None:
            # Empty string until ``_start`` knows the id: the KEY is what
            # holds the slot, and it has to be registered before the awaits
            # in ``create_run``.
            self._interactive_sessions[key] = ""

    def _release_interactive(self, session: InteractiveSession | None) -> None:
        """Give back one interactive slot. Idempotent per run, never negative."""
        self._interactive_inflight = max(0, self._interactive_inflight - 1)
        key = self._session_key(session)
        if key is not None:
            self._interactive_sessions.pop(key, None)

    # ── the queue ─────────────────────────────────────────────────────────

    def _enqueue(self, entry: _QueuedRun) -> None:
        """Put a run at the back of its device's line. SYNCHRONOUS."""
        self._pending.setdefault(entry.queue_key, deque()).append(entry)
        self._pending_by_id[entry.run_id] = entry

    def _dequeue(self, run_id: str) -> _QueuedRun | None:
        """Take a run out of the line wherever it is. ``None`` if not waiting.

        O(n) in the length of ONE device's queue, and only ever called by
        cancel and by shutdown. A run that reaches the front is removed by
        ``_pump``'s ``popleft`` instead, which is the hot path.
        """
        entry = self._pending_by_id.pop(run_id, None)
        if entry is None:
            return None
        queue = self._pending.get(entry.queue_key)
        if queue is not None:
            try:
                queue.remove(entry)
            except ValueError:  # pragma: no cover - indexes move together
                pass
            if not queue:
                self._pending.pop(entry.queue_key, None)
        return entry

    def _pump(self, queue_key: str) -> None:
        """Start as many waiting runs as this device has room for. SYNCHRONOUS.

        Called on submit and again whenever a run of this key ends, so there
        is no scheduler task, no polling interval, and nothing to drain at
        shutdown. Being synchronous is what makes it safe to call from
        ``_drive``'s ``finally``: the slot is released and the next run
        registered without an await in between, so no third party can slip
        into a slot that has already been promised.

        Refusing to start anything during shutdown is deliberate — the
        pending runs are being retired by ``shutdown`` at that moment, and
        promoting one into a task nobody drains is the exact failure
        ``_start``'s own re-check exists to prevent.
        """
        if self._shutting_down:
            return
        queue = self._pending.get(queue_key)
        while queue and self._slots.get(queue_key, 0) < self._limits.for_key(
                queue_key):
            entry = queue.popleft()
            self._pending_by_id.pop(entry.run_id, None)
            try:
                # The graph is NOT passed: ``_drive`` re-reads it from the
                # row. The signal and the subscriber set ARE — same objects,
                # so a client that parked on the queued run keeps both its
                # generation and its live feed across the promotion.
                self._start(entry.run_id, None, entry.options,
                            queue_key=queue_key, lane=entry.options.get(
                                "lane", DEFAULT_LANE),
                            signal=entry.signal, subscribers=entry.subscribers)
            except Exception:
                # Two things must not happen here, and both did before this
                # guard. The entry must not be LOST — it has already left
                # both indexes — so it goes back at the FRONT, keeping its
                # place in line for the next pump. And the exception must
                # not escape: this method is called from ``_finalize``,
                # inside the PREVIOUS run's task and before that run's
                # terminal event and row write, so letting it through would
                # make one run's failed promotion silently strand another
                # run's outcome.
                logger.exception(
                    "run %s could not be started off the %s queue; it stays "
                    "queued", entry.run_id, queue_key)
                queue.appendleft(entry)
                self._pending_by_id[entry.run_id] = entry
                break
        if not queue:
            self._pending.pop(queue_key, None)

    def _start(
        self, run_id: str, graph: dict[str, Any] | None,
        options: dict[str, Any], *, queue_key: str,
        lane: str = DEFAULT_LANE,
        session: InteractiveSession | None = None,
        signal: _Broadcast | None = None,
        subscribers: set[RunSubscription] | None = None,
    ) -> SubmitResult:
        """Register the run and hand it to its own task. SYNCHRONOUS.

        Not a coroutine, and that is the load-bearing property: there is no
        ``await`` anywhere between putting the run in the registry and
        creating the task that owns it, so no cancellation of the SUBMITTING
        coroutine can land in the middle and strand a registry entry that
        nothing will ever finish. Everything that needs the database
        (reading the graph back, ``mark_running``, the start event) happens
        inside the task, where a cancellation is the RUN's cancellation and
        unwinds through the normal terminal path.

        It is also what makes promotion out of the FIFO safe: a slot is taken
        and its run registered in one uninterruptible step, so a cancel can
        only ever find the run in exactly one of the two places that know how
        to stop it.

        The reported status is therefore a promise the task fulfils on its
        first step; a client that reads the row in that microsecond sees
        ``queued``, which is simply true.

        *graph* may be ``None``, meaning "read it from the row" — the shape
        every promoted run takes, since a deferred start cannot rely on the
        submitting request still being in memory.

        The shutdown re-check is not redundant with ``submit``'s. That one
        happens before ``create_run`` awaits (provenance capture can shell
        out to git), and ``shutdown`` snapshots the registry at any await —
        so a submit parked in that window would resume here and register a
        task nobody drains, with ``main.py`` closing the database underneath
        it. Refusing here leaves the durable ``queued`` row for startup
        recovery, exactly like a cancelled submit.
        """
        if self._shutting_down:
            raise RunServiceUnavailable(SHUTDOWN_AFTER_PERSIST)
        context = ExecutionContext(
            execution_id=run_id,
            device=queue_key,
            max_workers=settings.MAX_PARALLEL_NODES,
            verbose=bool(options.get("verbose")),
            graph_id=str(options.get("graph_id") or ""),
            backward_mode=bool(options.get("backward_mode")),
            auto_backward=bool(options.get("auto_backward")),
            # Reproducibility (#134). The seed reaching the CONTEXT — not
            # just ``apply_seed`` before the graph starts — is what lets the
            # engine derive a per-node seed and lets a node build its own
            # generator, which is the difference between "seeded" and
            # "actually reproducible".
            seed=options.get("seed"),
            deterministic=bool(options.get("deterministic")),
            # A server-owned run is an isolated, reproducible unit by
            # default: its stored graph_snapshot + options must fully
            # describe what ran, and inheriting live nn.Module weights from
            # a previous run would make that description a lie. The canvas
            # opts out by handing over the module store in its
            # InteractiveSession — a capability, not a flag, so a run
            # nobody lent one to is isolated whatever its options say.
            weights_persistent=bool(options.get("weights_persistent")),
            node_state_store=(None if session is None
                              else session.node_state_store),
        )
        active = _ActiveRun(
            run_id=run_id, context=context,
            signal=_Broadcast() if signal is None else signal,
            queue_key=queue_key, lane=lane, session=session,
            record_outputs=bool(options.get("record_outputs")),
            subscribers=set() if subscribers is None else subscribers,
        )
        # The task is created BEFORE anything is mutated, so this method is
        # all-or-nothing: if scheduling the task fails (a closed loop during
        # teardown), no slot has been taken and no half-registered run is
        # left for ``_pump``'s recovery to reason about. Nothing runs in
        # between — ``create_task`` only schedules, and this function never
        # awaits — so ``_drive`` cannot observe the registry before the next
        # three statements have landed.
        task = asyncio.create_task(
            self._drive(active, graph, options, session), name=f"run:{run_id}")
        if lane != LANE_INTERACTIVE:
            # Taken HERE, not in ``_pump``, so the one path that starts a run
            # is the one path that accounts for it. The interactive lane's
            # slot was taken in ``_admit_interactive`` before its row existed.
            self._slots[queue_key] = self._slots.get(queue_key, 0) + 1
        elif self._session_key(session) is not None:
            self._interactive_sessions[self._session_key(session)] = run_id
        active.task = task
        self._runs[run_id] = active
        return SubmitResult(run_id=run_id, status=STATUS_RUNNING)

    def _release_admission(self, active: _ActiveRun) -> None:
        """Give back what a run held, then serve the next in line. Idempotent.

        Called as soon as the GRAPH is done — the first line of
        ``_finalize``, before the terminal event — and again from
        ``_drive``'s ``finally`` as the backstop for the paths that never
        reach ``_finalize`` (a hard ``Task.cancel``, a vanished row).

        Releasing early is not an optimisation, it is a correctness fix that
        a live canvas found: a client re-enables its Run button the moment
        ``execution_complete`` arrives, so releasing after that event left a
        window in which the user had been told the run was over and the very
        next click was still refused by the one-run-per-session rule. A fast
        double-click lands squarely in it. By ``_finalize`` the engine has
        returned, so the device is idle and nothing will touch the session's
        cache or weights again — everything left is bookkeeping.

        The pump call lives here rather than at ``_drive``'s cleanup because
        "a slot freed" and "the queue may move" are the same event, and
        separating them is how a queue silently stalls.
        """
        if active.admission_released:
            return
        active.admission_released = True
        if active.lane == LANE_INTERACTIVE:
            self._release_interactive(active.session)
            return
        remaining = self._slots.get(active.queue_key, 0) - 1
        if remaining > 0:
            self._slots[active.queue_key] = remaining
        else:
            self._slots.pop(active.queue_key, None)
        self._pump(active.queue_key)

    # ── the run task ──────────────────────────────────────────────────────

    async def _drive(
        self, active: _ActiveRun, graph: dict[str, Any] | None,
        options: dict[str, Any], session: InteractiveSession | None = None,
    ) -> None:
        """The server-owned task. Nothing outside this module awaits it.

        A hard ``Task.cancel`` (shutdown's last resort) is BaseException, so
        it skips every ``except`` below and with them ``_finalize`` — the
        row stays ``running`` for the next boot's recovery to retire, which
        is deliberate: a forcibly-killed run must not write a tidy terminal
        status it cannot vouch for. The ``finally`` still runs (that is what
        closes subscriptions, deregisters the run and gives its slot back);
        only the terminal bookkeeping is skipped. Everything else —
        including a broken database — is caught, because an unretrieved task
        exception is a silent death nobody would ever see.

        *graph* is ``None`` for a run promoted out of the queue: the first
        thing the task does is read the snapshot back off its own row.
        Reading it HERE rather than in ``_pump`` is what keeps promotion
        synchronous, and therefore keeps a cancel from finding the run in
        neither the queue nor the registry.
        """
        run_id = active.run_id
        try:
            if graph is None:
                graph = await self._load_graph(active)
                if graph is None:
                    return
            # A cancel that arrived between promotion and here (the queue
            # front is the likeliest place for a user to catch a run) stops
            # it before the first node rather than after it: the engine's
            # earliest checkpoint is inside the level loop, and "cancelled"
            # should never mean "ran one node first".
            if active.context.should_stop():
                await self._finalize(active, self._stopped_status(active), None)
                return
            # Taken BEFORE ``_begin`` so a seeded run waiting its turn is
            # still reported as `queued` rather than as a `running` run
            # emitting nothing — the status stays true while it waits.
            async with run_exclusion().for_seed(options.get("seed")):
                if not await self._begin(active):
                    return
                status, error = await self._execute(active, graph, options,
                                                    session)
            await self._finalize(active, status, error)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("run %s: unhandled error in the run task", run_id)
        finally:
            active.close()
            self._runs.pop(run_id, None)
            self._release_admission(active)

    @staticmethod
    def _stopped_status(active: _ActiveRun) -> str:
        """``interrupted`` when the server asked, ``cancelled`` when a user did."""
        return (STATUS_INTERRUPTED
                if active.stop_reason == STOP_REASON_INTERRUPTED
                else STATUS_CANCELLED)

    async def _load_graph(self, active: _ActiveRun) -> dict[str, Any] | None:
        """Read a promoted run's graph back off its row. ``None`` means stop.

        Two ways it can fail, and they deserve different answers:

        - the row is GONE (deleted while the run waited) — there is nothing
          left to write a status or an event to, so this is the same silent
          give-up ``_begin`` makes for the same reason.
        - the snapshot is there but unusable. That is a failed run, not a
          hung one: filing it as ``failed`` with the reason beats leaving a
          row claiming to be running until the next restart.
        """
        snapshot = await self.store.get_graph_snapshot(active.run_id)
        if snapshot is None:
            logger.warning("run %s vanished while it was queued",
                           active.run_id)
            return None
        if not isinstance(snapshot, dict) or not snapshot.get("nodes"):
            await self._finalize(
                active, STATUS_FAILED,
                "the stored graph snapshot is empty or malformed")
            return None
        snapshot.setdefault("edges", [])
        snapshot.setdefault("presets", [])
        return snapshot

    async def _begin(self, active: _ActiveRun) -> bool:
        """Claim the row and announce the run. False means "do not run".

        The only way this fails is the row not existing any more — deleted
        out from under a submission — in which case there is nothing left to
        write events or a status to.
        """
        started = await self.store.mark_running(
            active.run_id, queue_key=active.context.device)
        if not started:
            logger.warning("run %s vanished before it could start",
                           active.run_id)
            return False
        await self._emit(active.run_id, EVENT_RUN_STARTED,
                         {"run_id": active.run_id})
        return True

    async def _execute(
        self, active: _ActiveRun, graph: dict[str, Any],
        options: dict[str, Any], session: InteractiveSession | None = None,
    ) -> tuple[str, str | None]:
        """Run the graph and classify the outcome. Never raises but cancel."""
        try:
            apply_seed(options.get("seed"))
            # Determinism is NOT applied here any more: ``execute_graph``
            # owns it through ``deterministic_scope``, which also takes it
            # back when the run ends. Setting it here would latch the
            # process outside that scope — the one-way latch the #188
            # review found.
            await execute_graph(
                graph["nodes"],
                graph["edges"],
                on_progress=self._progress_bridge(active, graph["nodes"]),
                on_signal=self._signal_bridge(active),
                context=active.context,
                error_mode=options.get("error_mode", DEFAULT_ERROR_MODE),
                max_retries=int(options.get("max_retries", 0)),
                # No ExecutionCache by default: nothing is re-executed
                # within a run, and sharing one ACROSS server-owned runs
                # would serve a later run stale tensors under a matching
                # key. The canvas lends its own — one per socket, cleared by
                # the same reconnect that always cleared it — because
                # incremental re-execution IS the editing loop.
                cache=None if session is None else session.cache,
                changed_nodes=(None if session is None
                               else list(session.changed_nodes) or None),
                run_id=active.run_id,
                output_store=self._output_store,
                record_outputs=active.record_outputs,
                preset_fallback=build_preset_fallback(graph["presets"]),
            )
            return STATUS_SUCCEEDED, None
        except CancellationError:
            # Why it stopped is a different fact from that it stopped: a user
            # pressing Stop is `cancelled`, the server going away is
            # `interrupted`. The engine raises the same exception for both.
            return self._stopped_status(active), None
        except GraphValidationError as exc:
            return STATUS_FAILED, str(exc)
        except Exception as exc:
            logger.warning("run %s failed: %s", active.run_id, exc,
                           exc_info=True)
            return STATUS_FAILED, str(exc)

    async def _finalize(
        self, active: _ActiveRun, status: str, error: str | None,
    ) -> None:
        """Terminal bookkeeping, in the one order that stays consistent.

        Metrics before the terminal event (so a reader that stops at the
        terminal event has the full series), the terminal event before the
        row (so an event-log follower never sees a finished row with no
        closing frame), and ``finished`` set before the row write so a long
        poller woken by the terminal event does not go back to sleep.

        ``terminating`` is set FIRST, before any await: from here the
        outcome is decided, and a cancel that arrives during the metric
        flush must not report that it cancelled a run about to file
        ``succeeded``.

        The run's ADMISSION goes back in the same breath, and before the
        terminal event rather than after it — see ``_release_admission`` for
        the window that closes.
        """
        active.terminating = True
        self._release_admission(active)
        await self._flush_metrics(active, force=True)
        if status == STATUS_SUCCEEDED:
            await self._emit(active.run_id, EVENT_RUN_COMPLETED, None)
        elif status == STATUS_FAILED:
            await self._emit(active.run_id, EVENT_RUN_FAILED,
                             {"error": error or "run failed"})
        else:
            reason = (STOP_REASON_INTERRUPTED if status == STATUS_INTERRUPTED
                      else STOP_REASON_CANCELLED)
            await self._emit(active.run_id, EVENT_RUN_STOPPED,
                             {"reason": reason})
        active.finished = True

        # mark_finished guards on the ACTIVE statuses and reports whether it
        # landed, which is how a cancel racing a completion is resolved
        # (#119): the loser simply did not write, instead of a succeeded run
        # being filed forever as cancelled by a late writer.
        landed = await self.store.mark_finished(active.run_id, status,
                                                error=error)
        if not landed:
            logger.info(
                "run %s was already terminal when %s was reported",
                active.run_id, status)
        await self.prune_retention()

    def _progress_bridge(
        self, active: _ActiveRun, nodes: list[dict[str, Any]],
    ) -> Callable[[str, str, dict[str, Any] | None], Any]:
        """Build the engine's ``on_progress`` callback for one run.

        The message body is byte-for-byte what ``ws_execution`` puts on the
        wire minus its ``type`` key — see the event-vocabulary block above.
        """
        media_ports = declared_media_ports(nodes)

        async def on_progress(
            node_id: str, status: str, result: dict[str, Any] | None,
        ) -> None:
            message: dict[str, Any] = {"node_id": node_id, "status": status}
            if result and status == "error":
                message["error"] = result.get("error", "")
            entries = build_node_output_entries(
                status, result, media_ports.get(node_id))
            if entries:
                message["outputs"] = entries
            await self._emit(active.run_id, EVENT_NODE_STATUS, message)
            if status == "progress":
                self._collect_metrics(active, node_id, result)
                await self._flush_metrics(active)

        return on_progress

    def _signal_bridge(
        self, active: _ActiveRun,
    ) -> Callable[[Any], Any]:
        """Build the engine's ``on_signal`` callback for one run (#122).

        Everything a node reports that is not a node status arrives here,
        already serialised onto the loop by the engine's single pump — so
        this can await the database without a lock and without reordering
        anything.

        Nothing in here is allowed to raise. A metric, an artifact row and a
        dropped-signal notice are all observability; a run must not fail
        because the thing describing it did.
        """
        async def on_signal(signal: Any) -> None:
            try:
                if isinstance(signal, MetricSignal):
                    await self._record_metric(active, signal)
                elif isinstance(signal, ArtifactSignal):
                    await self._record_artifact(active, signal)
                elif isinstance(signal, DroppedSignal):
                    await self._emit(active.run_id, EVENT_WARNING, {
                        "kind": "dropped_signals", "count": signal.count,
                        "detail": ("progress/metric signals were dropped to "
                                   "keep the run's outbox bounded; the "
                                   "series may have gaps"),
                    })
                else:  # pragma: no cover - a node pack's own signal type
                    logger.debug("run %s: ignoring unknown signal %r",
                                 active.run_id, type(signal).__name__)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("run %s: could not record a %s", active.run_id,
                               type(signal).__name__, exc_info=True)

        return on_signal

    async def _record_metric(
        self, active: _ActiveRun, signal: MetricSignal,
    ) -> None:
        """Buffer one explicitly-logged point and claim the node for it."""
        active.explicit_metric_nodes.add(signal.node_id or "")
        active.pending_metrics.append(MetricPoint(
            name=signal.name, value=signal.value, step=signal.step,
            node_id=signal.node_id))
        await self._flush_metrics(active)

    async def _record_artifact(
        self, active: _ActiveRun, signal: ArtifactSignal,
    ) -> None:
        """Register a file the run produced, then announce it.

        Store first: an ``artifact`` event pointing at a row that failed to
        insert would send a client looking for something that is not there.
        """
        meta = signal.meta
        if signal.node_id:
            meta = {**(meta or {}), "node_id": signal.node_id}
        record = await self.store.add_artifact(
            active.run_id, signal.kind, signal.path, meta=meta)
        await self._emit(active.run_id, EVENT_ARTIFACT, {
            "artifact_id": record.id, "kind": record.kind,
            "path": record.path, "meta": record.meta,
            "node_id": signal.node_id,
        })

    # ── events ────────────────────────────────────────────────────────────

    async def _emit(
        self, run_id: str, event_type: str, payload: Any,
    ) -> EventRecord | None:
        """Append one event durably, then hand it to live subscribers.

        Persist-then-fan-out, never the reverse: the store is the source of
        truth and a subscriber must not learn of an event that failed to
        land. A failed append is logged and swallowed — the event log is
        observability, and losing a progress row must not fail the run that
        was only trying to describe itself. ``CancelledError`` is re-raised
        untouched (it is the run stopping, not a logging problem); note that
        an append cancelled AFTER its COMMIT leaves a durable event whose
        cursor is lost here, which is safe precisely because every consumer
        re-reads the tail from the store rather than trusting this return.

        Subscribers receive EXACTLY the object that was stored — sanitised
        (``json_safe``: no NaN/Infinity, which are CPython tokens that
        ``JSON.parse`` rejects, and which would otherwise reach #121's
        WebSocket) and capped (``cap_event_payload``). The tempting
        alternative — store the capped copy, push the full one, since
        subscribers are in-process — is rejected because it would break the
        drop-tolerance contract: a subscriber is told that when it lags it
        may re-read the tail from the store "and lose nothing". If the live
        copy carried more than the stored one, a lagging consumer would
        silently get a DIFFERENT payload, so a UI would show an image live
        and a placeholder after any hiccup. One payload, one truth; #121
        looked at the split again and kept it.

        No write-behind buffer (#121, measured)
        --------------------------------------
        One transaction per event, on a path where a training thread is
        blocked in ``run_coroutine_threadsafe(...).result()`` until it
        returns, invites the obvious "batch the progress writes" reflex. It
        was measured instead of assumed: 300 realistic ``node_status``
        progress payloads through ``append_event`` on the WAL-mode database
        cost a MEAN of 0.68 ms, p95 0.92 ms, p99 1.46 ms, max 1.71 ms. The
        producers are throttled per epoch today (#122 adds a 0.5 s floor),
        so an interactive run spends well under a millisecond per second of
        wall clock here — three orders of magnitude below the epoch it is
        reporting on.

        Against that, a write-behind buffer would have to keep the attach
        contract intact: cursors are allocated INSIDE the transaction, so a
        buffered event has no cursor to fan out, and a re-attaching client
        that re-read the tail would legitimately not find events a
        subscriber had already been shown — the gapless-cursor guarantee
        #121 depends on. Real cost, no measured benefit; deferred with the
        numbers rather than "for now".
        """
        stamp = utc_now_iso()
        stored = cap_event_payload(json_safe(payload),
                                   cap_bytes=self._event_payload_cap_bytes)
        try:
            cursor = await self.store.append_event(run_id, event_type, stored,
                                                   ts=stamp)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("run %s: could not append %s event", run_id,
                           event_type, exc_info=True)
            self._wake(run_id)
            return None
        record = EventRecord(run_id=run_id, cursor=cursor, type=event_type,
                             payload=stored, ts=stamp)
        self._fan_out(run_id, record)
        return record

    def _fan_out(self, run_id: str, record: EventRecord) -> None:
        active = self._runs.get(run_id)
        if active is None:
            return
        for subscription in list(active.subscribers):
            subscription.offer(record)
        active.signal.notify()

    def _wake(self, run_id: str) -> None:
        active = self._runs.get(run_id)
        if active is not None:
            active.signal.notify()

    @asynccontextmanager
    async def subscribe(
        self, run_id: str, *, maxsize: int | None = None,
    ) -> AsyncIterator[RunSubscription]:
        """Live event feed for an in-process consumer (#121's WS bridge).

        A run that is already finished yields an immediately-closed
        subscription: there is nothing left to push, and the consumer should
        replay from ``RunStore.get_events``. Always used as a context manager
        so a dropped consumer cannot leak a queue onto a long-lived run.

        A run still WAITING in the queue (#123) yields a live subscription,
        not a closed one — it has not finished, it has not started, and a
        client that attached to it is entitled to see it start. The set the
        subscription joins is handed to the ``_ActiveRun`` at promotion, so
        no re-subscribe is needed and there is no window between the two in
        which an event could be missed. Discarding on the way out works
        either way BECAUSE it is the same set object.

        *maxsize* overrides the queue depth; ``0`` means unbounded, as
        ``asyncio.Queue`` defines it. (``maxsize or default`` would have
        quietly turned that request into the bounded default.)
        """
        depth = self._subscriber_queue_size if maxsize is None else maxsize
        subscription = RunSubscription(run_id=run_id,
                                       queue=asyncio.Queue(depth))
        active = self._runs.get(run_id)
        waiting = self._pending_by_id.get(run_id)
        if active is not None and not active.finished:
            subscribers = active.subscribers
        elif waiting is not None:
            subscribers = waiting.subscribers
        else:
            subscription.closed = True
            yield subscription
            return
        subscribers.add(subscription)
        try:
            yield subscription
        finally:
            subscribers.discard(subscription)

    async def wait_for_events(
        self,
        run_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 500,
        wait: float = 0.0,
    ) -> list[EventRecord]:
        """Events after *after_cursor*, optionally long-polling for the next.

        The wake-up is edge-triggered (``_Broadcast``) — there is no sleep
        loop and no polling interval anywhere in this function. Order is
        load-bearing: the generation is captured BEFORE the store read, so an
        event that lands during the read still satisfies the wait instead of
        being missed until the next one.

        Returns as soon as anything is available, and immediately (possibly
        empty) once the run is finished or unknown — a long poll on a dead
        run must never hang for its full timeout.

        A run WAITING in the queue parks on its pending entry's channel
        (#123). Without that it would fall into the "unknown" branch and
        return instantly every time, turning every follower of a queued run
        — the Runs panel, ``cdui run --wait`` — into a busy loop against the
        database for as long as the queue is deep.
        """
        deadline = time.monotonic() + max(wait, 0.0)
        while True:
            waiter = self._waiter_for(run_id)
            events = await self.store.get_events(
                run_id, after_cursor=after_cursor, limit=limit)
            if events or waiter is None:
                return events
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            try:
                await asyncio.wait_for(waiter.wait(), remaining)
            except asyncio.TimeoutError:
                return []

    def _waiter_for(self, run_id: str) -> asyncio.Event | None:
        """The generation to park on for a run, or ``None`` to not park.

        ``None`` means "there is nothing more coming for this id": the run
        finished, or this process never had it. Running beats waiting when
        both somehow answer, because a promoted run's registry entry is the
        newer fact.
        """
        active = self._runs.get(run_id)
        if active is not None:
            return None if active.finished else active.signal.waiter()
        waiting = self._pending_by_id.get(run_id)
        return None if waiting is None else waiting.signal.waiter()

    # ── metrics ───────────────────────────────────────────────────────────

    def _collect_metrics(
        self, active: _ActiveRun, node_id: str, payload: Any,
    ) -> None:
        """Infer series from a progress payload — unless the node speaks.

        Explicit beats inferred (#122). ``TrainingLoop`` logs ``train_loss``
        and ``val_loss`` through ``context.log_metric`` AND keeps
        ``loss``/``val_loss`` in its epoch payload for the live chart that
        has always read them; mining both would file every validation loss
        twice. The first explicit point from a node retires the inference
        for that node, which also leaves every node that has no metric API
        (a plugin, a chart pack) working exactly as it did.
        """
        if node_id in active.explicit_metric_nodes:
            return
        step = metric_step(payload, fallback=active.last_metric_step + 1)
        active.last_metric_step = step
        active.pending_metrics.extend(
            scalar_metrics(payload, node_id=node_id, step=step))

    async def _flush_metrics(
        self, active: _ActiveRun, *, force: bool = False,
    ) -> int:
        """Write buffered points in ONE transaction (``log_metrics``).

        Never one row per point: a training loop emits several series per
        epoch and per-point inserts would mean a transaction (and an fsync)
        each. The buffer drains when it is big enough OR when it has been
        sitting for ``metric_flush_interval_s`` — the second rule is what
        keeps a live chart current on a slow producer, checked on the emit
        path so no timer task is needed. ``force`` drains it at terminal.

        A successful flush emits ONE ``metric`` event for the whole batch
        (#122). Per point would mean a durable event row per training batch
        under ``batch_metrics``; per flush reuses a cadence that is already
        bounded by size and by time. Emitted AFTER the write, so the event
        never advertises a point that did not land.
        """
        if not active.pending_metrics:
            return 0
        now = time.monotonic()
        if (not force
                and len(active.pending_metrics) < self._metric_flush_max_points
                and now - active.last_flush_monotonic
                < self._metric_flush_interval_s):
            return 0
        batch, active.pending_metrics = active.pending_metrics, []
        active.last_flush_monotonic = now
        try:
            written = await self.store.log_metrics(active.run_id, batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("run %s: metric flush failed (%d point(s) lost)",
                           active.run_id, len(batch), exc_info=True)
            return 0
        if written:
            await self._emit(active.run_id, EVENT_METRIC, {"points": [
                {"name": p.name, "value": p.value, "step": p.step,
                 "node_id": p.node_id}
                for p in batch
            ]})
        return written

    # ── cancel ────────────────────────────────────────────────────────────

    async def cancel(self, run_id: str) -> CancelOutcome | None:
        """Ask a run to stop. ``None`` when the run does not exist.

        COOPERATIVE, never ``Task.cancel``: this sets the
        ``ExecutionContext`` flag, and where the run actually stops depends
        on what it is doing.

        - **Inside a long-loop node** (``TrainingLoop``,
          ``DiffusionTrainingLoop``, ``EvaluateModel``, ``DDPMSampler``,
          ``Map``): within ONE ITERATION. Since #122 those nodes poll
          ``context.should_stop()`` every batch/step/item, then return their
          partial outputs — a training node also writes an interrupt
          checkpoint on the way out. Stopping a multi-hour training is a
          matter of a batch, not of the epoch it is in.
        - **Inside any other node**: at the node boundary, as before. The
          node runs to the end of its ``run_in_executor`` call and the
          engine unwinds at its next checkpoint (between nodes, between
          levels, and after the last level). There is no safe way to
          interrupt arbitrary third-party node code, and killing the task
          outright is what leaves half-written rows and wedged CUDA state.

        Either way the answer here is an ACKNOWLEDGEMENT — the flag is set,
        and the outcome is OBSERVED on the row when the run unwinds. What
        #122 changed is how long that takes for the nodes where it mattered.

        A run still WAITING in the queue is dequeued instead, and that is the
        cheap case by a wide margin: nothing has been started, so there is
        nothing to unwind, no device to release and no partial output to
        reconcile. It is retired through the same guarded ``mark_finished``
        as any other never-started row (an orphan from a previous boot, or a
        row someone created directly), so a cancel racing a promotion cannot
        rewrite a run that is already going — the promotion happens in one
        synchronous step, which is what makes "in the queue" and "in the
        registry" exhaustive and mutually exclusive.

        The ``terminating`` check is what keeps the answer truthful once a
        run's outcome is decided but it is still in the registry — from the
        first line of ``_finalize`` (before the metric flush and the
        terminal event, both of which await) until the task deregisters
        itself. A registry hit alone would report "cancelled: true" for a
        run that is on its way to filing ``succeeded``.
        """
        entry = self._dequeue(run_id)
        if entry is not None:
            outcome = await self._retire_waiting(entry, STATUS_CANCELLED,
                                                 STOP_REASON_CANCELLED)
            if outcome is not None:
                return outcome

        active = self._runs.get(run_id)
        if active is not None and not active.terminating:
            if active.stop_reason is None:
                active.stop_reason = STOP_REASON_CANCELLED
            active.context.cancel()
            return CancelOutcome(run_id=run_id, status=STATUS_RUNNING,
                                 cancelled=True)

        record = await self.store.get_run(run_id)
        if record is None:
            return None
        if record.status == STATUS_QUEUED:
            landed = await self.store.mark_finished(
                run_id, STATUS_CANCELLED, expected=(STATUS_QUEUED,))
            if landed:
                return CancelOutcome(run_id=run_id, status=STATUS_CANCELLED,
                                     cancelled=True)
            record = await self.store.get_run(run_id) or record
        return CancelOutcome(run_id=run_id, status=record.status,
                             cancelled=False)

    async def _retire_waiting(
        self, entry: _QueuedRun, status: str, reason: str,
    ) -> CancelOutcome | None:
        """File an already-dequeued run as terminal and close its story.

        ``None`` when the guarded write did not land — the row was already
        terminal (deleted, or retired by another path) — so the caller falls
        through to reporting whatever the row now says instead of claiming a
        transition it did not make.

        The ``execution_stopped`` event is written even though there was no
        ``execution_start`` to balance it. A follower (the WS attach view,
        ``cdui run --wait``) needs a frame that says the run ENDED; going
        quiet is indistinguishable from a run that is simply slow.
        Manufacturing a start event to pair with it would be the actual lie.
        """
        landed = await self.store.mark_finished(
            entry.run_id, status, expected=(STATUS_QUEUED,))
        if not landed:
            entry.close()
            return None
        await self._emit(entry.run_id, EVENT_RUN_STOPPED, {"reason": reason})
        entry.close()
        return CancelOutcome(run_id=entry.run_id, status=status,
                             cancelled=True)

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def recover_interrupted(self) -> int:
        """Retire rows a dead process left active. STARTUP ONLY.

        Nothing resumes a ``running`` row after a restart, so leaving one
        claiming to be alive is both a lie to the user and an unbounded
        metrics leak: retention never deletes an active run, so an
        abandoned one would keep its events and metrics forever. This is why
        it must run BEFORE ``prune_retention``.

        Guarded rather than merely documented — calling it while this process
        has live runs would mark those very runs interrupted underneath
        themselves. Waiting runs count for the same reason: their rows are
        exactly the ``queued`` ones this sweep retires, and retiring a run
        the scheduler is still holding would leave the queue pointing at a
        terminal row.
        """
        if self._runs or self._pending_by_id:
            raise RuntimeError(
                "recover_interrupted is a startup call; "
                f"{len(self._runs)} active run(s) and "
                f"{len(self._pending_by_id)} queued run(s) are in flight")
        count = await self.store.interrupt_active_runs()
        if count:
            logger.warning(
                "marked %d run(s) interrupted (they did not survive a "
                "restart)", count)
        return count

    async def prune_retention(self) -> int:
        """Apply the keep-last-N retention policy. ``None`` disables it."""
        keep_last = self._retention_keep_last
        if keep_last is None or keep_last < 0:
            return 0
        deleted = await self.store.prune(keep_last=keep_last)
        if deleted:
            logger.info(
                "run retention: removed %d finished run(s), keeping the "
                "newest %d", deleted, keep_last)
        return deleted

    async def shutdown(self) -> None:
        """Stop every run and DRAIN its task. Must complete before db.close().

        The #119 carry-forward this closes: ``Database.close`` races any
        merely-slow ``db.run`` that is still in flight, and the run tasks are
        the only thing in the process that issues database work nobody is
        awaiting. Every one of them is awaited to completion here, so by the
        time this returns there is no worker thread left holding the
        connection.

        Three phases, in this order:

        0. **Retire the queue.** Every WAITING run is filed ``interrupted``
           on the spot, with the same ``execution_stopped`` frame a running
           run gets. Nothing resumes a queue across a restart — the
           scheduler is in memory and ``recover_interrupted`` retires
           ``queued`` rows precisely because "a queued row would otherwise
           wait forever on a scheduler that lost its queue" — so the only
           question is WHEN the row becomes honest, not what it becomes.
           Writing it here, while the fact is fresh and the database is
           still open, converges on exactly the row the next boot's recovery
           would have written; a hard kill (no graceful shutdown at all)
           still leaves them ``queued`` and recovery still retires them.
           Two paths, one outcome, by construction.

           This comes FIRST so a slot freed by a draining run cannot promote
           a run that is about to be retired. ``_pump`` also refuses to
           start anything once ``_shutting_down`` is set, which is the same
           guarantee from the other end.
        1. Cooperative. The stop reason is ``interrupted`` (the server is
           going away, the user did not ask), so the tasks write an honest
           terminal row on their way out.
        2. Hard, after ``shutdown_grace_s``. ``Task.cancel`` on whatever is
           still stuck in a long node; those rows stay ``running`` and the
           next boot's ``recover_interrupted`` retires them. The cancelled
           tasks are still awaited — cancelling is not draining.
        """
        self._shutting_down = True
        await self._retire_queue()
        active = list(self._runs.values())
        for entry in active:
            # First reason wins: a run the user already cancelled is filed as
            # cancelled even if the server goes down before it unwinds. Their
            # intent is the more specific fact, and it came first.
            if entry.stop_reason is None:
                entry.stop_reason = STOP_REASON_INTERRUPTED
            entry.context.cancel()
        tasks = [entry.task for entry in active if entry.task is not None]
        if not tasks:
            return
        logger.info("shutdown: draining %d in-flight run(s)", len(tasks))
        _done, pending = await asyncio.wait(tasks,
                                            timeout=self._shutdown_grace_s)
        if pending:
            logger.warning(
                "shutdown: %d run(s) did not stop within %.1fs; cancelling "
                "(they will be marked interrupted on the next start)",
                len(pending), self._shutdown_grace_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _retire_queue(self) -> int:
        """File every waiting run ``interrupted``. Returns how many.

        Both indexes are emptied SYNCHRONOUSLY up front, before the first
        await, so nothing can be promoted, cancelled or double-retired while
        the rows are being written one at a time.
        """
        waiting = list(self._pending_by_id.values())
        self._pending_by_id.clear()
        self._pending.clear()
        if not waiting:
            return 0
        logger.info("shutdown: retiring %d queued run(s)", len(waiting))
        for entry in waiting:
            await self._retire_waiting(entry, STATUS_INTERRUPTED,
                                       STOP_REASON_INTERRUPTED)
        return len(waiting)
