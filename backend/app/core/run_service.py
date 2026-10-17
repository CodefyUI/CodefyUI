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
  ``context.log_artifact`` — through the engine's ``on_signal`` callback
  (``core.executors``, since #766).
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

Since #766 the service no longer calls the engine itself. Three layers, each
importing only the one below it:

- ``core.run_events`` -- pure functions that turn engine callback arguments
  into event payloads (``node_status``, ``inner_node_status``, metric points,
  the signal frames), plus the event types. No I/O, no ``RunService``, so a
  worker subprocess can build the very same bytes.
- ``core.executors`` -- an ``Executor`` runs one ``ExecutionRequest`` and
  returns an ``ExecutionOutcome``. ``InProcessExecutor`` (the default, and
  the only one a run without a worker target meets) is what ``_execute``
  used to do: ``execute_graph`` in this process, its callbacks bridged
  through ``run_events`` into the run's ``RunSinks``. ``WorkerExecutor``
  (#772) executes a run placed on a worker agent; see Workers below.
- this module -- scheduling, the row, and persistence: ``RunSinks`` lands in
  ``_emit`` (where ``cap_event_payload`` applies), the metric batcher and the
  artifact table. Classifying a stop as ``cancelled`` or ``interrupted``
  stays here too (``_stopped_status``); an executor only knows that the
  engine stopped.

Lanes
-----
The run's device is resolved once, in :meth:`RunService.submit`: the
effective value is the first entry of ``options["target"]["devices"]`` when
the caller sent a device set (#769), else ``options["device"]`` when the key
is present, else the graph's ``settings.device``, else ``cpu``; ``auto``
resolves to the best accelerator this server has. The same string becomes ``exec_runs.queue_key``
and ``ExecutionContext.device``. The seed follows the same rule:
``options["seed"]`` when the key is present (``null`` is an unseeded run),
else the graph's ``settings.seed``, else none.

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

Who may start is asked of a :class:`~app.core.scheduling.Scheduler`
(#768): ``_pump`` hands it the front of a FIFO and the slots already taken,
and gets back a ``Placement`` (start it, holding these slots) or ``None``
(it stays first in line). Slots are keyed by ``(worker_id, device)``;
:class:`~app.core.scheduling.LocalScheduler`, the default, gives a run the
one slot ``("local", queue_key)`` under the cap above, which is the #123
rule unchanged. The FIFOs themselves stay keyed by the device string.
Since #773 the caps are per slot (``QueueLimits.for_slot``; an override
is ``cuda:0=2`` for this process, ``<worker>/cuda:0=2`` for a worker), a
run with a device set holds a slot on every device of it (all free, or it
waits), and a worker on this machine shares this process's slots.

Workers (#772)
--------------
With a :class:`~app.core.workers.WorkerRegistry` (the lifespan always
passes one), ``options["target"]`` can name a connected worker agent
(``cdui agent start``) or ``any``. Such a run is resolved against the
WORKER, not this process: its FIFO key is ``<worker>/<device as asked>``
(``gpu-box/cuda``, ``any/cpu``), a named worker must be one this server has
seen since it started and offer the device at submit (else
``RunSubmitError``; one that is away is waited for, #777), and the
interactive lane is refused (the canvas always executes here).
``record_outputs`` is honoured on a worker whose agent lists the
``record_outputs`` capability (#789): its captures stay there and the
outputs routes read them over its connection.
:class:`~app.core.scheduling.WorkerScheduler` places it on
``(worker, device)`` slots, capped per device like a local slot; ``any``
tries the connected workers in name order and this process last (a CUDA
set a worker would renumber: the workers only). The
placement decides the executor (``WorkerExecutor`` for a worker,
``InProcessExecutor`` here), the context device (the placement's first
device), and what the row records once the run starts: ``queue_key``
``<worker>/<device>`` and the ``worker_id`` / ``devices`` it got. A worker
run does not take this process's seed gate (its subprocess has RNGs of its
own) and records the WORKER's environment fingerprint, the one its
``hello`` carried (#781). A worker whose node catalog differs from this
server's is ``degraded``. ``any`` skips a worker that LACKS types this
server has, or reported no catalog; a run naming a degraded worker is
refused at submit unless it has every node type the graph uses
(``worker_handshake.catalog_refusal``). The type set is computed once at
submit, in a thread, for every run that targets a worker, and travels with
the queued run to ``WorkerExecutor``, which checks it again before
``assign`` against the worker as it is then. A worker
connecting, draining, undraining or going away calls
:meth:`RunService.reschedule`.
A run without a target never meets any of this.

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


Secrets (#251)
--------------
``exec_runs.graph_snapshot`` stores a SCRUBBED graph. SECRET-typed param
values are lifted out before the insert -- the same promise save, export,
publish and codegen already made -- and held in ``_run_secrets``, keyed by
run id, until ``_load_graph`` puts them back at promotion. A queued run
therefore executes with the key its submitter typed while the shared
database file never contains one. Every value of a node whose type this
server does not know is lifted out the same way (#537), because nothing
here can say which of them is the key.

Why memory is the right place for them, and not a hedge: the queue is
memory too. ``_pending``/``_pending_by_id`` do not survive a restart and
``recover_interrupted`` retires every ``queued`` row a dead process left, so
a snapshot can only ever be read back by the process that wrote it. The
vault's lifetime is exactly the window in which it can be needed. Encrypting
the column instead would need key material this project does not have (the
session token rotates every boot, by design), and refusing to queue a
secret-bearing graph would be a behaviour change that buys nothing this
does not.

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
import contextlib
import functools
import heapq
import itertools
import json
import logging
import math
import time
import weakref
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    NamedTuple,
)

from ..config import settings
from .cache import ExecutionCache
from .db import utc_now_iso
#: ``_current_cuda_index`` is imported rather than re-implemented here:
#: "which card does a bare ``cuda`` mean in this process" is one policy
#: decision, and #135 gave ``device_utils`` a second copy of it for the
#: index-validation fallback. Two copies of a policy drift.
from .device_utils import (
    get_available_devices,
    DEVICE_PATTERN,
    _current_cuda_index,
    graph_settings_device,
    graph_settings_seed,
    resolve_device,
    split_device,
)
from .env_fingerprint import (
    collect_fingerprint,
    execution_facts,
    worker_execution_facts,
)
from .execution_context import (
    ArtifactSignal,
    ExecutionContext,
    WarningSignal,
)
from .executors import (
    ExecutionOutcome,
    ExecutionRequest,
    Executor,
    InProcessExecutor,
    WORKER_S3_KEY,
    RunSinks,
    WorkerExecutor,
)
from .mounts import (
    BACKEND_LOCAL,
    BACKEND_MOUNT,
    BACKEND_S3,
    BACKEND_TRANSFER,
    holds_mount,
)
from .node_state_store import NodeStateStore
from .run_events import (
    DEFAULT_ERROR_MODE,
    EVENT_ARTIFACT,
    EVENT_METRIC,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_FAILED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STOPPED,
    EVENT_WARNING,
    artifact_meta,
    artifact_payload,
    warning_payload,
)
# Moved to ``run_events`` by #766 and re-exported under the names this module
# has always published (the redundant ``as`` marks each one as a re-export).
from .run_events import EVENT_INNER_NODE_STATUS as EVENT_INNER_NODE_STATUS
from .run_events import EVENT_NODE_STATUS as EVENT_NODE_STATUS
from .run_events import NON_METRIC_EVENTS as NON_METRIC_EVENTS
from .run_events import NON_METRIC_KEYS as NON_METRIC_KEYS
from .run_events import metric_step as metric_step
from .run_events import scalar_metrics as scalar_metrics
# The payload cap moved to ``run_events`` by #771 (the worker subprocess
# applies it too); re-exported under the names this module published.
from .run_events import TRUNCATION_MARKER as TRUNCATION_MARKER
from .run_events import cap_event_payload as cap_event_payload
from .run_events import json_size as json_size
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
    WORKER_LOCAL,
    RunProvenance,
    RunRecord,
    RunStore,
    json_safe,
)
from . import memory_admission as mem
from .scheduling import (
    LOCAL_WORKER_ID,
    RESERVED_SLOT,
    LocalScheduler,
    Placement,
    Scheduler,
    SlotKey,
    WorkerScheduler,
    candidate_devices,
    entry_budget,
    entry_pool,
    entry_target,
    no_mount,
    on_a_worker,
    renumbered_on_a_worker,
    target_wait,
)
from .secret_params import (
    SecretVault,
    restore_graph_secrets,
    split_graph_secrets,
)
from .seeding import MAX_SEED as SEEDING_MAX_SEED
from .worker_handshake import catalog_refusal, graph_node_types
from .worker_protocol import (
    CAPABILITY_RECORD_OUTPUTS,
    POOL_ONE_SHOT,
    TARGET_ANY,
    TARGET_LOCAL,
    WORKER_NAME_PATTERN,
    canonical_device,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .workers import WorkerRegistry

logger = logging.getLogger(__name__)


# ── event vocabulary ──────────────────────────────────────────────────────
#
# The event TYPES and their payload builders live in ``core.run_events``
# since #766 (re-exported above); what stays here is how a run's log ENDS.

#: ``execution_stopped`` payload discriminator. The WS protocol has one
#: "stopped" frame; the run ROW distinguishes cancelled from interrupted,
#: and this key carries the same fact into the replay log.
STOP_REASON_CANCELLED = "cancelled"
STOP_REASON_INTERRUPTED = "interrupted"

#: What an executor may report (``ExecutionOutcome.status``). Anything else
#: is filed as a failure naming the bad value (#766): an executor's bug must
#: not leave a row in a status the rest of the service does not expect.
EXECUTOR_STATUSES = frozenset({STATUS_SUCCEEDED, STATUS_FAILED,
                               STATUS_CANCELLED, STATUS_INTERRUPTED})

#: Between the worker and the device in a worker run's queue key
#: (``gpu-box/cuda:0``, ``any/cpu``). Never part of a device string.
WORKER_QUEUE_SEPARATOR = "/"

#: An outcome object without a ``peak_memory`` attribute (#770): the
#: executor did not measure, which is different from measuring None.
_NOT_MEASURED = object()

#: What a slot reserved for an earlier waiting run counts as in
#: ``RunService._admission_view``: more than any cap, so it reads as full.
_RESERVED = RESERVED_SLOT

#: The frames that END a run's log: a client replaying it over the WS
#: attach view (the canvas) takes one of them as the run's end.
#: ``cdui run --wait`` stops on the row's status instead and only prints
#: them. Startup recovery checks a log's last event against this before
#: closing it, so no run is closed twice (#552).
CLOSING_EVENTS = frozenset({EVENT_RUN_COMPLETED, EVENT_RUN_FAILED,
                            EVENT_RUN_STOPPED})

#: The ``execution_facts`` key of a worker run that a restarted server
#: adopts it by (#793): ``{"last_seq": int, "slots": [[worker, device]]}``.
WORKER_RESUME_KEY = "worker_resume"

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
    # Where the run executes (#769): ``{"worker": ..., "devices": [...]}``.
    # Stored only when the caller sent it, so a run without a target keeps
    # exactly the options row it had before; see :func:`normalize_target`.
    "target",
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
#: ``DEFAULT_ERROR_MODE`` is defined in ``run_events`` (#766) because the
#: executor that passes it to the engine needs it too, and imported above.
ERROR_MODES = frozenset({"fail_fast", "continue", "retry"})

#: Retry ceiling. ``retry`` mode re-runs a failing node ``max_retries``
#: times; unbounded, one client typo parks a worker thread forever.
MAX_RETRIES_LIMIT = 10

#: ``DEVICE_PATTERN`` (defined in ``device_utils`` and re-exported here) is
#: the device vocabulary ``resolve_device`` understands, validated as a
#: VALUE: strict option keys exist to stop a typo from silently running
#: forty minutes on the wrong device, and ``{"device": "cudda"}`` defeats
#: that entirely if only the key is checked. ``auto`` resolves to the best
#: accelerator present (``describe_accelerator()["default"]``); an absent
#: device is ``cpu`` unless the graph carries ``settings.device``.

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


def _local_queue_key(device: Any) -> str:
    """The FIFO key of a run executing in this process: its device,
    resolved here and canonical (the rule since #123)."""
    return canonical_queue_key(resolve_device(device))


def _local_device_values() -> list[str]:
    """This process's devices as queue keys (``cpu``, ``cuda:0``, ``mps:0``):
    what an ``any`` run can be placed on here (#772)."""
    values: list[str] = []
    for device in get_available_devices():
        key = canonical_queue_key(device)
        if key not in values:
            values.append(key)
    return values


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
    """How many runs each device slot admits at once.

    A value object rather than four ``settings`` reads scattered through the
    scheduler: the limits are consulted on every promotion, tests need to
    set them without touching global state, and "what is the cap for
    ``cuda:0``" should have exactly one answer computed in one place.

    ``overrides`` beats the per-class default, which is the point of it:
    ``cuda:0=2`` on a 48 GB card is a legitimate thing to want without
    changing what ``cuda:1`` does. A slot is ``(worker_id, device)`` (#768);
    an override key is the bare device for this process (``cuda:0``) and
    ``<worker>/<device>`` for a worker agent (``gpu-box-1/cuda:0``, #773),
    so raising one card's cap never moves the same card number elsewhere.
    """

    cpu: int = 2
    gpu: int = 1
    interactive: int = 2
    overrides: tuple[tuple[str, int], ...] = ()

    def for_slot(self, slot: SlotKey) -> int:
        """The concurrency cap of one ``(worker_id, device)`` slot. >= 1.

        The override for the slot's key, else the per-kind default: ``cpu``
        for ``cpu``, ``gpu`` for every accelerator (including a backend
        nobody has added yet: the classification is "not cpu").

        Never zero: a cap of 0 would accept runs into a queue that can never
        drain, which is a hang dressed up as a configuration. A user who
        wants that should not submit the run.
        """
        worker_id, device = slot
        key = (device if worker_id == LOCAL_WORKER_ID
               else f"{worker_id}{WORKER_QUEUE_SEPARATOR}{device}")
        for override, value in self.overrides:
            if override == key:
                return max(1, value)
        return max(1, self.cpu if device == CPU_QUEUE_KEY else self.gpu)

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
    """``"cuda:0=2,gpu-box/cuda:1=3"`` -> ``(("cuda:0", 2), ("gpu-box/cuda:1", 3))``.

    Two key shapes (#773):

    - ``device`` (``cuda:0``, ``cpu``): this process's slot only, as before
      workers existed, stored as written (lower-cased). It never applies to
      a worker's card of the same number.
    - ``local/<device>``: the same, validated and canonicalised like a
      worker key's device and stored as the bare device.
    - ``<worker>/<device>`` (``gpu-box-1/cuda:0``): that worker's slot only.
      The worker part must be a valid worker name (not ``any``), the device
      a concrete one (``cpu``, ``cuda:N``, ``mps:N``), which is stored in
      canonical form (``mps`` -> ``mps:0``, ``cuda:01`` -> ``cuda:1``). A
      card a same-host worker shares with this process has this process's
      slot and cap, so such an override does not apply to it
      (``RunService.check_worker_overrides`` warns).

    Split on the LAST ``=`` so a key may contain one; a key legitimately
    contains a colon (``cuda:0``), which is why this is not a simple
    ``dict(pair.split("=") for ...)``. Keys are lower-cased.

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
        if WORKER_QUEUE_SEPARATOR in key:
            key = _worker_override_key(entry, key)
            if key is None:
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


def _worker_override_key(entry: str, key: str) -> str | None:
    """The stored key of a ``<worker>/<device>`` override, or None (warned)."""
    worker, _, device = key.partition(WORKER_QUEUE_SEPARATOR)
    worker, device = worker.strip(), device.strip()
    if worker != TARGET_LOCAL and (
            worker == TARGET_ANY or not WORKER_NAME_PATTERN.fullmatch(worker)):
        logger.warning("ignoring run queue override %r: %r is not a worker "
                       "name", entry, worker)
        return None
    try:
        device = canonical_device(device)
    except ValueError:
        device = ""
    # A bare ``cuda`` canonicalises to itself, but a slot is always one card.
    if device and device != "cuda":
        return (device if worker == TARGET_LOCAL
                else f"{worker}{WORKER_QUEUE_SEPARATOR}{device}")
    logger.warning("ignoring run queue override %r: expected "
                   "<worker>/<device> with a concrete device such as "
                   "cuda:0, mps:0 or cpu", entry)
    return None


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


#: ``submit(needs=...)`` default: compute the run's storage needs (#782).
NEEDS_UNKNOWN = object()

#: The worker that is this control-plane process: what a run without a
#: target runs on, and the string ``run_store`` writes for one.
TARGET_WORKER_LOCAL = WORKER_LOCAL
#: "Place it wherever there is room": the connected workers first, then
#: this process (#772).
TARGET_WORKER_ANY = TARGET_ANY
#: Worker names (#769), matched with ``fullmatch``. ``local`` and ``any``
#: match it too; they are the two reserved names a worker cannot enroll
#: under.
TARGET_WORKER_PATTERN = WORKER_NAME_PATTERN
#: The queue-key family of ``target.pool: "one-shot"`` runs (#790):
#: ``pool:one-shot/<device>``. The ``:`` cannot occur in a worker name, so
#: it never collides with a ``<worker>/<device>`` key, and the pool's FIFO
#: is its own: a run waiting for a one-shot worker holds up no ``any`` run.
#: An agent started with ``--run-id`` takes its run out of that line
#: (:meth:`RunService._start_pinned`).
ONE_SHOT_QUEUE_PREFIX = f"pool:{POOL_ONE_SHOT}"
#: Upper bound on ``target.devices``. A device set names cards on ONE
#: host; the cap keeps a stored row bounded like every other option.
MAX_TARGET_DEVICES = 64
#: ``options.target``'s keys (#769, #788).
TARGET_KEYS = frozenset({"worker", "devices", "pool", "memory_gib",
                         "wait"})


def _content_index(store: Any) -> Any:
    """The transfer backend's hash index over *store*'s database (#785),
    or None for a store without one (a test double)."""
    from .transfer import ContentIndex

    db = getattr(store, "db", None)
    return None if db is None else ContentIndex(db)


#: Seconds before a one-shot expiry whose database write failed is tried
#: again (#790).
ONE_SHOT_EXPIRY_RETRY_S = 30.0


def one_shot_timed_out(timeout_s: float) -> str:
    """The error of a run no one-shot worker took in time (#790)."""
    if timeout_s >= 3600 and timeout_s % 3600 == 0:
        span = f"{timeout_s / 3600:.0f} h"
    elif timeout_s >= 60 and timeout_s % 60 == 0:
        span = f"{timeout_s / 60:.0f} min"
    else:
        span = f"{timeout_s:g} s"
    return (f"no one-shot worker took this run within {span} "
            "(CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S); start one with "
            "cdui agent start --one-shot, then submit the run again")


def worker_not_connected(name: str) -> str:
    """The refusal for a ``target.worker`` no agent has connected as since
    this server started (one that connected and went away is waited for).

    One function so the REST layer, the CLI and the tests cannot drift
    apart.
    """
    return (f"worker {name!r} has not connected to this server since it "
            "started; start it with `cdui agent start`, or target 'local' "
            "or 'any'")


def worker_cannot_record_outputs(name: str) -> str:
    """The refusal for ``record_outputs`` on a worker whose agent does not
    list the ``record_outputs`` capability (#789): one from before it."""
    return (f"worker {name!r} cannot record outputs: its agent does not "
            "support it; update the agent, or target 'local' to capture "
            "outputs")


#: Refusals for what a worker run cannot do.
WORKER_REFUSES_INTERACTIVE = (
    "the interactive lane always runs on this server; drop the target or "
    "target 'local'")

#: What each device field accepts, for its refusal message.
_DEVICE_VOCABULARY = "cpu, auto, cuda, cuda:N, mps or mps:N"
_TARGET_DEVICE_VOCABULARY = "cpu, cuda:N, mps or mps:N"


def _checked_device(value: str, *, field: str,
                    allow_auto: bool = True) -> str:
    """Trim, lowercase and check one device string against ``DEVICE_PATTERN``.

    The one device check shared by ``options.device``,
    ``graph.settings.device`` and every ``target.devices`` entry; callers
    have already established *value* is a string. ``allow_auto=False`` is
    the device-set rule: ``auto`` names no card.
    """
    device = value.strip().lower()
    if DEVICE_PATTERN.match(device) is None \
            or (device == "auto" and not allow_auto):
        vocabulary = (_DEVICE_VOCABULARY if allow_auto
                      else _TARGET_DEVICE_VOCABULARY)
        raise RunSubmitError(
            f"unknown {field} {device!r}; expected {vocabulary}")
    return device


def _concrete_device(device: str) -> str:
    """One ``target.devices`` entry in canonical form, or ``RunSubmitError``.

    A device set lists cards, so an entry must name one: ``cpu``,
    ``cuda:N`` or ``mps:N``. ``mps`` alone is ``mps:0`` (there is one), and
    the index is re-spelled from its integer so ``cuda:00`` and ``cuda:0``
    are the same entry. A bare ``cuda`` is refused: which card it means
    depends on the process, and the set is supposed to say.
    """
    kind, index = split_device(device)
    if kind == "cpu":
        return "cpu"
    if index is None and kind == "mps":
        index = 0
    if index is None:
        raise RunSubmitError(
            f"target.devices entry {device!r} names a device kind, not a "
            f"device; use {kind}:N")
    return f"{kind}:{index}"


def normalize_target(raw: Any) -> dict[str, Any] | None:
    """Validate ``options.target`` (#769). ``None`` means "not given".

    Shape: ``{"worker": "local" | "any" | <name>, "devices": [...],
    "pool": "one-shot", "memory_gib": N, "wait": bool}``, every key
    optional and no others. ``worker`` is trimmed and lowercased; absent,
    it is ``local`` (``any`` with a ``pool``).

    ``pool`` (#790): ``"one-shot"`` places the run on a one-shot worker
    (``cdui agent start --one-shot``), any one that offers the device and
    has not taken a run yet, and on nothing else; the run waits for one to
    appear. It goes with ``worker`` absent or ``any`` only. Without
    ``pool`` the normalised target has no ``pool`` key, so a stored target
    reads exactly as before.

    ``memory_gib`` (#788) is the run's memory budget per device, for a card
    whose policy is ``shared``; ``wait`` says whether a run that does not
    fit a shared card now waits (default: yes for ``any``, no for a named
    worker). Both appear in the result only when given. Whether a named
    worker is connected is not
    known here: :meth:`RunService.submit` checks it against the worker
    registry (#772).

    ``devices`` absent (or null) means the run's single ``device``
    option, as before #769. Given, it is a non-empty list of concrete
    devices in canonical form (see :func:`_concrete_device`), with no
    duplicates after
    canonicalising and one kind (``cuda:0`` with ``cpu`` is not a device
    set).

    ``devices`` records the REQUESTED set. Its first entry becomes the
    run's ``device`` (see :func:`normalize_options`). On this process that
    entry is the queue and the device the run executes on, and the run
    holds a slot on every device of the set (#773). On a worker (#772) the
    run holds a slot for every device in the set and its subprocess sees
    all of them (#774), renumbered from ``cuda:0`` in the set's order.
    An ``any`` run with a CUDA set that renumbering changes is therefore
    placed on a worker only (``scheduling.renumbered_on_a_worker``).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RunSubmitError(
            f"target must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - TARGET_KEYS)
    if unknown:
        raise RunSubmitError(
            f"unknown target key(s) {unknown}; supported: "
            f"{sorted(TARGET_KEYS)}")

    pool = raw.get("pool")
    if pool is not None:
        if isinstance(pool, str):
            pool = pool.strip().lower()
        if pool != POOL_ONE_SHOT:
            raise RunSubmitError(
                f"target.pool must be {POOL_ONE_SHOT!r} (the one-shot "
                "workers, #790)")
    worker = raw.get("worker")
    if worker is None:
        worker = TARGET_WORKER_ANY if pool else TARGET_WORKER_LOCAL
    if isinstance(worker, str):
        worker = worker.strip().lower()
    if not isinstance(worker, str) \
            or TARGET_WORKER_PATTERN.fullmatch(worker) is None:
        raise RunSubmitError(
            "target.worker must be 'local', 'any' or a worker name "
            "(lowercase letters, digits and '-', at most 63 chars)")
    if pool and worker != TARGET_WORKER_ANY:
        raise RunSubmitError(
            "target.pool picks the worker: leave target.worker out (or "
            "'any'), or name a one-shot worker in target.worker without a "
            "pool")

    devices = raw.get("devices")
    if devices is not None:
        if not isinstance(devices, list) or not devices:
            raise RunSubmitError("target.devices must be a non-empty list")
        if len(devices) > MAX_TARGET_DEVICES:
            raise RunSubmitError(
                f"target.devices must list at most {MAX_TARGET_DEVICES} "
                "devices")
        cleaned: list[str] = []
        for entry in devices:
            if not isinstance(entry, str):
                raise RunSubmitError("target.devices entries must be strings")
            device = _concrete_device(_checked_device(
                entry, field="target device", allow_auto=False))
            if device in cleaned:
                raise RunSubmitError(
                    f"target.devices lists {device!r} twice")
            cleaned.append(device)
        kinds = sorted({split_device(device)[0] for device in cleaned})
        if len(kinds) > 1:
            raise RunSubmitError(
                f"target.devices mixes device kinds {kinds}; a device set is "
                "one kind")
        devices = cleaned
    target: dict[str, Any] = {"worker": worker, "devices": devices}
    if pool:
        target["pool"] = pool
    # #788: present only when given, so a target without them stores the
    # row it stored before.
    memory_gib = raw.get("memory_gib")
    if memory_gib is not None:
        if isinstance(memory_gib, bool) \
                or not isinstance(memory_gib, (int, float)) \
                or not math.isfinite(memory_gib) \
                or not 0 < memory_gib <= mem.MAX_MEMORY_GIB:
            raise RunSubmitError(
                f"target.memory_gib must be a number above 0 and at most "
                f"{mem.MAX_MEMORY_GIB:g}")
        if worker == TARGET_WORKER_LOCAL:
            raise RunSubmitError(
                "target.memory_gib needs a worker: a budget is enforced in "
                "a worker's run subprocess, and this server runs its own "
                "runs in one shared process")
        target["memory_gib"] = float(memory_gib)
    wait = raw.get("wait")
    if wait is not None:
        if not isinstance(wait, bool):
            raise RunSubmitError("target.wait must be true or false")
        target["wait"] = wait
    return target


def target_devices(options: Any) -> Any:
    """``options["target"]["devices"]``, or None when there is none.

    Reads raw and normalised options alike. On a normalised dict the
    answer is the canonical device list; on raw options, ``is not None``
    is the test "the caller named a device set" (#769).
    """
    target = options.get("target") if isinstance(options, dict) else None
    return target.get("devices") if isinstance(target, dict) else None


def target_worker(options: Any) -> str:
    """The worker a NORMALISED options dict targets; ``local`` without one."""
    target = options.get("target") if isinstance(options, dict) else None
    worker = target.get("worker") if isinstance(target, dict) else None
    return worker or TARGET_WORKER_LOCAL


def normalize_options(raw: Any) -> dict[str, Any]:
    """Validate the submit options and fill in defaults.

    Strict on purpose (see ``OPTION_KEYS``). Returns a NEW dict carrying
    every key except ``target`` -- which appears only when the caller sent
    one, so a run without a target stores the row it stored before #769 --
    so ``exec_runs.options`` always records the full effective
    configuration rather than "whatever the client happened to send" — a
    stored run has to be readable years later without knowing which defaults
    were in force at submit time.

    ``device`` is normalised (trimmed, lowercased) but NOT availability-
    checked here: that resolution happens at start time via
    ``resolve_device`` and lands in ``exec_runs.queue_key``, so the row keeps
    both what was asked for and what was used.

    With ``target.devices`` (#769) the list is the more specific statement,
    and ``devices[0]`` relates to ``device`` in one of three ways:

    - ``device`` absent, or ``"auto"``: ``devices[0]`` becomes the device.
    - ``device`` a bare kind (``cuda``, ``mps``, ``cpu``) that matches the
      kind of ``devices[0]``: they agree, and ``devices[0]`` becomes the
      device (``cuda`` + ``["cuda:1"]`` is ``cuda:1``).
    - anything else that names a different device after trimming,
      lowercasing and canonicalising (``cuda:0`` + ``["cuda:1"]``, ``cpu``
      + ``["cuda:0"]``): ``RunSubmitError``
      ``target.devices and device disagree``. ``cuda:00`` + ``["cuda:0"]``
      agrees, and the device becomes ``cuda:0``.

    How the rest of the set is used depends on where the run executes;
    see :func:`normalize_target`.
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

    target = normalize_target(raw.get("target"))
    devices = None if target is None else target["devices"]

    # ``target.devices`` against ``device`` (#769); see the docstring.
    device = raw.get("device", devices[0] if devices else "cpu")
    if not isinstance(device, str) or not device.strip():
        raise RunSubmitError("device must be a non-empty string")
    device = _checked_device(device, field="device")
    if devices and device != devices[0]:
        kind, index = split_device(device)
        first_kind, first_index = split_device(devices[0])
        agrees = (device == "auto"
                  or (kind == first_kind
                      and (index is None or index == first_index)))
        if not agrees:
            raise RunSubmitError("target.devices and device disagree")
        device = devices[0]

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

    normalized = {
        "device": device,
        "seed": seed,
        "lane": lane.strip(),
        "graph_id": graph_id,
        "error_mode": error_mode,
        "max_retries": max_retries,
        **flags,
    }
    # Only when sent: a run without a target stores the same options row it
    # stored before #769.
    if target is not None:
        normalized["target"] = target
    return normalized


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
    subgraphs = [] if raw.get("subgraphs") is None else raw["subgraphs"]
    if not isinstance(nodes, list):
        raise RunSubmitError("graph.nodes must be a list")
    if not nodes:
        raise RunSubmitError("graph.nodes is empty")
    if not isinstance(edges, list):
        raise RunSubmitError("graph.edges must be a list")
    if not isinstance(presets, list):
        raise RunSubmitError("graph.presets must be a list")
    if not isinstance(subgraphs, list):
        raise RunSubmitError("graph.subgraphs must be a list")
    settings = raw.get("settings")
    if settings is not None and not isinstance(settings, dict):
        raise RunSubmitError("graph.settings must be an object")
    device = (settings or {}).get("device")
    if isinstance(device, str) and not device.strip():
        # A blank string is "no assignment", the reading ``GraphSettings``
        # (save/validate), ``graph_settings_device`` and the canvas share.
        device = None
    if device is not None:
        if not isinstance(device, str):
            raise RunSubmitError("graph.settings.device must be a string")
        device = _checked_device(device, field="graph.settings.device")
    seed = (settings or {}).get("seed")
    if seed is not None:
        # Strict like ``GraphSettings.seed``: ``"7"``, ``true`` or ``2.0``
        # is refused, never coerced (bool is an int subclass).
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RunSubmitError("graph.settings.seed must be an integer")
        if not 0 <= seed <= MAX_SEED:
            raise RunSubmitError(
                f"graph.settings.seed must be between 0 and {MAX_SEED}")
    normalized = {
        "nodes": nodes,
        "edges": edges,
        "presets": presets,
        "subgraphs": subgraphs,
    }
    # Kept only when set, so the snapshot records the assignment and a
    # graph with none stays shaped as before. A seed of 0 is set.
    kept = {}
    if device:
        kept["device"] = device
    if seed is not None:
        kept["seed"] = seed
    if kept:
        normalized["settings"] = kept
    return normalized


def fill_graph_device(graph: Any, options: Any) -> Any:
    """Fill an absent ``options["device"]`` from the graph's ``settings.device``.

    The device half of :func:`apply_graph_settings`, shared with the sweep
    route, which applies the device rule but owns its seed. Returns
    *options* itself when there is nothing to fill: a non-dict (for
    :func:`normalize_options` to refuse), a ``device`` key already present,
    a graph with no device, or a ``target.devices`` list (#769) -- that
    names the device itself, so ``settings.device`` is not consulted and
    cannot "disagree" with a set the caller chose on purpose.
    """
    if options is not None and not isinstance(options, dict):
        return options
    if options is not None and ("device" in options
                                or target_devices(options) is not None):
        return options
    device = graph_settings_device(graph)
    if not device:
        return options
    return {**(options or {}), "device": device}


def apply_graph_settings(graph: dict[str, Any], options: Any) -> Any:
    """Fill the run options the caller left out from the graph's settings.

    ``options["device"]`` absent takes ``settings.device`` (unless
    ``target.devices`` names the device; see :func:`fill_graph_device`) and
    ``options["seed"]`` absent takes ``settings.seed`` (#704). Key absence
    is the test, so a caller that names a value keeps it: an explicit
    ``"seed": null`` is an unseeded run of a seeded graph, and an explicit
    ``"device": null`` still fails in :func:`normalize_options`. A non-dict
    *options* is returned untouched for that function to refuse.

    *graph* is read through :func:`graph_settings_device` and
    :func:`graph_settings_seed`, which drop an invalid value; the submit
    path has already refused one in :func:`normalize_graph`.
    """
    if options is not None and not isinstance(options, dict):
        return options
    filled = dict(fill_graph_device(graph, options) or {})
    if "seed" not in filled:
        seed = graph_settings_seed(graph)
        if seed is not None:
            filled["seed"] = seed
    return filled if filled or options is not None else options


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

        "Never let a failure here replace the exception on its way out" is
        the rule, and it used to be enforced over the ``ensure_future`` line
        alone (#190) — while the await nine lines below it, which is where
        a failing notification actually SURFACES, was unguarded. Both call
        sites are ``finally`` blocks of an ``@asynccontextmanager``, so a
        raise out of here does not merely add noise: it substitutes itself
        for whatever the run was already failing with, and the original is
        gone.

        Swallowing costs nothing that was not already lost. ``_notify_all``
        failing means the waiters were not woken either way; the difference
        is only whether the caller's own exception survives to be reported.
        Logged at warning, because the one thing that can raise here is a
        condition bound to another loop, which is a bug worth seeing.
        """
        try:
            notify = asyncio.ensure_future(self._notify_all())
        except RuntimeError:  # pragma: no cover - loop already gone
            # Nothing can be waiting without a loop to wait on, and the
            # counters are already back.
            return
        self._wakes.add(notify)
        notify.add_done_callback(self._wakes.discard)
        notify.add_done_callback(_swallow_task_result)
        try:
            await asyncio.shield(notify)
        except Exception:  # noqa: BLE001 - see above; CancelledError is a
            # BaseException and so still propagates, which is required:
            # ``shield`` re-raises the cancel that landed on this await.
            logger.warning("run gate: waking the waiters failed",
                           exc_info=True)

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


def _adopted_budget(facts: Mapping[str, Any]) -> "mem.Budget | None":
    """The memory budget an adopted run (#793) was admitted with (#788):
    ``execution_facts.memory_budget``, or None."""
    recorded = facts.get("memory_budget")
    if not isinstance(recorded, Mapping):
        return None
    nbytes = recorded.get("bytes")
    if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes <= 0:
        return None
    source = recorded.get("source")
    from_run = recorded.get("from_run")
    return mem.Budget(nbytes,
                      source if isinstance(source, str) else mem.SOURCE_DECLARED,
                      from_run if isinstance(from_run, str) else None)


def _adopted_placement(worker_id: str, devices: tuple[str, ...],
                       slots: tuple[SlotKey, ...],
                       facts: Mapping[str, Any]) -> Placement:
    """An adopted run's placement (#793), with the memory budget it was
    admitted with on its shared cards (#788): ``execution_facts.
    memory_budget.devices`` and their slots (``slots`` is in ``devices``'
    order, as the placement recorded it), so the ledger holds it again."""
    budget = _adopted_budget(facts)
    recorded = facts.get("memory_budget")
    budget_devices = tuple(
        d for d in (recorded.get("devices") or ())
        if isinstance(d, str) and d in devices) \
        if budget is not None and isinstance(recorded, Mapping) else ()
    if budget is None or not budget_devices or len(slots) != len(devices):
        return Placement(worker_id=worker_id, devices=devices, slots=slots)
    by_device = dict(zip(devices, slots))
    return Placement(worker_id=worker_id, devices=devices, slots=slots,
                     memory_budget=budget.bytes,
                     budget_devices=budget_devices,
                     budget_slots=tuple(by_device[d] for d in budget_devices))


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
    #: Submit order across EVERY queue (``RunService._submit_seq``):
    #: ``reschedule`` offers a freed slot to the oldest front first.
    seq: int = 0
    #: ``run_needs.needs`` for a run that targets a worker (#782): the
    #: scheduler places it only where its mount is. None for a local run.
    #: In memory only, like the entry itself: a restart retires the queue.
    needs: dict[str, Any] | None = None
    #: The node types the run executes (``graph_node_types``), for a run
    #: that targets a worker (#781); None for a local one.
    node_types: frozenset[str] | None = None
    #: Its memory budget (#788): declared (``target.memory_gib``) or from
    #: the budget source. None: none, so it never takes a shared card.
    memory_budget: "mem.Budget | None" = None
    #: ``memory_admission.graph_hash`` of its stored graph, for a run that
    #: targets a worker: recorded in ``execution_facts`` so a later run of
    #: the same graph finds this one's peak (#788). None for a local run.
    graph_hash: str | None = None

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
    #: The slots a queued-lane run took at promotion (#768); every one is
    #: given back on release. ``None`` on the interactive lane, which holds
    #: no device slot.
    placement: Placement | None = None
    #: ``run_needs.needs`` of a run that targets a worker (#782), handed to
    #: ``WorkerExecutor``, which picks the storage backend from it (#785).
    needs: dict[str, Any] | None = None
    #: What ``exec_runs.queue_key`` records once the run starts: the device
    #: for a run in this process, ``<worker>/<device>`` on a worker (#772).
    row_queue_key: str = CPU_QUEUE_KEY
    session: "InteractiveSession | None" = None
    record_outputs: bool = False
    #: The node types it executes, computed at submit for a run that
    #: targets a worker (#781); the executor's pre-assignment check.
    node_types: frozenset[str] | None = None
    #: ``_QueuedRun.memory_budget`` and ``.graph_hash`` (#788).
    memory_budget: "mem.Budget | None" = None
    graph_hash: str | None = None
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
    last_flush_monotonic: float = field(default_factory=time.monotonic)
    #: Nodes that have logged an EXPLICIT metric (#122). Progress payloads
    #: from these are no longer mined by ``scalar_metrics`` — otherwise a
    #: training loop that logs ``val_loss`` and also puts ``val_loss`` in its
    #: epoch payload would write the same point twice.
    explicit_metric_nodes: set[str] = field(default_factory=set)
    #: True once the executor reported a measurement for this run, and the
    #: ``peak_memory`` it reported (#770; None is a measurement: a CPU
    #: run). ``_finalize`` records only a measured run: one stopped before
    #: its first node, an unusable snapshot, or an executor that raised has
    #: no number to report, not a null one.
    peak_measured: bool = False
    peak_memory: dict[str, Any] | None = None
    #: A worker run adopted after a restart of this server (#793): the
    #: last frame its previous process recorded. None for a run started
    #: here.
    resume_seq: int | None = None
    #: The last seq stored in ``execution_facts.worker_resume`` (#793).
    stored_seq: int = 0
    #: When this process adopted the run (UTC ISO), or None (#793).
    adopted_at: str | None = None
    #: An adopted ``s3`` run's ``execution_facts.worker_s3`` (#793, #791).
    resume_s3: dict[str, Any] | None = None

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
        executor: Executor | None = None,
        scheduler: Scheduler | None = None,
        workers: "WorkerRegistry | None" = None,
        budget_source: "mem.BudgetSource | None" = None,
        one_shot_queue_timeout_s: float | None = None,
    ) -> None:
        self.store = store
        # How long a run waiting for a one-shot worker waits before it
        # fails (#790, ``CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S``); 0:
        # forever. Only such runs have a timer; nothing else changes.
        self._one_shot_timeout_s = max(0.0, float(
            settings.WORKER_ONE_SHOT_QUEUE_TIMEOUT_S
            if one_shot_queue_timeout_s is None
            else one_shot_queue_timeout_s))
        #: Run id -> the timer that fails it, for each run waiting for a
        #: one-shot worker (#790).
        self._one_shot_timers: dict[str, asyncio.TimerHandle] = {}
        self._one_shot_expiries: set[asyncio.Task] = set()
        # The connected worker agents (#772). With a registry, a run that
        # targets a worker is placed by ``WorkerScheduler`` and executed by
        # ``WorkerExecutor``; a run without a target never meets either.
        self._workers = workers
        # Memory admission (#788): the budgets promised on shared cards,
        # read by the scheduler and moved by the executor's ``run_started``;
        # and where a run that declares no ``memory_gib`` gets its default.
        # Both idle unless a card's policy is ``shared``.
        ledger = getattr(scheduler, "ledger", None)
        self._ledger: mem.MemoryLedger = (
            ledger if isinstance(ledger, mem.MemoryLedger)
            else mem.MemoryLedger())
        self._budget_source: mem.BudgetSource = (
            mem.HistoryBudgetSource(self._last_peak_memory)
            if budget_source is None else budget_source)
        self._worker_executor: Executor | None = (
            None if workers is None else WorkerExecutor(
                workers, content_index=_content_index(store),
                on_started=self._ledger.started))
        # Where a run's graph executes (#766). In this process unless a
        # caller hands over another executor; the service schedules, records
        # and fans out the same way whichever one it is.
        self._executor: Executor = (InProcessExecutor() if executor is None
                                    else executor)
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
        # Who may leave the front of a FIFO, and on which slots (#768).
        if scheduler is None:
            scheduler = (LocalScheduler(self._limits) if workers is None
                         else WorkerScheduler(self._limits, workers,
                                              _local_device_values,
                                              ledger=self._ledger))
        self._scheduler: Scheduler = scheduler
        if workers is not None:
            # A worker connecting, draining, undraining or going away is
            # capacity appearing or vanishing somewhere no queue key names.
            workers.add_listener(self.reschedule)
            # New memory figures reschedule only when a run with a budget
            # waits for that worker (#788).
            workers.memory_waiting = self._memory_waiting
            # A run that ended here while its worker was away, and that the
            # worker later reports finished (#783).
            workers.on_late_result = self._record_late_result
        self._runs: dict[str, _ActiveRun] = {}
        # ── the queue (#123) ─────────────────────────────────────────────
        # One FIFO per resolved device, plus a by-id index so cancel is O(1)
        # instead of a scan of every queue. The two are written together and
        # only ever from the event loop; ``_enqueue``/``_dequeue`` are the
        # only places that touch both, so they cannot drift.
        self._pending: dict[str, deque[_QueuedRun]] = {}
        self._pending_by_id: dict[str, _QueuedRun] = {}
        # One monotonic submit counter for every queue: several FIFOs can
        # share one slot (``cuda:0``, ``any/cuda`` and a same-host worker's
        # ``<worker>/cuda:0``), and ``reschedule`` serves their fronts in
        # this order so no queue starves the others.
        self._submit_seq = itertools.count(1)
        # A reschedule asked for by a cancel runs on the loop AFTER the
        # cancel, coalesced (``_request_reschedule``); while a bulk cancel
        # holds ``deferred_reschedule`` it waits for the whole batch.
        self._reschedule_holds = 0
        self._reschedule_requested = False
        # Runs holding each ``(worker_id, device)`` slot (#768). NOT derived
        # from ``_runs`` on demand: counting a dict by predicate on every
        # promotion is both slower and easier to get subtly wrong than
        # incrementing where the slot is actually taken and released.
        self._slots: dict[SlotKey, int] = {}
        # Slots a refused device-set run at the front of its FIFO needs,
        # kept off-limits to every later run until it starts (#773; see
        # ``scheduling``'s module docstring). Insertion order is the
        # first-come, first-served order ``_admission_view`` relies on.
        self._reservations: dict[str, tuple[SlotKey, ...]] = {}
        # ``(worker, device)`` overrides already reported as shadowed.
        self._shadowed_overrides_logged: set[tuple[str, str]] = set()
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
        # ── SECRET params (#251) ─────────────────────────────────────────
        # ``run_id -> {address: value}`` for the secrets scrubbed out of a
        # run's stored snapshot. Process memory, never persisted: this is
        # the half that must not reach the shared database file.
        #
        # An in-memory store is safe here for one specific reason -- the
        # queue it shadows is in memory too. ``_pending``/``_pending_by_id``
        # do not survive a restart, and ``recover_interrupted`` retires every
        # ``queued`` row a dead process left behind, so a snapshot can only
        # ever be read back by the SAME process that wrote it. The vault's
        # lifetime is therefore exactly the lifetime over which it can be
        # needed; nothing is lost by not persisting it, because nothing
        # resumes. If queued runs ever DO start surviving a restart, this
        # becomes the wrong answer and the test named for that premise in
        # test_run_secrets.py is what fails first.
        #
        # It is bounded by the queue: an entry is added on submit and
        # removed on exactly one of two exhaustive paths -- ``_drive``'s
        # ``finally`` for a run that started, ``_retire_waiting`` for one
        # that never did.
        self._run_secrets: dict[str, SecretVault] = {}
        self._shutting_down = False
        # Called with ``(run_id, status)`` once a run's terminal status has
        # LANDED on its row (#343). The study coordinator copies a child's
        # evidence onto its attempt here, before ``_finalize`` runs
        # retention -- which may delete the very row it reads.
        self._terminal_listeners: list[
            Callable[[str, str], Awaitable[None]]] = []

    # ── introspection ─────────────────────────────────────────────────────

    def active_run_ids(self) -> list[str]:
        """Runs THIS process is driving, submit order."""
        return list(self._runs)

    def is_active(self, run_id: str) -> bool:
        return run_id in self._runs

    def running_on(self, run_id: str) -> str | None:
        """The worker *run_id* is executing on now, or None: not active
        here, or running in this process. What the transfer upload route
        (#785) checks a worker's upload against."""
        active = self._runs.get(run_id)
        placement = None if active is None else active.placement
        return placement.worker_id if on_a_worker(placement) else None

    def pending_secret_count(self) -> int:
        """How many runs this process is holding SECRET params for (#251).

        A COUNT, never the values — this exists so tests can pin that the
        vault drains, and a diagnostic that returned the keys themselves
        would be a new way to leak them.
        """
        return len(self._run_secrets)

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

        An approximation once a run needs more than one slot (#773): the
        position is counted in the ONE FIFO the run waits in (its queue
        key; for a device set, its first device's). Position 1 means
        nothing in that line is ahead of it, not that it starts next: it
        also needs every other slot of its set, and the runs waiting for
        those cards in their own FIFOs are not counted here, nor are the
        runs a same-host worker has on a shared card.
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

    def queue_reason(self, run_id: str) -> str | None:
        """Why a waiting run cannot start, beyond a busy slot, or None.

        The scheduler's word (``WorkerScheduler.explain``, #772): "waiting
        for worker X to reconnect", "worker X is draining". None for a run
        that is not waiting, or that waits only for its turn.
        """
        entry = self._pending_by_id.get(run_id)
        explain = getattr(self._scheduler, "explain", None)
        if entry is None or explain is None:
            return None
        try:
            return explain(entry, self._admission_view(run_id))
        except Exception:  # noqa: BLE001 -- a description, never an error
            logger.debug("could not explain run %s's wait", run_id,
                         exc_info=True)
            return None

    def queue_snapshot(self) -> dict[str, list[str]]:
        """``{queue_key: [run_id, ...]}`` in service order. Waiting runs only.

        The scheduler's own view, for tests and diagnostics. Running runs are
        NOT in it — they no longer occupy a place in line, they occupy a
        slot.
        """
        return {key: [entry.run_id for entry in queue]
                for key, queue in self._pending.items() if queue}

    def slot_usage(self, slot: SlotKey) -> dict[str, int]:
        """``{"limit": N, "taken": M}`` for one ``(worker_id, device)`` slot.

        What ``GET /api/workers`` shows per device (#773): the cap
        :class:`QueueLimits` gives the slot and how many running runs hold
        it now. Waiting runs are not counted; they hold no slot.
        """
        return {"limit": self._limits.for_slot(slot),
                "taken": self._slots.get(slot, 0)}

    def check_worker_overrides(self, worker: Any) -> None:
        """Warn once per card when a ``<worker>/<device>`` override cannot
        apply because the card is shared with this process (#773): a
        shared card has this process's slot, and this process's cap."""
        for device, local in (getattr(worker, "shared", None) or {}).items():
            key = f"{worker.id}{WORKER_QUEUE_SEPARATOR}{device}"
            if not any(k == key for k, _ in self._limits.overrides):
                continue
            if (worker.id, device) in self._shadowed_overrides_logged:
                continue
            self._shadowed_overrides_logged.add((worker.id, device))
            logger.warning(
                "run queue override %r is ignored: worker %s runs on this "
                "machine and its %s is this server's %s, which uses the "
                "local limit (%d)", key, worker.id, device, local,
                self._limits.for_slot((LOCAL_WORKER_ID, local)))

    @staticmethod
    def local_devices() -> list[str]:
        """This process's devices, canonical (``cpu``, ``cuda:0``, ``mps:0``)."""
        return _local_device_values()

    # ── terminal listeners (#343) ─────────────────────────────────────────

    def add_terminal_listener(
        self, listener: Callable[[str, str], Awaitable[None]],
    ) -> None:
        """Register *listener* to be awaited as ``listener(run_id, status)``
        whenever a run's terminal status lands on its row in this process.

        Awaited inline, in registration order, BEFORE retention runs, so a
        listener can still read the run's metrics. A listener must be quick
        and must not wait on anything that waits on a run: ``cancel`` and
        ``shutdown`` reach it too. Exceptions are logged and swallowed; a
        bookkeeping failure must never fail the run that triggered it.
        """
        self._terminal_listeners.append(listener)

    def remove_terminal_listener(
        self, listener: Callable[[str, str], Awaitable[None]],
    ) -> None:
        with contextlib.suppress(ValueError):
            self._terminal_listeners.remove(listener)

    async def _notify_terminal(self, run_id: str, status: str) -> None:
        for listener in list(self._terminal_listeners):
            try:
                await listener(run_id, status)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("run %s: a terminal listener failed",
                               run_id, exc_info=True)

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(
        self,
        graph: Any,
        *,
        options: Any = None,
        name: str | None = None,
        session: InteractiveSession | None = None,
        provenance: RunProvenance | None = None,
        sweep_id: str | None = None,
        sweep_variant: int | None = None,
        run_id: str | None = None,
        study_id: str | None = None,
        study_attempt_id: str | None = None,
        needs: Any = NEEDS_UNKNOWN,
        memory_budget: "mem.Budget | None" = None,
    ) -> SubmitResult:
        """Persist a run, then SCHEDULE it. Raises ``RunSubmitError`` on input.

        *memory_budget* (#788) is a sweep's or study's child's default budget
        as :meth:`check_batch_budgets` derived it, so the child does not look
        its history up again; None derives it here.

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
        disagree. The effective device is ``target.devices[0]`` when the
        options carry a device set (#769; ``settings.device`` is then not
        consulted, and see :func:`normalize_options` for how it combines
        with ``device``), else ``options["device"]`` when that key is
        present, else the graph's ``settings.device``, else ``cpu``;
        the effective value is written into the stored options so the row
        stays self-describing. ``options["seed"]`` falls back to the graph's
        ``settings.seed`` the same way (see :func:`apply_graph_settings`). ``auto`` resolves to the best accelerator
        this server has.

        *session* is the canvas's process-local state; see
        :class:`InteractiveSession`. Passing one on any other lane is a
        submit error rather than a silent no-op: it would mean a run that
        behaves statefully while its stored row says ``queued``.

        *provenance*, *sweep_id* and *sweep_variant* are passed straight
        through to ``create_run``, and read here only by the lane check
        below — which is *sweep_id*'s single other use (#140). A sweep
        captures ONE ``RunProvenance`` and hands the same object to all N
        variants: the default capture shells out to git twice per run in
        project mode, and thirty-two of those is thirty-two times the cost
        for one answer.

        *run_id*, *study_id* and *study_attempt_id* serve the study
        coordinator (#343): it mints the run id and records it on the
        attempt BEFORE calling this, and the stamp on the row is how a child
        whose linkage write was lost is found again. Like a sweep, a study
        child cannot use the interactive lane.

        *needs* is this run's ``run_needs.needs`` when the caller already
        has it (:meth:`check_placeable_batch` computes it for every child
        of a sweep or study), so it is not computed twice (#782).
        """
        normalized_graph, normalized_options, queue_key, needs = self._prepare(
            graph, options,
            batch_child=sweep_id is not None or study_id is not None,
            needs=needs)
        node_types = None
        if needs is not None:
            # A run that targets a worker (``_prepare`` leaves ``needs`` None
            # only for a local one): the node types it runs, once, off the
            # loop (#781). Checked here for a named worker and carried to the
            # executor, which checks them again before ``assign``.
            node_types = frozenset(await asyncio.to_thread(
                graph_node_types, normalized_graph))
            self._check_worker_catalog(normalized_options, node_types)
        normalized_name = normalize_name(name)
        lane = normalized_options["lane"]
        if session is not None and lane != LANE_INTERACTIVE:
            raise RunSubmitError(
                f"an interactive session requires lane={LANE_INTERACTIVE!r}, "
                f"got {lane!r}")

        if lane == LANE_INTERACTIVE:
            return await self._submit_interactive(
                normalized_graph, normalized_options, normalized_name,
                queue_key, session)

        # SECRET params never reach the row (#251), and neither does any
        # value of a node whose type this server does not know (#537): it
        # cannot tell which of those are secret. The stored copy is
        # scrubbed and the real values are held in memory until promotion
        # re-injects them; see ``self._run_secrets`` for why that is safe.
        stored_graph, secrets = split_graph_secrets(
            normalized_graph, unknown_types_as_secret=True)
        budget, graph_digest = None, None
        if needs is not None:
            # A run that targets a worker (#788): its budget, and the
            # refusal of one that cannot fit a shared card. Before the row
            # is written, so a refusal leaves nothing behind.
            graph_digest = mem.graph_hash(stored_graph, normalized_options)
            budget = await self._memory_budget(
                normalized_options, stored_graph,
                batch_child=sweep_id is not None or study_id is not None,
                derived=memory_budget)
        record = await self.store.create_run(
            graph_snapshot=stored_graph,
            options=normalized_options,
            name=normalized_name,
            status=STATUS_QUEUED,
            queue_key=queue_key,
            provenance=provenance,
            sweep_id=sweep_id,
            sweep_variant=sweep_variant,
            run_id=run_id,
            study_id=study_id,
            study_attempt_id=study_attempt_id,
            worker_id=target_worker(normalized_options),
            devices=target_devices(normalized_options),
            graph_hash=graph_digest,
        )
        if self._shutting_down:
            # The same re-check ``_start`` makes, for the same reason and at
            # the same seam: ``create_run`` awaits (provenance capture can
            # shell out to git), and ``shutdown`` retires the queue at any
            # await. A submit parked in that window would otherwise join a
            # queue that has already been drained and wait for a pump that
            # will never run again.
            raise RunServiceUnavailable(SHUTDOWN_AFTER_PERSIST)
        # After the shutdown re-check, before the run can possibly be
        # promoted: the next three statements do not await, and ``_pump`` is
        # what starts a run, so the vault is populated by the time anything
        # can read the scrubbed snapshot back. Registering it above the
        # re-check would strand an entry on the raise.
        if secrets:
            self._run_secrets[record.id] = secrets
        # Enqueue THEN pump, always — even when the device is idle and the
        # run will start on the very next statement. A fast path that started
        # an empty-queue submit directly would be a second admission rule to
        # keep in step with this one, and the first bug it produced would be
        # a run overtaking a queue that filled between the check and the
        # start.
        entry = _QueuedRun(run_id=record.id, queue_key=queue_key,
                           options=normalized_options, signal=_Broadcast(),
                           seq=next(self._submit_seq),
                           needs=needs, node_types=node_types,
                           memory_budget=budget, graph_hash=graph_digest)
        self._enqueue(entry)
        self._arm_one_shot_timeout(entry)
        self._pump(queue_key)
        return SubmitResult(
            run_id=record.id,
            status=(STATUS_RUNNING if record.id in self._runs
                    else STATUS_QUEUED))

    def _prepare(
        self, graph: Any, options: Any, *, batch_child: bool = False,
        needs: Any = NEEDS_UNKNOWN,
    ) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any] | None]:
        """``(graph, options, queue_key, needs)`` for a submit, or the refusal.

        The one place a submission is validated, shared by :meth:`submit`
        and :meth:`check_placeable_batch` so the two cannot drift: the
        shutdown check, the graph and options normalisation (the graph's
        settings fill what the options leave out), the batch-child lane
        guard and the queue key, which is where a ``target`` is checked.
        *batch_child* is a sweep variant or a study child. ``needs`` is
        ``run_needs.needs`` for a run that targets a worker (#782) and None
        for a local one, which never computes it: the 3.0 path unchanged.
        The queue key's refusals (an unknown worker, a device nobody
        offers, ``record_outputs``) come first; only a run they let through
        has its graph read for storage (:meth:`_check_storage`). A *needs*
        given by the caller (anything but :data:`NEEDS_UNKNOWN`) is used as
        it is.
        """
        if self._shutting_down:
            raise RunServiceUnavailable("run service is shutting down")
        normalized_graph = normalize_graph(graph)
        normalized_options = normalize_options(
            apply_graph_settings(normalized_graph, options))
        if batch_child and normalized_options["lane"] == LANE_INTERACTIVE:
            # Refused for the same reason a session on the wrong lane is:
            # _submit_interactive admits against RUN_INTERACTIVE_MAX_CONCURRENT
            # and REFUSES past it rather than queuing, so a 32-variant
            # interactive sweep would be two runs and thirty 503s.
            raise RunSubmitError(
                f"a sweep cannot use lane={LANE_INTERACTIVE!r}; its variants "
                "are queued, and the interactive lane refuses past its cap "
                "instead of queuing")
        queue_key = self._submit_queue_key(normalized_options)
        if entry_target(normalized_options)[0] == TARGET_WORKER_LOCAL:
            return normalized_graph, normalized_options, queue_key, None
        if needs is NEEDS_UNKNOWN:
            needs = self._needs_of(normalized_graph)
        self._check_storage(normalized_options, needs)
        return normalized_graph, normalized_options, queue_key, needs

    async def _memory_budget(self, options: dict[str, Any],
                             graph: dict[str, Any], *,
                             batch_child: bool = False,
                             derived: "mem.Budget | None" = None,
                             ) -> "mem.Budget | None":
        """A worker run's memory budget (#788), or ``RunSubmitError``.

        ``target.memory_gib`` when declared. Otherwise, only when a shared
        card could take the run, *derived* (a batch child's, from
        :meth:`check_batch_budgets`) or the budget source's suggestion (the
        last peak of this graph on these devices, times 1.2). A run that
        waits for an exclusive card gets its default later, when the card is
        set to ``shared`` or a run of its graph reports a peak
        (:meth:`refresh_budgets`). Then the scheduler's refusal, if
        any (``WorkerScheduler.memory_refusal``): no budget for a run that
        can only go to shared cards, a budget larger than every card, or
        (unless the run may wait) one that fits no card now. A named
        worker's run waits only with ``target.wait: true``; ``any``, and a
        sweep's or a study's child (refusing one would leave half a batch
        submitted), wait unless ``target.wait: false``.
        """
        target = options.get("target") or {}
        declared = target.get("memory_gib")
        budget = (None if declared is None
                  else mem.Budget(mem.gib_to_bytes(declared)))
        wants = getattr(self._scheduler, "wants_budget", None)
        if wants is None or not wants(options):
            return budget
        if budget is None:
            budget = derived
        if budget is None:
            suggestion = await self._budget_source.suggest(graph, options)
            budget = None if suggestion is None else suggestion.budget
        refusal = getattr(self._scheduler, "memory_refusal", None)
        if refusal is not None:
            worker_id, _ = entry_target(options)
            message = refusal(
                options, None if budget is None else budget.bytes,
                self._slots,
                wait=target_wait(options, default=(
                    batch_child or worker_id == TARGET_WORKER_ANY)))
            if message is not None:
                raise RunSubmitError(message)
        return budget

    async def _last_peak_memory(self, digest: str) -> Any:
        lookup = getattr(self.store, "last_peak_memory", None)
        return None if lookup is None else await lookup(digest)

    async def check_batch_budgets(self, graphs: Iterable[Any],
                                  options: Any) -> list["mem.Budget | None"]:
        """Refuse, before anything is written, a sweep or study one of whose
        children would be refused for memory at its submit (#788): every
        child's budget as :meth:`_memory_budget` derives it (declared, or
        its graph's last peak on its devices), judged as that child's submit
        judges it (``target.wait`` as given, else it waits). Returns each
        child's derived default budget in order (None: declared, not on a
        shared card, or none), for the caller to hand to :meth:`submit`
        (``memory_budget=``). Call after :meth:`check_placeable_batch`, whose
        checks this assumes."""
        graphs = list(graphs)
        result: list[mem.Budget | None] = [None] * len(graphs)
        wants = getattr(self._scheduler, "wants_budget", None)
        refusal = getattr(self._scheduler, "memory_refusal", None)
        if wants is None or refusal is None:
            return result
        checked: dict[str, dict[str, Any]] = {}
        seen: set[tuple[str, int | None]] = set()
        for index, graph in enumerate(graphs):
            settings_ = graph.get("settings") if isinstance(graph, dict) \
                else None
            key = json.dumps(settings_, sort_keys=True, default=str)
            if key not in checked:
                _, checked[key], _, _ = self._prepare(
                    graph, options, batch_child=True, needs=None)
            child_options = checked[key]
            if not wants(child_options):
                continue
            declared = (child_options.get("target") or {}).get("memory_gib")
            if declared is not None:
                budget: int | None = mem.gib_to_bytes(declared)
            else:
                stored, _ = split_graph_secrets(
                    normalize_graph(graph), unknown_types_as_secret=True)
                suggestion = await self._budget_source.suggest(
                    stored, child_options)
                if suggestion is not None:
                    result[index] = suggestion.budget
                budget = None if suggestion is None else suggestion.budget.bytes
            if (key, budget) in seen:
                continue
            seen.add((key, budget))
            # The child's own wait (``_memory_budget``: a batch child waits
            # unless ``target.wait: false``), so a child the submit would
            # refuse is refused here, before any row.
            message = refusal(child_options, budget, self._slots,
                              wait=target_wait(child_options, default=True))
            if message is not None:
                raise RunSubmitError(message)
        return result

    def _budgetless(self, graph_hash: str | None = None) -> list[_QueuedRun]:
        """Waiting worker runs a shared card could take that have no budget
        (a declared one is set at submit), of *graph_hash* when given."""
        wants = getattr(self._scheduler, "wants_budget", None)
        if wants is None:
            return []
        return [entry for entry in self._pending_by_id.values()
                if entry.needs is not None and entry.memory_budget is None
                and (graph_hash is None or entry.graph_hash == graph_hash)
                and wants(entry.options)]

    async def refresh_budgets(self, graph_hash: str | None = None) -> None:
        """Give each waiting run :meth:`_budgetless` finds its default
        budget (#788), then reschedule. Called when a card is set to
        ``shared`` (a run submitted while it was exclusive has none) and when
        a run of *graph_hash* succeeds with a peak (a run submitted before
        its graph had one). One that still has none keeps waiting, and its
        reason says how to get it going."""
        entries = self._budgetless(graph_hash)
        for entry in entries:
            graph = await self.store.get_graph_snapshot(entry.run_id)
            if graph is None or entry.run_id not in self._pending_by_id:
                continue
            suggestion = await self._budget_source.suggest(graph,
                                                           entry.options)
            if suggestion is not None:
                entry.memory_budget = suggestion.budget
        if entries:
            self.reschedule()

    def _memory_waiting(self, worker_id: str) -> bool:
        """Whether a waiting run with a memory budget may go to *worker_id*
        (it names it, or ``any``): the registry's question before it
        reschedules on a heartbeat's new memory figures (#788)."""
        for entry in self._pending_by_id.values():
            if entry_budget(entry) is None:
                continue
            target, _ = entry_target(entry.options)
            if target in (worker_id, TARGET_WORKER_ANY):
                return True
        return False

    def card_admission(self, worker: Any, device: str
                       ) -> dict[str, Any] | None:
        """A card's admission figures for ``GET /api/workers`` (#788), the
        scheduler's own (``WorkerScheduler.admission``); None without a
        worker scheduler."""
        admission = getattr(self._scheduler, "admission", None)
        return None if admission is None else admission(worker, device)

    async def suggest_memory(self, graph: Any, options: Any = None
                             ) -> "mem.Suggestion | None":
        """The default budget the budget source gives *graph* run with
        *options* (its ``device`` and ``target.devices``: the history is per
        device kind and count) (#788), for the Queue on... dialog. Raises
        ``RunSubmitError`` on a malformed graph or options."""
        normalized = normalize_graph(graph)
        stored, _secrets = split_graph_secrets(
            normalized, unknown_types_as_secret=True)
        shape = options if isinstance(options, dict) else {}
        target = shape.get("target") if isinstance(shape.get("target"),
                                                   dict) else {}
        return await self._budget_source.suggest(stored, {
            "device": shape.get("device"),
            "target": {"devices": target.get("devices")}})

    def _check_worker_catalog(self, options: dict[str, Any],
                              node_types: frozenset[str]) -> None:
        """Refuse a run naming a worker whose node catalog differs from this
        server's and that lacks a type the run uses (#781). ``any`` is not
        checked here: placement skips the workers that lack types, and the
        executor checks the one it picked."""
        worker_id, _devices = entry_target(options)
        if worker_id in (TARGET_WORKER_LOCAL, TARGET_WORKER_ANY) \
                or self._workers is None:
            return
        worker = self._workers.get(worker_id)
        refusal = (None if worker is None
                   else catalog_refusal(worker, node_types))
        if refusal is not None:
            raise RunSubmitError(refusal)

    @staticmethod
    def _needs_of(graph: dict[str, Any]) -> dict[str, Any]:
        """``run_needs.needs``, with the transfer backend's verdict
        (``needs["transfer"]``, #785) when the run names a mount."""
        from .run_needs import needs as run_needs
        from .transfer import annotate

        return annotate(run_needs(graph))

    def check_placeable_batch(self, graphs: Iterable[Any],
                              options: Any) -> list[dict[str, Any] | None]:
        """Refuse, before anything is written, what :meth:`submit` would
        refuse for any child of a sweep or study (#777).

        Every child gets *options*; they differ only in their graph, and
        of the graph only ``settings`` (device, seed) reaches the options,
        so the options and the queue key are checked once per distinct
        ``settings``. Storage (#782) depends on the whole graph (a variant
        may switch ``checkpoint_every`` on), so each child's needs are
        computed and checked; they are returned in order, for the caller
        to hand to :meth:`submit` (``needs=``) instead of computing them
        again. None for a child that runs here. Raises ``RunSubmitError``
        (or ``RunServiceUnavailable``) with the message ``submit`` would
        give. A ``target`` naming a worker this server has seen but that
        is away is not refused: its children wait for it, as one run does.
        """
        checked: dict[str, dict[str, Any]] = {}
        result: list[dict[str, Any] | None] = []
        for graph in graphs:
            settings_ = graph.get("settings") if isinstance(graph, dict) \
                else None
            key = json.dumps(settings_, sort_keys=True, default=str)
            if key not in checked:
                _, checked[key], _, _ = self._prepare(
                    graph, options, batch_child=True, needs=None)
            child_options = checked[key]
            if entry_target(child_options)[0] == TARGET_WORKER_LOCAL:
                result.append(None)
                continue
            child_graph = normalize_graph(graph)
            child_needs = self._needs_of(child_graph)
            self._check_storage(child_options, child_needs)
            # Per child, not per ``settings``: a variant may change what
            # runs (a Map's preset, a bypass) (#781). Only a named worker
            # whose catalog differs is checked, so the common case costs
            # nothing.
            worker = (None if self._workers is None
                      else self._workers.get(entry_target(child_options)[0]))
            if worker is not None and worker.catalog.degraded:
                self._check_worker_catalog(
                    child_options, frozenset(graph_node_types(child_graph)))
            result.append(child_needs)
        return result

    def _submit_queue_key(self, options: dict[str, Any]) -> str:
        """The FIFO a submitted run joins, or ``RunSubmitError``.

        Without a target, or with ``target.worker == "local"``: the
        resolved device, exactly as before #772. On a worker the device is
        the WORKER's to resolve, so the key is ``<worker>/<device as
        asked>`` (``gpu-box/cuda:1``, ``gpu-box/cuda``) and ``any/<device>``
        for ``any``; :data:`WORKER_QUEUE_SEPARATOR` cannot occur in a device
        string, so the two families never collide. A worker run's row
        records ``<worker>/<device it got>`` once it starts.

        A named worker must be one this server has seen since it started
        (connected now or away: a run for a worker that is away WAITS for
        it to reconnect, #777), and must offer every device of the set (or
        the requested device, or one of the requested kind) as it last
        reported them; ``any`` likewise needs this server or a worker it
        has seen to offer the device (a CUDA set a worker would renumber:
        a worker it has seen, since this server never takes such an
        ``any`` run, see ``WorkerScheduler``). The run must be one a
        worker can do: not the interactive lane, and ``record_outputs``
        only on a named worker whose last ``hello`` listed the capability
        (#789; one that never connected since this server started is
        checked when the run starts, ``WorkerExecutor``).
        Only an unknown worker name, a device nobody offers, or one of
        those options is refused, so a worker disconnecting halfway
        through a sweep's or a study's submissions does not fail it.
        Storage is checked after this, by :meth:`_check_storage`.
        """
        worker_id, devices = entry_target(options)
        if worker_id == TARGET_WORKER_LOCAL:
            return _local_queue_key(options.get("device"))
        if options.get("lane") == LANE_INTERACTIVE:
            raise RunSubmitError(WORKER_REFUSES_INTERACTIVE)
        requested = str(options.get("device") or "cpu")
        if entry_pool(options) == POOL_ONE_SHOT:
            # Never refused for want of a worker (#790): the run waits for
            # a one-shot worker to appear, up to
            # ``CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S``.
            if self._workers is None:
                raise RunSubmitError(
                    "target.pool 'one-shot' needs worker agents, and this "
                    "server has none")
            return f"{ONE_SHOT_QUEUE_PREFIX}{WORKER_QUEUE_SEPARATOR}{requested}"
        if worker_id == TARGET_WORKER_ANY:
            can_serve = getattr(self._scheduler, "can_serve_any", None)
            if can_serve is not None and not can_serve(requested, devices):
                if renumbered_on_a_worker(devices):
                    raise RunSubmitError(
                        f"no known worker offers the device set "
                        f"{', '.join(devices or [])} (with target 'any', a "
                        f"CUDA set other than cuda:0 alone runs on a worker "
                        f"only, never on this server)")
                wanted = ", ".join(devices) if devices else requested
                raise RunSubmitError(
                    f"neither this server nor a known worker offers "
                    f"{wanted}")
            return f"{TARGET_WORKER_ANY}{WORKER_QUEUE_SEPARATOR}{requested}"
        worker = None if self._workers is None else self._workers.get(worker_id)
        if worker is None:
            raise RunSubmitError(worker_not_connected(worker_id))
        capabilities = getattr(worker, "capabilities", None) or ()
        if options.get("record_outputs") and capabilities \
                and CAPABILITY_RECORD_OUTPUTS not in capabilities:
            raise RunSubmitError(worker_cannot_record_outputs(worker_id))
        offered = worker.device_values()
        wanted = devices or []
        missing = [d for d in wanted if not worker.offers(d)]
        if not wanted and not candidate_devices(offered, requested):
            missing = [requested]
        if missing:
            raise RunSubmitError(
                f"worker {worker_id!r} does not offer {', '.join(missing)} "
                f"(it offers {', '.join(offered) or 'nothing'})")
        return f"{worker_id}{WORKER_QUEUE_SEPARATOR}{requested}"

    def _check_storage(self, options: dict[str, Any],
                       needs: Mapping[str, Any] | None) -> None:
        """Refuse a worker run whose *needs* name a mount nobody it may go
        to holds (#782), unless the transfer backend can ship its files
        instead (#785).

        A named worker must hold the mount, as it last reported its mounts
        (``mounts.holds_mount``: a same-host worker holds this server's
        label without declaring it), or the run must fit the transfer
        backend (``needs["transfer"]``, decided by ``core.transfer``). For
        ``any``, this server or a known worker that can serve the files
        must offer the device; this server always holds its own label.
        Called after :meth:`_submit_queue_key`, so the worker is known and
        offers the device.
        """
        from .transfer import refusal, transferable

        mount = needs.get("mount") if isinstance(needs, Mapping) else None
        if not mount:
            return
        transfer = transferable(needs)
        worker_id, devices = entry_target(options)
        requested = str(options.get("device") or "cpu")
        if entry_pool(options) == POOL_ONE_SHOT:
            return  # whoever appears; placement checks the mount (#790)
        if worker_id == TARGET_WORKER_ANY:
            can_serve = getattr(self._scheduler, "can_serve_any", None)
            if can_serve is not None and not can_serve(
                    requested, devices, mount=mount, transfer=transfer):
                wanted = ", ".join(devices) if devices else requested
                why = refusal(needs)
                raise RunSubmitError(
                    f"this run reads or writes files under the `{mount}` "
                    f"mount, and neither this server nor a known worker "
                    f"with that mount offers {wanted}"
                    + (f" (the transfer backend cannot ship them: {why})"
                       if why else ""))
            return
        worker = None if self._workers is None else self._workers.get(worker_id)
        if worker is not None and not holds_mount(worker, mount) \
                and not transfer:
            raise RunSubmitError(no_mount(worker_id, mount, refusal(needs)))

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
            # Scrubbed on the way in, exactly like the queued lane (#251) --
            # but with no vault entry, because this lane never reads its
            # snapshot back. ``_start`` gets the LIVE graph, so the row is
            # write-only and the key in it would be pure retained liability.
            stored_graph, _secrets = split_graph_secrets(
                graph, unknown_types_as_secret=True)
            record = await self.store.create_run(
                graph_snapshot=stored_graph, options=options, name=name,
                status=STATUS_QUEUED, queue_key=queue_key,
                worker_id=target_worker(options),
                devices=target_devices(options),
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

    # ── waiting for a one-shot worker (#790) ──────────────────────────────

    def _waits_for_one_shot(self, options: dict[str, Any]) -> bool:
        """Whether a run with *options* waits for a one-shot worker: the
        one-shot pool, or a worker that is known to be one-shot."""
        if entry_pool(options) == POOL_ONE_SHOT:
            return True
        worker_id, _ = entry_target(options)
        if worker_id in (TARGET_WORKER_LOCAL, TARGET_WORKER_ANY) \
                or self._workers is None:
            return False
        worker = self._workers.get(worker_id)
        return bool(getattr(worker, "one_shot", False))

    def _arm_one_shot_timeout(self, entry: _QueuedRun) -> None:
        """Fail *entry* if no one-shot worker takes it in time (#790, the
        GitHub runner rule: queued until a matching runner comes online,
        failed after 24 h). SYNCHRONOUS; a no-op for every other run."""
        if self._one_shot_timeout_s <= 0 \
                or not self._waits_for_one_shot(entry.options):
            return
        loop = asyncio.get_running_loop()
        self._one_shot_timers[entry.run_id] = loop.call_later(
            self._one_shot_timeout_s, self._one_shot_timed_out, entry.run_id)

    def _disarm_one_shot_timeout(self, run_id: str) -> None:
        timer = self._one_shot_timers.pop(run_id, None)
        if timer is not None:
            timer.cancel()

    def _one_shot_timed_out(self, run_id: str) -> None:
        self._one_shot_timers.pop(run_id, None)
        if run_id not in self._pending_by_id or self._shutting_down:
            return
        task = asyncio.ensure_future(self._expire_one_shot(run_id))
        self._one_shot_expiries.add(task)
        task.add_done_callback(self._one_shot_expiries.discard)

    async def _expire_one_shot(self, run_id: str) -> None:
        entry = self._dequeue(run_id)
        if entry is None:
            return
        error = one_shot_timed_out(self._one_shot_timeout_s)
        try:
            await self._fail_waiting(entry, error)
        except asyncio.CancelledError:
            # Shutdown: its queue is being retired; this run with it.
            self._requeue_unexpired(entry)
            raise
        except Exception:  # noqa: BLE001 -- retried below
            logger.exception("run %s: could not fail it after its one-shot "
                             "wait; trying again in %.0f s", run_id,
                             ONE_SHOT_EXPIRY_RETRY_S)
            self._requeue_unexpired(entry, retry=True)
            return
        logger.warning("run %s failed: %s", run_id, error)

    def _requeue_unexpired(self, entry: _QueuedRun, *,
                           retry: bool = False) -> None:
        """Put back a run whose expiry did not land, so it is neither lost
        from the queue nor left without a timer (#790). A run whose row is
        no longer ``queued`` is caught by the guarded write on the retry,
        which then closes it."""
        self._enqueue(entry)  # during shutdown ``_retire_queue`` files it
        if retry and not self._shutting_down:
            loop = asyncio.get_running_loop()
            self._one_shot_timers[entry.run_id] = loop.call_later(
                ONE_SHOT_EXPIRY_RETRY_S, self._one_shot_timed_out,
                entry.run_id)

    async def one_shot_run_refusal(self, run_id: str,
                                   worker_name: str) -> str | None:
        """Why a one-shot agent started with ``--run-id`` *run_id* may not
        connect as *worker_name* (#790), or None: the run must be waiting
        for it (naming it, or the one-shot pool) or already running there.
        A refusal ends the agent at once instead of leaving it to wait for
        a run that will never be placed on it."""
        active = self._runs.get(run_id)
        if active is not None:
            placement = active.placement
            if placement is not None and placement.worker_id == worker_name:
                return None
            return (f"run {run_id} is already running "
                    f"({'on ' + placement.worker_id if placement else 'here'})")
        entry = self._pending_by_id.get(run_id)
        if entry is not None:
            worker_id, _ = entry_target(entry.options)
            if worker_id == worker_name \
                    or entry_pool(entry.options) == POOL_ONE_SHOT:
                return None
            return (f"run {run_id} does not wait for a one-shot worker (its "
                    f"target is {worker_id!r}); submit it with target.pool "
                    f"{POOL_ONE_SHOT!r} or target.worker {worker_name!r}")
        record = await self.store.get_run(run_id)
        if record is None:
            return f"run {run_id} does not exist on this server"
        return f"run {run_id} is not waiting to start (it is {record.status})"

    def _dequeue(self, run_id: str) -> _QueuedRun | None:
        """Take a run out of the line wherever it is. ``None`` if not waiting.

        O(n) in the length of ONE device's queue, and only ever called by
        cancel and by shutdown. A run that reaches the front is removed by
        ``_pump``'s ``popleft`` instead, which is the hot path.
        """
        entry = self._pending_by_id.pop(run_id, None)
        if entry is None:
            return None
        self._disarm_one_shot_timeout(run_id)
        queue = self._pending.get(entry.queue_key)
        was_front = bool(queue) and queue[0] is entry
        if queue is not None:
            try:
                queue.remove(entry)
            except ValueError:  # pragma: no cover - indexes move together
                pass
            if not queue:
                self._pending.pop(entry.queue_key, None)
        reserved = self._reservations.pop(run_id, None) is not None
        if was_front or reserved:
            # The run behind it is the new front, and nothing else would
            # offer it to the scheduler: a front that was waiting for a
            # worker, or whose placement raised, holds no slot whose
            # release would pump its queue. Cards it reserved are open
            # to every other run again too. Deferred, not done here: a
            # "cancel all" dequeues its runs one by one, and the sibling
            # behind this one must not start just to be cancelled next.
            self._request_reschedule()
        return entry

    def _request_reschedule(self) -> None:
        """Reschedule once, on the loop, after the current step. Coalesced.

        Inside :meth:`deferred_reschedule` it waits for the block to end.
        Outside a running loop (a synchronous caller) it reschedules now.
        """
        if self._reschedule_requested:
            return
        self._reschedule_requested = True
        if self._reschedule_holds:
            return
        try:
            asyncio.get_running_loop().call_soon(self._run_requested_reschedule)
        except RuntimeError:
            self._run_requested_reschedule()

    def _run_requested_reschedule(self) -> None:
        if self._reschedule_holds or not self._reschedule_requested:
            return  # the hold's end runs it
        self._reschedule_requested = False
        self.reschedule()

    @contextlib.contextmanager
    def deferred_reschedule(self) -> Iterator[None]:
        """Hold back rescheduling for a bulk cancel; reschedule once after.

        Cancelling a sweep's or a study's runs one at a time would
        otherwise start each queued sibling as the one ahead of it leaves
        (a cancelled front, a running sibling releasing its slot), only to
        cancel it a moment later. Nestable; the outermost exit reschedules
        if anything asked for it.
        """
        self._reschedule_holds += 1
        try:
            yield
        finally:
            self._reschedule_holds -= 1
            if not self._reschedule_holds and self._reschedule_requested:
                self._reschedule_requested = False
                self.reschedule()

    def _reserve(self, entry: _QueuedRun) -> None:
        """Record (or drop) what the refused front-of-line *entry* reserves.

        Asked of the scheduler each time the entry is refused, so a run
        that stops qualifying (its worker went away or is draining) stops
        holding cards idle. An existing reservation keeps its place in the
        first-come order.
        """
        reservation = getattr(self._scheduler, "reservation", None)
        try:
            slots = None if reservation is None else reservation(entry)
        except Exception:  # noqa: BLE001 -- a reservation is an optimisation
            logger.exception("could not compute run %s's reservation",
                             entry.run_id)
            slots = None
        if slots:
            self._reservations[entry.run_id] = tuple(slots)
        else:
            self._reservations.pop(entry.run_id, None)

    def _admission_view(self, run_id: str) -> Mapping[SlotKey, int]:
        """``taken`` as *run_id* may see it: slots reserved ahead of it full.

        A reservation blocks every run that is not itself holding an
        EARLIER one (first come, first served), so the oldest waiting set
        run always makes progress and two set runs cannot block each other.
        """
        if not self._reservations:
            return self._slots
        view = dict(self._slots)
        for holder, slots in self._reservations.items():
            if holder == run_id:
                break
            for slot in slots:
                view[slot] = _RESERVED
        return view

    def _pump(self, queue_key: str) -> None:
        """Start waiting runs while the scheduler places them. SYNCHRONOUS.

        Strict FIFO per queue key: the scheduler is asked about the FRONT
        entry only, and a ``None`` stops the pump, so nothing overtakes a
        run that cannot start yet.

        Called on submit; a run ending, a waiting run leaving the front of
        its line and a worker changing go through :meth:`reschedule`, which
        asks every queue. So there is no scheduler task, no polling
        interval, and nothing to drain at shutdown. Being synchronous is
        what makes it safe to call from ``_drive``'s ``finally``: the slot
        is released and the next run
        registered without an await in between, so no third party can slip
        into a slot that has already been promised.

        Refusing to start anything during shutdown is deliberate — the
        pending runs are being retired by ``shutdown`` at that moment, and
        promoting one into a task nobody drains is the exact failure
        ``_start``'s own re-check exists to prevent.
        """
        while self._pump_one(queue_key):
            pass
        self._start_pinned()

    def _start_pinned(self) -> None:
        """Start the run each ``--run-id`` one-shot agent waits for, out of
        its line (#790). SYNCHRONOUS.

        Such an agent takes that run and no other, so the strict FIFO would
        leave it idle (holding a batch allocation) while older runs wait in
        front of its run in the one-shot pool, or in its own name's line.
        The scheduler still decides: the run starts only where and when
        :meth:`place` puts it. One registry pass, a dict lookup per agent.
        """
        if self._shutting_down or self._workers is None:
            return
        for worker in self._workers.workers():
            run_id = getattr(worker, "one_shot_run_id", None)
            if not run_id or not getattr(worker, "one_shot_available", False) \
                    or not worker.accepts_runs:
                continue
            entry = self._pending_by_id.get(run_id)
            if entry is None:
                continue
            try:
                placement = self._scheduler.place(
                    entry, self._admission_view(entry.run_id))
            except Exception:
                logger.exception("the scheduler failed to place run %s for "
                                 "one-shot worker %s", run_id, worker.name)
                continue
            if placement is None or not placement.slots:
                continue
            queue = self._pending.get(entry.queue_key)
            if queue is None:  # pragma: no cover - the indexes move together
                continue
            position = queue.index(entry)
            del queue[position]
            if not queue:
                self._pending.pop(entry.queue_key, None)
            self._pending_by_id.pop(entry.run_id, None)
            self._reservations.pop(entry.run_id, None)
            try:
                self._start(entry.run_id, None, entry.options,
                            queue_key=entry.queue_key,
                            lane=entry.options.get("lane", DEFAULT_LANE),
                            placement=placement, signal=entry.signal,
                            subscribers=entry.subscribers,
                            node_types=entry.node_types, needs=entry.needs)
            except Exception:
                logger.exception("run %s could not be started for one-shot "
                                 "worker %s; it stays queued", run_id,
                                 worker.name)
                self._pending.setdefault(entry.queue_key,
                                         deque()).insert(position, entry)
                self._pending_by_id[entry.run_id] = entry
                continue
            self._disarm_one_shot_timeout(entry.run_id)

    def _pump_one(self, queue_key: str) -> bool:
        """Start the front of *queue_key* if the scheduler places it.

        True when a run started (the queue may move again), False when the
        front stays where it is or the queue is empty. ``_pump``'s body,
        one entry at a time, so :meth:`reschedule` can interleave queues.
        """
        if self._shutting_down:
            return False
        queue = self._pending.get(queue_key)
        if not queue:
            self._pending.pop(queue_key, None)
            return False
        front = queue[0]
        try:
            # Inside the guard: a scheduler that raises must not unwind
            # ``_finalize`` or ``submit`` any more than a failed start
            # may. The run stays first in line.
            placement = self._scheduler.place(
                front, self._admission_view(front.run_id))
        except Exception:
            logger.exception(
                "the scheduler failed to place run %s off the %s queue; "
                "it stays queued", front.run_id, queue_key)
            return False
        if placement is None:
            self._reserve(front)
            return False
        if not placement.slots:
            # A placement that holds no slot would never be counted,
            # so the loop would drain the whole FIFO uncapped.
            logger.warning(
                "the scheduler placed run %s without any slot; treating "
                "it as a refusal, the run stays queued", front.run_id)
            return False
        entry = queue.popleft()
        if not queue:
            self._pending.pop(queue_key, None)
        self._pending_by_id.pop(entry.run_id, None)
        self._reservations.pop(entry.run_id, None)
        try:
            # The graph is NOT passed: ``_drive`` re-reads it from the
            # row. The signal and the subscriber set ARE — same objects,
            # so a client that parked on the queued run keeps both its
            # generation and its live feed across the promotion.
            self._start(entry.run_id, None, entry.options,
                        queue_key=queue_key, lane=entry.options.get(
                            "lane", DEFAULT_LANE),
                        placement=placement,
                        signal=entry.signal, subscribers=entry.subscribers,
                        node_types=entry.node_types,
                        needs=entry.needs,
                        memory_budget=entry.memory_budget,
                        graph_hash=entry.graph_hash)
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
            self._pending.setdefault(queue_key, deque()).appendleft(entry)
            self._pending_by_id[entry.run_id] = entry
            return False
        self._disarm_one_shot_timeout(entry.run_id)
        return True

    def _start(
        self, run_id: str, graph: dict[str, Any] | None,
        options: dict[str, Any], *, queue_key: str,
        lane: str = DEFAULT_LANE,
        placement: Placement | None = None,
        session: InteractiveSession | None = None,
        signal: _Broadcast | None = None,
        subscribers: set[RunSubscription] | None = None,
        node_types: frozenset[str] | None = None,
        needs: dict[str, Any] | None = None,
        memory_budget: "mem.Budget | None" = None,
        graph_hash: str | None = None,
        resume_seq: int | None = None,
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

        *placement* is the scheduler's answer for a queued-lane run (the
        slots it takes here); the interactive lane passes none.

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
        # Where it executes: the first device of its placement (#772) --
        # for a run placed in this process by ``LocalScheduler`` that is its
        # queue key, as it always was -- else (the interactive lane) the
        # queue key.
        on_worker = on_a_worker(placement)
        device = (placement.devices[0]
                  if placement is not None and placement.devices
                  and lane != LANE_INTERACTIVE else queue_key)
        context = ExecutionContext(
            execution_id=run_id,
            device=device,
            max_workers=settings.MAX_PARALLEL_NODES,
            verbose=bool(options.get("verbose")),
            graph_id=str(options.get("graph_id") or ""),
            backward_mode=bool(options.get("backward_mode")),
            auto_backward=bool(options.get("auto_backward")),
            # Reproducibility (#134). The seed reaching the CONTEXT — not
            # just the executor's ``seed_rngs`` before the graph starts — is
            # what lets the engine derive a per-node seed and lets a node
            # build its own generator, which is the difference between
            # "seeded" and "actually reproducible".
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
            queue_key=queue_key, lane=lane,
            placement=None if lane == LANE_INTERACTIVE else placement,
            row_queue_key=(f"{placement.worker_id}{WORKER_QUEUE_SEPARATOR}"
                           f"{device}" if on_worker else device),
            session=session,
            record_outputs=bool(options.get("record_outputs")),
            subscribers=set() if subscribers is None else subscribers,
            node_types=node_types,
            needs=needs,
            memory_budget=memory_budget,
            graph_hash=graph_hash,
            resume_seq=resume_seq,
            stored_seq=resume_seq or 0,
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
            for slot in active.placement.slots if active.placement else ():
                self._slots[slot] = self._slots.get(slot, 0) + 1
            if active.placement is not None \
                    and active.placement.memory_budget is not None:
                # Reserved until the worker's ``run_started`` says its cap
                # is in place (#788), released with the slots.
                # The footprint: its budget plus the CUDA context the cap
                # does not cover.
                self._ledger.reserve(run_id, active.placement.budget_slots,
                                     mem.footprint(
                                         active.placement.memory_budget,
                                         active.placement.budget_devices[0]))
            if on_worker and self._workers is not None:
                # In the same step as the slots: a one-shot worker takes
                # this run and no other (#790).
                self._workers.claim_one_shot(placement.worker_id, run_id)
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
        slots = active.placement.slots if active.placement else ()
        self._ledger.release(active.run_id)
        for slot in slots:
            remaining = self._slots.get(slot, 0) - 1
            if remaining > 0:
                self._slots[slot] = remaining
            else:
                self._slots.pop(slot, None)
        if on_a_worker(active.placement) and self._workers is not None:
            # A one-shot worker whose ``assign`` never went out (cancelled
            # while starting, refused before it, a failed send) may take
            # another run (#790). One place every ending path passes.
            self._workers.release_one_shot(active.placement.worker_id,
                                           active.run_id)
        if self._reschedule_holds:
            self._request_reschedule()  # a bulk cancel is in progress
        else:
            self.reschedule()

    def reschedule(self) -> None:
        """Offer the front of every waiting FIFO to the scheduler. SYNCHRONOUS.

        For the moments capacity can appear somewhere a single queue key
        cannot name: a run releasing its slots, a waiting run leaving the
        front of its line, and (#772) a worker coming online or draining.
        Every non-empty FIFO is asked, because which queue a freed slot
        unblocks is the scheduler's knowledge, not this method's; there are
        a handful of FIFOs, so asking all of them is cheap.

        One run at a time, oldest front first (by submit order across all
        queues, ``_QueuedRun.seq``): several FIFOs can share one slot
        (``cuda:0``, ``any/cuda`` and a same-host worker's
        ``<worker>/cuda:0`` all take ``("local", "cuda:0")``), and serving
        the releasing run's own queue first would let one long line keep
        the slot forever. A queue whose front is refused is not asked
        again in this pass, so strict FIFO per queue still holds.
        """
        fronts = [(queue[0].seq, key)
                  for key, queue in self._pending.items() if queue]
        heapq.heapify(fronts)
        while fronts:
            seq, key = heapq.heappop(fronts)
            queue = self._pending.get(key)
            if not queue or queue[0].seq != seq:
                continue  # stale: that queue moved since it was pushed
            if not self._pump_one(key):
                continue  # refused: not asked again in this pass
            queue = self._pending.get(key)
            if queue:
                heapq.heappush(fronts, (queue[0].seq, key))
        self._start_pinned()

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
            # Not for an adopted worker run (#793): it is executing on its
            # worker, which must be told to stop (``WorkerExecutor``).
            if active.resume_seq is None and active.context.should_stop():
                await self._finalize(active, self._stopped_status(active), None)
                return
            # Taken BEFORE ``_begin`` so a seeded run waiting its turn is
            # still reported as `queued` rather than as a `running` run
            # emitting nothing — the status stays true while it waits.
            # A worker run executes in a subprocess with RNGs of its own, so
            # it neither waits for nor holds this process's seed gate.
            gate = (contextlib.nullcontext() if on_a_worker(active.placement)
                    else run_exclusion().for_seed(options.get("seed")))
            async with gate:
                # An adopted worker run (#793) began under the previous
                # process: its row and start event are already there.
                if active.resume_seq is None \
                        and not await self._begin(active, graph, options):
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
            # One of the vault's two exhaustive drop points (#251): every run
            # that STARTED passes through here, including the hard-cancelled
            # ones that skip ``_finalize`` entirely. Before
            # ``_release_admission``, which pumps the next run.
            self._run_secrets.pop(run_id, None)
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
        snapshot.setdefault("subgraphs", [])
        # Put the SECRET params back (#251). The row was stored scrubbed, so
        # this is the step that makes a queued run execute with the key its
        # submitter typed -- and with every value of a node whose type was
        # unknown at submit (#537), even if that type has been registered
        # since: restore takes the vault's word for what was withheld, not
        # the registry's. A run whose vault entry is gone -- an orphan row
        # from a previous boot, a graph submitted before this existed --
        # simply runs with the blank the snapshot carries and fails the way
        # a missing key already fails.
        restore_graph_secrets(snapshot, self._run_secrets.get(active.run_id))
        return snapshot

    async def _begin(
        self, active: _ActiveRun, graph: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> bool:
        """Claim the row and announce the run. False means "do not run".

        The only way this fails is the row not existing any more — deleted
        out from under a submission — in which case there is nothing left to
        write events or a status to.

        This is also where the run's environment is recorded (#344): HERE,
        at execution start, rather than at submit, because a queued run can
        wait hours and the server can be updated in between -- the record
        must name the environment that actually executes it. Collecting it
        never fails the run; a failure is logged and leaves the columns
        NULL, which reads as "not recorded" rather than as a wrong answer.
        """
        on_worker = on_a_worker(active.placement)
        if on_worker:
            # This process's environment is not the one executing the run:
            # the worker's fingerprint, as its ``hello`` reported it (#781;
            # None, which reads as "not recorded", when it reported none),
            # and only the facts that do not depend on this process. The
            # worker's ``run_started`` merges in the rest (``WorkerExecutor``).
            worker = (None if self._workers is None
                      else self._workers.get(active.placement.worker_id))
            fingerprint = (None if worker is None
                           else worker.fingerprint_document)
            digest = None if worker is None else worker.fingerprint
            facts = worker_execution_facts(
                options=options or {}, graph=graph,
                worker_id=active.placement.worker_id,
                devices=list(active.placement.devices))
            # What a restarted server needs to adopt the run (#793).
            facts[WORKER_RESUME_KEY] = self._worker_resume(active, 0)
        else:
            digest = None
            fingerprint, facts = await self._environment_record(
                active, graph, options)
        if facts is not None:
            facts = {**facts, **self._memory_facts(active)}
        started = await self.store.mark_running(
            active.run_id, queue_key=active.row_queue_key,
            # The worker it got (an ``any`` run's row said ``any``); a local
            # run rewrites the ``local`` it already had.
            worker_id=(active.placement.worker_id
                       if active.placement is not None else None),
            devices=list(active.placement.devices) if on_worker else None,
            fingerprint=fingerprint, fingerprint_digest=digest,
            execution_facts=facts)
        if not started:
            logger.warning("run %s vanished before it could start",
                           active.run_id)
            return False
        await self._emit(active.run_id, EVENT_RUN_STARTED,
                         {"run_id": active.run_id})
        return True

    @staticmethod
    def _memory_facts(active: _ActiveRun) -> dict[str, Any]:
        """The run's ``execution_facts`` keys of #788: ``graph_hash`` (a
        worker-targeted run's), and ``memory_budget`` (``{"bytes",
        "source", "devices", ["from_run"]}``) when it was admitted to a
        shared card by memory. Nothing for a local run."""
        facts: dict[str, Any] = {}
        if active.graph_hash:
            facts["graph_hash"] = active.graph_hash
        placement = active.placement
        if placement is not None and placement.memory_budget is not None:
            budget = active.memory_budget or mem.Budget(
                placement.memory_budget)
            facts["memory_budget"] = {
                **budget.describe(), "bytes": placement.memory_budget,
                "devices": list(placement.budget_devices)}
        return facts

    @staticmethod
    async def _environment_record(
        active: _ActiveRun, graph: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """``(fingerprint, execution_facts)`` for a run about to start.

        The fingerprint is blocking (a lockfile read, package metadata,
        git once per process), so it goes to a thread like
        ``RunProvenance.capture`` does.
        """
        try:
            fingerprint = await asyncio.to_thread(collect_fingerprint)
            facts = execution_facts(
                options=options or {}, graph=graph,
                resolved_device=active.context.device)
            return fingerprint, facts
        except Exception:
            logger.warning("run %s: could not record its environment",
                           active.run_id, exc_info=True)
            return None, None

    async def _execute(
        self, active: _ActiveRun, graph: dict[str, Any],
        options: dict[str, Any], session: InteractiveSession | None = None,
    ) -> tuple[str, str | None]:
        """Hand the run to the executor and read back its outcome.

        Never raises but cancel. Where the graph executes is the executor's
        business (#766); what a stop MEANS is this service's: an executor
        reports ``cancelled`` for any cooperative stop, and
        ``_stopped_status`` turns that into ``interrupted`` when the server
        asked rather than a user.
        """
        executor = (self._worker_executor
                    if on_a_worker(active.placement) and self._worker_executor is not None
                    else self._executor)
        try:
            outcome = await executor.run(ExecutionRequest(
                run_id=active.run_id,
                graph=graph,
                options=options,
                context=active.context,
                session=session,
                output_store=self._output_store,
                record_outputs=active.record_outputs,
                sinks=self._sinks(active),
                explicit_metric_nodes=active.explicit_metric_nodes,
                placement=active.placement,
                node_types=active.node_types,
                needs=active.needs,
                resume_seq=active.resume_seq,
                resume_s3=active.resume_s3,
            ))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The executor is supposed to classify every failure itself;
            # one that raises anyway must still leave a terminal row and a
            # closing event, so the run is FAILED rather than abandoned.
            logger.warning("run %s: the executor raised: %s", active.run_id,
                           exc, exc_info=True)
            outcome = ExecutionOutcome(STATUS_FAILED, str(exc))
        else:
            # Only an outcome the executor itself reported carries a
            # measurement; a raise, or a value that is neither a map nor
            # null, records nothing (key absent = not recorded).
            peak_memory = getattr(outcome, "peak_memory", _NOT_MEASURED)
            if peak_memory is None or isinstance(peak_memory, dict):
                active.peak_measured = True
                active.peak_memory = peak_memory
        status = getattr(outcome, "status", None)
        if status not in EXECUTOR_STATUSES:
            logger.warning("run %s: the executor reported status %r",
                           active.run_id, status)
            return STATUS_FAILED, (
                f"the executor reported an unknown status {status!r}")
        if status == STATUS_CANCELLED:
            # A stop carries no error, whatever the executor attached.
            return self._stopped_status(active), None
        # ``interrupted`` (#772) falls through with its error: nobody here
        # asked for the stop -- the run's worker went away, or ended it on
        # its own -- and the error says which.
        return status, outcome.error

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

        A run its executor measured has its device memory merged into
        ``execution_facts`` after the metric flush (#770, see
        ``_record_peak_memory``).
        """
        active.terminating = True
        self._release_admission(active)
        await self._flush_metrics(active, force=True)
        if active.peak_measured:
            await self._record_peak_memory(active)
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
        else:
            await self._notify_terminal(active.run_id, status)
            if status == STATUS_SUCCEEDED and active.graph_hash \
                    and active.peak_memory and self._budgetless(
                        active.graph_hash):
                # Its peak is a default for runs of the same graph that
                # wait without a budget (#788).
                try:
                    await self.refresh_budgets(active.graph_hash)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 -- a default, never an error
                    logger.warning("could not give waiting runs of %s a "
                                   "memory budget", active.graph_hash,
                                   exc_info=True)
        await self.prune_retention()

    async def _record_peak_memory(self, active: _ActiveRun) -> None:
        """Merge the executor's peak memory into ``execution_facts`` (#770).

        Before the terminal event, so a reader woken by it finds the number
        on the row. ``null`` is a recorded answer (a CPU run, no torch);
        a facts document that was never written stays unwritten (see
        ``RunStore.update_execution_facts``). Never raises but cancel: a
        run must not fail because the thing describing it did.
        """
        try:
            landed = await self.store.update_execution_facts(
                active.run_id, {"peak_memory": active.peak_memory})
            if not landed:
                logger.info(
                    "run %s: no execution facts recorded, so its peak "
                    "memory is dropped", active.run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("run %s: could not record its peak memory",
                           active.run_id, exc_info=True)

    # ── sinks (#766) ──────────────────────────────────────────────────────
    #
    # The persistence half of what used to be the three engine bridges
    # (``_progress_bridge``, ``_inner_status_bridge``, ``_signal_bridge``).
    # The payloads are built by ``core.run_events``; the executor calls these
    # through the run's ``RunSinks`` (see ``_sinks``).

    def _sinks(self, active: _ActiveRun) -> RunSinks:
        """The ``RunSinks`` one run's executor reports into."""
        return RunSinks(
            emit=functools.partial(self._emit, active.run_id),
            record_metrics=functools.partial(self._record_metrics, active),
            record_artifact=functools.partial(self._record_artifact, active),
            warn=functools.partial(self._warn, active),
            record_execution_facts=functools.partial(
                self.store.update_execution_facts, active.run_id),
            record_worker_connection=functools.partial(
                self._record_worker_connection, active),
            # Only read by a restarted server's adoption (#793): with no
            # resume grace nothing would ever read it.
            record_worker_seq=(functools.partial(self._record_worker_seq,
                                                 active)
                               if on_a_worker(active.placement)
                               and self._resume_grace() > 0 else None),
        )

    @staticmethod
    def _worker_resume(active: _ActiveRun, seq: int) -> dict[str, Any]:
        """``execution_facts.worker_resume`` (#793): the last frame recorded
        and the slots the run holds, which a restarted server cannot work
        out before the worker is back (a same-host card's slot is this
        process's)."""
        slots = active.placement.slots if active.placement else ()
        return {"last_seq": int(seq),
                "slots": [list(slot) for slot in slots]}

    async def _record_worker_seq(self, active: _ActiveRun, seq: int) -> None:
        """Before a worker run's frames up to *seq* are acknowledged (#793):
        flush the metric buffer, so nothing the agent then forgets lives
        only in this process's memory, and persist *seq* for a restart."""
        if seq <= active.stored_seq:
            return  # nothing moved: no write
        batch = list(active.pending_metrics)
        # One transaction: the buffered points and the seq, both or
        # neither. A failure raises (no ack goes out) and keeps the points
        # buffered for the next try.
        if not await self.store.checkpoint_worker_run(active.run_id, batch,
                                                      seq):
            raise RuntimeError(f"run {active.run_id} has no worker_resume "
                               "record to store its seq in")
        del active.pending_metrics[:len(batch)]
        active.stored_seq = seq
        if batch:
            active.last_flush_monotonic = time.monotonic()
            await self._announce_metrics(active, batch)

    def _resume_grace(self) -> float:
        """``CODEFYUI_WORKER_RESUME_GRACE_S`` as the registry has it; 0
        without a registry."""
        return float(getattr(self._workers, "resume_grace_s", 0) or 0)

    async def _record_worker_connection(self, active: _ActiveRun,
                                        connected: bool) -> None:
        """The row's ``running`` <-> ``disconnected`` (#783). The run keeps
        its slots either way: they go back when it ends."""
        if active.terminating:
            return
        await self.store.mark_worker_connection(active.run_id, connected)
        self._wake(active.run_id)

    async def _record_late_result(self, worker: str, run_id: str,
                                  frame: Any) -> None:
        """A worker reported how a run ended after it had ended here (#783).

        Recorded as ``execution_facts.late_result`` (``{status, error,
        at}``) and logged. Never as an event: the run's log already ended
        with its terminal event, and a reader that stops there must not
        find more. Never as a status: ``mark_finished`` only lands on
        active rows, and a late report does not overwrite history.
        """
        record = await self.store.get_run(run_id)
        if record is None:
            logger.info("run %s: late result from worker %s, but the run "
                        "is gone", run_id, worker)
            return
        logger.warning("run %s ended here as %s; worker %s reports it %s",
                       run_id, record.status, worker, frame.status)
        await self.store.update_execution_facts(run_id, {"late_result": {
            "status": frame.status, "error": frame.error,
            "worker_id": worker, "at": utc_now_iso()}})

    def worker_status(self, run_id: str) -> dict[str, Any] | None:
        """Where an active worker run's worker stands (#783), or None.

        ``{"worker_id", "state", "disconnected_at", "resume_deadline"}``:
        ``state`` is ``connected`` or ``disconnected``; the two times are
        Unix seconds while it is disconnected, else None. None for a run
        that is not active or does not execute on a worker.
        """
        active = self._runs.get(run_id)
        if active is None or not on_a_worker(active.placement) \
                or self._workers is None:
            return None
        worker_id = active.placement.worker_id
        run = self._workers.run_state(worker_id, run_id)
        since = None if run is None else run.detached_wall
        grace = getattr(self._workers, "resume_grace_s", 0.0)
        return {
            "worker_id": worker_id,
            "state": "connected" if since is None else "disconnected",
            "disconnected_at": since,
            "resume_deadline": None if since is None else since + grace,
        }

    async def _warn(self, active: _ActiveRun, signal: WarningSignal) -> None:
        """A ``WarningSignal`` a node sent, as a ``run_warning`` event."""
        await self._emit(active.run_id, EVENT_WARNING, warning_payload(signal))

    async def _record_metrics(
        self, active: _ActiveRun, points: list[MetricPoint],
    ) -> None:
        """Buffer points, then let the batcher decide whether to flush.

        Called on every progress frame, even with no points: the time rule
        in ``_flush_metrics`` is checked on this path.
        """
        active.pending_metrics.extend(points)
        await self._flush_metrics(active)

    async def _record_artifact(
        self, active: _ActiveRun, signal: ArtifactSignal,
    ) -> None:
        """Register a file the run produced, then announce it.

        Store first: an ``artifact`` event pointing at a row that failed to
        insert would send a client looking for something that is not there.

        A signal carrying a ``mount_label`` (a worker child's file under a
        shared mount, #782) is stored as a ``mount`` row whose path is
        relative to that mount; every other one as before, a ``local`` row
        with the path as produced. Either way the row names the worker that
        produced it (``local`` for this process), so a path from another
        machine is never mistaken for one of this machine's.
        """
        mount_label = getattr(signal, "mount_label", None)
        placement = active.placement
        if getattr(signal, "backend", None) == BACKEND_TRANSFER:
            # Uploaded here by the worker (#785): the path is the key the
            # upload answered with, and only one of this run's own keys is
            # recorded (a row that pointed elsewhere would let retention
            # delete it).
            from .transfer import key_path

            if key_path(active.run_id, signal.path) is None:
                logger.warning("run %s: ignoring a transferred artifact "
                               "outside the run's directory: %r",
                               active.run_id, signal.path)
                return
            backend = BACKEND_TRANSFER
        elif getattr(signal, "backend", None) == BACKEND_S3:
            # Uploaded to the run's bucket by the worker (#791): the path is
            # the object key, recorded only under the run's own prefix.
            # The row records its bucket (#791), so it keeps naming its
            # object after a settings change.
            from .s3 import owns_key

            if not owns_key(active.run_id, signal.path,
                            getattr(signal, "prefix", None)):
                logger.warning("run %s: ignoring an S3 artifact outside the "
                               "run's prefix: %r", active.run_id,
                               signal.path)
                return
            backend = BACKEND_S3
        else:
            backend = BACKEND_MOUNT if mount_label else BACKEND_LOCAL
        adopted_at = getattr(active, "adopted_at", None)
        if adopted_at is not None and await self.store.has_artifact(
                active.run_id, signal.kind, signal.path,
                meta=artifact_meta(signal), before=adopted_at):
            # An adopted run (#793): the worker replays what the previous
            # process recorded after its last stored seq. Only a row that
            # process wrote, for the same file AND meta, is that replay; a
            # file rewritten later (``latest.pt``) is recorded as usual.
            logger.info("run %s: artifact %s already recorded; replay "
                        "ignored", active.run_id, signal.path)
            return
        record = await self.store.add_artifact(
            active.run_id, signal.kind, signal.path,
            meta=artifact_meta(signal),
            backend=backend,
            mount_label=mount_label or None,
            worker_id=(placement.worker_id if placement is not None
                       else LOCAL_WORKER_ID),
            bucket=getattr(signal, "bucket", None))
        await self._emit(active.run_id, EVENT_ARTIFACT,
                         artifact_payload(record, signal.node_id))

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
            await self._announce_metrics(active, batch)
        return written

    async def _announce_metrics(self, active: _ActiveRun,
                                batch: list[MetricPoint]) -> None:
        """The ``metric`` event for a batch that landed."""
        await self._emit(active.run_id, EVENT_METRIC, {"points": [
            {"name": p.name, "value": p.value, "step": p.step,
             "node_id": p.node_id}
            for p in batch
        ]})

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

        **And an acknowledgement is not a promise (#404).** Every bullet
        above assumes the node co-operates. ``should_stop()`` is a flag a
        node has to READ, and nothing makes it: a third-party node, or a
        first-party one with a loop that was never instrumented, runs to
        completion exactly as if nobody had asked. ``cancelled=True`` here
        means "the request was delivered", never "the run will stop", and
        there is no deadline behind it — the alternative is ``Task.cancel``
        mid-step, which is what leaves half-written rows and wedged CUDA
        state, and which the whole cooperative design exists to avoid. A
        caller that renders this as "stopped" is reporting something this
        method does not know. ``shutdown`` is the one place that escalates,
        and only because the process is going away regardless.

        This is where a sweep's ``cancelling`` state gets its meaning, and
        its open end: ``POST /api/sweeps/{id}/cancel`` is N of these calls,
        so the sweep sits at ``cancelling`` until every child is terminal,
        and a single node that never polls holds it there indefinitely. See
        ``sweep_store.SWEEP_STATE_CANCELLING`` for what a reader can do
        about that (count what is still active; do not watch the clock).

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
            if on_a_worker(active.placement) and self._resume_grace() > 0:
                # The stop may wait for an absent worker (#783); written
                # down so a restarted server that adopts the run asks again
                # instead of letting it finish against the user's Stop.
                try:
                    await self.store.mark_stop_requested(
                        run_id, active.stop_reason)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 -- the stop still goes out
                    logger.warning("run %s: could not record the stop",
                                   run_id, exc_info=True)
            return CancelOutcome(run_id=run_id, status=STATUS_RUNNING,
                                 cancelled=True)

        record = await self.store.get_run(run_id)
        if record is None:
            return None
        if record.status == STATUS_QUEUED:
            landed = await self.store.mark_finished(
                run_id, STATUS_CANCELLED, expected=(STATUS_QUEUED,))
            if landed:
                await self._notify_terminal(run_id, STATUS_CANCELLED)
                return CancelOutcome(run_id=run_id, status=STATUS_CANCELLED,
                                     cancelled=True)
            record = await self.store.get_run(run_id) or record
        return CancelOutcome(run_id=run_id, status=record.status,
                             cancelled=False)

    async def fail_queued_for_worker(self, worker_id: str,
                                     error: str) -> list[str]:
        """Fail every run still waiting for *worker_id* (#780: the worker
        was removed, so nothing will ever place them). Running runs are not
        touched here; closing the worker's socket interrupts those.
        Returns the run ids that were failed."""
        failed: list[str] = []
        for run_id, waiting in list(self._pending_by_id.items()):
            if entry_target(waiting.options)[0] != worker_id:
                continue
            entry = self._dequeue(run_id)
            if entry is None:  # pragma: no cover - same loop, same index
                continue
            if await self._fail_waiting(entry, error):
                failed.append(run_id)
        if failed:
            logger.warning("%d queued run(s) failed: %s", len(failed), error)
        return failed

    async def _fail_waiting(self, entry: _QueuedRun, error: str) -> bool:
        """File an already-dequeued run ``failed`` with *error*. True when
        the guarded write landed (the row was still ``queued``)."""
        run_id = entry.run_id
        landed = await self.store.mark_finished(
            run_id, STATUS_FAILED, error=error, expected=(STATUS_QUEUED,))
        # After the write: one that raised leaves the run queued with its
        # secrets, for a retry (#790).
        self._run_secrets.pop(run_id, None)
        if landed:
            await self._emit(run_id, EVENT_RUN_FAILED, {"error": error})
        entry.close()
        if landed:
            await self._notify_terminal(run_id, STATUS_FAILED)
        return landed

    async def _retire_waiting(
        self, entry: _QueuedRun, status: str, reason: str,
    ) -> CancelOutcome | None:
        """File an already-dequeued run as terminal and close its story.

        ``None`` when the guarded write did not land — the row was already
        terminal (deleted, or retired by another path) — so the caller falls
        through to reporting whatever the row now says instead of claiming a
        transition it did not make.

        The ``execution_stopped`` event is written even though there was no
        ``execution_start`` to balance it. A client following the log over
        the WS attach view needs a frame that says the run ENDED; going
        quiet is indistinguishable from a run that is simply slow.
        (``cdui run --wait`` stops on the row's status and prints the
        frame.) Manufacturing a start event to pair with it would be the
        actual lie.
        """
        # The vault's other drop point (#251), and the reason it lives here
        # rather than in each caller: this is the one function BOTH cancel
        # and shutdown route a never-started run through, so a run that
        # leaves the queue without reaching ``_drive`` cannot keep its key.
        self._run_secrets.pop(entry.run_id, None)
        landed = await self.store.mark_finished(
            entry.run_id, status, expected=(STATUS_QUEUED,))
        if not landed:
            entry.close()
            return None
        await self._emit(entry.run_id, EVENT_RUN_STOPPED, {"reason": reason})
        entry.close()
        await self._notify_terminal(entry.run_id, status)
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

        Each retired run's log is closed with the ``execution_stopped``
        frame a graceful shutdown writes (#552), in the same transaction as
        the row. A canvas tab that re-attaches after the restart replays
        that log, and without a closing frame nothing on the wire ever told
        it the run was over: it stayed on Running until the page was
        reloaded. The frame goes straight to the store rather than through
        ``_emit`` because the guard above means no subscriber can be waiting
        for it.

        Worker runs are the exception (#793). A run a worker agent was
        executing did not die with this process: its subprocess keeps
        running and the agent spools its frames until a control plane
        acknowledges them. With a worker registry and a resume grace
        (``CODEFYUI_WORKER_RESUME_GRACE_S`` > 0), every ``running`` or
        ``disconnected`` row placed on a worker is ADOPTED instead of
        retired (:meth:`_adopt`): it goes ``disconnected`` and waits for
        its worker as if the connection had just dropped. Runs that
        executed in this process, and every ``queued`` row, are retired
        exactly as before. Returns how many rows were retired.
        """
        if self._runs or self._pending_by_id:
            raise RuntimeError(
                "recover_interrupted is a startup call; "
                f"{len(self._runs)} active run(s) and "
                f"{len(self._pending_by_id)} queued run(s) are in flight")
        adoptable = await self._adoptable_worker_runs()
        count = await self.store.interrupt_active_runs(
            closing_event=EVENT_RUN_STOPPED,
            closing_payload={"reason": STOP_REASON_INTERRUPTED},
            closing_types=CLOSING_EVENTS,
            keep=[record.id for record, _placement, _seq in adoptable],
        )
        if count:
            logger.warning(
                "marked %d run(s) interrupted (they did not survive a "
                "restart)", count)
        for record, placement, last_seq in adoptable:
            try:
                # Before anything can read it: the row says the worker is
                # away from the first moment of this process.
                await self.store.mark_worker_connection(record.id, False)
                self._adopt(record, placement, last_seq)
            except Exception:
                logger.exception("run %s could not be adopted; it ends "
                                 "interrupted", record.id)
                if await self.store.mark_finished(
                        record.id, STATUS_INTERRUPTED,
                        error="this server restarted and could not take "
                              "the run over"):
                    await self.store.append_event(
                        record.id, EVENT_RUN_STOPPED,
                        {"reason": STOP_REASON_INTERRUPTED})
                    count += 1
        if adoptable:
            logger.warning(
                "%d run(s) on worker agents were in flight; they wait up "
                "to %.0f s for their workers to reconnect", len(self._runs),
                self._workers.resume_grace_s)
        return count

    async def _adoptable_worker_runs(
        self,
    ) -> list[tuple[RunRecord, Placement, int]]:
        """``(row, placement, last seq)`` for each worker run a restart
        adopts (#793); none without a worker registry or a resume grace.

        The placement is rebuilt from the row: its worker and devices, and
        the slots ``execution_facts.worker_resume`` recorded. A row without
        that record (started by a server before 3.2, or one that never
        recorded it) is retired as before: its slots cannot be known (a
        same-host card's slot is this process's, ``("local", "cuda:0")``),
        and guessing them would let another run onto its card. So is a
        row naming no device.
        """
        if self._workers is None or self._resume_grace() <= 0:
            return []
        adoptable = []
        for record in await self.store.list_worker_runs_in_flight(
                closing_types=CLOSING_EVENTS):
            devices = [d for d in (record.devices or []) if isinstance(d, str)]
            facts = (record.execution_facts
                     if isinstance(record.execution_facts, dict) else {})
            resume = facts.get(WORKER_RESUME_KEY)
            resume = resume if isinstance(resume, dict) else {}
            last_seq = resume.get("last_seq")
            slots = tuple(
                (slot[0], slot[1]) for slot in resume.get("slots") or ()
                if isinstance(slot, list) and len(slot) == 2
                and all(isinstance(part, str) and part for part in slot))
            if not devices or not slots or not isinstance(last_seq, int) \
                    or isinstance(last_seq, bool) or last_seq < 0:
                logger.warning("run %s on worker %s did not record where it "
                               "can be resumed; it cannot be adopted",
                               record.id, record.worker_id)
                continue
            adoptable.append((record, _adopted_placement(
                record.worker_id, tuple(devices), slots, facts), last_seq))
        return adoptable

    def _adopt(self, record: RunRecord, placement: Placement,
               last_seq: int) -> None:
        """Drive *record*, a worker run a previous process left in flight
        (#793). SYNCHRONOUS: the registry, the slots and the task are set
        up with no await in between.

        Started through :meth:`_start` like any run, with *last_seq* as its
        resume point: it holds its slots again (its subprocess still holds
        the cards) and its row stays active, so retention keeps its files
        and the transfer backend still takes its uploads. ``_drive`` skips
        ``_begin`` (the previous process did it) and ``WorkerExecutor``
        sends no ``assign``: it waits for the worker's reconnect and
        replay, after *last_seq*.
        """
        options = record.options if isinstance(record.options, dict) else {}
        facts = (record.execution_facts
                 if isinstance(record.execution_facts, dict) else {})
        self._workers.adopt(placement.worker_id, record.id,
                            last_seq=last_seq)
        try:
            self._start(record.id, {}, options,
                        queue_key=record.queue_key or placement.devices[0],
                        lane=DEFAULT_LANE, placement=placement,
                        memory_budget=_adopted_budget(facts),
                        graph_hash=(facts.get("graph_hash") if isinstance(
                            facts.get("graph_hash"), str) else None),
                        resume_seq=last_seq)
        except BaseException:
            self._workers.close_run(placement.worker_id, record.id)
            raise
        if placement.memory_budget is not None:
            # Its subprocess still holds the card under its cap (#788): the
            # budget counts as running at once, so a restarted server does
            # not admit onto the card what the adopted run still uses.
            self._ledger.started(record.id)
        adopted = self._runs[record.id]
        adopted.adopted_at = utc_now_iso()
        worker_s3 = facts.get(WORKER_S3_KEY)
        if isinstance(worker_s3, dict):
            adopted.resume_s3 = worker_s3
        stop = facts.get("stop_requested")
        if stop in (STOP_REASON_CANCELLED, STOP_REASON_INTERRUPTED):
            # A Stop the previous process could not deliver: the worker
            # is asked again when it is back (``WorkerExecutor``).
            active = self._runs[record.id]
            active.stop_reason = stop
            active.context.cancel()

    async def scrub_stored_secrets(self) -> int:
        """Remove values an older build persisted. Returns rows changed.

        SECRET values, plus every value of a card whose graph-owned preset
        definition cannot be read (#541), since nothing says which are keys.

        A startup sweep, and the part of #251 that addresses the leak that
        already happened. Scrubbing on write only protects runs submitted
        from here on; retention is count-based with no age component, so a
        key written by a previous version sits in the shared database until
        200 further runs push its row out — which on a quiet install is
        never.

        FINISHED rows only (see ``list_terminal_graph_snapshots``). An active
        row may be one this process is about to promote and re-inject into,
        and blanking it would break the very run the vault exists to serve.
        After ``recover_interrupted`` there are none, which is why
        ``main.py`` orders it second; the guard is what makes the method safe
        to call anywhere regardless.

        Idempotent and cheap on the steady state: a row is only rewritten
        when the scrub actually finds something, so once swept, every later
        boot reads the rows and writes nothing. That standing sweep is also a
        backstop — if any future path ever persists an unscrubbed snapshot,
        the next restart takes the key back out.

        What this does and does not reach, measured rather than assumed
        (SQLite 3.50.4, real schema, explicit checkpoints on both sides):

        - The rows it rewrites come out clean, **because
          ``PRAGMA secure_delete=ON`` is set** (see ``db.py``) -- NOT because
          an UPDATE is self-cleaning. It is not. Blanking a secret shrinks
          the JSON, and a shrinking rewrite of a large TEXT value releases
          overflow pages; without the pragma those go to the freelist with
          the old content intact, in the MAIN file::

              12KB -> 0KB   secure_delete=OFF -> residue in main
              12KB -> 0KB   secure_delete=ON  -> none

          Whether a given scrub frees a page at all depends on where the new
          size lands against a page boundary -- a same-size rewrite rewrites
          the chain in place and happens to leave nothing either way. That is
          incidental. Do not read this sweep as safe without the pragma.
        - Rows retention already PRUNED on an older build are beyond it.
          Their pages were freed before the pragma shipped, so they went to
          the freelist with the key intact and stay that way until a
          ``VACUUM``. The row is gone, so there is nothing here to rewrite.

        That second case is why the docs still tell an operator who ran a
        secret-bearing graph on an earlier version to treat the key as
        disclosed and rotate it. For anything written from here on there is
        no such caveat: the value never reaches the row in the first place,
        and a freed page is zeroed on the way out.
        """
        cleaned = 0
        for run_id, graph in await self.store.list_terminal_graph_snapshots():
            # SECRET params, and every value of a card whose definition in
            # the graph's own presets[] does not parse (#541): that is a fact
            # about the stored graph, so its ordinary settings go too. NOT
            # ``unknown_types_as_secret``, which the submit lanes pass
            # (#537): "unknown" here would mean unknown at THIS boot, so a
            # plugin that failed to load once would have every value of its
            # nodes blanked in every past run, for good.
            scrubbed, secrets = split_graph_secrets(graph)
            if not secrets:
                continue
            # ``scrubbed`` is a copy; ``secrets`` is discarded unread. The
            # values are on their way out, not into memory.
            if await self.store.replace_graph_snapshot(run_id, scrubbed):
                cleaned += 1
        if cleaned:
            logger.warning(
                "blanked SECRET parameter values, and every value of preset "
                "nodes whose embedded definition cannot be read, in %d "
                "stored run snapshot(s) written before they were scrubbed "
                "on write", cleaned)
        return cleaned

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
        for timer in self._one_shot_timers.values():
            timer.cancel()
        self._one_shot_timers.clear()
        # An expiry in flight writes to the database: finish it, or cancel
        # it, before the database closes (#790).
        expiries = list(self._one_shot_expiries)
        for task in expiries:
            task.cancel()
        await asyncio.gather(*expiries, return_exceptions=True)
        await self._retire_queue()
        active = list(self._runs.values())
        for entry in active:
            # First reason wins: a run the user already cancelled is filed as
            # cancelled even if the server goes down before it unwinds. Their
            # intent is the more specific fact, and it came first.
            if entry.stop_reason is None:
                entry.stop_reason = STOP_REASON_INTERRUPTED
            entry.context.cancel()
            # A worker run waiting for its worker (a drop, or adopted after
            # a restart, #793) cannot be told to stop: end it now, with a
            # terminal row, instead of after the grace with none (which the
            # next start would adopt again). Its worker is told to stop it
            # when it next connects (an untracked run).
            if on_a_worker(entry.placement) and self._workers is not None:
                worker_id = entry.placement.worker_id
                self._workers.end_waiting(
                    worker_id, entry.run_id,
                    f"this server is shutting down and worker {worker_id} "
                    "is not connected")
        try:
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
                await self._retire_unstopped_worker_runs(
                    [entry for entry in active if entry.task in pending])
        finally:
            # Belt and braces (#251). ``_retire_queue`` above dropped every
            # waiting run's secrets and each drained task dropped its own, so
            # this should already be empty — but "should" is not a property a
            # credential store gets to rely on, and a service object that
            # outlives its shutdown (tests, an embedded host) must not still
            # be holding keys.
            self._run_secrets.clear()

    async def _retire_unstopped_worker_runs(
            self, entries: list[_ActiveRun]) -> None:
        """Worker runs a graceful shutdown could not stop within its grace
        (#793): their rows end ``interrupted`` now, with a closing frame,
        rather than staying active for the next start to adopt. A graceful
        stop stops worker runs; the hard cancel already told each worker
        to stop it, and a worker that still reports it later is told again
        (an untracked run)."""
        for entry in entries:
            if not on_a_worker(entry.placement):
                continue
            worker_id = entry.placement.worker_id
            try:
                if await self.store.mark_finished(
                        entry.run_id, STATUS_INTERRUPTED,
                        error=(f"this server shut down before the run on "
                               f"worker {worker_id} stopped")):
                    await self.store.append_event(
                        entry.run_id, EVENT_RUN_STOPPED,
                        {"reason": STOP_REASON_INTERRUPTED})
            except Exception:  # noqa: BLE001 -- the next start retires it
                logger.warning("run %s: could not record its end at "
                               "shutdown", entry.run_id, exc_info=True)

    async def _retire_queue(self) -> int:
        """File every waiting run ``interrupted``. Returns how many.

        Both indexes are emptied SYNCHRONOUSLY up front, before the first
        await, so nothing can be promoted, cancelled or double-retired while
        the rows are being written one at a time.
        """
        waiting = list(self._pending_by_id.values())
        self._pending_by_id.clear()
        self._pending.clear()
        self._reservations.clear()
        if not waiting:
            return 0
        logger.info("shutdown: retiring %d queued run(s)", len(waiting))
        for entry in waiting:
            await self._retire_waiting(entry, STATUS_INTERRUPTED,
                                       STOP_REASON_INTERRUPTED)
        return len(waiting)
