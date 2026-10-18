"""The control plane's view of its worker agents (#772).

A worker agent (``cdui agent start``, #771) dials ``/ws/worker`` and says
``hello``; :class:`WorkerRegistry` is where it is then known: its devices,
its heartbeat, the runs it is executing and the connection to reach it.
Enrolled workers (#780) are also rows of the ``workers`` table: the
lifespan loads them at boot (:meth:`WorkerRegistry.load_known`) as
``offline``, so a run can target one and queue until it dials in. A worker
that authenticated with the session token (a same-host agent) is known in
memory only, as before.

Identity: a worker's id IS its name (``[a-z0-9][a-z0-9-]{0,62}``, the
``target.worker`` alphabet, ``local`` and ``any`` reserved). A name is
unique: a second connection under a name REPLACES the first. That makes
the id stable across reconnects, which is what slots
(``(worker_id, device)``, #768) and ``exec_runs.worker_id`` need.

Status: ``online`` -> ``degraded`` (a heartbeat is late) -> ``disconnected``
(the socket closed, or :data:`MISSED_HEARTBEATS` heartbeats in a row
missed). ``offline`` is an enrolled worker that has not connected since
this server started, or a one-shot worker (#790) that left with no run in
flight (it reported its run and exited, as it should). Draining is a separate flag (no new assignments); the API's
``status`` reads ``draining`` for a connected worker that has it, because
that is what decides whether a run can go there, and keeps the
``draining`` flag beside it.

When a worker goes ``disconnected`` its runs are not ended (#783): the run
subprocesses keep running on the worker and the agent spools their frames.
Each run is told (:data:`WORKER_DISCONNECTED` in its inbox; the executor
marks the row ``disconnected``) and keeps its slots. A reconnect under the
same name whose ``run_alive`` lists the run within ``resume_grace_s``
(``CODEFYUI_WORKER_RESUME_GRACE_S``, 300 s) re-attaches it
(:data:`WORKER_RECONNECTED`); the agent replays what the control plane has
not acknowledged and :meth:`WorkerRegistry.route` drops the frames already
seen. After the grace the run ends ``interrupted`` and its slots are freed.
A run that ended while it may still be running on its worker is
remembered: when the worker later reports it finished, the
``run_finished`` goes to :attr:`WorkerRegistry.on_late_result`, which
records it as ``execution_facts.late_result`` without changing the run's
status or its event log. Frames for a run nothing waits for are
acknowledged anyway, so the agent does not keep them forever; a run the
worker still executes that nothing here tracks (and that did not just
finish here) is told to stop at once, so it does not hold a card the
scheduler has handed on.

A connected worker whose node catalog differs from this server's (#781,
``core.worker_handshake.check_catalog``) also reads ``degraded``, with
``node_catalog`` saying which types are missing or extra; ``degraded_reasons``
lists why (``node_catalog``, ``heartbeat``). Unlike a late heartbeat, a
catalog that LACKS types this server has keeps the worker out of
``target: any`` placement (:attr:`Worker.skipped_by_any`), and so does a
catalog the worker never reported; a worker that only has more types
stays eligible. A run on a worker whose catalog differs must use only
types it has (checked at submit for a named target and before ``assign``).
The comparison is redone whenever this server's own catalog changes (a
plugin installed or reloaded here), on the next lookup. A registry built
without a ``node_catalog`` source compares nothing and skips nothing.

A ``hello`` refused at the handshake (another CodefyUI version, a name the
credential does not allow) is remembered per name, in memory
(:meth:`WorkerRegistry.record_rejection`, with the peer address), so the
Workers panel can say why a worker never appeared; a later successful
connection under the name clears it. A refusal of an attempt to take the
name of a CONNECTED worker is marked ``live_owner`` so the panel lists it
as a refused attempt, never as that worker's own refusal.

Runs: :class:`WorkerRun` is one assigned run's inbox. The socket handler
routes every per-run frame into it and ``WorkerExecutor`` reads it; a
worker that goes away puts :data:`WORKER_GONE` in each inbox, which the
executor turns into ``interrupted``. An inbox is not waited on below
:data:`INBOX_HIGH_WATER` queued frames: the socket's read loop also carries
the heartbeats and control frames of every other run, so one run whose
consumer is briefly slow (a stalled database write) must not hold them up.
Live, the agent's own pipe backpressure bounds how fast one run can
produce. A replay after a long disconnect is not bounded that way (#783),
so past the high water mark a warning is logged once per run and the
socket handler stops reading until the inbox drains to half of it
(:meth:`WorkerRegistry.backpressure`).

One-shot workers (#790, ``cdui agent start --one-shot``): ``hello.one_shot``
marks the worker; ``target: any`` never places a run there (only a run
naming it or ``target.pool: "one-shot"`` does, see ``core.scheduling``),
and it takes ONE run: :meth:`WorkerRegistry.claim_one_shot` records it
the moment the run is placed (``Worker.one_shot_claim``), which keeps every
other run off it, across reconnects of the same agent (``instance_id``),
and across a restart of this server (the agent says which run it took in
``hello.one_shot_taken``). Once its ``assign`` went out
(:meth:`WorkerRegistry.one_shot_assigned`) the claim is for good; a run
that ends before that, however it ends, hands it back
(:meth:`release_one_shot`, called when the run gives back its slots).

When a replacing connection arrives while runs are still in flight on the
old one, those runs move to the new connection and wait for its
``run_alive``: a run it lists carries on, one it does not (or every one,
if no ``run_alive`` arrives within a heartbeat interval) ends
``interrupted``. Nothing is lost silently: each outcome is logged.

Instances (#845, ``cdui agent start --one-shot --instance``): one
credential serves several concurrent one-shot agents (a SLURM array's
tasks). Each registers under a worker id of its own,
``<name>.<instance>`` (``worker_protocol.instance_worker_id``;
:attr:`Worker.instance_of` names the credential), and is an ordinary
one-shot worker from then on: its own slots, its own run, its own row in
``GET /api/workers``. What is set on the credential applies to each
instance: its drain state (:meth:`WorkerRegistry.drain`), its device
admission policies (:func:`policy_owner`) and its ``<worker>/<device>``
queue caps (``QueueLimits.for_slot``). Revoking the credential removes
every instance (:meth:`WorkerRegistry.forget_instances`). Instances are
known in memory only, never in the credential's ``workers`` row; at most
:data:`MAX_LEFT_INSTANCES` that left are kept per credential.

After a restart of this server (#793), ``RunService.recover_interrupted``
hands every worker run its previous process left in flight to
:meth:`WorkerRegistry.adopt`: the run waits, detached, exactly as if its
worker's connection had just dropped, with the ``seq`` the previous
process last recorded, so the agent's replay is deduplicated against it.
A run on a worker this registry does not know yet (one that authenticated
with the session token, which is known in memory only) waits on a
placeholder that is not listed; the worker's first connection under that
name takes its runs like any reconnect. Within the resume grace a
``run_alive`` that lists the run re-attaches it; otherwise it ends
``interrupted``, saying that this server restarted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from . import worker_protocol as wp
from .scheduling import LOCAL_WORKER_ID, SlotKey
from .worker_handshake import (
    CATALOG_UNKNOWN,
    CatalogCheck,
    check_catalog,
    verified_fingerprint,
)
from .worker_outputs import DEFAULT_TIMEOUT_S, WorkerOutputs

logger = logging.getLogger(__name__)

STATUS_ONLINE = "online"
STATUS_DEGRADED = "degraded"
STATUS_DISCONNECTED = "disconnected"
STATUS_DRAINING = "draining"
STATUS_OFFLINE = "offline"

#: How the worker's connection authenticated (``Worker.credential``).
CREDENTIAL_SESSION = "session_token"
CREDENTIAL_WORKER = "worker_token"

#: Heartbeats a worker may miss before it is ``disconnected``: 3 x 10 s.
MISSED_HEARTBEATS = 3
#: Queued frames in one run's inbox past which a warning is logged and the
#: worker's socket is read no further until it drains (#783).
INBOX_HIGH_WATER = 10_000
#: How often a paused socket checks whether the inbox drained.
INBOX_DRAIN_POLL_S = 0.05
#: Names whose last handshake refusal is remembered; the oldest is dropped.
MAX_REJECTIONS = 100
#: Instances of one credential that left (#845) kept listed; the oldest
#: is dropped. A SLURM array of hundreds of tasks would otherwise list
#: every finished task until this server restarts.
MAX_LEFT_INSTANCES = 50
#: ``degraded_reasons`` entries.
DEGRADED_NODE_CATALOG = "node_catalog"
DEGRADED_HEARTBEAT = "heartbeat"

#: Seconds a disconnected worker's runs wait for it (#783).
DEFAULT_RESUME_GRACE_S = 300.0
#: An orphan run's frames (see :meth:`WorkerRegistry.route`) are acked
#: every this many, and on its ``run_finished``.
ORPHAN_ACK_EVERY = 64
#: Runs remembered for a late ``run_finished``; the oldest is forgotten.
LATE_RUNS_MAX = 1000

#: Put in a run's inbox when its worker is gone: ``(WORKER_GONE, reason)``.
WORKER_GONE = object()
#: Put in a run's inbox when its worker's connection dropped and the run
#: waits for it to come back: ``(WORKER_DISCONNECTED, reason)`` (#783).
WORKER_DISCONNECTED = object()
#: Put in a run's inbox when a reconnected worker confirmed it:
#: ``(WORKER_RECONNECTED,)`` (#783).
WORKER_RECONNECTED = object()
#: Seconds a run that finished here is not "untracked": its worker may
#: still list it while the subprocess exits (#783).
RECENTLY_CLOSED_S = 60.0


Send = Callable[[Any], Awaitable[None]]


@dataclass(eq=False)
class WorkerRun:
    """One run assigned to a worker: its frame inbox and its last ``seq``."""

    run_id: str
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    #: The highest ``seq`` delivered; anything at or below it is a
    #: duplicate (#783 replays after reconnects).
    last_seq: int = 0
    #: True while a replacing connection has not confirmed it (run_alive).
    awaiting_alive: bool = False
    gone: bool = False
    high_water_logged: bool = False
    #: When its worker's connection dropped (monotonic, and wall clock for
    #: the API), while it waits for the worker to come back (#783).
    detached_at: float | None = None
    detached_wall: float | None = None
    #: Taken over from a previous process of this server (#793): it waits
    #: for its worker since this server restarted, not since a drop.
    adopted: bool = False


@dataclass(eq=False)
class Worker:
    """One connected (or recently connected) worker agent."""

    id: str
    name: str
    version: str
    #: Its environment fingerprint digest (#781), or None.
    fingerprint: str | None
    platform: dict[str, str]
    mounts: dict[str, str]
    devices: list[wp.DeviceReport]
    send: Send
    close: Callable[[], Awaitable[None]]
    heartbeat_s: float
    status: str = STATUS_ONLINE
    draining: bool = False
    #: Its ``hello.machine_id`` equals this process's: the same machine
    #: (#773). Decided by machine identity, never by the peer address.
    same_host: bool = False
    #: Its ``hello.machine_id`` (a digest), or None: two agents on one
    #: machine share its CPU and Apple GPU (#842, :func:`worker_host`).
    machine_id: str | None = None
    #: Worker device -> this process's device for each card the two share
    #: (#773): ``cpu``, ``mps:0``, and each CUDA card whose UUID matches
    #: one here. Empty for another machine. A description (the API's
    #: ``shared_with``); the slots of one card are grouped by
    #: ``WorkerRegistry.card_slots`` (#842).
    shared: dict[str, str] = field(default_factory=dict)
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.monotonic)
    #: Epoch seconds of its last heartbeat (or, for a worker known from the
    #: table, when it was last seen); None for one never seen.
    last_heartbeat_at: float | None = field(default_factory=time.time)
    running: dict[str, WorkerRun] = field(default_factory=dict)
    #: When runs inherited from a replaced connection stop waiting for
    #: ``run_alive`` (monotonic), or None.
    alive_deadline: float | None = None
    #: It has a row in the ``workers`` table (#780): its credential can be
    #: revoked and it survives a restart of this server.
    enrolled: bool = False
    #: How its current (or last) connection authenticated:
    #: :data:`CREDENTIAL_WORKER`, :data:`CREDENTIAL_SESSION`, or None for
    #: an enrolled worker that has not connected since this server started.
    credential: str | None = None
    #: Its current (or last) connection reached this server over plain
    #: ``ws://`` from another machine (#786): the worker token and the
    #: secrets in its assignments crossed the network unencrypted. False for
    #: ``wss`` (as uvicorn reports the scheme) and for a loopback peer: a
    #: same-host agent, a TLS proxy on this machine, or an SSH tunnel's
    #: local end.
    plain_http: bool = False
    #: The fingerprint document behind :attr:`fingerprint` (#781): what a
    #: run on this worker records as its environment.
    fingerprint_document: dict[str, Any] | None = None
    #: ``hello.node_catalog_hash`` and ``hello.node_types`` (#781).
    node_catalog_hash: str | None = None
    node_types: tuple[str, ...] = ()
    #: The optional protocol features its ``hello`` listed (#781); the
    #: transfer storage backend goes only to one naming ``transfer`` (#785).
    capabilities: tuple[str, ...] = ()
    #: Its catalog against this server's, as last compared (#781).
    catalog: CatalogCheck = field(default_factory=CatalogCheck)
    #: The server catalog hash :attr:`catalog` was compared with.
    catalog_against: str | None = None
    #: Its registry compares catalogs (it has a ``node_catalog`` source).
    catalog_checked: bool = False
    #: A ``cdui agent start --one-shot`` agent (#790): ``any`` never
    #: places a run here, and it takes exactly one run.
    one_shot: bool = False
    #: The one run it takes (``--run-id``), or None.
    one_shot_run_id: str | None = None
    #: The run it took (set when the run is placed here), or None. Carried
    #: across a reconnect of the same agent; a new agent starts without.
    one_shot_claim: str | None = None
    #: The claimed run's ``assign`` went out: the agent has it, so the claim
    #: is not handed back when the run ends.
    one_shot_assigned: bool = False
    #: ``hello.instance_id``: which agent process holds the name (#790).
    instance_id: str | None = None
    #: The credential's name when this is one of its instances (#845,
    #: ``<instance_of>.<instance>``), else None.
    instance_of: str | None = None
    #: ``hello.instance`` (#845), else None.
    instance: str | None = None
    #: On a worker (or placeholder) holding a run adopted after a restart
    #: (#793, #845): the hashed ``instance_id`` of the agent process that
    #: ran it (``worker_handshake.instance_id_hash``). Only that process
    #: may upload the run's files, or take a one-shot id back
    #: (``worker_handshake.same_agent``).
    instance_hash: str | None = None

    @property
    def online(self) -> bool:
        return self.status not in (STATUS_DISCONNECTED, STATUS_OFFLINE)

    @property
    def one_shot_available(self) -> bool:
        """A one-shot worker that has not taken its run yet (#790)."""
        return self.one_shot and self.one_shot_claim is None

    def one_shot_takes(self, run_id: str) -> bool:
        """Whether a run may be placed here as far as one-shot rules go:
        any run on a regular worker; on a one-shot worker, only while it
        has not taken one, and only its ``--run-id`` when it named one."""
        if not self.one_shot:
            return True
        return self.one_shot_claim is None and (
            self.one_shot_run_id is None or self.one_shot_run_id == run_id)

    @property
    def skipped_by_any(self) -> bool:
        """``target: any`` does not place runs here (#781): its catalog lacks
        node types this server has, or it reported none."""
        return self.catalog_checked and (
            self.catalog.lacks_types
            or self.catalog.status == CATALOG_UNKNOWN)

    def degraded_reasons(self) -> list[str]:
        if not self.online:
            return []
        reasons = []
        if self.catalog.degraded:
            reasons.append(DEGRADED_NODE_CATALOG)
        if self.status == STATUS_DEGRADED:
            reasons.append(DEGRADED_HEARTBEAT)
        return reasons

    def api_status(self) -> str:
        """``status`` as the API reports it: ``draining`` first (it decides
        whether a run can go there), then ``degraded`` for either reason."""
        if self.online and self.draining:
            return STATUS_DRAINING
        if self.degraded_reasons():
            return STATUS_DEGRADED
        return self.status

    @property
    def accepts_runs(self) -> bool:
        return self.online and not self.draining

    def device_values(self) -> list[str]:
        """The devices this worker offers, canonical (``mps`` is ``mps:0``);
        a report this side cannot parse is left out."""
        values = (_canonical_or_none(d.value) for d in self.devices)
        return [v for v in values if v is not None]

    def offers(self, device: str) -> bool:
        return _canonical_or_none(device) in self.device_values()

    def describe(self) -> dict[str, Any]:
        """The ``GET /api/workers`` entry."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.api_status(),
            "draining": self.draining,
            "degraded_reasons": self.degraded_reasons(),
            "node_catalog": self.catalog.describe(),
            "same_host": self.same_host,
            "version": self.version,
            "fingerprint_digest": self.fingerprint,
            "platform": dict(self.platform),
            "mounts": dict(self.mounts),
            "devices": [{
                "value": d.value, "kind": d.kind, "index": d.index,
                "name": d.name,
                "memory": None if d.source == "none" else {
                    "total_bytes": d.total_bytes,
                    "free_bytes": d.free_bytes, "source": d.source},
            } for d in self.devices],
            "running": sorted(self.running),
            "connected_at": (None if self.status == STATUS_OFFLINE
                             and not self.one_shot else self.connected_at),
            "last_heartbeat_at": self.last_heartbeat_at,
            "enrolled": self.enrolled,
            "credential": self.credential,
            "plain_http": self.plain_http,
            # #790: None for a regular worker.
            "one_shot": None if not self.one_shot else {
                "run_id": self.one_shot_run_id,
                "took": self.one_shot_claim,
            },
            # #845: the credential this instance shares, and its own part
            # of the id; both None for a worker that is not an instance.
            "instance_of": self.instance_of,
            "instance": self.instance,
        }


def _is_cuda(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def policy_owner(worker_id: str) -> str:
    """Whose device admission policies (#788) *worker_id* uses: its own,
    or for an instance (#845, ``<name>.<instance>``) its credential's, so
    every instance of a SLURM array admits as the credential says."""
    return wp.instance_parts(worker_id)[0]


def _canonical_or_none(device: str) -> str | None:
    try:
        return wp.canonical_device(device)
    except ValueError:
        return None


@dataclass
class _CardIndex:
    """Which slots are one physical device (#842)."""

    #: Slot -> every slot on its device, for a device several slots name.
    slots: dict[SlotKey, tuple[SlotKey, ...]] = field(default_factory=dict)
    #: Slot -> its device identity (:func:`device_identity`).
    identity: dict[SlotKey, str] = field(default_factory=dict)


#: The machine part of this server's own devices' identities, and of a
#: same-host worker's (``Worker.same_host``): never a reported machine id,
#: so an agent elsewhere that claims this server's id joins nothing here.
HOST_LOCAL = "local"


def device_identity(value: str, uuid: str | None, *,
                    host: str | None) -> str | None:
    """What makes *value* one physical device across processes (#773,
    #842): ``gpu:<uuid>`` for a CUDA card with a UUID (unique on every
    machine), and ``host:<host>:cpu`` / ``host:<host>:mps:0`` for the CPU
    and the Apple GPU of machine *host* (one of each per machine): this
    server's (:data:`HOST_LOCAL`), or the machine id a worker reported, so
    two agents on one remote machine share them too. None for anything
    else: its slot is its own."""
    if value.startswith("cuda"):
        return f"gpu:{str(uuid).lower()}" if uuid else None
    if host and (value == "cpu" or value.startswith("mps")):
        return f"host:{host}:{value}"
    return None


def worker_host(worker: Any) -> str | None:
    """The machine part of *worker*'s CPU and Apple GPU identities:
    :data:`HOST_LOCAL` on this server's machine, else the machine id its
    ``hello`` reported (None: unknown, its CPU is its own)."""
    if getattr(worker, "same_host", False):
        return HOST_LOCAL
    machine = getattr(worker, "machine_id", None)
    return f"m-{machine}" if machine else None


def _device_identities(devices: list[wp.DeviceReport]) -> list[tuple]:
    """What placement reads off a device list: which devices, not their
    memory figures."""
    return [(d.value, d.kind, d.index, d.uuid) for d in devices]


class WorkerRegistry:
    """Every worker this control plane knows, by name. Event-loop only.

    *local_devices* returns this process's devices as ``DeviceFact``-like
    objects (``value``, ``uuid``); it is called once, off the event loop
    at startup (:meth:`probe_local_devices`), or else on first use, and is
    what workers' devices are matched against (#773, #842). Without it no
    device of this process is shared with a worker.
    """

    def __init__(self, *, heartbeat_s: float = wp.DEFAULT_HEARTBEAT_S,
                 grace_s: float = wp.DEFAULT_GRACE_S,
                 missed_heartbeats: int = MISSED_HEARTBEATS,
                 clock: Callable[[], float] = time.monotonic,
                 local_devices: Callable[[], list[Any]] | None = None,
                 resume_grace_s: float = DEFAULT_RESUME_GRACE_S,
                 node_catalog: Callable[[], Any] | None = None,
                 output_fetch_timeout_s: float = DEFAULT_TIMEOUT_S,
                 probe_off_loop: bool = False) -> None:
        self._local_devices_source = local_devices
        #: Reads of worker runs' recorded outputs in flight (#789).
        self.outputs = WorkerOutputs(self, timeout_s=output_fetch_timeout_s)
        #: Never list this process's devices on the event loop: the probe
        #: runs in a thread (:meth:`probe_local_devices`), and until it
        #: has, this process has no device to share. The server sets it; a
        #: registry built without it probes on first use.
        self._probe_off_loop = probe_off_loop
        self._probe_task: asyncio.Task | None = None
        #: This server's ``node_registry.node_catalog`` (#781); without it
        #: no worker's catalog is compared (every one reads ``unknown``).
        self._node_catalog = node_catalog
        #: Name -> ``{"name", "reason", "at"}``: each name's last refused
        #: handshake, oldest first (#781).
        self._rejections: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._local_devices_cache: list[Any] | None = None
        #: ``(worker, device, uuid)`` already logged as not shared.
        self._unshared_logged: set[tuple[str, str, str | None]] = set()
        #: The physical devices several slots name (#842), built on first
        #: use and dropped by :meth:`_cards_changed`; see :meth:`card_slots`.
        self._cards: _CardIndex | None = None
        #: ``(identity, slots)`` already logged as one device.
        self._cards_logged: set[tuple[str, tuple[SlotKey, ...]]] = set()
        #: Run id -> ``{slot: device identity}`` of each run adopted after a
        #: restart (#793), from its ``worker_resume``, until the run ends or
        #: its worker reports its own devices; see :meth:`remember_cards`.
        self._adopted_cards: dict[str, dict[SlotKey, str]] = {}
        self.heartbeat_s = heartbeat_s
        self.grace_s = grace_s
        self.missed_heartbeats = missed_heartbeats
        #: Seconds a disconnected worker's runs wait for it; 0: none (#783).
        self.resume_grace_s = max(0.0, float(resume_grace_s))
        #: ``(worker name, run id)`` -> last seq seen, for each run that
        #: ended here while it may still be running on its worker (#783).
        self._late: dict[tuple[str, str], int] = {}
        #: Run id -> when its worker reported it finished (monotonic), for
        #: :data:`RECENTLY_CLOSED_S`.
        self._recently_closed: dict[str, float] = {}
        #: Awaited as ``on_late_result(worker_name, run_id, run_finished)``
        #: when such a run reports its end (``RunService`` records it).
        self.on_late_result: Callable[[str, str, Any], Awaitable[Any]] | None \
            = None
        self._clock = clock
        self._workers: dict[str, Worker] = {}
        #: Credential name -> the ids of its instances (#845) this registry
        #: lists, oldest connection first (a dict as an ordered set), so
        #: nothing per connection scans every worker.
        self._instances: dict[str, dict[str, None]] = {}
        #: ``(worker id, canonical device)`` -> ``shared`` (#788); a pair
        #: with no entry is ``exclusive``. Kept by name, apart from the
        #: ``Worker`` objects, so a reconnect (which builds a new one) and a
        #: same-host worker known only in memory both keep theirs.
        self._policies: dict[tuple[str, str], str] = {}
        #: ``worker id -> bool``: whether a waiting run with a memory budget
        #: may go to that worker (#788). Set by ``RunService``; a heartbeat
        #: with new memory figures reschedules only when it says so. None
        #: (no run service): every such heartbeat reschedules.
        self.memory_waiting: Callable[[str], bool] | None = None
        #: Name -> an unlisted, offline placeholder holding the runs adopted
        #: after a restart for a worker not known here yet (#793).
        self._adopted: dict[str, Worker] = {}
        #: Run id -> its adopted run, until its executor claims it
        #: (:meth:`claim_adopted`), which may be after it already ended.
        self._unclaimed: dict[str, WorkerRun] = {}
        self._listeners: list[Callable[[], None]] = []
        self._watchdog: asyncio.Task | None = None
        #: Fire-and-forget closes, kept so they are not collected mid-flight
        #: and their failures are logged.
        self._tasks: set[asyncio.Task] = set()

    # ── queries ───────────────────────────────────────────────────────────

    def get(self, worker_id: str) -> Worker | None:
        worker = self._workers.get(worker_id)
        if worker is not None:
            self._check_catalog(worker)
        return worker

    def workers(self) -> list[Worker]:
        workers = sorted(self._workers.values(), key=lambda w: w.name)
        for worker in workers:
            self._check_catalog(worker)
        return workers

    # ── node catalogs and refusals (#781) ────────────────────────────────

    def _control_catalog(self) -> Any:
        if self._node_catalog is None:
            return None
        try:
            return self._node_catalog()
        except Exception:  # noqa: BLE001 -- a check degrades, never fails
            logger.exception("could not compute this server's node catalog")
            return None

    def _check_catalog(self, worker: Worker) -> None:
        """Compare *worker*'s catalog with this server's when either changed.

        Cheap when nothing did: the server's catalog is cached per node
        registry generation, and the comparison only reruns for a new hash.
        """
        if not worker.node_catalog_hash:
            return
        control = self._control_catalog()
        if control is None or control.hash == worker.catalog_against:
            return
        before = worker.catalog.status
        worker.catalog = check_catalog(control, worker.node_catalog_hash,
                                       worker.node_types)
        worker.catalog_against = control.hash
        if worker.catalog.degraded:
            logger.warning(
                "worker %s is degraded: its node catalog differs from this "
                "server's (missing %d type(s): %s; extra %d)", worker.name,
                len(worker.catalog.missing),
                ", ".join(worker.catalog.missing[:10]) or "-",
                len(worker.catalog.extra))
        elif before != worker.catalog.status:
            logger.info("worker %s: node catalog matches this server's",
                        worker.name)

    def record_rejection(self, name: str, reason: str, *,
                         peer: str | None = None) -> None:
        """Remember why a ``hello`` under *name* was refused (in memory).

        ``live_owner`` says a connected worker holds the name: the refusal
        was someone else's attempt, not that worker's own (#781).
        """
        name = str(name)[:63]
        owner = self._workers.get(name)
        self._rejections.pop(name, None)
        self._rejections[name] = {
            "name": name, "reason": reason, "at": time.time(), "peer": peer,
            "live_owner": owner is not None and owner.online}
        while len(self._rejections) > MAX_REJECTIONS:
            self._rejections.popitem(last=False)

    def rejections(self) -> list[dict[str, Any]]:
        """Each name's last refused handshake, newest first."""
        return [dict(entry) for entry in reversed(self._rejections.values())]

    def run_state(self, worker_id: str, run_id: str) -> WorkerRun | None:
        """The run *worker_id* is executing as *run_id*, or None."""
        holder = self._holder(worker_id, run_id)
        return None if holder is None else holder.running[run_id]

    def _holder(self, worker_id: str, run_id: str) -> Worker | None:
        """The worker (or the unlisted placeholder of runs adopted after a
        restart, #793) whose ``running`` holds *run_id*. Both are looked
        in: a worker of the name may become known (enrolled) while the
        placeholder still holds runs."""
        for holder in (self._workers.get(worker_id),
                       self._adopted.get(worker_id)):
            if holder is not None and run_id in holder.running:
                return holder
        return None

    # ── this process's devices (#773) ────────────────────────────────────

    async def probe_local_devices(self) -> None:
        """List this process's devices once, in a thread (the probe imports
        torch and reads NVML). Awaited where a worker could share a card
        with this server: ``/ws/worker`` before registering a same-host
        worker (after ``welcome``), and ``RunService`` before adopting runs
        after a restart, so neither is matched against a list that is not
        there yet. Anything else that needs it first starts it in the
        background (:meth:`_local_devices`). This process's devices do not
        change while it runs, so the list is never refreshed; matching is
        redone when it lands."""
        if self._local_devices_cache is not None:
            return
        if self._local_devices_source is None:
            self._local_devices_cache = []  # nothing to list, nothing to wait
            return
        if self._probe_task is None:
            self._probe_task = asyncio.ensure_future(self._probe())
        await asyncio.shield(self._probe_task)

    async def _probe(self) -> None:
        source = self._local_devices_source
        try:
            devices = list(await asyncio.to_thread(source)) if source else []
        except Exception:  # noqa: BLE001 -- sharing degrades, never fails
            logger.exception("could not list this server's devices; no "
                             "device is shared with workers")
            devices = []
        self._local_devices_cache = devices
        for worker in self._workers.values():
            worker.shared = self._shared_devices(worker)
        self._cards_changed()
        self._changed()

    def _local_devices(self) -> list[Any]:
        if self._local_devices_cache is None:
            if self._probe_off_loop:
                if self._probe_task is None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:  # no running loop: nothing to do
                        return []
                    self._probe_task = loop.create_task(self._probe())
                return []
            source = self._local_devices_source
            try:
                self._local_devices_cache = list(source()) if source else []
            except Exception:  # noqa: BLE001 -- sharing degrades, never fails
                logger.exception("could not list this server's devices; no "
                                 "device is shared with workers")
                self._local_devices_cache = []
        return self._local_devices_cache

    def _local_identities(self, *, cards: bool = True) -> dict[str, str]:
        """Device identity (:func:`device_identity`) -> this process's
        device: its CPU and Apple GPU always (no probe needed), and with
        *cards* every CUDA card with a UUID (from the probe). The one place
        a device of this process is matched to a worker's (#773, #842)."""
        identities = {
            device_identity(value, None, host=HOST_LOCAL): value
            for value in ("cpu", "mps:0")}
        if not cards:
            return identities
        for fact in self._local_devices():
            value = _canonical_or_none(getattr(fact, "value", ""))
            if value is None or not value.startswith("cuda"):
                continue
            identity = device_identity(value, getattr(fact, "uuid", None),
                                       host=HOST_LOCAL)
            if identity is not None:
                identities.setdefault(identity, value)
        return identities

    def _shared_devices(self, worker: Worker) -> dict[str, str]:
        """Worker device -> this process's device, for the devices a
        same-host worker shares with this process (``shared_with`` in
        ``GET /api/workers``): its ``cpu`` and ``mps:0`` always, and each
        CUDA card whose UUID matches one here, whatever its number on
        either side. A CUDA card without a UUID, or whose UUID matches
        nothing here, is logged once (once this process's cards are
        known). Description only: which runs share a device is the card
        index's (:meth:`card_slots`)."""
        if not worker.same_host:
            return {}
        cards = any(_is_cuda(r.value) for r in worker.devices)
        local = self._local_identities(cards=cards)
        probed = self._local_devices_cache is not None
        shared: dict[str, str] = {}
        for report in worker.devices:
            value = _canonical_or_none(report.value)
            if value is None:
                continue
            identity = device_identity(value, report.uuid, host=HOST_LOCAL)
            match = None if identity is None else local.get(identity)
            if match is not None:
                shared[value] = match
                continue
            if not value.startswith("cuda") or not probed:
                continue
            key = (worker.name, value, report.uuid)
            if key not in self._unshared_logged:
                self._unshared_logged.add(key)
                logger.warning(
                    "worker %s runs on this machine, but its %s (uuid %s) "
                    "matches no card of this server; it keeps a slot of "
                    "its own", worker.name, value, report.uuid or "unknown")
        return shared

    # ── physical cards (#773, #842) ──────────────────────────────────────

    def _card_index(self) -> "_CardIndex":
        if self._cards is None:
            self._cards = self._build_card_index()
        return self._cards

    def _build_card_index(self) -> "_CardIndex":
        """Group every slot by the physical device it is
        (:func:`device_identity`): each worker's devices (connected or not:
        a disconnected worker's runs keep their slots while they wait for
        it, #783), the slots of runs adopted after a restart whose worker
        is not known yet (#793, :meth:`remember_cards`), and this
        process's devices. Rebuilt only when one of those changes
        (:meth:`_cards_changed`)."""
        # One identity per slot: a worker's own report wins over what an
        # adopted run recorded for the same slot.
        identity_of: dict[SlotKey, str] = {}
        for cards in self._adopted_cards.values():
            identity_of.update(cards)
        for worker in sorted(self._workers.values(), key=lambda w: w.name):
            host = worker_host(worker)
            for report in worker.devices:
                value = _canonical_or_none(report.value)
                if value is None:
                    continue
                identity = device_identity(value, report.uuid, host=host)
                slot = (worker.id, value)
                if worker.status == STATUS_OFFLINE and slot in identity_of:
                    # Known from the table only (it has not connected
                    # since this server started): what an adopted run
                    # recorded is fresher than the stored report.
                    continue
                if identity is None:
                    identity_of.pop(slot, None)
                else:
                    identity_of[slot] = identity
        index = _CardIndex()
        if not identity_of:
            # No worker names a device: nothing to group, and this
            # process's devices are not probed for it.
            return index
        members: dict[str, list[SlotKey]] = {}
        # This server's cards are listed only where one could be shared: a
        # same-host worker, or a run adopted after a restart. Once listed,
        # every worker is matched against them.
        cards = any(i.startswith("gpu:") for i in identity_of.values()) and (
            self._local_devices_cache is not None or bool(self._adopted_cards)
            or any(w.same_host for w in self._workers.values()))
        for identity, value in self._local_identities(cards=cards).items():
            members[identity] = [(LOCAL_WORKER_ID, value)]
        for slot, identity in identity_of.items():
            members.setdefault(identity, []).append(slot)
        for identity, slots in members.items():
            for slot in slots:
                index.identity[slot] = identity
            if len(slots) < 2:
                continue
            group = tuple(slots)
            for slot in group:
                index.slots[slot] = group
            key = (identity, group)
            if key not in self._cards_logged:
                self._cards_logged.add(key)
                logger.info(
                    "one physical device (%s) is offered as %s; runs on it "
                    "count against one budget", identity,
                    ", ".join(f"{w}/{d}" for w, d in group))
        return index

    def _cards_changed(self) -> None:
        """The devices behind the card index changed (a worker connected,
        was loaded or forgotten, reported other devices, or a run was
        adopted): rebuild it on next use. A heartbeat's memory figures or a
        status change do not touch it."""
        self._cards = None

    def card_slots(self, slot: SlotKey) -> tuple[SlotKey, ...]:
        """Every slot on the physical device *slot* is on (#842), *slot*
        included; just ``(slot,)`` for a device no other slot names.

        Two workers that offer one CUDA card report the same UUID, whatever
        number each gives it (another ``CUDA_VISIBLE_DEVICES``, a SLURM
        allocation); their slots ``(a, cuda:0)`` and ``(b, cuda:1)`` are
        one card, and so is this process's ``("local", cuda:N)`` when it
        has the card. A same-host worker's ``cpu`` and ``mps:0`` are this
        process's (#773), and two agents that report one machine id share
        theirs. The scheduler counts the runs on every slot of a
        card against each of them (``scheduling.CardBudget``). A CUDA card
        reported without a UUID keeps a slot of its own.
        """
        return self._card_index().slots.get(slot, (slot,))

    def card_identity(self, slot: SlotKey) -> str | None:
        """The physical identity of *slot*'s device (``gpu:<uuid>``,
        ``host:cpu``, ...), or None. What ``execution_facts.worker_resume``
        records beside a run's slots, so a restarted server groups an
        adopted run's slots before its worker is back (#793)."""
        return self._card_index().identity.get(slot)

    def remember_cards(self, run_id: str,
                       cards: Mapping[SlotKey, str]) -> None:
        """The identities of adopted run *run_id*'s slots (#793), from its
        ``worker_resume``: they group its slots with the other offerers of
        the device while its worker is away. Dropped when the run ends,
        when its worker connects (its own report replaces them) and when
        the worker is stopped or forgotten."""
        if cards:
            self._adopted_cards[run_id] = dict(cards)
            self._cards_changed()

    def _forget_adopted_cards(self, *, run_id: str | None = None,
                              worker_id: str | None = None) -> None:
        dropped = False
        for rid in list(self._adopted_cards):
            cards = self._adopted_cards[rid]
            if rid == run_id:
                del self._adopted_cards[rid]
                dropped = True
            elif worker_id is not None and any(
                    slot[0] == worker_id for slot in cards):
                kept = {s: i for s, i in cards.items() if s[0] != worker_id}
                if kept:
                    self._adopted_cards[rid] = kept
                else:
                    del self._adopted_cards[rid]
                dropped = True
        if dropped:
            self._cards_changed()

    def card_peers(self, worker_id: str,
                   device: str) -> tuple[tuple[str, str], ...]:
        """The other ``(worker id, device)`` pairs that offer the physical
        device *worker_id*'s *device* is (#842); ``local`` is this process.
        Empty for a device nobody else offers."""
        value = _canonical_or_none(device)
        if value is None:
            return ()
        slot = (worker_id, value)
        return tuple(s for s in self.card_slots(slot) if s != slot)

    def slot_online(self, slot: SlotKey) -> bool:
        """Whether *slot*'s offerer can take a run now: this process
        always, a worker while it is connected."""
        if slot[0] == LOCAL_WORKER_ID:
            return True
        worker = self._workers.get(slot[0])
        return worker is not None and worker.online

    # ── device admission policies (#788) ─────────────────────────────────

    def load_policies(self, rows: Any) -> None:
        """Load stored ``(worker, device, policy)`` rows
        (``WorkerStore.list_device_policies``). An unknown policy or a
        device this side cannot parse is skipped (logged)."""
        from .memory_admission import POLICIES

        for worker_id, device, policy in rows:
            canonical = _canonical_or_none(device)
            if canonical is None or policy not in POLICIES:
                logger.warning("ignoring stored policy %r for %s on %s",
                               policy, device, worker_id)
                continue
            self._policies[(worker_id, canonical)] = policy

    def device_policy(self, worker_id: str, device: str) -> str:
        """``exclusive`` (the default) or ``shared`` for *worker_id*'s
        *device*, as that worker numbers it."""
        from .memory_admission import POLICY_EXCLUSIVE

        canonical = _canonical_or_none(device)
        if canonical is None:
            return POLICY_EXCLUSIVE
        return self._policies.get((policy_owner(worker_id), canonical),
                                  POLICY_EXCLUSIVE)

    def set_device_policy(self, worker_id: str, device: str,
                          policy: str) -> None:
        """Set a policy (the caller stored it) and reschedule: a card that
        became shared may take waiting runs now."""
        from .memory_admission import POLICY_EXCLUSIVE

        canonical = wp.canonical_device(device)
        worker_id = policy_owner(worker_id)
        if policy == POLICY_EXCLUSIVE:
            self._policies.pop((worker_id, canonical), None)
        else:
            self._policies[(worker_id, canonical)] = policy
        logger.info("worker %s: %s is now %s", worker_id, canonical, policy)
        self._changed()

    def _shares_by_memory(self, worker: Worker) -> bool:
        """Whether one of *worker*'s cards is set to ``shared``."""
        from .memory_admission import POLICY_SHARED

        owner = policy_owner(worker.id)
        return any(key[0] == owner and policy == POLICY_SHARED
                   for key, policy in self._policies.items())

    def forget_policies(self, worker_id: str) -> None:
        for key in [k for k in self._policies if k[0] == worker_id]:
            del self._policies[key]

    # ── change notification ───────────────────────────────────────────────

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Call *listener* (sync) whenever capacity may have changed."""
        self._listeners.append(listener)

    def _changed(self) -> None:
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:  # noqa: BLE001 -- a listener must not break us
                logger.exception("a worker registry listener failed")

    # ── connections ───────────────────────────────────────────────────────

    def connect(self, hello: wp.Hello, *, send: Send,
                close: Callable[[], Awaitable[None]],
                same_host: bool = False,
                credential: str | None = None,
                enrolled: bool = False,
                plain_http: bool = False) -> tuple[Worker, Worker | None]:
        """Register *hello*'s worker. Returns ``(worker, replaced)``.

        *same_host*: the worker's machine identity equals this process's
        (``ws_worker``), so the cards it shares with this process (see
        :meth:`_shared_devices`) share their slots. *credential* says how
        the connection authenticated; *enrolled* that the worker has a
        ``workers`` row (a known worker keeps the flag either way).
        *plain_http*: the connection is unencrypted from another machine.

        A worker already registered under the name is replaced: its runs
        move to the new connection and wait for ``run_alive``, and the
        caller closes the old socket (``replaced.close``).
        """
        old = self._workers.get(hello.name)
        placeholder = self._adopted.pop(hello.name, None)
        if placeholder is not None:
            # Runs adopted after a restart (#793) for a worker first seen
            # now: they move to it like a reconnect's.
            if old is None:
                old = placeholder
            else:
                # The name became known (enrolled) during the grace.
                old.running.update(placeholder.running)
        document, digest = verified_fingerprint(hello.fingerprint,
                                                worker=hello.name)
        worker = Worker(
            id=hello.name, name=hello.name, version=hello.version,
            fingerprint=digest, platform=dict(hello.platform),
            mounts=dict(hello.mounts), devices=list(hello.devices),
            send=send, close=close, heartbeat_s=self.heartbeat_s,
            last_heartbeat=self._clock(), same_host=same_host,
            credential=credential, plain_http=plain_http,
            enrolled=enrolled or (old is not None and old.enrolled),
            fingerprint_document=document,
            node_catalog_hash=hello.node_catalog_hash,
            node_types=tuple(hello.node_types),
            capabilities=tuple(hello.capabilities),
            catalog_checked=self._node_catalog is not None,
            one_shot=hello.one_shot, one_shot_run_id=hello.one_shot_run_id,
            instance_id=hello.instance_id, machine_id=hello.machine_id)
        if hello.instance is not None:
            # ``ws_worker`` registered it under ``<name>.<instance>`` (#845).
            worker.instance_of = wp.instance_parts(hello.name)[0]
            worker.instance = hello.instance
        if old is not None and old.one_shot and hello.one_shot \
                and old.instance_id == hello.instance_id:
            # The same one-shot agent reconnecting (#790): the run it took
            # is still the one it took.
            worker.one_shot_claim = old.one_shot_claim
            worker.one_shot_assigned = old.one_shot_assigned
        if hello.one_shot and hello.one_shot_taken is not None:
            # The agent says it already took a run (#790): after a restart
            # of this server nothing else remembers that.
            worker.one_shot_claim = hello.one_shot_taken
            worker.one_shot_assigned = True
        worker.shared = self._shared_devices(worker)
        self._rejections.pop(hello.name, None)
        self._check_catalog(worker)
        replaced = None
        base = (None if worker.instance_of is None
                else self._workers.get(worker.instance_of))
        if base is not None:
            # An instance takes its credential's drain state (#845): draining
            # the credential drains every instance (:meth:`drain`).
            worker.draining = base.draining
        if old is not None:
            worker.draining = worker.draining or old.draining
            if old.online:
                replaced = old
                old.status = STATUS_DISCONNECTED
                # Its socket closes: nothing it was asked will be answered.
                self.outputs.worker_gone(hello.name, "a new connection "
                                                     "replaced it")
            # A replaced connection's runs, or a disconnected worker's runs
            # still inside their resume grace (#783), move to this one.
            if old.running:
                for run in old.running.values():
                    run.awaiting_alive = True
                worker.running = old.running
                old.running = {}
                worker.alive_deadline = self._clock() + self.heartbeat_s
                logger.warning(
                    "worker %s reconnected with %d run(s) in flight; waiting "
                    "for run_alive", hello.name, len(worker.running))
            elif replaced is not None:
                logger.warning("worker %s: a new connection replaced the "
                               "old one", hello.name)
        self._workers[hello.name] = worker
        if worker.instance_of is not None:
            ids = self._instances.setdefault(worker.instance_of, {})
            ids.pop(worker.id, None)  # its newest connection goes last
            ids[worker.id] = None
            self._prune_left_instances(worker.instance_of)
        self._forget_adopted_cards(worker_id=worker.id)
        self._cards_changed()
        self._ensure_watchdog()
        logger.info("worker %s connected (%s, devices %s)", worker.name,
                    worker.version, worker.device_values())
        self._changed()
        return worker, replaced

    # ── instances (#845) ──────────────────────────────────────────────────

    def instances_of(self, name: str) -> list[Worker]:
        """Every instance of *name*'s credential this registry lists,
        oldest connection first."""
        return [self._workers[i] for i in self._instances.get(name, ())]

    def _unindex(self, worker: Worker) -> None:
        if worker.instance_of is None:
            return
        ids = self._instances.get(worker.instance_of)
        if ids is not None:
            ids.pop(worker.id, None)
            if not ids:
                del self._instances[worker.instance_of]

    def _prune_left_instances(self, name: str) -> None:
        """Keep at most :data:`MAX_LEFT_INSTANCES` instances of *name* that
        left (not connected and nothing in flight: ``offline`` after its
        run, or ``disconnected`` once the resume grace ended its run); the
        oldest connection goes first."""
        left = [w for w in self.instances_of(name)
                if not w.online and not w.running]
        for worker in left[:max(0, len(left) - MAX_LEFT_INSTANCES)]:
            self._unindex(worker)
            del self._workers[worker.id]

    def forget_instances(self, name: str, reason: str) -> list[str]:
        """Drop every instance of *name*'s credential (it was revoked), and
        the runs adopted after a restart under an instance's id, as
        :meth:`forget` does for one worker. Returns the dropped ids."""
        prefix = f"{name}{wp.INSTANCE_SEPARATOR}"
        ids = sorted(set(self._instances.get(name, ()))
                     | {i for i in self._adopted if i.startswith(prefix)})
        for worker_id in ids:
            self.forget(worker_id, reason)
        return ids

    def holder(self, worker_id: str) -> Worker | None:
        """The worker registered as *worker_id*, else the unlisted
        placeholder of the runs adopted under it after a restart (#793),
        else None: whom a ``hello`` under that id would take over."""
        return self._workers.get(worker_id) or self._adopted.get(worker_id)

    # ── known workers (#780) ──────────────────────────────────────────────

    def load_known(self, known: list[Any]) -> list[Worker]:
        """Add enrolled workers that are not connected, as ``offline``.

        *known* holds ``worker_store.KnownWorker``-like objects (``name``,
        ``platform``, ``mounts``, ``devices`` as dicts, ``last_seen_at``
        ISO text). A name already registered is left as it is. The devices
        are the ones it last reported, so a run can target it and wait.
        """
        from .db import iso_to_epoch

        added = []
        for entry in known:
            if entry.name in self._workers:
                continue
            devices = []
            for raw in entry.devices:
                try:
                    devices.append(wp.DeviceReport.model_validate(raw))
                except ValueError:
                    logger.warning("worker %s: a stored device report "
                                   "could not be read; left out", entry.name)
            seen = iso_to_epoch(entry.last_seen_at)
            worker = Worker(
                id=entry.name, name=entry.name, version="",
                fingerprint=getattr(entry, "fingerprint_digest", None),
                platform=dict(entry.platform), mounts=dict(entry.mounts),
                devices=devices, send=_not_connected, close=_nothing,
                heartbeat_s=self.heartbeat_s, status=STATUS_OFFLINE,
                connected_at=seen or 0.0, last_heartbeat_at=seen,
                enrolled=True,
                # Its last agent was one-shot (#845, migration 016): a run
                # naming it is timed by the one-shot queue rule before that
                # agent is back.
                one_shot=bool(getattr(entry, "one_shot", False)))
            self._workers[entry.name] = worker
            added.append(worker)
        if added:
            self._cards_changed()
            logger.info("%d enrolled worker(s) known, offline until they "
                        "connect: %s", len(added),
                        ", ".join(w.name for w in added))
            self._changed()
        return added

    def forget(self, worker_id: str,
               reason: str = "the worker was removed") -> Worker | None:
        """Drop *worker_id*: a connected one is disconnected and its socket
        closed. Every run on it ends ``interrupted`` at once, including runs
        already waiting for it to reconnect: once it is gone from the
        registry nothing (the watchdog included) would end them. Returns the
        dropped worker."""
        self._end_adopted(worker_id, reason)
        self._forget_adopted_cards(worker_id=worker_id)
        worker = self._workers.get(worker_id)
        if worker is None:
            return None
        if worker.online:
            # Removed: nothing will come back for its runs (#783).
            self.disconnect(worker, reason, resume=False)
            self._spawn(_close_quietly(worker))
        for run in list(worker.running.values()):
            self._end(worker, run, reason)
        self._unindex(worker)
        del self._workers[worker_id]
        self._cards_changed()
        logger.info("worker %s removed", worker_id)
        self._changed()
        return worker

    def kick(self, worker_id: str, reason: str) -> None:
        """Disconnect *worker_id*'s live connection, if any, keeping it known."""
        worker = self._workers.get(worker_id)
        if worker is not None and worker.online:
            self.disconnect(worker, reason)
            self._spawn(_close_quietly(worker))

    def stopped(self, worker_id: str, reason: str) -> None:
        """*worker_id* was stopped on purpose (an SSH worker's Disable,
        Remove or a shutdown, #784): its runs will not come back, so the
        ones still on it end ``interrupted`` now instead of after the
        resume grace. A live connection is closed the same way."""
        self._end_adopted(worker_id, reason)
        self._forget_adopted_cards(worker_id=worker_id)
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        if worker.online:
            self.disconnect(worker, reason, resume=False)
            self._spawn(_close_quietly(worker))
            return
        if worker.running:
            for run in list(worker.running.values()):
                self._end(worker, run, reason)
            self._changed()

    def disconnect(self, worker: Worker, reason: str = "the connection closed",
                   *, resume: bool = True) -> None:
        """Mark *worker* disconnected. Idempotent.

        Its runs wait ``resume_grace_s`` for it to come back (#783); with
        no grace, or *resume* False (this process is stopping), they end
        at once. A worker that a newer connection already replaced is left
        alone: its runs moved to the replacement.
        """
        if self._workers.get(worker.name) is not worker:
            return
        if worker.status == STATUS_DISCONNECTED:
            return
        self.outputs.worker_gone(worker.name, reason)
        if worker.one_shot and not worker.running:
            # A one-shot agent leaving with nothing in flight (#790): it
            # reported its run and exited, or was stopped before one came.
            # Nothing waits for it to come back.
            worker.status = STATUS_OFFLINE
            logger.info("one-shot worker %s left (%s)", worker.name,
                        f"after run {worker.one_shot_claim}"
                        if worker.one_shot_claim else "without a run")
            self._changed()
            return
        worker.status = STATUS_DISCONNECTED
        logger.warning("worker %s disconnected: %s", worker.name, reason)
        for run in list(worker.running.values()):
            if resume and self.resume_grace_s > 0:
                self._detach(worker, run, reason)
            else:
                self._end(worker, run, reason)
        self._changed()

    def _detach(self, worker: Worker, run: WorkerRun, reason: str) -> None:
        """*run* waits for its worker to reconnect (#783)."""
        if run.gone or run.detached_at is not None:
            return
        run.detached_at = self._clock()
        run.detached_wall = time.time()
        run.awaiting_alive = False
        logger.warning("run %s on worker %s waits up to %.0f s for it to "
                       "reconnect: %s", run.run_id, worker.name,
                       self.resume_grace_s, reason)
        run.inbox.put_nowait((WORKER_DISCONNECTED, reason))

    def _end(self, worker: Worker, run: WorkerRun, reason: str) -> None:
        if run.gone:
            return
        run.gone = True
        worker.running.pop(run.run_id, None)
        self._forget_adopted_cards(run_id=run.run_id)
        logger.warning("run %s on worker %s ends interrupted: %s",
                       run.run_id, worker.name, reason)
        # It may still be running there: a later run_finished is a late
        # result (#783).
        self._late[(worker.name, run.run_id)] = run.last_seq
        while len(self._late) > LATE_RUNS_MAX:
            del self._late[next(iter(self._late))]
        run.inbox.put_nowait((WORKER_GONE, reason))

    def heartbeat(self, worker: Worker, frame: wp.Heartbeat) -> None:
        worker.last_heartbeat = self._clock()
        worker.last_heartbeat_at = time.time()
        if frame.devices != worker.devices:
            identity_changed = (_device_identities(frame.devices)
                                != _device_identities(worker.devices))
            worker.devices = list(frame.devices)
            if identity_changed:
                worker.shared = self._shared_devices(worker)
                self._cards_changed()
                # A device appearing (or a card changing its slot) is
                # capacity a waiting run may now take. Memory figures
                # change on every heartbeat and are not, except on a card
                # shared by memory (#788): there free memory IS capacity.
                self._changed()
            elif self._shares_by_memory(worker) and (
                    self.memory_waiting is None
                    or self.memory_waiting(worker.id)):
                self._changed()
        if frame.mounts is not None and frame.mounts != worker.mounts:
            # A share that appeared (or went away) on the worker (#782):
            # waiting runs that need it may now be placed.
            logger.info("worker %s now serves mounts %s", worker.name,
                        sorted(frame.mounts) or "none")
            worker.mounts = dict(frame.mounts)
            self._changed()
        if worker.status == STATUS_DEGRADED:
            worker.status = STATUS_ONLINE
            logger.info("worker %s is back on time", worker.name)
        self._cancel_untracked(worker, frame.running)

    def _cancel_untracked(self, worker: Worker, run_ids: Any) -> None:
        """Stop every run *worker* executes that nothing here tracks (#783).

        A run that ended here (past its resume grace, or never re-attached)
        gave its slots back; left running on the worker, its subprocess
        would keep the card while the scheduler hands it to another run.
        So the worker is told to stop it at once (``cancel``, ``grace_s``
        0). Its ``run_finished`` is still recorded as a late result. Sent
        on every ``run_alive`` and heartbeat that lists it; the agent
        ignores a cancel for a run it already stopped.
        """
        now = self._clock()
        for run_id in run_ids:
            if run_id in worker.running:
                continue
            closed = self._recently_closed.get(run_id)
            if closed is not None and now - closed < RECENTLY_CLOSED_S:
                continue  # finished here; its subprocess is exiting
            logger.warning("worker %s is running %s, which nothing here "
                           "tracks; telling it to stop", worker.name, run_id)
            self._spawn(worker.send(wp.Cancel(run_id=run_id, grace_s=0)))

    def run_alive(self, worker: Worker, frame: wp.RunAlive) -> None:
        """Keep the inherited runs the agent lists; end the rest."""
        alive = {entry.run_id for entry in frame.runs}
        for run in list(worker.running.values()):
            if not run.awaiting_alive:
                continue
            if run.run_id in alive:
                run.awaiting_alive = False
                logger.info("run %s re-attached to worker %s", run.run_id,
                            worker.name)
                if run.detached_at is not None:
                    run.detached_at = run.detached_wall = None
                    # Back: a later drop is an ordinary one (#793).
                    run.adopted = False
                    run.inbox.put_nowait((WORKER_RECONNECTED,))
            else:
                self._end(worker, run, "the reconnected agent does not have it")
        worker.alive_deadline = None
        self._cancel_untracked(worker, alive)

    def drain(self, worker_id: str, draining: bool = True) -> Worker | None:
        worker = self._workers.get(worker_id)
        if worker is None:
            return None
        # Draining a credential drains its instances too (#845).
        changed = False
        for each in [worker, *self.instances_of(worker_id)]:
            if each.draining != draining:
                each.draining = draining
                changed = True
                logger.info("worker %s %s", each.name,
                            "draining" if draining else "undrained")
        if changed:
            self._changed()
        return worker

    # ── one-shot workers (#790) ───────────────────────────────────────────

    def claim_one_shot(self, worker_id: str, run_id: str) -> None:
        """*run_id* was placed on *worker_id*: a one-shot worker takes no
        other. Called synchronously where the run's slots are taken, so no
        second placement can slip in before the assignment goes out."""
        worker = self._workers.get(worker_id)
        if worker is not None and worker.one_shot \
                and worker.one_shot_claim is None:
            worker.one_shot_claim = run_id
            logger.info("one-shot worker %s takes run %s", worker.name,
                        run_id)

    def one_shot_assigned(self, worker_id: str, run_id: str) -> None:
        """*run_id*'s ``assign`` went out to *worker_id*: its claim stays."""
        worker = self._workers.get(worker_id)
        if worker is not None and worker.one_shot_claim == run_id:
            worker.one_shot_assigned = True

    def release_one_shot(self, worker_id: str, run_id: str) -> None:
        """*run_id* gave back its slots: if its ``assign`` never went out
        to *worker_id* (cancelled while starting, refused here, the socket
        failed under the send), the one-shot worker may take another run.
        Idempotent; an assigned claim stays."""
        worker = self._workers.get(worker_id)
        if worker is not None and worker.one_shot_claim == run_id \
                and not worker.one_shot_assigned:
            worker.one_shot_claim = None
            self._changed()

    # ── runs ──────────────────────────────────────────────────────────────

    def open_run(self, worker: Worker, run_id: str) -> WorkerRun:
        run = WorkerRun(run_id=run_id)
        worker.running[run_id] = run
        return run

    def adopt(self, worker_id: str, run_id: str, *,
              last_seq: int = 0,
              instance_hash: str | None = None,
              one_shot: bool = False) -> WorkerRun:
        """Take over *run_id*, left in flight on *worker_id* by a previous
        process of this server (#793). STARTUP, before workers connect.

        The run waits detached, as after a dropped connection: the worker's
        reconnect and ``run_alive`` re-attach it within ``resume_grace_s``
        (counted from now), else it ends ``interrupted``. *last_seq* is the
        last frame the previous process recorded; the replay's frames up to
        it are dropped as duplicates (:meth:`route`). A worker known here
        (an enrolled one, offline) holds it in its ``running``; for one not
        known yet an unlisted placeholder does, until it connects.
        *instance_hash*: the hashed ``instance_id`` of the agent process
        that ran it (#845): only that process may upload the run's files
        (``routes_runs``) and, for a *one_shot* run, take the id back
        (``worker_handshake.same_agent``). A placeholder of a one-shot run
        is marked one-shot, so the one-shot rule
        (``worker_handshake.one_shot_refusal``) applies to it.
        """
        worker = self._workers.get(worker_id)
        if worker is None:
            worker = self._adopted.get(worker_id)
        if worker is None:
            worker = self._adopted[worker_id] = Worker(
                id=worker_id, name=worker_id, version="", fingerprint=None,
                platform={}, mounts={}, devices=[], send=_not_connected,
                close=_nothing, heartbeat_s=self.heartbeat_s,
                status=STATUS_OFFLINE, last_heartbeat_at=None)
        if instance_hash is not None:
            worker.instance_hash = instance_hash
        if one_shot and worker_id in self._adopted:
            worker.one_shot = True
        run = WorkerRun(run_id=run_id, last_seq=max(0, int(last_seq)),
                        detached_at=self._clock(), detached_wall=time.time(),
                        adopted=True)
        worker.running[run_id] = run
        self._unclaimed[run_id] = run
        logger.warning("run %s on worker %s was in flight when this server "
                       "restarted; waiting up to %.0f s for the worker to "
                       "report it", run_id, worker_id, self.resume_grace_s)
        self._ensure_watchdog()
        return run

    def end_waiting(self, worker_id: str, run_id: str, reason: str) -> bool:
        """End *run_id* now if it waits for *worker_id* to come back (a
        drop, or adopted after a restart): this server is shutting down
        and nothing will come back for it here (#793). True if ended."""
        worker = self._holder(worker_id, run_id)
        run = None if worker is None else worker.running[run_id]
        if run is None or run.detached_at is None:
            return False
        self._end(worker, run, reason)
        return True

    def claim_adopted(self, run_id: str) -> WorkerRun | None:
        """The run :meth:`adopt` took over as *run_id*, once: its inbox
        holds what happened to it meanwhile (an end included)."""
        return self._unclaimed.pop(run_id, None)

    def _end_adopted(self, worker_id: str, reason: str) -> None:
        """End the runs an unlisted placeholder holds for *worker_id*."""
        placeholder = self._adopted.pop(worker_id, None)
        if placeholder is not None:
            for run in list(placeholder.running.values()):
                self._end(placeholder, run, reason)

    def close_run(self, worker_id: str, run_id: str, *,
                  finished: bool = False) -> None:
        """*run_id* ended here. *finished*: its worker reported the end
        (``run_finished``), so a heartbeat that still lists it for a moment
        is not a stray run (:meth:`_cancel_untracked`)."""
        worker = self._workers.get(worker_id)
        if worker is not None:
            worker.running.pop(run_id, None)
        self._forget_adopted_cards(run_id=run_id)
        # A run adopted for a worker not known yet (#793): its placeholder
        # must not hand a run nothing reads to the worker's connection.
        self._unclaimed.pop(run_id, None)
        placeholder = self._adopted.get(worker_id)
        if placeholder is not None:
            placeholder.running.pop(run_id, None)
            if not placeholder.running:
                del self._adopted[worker_id]
        if finished:
            now = self._clock()
            self._recently_closed[run_id] = now
            for stale in [r for r, at in self._recently_closed.items()
                          if now - at >= RECENTLY_CLOSED_S]:
                del self._recently_closed[stale]

    def touch(self, worker: Worker) -> None:
        """Any frame from *worker* says it is alive (#783): a long replay
        must not read as missed heartbeats."""
        worker.last_heartbeat = self._clock()
        if worker.status == STATUS_DEGRADED:
            worker.status = STATUS_ONLINE

    def route(self, worker: Worker, frame: Any) -> None:
        """Deliver a per-run frame to its run's inbox, dropping duplicates."""
        run = worker.running.get(frame.run_id)
        if run is None:
            self._route_orphan(worker, frame)
            return
        if frame.seq <= run.last_seq:
            logger.debug("run %s: duplicate seq %s ignored", run.run_id,
                         frame.seq)
            return
        if frame.seq > run.last_seq + 1:
            # The agent sends a run's frames in order and replays every
            # unacknowledged one after a reconnect, so a gap means frames
            # were lost on its side (an unwritable spool). Nothing can
            # resend them; say so and carry on with what arrived.
            logger.warning("run %s on worker %s: frames %d-%d never "
                           "arrived", run.run_id, worker.name,
                           run.last_seq + 1, frame.seq - 1)
        run.last_seq = frame.seq
        run.inbox.put_nowait(frame)
        if run.inbox.qsize() > INBOX_HIGH_WATER and not run.high_water_logged:
            run.high_water_logged = True
            logger.warning(
                "run %s: %d frames from worker %s are waiting to be "
                "recorded; the database is falling behind", run.run_id,
                run.inbox.qsize(), worker.name)

    async def backpressure(self, worker: Worker, frame: Any) -> None:
        """Pause reading *worker*'s socket while *frame*'s run is far
        behind (#783).

        A replay after a long disconnect can deliver many thousands of
        frames at once, faster than the database records them. Past
        :data:`INBOX_HIGH_WATER` queued frames the socket handler waits
        here until the inbox is down to half of that, so TCP slows the
        agent instead of the inbox growing without bound. The worker's
        heartbeat clock is kept current meanwhile: it is silent because
        this side stopped reading, not because it is gone.
        """
        run = worker.running.get(getattr(frame, "run_id", None) or "")
        if run is None or run.inbox.qsize() <= INBOX_HIGH_WATER:
            return
        while run.inbox.qsize() > INBOX_HIGH_WATER // 2 and not run.gone \
                and worker.online:
            worker.last_heartbeat = self._clock()
            await asyncio.sleep(INBOX_DRAIN_POLL_S)

    def _route_orphan(self, worker: Worker, frame: Any) -> None:
        """A frame for a run nothing here waits for (#783).

        A run that already ended here (``_late``) may still finish on its
        worker: its ``run_finished`` goes to :attr:`on_late_result`, once.
        Every orphan frame is acknowledged now and then (the last one
        always), or the agent would keep them spooled and replay them on
        every reconnect.
        """
        key = (worker.name, frame.run_id)
        last = self._late.get(key)
        if last is None:
            logger.info("worker %s sent a %s for run %s, which is not "
                        "running there; ignored", worker.name, frame.type,
                        frame.run_id)
        elif frame.seq > last:
            self._late[key] = frame.seq
            if isinstance(frame, wp.RunFinished):
                del self._late[key]
                logger.warning("run %s on worker %s reported %s after it "
                               "ended here", frame.run_id, worker.name,
                               frame.status)
                if self.on_late_result is not None:
                    self._spawn(self.on_late_result(worker.name,
                                                    frame.run_id, frame))
        if isinstance(frame, wp.RunFinished) \
                or frame.seq % ORPHAN_ACK_EVERY == 0:
            self._spawn(worker.send(wp.Ack(run_id=frame.run_id,
                                           seq=frame.seq)))

    # ── the watchdog ──────────────────────────────────────────────────────

    def check(self) -> None:
        """Apply the heartbeat rules once (the watchdog calls this)."""
        now = self._clock()
        for name, placeholder in list(self._adopted.items()):
            self._expire_detached(placeholder, now)
            if not placeholder.running:
                del self._adopted[name]
        for worker in list(self._workers.values()):
            if not worker.online:
                self._expire_detached(worker, now)
                continue
            silent = now - worker.last_heartbeat
            if silent > worker.heartbeat_s * self.missed_heartbeats:
                self.disconnect(worker, f"no heartbeat for {silent:.0f} s")
                self._spawn(_close_quietly(worker))
                continue
            if silent > worker.heartbeat_s * 1.5 \
                    and worker.status == STATUS_ONLINE:
                worker.status = STATUS_DEGRADED
                logger.warning("worker %s: heartbeat late (%.0f s)",
                               worker.name, silent)
            if worker.alive_deadline is not None \
                    and now >= worker.alive_deadline:
                worker.alive_deadline = None
                for run in list(worker.running.values()):
                    if run.awaiting_alive:
                        self._end(worker, run, "no run_alive after reconnect")

    def _expire_detached(self, worker: Worker, now: float) -> None:
        """End the runs of a disconnected *worker* whose grace ran out."""
        for run in list(worker.running.values()):
            if run.detached_at is not None \
                    and now - run.detached_at >= self.resume_grace_s:
                if run.adopted:
                    reason = (f"it did not reconnect with the run within "
                              f"{self.resume_grace_s:.0f} s after this "
                              f"server restarted")
                else:
                    reason = (f"worker {worker.name} did not reconnect "
                              f"within {self.resume_grace_s:.0f} s")
                self._end(worker, run, reason)
                self._changed()

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("a worker registry task failed",
                           exc_info=task.exception())

    def _ensure_watchdog(self) -> None:
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.ensure_future(self._watch())

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(min(1.0, self.heartbeat_s / 4))
            try:
                self.check()
            except Exception:  # noqa: BLE001
                logger.exception("worker watchdog failed")

    async def shutdown(self) -> None:
        """Stop the watchdog and close every connection."""
        if self._probe_task is not None and not self._probe_task.done():
            # The thread itself finishes on its own; nothing waits for it.
            self._probe_task.cancel()
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for worker in list(self._workers.values()):
            if worker.online:
                # ``drain`` first: the agent takes no new work until its next
                # connection (research report 5.9).
                try:
                    await worker.send(wp.Drain())
                except Exception:  # noqa: BLE001 -- already gone
                    pass
                self.disconnect(worker, "the control plane is shutting down",
                                resume=False)
                await _close_quietly(worker)


async def _not_connected(_frame: Any) -> None:
    raise ConnectionError("the worker is not connected")


async def _nothing() -> None:
    return None


async def _close_quietly(worker: Worker) -> None:
    try:
        await worker.close()
    except Exception:  # noqa: BLE001 -- already gone
        pass
