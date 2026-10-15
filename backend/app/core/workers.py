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
this server started. Draining is a separate flag (no new assignments); the API's
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
records it as a ``late_result`` warning without changing the run's status.
Frames for a run nothing waits for are acknowledged anyway, so the agent
does not keep them forever.

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

When a replacing connection arrives while runs are still in flight on the
old one, those runs move to the new connection and wait for its
``run_alive``: a run it lists carries on, one it does not (or every one,
if no ``run_alive`` arrives within a heartbeat interval) ends
``interrupted``. Nothing is lost silently: each outcome is logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import worker_protocol as wp

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
#: ``(WORKER_RECONNECTED, last_seq)`` (#783).
WORKER_RECONNECTED = object()


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


@dataclass(eq=False)
class Worker:
    """One connected (or recently connected) worker agent."""

    id: str
    name: str
    version: str
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
    #: Worker device -> this process's device for each card the two share
    #: (#773, ``scheduling.slot_for``): ``cpu``, ``mps:0``, and each CUDA
    #: card whose UUID matches one here. Empty for another machine.
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

    @property
    def online(self) -> bool:
        return self.status not in (STATUS_DISCONNECTED, STATUS_OFFLINE)

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
            "status": (STATUS_DRAINING if self.draining and self.online
                       else self.status),
            "draining": self.draining,
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
                             else self.connected_at),
            "last_heartbeat_at": self.last_heartbeat_at,
            "enrolled": self.enrolled,
            "credential": self.credential,
        }


def _canonical_or_none(device: str) -> str | None:
    try:
        return wp.canonical_device(device)
    except ValueError:
        return None


def _device_identities(devices: list[wp.DeviceReport]) -> list[tuple]:
    """What placement reads off a device list: which devices, not their
    memory figures."""
    return [(d.value, d.kind, d.index, d.uuid) for d in devices]


class WorkerRegistry:
    """Every worker this control plane knows, by name. Event-loop only.

    *local_devices* returns this process's devices as ``DeviceFact``-like
    objects (``value``, ``uuid``); it is called once, on the first
    same-host worker, and is what a same-host worker's CUDA cards are
    matched against by UUID. Without it no CUDA card is shared.
    """

    def __init__(self, *, heartbeat_s: float = wp.DEFAULT_HEARTBEAT_S,
                 grace_s: float = wp.DEFAULT_GRACE_S,
                 missed_heartbeats: int = MISSED_HEARTBEATS,
                 clock: Callable[[], float] = time.monotonic,
                 local_devices: Callable[[], list[Any]] | None = None,
                 resume_grace_s: float = DEFAULT_RESUME_GRACE_S) -> None:
        self._local_devices_source = local_devices
        self._local_devices_cache: list[Any] | None = None
        #: ``(worker, device, uuid)`` already logged as not shared.
        self._unshared_logged: set[tuple[str, str, str | None]] = set()
        self.heartbeat_s = heartbeat_s
        self.grace_s = grace_s
        self.missed_heartbeats = missed_heartbeats
        #: Seconds a disconnected worker's runs wait for it; 0: none (#783).
        self.resume_grace_s = max(0.0, float(resume_grace_s))
        #: ``(worker name, run id)`` -> last seq seen, for each run that
        #: ended here while it may still be running on its worker (#783).
        self._late: dict[tuple[str, str], int] = {}
        #: Awaited as ``on_late_result(worker_name, run_id, run_finished)``
        #: when such a run reports its end (``RunService`` records it).
        self.on_late_result: Callable[[str, str, Any], Awaitable[Any]] | None \
            = None
        self._clock = clock
        self._workers: dict[str, Worker] = {}
        self._listeners: list[Callable[[], None]] = []
        self._watchdog: asyncio.Task | None = None
        #: Fire-and-forget closes, kept so they are not collected mid-flight
        #: and their failures are logged.
        self._tasks: set[asyncio.Task] = set()

    # ── queries ───────────────────────────────────────────────────────────

    def get(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    def workers(self) -> list[Worker]:
        return sorted(self._workers.values(), key=lambda w: w.name)

    def run_state(self, worker_id: str, run_id: str) -> WorkerRun | None:
        """The run *worker_id* is executing as *run_id*, or None."""
        worker = self._workers.get(worker_id)
        return None if worker is None else worker.running.get(run_id)

    # ── same-host devices (#773) ─────────────────────────────────────────

    def _local_devices(self) -> list[Any]:
        if self._local_devices_cache is None:
            source = self._local_devices_source
            try:
                self._local_devices_cache = list(source()) if source else []
            except Exception:  # noqa: BLE001 -- sharing degrades, never fails
                logger.exception("could not list this server's devices; no "
                                 "CUDA card is shared with same-host workers")
                self._local_devices_cache = []
        return self._local_devices_cache

    def _shared_devices(self, worker: Worker) -> dict[str, str]:
        """Worker device -> this process's device, for the cards both use.

        Only for a same-host worker. ``cpu`` is shared, and ``mps:0`` when
        this process has it too (one Apple GPU per machine). A CUDA card is
        shared only when its UUID matches one of this process's cards, so
        two processes that number the cards differently (another
        ``CUDA_VISIBLE_DEVICES``) still agree on which card is which. A
        CUDA card without a UUID, or whose UUID matches nothing here, keeps
        a slot of its own; that is logged once.
        """
        if not worker.same_host:
            return {}
        local = self._local_devices()
        by_uuid: dict[str, str] = {}
        local_values: set[str] = set()
        for fact in local:
            value = _canonical_or_none(getattr(fact, "value", ""))
            if value is None:
                continue
            local_values.add(value)
            uuid = getattr(fact, "uuid", None)
            if uuid and value.startswith("cuda"):
                by_uuid[str(uuid).lower()] = value
        shared: dict[str, str] = {}
        for report in worker.devices:
            value = _canonical_or_none(report.value)
            if value is None:
                continue
            if value == "cpu" or (value.startswith("mps")
                                  and value in local_values):
                shared[value] = value
                continue
            match = by_uuid.get(report.uuid.lower()) if report.uuid else None
            if match is not None:
                shared[value] = match
                continue
            key = (worker.name, value, report.uuid)
            if key not in self._unshared_logged:
                self._unshared_logged.add(key)
                logger.warning(
                    "worker %s runs on this machine, but its %s (uuid %s) "
                    "matches no card of this server; it keeps a slot of "
                    "its own", worker.name, value, report.uuid or "unknown")
        return shared

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
                enrolled: bool = False) -> tuple[Worker, Worker | None]:
        """Register *hello*'s worker. Returns ``(worker, replaced)``.

        *same_host*: the worker's machine identity equals this process's
        (``ws_worker``), so the cards it shares with this process (see
        :meth:`_shared_devices`) share their slots. *credential* says how
        the connection authenticated; *enrolled* that the worker has a
        ``workers`` row (a known worker keeps the flag either way).

        A worker already registered under the name is replaced: its runs
        move to the new connection and wait for ``run_alive``, and the
        caller closes the old socket (``replaced.close``).
        """
        old = self._workers.get(hello.name)
        worker = Worker(
            id=hello.name, name=hello.name, version=hello.version,
            fingerprint=hello.fingerprint, platform=dict(hello.platform),
            mounts=dict(hello.mounts), devices=list(hello.devices),
            send=send, close=close, heartbeat_s=self.heartbeat_s,
            last_heartbeat=self._clock(), same_host=same_host,
            credential=credential,
            enrolled=enrolled or (old is not None and old.enrolled))
        worker.shared = self._shared_devices(worker)
        replaced = None
        if old is not None:
            worker.draining = old.draining
            if old.online:
                replaced = old
                old.status = STATUS_DISCONNECTED
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
        self._ensure_watchdog()
        logger.info("worker %s connected (%s, devices %s)", worker.name,
                    worker.version, worker.device_values())
        self._changed()
        return worker, replaced

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
                id=entry.name, name=entry.name, version="", fingerprint=None,
                platform=dict(entry.platform), mounts=dict(entry.mounts),
                devices=devices, send=_not_connected, close=_nothing,
                heartbeat_s=self.heartbeat_s, status=STATUS_OFFLINE,
                connected_at=seen or 0.0, last_heartbeat_at=seen,
                enrolled=True)
            self._workers[entry.name] = worker
            added.append(worker)
        if added:
            logger.info("%d enrolled worker(s) known, offline until they "
                        "connect: %s", len(added),
                        ", ".join(w.name for w in added))
            self._changed()
        return added

    def forget(self, worker_id: str,
               reason: str = "the worker was removed") -> Worker | None:
        """Drop *worker_id*: a connected one is disconnected (its runs end
        ``interrupted``) and its socket closed. Returns the dropped worker."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return None
        if worker.online:
            # Removed: nothing will come back for its runs (#783).
            self.disconnect(worker, reason, resume=False)
            self._spawn(_close_quietly(worker))
        del self._workers[worker_id]
        logger.info("worker %s removed", worker_id)
        self._changed()
        return worker

    def kick(self, worker_id: str, reason: str) -> None:
        """Disconnect *worker_id*'s live connection, if any, keeping it known."""
        worker = self._workers.get(worker_id)
        if worker is not None and worker.online:
            self.disconnect(worker, reason)
            self._spawn(_close_quietly(worker))

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
                # A device appearing (or a card changing its slot) is
                # capacity a waiting run may now take. Memory figures
                # change on every heartbeat and are not.
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
        for run_id in run_ids:
            if run_id in worker.running:
                continue
            logger.warning("worker %s is running %s, which nothing here "
                           "tracks; telling it to stop", worker.name, run_id)
            self._spawn(worker.send(wp.Cancel(run_id=run_id, grace_s=0)))

    def run_alive(self, worker: Worker, frame: wp.RunAlive) -> None:
        """Keep the inherited runs the agent lists; end the rest."""
        alive = {entry.run_id: entry.last_seq for entry in frame.runs}
        for run in list(worker.running.values()):
            if not run.awaiting_alive:
                continue
            if run.run_id in alive:
                run.awaiting_alive = False
                logger.info("run %s re-attached to worker %s", run.run_id,
                            worker.name)
                if run.detached_at is not None:
                    run.detached_at = run.detached_wall = None
                    run.inbox.put_nowait(
                        (WORKER_RECONNECTED, alive[run.run_id]))
            else:
                self._end(worker, run, "the reconnected agent does not have it")
        worker.alive_deadline = None
        self._cancel_untracked(worker, alive)

    def drain(self, worker_id: str, draining: bool = True) -> Worker | None:
        worker = self._workers.get(worker_id)
        if worker is None:
            return None
        if worker.draining != draining:
            worker.draining = draining
            logger.info("worker %s %s", worker.name,
                        "draining" if draining else "undrained")
            self._changed()
        return worker

    # ── runs ──────────────────────────────────────────────────────────────

    def open_run(self, worker: Worker, run_id: str) -> WorkerRun:
        run = WorkerRun(run_id=run_id)
        worker.running[run_id] = run
        return run

    def close_run(self, worker_id: str, run_id: str) -> None:
        worker = self._workers.get(worker_id)
        if worker is not None:
            worker.running.pop(run_id, None)

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
                self._end(worker, run, f"worker {worker.name} did not "
                          f"reconnect within {self.resume_grace_s:.0f} s")
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
