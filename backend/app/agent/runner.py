"""One spawned subprocess per assigned run, and the frames it produces (#771).

:class:`Runner` is the agent's half of a run. For each ``assign`` it

1. refuses what this wave cannot do (``record_outputs``, a device this
   worker does not offer, a second copy of a running run, any run while
   draining) with an immediate ``run_finished`` ``failed``;
2. starts ``child.main`` in a ``multiprocessing`` *spawn* context with an
   explicit environment (:func:`child_environment`): the assigned CUDA
   cards as ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``,
   ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so the indices this agent reports
   (``core.resources``, NVML order) are the ones the child's torch uses,
   and no ``CODEFYUI_DB_PATH``: the child never opens SQLite. The cards
   keep the order the assignment lists them in, so the run's ``cuda:K`` is
   the K-th device of its set; ``run_started`` carries that mapping as an
   ``execution_facts`` patch (:func:`run_facts`, #774). Node ``device``
   params are not rewritten: ``CUDA_VISIBLE_DEVICES`` is what renumbers.
   With a directory for the control plane's mount label (``--mount``),
   the child's ``CODEFYUI_*`` data directories are derived from it
   (``core.mounts.data_root_env``, #782) and, unless the worker is on the
   control plane's machine, the files it reports under it are reported
   relative to it; a same-host worker without one gets the control plane's
   own directories as ``assign.storage.dirs`` (:meth:`Runner.storage_plan`);
3. reads the child's pipe on a thread of its own and forwards every message
   as a protocol frame, in order, stamped with the run's next ``seq``. The
   thread waits for each frame to be handed to the connection before it
   reads the next message, so a slow control plane slows the child's pipe
   writes instead of growing a buffer here;
4. reports how the run ended, once the child process has exited (one that
   lingers after reporting is terminated, then killed), so the control
   plane never frees the device slot while the old process still holds
   it. The child says so itself (``finished``);
   a child that exits without saying is a crash (``failed``, with the exit
   code and the tail of its output) or, after a ``cancel`` it did not
   honour within ``grace_s``, a run this agent terminated
   (``interrupted``, naming the last node seen running).

Cancel is cooperative first: ``("cancel",)`` down the pipe, which the child
turns into ``ExecutionContext.cancel()``. ``grace_s`` later the process is
terminated, and :data:`KILL_AFTER_S` after that killed.

This module imports neither torch nor the engine: it runs in the agent.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import multiprocessing
import re
import shutil
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..core import mounts as mounts_module
from ..core import worker_protocol as wp
from . import child as child_module
from .spool import SPOOL_FILE

logger = logging.getLogger(__name__)

#: Seconds between ``terminate()`` and ``kill()`` for a cancelled run.
KILL_AFTER_S = 5.0
#: Seconds a child gets to exit after it reported its outcome.
EXIT_AFTER_FINISHED_S = 10.0
#: How often the reader checks a reported child's liveness while draining.
DRAIN_POLL_S = 0.2
#: How much of a crashed child's output ``run_finished`` carries.
STDERR_TAIL_BYTES = 4096
#: Per-run output file, under ``<runs_dir>/<run_id>/``.
OUTPUT_LOG = "output.log"

REFUSE_RECORD_OUTPUTS = ("record_outputs is not supported on workers yet "
                         "(#789); submit without it")
REFUSE_DRAINING = "this worker is draining and takes no new runs"
REFUSE_DUPLICATE = "this worker is already running {run_id}"

_DEVICE_SYNTAX = re.compile(r"^(cpu|cuda|mps)(?::(\d+))?$")

Send = Callable[[Any], Awaitable[None]]


class AssignmentRefused(ValueError):
    """An assignment this worker cannot run; the message says why."""


def split_device(device: str) -> tuple[str, int | None]:
    """``"cuda:1"`` -> ``("cuda", 1)``; ``"cpu"`` -> ``("cpu", None)``.

    ``device_utils.split_device`` without its torch import.
    """
    match = _DEVICE_SYNTAX.match(device.strip().lower())
    if match is None:
        raise AssignmentRefused(f"unknown device {device!r}")
    kind, index = match.groups()
    return kind, None if index is None else int(index)


def cuda_indices(devices: list[str]) -> list[int]:
    """The CUDA card indices in *devices*, in the order given."""
    indices = []
    for device in devices:
        kind, index = split_device(device)
        if kind == "cuda" and index is not None:
            indices.append(index)
    return indices


def device_aliases(devices: list[str]) -> dict[str, str]:
    """Physical CUDA card -> its name inside the run, for *devices* (#774).

    The one statement of the alias rule; :func:`child_environment` and
    :func:`child_device` both read it. The run subprocess gets the cards
    as ``CUDA_VISIBLE_DEVICES`` in the order listed and the CUDA runtime
    numbers them in that order, so with ``["cuda:2", "cuda:0"]`` the run's
    ``cuda:0`` is this machine's ``cuda:2`` and its ``cuda:1`` is
    ``cuda:0``: the alias is the card's position in *devices*, which is
    why the order of a device set matters. Assumes no card is listed twice
    (``Runner._check`` refuses that). ``{}`` when the set names no card.
    """
    return {f"cuda:{index}": f"cuda:{position}"
            for position, index in enumerate(cuda_indices(devices))}


def child_device(device: str, devices: list[str]) -> str:
    """*device* renumbered for a child that sees only *devices*' cards.

    With ``CUDA_VISIBLE_DEVICES=2,5`` the child's ``cuda:0`` is this
    machine's ``cuda:2``. A device outside the set is refused: the child
    could not reach it.
    """
    kind, index = split_device(device)
    aliases = device_aliases(devices)
    if kind != "cuda" or index is None or not aliases:
        return device
    alias = aliases.get(f"cuda:{index}")
    if alias is None:
        raise AssignmentRefused(
            f"device {device!r} is not in the assigned set {devices}")
    return alias


def canonical_device(device: str) -> str:
    """``worker_protocol.canonical_device``, refusing what it cannot parse."""
    try:
        return wp.canonical_device(device)
    except ValueError as exc:
        raise AssignmentRefused(str(exc)) from None


#: Removed from every run subprocess's environment: the database it must
#: never open, the agent's connection token, and the ROCm visibility
#: filters (``CUDA_VISIBLE_DEVICES`` alone selects the cards, and ROCm
#: honours it; an inherited ``HIP_``/``ROCR_VISIBLE_DEVICES`` would filter
#: the cards twice).
CHILD_UNSET_ENV = ("CODEFYUI_DB_PATH", "CODEFYUI_AGENT_TOKEN",
                   "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def child_environment(device: str, devices: list[str],
                      data_env: tuple[dict[str, str], list[str]] | None = None
                      ) -> dict[str, Any]:
    """``{"set", "unset", "devices"}`` for ``child.apply_environment``.

    ``devices`` is the effective set: *devices*, or ``[device]`` when the
    set is empty. The run sees only its cards, as ``CUDA_VISIBLE_DEVICES``
    in :func:`device_aliases` order, and no card at all when it names none
    (a ``cpu`` or ``mps`` run). A bare ``cuda`` with no set keeps the
    visibility the agent inherited. ``apply_environment`` reads only
    ``set`` and ``unset``.

    *data_env* is ``(set, unset)`` for the run's data settings (#782,
    :meth:`Runner.storage_plan`). Without it the child keeps the data
    settings the agent has.
    """
    env: dict[str, str] = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    effective = list(devices) or [device]
    aliases = device_aliases(effective)
    if aliases:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            physical.split(":", 1)[1] for physical in aliases)
    elif "cuda" not in {split_device(d)[0] for d in effective}:
        env["CUDA_VISIBLE_DEVICES"] = ""
    unset = list(CHILD_UNSET_ENV)
    if data_env is not None:
        derived, cleared = data_env
        env.update(derived)
        unset.extend(cleared)
    return {"set": env, "unset": unset, "devices": effective}


def run_facts(env: dict[str, Any]) -> dict[str, Any]:
    """What the run subprocess sees of this machine's devices (#774).

    *env* is :func:`child_environment`'s result. Merged into the run's
    ``execution_facts`` by the control plane from ``run_started``.
    ``visible_devices`` is ``CUDA_VISIBLE_DEVICES`` exactly as the
    subprocess gets it (``""`` for a run that sees no card, ``None`` when
    the agent left the inherited value in place); ``device_aliases`` is
    :func:`device_aliases` of the same effective set.
    """
    return {
        "visible_devices": (env.get("set") or {}).get("CUDA_VISIBLE_DEVICES"),
        "device_aliases": device_aliases(env.get("devices") or []),
    }


def read_tail(path: Path, limit: int = STDERR_TAIL_BYTES) -> str:
    """The last *limit* bytes of *path*, decoded; ``""`` when unreadable."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


#: Touched in a run's directory when the run is retired. Retention orders
#: by it, so a long run that ended last is kept over a short one that
#: started later, and a directory without it (a run in flight) is never
#: pruned.
FINISHED_MARKER = ".finished"


def mark_finished(run_dir: Path) -> None:
    try:
        (run_dir / FINISHED_MARKER).touch()
    except OSError:
        pass


def mark_leftovers(runs_dir: Path) -> None:
    """Mark the directories a previous agent process left unmarked.

    Called before this agent starts any run: every unmarked directory then
    belongs to a run that ended with the old process, and marking it makes
    it eligible for retention.
    """
    try:
        for path in runs_dir.iterdir():
            if path.is_dir() and not (path / FINISHED_MARKER).exists():
                mark_finished(path)
    except OSError:
        pass


def prune_run_dirs(runs_dir: Path, keep_last: int) -> None:
    """Delete all but the *keep_last* most recently finished run directories.

    Same rule as the control plane's run retention (keep the last N),
    counted by retirement time (:data:`FINISHED_MARKER`). A directory that
    still holds a spool (``agent.spool``) is kept. Blocking: call it
    in a thread. Tolerates directories vanishing underneath it.
    """
    if keep_last <= 0:
        return
    try:
        candidates = list(runs_dir.iterdir())
    except OSError:
        return
    finished: list[tuple[float, Path]] = []
    for path in candidates:
        if (path / SPOOL_FILE).exists():
            continue  # frames the control plane has not acknowledged (#783)
        try:
            finished.append(((path / FINISHED_MARKER).stat().st_mtime, path))
        except OSError:  # no marker (in flight), not a dir, or gone
            continue
    finished.sort(key=lambda item: item[0], reverse=True)
    for _, stale in finished[keep_last:]:
        shutil.rmtree(stale, ignore_errors=True)


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return None


@dataclass
class RunHandle:
    """One run in flight on this agent."""

    run_id: str
    process: Any
    events_in: Any
    control_out: Any
    log_path: Path
    #: The assignment's own ``grace_s``; ``None`` defers to the runner's.
    grace_s: float | None = None
    seq: int = 0
    last_node: str | None = None
    cancel_requested: bool = False
    terminated: bool = False
    killed: bool = False
    #: How the run ended is known: the child reported it (the report is
    #: held until the process exits) or the reader is writing it.
    finished: bool = False
    #: Pipe messages read and dropped after the child's ``finished``.
    discarded: int = 0
    #: The escalation in flight stops the process now (a cancel that came
    #: after the child had reported), not after a grace period.
    stopping_now: bool = False
    reader: threading.Thread | None = None
    escalation: asyncio.Task | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Set once the reader has joined the process (it is gone).
    exited: asyncio.Event = field(default_factory=asyncio.Event)
    #: Set once the run is retired (its ``run_finished`` has been sent).
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def next_seq(self) -> int:
        with self.lock:
            self.seq += 1
            return self.seq


class Runner:
    """Starts, watches and stops this agent's run subprocesses.

    Every method is called on the agent's event loop; the per-run reader
    threads reach the loop through ``run_coroutine_threadsafe``.
    """

    def __init__(
        self, *, runs_dir: Path, offered_devices: set[str],
        mounts: dict[str, str] | None = None,
        send: Send | None = None, keep_last: int = 200,
        default_grace_s: float = wp.DEFAULT_GRACE_S,
        kill_after_s: float = KILL_AFTER_S,
        exit_after_finished_s: float = EXIT_AFTER_FINISHED_S,
        child_entry: Callable[..., None] = child_module.main,
    ) -> None:
        self.runs_dir = runs_dir
        self.offered_devices = {canonical_device(d) for d in offered_devices}
        #: ``--mount`` label -> directory on this machine (#782), sent in
        #: ``hello.mounts``.
        self.mounts: dict[str, str] = dict(mounts or {})
        self._missing_logged: set[str] = set()
        self.send: Send = send if send is not None else _drop
        self.keep_last = keep_last
        self.default_grace_s = default_grace_s
        self.kill_after_s = kill_after_s
        self.exit_after_finished_s = exit_after_finished_s
        self.child_entry = child_entry
        self.draining = False
        #: Grace a run gets when the agent itself stops (``cdui agent stop``).
        self.shutdown_grace_s = 10.0
        self.runs: dict[str, RunHandle] = {}
        #: Runs refused at assignment whose ``run_finished`` is still to be
        #: acknowledged: they have no handle to retire their directory.
        self._refused: set[str] = set()
        self._ctx = multiprocessing.get_context("spawn")
        self._housekeeping: set[asyncio.Future] = set()
        mark_leftovers(runs_dir)
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── queries ───────────────────────────────────────────────────────────

    def running_ids(self) -> list[str]:
        return list(self.runs)

    def take_refused(self, run_id: str) -> bool:
        """True (once) when *run_id* was refused at assignment."""
        if run_id in self._refused:
            self._refused.discard(run_id)
            return True
        return False

    # ── assign ────────────────────────────────────────────────────────────

    def _check(self, frame: wp.Assign) -> tuple[str, list[str]]:
        """``(child device, devices)`` for *frame*, or ``AssignmentRefused``.

        Every device is compared in canonical form (``mps`` is ``mps:0``).
        The run's own ``device`` must be offered too. A set lists each
        device once, and the run's ``device`` is its first entry (the
        control plane always sends that): the run's device is the set's
        ``cuda:0``.
        """
        if self.draining:
            raise AssignmentRefused(REFUSE_DRAINING)
        if frame.run_id in self.runs:
            raise AssignmentRefused(REFUSE_DUPLICATE.format(run_id=frame.run_id))
        if frame.options.get("record_outputs"):
            raise AssignmentRefused(REFUSE_RECORD_OUTPUTS)
        offered = self.offered_devices
        devices = list(frame.devices)
        canonical_set = [canonical_device(device) for device in devices]
        for device, canonical in zip(devices, canonical_set):
            if canonical not in offered:
                raise AssignmentRefused(
                    f"device {device!r} is not offered by this worker "
                    f"(offered: {sorted(offered)})")
        for position, canonical in enumerate(canonical_set):
            if canonical in canonical_set[:position]:
                raise AssignmentRefused(
                    f"the assigned set {devices} lists {canonical!r} twice")
        device = canonical_device(frame.device)
        kind, index = split_device(device)
        if index is None:  # a bare kind: any card of that kind will do
            if not any(split_device(o)[0] == kind for o in offered):
                raise AssignmentRefused(
                    f"device {frame.device!r} is not offered by this worker "
                    f"(offered: {sorted(offered)})")
        elif device not in offered:
            raise AssignmentRefused(
                f"device {frame.device!r} is not offered by this worker "
                f"(offered: {sorted(offered)})")
        if canonical_set and device != canonical_set[0]:
            raise AssignmentRefused(
                f"device {frame.device!r} is not the first device of the "
                f"assigned set {devices}")
        named = device if kind == "cuda" else frame.device.strip().lower()
        return child_device(named, canonical_set), canonical_set

    async def assign(self, frame: wp.Assign) -> None:
        """Start *frame*'s run, or refuse it with a ``run_finished``.

        Never raises: whatever goes wrong becomes that run's
        ``run_finished`` ``failed``, and the connection keeps serving.
        """
        self._loop = asyncio.get_running_loop()
        try:
            device, devices = self._check(frame)
        except AssignmentRefused as exc:
            logger.warning("run %s refused: %s", frame.run_id, exc)
            if frame.run_id not in self.runs:  # never talk over a live run
                self._refused.add(frame.run_id)
                await self.send(wp.RunFinished(
                    run_id=frame.run_id, seq=1, status=wp.FINISHED_FAILED,
                    error=str(exc)))
            return
        try:
            await self._start(frame, device, devices)
        except Exception as exc:  # noqa: BLE001 -- reported, agent keeps serving
            logger.exception("run %s: could not start", frame.run_id)
            await self._abandon(frame.run_id, f"could not start the run "
                                              f"subprocess: {exc}")

    async def _abandon(self, run_id: str, error: str) -> None:
        """Fail a run whose start went wrong part-way, and clean it up."""
        handle = self.runs.pop(run_id, None)
        if handle is not None:
            handle.finished = True
            try:
                if handle.process.is_alive():
                    handle.process.kill()
            except Exception:  # noqa: BLE001
                pass
            handle.exited.set()
            handle.done.set()
        seq = handle.next_seq() if handle is not None else 1
        await self.send(wp.RunFinished(run_id=run_id, seq=seq,
                                       status=wp.FINISHED_FAILED, error=error))

    def available_mounts(self) -> dict[str, str]:
        """The configured mounts whose directory exists now (#782).

        What ``hello`` and every ``heartbeat`` advertise: a saved mount
        whose share is not mounted yet is left out until it appears, so
        the control plane never places a run that needs it here.
        """
        available: dict[str, str] = {}
        for label, path in sorted(self.mounts.items()):
            if not Path(path).is_dir():
                if label not in self._missing_logged:
                    self._missing_logged.add(label)
                    logger.warning("mount %s=%s is not a directory; not "
                                   "offering it until it is", label, path)
                continue
            available[label] = path
            if label in self._missing_logged:
                self._missing_logged.discard(label)
                logger.info("mount %s=%s is available now", label, path)
        return available

    def storage_plan(self, storage: wp.Storage | None
                     ) -> tuple[tuple[dict[str, str], list[str]] | None,
                                dict[str, str] | None]:
        """``(data_env, report_mount)`` for a run assigned with *storage*.

        The one place an assignment's storage becomes the child's paths
        (#782). ``data_env`` is the ``(set, unset)`` of the child's data
        settings: derived from this agent's own directory for the control
        plane's label (``--mount``), else the control plane's directories
        it lends a same-host worker (``storage.dirs``), else None.
        ``report_mount`` is ``{"label", "root"}`` when the child reports
        files under the mount relative to it: only for this agent's own
        mount, and not when the control plane asks for absolute paths
        (``relative_artifacts`` off: a same-host worker). Without
        ``storage`` (a peer from before #782) the label is ``data``.
        """
        label = storage.mount if storage is not None else \
            mounts_module.DEFAULT_LABEL
        own = self.available_mounts().get(label)
        if own:
            report = (storage is None or storage.relative_artifacts)
            return (mounts_module.data_root_env(own),
                    {"label": label, "root": own} if report else None)
        if storage is not None and storage.dirs:
            return (dict(storage.dirs), ["CODEFYUI_PROJECT_DIR"]), None
        return None, None

    async def _start(self, frame: wp.Assign, device: str,
                     devices: list[str]) -> None:
        run_dir = self.runs_dir / frame.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / OUTPUT_LOG
        # Off the loop: the mount check stats a network share.
        data_env, mount = await asyncio.to_thread(self.storage_plan,
                                                  frame.storage)
        env = child_environment(frame.device, devices, data_env)
        facts = run_facts(env)
        spec = {
            "run_id": frame.run_id, "graph": frame.graph,
            "options": frame.options, "device": device,
            "env": env, "log_path": str(log_path), "mount": mount,
        }
        events_in, events_out = self._ctx.Pipe(duplex=False)
        control_in, control_out = self._ctx.Pipe(duplex=False)
        process = self._ctx.Process(
            target=self.child_entry, args=(spec, events_out, control_in),
            name=f"codefyui-run-{frame.run_id}", daemon=False)
        try:
            process.start()
        except BaseException:
            for end in (events_in, events_out, control_in, control_out):
                end.close()
            raise
        # The child holds its own copies now. Closing ours is what lets the
        # reader see EOF when the child exits, however it exits.
        events_out.close()
        control_in.close()
        handle = RunHandle(
            run_id=frame.run_id, process=process, events_in=events_in,
            control_out=control_out, log_path=log_path,
            grace_s=frame.grace_s)
        self.runs[frame.run_id] = handle
        logger.info("run %s started in pid %s (device %s, devices %s)",
                    frame.run_id, process.pid, device, devices or "-")
        await self.send(wp.RunStarted(run_id=frame.run_id,
                                      seq=handle.next_seq(), pid=process.pid,
                                      execution_facts=facts))
        handle.reader = threading.Thread(
            target=self._read, args=(handle, self._loop),
            name=f"run-reader-{frame.run_id}", daemon=True)
        handle.reader.start()

    # ── cancel ────────────────────────────────────────────────────────────

    async def cancel(self, run_id: str, grace_s: float | None = None) -> bool:
        """Ask *run_id* to stop; escalate after *grace_s*. False if unknown.

        A run whose child already reported how it ended, but has not
        exited yet, is stopped at once (``terminate()``, then ``kill()``):
        there is nothing left to cancel cooperatively. Its report is not
        changed; it goes out once the process is gone.
        """
        handle = self.runs.get(run_id)
        if handle is None or handle.exited.is_set():
            return False
        if handle.finished:
            if not handle.stopping_now:
                handle.stopping_now = True
                if handle.escalation is not None:
                    handle.escalation.cancel()
                handle.escalation = asyncio.create_task(
                    self._escalate(handle, 0.0, "was cancelled after "
                                                "reporting"),
                    name=f"escalate:{run_id}")
            return True
        # cancel's grace, else the assignment's, else welcome's (else 30 s).
        grace = next(g for g in (grace_s, handle.grace_s,
                                 self.default_grace_s, wp.DEFAULT_GRACE_S)
                     if g is not None)
        if not handle.cancel_requested:
            handle.cancel_requested = True
            try:
                handle.control_out.send((child_module.MSG_CANCEL,))
            except (OSError, ValueError):
                pass  # the child is already gone; the reader reports it
        if handle.escalation is None:
            handle.escalation = asyncio.create_task(
                self._escalate(handle, grace, "did not stop"),
                name=f"escalate:{run_id}")
        return True

    def _watch_exit(self, handle: RunHandle) -> None:
        """The child reported: give it ``exit_after_finished_s`` to exit.

        Runs on the loop (the reader schedules it). An escalation already
        in flight (a cancel's) is left to finish the job.
        """
        if handle.escalation is None or handle.escalation.done():
            handle.escalation = asyncio.create_task(
                self._escalate(handle, self.exit_after_finished_s,
                               "is still alive after reporting"),
                name=f"exit:{handle.run_id}")

    async def _exits_within(self, handle: RunHandle, timeout_s: float) -> bool:
        if timeout_s <= 0:
            return handle.exited.is_set()
        try:
            await asyncio.wait_for(handle.exited.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    async def _escalate(self, handle: RunHandle, grace_s: float,
                        why: str) -> None:
        """Stop *handle*'s process: wait *grace_s*, terminate, then kill.

        The one escalation path, for a cancel (after its grace period), for
        a child that lingers after reporting, and for a cancel of such a
        child (no grace). ``terminated`` / ``killed`` record what was sent,
        which is what ``_unreported_end`` reads. Only the reader thread
        waits on the process; this waits for ``exited``, which it sets.
        """
        if await self._exits_within(handle, grace_s):
            return
        logger.warning("run %s %s (%.1f s); terminating", handle.run_id, why,
                       grace_s)
        handle.terminated = True
        handle.process.terminate()
        if await self._exits_within(handle, self.kill_after_s):
            return
        logger.warning("run %s survived terminate(); killing", handle.run_id)
        handle.killed = True
        handle.process.kill()

    async def shutdown(self, grace_s: float | None = None) -> None:
        """Stop every run (cooperatively, then by force) and wait for them."""
        if grace_s is None:
            grace_s = self.shutdown_grace_s
        self.draining = True
        handles = list(self.runs.values())
        for handle in handles:
            await self.cancel(handle.run_id, grace_s)
        for handle in handles:
            await handle.done.wait()

    # ── the reader thread ─────────────────────────────────────────────────

    def _forward(self, loop: asyncio.AbstractEventLoop, frame: Any) -> None:
        """Hand *frame* to the connection and wait until it has taken it."""
        try:
            asyncio.run_coroutine_threadsafe(self.send(frame), loop).result()
        except Exception:  # noqa: BLE001 -- a dead loop: nobody to tell
            logger.debug("could not forward a frame", exc_info=True)

    def _frame(self, handle: RunHandle, message: Any) -> Any:
        """The protocol frame for one pipe message, or None to skip it."""
        if not isinstance(message, tuple) or not message:
            return None
        kind = message[0]
        run_id = handle.run_id
        if kind == child_module.MSG_EVENT:
            _, event_type, payload = message
            if (event_type == "node_status" and isinstance(payload, dict)
                    and payload.get("status") in ("running", "progress")):
                handle.last_node = payload.get("node_id")
            return wp.Event(run_id=run_id, seq=handle.next_seq(),
                            event_type=event_type, payload=payload)
        if kind == child_module.MSG_METRICS:
            return wp.Metrics(run_id=run_id, seq=handle.next_seq(),
                              points=message[1])
        if kind == child_module.MSG_ARTIFACT:
            return wp.Artifact(run_id=run_id, seq=handle.next_seq(),
                               **message[1])
        if kind == child_module.MSG_WARNING:
            return wp.RunWarning(run_id=run_id, seq=handle.next_seq(),
                                 **message[1])
        if kind == child_module.MSG_FINISHED:
            body = message[1]
            status = body.get("status")
            if status not in ("succeeded", "failed", "cancelled"):
                status, error = wp.FINISHED_FAILED, (
                    f"the run subprocess reported an unknown status {status!r}")
            else:
                error = body.get("error")
            return wp.RunFinished(
                run_id=run_id, seq=handle.next_seq(), status=status,
                error=error, peak_memory=_as_dict(body.get("peak_memory")))
        logger.debug("run %s: ignoring pipe message %r", run_id, kind)
        return None

    def _read(self, handle: RunHandle, loop: asyncio.AbstractEventLoop) -> None:
        """Thread body: forward the child's messages, then report its end.

        ``run_finished`` goes out only once the child process is gone,
        whether it said how the run ended or not. This thread is the only
        one that waits on the process.
        """
        reported: wp.RunFinished | None = None
        while True:
            try:
                message = handle.events_in.recv()
            except (EOFError, OSError):
                break
            except Exception:  # noqa: BLE001 -- an unpicklable message
                logger.warning("run %s: unreadable pipe message",
                               handle.run_id, exc_info=True)
                continue
            try:
                frame = self._frame(handle, message)
            except Exception:  # noqa: BLE001 -- a malformed message
                logger.warning("run %s: dropped a malformed pipe message",
                               handle.run_id, exc_info=True)
                continue
            if frame is None:
                continue
            if isinstance(frame, wp.RunFinished):
                # Held until the child has exited: the control plane frees
                # the run's device slot on run_finished, and the next run
                # must not start next to this one's CUDA context.
                handle.finished = True
                reported = frame
                break
            self._forward(loop, frame)
        if reported is not None:
            try:
                loop.call_soon_threadsafe(self._watch_exit, handle)
            except RuntimeError:  # the loop is gone: nobody to escalate
                pass
            self._drain(handle)
        handle.process.join()
        try:
            loop.call_soon_threadsafe(handle.exited.set)
        except RuntimeError:
            pass
        if reported is None:
            # Decided before ``finished`` is set, so a cancel landing now
            # takes the ordinary path and cannot touch the flags it reads.
            reported = self._unreported_end(handle)
            handle.finished = True
        elif handle.discarded:
            logger.warning("run %s: dropped %d pipe message(s) the subprocess "
                           "sent after reporting", handle.run_id,
                           handle.discarded)
        self._forward(loop, reported)
        for end in (handle.events_in, handle.control_out):
            try:
                end.close()
            except OSError:
                pass
        logger.info("run %s: subprocess exited with code %s", handle.run_id,
                    handle.process.exitcode)
        try:
            loop.call_soon_threadsafe(self._retire, handle)
        except RuntimeError:  # the agent's loop is gone: nobody to retire for
            pass

    def _drain(self, handle: RunHandle) -> None:
        """Read and drop the pipe until EOF or the process exits. Blocking.

        After ``finished`` nothing the child sends is forwarded, but it is
        still read, so a child still writing cannot block on a full pipe
        while the escalation waits for it to exit.
        """
        while True:
            try:
                if handle.events_in.poll(DRAIN_POLL_S):
                    handle.events_in.recv()
                    handle.discarded += 1
                    logger.debug("run %s: dropped a pipe message sent after "
                                 "reporting", handle.run_id)
                    continue
            except (EOFError, OSError):
                return
            except Exception:  # noqa: BLE001 -- an unpicklable message
                handle.discarded += 1
                continue
            if not handle.process.is_alive():
                return

    def _unreported_end(self, handle: RunHandle) -> wp.RunFinished:
        """``run_finished`` for a child that exited without saying how.

        ``interrupted`` only when the exit is the one a stop explains: the
        signal this agent sent after the grace period, or a clean exit
        while a cancel was pending. Any other exit is a crash (``failed``),
        cancel or not. The output tail rides on every one of them.
        """
        code = handle.process.exitcode
        tail = read_tail(handle.log_path)
        expected: set[int] = set()
        if handle.terminated:
            expected.add(-signal.SIGTERM)
        if handle.killed:
            expected.add(-getattr(signal, "SIGKILL", signal.SIGTERM))
        if code in expected:
            error = ("the run did not stop within its grace period and was "
                     "terminated")
        elif handle.cancel_requested and code == 0:
            error = "the run subprocess exited while stopping"
        else:
            return wp.RunFinished(
                run_id=handle.run_id, seq=handle.next_seq(),
                status=wp.FINISHED_FAILED, exit_code=code,
                error=f"run subprocess exited with code {code}",
                stderr_tail=tail, last_node=handle.last_node)
        return wp.RunFinished(
            run_id=handle.run_id, seq=handle.next_seq(),
            status=wp.FINISHED_INTERRUPTED, exit_code=code, error=error,
            stderr_tail=tail, last_node=handle.last_node)

    def _retire(self, handle: RunHandle) -> None:
        self.runs.pop(handle.run_id, None)
        if handle.escalation is not None and not handle.escalation.done():
            handle.escalation.cancel()
        handle.done.set()
        task = asyncio.ensure_future(asyncio.to_thread(
            self._retire_files, handle.log_path.parent))
        self._housekeeping.add(task)
        task.add_done_callback(self._housekeeping.discard)

    def _retire_files(self, run_dir: Path) -> None:
        """Mark *run_dir* finished and apply retention. Blocking."""
        mark_finished(run_dir)
        prune_run_dirs(self.runs_dir, self.keep_last)


async def _drop(frame: Any) -> None:  # pragma: no cover - placeholder sink
    logger.debug("no connection; dropped %s", type(frame).__name__)
