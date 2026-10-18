"""The control plane's reads of a worker run's recorded outputs (#789).

A worker run with ``record_outputs`` keeps its captures on its worker
(``agent.outputs``). :class:`WorkerOutputs` turns one read of
``/api/execution/outputs/*`` into a ``fetch_output`` frame over the
worker's ``/ws/worker`` connection and waits for its ``output_answer``
parts. The worker dialled in, so the request rides on its connection: the
control plane still never connects to a worker.

A read never hangs on a worker that cannot answer:

- the worker is not connected (or never was since this server started,
  or its agent does not list ``record_outputs``): :class:`WorkerUnreachable`
  at once;
- its connection closes while the read waits: every read it had in flight
  ends with :class:`WorkerUnreachable` (``WorkerRegistry.disconnect`` and a
  replacing connection call :meth:`WorkerOutputs.worker_gone`);
- no answer within ``CODEFYUI_WORKER_OUTPUT_FETCH_TIMEOUT_S``:
  :class:`WorkerTimeout`.

The wait rides on ``fetch_output`` (``timeout_s``), so the agent gives up
first; a read this side stops waiting for (a timeout, or the browser went
away) is cancelled on the worker with ``cancel_fetch``. A ``drop_outputs``
for a worker that is not connected waits for its next connection
(:meth:`WorkerOutputs.flush_drops`).

The routes answer the first with 503 and the second with 504, each with a
detail naming the worker. Event-loop only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any

from . import worker_protocol as wp

logger = logging.getLogger(__name__)

#: Seconds a read waits for its answer unless the caller says otherwise.
DEFAULT_TIMEOUT_S = 60.0
#: ``drop_outputs`` kept per worker while it is away; the oldest go first
#: (its own budget removes those in time).
PENDING_DROPS_MAX = 1000


class WorkerUnreachable(RuntimeError):
    """The worker holding the run's outputs cannot be asked now."""


class WorkerTimeout(RuntimeError):
    """The worker did not answer in time."""


def unreachable_detail(worker: str, why: str) -> str:
    """The 503 detail: which worker, why, and what to do."""
    return (f"the outputs this run recorded are kept on worker {worker!r}, "
            f"which {why}; they can be read again once it is connected")


@dataclass(frozen=True)
class RemoteAnswer:
    """A worker's answer to one read."""

    status: int
    #: The JSON body, parsed; None for a 204.
    body: Any
    #: The run has ended on the worker (its captures are final).
    final: bool


@dataclass(eq=False)
class _Pending:
    worker: str
    future: asyncio.Future
    parts: dict[int, str] = field(default_factory=dict)
    expected: int | None = None


class WorkerOutputs:
    """Reads in flight to workers, by request id. One per registry."""

    def __init__(self, registry: Any, *,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._registry = registry
        self.timeout_s = timeout_s
        self._pending: dict[str, _Pending] = {}
        self._tasks: set[asyncio.Task] = set()
        #: Worker name -> run ids to drop there once it is connected.
        self._drops: dict[str, dict[str, None]] = {}

    def _reachable(self, worker_id: str) -> Any:
        worker = self._registry.get(worker_id)
        if worker is None:
            raise WorkerUnreachable(unreachable_detail(
                worker_id, "has not connected since this server started"))
        if not worker.online:
            raise WorkerUnreachable(unreachable_detail(
                worker_id, "is not connected"))
        if wp.CAPABILITY_RECORD_OUTPUTS not in (worker.capabilities or ()):
            raise WorkerUnreachable(unreachable_detail(
                worker_id, "runs an agent that cannot serve them"))
        return worker

    async def fetch(self, worker_id: str, run_id: str, op: str,
                    params: dict[str, Any] | None = None, *,
                    timeout_s: float | None = None) -> RemoteAnswer:
        """Ask *worker_id* for one read of *run_id*'s outputs."""
        worker = self._reachable(worker_id)
        request_id = secrets.token_hex(12)
        pending = _Pending(worker=worker_id,
                           future=asyncio.get_running_loop().create_future())
        self._pending[request_id] = pending
        wait = self.timeout_s if timeout_s is None else timeout_s
        sent = False
        try:
            try:
                await worker.send(wp.FetchOutput(
                    request_id=request_id, run_id=run_id, op=op,
                    params=dict(params or {}), timeout_s=wait))
                sent = True
            except Exception as exc:  # noqa: BLE001 -- a closing socket
                raise WorkerUnreachable(unreachable_detail(
                    worker_id, f"could not be sent the request ({exc})")) \
                    from None
            try:
                return await asyncio.wait_for(asyncio.shield(pending.future),
                                              timeout=wait)
            except asyncio.TimeoutError:
                raise WorkerTimeout(
                    f"worker {worker_id!r} did not answer within {wait:g} s "
                    "(CODEFYUI_WORKER_OUTPUT_FETCH_TIMEOUT_S)") from None
        finally:
            self._pending.pop(request_id, None)
            if not pending.future.done():
                # Nobody waits for it any more: the worker stops answering.
                pending.future.cancel()
                if sent:
                    self._send_soon(worker_id, wp.CancelFetch(
                        request_id=request_id))

    def deliver(self, worker_id: str, frame: wp.OutputAnswer) -> None:
        """One ``output_answer`` part from *worker_id*'s connection."""
        pending = self._pending.get(frame.request_id)
        if pending is None or pending.worker != worker_id \
                or pending.future.done():
            logger.debug("worker %s answered request %s, which nothing "
                         "waits for", worker_id, frame.request_id)
            return
        if pending.expected is None:
            pending.expected = frame.parts
        if frame.parts != pending.expected or frame.part >= frame.parts:
            pending.future.set_exception(WorkerUnreachable(
                f"worker {worker_id!r} sent a malformed answer"))
            return
        pending.parts[frame.part] = frame.body
        if len(pending.parts) < pending.expected:
            return
        text = "".join(pending.parts[i] for i in range(pending.expected))
        try:
            body = json.loads(text) if text else None
        except ValueError as exc:
            pending.future.set_exception(WorkerUnreachable(
                f"worker {worker_id!r} sent an answer that is not JSON: "
                f"{exc}"))
            return
        pending.future.set_result(RemoteAnswer(status=frame.status,
                                               body=body, final=frame.final))

    def worker_gone(self, worker_id: str, reason: str) -> None:
        """End every read in flight to *worker_id*: its connection is gone."""
        for pending in list(self._pending.values()):
            if pending.worker == worker_id and not pending.future.done():
                pending.future.set_exception(WorkerUnreachable(
                    unreachable_detail(worker_id, f"disconnected ({reason})")))

    def in_flight(self) -> int:
        return len(self._pending)

    def _capable(self, worker_id: str) -> Any:
        worker = self._registry.get(worker_id)
        if worker is None or not worker.online or \
                wp.CAPABILITY_RECORD_OUTPUTS not in (worker.capabilities or ()):
            return None
        return worker

    def _send_soon(self, worker_id: str, frame: Any) -> bool:
        """Send *frame* to *worker_id* in a task; False when it cannot
        be (not connected, or no loop)."""
        worker = self._capable(worker_id)
        if worker is None:
            return False
        try:
            task = asyncio.get_running_loop().create_task(worker.send(frame))
        except RuntimeError:  # no loop: nothing can be sent
            return False
        self._tasks.add(task)
        task.add_done_callback(self._sent)
        return True

    def drop(self, run_id: str, worker_id: str) -> None:
        """Tell *worker_id* its captures of *run_id* are no longer wanted.

        Fire and forget, for ``RunOutputStore.on_remote_drop``. A worker
        that is not connected is told when it next connects
        (:meth:`flush_drops`); its own budget removes the captures in time
        regardless (``agent.outputs``).
        """
        if self._send_soon(worker_id, wp.DropOutputs(run_id=run_id)):
            return
        waiting = self._drops.setdefault(worker_id, {})
        waiting[run_id] = None
        while len(waiting) > PENDING_DROPS_MAX:
            del waiting[next(iter(waiting))]
        logger.info("run %s's outputs are no longer retained; worker %s is "
                    "told when it connects", run_id, worker_id)

    def pending_drops(self, worker_id: str) -> list[str]:
        return list(self._drops.get(worker_id, ()))

    def flush_drops(self, worker_id: str) -> int:
        """*worker_id* connected: send the drops it missed. Returns how
        many were sent. An agent that cannot serve outputs has none to
        drop: its list is discarded."""
        waiting = self._drops.pop(worker_id, None)
        if not waiting:
            return 0
        if self._capable(worker_id) is None:
            worker = self._registry.get(worker_id)
            if worker is None or not worker.online:
                self._drops[worker_id] = waiting  # not connected after all
            return 0
        for run_id in waiting:
            self._send_soon(worker_id, wp.DropOutputs(run_id=run_id))
        return len(waiting)

    def _sent(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.info("drop_outputs not sent: %s", task.exception())
