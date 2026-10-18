"""A worker run's recorded outputs, kept and served by the agent (#789).

A run assigned with ``record_outputs`` records its captures in its run
subprocess's own ``RunOutputStore`` (``agent.child``), exactly as a run in
the control plane does. When the run ends the subprocess copies them off
its cards, reports, and writes them to ``<runs_dir>/<run_id>/outputs/``
(``core.output_queries.write_slots``: one ``torch.save`` file per captured
port and an index), so they outlive it.

The control plane reads them on demand: ``fetch_output`` frames, one per
``/api/execution/outputs/*`` read (``core.worker_protocol``).
:class:`AgentOutputs` answers each one, in a task of its own,

- from the run subprocess while it lives (``Runner.ask_child``): it reads
  its live store, then the copies, then the files, with the same
  ``output_queries`` functions the control plane uses;
- after that, from the files, through :class:`OutputsHelper`: a
  subprocess of its own that loads them (memory-mapped) and answers.
  The agent process never imports torch; the helper does, with no CUDA
  card visible. It starts on the first read (or when a recorded run
  ends) and exits after :data:`HELPER_IDLE_S` without one.

Every wait here ends before the control plane's (``fetch_output``'s
``timeout_s`` less :data:`ANSWER_MARGIN_S`), and a ``cancel_fetch`` ends
one at once, so a read the control plane gave up on never holds one of
the :data:`MAX_CONCURRENT` places.

The answer goes back as ``output_answer`` frames (the connection cuts the
body into parts). ``delete`` and ``drop_outputs`` remove a run's captures:
the dump's directory, and what a run subprocess still holds.

Budget: the dumps on this worker are held to the in-process store's
limits, ``CODEFYUI_RUN_OUTPUT_STORE_MAX_MB`` (as set on THIS machine) and
:data:`MAX_RUNS` runs, oldest first and never the run just written
(:func:`prune_outputs`), the same rule ``RunOutputStore`` applies. The
control plane's own retention also evicts runs by id (``drop_outputs``).

This module imports neither torch nor the engine at module level; only
:func:`helper_main` (in the helper subprocess) does.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..core import worker_protocol as wp
from ..core.output_queries import INDEX_FILE, OUTPUTS_DIRNAME, not_captured
from ..core.run_output_store import MAX_RUNS as STORE_MAX_RUNS

logger = logging.getLogger(__name__)

#: Runs whose dumps a worker keeps: the in-process store's run bound.
MAX_RUNS = STORE_MAX_RUNS
#: Seconds the helper subprocess stays up without a read.
HELPER_IDLE_S = 300.0
#: The control plane's wait for an answer when ``fetch_output`` does not
#: say (its ``CODEFYUI_WORKER_OUTPUT_FETCH_TIMEOUT_S`` default).
DEFAULT_FETCH_TIMEOUT_S = 60.0
#: Seconds the agent's own wait ends before the control plane's: its 504
#: then reaches the browser instead of the control plane's.
ANSWER_MARGIN_S = 5.0
#: Reads answered at once (each may be a large tensor's slice).
MAX_CONCURRENT = 4

#: Helper pipe messages: agent -> helper, then helper -> agent.
MSG_FETCH = "fetch"
MSG_ANSWER = "output_answer"

#: ``(status, body text, final)``.
Answer = tuple[int, str, bool]
SendAnswer = Callable[[str, int, str, bool], Awaitable[Any]]


def outputs_dir(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id / OUTPUTS_DIRNAME


def answer_timeout(frame: wp.FetchOutput) -> float:
    """Seconds the agent works on *frame*: less than the control plane
    waits for it."""
    wait = frame.timeout_s or DEFAULT_FETCH_TIMEOUT_S
    return max(wait - ANSWER_MARGIN_S, wait / 2)


def _error(status: int, detail: str) -> Answer:
    return status, json.dumps({"detail": detail}), True


def _dumps(directory: Path) -> list[tuple[float, int, Path]]:
    """``(written at, bytes, directory)`` of every complete dump under the
    runs directory, newest first."""
    found: list[tuple[float, int, Path]] = []
    try:
        run_dirs = list(directory.iterdir())
    except OSError:
        return found
    for run_dir in run_dirs:
        index = run_dir / OUTPUTS_DIRNAME / INDEX_FILE
        try:
            written = index.stat().st_mtime
            total = int(json.loads(index.read_text(encoding="utf-8"))
                        .get("bytes") or 0)
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        found.append((written, total, run_dir / OUTPUTS_DIRNAME))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def prune_outputs(runs_dir: Path, max_bytes: int, max_runs: int = MAX_RUNS,
                  keep: str | None = None) -> list[str]:
    """Delete the oldest dumps until at most *max_runs* remain and they
    hold at most *max_bytes* (0: no byte bound). *keep* (the run just
    written) is never deleted, so a run whose own captures exceed the
    budget exceeds it alone, as in ``RunOutputStore``. Returns the run ids
    deleted. Blocking."""
    dumps = _dumps(runs_dir)
    total = sum(nbytes for _, nbytes, _ in dumps)
    count = len(dumps)
    dropped: list[str] = []
    for _, nbytes, directory in reversed(dumps):  # oldest first
        if count <= max_runs and (not max_bytes or total <= max_bytes):
            break
        run_id = directory.parent.name
        if run_id == keep:
            continue
        shutil.rmtree(directory, ignore_errors=True)
        dropped.append(run_id)
        count -= 1
        total -= nbytes
    if dropped:
        logger.info("recorded outputs over this worker's budget; dropped "
                    "those of %s", ", ".join(dropped))
    return dropped


# ── the helper subprocess ────────────────────────────────────────────────────


def helper_main(requests: Any, answers: Any) -> None:
    """Helper ``Process`` target: answer reads of dumps until the pipe
    closes. Imports torch (CPU only)."""
    import os
    import signal
    from concurrent.futures import ThreadPoolExecutor

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ..core import output_queries as q

    lock = threading.Lock()

    def answer(request_id: str, directory: str, op: str, run_id: str,
               params: dict[str, Any]) -> None:
        status, text = q.answer_read(q.DiskRunReader(Path(directory)), op,
                                     run_id, params)
        with lock:
            try:
                answers.send((MSG_ANSWER, request_id, status, text, True))
            except (OSError, ValueError, EOFError):
                pass

    with ThreadPoolExecutor(MAX_CONCURRENT,
                            thread_name_prefix="outputs") as pool:
        while True:
            try:
                message = requests.recv()
            except (EOFError, OSError):
                break
            if isinstance(message, tuple) and len(message) == 6 \
                    and message[0] == MSG_FETCH:
                pool.submit(answer, *message[1:])


class _Instance:
    """One helper process: its pipe and the reads waiting on it. Its
    state never passes to another instance, so a helper exiting fails
    only its own reads."""

    def __init__(self, process: Any, requests: Any) -> None:
        self.process = process
        self.requests = requests
        self.pending: dict[str, asyncio.Future] = {}
        #: Closed by :meth:`OutputsHelper.close`, or gone: takes no reads.
        self.closed = False

    @property
    def usable(self) -> bool:
        return not self.closed and self.process.is_alive()

    def fail(self, answer: Answer) -> None:
        for future in list(self.pending.values()):
            if not future.done():
                future.set_result(answer)

    def close(self) -> None:
        self.closed = True
        try:
            self.requests.close()
        except OSError:
            pass


class OutputsHelper:
    """The helper subprocess, started on demand and stopped when idle.

    Every method runs on the agent's event loop; the reader thread reaches
    it through ``call_soon_threadsafe``. Each start makes a new
    :class:`_Instance`, and every transition (start, close, exit) changes
    only the instance it is about.
    """

    def __init__(self, *, idle_s: float = HELPER_IDLE_S,
                 timeout_s: float = DEFAULT_FETCH_TIMEOUT_S - ANSWER_MARGIN_S,
                 entry: Callable[..., None] = helper_main) -> None:
        self.idle_s = idle_s
        self.timeout_s = timeout_s
        self.entry = entry
        self._ctx = multiprocessing.get_context("spawn")
        self._current: _Instance | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_used = time.monotonic()
        self._idle_check: asyncio.TimerHandle | None = None
        self._counter = 0

    @property
    def running(self) -> bool:
        return self._current is not None and self._current.usable

    def start(self) -> _Instance:
        """The helper, started now unless it is up (a recorded run ended:
        its first read will not wait for torch to import)."""
        self._loop = asyncio.get_running_loop()
        self._last_used = time.monotonic()
        current = self._current
        if current is not None and current.usable:
            return current
        if current is not None:
            current.close()  # exiting: its reader fails its reads
        requests_out, requests_in = self._ctx.Pipe(duplex=False)
        answers_in, answers_out = self._ctx.Pipe(duplex=False)
        process = self._ctx.Process(target=self.entry,
                                    args=(requests_out, answers_out),
                                    name="codefyui-outputs", daemon=True)
        process.start()
        requests_out.close()
        answers_out.close()
        instance = _Instance(process, requests_in)
        self._current = instance
        threading.Thread(target=self._read, args=(instance, answers_in,
                                                  self._loop),
                         name="outputs-helper-reader", daemon=True).start()
        logger.info("outputs helper started (pid %s)", process.pid)
        self._schedule_idle_check()
        return instance

    async def ask(self, directory: Path, run_id: str, op: str,
                  params: dict[str, Any], *,
                  timeout_s: float | None = None) -> Answer:
        """One read of the dump in *directory*. Cancelling it forgets it."""
        instance = self.start()
        self._counter += 1
        request_id = f"h{self._counter}"
        future = asyncio.get_running_loop().create_future()
        instance.pending[request_id] = future
        wait = self.timeout_s if timeout_s is None else timeout_s
        try:
            try:
                instance.requests.send((MSG_FETCH, request_id,
                                        str(directory), op, run_id, params))
            except (OSError, ValueError) as exc:
                return _error(500, f"the outputs helper is not reachable: "
                                   f"{exc}")
            try:
                return await asyncio.wait_for(asyncio.shield(future), wait)
            except asyncio.TimeoutError:
                return _error(504, f"reading the recorded outputs took "
                                   f"over {wait:g} s on the worker")
        finally:
            instance.pending.pop(request_id, None)
            self._last_used = time.monotonic()

    def _read(self, instance: _Instance, answers: Any,
              loop: asyncio.AbstractEventLoop) -> None:
        """Thread body: hand each answer to its read; on EOF, fail the
        reads still waiting on *instance* (its process is gone)."""
        while True:
            try:
                message = answers.recv()
            except (EOFError, OSError):
                break
            if isinstance(message, tuple) and len(message) == 5 \
                    and message[0] == MSG_ANSWER:
                _, request_id, status, text, final = message
                try:
                    loop.call_soon_threadsafe(self._resolve, instance,
                                              request_id,
                                              (status, text, final))
                except RuntimeError:
                    break
        try:
            answers.close()
        except OSError:
            pass
        instance.process.join()
        try:
            loop.call_soon_threadsafe(self._exited, instance)
        except RuntimeError:
            pass

    @staticmethod
    def _resolve(instance: _Instance, request_id: str,
                 answer: Answer) -> None:
        future = instance.pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(answer)

    def _exited(self, instance: _Instance) -> None:
        instance.closed = True
        if self._current is instance:
            self._current = None
        instance.fail(_error(500, "the outputs helper exited "
                                  f"(code {instance.process.exitcode})"))

    def _schedule_idle_check(self) -> None:
        if self._loop is None or self._idle_check is not None:
            return
        self._idle_check = self._loop.call_later(self.idle_s / 4 or 1,
                                                 self._check_idle)

    def _check_idle(self) -> None:
        self._idle_check = None
        current = self._current
        if current is None or not current.usable:
            return
        if not current.pending \
                and time.monotonic() - self._last_used >= self.idle_s:
            logger.info("outputs helper idle for %.0f s; stopping it",
                        self.idle_s)
            self.close()
            return
        self._schedule_idle_check()

    def close(self) -> None:
        """Close the request pipe: the helper finishes and exits. The next
        read starts a new one."""
        if self._idle_check is not None:
            self._idle_check.cancel()
            self._idle_check = None
        current, self._current = self._current, None
        if current is not None:
            current.close()


# ── the agent's side ─────────────────────────────────────────────────────────


class AgentOutputs:
    """Answers ``fetch_output``, ``cancel_fetch`` and ``drop_outputs`` for
    one agent, each in a task of its own: a large read or a slow delete
    never holds up the cancels and acks behind it."""

    def __init__(self, runner: Any, send_answer: SendAnswer, *,
                 helper: OutputsHelper | None = None) -> None:
        self.runner = runner
        self.send_answer = send_answer
        self.helper = helper if helper is not None else OutputsHelper()
        self._slots = asyncio.Semaphore(MAX_CONCURRENT)
        #: Reads being answered, by request id.
        self._reads: dict[str, asyncio.Task] = {}
        #: Drops in progress, kept so they are not collected mid-flight.
        self._drops: set[asyncio.Task] = set()

    def start(self, frame: wp.FetchOutput) -> asyncio.Task:
        """Answer *frame* in a task of its own."""
        task = asyncio.create_task(self.serve(frame),
                                   name=f"outputs:{frame.run_id}")
        self._reads[frame.request_id] = task

        def done(_: asyncio.Task) -> None:
            if self._reads.get(frame.request_id) is task:
                del self._reads[frame.request_id]

        task.add_done_callback(done)
        return task

    def cancel(self, request_id: str) -> bool:
        """``cancel_fetch``: stop answering *request_id*. False when it is
        not being answered (already done)."""
        task = self._reads.get(request_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def start_drop(self, run_id: str) -> asyncio.Task:
        task = asyncio.create_task(self.drop(run_id),
                                   name=f"drop-outputs:{run_id}")
        self._drops.add(task)
        task.add_done_callback(self._drops.discard)
        return task

    async def serve(self, frame: wp.FetchOutput) -> None:
        """Answer *frame*. Never raises but cancel (``cancel_fetch``: no
        answer is sent; the control plane stopped waiting)."""
        try:
            async with self._slots:
                status, text, final = await self._answer(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- answered, not raised
            logger.exception("answering a read of run %s's outputs failed",
                             frame.run_id)
            status, text, final = _error(500, f"the worker could not read "
                                              f"the outputs: {exc}")
        try:
            await self.send_answer(frame.request_id, status, text, final)
        except Exception:  # noqa: BLE001 -- the connection is gone
            logger.debug("could not send an output answer", exc_info=True)

    async def _answer(self, frame: wp.FetchOutput) -> Answer:
        if frame.op == "delete":
            return await self._delete(frame.run_id)
        timeout_s = answer_timeout(frame)
        answer = await self.runner.ask_child(frame.run_id, frame.op,
                                             dict(frame.params),
                                             timeout_s=timeout_s)
        if answer is not None:
            return answer
        directory = outputs_dir(self.runner.runs_dir, frame.run_id)
        if not (directory / INDEX_FILE).exists():
            # Nothing on disk (never recorded, evicted, or deleted): the
            # answer the reads give for a run they do not hold, without
            # starting the helper for it.
            return _error(404, f"run '{frame.run_id}' not found"
                          if frame.op != "stats" else not_captured(
                              f"run '{frame.run_id}' has no captured "
                              "outputs"))
        return await self.helper.ask(directory, frame.run_id, frame.op,
                                     dict(frame.params), timeout_s=timeout_s)

    async def _delete(self, run_id: str) -> Answer:
        live = self.runner.child_drop(run_id)
        removed = await asyncio.to_thread(self._remove, run_id)
        if not (live or removed):
            return _error(404, f"run '{run_id}' not found")
        return 200, json.dumps({"run_id": run_id, "deleted": True}), True

    def _remove(self, run_id: str) -> bool:
        directory = outputs_dir(self.runner.runs_dir, run_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    async def drop(self, run_id: str) -> None:
        """``drop_outputs``: the control plane no longer retains the run."""
        self.runner.child_drop(run_id)
        if await asyncio.to_thread(self._remove, run_id):
            logger.info("dropped the recorded outputs of run %s (the control "
                        "plane no longer retains it)", run_id)

    def close(self) -> None:
        for task in list(self._reads.values()):
            task.cancel()
        self.helper.close()
