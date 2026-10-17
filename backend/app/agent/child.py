"""The run subprocess: executes one assigned run and reports over a pipe (#771).

The agent starts one of these per assignment with
``multiprocessing.get_context("spawn")``, so it is a fresh interpreter: the
first thing :func:`main` does is put the environment the agent chose in
place (``CUDA_VISIBLE_DEVICES`` and friends), BEFORE anything imports
torch. This module therefore imports nothing heavy at module level; the
engine comes in inside :func:`run_assignment`.

How the run executes is ``InProcessExecutor``'s business, unchanged: the
child builds the ``ExecutionContext`` the way ``RunService._start`` does,
hands ``InProcessExecutor`` a ``RunSinks`` whose four callables write
messages to the pipe instead of to a ``RunService``, and reports the
outcome. The payloads are built by the same ``core.run_events`` builders in
the same order, which is what makes a worker run's event log byte-identical
to a local one's.

What the child never does: open SQLite (it has no ``RunStore``; the agent
does not even pass ``CODEFYUI_DB_PATH``), touch a ``NodeStateStore`` (a
worker run is isolated, ``weights_persistent=False``).

Recorded outputs (#789): a run with ``options.record_outputs`` captures
into a ``RunOutputStore`` of this process's own, exactly as a run on the
control plane does (:class:`LiveOutputs`). When the run ends, before
``finished``, the store is written to ``spec["outputs_dir"]``
(``output_queries.dump_run``), so the agent can serve it after this process
exits. Until this process exits it answers the agent's reads itself: from
the store while the run goes, from the files once they are written.

Pipe messages, child -> agent (tuples, pickled by ``multiprocessing``):

- ``("event", event_type, payload)``
- ``("metrics", [{"name", "value", "step", "node_id"}, ...])`` (never empty)
- ``("artifact", {"kind", "path", "meta", "node_id", "mount_label"})``:
  ``path`` relative to the run's mount (``spec["mount"]``, #782) when the
  file is under it, with ``mount_label`` naming it; else absolute, with
  ``mount_label`` None.
- ``("warning", {"kind", "detail", "node_id"})``
- ``("output_answer", request_id, status, body_text, final)``: the answer
  to a ``fetch`` (#789), the read's HTTP status and JSON body text;
  ``final`` once it came from the files.
- ``("finished", {"status", "error", "peak_memory"})``, last (an
  ``output_answer`` may still follow it).
- ``("memory_limit", {device: {"budget_bytes", "total_bytes",
  "fraction"}})``, FIRST, for a run with memory budgets (#788,
  ``spec["memory_budgets"]``, as this process numbers its cards): each
  card's caching allocator is capped with
  ``set_per_process_memory_fraction(budget / total)`` before anything
  else touches torch, so a run that outgrows its budget runs out of memory
  alone. The agent sends ``run_started`` on this message. A cap that
  cannot be set fails the run before it starts (``finished`` with no
  ``memory_limit``).

Paths on a shared mount (#782) are mapped at exactly two places, both fed
by the assignment (``spec["mount"]``, built once by ``Runner.storage_plan``):
in, the data settings are derived from the mount root before anything
imports ``app.config`` (``runner.child_environment``), so the graph's own
paths (an upload's bare name, ``models/x.pt``) resolve unchanged and the
graph is never rewritten; out, every artifact path goes through
:func:`artifact_location`, the one place a path becomes relative.

Agent -> child: ``("cancel",)``; for a recorded run (#789) also
``("fetch", request_id, op, run_id, params)`` (a read of its outputs, see
``core.output_queries.dispatch``) and ``("drop_outputs", run_id)``. The
pipe closing means the agent is gone, which is treated as a cancel: nobody
is left to report to.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable

MSG_EVENT = "event"
MSG_METRICS = "metrics"
MSG_ARTIFACT = "artifact"
MSG_WARNING = "warning"
MSG_FINISHED = "finished"
MSG_MEMORY_LIMIT = "memory_limit"
MSG_CANCEL = "cancel"
MSG_FETCH = "fetch"
MSG_DROP_OUTPUTS = "drop_outputs"
MSG_OUTPUT_ANSWER = "output_answer"

Send = Callable[[tuple], None]


def main(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """``Process`` target. *spec* is built by ``Runner.assign``."""
    ignore_interrupts()
    apply_environment(spec.get("env") or {})
    _redirect_output(spec.get("log_path"))
    # Logging to the redirected stderr, so a node's warnings land in the
    # per-run log next to its prints.
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_assignment(spec, _safe_sender(events_out), control_in)
    try:
        events_out.close()
    except OSError:
        pass


def ignore_interrupts() -> None:
    """Leave Ctrl+C to the agent.

    A foreground agent shares its terminal's process group with its
    children, so Ctrl+C reaches every run subprocess at once. The agent
    answers it by cancelling each run (cooperatively, then by force); a
    child that died of the same keystroke would skip that and be reported
    as a crash.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def apply_environment(env: dict[str, Any]) -> None:
    """Apply ``{"set": {name: value}, "unset": [names]}`` to ``os.environ``."""
    for name in env.get("unset", ()):
        os.environ.pop(name, None)
    for name, value in (env.get("set") or {}).items():
        os.environ[name] = str(value)


def _redirect_output(log_path: str | None) -> None:
    """Send this process's stdout and stderr, Python's and the C runtime's,
    to the run's log file. The agent reads its tail when the run crashes."""
    if not log_path:
        return
    stream = open(log_path, "a", buffering=1, encoding="utf-8",  # noqa: SIM115
                  errors="replace")
    for fd in (1, 2):
        try:
            os.dup2(stream.fileno(), fd)
        except OSError:  # no such descriptor (a GUI-less Windows parent)
            pass
    sys.stdout = stream
    sys.stderr = stream


def _safe_sender(conn: Any) -> Send:
    """``conn.send``, minus the exception when the agent has gone away.

    A sink must not raise into the engine: a run whose reader vanished
    is stopped by the control listener (EOF is a cancel), not by a
    ``BrokenPipeError`` surfacing as a node failure. Locked: the answers to
    reads of the run's outputs (#789) are sent from other threads, and two
    writers must not interleave one message's bytes.
    """
    lock = threading.Lock()

    def send(message: tuple) -> None:
        try:
            with lock:
                conn.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    return send


def build_context(run_id: str, options: dict[str, Any], device: str) -> Any:
    """The run's ``ExecutionContext``, as ``RunService._start`` builds it.

    Same fields from the same options, except the two that would reach
    the control plane's process-local state: no ``node_state_store`` (and
    ``weights_persistent`` off, which with no store is what a server-owned
    run already gets) -- a worker run is isolated.
    """
    from ..config import settings
    from ..core.execution_context import ExecutionContext

    return ExecutionContext(
        execution_id=run_id,
        device=device,
        max_workers=settings.MAX_PARALLEL_NODES,
        verbose=bool(options.get("verbose")),
        graph_id=str(options.get("graph_id") or ""),
        backward_mode=bool(options.get("backward_mode")),
        auto_backward=bool(options.get("auto_backward")),
        seed=options.get("seed"),
        deterministic=bool(options.get("deterministic")),
        weights_persistent=False,
        node_state_store=None,
    )


def wire_payload(payload: Any, cap_bytes: int) -> Any:
    """*payload* as the control plane would store it: NaN-free
    (``json_safe``), capped (``cap_event_payload``, the same policy and
    ceiling ``RunService._emit`` applies) and plain JSON values.

    Applying both here keeps a giant payload off the pipe and the socket;
    the control plane applies them again, which changes nothing. The JSON
    round trip (``default=str``, the rule ``json_size`` measures with)
    turns anything without a JSON type into its ``str`` so the message can
    always be pickled. Raises when even that fails (a circular structure).
    """
    import json

    from ..core.run_events import cap_event_payload
    from ..core.run_store import json_safe

    capped = cap_event_payload(json_safe(payload), cap_bytes=cap_bytes)
    return json.loads(json.dumps(capped, allow_nan=False, default=str))


def artifact_location(path: str, mount: dict[str, Any] | None
                      ) -> tuple[str, str | None]:
    """``(path, mount_label)`` an artifact is reported with (#782).

    Relative to the run's mount when *path* is under its root, so the
    control plane can find it at its own path for the same label; else
    *path* as produced, with no label.
    """
    if mount and mount.get("root") and mount.get("label"):
        from ..core.mounts import relative_to_mount

        relative = relative_to_mount(path, mount["root"])
        if relative is not None:
            return relative, str(mount["label"])
    return path, None


def _pipe_sinks(send: Send, cap_bytes: int | None = None,
                mount: dict[str, Any] | None = None) -> Any:
    """A ``RunSinks`` whose callables write pipe messages. *mount* is the
    run's ``{"label", "root"}`` (#782), or None."""
    import logging

    from ..config import settings
    from ..core.executors import RunSinks

    log = logging.getLogger(__name__)
    if cap_bytes is None:
        cap_bytes = settings.RUN_EVENT_PAYLOAD_CAP_BYTES

    async def emit(event_type: str, payload: dict[str, Any] | None) -> None:
        try:
            wired = wire_payload(payload, cap_bytes)
        except (TypeError, ValueError, RecursionError) as exc:
            log.warning("a %s event could not be encoded: %s", event_type, exc)
            node_id = payload.get("node_id") if isinstance(payload, dict) else None
            send((MSG_WARNING, {
                "kind": "unserialisable_event",
                "detail": f"a {event_type} event could not be encoded: {exc}",
                "node_id": node_id if isinstance(node_id, str) else None}))
            return
        send((MSG_EVENT, event_type, wired))

    async def record_metrics(points: list[Any]) -> None:
        # The engine bridge calls this with [] on every progress frame to
        # let the control plane's batcher check its clock; that poke is the
        # consumer's to make (see worker_protocol), not worth a message.
        if points:
            send((MSG_METRICS, [
                {"name": p.name, "value": p.value, "step": p.step,
                 "node_id": p.node_id} for p in points]))

    async def record_artifact(signal: Any) -> None:
        try:
            meta = wire_payload(signal.meta, 0)  # 0: no cap; a row, not an event
        except (TypeError, ValueError, RecursionError) as exc:
            log.warning("an artifact's meta could not be encoded: %s", exc)
            meta = None
        path, mount_label = artifact_location(signal.path, mount)
        send((MSG_ARTIFACT, {"kind": signal.kind, "path": path,
                             "meta": meta, "node_id": signal.node_id,
                             "mount_label": mount_label}))

    async def warn(signal: Any) -> None:
        send((MSG_WARNING, {"kind": signal.kind, "detail": signal.detail,
                            "node_id": signal.node_id}))

    return RunSinks(emit=emit, record_metrics=record_metrics,
                    record_artifact=record_artifact, warn=warn)


def _listen_for_cancel(control_in: Any, context: Any,
                       outputs: "LiveOutputs | None" = None) -> None:
    """Thread body: cancel the run on a ``cancel`` message or a closed pipe;
    hand reads of a recorded run's outputs to *outputs* (#789)."""
    while True:
        try:
            message = control_in.recv()
        except (EOFError, OSError):
            context.cancel()
            return
        if not isinstance(message, tuple) or not message:
            continue
        if message[0] == MSG_CANCEL:
            context.cancel()
        elif outputs is not None and message[0] == MSG_FETCH \
                and len(message) == 5:
            outputs.answer(*message[1:])
        elif outputs is not None and message[0] == MSG_DROP_OUTPUTS \
                and len(message) == 2:
            outputs.drop(message[1])


class _LoopReader:
    """The live store, read from another thread: each call runs on the
    run's event loop (the store's lock lives there). With no loop (before
    the run starts, or after it ends) it holds nothing."""

    def __init__(self, store: Any, loop: Any) -> None:
        self._store = store
        self._loop = loop

    async def _call(self, name: str, *args: Any) -> Any:
        import asyncio

        loop = self._loop
        if loop is None or loop.is_closed():
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(
                getattr(self._store, name)(*args), loop)
        except RuntimeError:  # the loop closed in between
            return None
        return await asyncio.wrap_future(future)

    async def has_run(self, run_id: str) -> bool:
        return bool(await self._call("has_run", run_id))

    async def list_ports(self, run_id: str) -> Any:
        return await self._call("list_ports", run_id)

    async def get_with_version(self, run_id: str, node_id: str,
                               port: str) -> Any:
        return await self._call("get_with_version", run_id, node_id, port)

    async def get(self, run_id: str, node_id: str, port: str) -> Any:
        return await self._call("get", run_id, node_id, port)


class LiveOutputs:
    """A recorded run's captures in this process, and the reads of them
    (#789). Reads run on a small pool of their own, each on its own event
    loop, so serialising a large tensor never holds up the run's loop:
    only the store lookups hop to it."""

    def __init__(self, store: Any, run_id: str, directory: str | None,
                 send: Send) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.store = store
        self.run_id = run_id
        self.directory = directory
        self.send = send
        #: The run's event loop, once it runs.
        self.loop: Any = None
        #: The dump's reader, once written: reads then come from the files
        #: (and are ``final``).
        self.disk: Any = None
        self._pool = ThreadPoolExecutor(2, thread_name_prefix="outputs")

    def answer(self, request_id: str, op: str, run_id: str,
               params: dict[str, Any]) -> None:
        self._pool.submit(self._answer, request_id, op, run_id,
                          dict(params or {}))

    def _answer(self, request_id: str, op: str, run_id: str,
                params: dict[str, Any]) -> None:
        import asyncio

        from ..core import output_queries as q
        from ..core.port_stats import compute_port_stats

        async def compute(value: Any) -> dict[str, Any]:
            return compute_port_stats(value)

        disk = self.disk
        reader = disk if disk is not None else _LoopReader(self.store,
                                                           self.loop)
        try:
            status, body = asyncio.run(q.dispatch(reader, op, run_id, params,
                                                  compute=compute))
            text = "" if body is None else q.encode_body(body)
        except BaseException as exc:  # noqa: BLE001 -- answered, not raised
            status, text = 500, q.encode_body(
                {"detail": f"reading the recorded outputs failed: {exc}"})
        self.send((MSG_OUTPUT_ANSWER, request_id, status, text,
                   disk is not None))

    def drop(self, run_id: str) -> None:
        import asyncio

        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.store.delete_run(run_id),
                                             loop)
        except RuntimeError:
            pass

    async def dump(self) -> None:
        """Write the run's captures to :attr:`directory`, then read from
        there. A failure is a ``warning`` for the run, never its end."""
        if not self.directory:
            return
        from pathlib import Path

        from ..core import output_queries as q

        try:
            if not await self.store.has_run(self.run_id):
                return  # recorded nothing: there is nothing to keep
            await q.dump_run(self.store, self.run_id, Path(self.directory))
            self.disk = q.DiskRunReader(Path(self.directory))
        except Exception as exc:  # noqa: BLE001 -- reported, the run stands
            import logging
            import traceback

            logging.getLogger(__name__).warning(
                "the recorded outputs could not be written: %s", exc)
            traceback.print_exc()
            self.send((MSG_WARNING, {
                "kind": "outputs_not_kept",
                "detail": (f"the outputs this run recorded could not be "
                           f"written on the worker, so they cannot be read "
                           f"after it ends: {type(exc).__name__}: {exc}"),
                "node_id": None}))


def apply_memory_budgets(budgets: dict[str, int]) -> dict[str, Any]:
    """Cap each card's allocator to its budget (#788). Imports torch.

    *budgets* maps a device as THIS process numbers it (``cuda:0`` is the
    first card of its ``CUDA_VISIBLE_DEVICES``) to bytes. The fraction is
    of the card's total memory, clamped to 1. Returns what was applied, for
    ``run_started``'s ``execution_facts.memory_limit``. Raises
    ``RuntimeError`` when a cap cannot be set: a run admitted to a shared
    card must not run uncapped.
    """
    import torch

    applied: dict[str, Any] = {}
    for device, budget in budgets.items():
        kind, _, index = str(device).partition(":")
        try:
            if kind == "cuda":
                card = int(index or 0)
                total = int(torch.cuda.get_device_properties(card).total_memory)
                fraction = min(1.0, budget / total)
                torch.cuda.set_per_process_memory_fraction(fraction, card)
            elif kind == "mps":
                total = int(torch.mps.recommended_max_memory())
                fraction = min(1.0, budget / total)
                torch.mps.set_per_process_memory_fraction(fraction)
            else:
                raise ValueError(f"{kind} has no allocator to cap")
        except Exception as exc:  # noqa: BLE001 -- re-raised with context
            raise RuntimeError(
                f"could not cap {device}'s memory to its budget of "
                f"{budget} bytes: {type(exc).__name__}: {exc}") from exc
        applied[str(device)] = {"budget_bytes": int(budget),
                                "total_bytes": total,
                                "fraction": round(fraction, 6)}
    return applied


def _normalised_graph(graph: dict[str, Any]) -> dict[str, Any]:
    graph = dict(graph)
    graph.setdefault("edges", [])
    graph.setdefault("presets", [])
    graph.setdefault("subgraphs", [])
    return graph


def run_assignment(spec: dict[str, Any], send: Send,
                   control_in: Any = None, *, discover: bool = True) -> None:
    """Discover nodes, execute the run, send ``finished``. Never raises.

    ``discover=False`` keeps the node registry as it is (tests that run
    this in a process whose registry is already populated).
    """
    status, error, peak_memory = "failed", None, None
    try:
        import asyncio

        from ..core.executors import ExecutionRequest, InProcessExecutor
        from ..core.runtime import initialize_runtime

        # Before discovery: a node module that touches the card at import
        # must already find it capped (#788).
        budgets = spec.get("memory_budgets") or {}
        if budgets:
            send((MSG_MEMORY_LIMIT, apply_memory_budgets(budgets)))

        # The same discovery the server's lifespan runs: built-in, custom
        # and plugin nodes, presets. Payloads depend on the registry (media
        # detection), so a worker must see the node set the server sees.
        if discover:
            initialize_runtime()
        run_id = spec["run_id"]
        options = dict(spec.get("options") or {})
        context = build_context(run_id, options, spec.get("device") or "cpu")
        outputs = None
        if options.get("record_outputs"):
            from ..core.run_output_store import RunOutputStore

            # This run's alone, so no budget evicts within it (a run
            # exceeding the budget exceeds it in the control plane's store
            # too); the agent holds the dumps to the budget (#789).
            outputs = LiveOutputs(RunOutputStore(max_bytes=0), run_id,
                                  spec.get("outputs_dir"), send)
        if control_in is not None:
            threading.Thread(target=_listen_for_cancel,
                             args=(control_in, context, outputs),
                             name="cancel-listener", daemon=True).start()
        request = ExecutionRequest(
            run_id=run_id,
            graph=_normalised_graph(spec["graph"]),
            options=options,
            context=context,
            session=None,
            output_store=None if outputs is None else outputs.store,
            record_outputs=outputs is not None,
            sinks=_pipe_sinks(send, mount=spec.get("mount")),
            explicit_metric_nodes=set(),
        )

        async def execute() -> Any:
            if outputs is not None:
                outputs.loop = asyncio.get_running_loop()
            # This process runs this one run and nothing else, so its peak
            # memory figures are exact (#770).
            result = await InProcessExecutor(exclusive_process=True).run(
                request)
            if outputs is not None:
                await outputs.dump()
            return result

        outcome = asyncio.run(execute())
        status, error = outcome.status, outcome.error
        peak_memory = getattr(outcome, "peak_memory", None)
    except BaseException as exc:  # noqa: BLE001 -- reported, then we exit
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    send((MSG_FINISHED, {"status": status, "error": error,
                         "peak_memory": peak_memory}))
