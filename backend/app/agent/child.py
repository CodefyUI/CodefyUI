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
does not even pass ``CODEFYUI_DB_PATH``), record outputs (refused at
assignment time until #789), touch a ``NodeStateStore`` (a worker run is
isolated, ``weights_persistent=False``).

Pipe messages, child -> agent (tuples, pickled by ``multiprocessing``):

- ``("event", event_type, payload)``
- ``("metrics", [{"name", "value", "step", "node_id"}, ...])`` (never empty)
- ``("artifact", {"kind", "path", "meta", "node_id"})``
- ``("warning", {"kind", "detail", "node_id"})``
- ``("finished", {"status", "error", "peak_memory"})``, last.

Agent -> child: ``("cancel",)``. The pipe closing means the agent is gone,
which is treated as a cancel: nobody is left to report to.
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
MSG_CANCEL = "cancel"

Send = Callable[[tuple], None]


def main(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """``Process`` target. *spec* is built by ``runner.child_spec``."""
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
    ``BrokenPipeError`` surfacing as a node failure.
    """
    def send(message: tuple) -> None:
        try:
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


def _pipe_sinks(send: Send) -> Any:
    """A ``RunSinks`` whose callables write pipe messages."""
    from ..core.executors import RunSinks

    async def emit(event_type: str, payload: dict[str, Any] | None) -> None:
        send((MSG_EVENT, event_type, payload))

    async def record_metrics(points: list[Any]) -> None:
        # The engine bridge calls this with [] on every progress frame to
        # let the control plane's batcher check its clock; that poke is the
        # consumer's to make (see worker_protocol), not worth a message.
        if points:
            send((MSG_METRICS, [
                {"name": p.name, "value": p.value, "step": p.step,
                 "node_id": p.node_id} for p in points]))

    async def record_artifact(signal: Any) -> None:
        send((MSG_ARTIFACT, {"kind": signal.kind, "path": signal.path,
                             "meta": signal.meta, "node_id": signal.node_id}))

    async def warn(signal: Any) -> None:
        send((MSG_WARNING, {"kind": signal.kind, "detail": signal.detail,
                            "node_id": signal.node_id}))

    return RunSinks(emit=emit, record_metrics=record_metrics,
                    record_artifact=record_artifact, warn=warn)


def _listen_for_cancel(control_in: Any, context: Any) -> None:
    """Thread body: cancel the run on a ``cancel`` message or a closed pipe."""
    while True:
        try:
            message = control_in.recv()
        except (EOFError, OSError):
            context.cancel()
            return
        if isinstance(message, tuple) and message and message[0] == MSG_CANCEL:
            context.cancel()


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

        # The same discovery the server's lifespan runs: built-in, custom
        # and plugin nodes, presets. Payloads depend on the registry (media
        # detection), so a worker must see the node set the server sees.
        if discover:
            initialize_runtime()
        run_id = spec["run_id"]
        options = dict(spec.get("options") or {})
        context = build_context(run_id, options, spec.get("device") or "cpu")
        if control_in is not None:
            threading.Thread(target=_listen_for_cancel,
                             args=(control_in, context),
                             name="cancel-listener", daemon=True).start()
        request = ExecutionRequest(
            run_id=run_id,
            graph=_normalised_graph(spec["graph"]),
            options=options,
            context=context,
            session=None,
            output_store=None,
            record_outputs=False,
            sinks=_pipe_sinks(send),
            explicit_metric_nodes=set(),
        )
        outcome = asyncio.run(InProcessExecutor().run(request))
        status, error = outcome.status, outcome.error
        peak_memory = getattr(outcome, "peak_memory", None)
    except BaseException as exc:  # noqa: BLE001 -- reported, then we exit
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    send((MSG_FINISHED, {"status": status, "error": error,
                         "peak_memory": peak_memory}))
