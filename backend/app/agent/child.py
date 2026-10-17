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
- ``("artifact", {"kind", "path", "meta", "node_id", "mount_label"})``:
  ``path`` relative to the run's mount (``spec["mount"]``, #782) when the
  file is under it, with ``mount_label`` naming it; else absolute, with
  ``mount_label`` None.
- ``("warning", {"kind", "detail", "node_id"})``
- ``("finished", {"status", "error", "peak_memory"})``, last.

Paths on a shared mount (#782) are mapped at exactly two places, both fed
by the assignment (``spec["mount"]``, built once by ``Runner.storage_plan``):
in, the data settings are derived from the mount root before anything
imports ``app.config`` (``runner.child_environment``), so the graph's own
paths (an upload's bare name, ``models/x.pt``) resolve unchanged and the
graph is never rewritten; out, every artifact path goes through
:func:`artifact_location`, the one place a path becomes relative.

Agent -> child: ``("cancel",)``. The pipe closing means the agent is gone,
which is treated as a cancel: nobody is left to report to.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

MSG_EVENT = "event"
MSG_METRICS = "metrics"
MSG_ARTIFACT = "artifact"
MSG_WARNING = "warning"
MSG_FINISHED = "finished"
MSG_CANCEL = "cancel"

Send = Callable[[tuple], None]


def main(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """``Process`` target. *spec* is built by ``Runner.assign``."""
    ignore_interrupts()
    apply_environment(spec.get("env") or {})
    _redirect_output(spec.get("log_path"))
    enter_install_dir()
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


#: The install's ``backend/`` directory: the working directory the server
#: runs in (``cdui start``), which a graph's relative paths are read from.
INSTALL_DIR = Path(__file__).resolve().parents[2]


def enter_install_dir(install_dir: Path = INSTALL_DIR) -> None:
    """Run in the install's ``backend/``, as the server does (#839).

    A graph's relative paths resolve against the working directory: the
    bundled samples (``data/samples/iris.csv``, the ``data/samples/rag``
    folder) are spelled that way, and ``core.run_needs`` lets such a run go
    to a worker without the ``data`` mount because every install has them.
    An agent started over SSH or from a terminal keeps the directory it was
    started in (the home directory, say), so without this the sample was
    looked for there and the run failed on the worker.

    The ``CODEFYUI_*_DIR`` / ``*_PATH`` settings the child inherited are
    made absolute first, against the directory they were given in, so a
    relative one keeps naming the same place.
    """
    here = Path.cwd()
    for name, value in list(os.environ.items()):
        if not name.startswith("CODEFYUI_") or not value:
            continue
        if not (name.endswith("_DIR") or name.endswith("_PATH")):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            os.environ[name] = str(here / path)
    os.chdir(install_dir)


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


def run_cards(spec: dict[str, Any]) -> dict[str, str] | None:
    """In-run CUDA name -> the worker's card, for this run (#839).

    Read off the environment the agent chose (``runner.child_environment``):
    the run sees its set's cards as ``cuda:0..N-1`` in set order. ``{}``
    when it sees no card (a CPU or MPS run); None when the agent left the
    inherited visibility in place (a bare ``cuda`` with no set), where the
    numbering is not this run's to state.
    """
    from .runner import device_aliases

    env = spec.get("env") or {}
    if "CUDA_VISIBLE_DEVICES" not in (env.get("set") or {}):
        return None
    aliases = device_aliases(list(env.get("devices") or []))
    return {in_run: physical for physical, in_run in aliases.items()}


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
            sinks=_pipe_sinks(send, mount=spec.get("mount")),
            explicit_metric_nodes=set(),
        )
        # This process runs this one run and nothing else, so its peak
        # memory figures are exact (#770).
        outcome = asyncio.run(
            InProcessExecutor(exclusive_process=True,
                              run_cards=run_cards(spec)).run(request))
        status, error = outcome.status, outcome.error
        peak_memory = getattr(outcome, "peak_memory", None)
    except BaseException as exc:  # noqa: BLE001 -- reported, then we exit
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    send((MSG_FINISHED, {"status": status, "error": error,
                         "peak_memory": peak_memory}))
