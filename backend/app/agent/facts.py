"""The agent's environment facts for ``hello`` (#781).

The control plane checks a worker by what its ``hello`` reports: the
environment fingerprint (``core.env_fingerprint``) and the node catalog
(``core.node_registry.node_catalog``). Both need what the agent process
must never load: torch (the fingerprint reads its build, the accelerators
and the device list) and every node module (the catalog is built from the
registry after discovery). So they are collected ONCE, when the agent
starts, by a short-lived helper subprocess (``multiprocessing`` "spawn",
like a run subprocess) that runs the same discovery a run does
(``initialize_runtime``), and cached for every connection after.

What the fingerprint describes is the agent's environment as its run
subprocesses inherit it: the same interpreter, packages, plugins, custom
nodes and environment variables, all CUDA cards the agent sees (a run's
device set narrows ``CUDA_VISIBLE_DEVICES`` afterwards, per run). Plugins
or packs installed while the agent runs are reported after the agent
restarts.

When the helper fails (it crashed, or took longer than
:data:`FACTS_TIMEOUT_S`), the agent still connects, with no fingerprint and
no catalog; the failure is logged here and the control plane logs that the
worker's node types are not checked.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Seconds the helper has to import torch, discover every node and report.
FACTS_TIMEOUT_S = 300.0
#: How often a waiting :func:`gather` looks at its *cancel* event.
CANCEL_POLL_S = 0.1


@dataclass(frozen=True)
class AgentFacts:
    """What the helper reported, ready for ``hello``."""

    fingerprint: dict[str, Any] | None = None
    fingerprint_digest: str | None = None
    node_catalog_hash: str | None = None
    node_types: list[str] = field(default_factory=list)


def collect(*, discover: bool = True) -> dict[str, Any]:
    """The facts of THIS process, after node discovery. Imports torch.

    ``discover=False`` keeps the node registry as it is (tests whose
    registry is already populated).
    """
    from ..core.env_fingerprint import collect_with_digest
    from ..core.node_registry import node_catalog
    from ..core.runtime import initialize_runtime

    if discover:
        initialize_runtime()
    document, digest = collect_with_digest()
    catalog = node_catalog()
    return {"fingerprint": document, "fingerprint_digest": digest,
            "node_catalog_hash": catalog.hash,
            "node_types": list(catalog.node_types)}


def helper_main(conn: Any) -> None:
    """The helper subprocess: report :func:`collect` on *conn*, or why not."""
    try:
        result: tuple = ("ok", collect())
    except BaseException as exc:  # noqa: BLE001 -- reported, then we exit
        result = ("error", f"{type(exc).__name__}: {exc}")
    try:
        conn.send(result)
    finally:
        conn.close()


def gather(*, timeout_s: float = FACTS_TIMEOUT_S,
           target: Callable[[Any], None] = helper_main,
           cancel: threading.Event | None = None) -> AgentFacts | None:
    """Run the helper subprocess and read its report. Blocking.

    Never raises: a helper that fails or times out gives an empty
    :class:`AgentFacts` (logged). *cancel* set while it waits (the agent
    is stopping) kills the helper at once and returns None. *target* is
    the helper's entry point (tests swap it).
    """
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(send,),
                          name="codefyui-agent-facts", daemon=True)
    try:
        process.start()
    except Exception:  # noqa: BLE001
        logger.warning("could not start the facts helper", exc_info=True)
        send.close()
        receive.close()
        return AgentFacts()
    send.close()
    deadline = time.monotonic() + timeout_s
    try:
        while not receive.poll(CANCEL_POLL_S):
            if cancel is not None and cancel.is_set():
                logger.info("stopping: the facts helper was ended")
                process.kill()
                return None
            if time.monotonic() >= deadline:
                logger.warning(
                    "the facts helper did not report within %.0f s; "
                    "connecting without a fingerprint or node catalog",
                    timeout_s)
                return AgentFacts()
        status, value = receive.recv()
    except (EOFError, OSError, ValueError, TypeError):
        logger.warning("the facts helper exited without reporting (exit code "
                       "%s); connecting without a fingerprint or node "
                       "catalog", _exit_code(process))
        return AgentFacts()
    finally:
        receive.close()
        _reap(process)
    if status != "ok" or not isinstance(value, dict):
        logger.warning("the facts helper failed: %s; connecting without a "
                       "fingerprint or node catalog", value)
        return AgentFacts()
    return AgentFacts(
        fingerprint=value.get("fingerprint"),
        fingerprint_digest=value.get("fingerprint_digest"),
        node_catalog_hash=value.get("node_catalog_hash"),
        node_types=list(value.get("node_types") or []))


def _exit_code(process: Any) -> Any:
    process.join(5)
    return process.exitcode


def _reap(process: Any) -> None:
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
