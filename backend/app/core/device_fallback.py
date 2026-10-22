"""Which devices a run asked for and will not get, as run warnings (#839).

A device this process does not have degrades: ``resolve_device`` falls back
(``cuda`` on a Mac runs on the CPU, ``cuda:3`` on a one-card box on the
current card) and logs a line in the server console. That fallback is
behaviour exported scripts and saved graphs rely on, so it stays. What
this module adds is the run's own record of it: :func:`device_fallbacks`
lists every substitution the run will meet, and ``InProcessExecutor`` sends
each one as a ``run_warning`` of kind :data:`WARNING_DEVICE_FALLBACK`,
which the Runs panel shows in the run's event log.

Two places a run names a device:

- the run's own device (``options["device"]``), resolved once at submit
  into the device the run executes on (``ExecutionContext.device``);
- a node's ``device`` param (TrainingLoop, Inference, ...), resolved by the
  node through ``device_utils.resolve_node_device`` when it executes. The
  same guards ``device_utils.node_target_device`` applies hold here: only a
  param the node declares, only a device-shaped value, ``auto`` (the run's
  device) never.

The nodes are the ones the run executes, as ``execute_graph`` expanded them
(its ``on_expanded`` callback): bypassed nodes, unused subgraph definitions
and a graph that does not prepare at all produce no warning, and the graph
is not expanded a second time.

A node's ``device`` is a name INSIDE the run. In a worker's run subprocess
the cards of the run's device set are renumbered from ``cuda:0`` (#774), so
the sentence describes the run's cards and which of the worker's cards
they are (*run_cards*), not the worker's own numbering.

Nothing here changes what a run does: the answer is computed with the very
``resolve_device`` the node will call (its log line suppressed for this
call, so a fallback is still logged once, by the node), and only reported.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Iterable, Mapping

from . import device_utils
from .device_utils import DEVICE_SYNTAX, split_device

#: ``run_warning`` kind for a device the run asked for and does not get.
WARNING_DEVICE_FALLBACK = "device_fallback"

#: ``code`` of each sentence, with ``params`` beside the English ``detail``,
#: so the Runs panel words it in its own language (#841's coded messages,
#: ``frontend/src/utils/runReasons.ts``). ``where`` is ``server`` or
#: ``worker``; ``cards`` is ``[[in_run, worker_card], ...]``.
CODE_RUN = "device_fallback_run"
CODE_NODE = "device_fallback_node"
CODE_NODE_IN_RUN = "device_fallback_node_in_run"

#: Set while :func:`_node_device` asks ``resolve_device``; the filter
#: below drops that call's log records and nothing else's (a context
#: variable, so a node resolving on another thread still logs).
_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "device_fallback_quiet", default=False)


class _QuietFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _QUIET.get()


_FILTER = _QuietFilter()
if not any(isinstance(f, _QuietFilter) for f in device_utils.logger.filters):
    device_utils.logger.addFilter(_FILTER)


def substituted(requested: str, resolved: str) -> bool:
    """Whether *resolved* is a different device from the one *requested*.

    A different kind (``cuda`` -> ``cpu``), or a different card when the
    request named one (``cuda:3`` -> ``cuda:0``). A bare kind is any card
    of it (``cuda`` -> ``cuda:1`` is not a substitution); ``mps`` and
    ``mps:0`` are one device. A malformed spelling (``cuda:abc``,
    ``cuda: 0``) is always a substitution: ``resolve_device`` never hands
    it back as written.
    """
    want = requested.strip().lower()
    if DEVICE_SYNTAX.match(want) is None:
        return True
    want_kind, want_index = split_device(want)
    got_kind, got_index = split_device(resolved.strip().lower())
    if want_kind != got_kind:
        return True
    if want_index is None:
        return False
    if want_kind == "mps":
        return (got_index or 0) != want_index
    return got_index is not None and got_index != want_index


#: Stands for the run's device in :func:`_node_device`: a value no real
#: device spelling can equal.
_RUN_DEVICE = "<run device>"


class _RunContext:
    device = _RUN_DEVICE


def _node_device(params: Mapping[str, Any], node_cls: Any) -> str | None:
    """The device a node pinned through its own ``device`` param lands on,
    or None when it follows the run's device.

    Asked of ``device_utils.node_target_device`` itself, the function the
    engine uses for the node, so the guards cannot drift from it: only a
    param the node declares, only a device-shaped value, ``auto`` never.
    The run's device is a marker here, so "follows the run" is told apart
    from "resolved to the same device". Quiet: no second log line.
    """
    token = _QUIET.set(True)
    try:
        resolved = device_utils.node_target_device(dict(params), _RunContext,
                                                    node_cls)
    finally:
        _QUIET.reset(token)
    return None if resolved == _RUN_DEVICE else resolved


def _cards_text(run_cards: Mapping[str, str]) -> str:
    """``1 card (cuda:0 = the worker's cuda:1)``."""
    count = len(run_cards)
    pairs = ", ".join(f"{in_run} = the worker's {physical}"
                      for in_run, physical in run_cards.items())
    return f"{count} card{'s' if count != 1 else ''} ({pairs})"


def _node_entry(node_id: Any, node_type: Any, value: str, resolved: str,
                where: str, run_cards: Mapping[str, str] | None
                ) -> dict[str, Any]:
    """One node's report: the English ``detail`` and its ``code`` and
    ``params``."""
    # Worded as the node's setting: a node the run serves from its cache
    # never executes, and the sentence stays true for it.
    head = f"Node {node_id} ({node_type}) is set to device {value}"
    tail = f"when it executes, it runs on {resolved}."
    params: dict[str, Any] = {"node": str(node_id), "type": str(node_type),
                              "requested": value, "device": resolved}
    if run_cards is not None and split_device(value)[0] == "cuda":
        # In a worker's run subprocess: a cuda name is an in-run name.
        params["cards"] = [[in_run, physical]
                           for in_run, physical in run_cards.items()]
        detail = (f"{head}; this run has {_cards_text(run_cards)}, so {tail}"
                  if run_cards else
                  f"{head}; this run has no CUDA card, so {tail}")
        return {"detail": detail, "code": CODE_NODE_IN_RUN, "params": params}
    params["where"] = _where_code(where)
    return {"detail": f"{head}, which {where} does not have; {tail}",
            "code": CODE_NODE, "params": params}


def _where_code(where: str) -> str:
    return "worker" if "worker" in where else "server"


def device_fallbacks(nodes: Iterable[Mapping[str, Any]], requested: Any,
                     run_device: str, *, where: str = "this server",
                     check_run: bool = True,
                     run_cards: Mapping[str, str] | None = None,
                     ) -> list[dict[str, Any]]:
    """``[{"detail", "node_id", "code", "params"}, ...]``: every device
    substitution a run of
    *nodes* (the expanded node list it executes) will meet, the run's own
    first.

    *requested* is the run's ``options["device"]``, *run_device* the device
    it executes on. *where* names the machine in the sentence. *check_run*
    False skips the run's own device: a worker's run subprocess sees its
    cards renumbered, and its device was checked against the worker's at
    submit. *run_cards* (a worker's run subprocess) maps each in-run CUDA
    name to the worker's card (``{"cuda:0": "cuda:1"}``; empty when the run
    sees no card); the sentence for a CUDA node is then about the run's
    cards.
    """
    from .node_registry import registry

    found: list[dict[str, Any]] = []
    asked = requested.strip().lower() if isinstance(requested, str) else ""
    if (check_run and asked and asked != "auto"
            and substituted(asked, run_device)):
        found.append({
            "detail": (f"The run asked for device {asked}, which {where} "
                       f"does not have; it runs on {run_device}."),
            "node_id": None, "code": CODE_RUN,
            "params": {"requested": asked, "device": run_device,
                       "where": _where_code(where)},
        })
    seen: dict[tuple[Any, Any], str | None] = {}
    for node in nodes:
        data = node.get("data")
        params = data.get("params") if isinstance(data, Mapping) else None
        if not isinstance(params, Mapping):
            continue
        raw = params.get("device")
        if not isinstance(raw, str):
            continue
        node_cls = registry.get(node.get("type"))
        if node_cls is None:
            continue
        key = (node.get("type"), raw)
        if key not in seen:
            seen[key] = _node_device(params, node_cls)
        resolved = seen[key]
        value = raw.strip().lower()
        if resolved is None or not substituted(value, resolved):
            continue
        node_id = node.get("id")
        found.append({
            **_node_entry(node_id, node.get("type"), value, resolved, where,
                          run_cards),
            "node_id": node_id if isinstance(node_id, str) else None,
        })
    return found
