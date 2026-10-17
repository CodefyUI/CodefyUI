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

_DEVICE_KINDS = ("cpu", "cuda", "mps")

#: Set while :func:`_resolve_quietly` asks ``resolve_device``; the filter
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


def _resolve_quietly(value: str) -> str:
    """``resolve_device(value)`` without its fallback log line."""
    token = _QUIET.set(True)
    try:
        return device_utils.resolve_device(value)
    finally:
        _QUIET.reset(token)


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


def _device_shaped(value: Any) -> str | None:
    """*value* trimmed and lowercased when it is meant as a device (its kind
    is cpu, cuda or mps, as ``node_target_device`` decides), ``auto`` and
    blanks excepted. Malformed spellings of those kinds are kept: they
    fall back too."""
    if not isinstance(value, str):
        return None
    probe = value.strip().lower()
    if not probe or probe == "auto":
        return None
    if split_device(probe)[0] not in _DEVICE_KINDS:
        return None
    return probe


def _declares_device(node_cls: Any) -> bool:
    try:
        return any(getattr(p, "name", None) == "device"
                   for p in node_cls.define_params())
    except Exception:  # noqa: BLE001 -- a broken node is not ours to judge
        return False


def _cards_text(run_cards: Mapping[str, str]) -> str:
    """``1 card (cuda:0 = the worker's cuda:1)``."""
    count = len(run_cards)
    pairs = ", ".join(f"{in_run} = the worker's {physical}"
                      for in_run, physical in run_cards.items())
    return f"{count} card{'s' if count != 1 else ''} ({pairs})"


def _node_sentence(node_id: Any, node_type: Any, value: str, resolved: str,
                   where: str, run_cards: Mapping[str, str] | None) -> str:
    head = f"Node {node_id} ({node_type}) asks for device {value}"
    if run_cards is not None and split_device(value)[0] == "cuda":
        # In a worker's run subprocess: a cuda name is an in-run name.
        if run_cards:
            return (f"{head}; this run has {_cards_text(run_cards)}, "
                    f"so it runs on {resolved}.")
        return f"{head}; this run has no CUDA card, so it runs on {resolved}."
    return f"{head}, which {where} does not have; it runs on {resolved}."


def device_fallbacks(nodes: Iterable[Mapping[str, Any]], requested: Any,
                     run_device: str, *, where: str = "this server",
                     check_run: bool = True,
                     run_cards: Mapping[str, str] | None = None,
                     ) -> list[dict[str, Any]]:
    """``[{"detail", "node_id"}, ...]``: every device substitution a run of
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
    asked = _device_shaped(requested)
    if check_run and asked is not None and substituted(asked, run_device):
        found.append({
            "detail": (f"The run asked for device {asked}, which {where} "
                       f"does not have; it runs on {run_device}."),
            "node_id": None,
        })
    seen: dict[str, str] = {}
    for node in nodes:
        data = node.get("data")
        params = data.get("params") if isinstance(data, Mapping) else None
        if not isinstance(params, Mapping):
            continue
        value = _device_shaped(params.get("device"))
        if value is None:
            continue
        node_cls = registry.get(node.get("type"))
        if node_cls is None or not _declares_device(node_cls):
            continue
        if value not in seen:
            seen[value] = _resolve_quietly(value)
        resolved = seen[value]
        if not substituted(value, resolved):
            continue
        node_id = node.get("id")
        found.append({
            "detail": _node_sentence(node_id, node.get("type"), value,
                                     resolved, where, run_cards),
            "node_id": node_id if isinstance(node_id, str) else None,
        })
    return found
