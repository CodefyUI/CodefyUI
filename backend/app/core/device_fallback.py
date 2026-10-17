"""Which devices a run asked for and will not get, as run warnings (#839).

A device this process does not have degrades: ``resolve_device`` falls back
(``cuda`` on a Mac runs on the CPU, ``cuda:3`` on a one-card box on the
current card) and logs a line in the server console. That fallback is
behaviour exported scripts and saved graphs rely on, so it stays. What
this module adds is the run's own record of it: before the graph executes,
:func:`device_fallbacks` lists every substitution the run will meet, and
``InProcessExecutor`` sends each one as a ``run_warning`` of kind
:data:`WARNING_DEVICE_FALLBACK`, which the Runs panel shows in the run's
event log.

Two places a run names a device:

- the run's own device (``options["device"]``), resolved once at submit
  into the device the run executes on (``ExecutionContext.device``);
- a node's ``device`` param (TrainingLoop, Inference, ...), resolved by the
  node through ``device_utils.resolve_node_device`` when it executes. The
  same guards ``device_utils.node_target_device`` applies hold here: only a
  param the node declares, only a device-shaped value, ``auto`` (the run's
  device) never.

Nothing here changes what a run does: the answer is computed with the
very ``resolve_device`` the node will call, and only reported.
"""

from __future__ import annotations

from typing import Any

from .device_utils import resolve_device, split_device

#: ``run_warning`` kind for a device the run asked for and does not get.
WARNING_DEVICE_FALLBACK = "device_fallback"

_DEVICE_KINDS = ("cpu", "cuda", "mps")


def substituted(requested: str, resolved: str) -> bool:
    """Whether *resolved* is a different device from the one *requested*.

    A different kind (``cuda`` -> ``cpu``), or a different card when the
    request named one (``cuda:3`` -> ``cuda:0``). A bare kind is any card
    of it (``cuda`` -> ``cuda:1`` is not a substitution); ``mps`` and
    ``mps:0`` are one device.
    """
    want_kind, want_index = split_device(requested.strip().lower())
    got_kind, got_index = split_device(resolved.strip().lower())
    if want_kind != got_kind:
        return True
    if want_index is None:
        return False
    if want_kind == "mps":
        return (got_index or 0) != want_index
    return got_index is not None and got_index != want_index


def _device_shaped(value: Any) -> str | None:
    """*value* trimmed and lowercased when it names a device (not ``auto``)."""
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


def device_fallbacks(graph: dict[str, Any], requested: Any, run_device: str,
                     *, where: str = "this server",
                     check_run: bool = True) -> list[dict[str, Any]]:
    """``[{"detail", "node_id"}, ...]``: every device substitution the run
    of *graph* will meet, the run's own first.

    *requested* is the run's ``options["device"]``, *run_device* the device
    it executes on. *where* names the machine in the sentence. *check_run*
    False skips the run's own device: a worker's run subprocess sees its
    cards renumbered (``cuda:1`` alone is its ``cuda:0``), and its device
    was checked against the worker's at submit.
    """
    found: list[dict[str, Any]] = []
    asked = _device_shaped(requested)
    if check_run and asked is not None and substituted(asked, run_device):
        found.append({
            "detail": (f"The run asked for device {asked}, which {where} "
                       f"does not have; it runs on {run_device}."),
            "node_id": None,
        })
    from .node_registry import registry
    from .run_needs import executable_nodes

    seen: dict[str, str] = {}
    for node in executable_nodes(graph):
        data = node.get("data")
        params = data.get("params") if isinstance(data, dict) else None
        if not isinstance(params, dict):
            continue
        value = _device_shaped(params.get("device"))
        if value is None:
            continue
        node_cls = registry.get(node.get("type"))
        if node_cls is None or not _declares_device(node_cls):
            continue
        if value not in seen:
            seen[value] = resolve_device(value)
        resolved = seen[value]
        if not substituted(value, resolved):
            continue
        node_id = node.get("id")
        found.append({
            "detail": (f"Node {node_id} ({node.get('type')}) asks for device "
                       f"{value}, which {where} does not have; it runs on "
                       f"{resolved}."),
            "node_id": node_id if isinstance(node_id, str) else None,
        })
    return found
