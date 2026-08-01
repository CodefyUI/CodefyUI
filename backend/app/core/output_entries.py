"""The ``node_status`` output contract — typed renderable payloads (#117).

Every renderable payload a node produces rides in a ``node_status``
message's ``outputs`` list as ``{"output_kind": <kind>, <kind>: <payload>}``
(plus ``"port"`` when the payload came from a named output port). Kinds are
plain strings, not a closed enum, so a later node pack can add its own (e.g.
"chart") without a core release.

This replaced two guessing games: the WS layer sniffing "long alphanumeric
string => base64 PNG" (which mislabelled ordinary long text as an image),
and the frontend smuggling images/progress through the log stream as
``__IMAGE__:`` / ``__PROGRESS__:`` prefixed strings.

Why it lives in ``core``
------------------------
It shipped in ``api/ws_execution.py`` because the WebSocket was the only
producer of ``node_status``. #120 gave ``RunService`` its own progress
bridge and had to reach back into ``api`` through a function-local import to
avoid a second copy of this logic — the documented layering debt this module
pays off. Since #121 the WebSocket is a *view* over the run service and
produces no output entries at all; the builders belong to the run, not to
one transport.

Pure functions over plain dicts, with exactly one dependency on live server
state: :func:`declared_image_ports` asks the node registry what a node
declares. Nothing here touches a socket, a request or a database.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import settings
from .node_base import MEDIA_IMAGE
from .node_registry import registry

logger = logging.getLogger(__name__)

OUTPUT_KIND_TEXT = "text"
OUTPUT_KIND_IMAGE = "image"
OUTPUT_KIND_PROGRESS = "progress"
OUTPUT_KIND_TENSOR_SUMMARY = "tensor_summary"

# Statuses whose payload is a node result worth summarizing. "interrupted"
# (#122) carries PARTIAL outputs — the epochs a stopped training run did
# manage — and those are exactly what the user wants to look at afterwards,
# so it summarizes like a completed one.
_STATUSES_WITH_RESULT = ("completed", "cached", "interrupted")


def _summarize_single(value: Any) -> dict[str, Any]:
    """Generate a human-readable summary for a single output value."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            summary: dict[str, Any] = {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if value.numel() > 0 and value.is_floating_point():
                summary["min"] = round(float(value.min()), 4)
                summary["max"] = round(float(value.max()), 4)
                summary["mean"] = round(float(value.mean()), 4)
            elif value.numel() > 0:
                summary["min"] = int(value.min())
                summary["max"] = int(value.max())
            # Embed values for small tensors so per-node viz (e.g. the
            # embedding scatter plot) can render without a separate REST
            # round-trip. Larger tensors keep going through /api/execution/outputs.
            if value.numel() <= 256:
                summary["values"] = value.detach().cpu().tolist()
            return summary
        if isinstance(value, torch.nn.Module):
            param_count = sum(p.numel() for p in value.parameters())
            return {
                "type": "model",
                "class": value.__class__.__name__,
                "params": param_count,
                "trainable": sum(p.numel() for p in value.parameters() if p.requires_grad),
            }
    except ImportError:
        pass
    if isinstance(value, (int, float, bool)):
        return {"type": "scalar", "value": value}
    if isinstance(value, str):
        summary: dict[str, Any] = {"type": "string", "value": value[:200]}
        rel = _models_dir_relative(value)
        if rel is not None:
            summary["download_path"] = rel
        return summary
    if isinstance(value, (list, tuple)):
        out: dict[str, Any] = {
            "type": "list",
            "length": len(value),
            "repr": repr(value)[:200],
        }
        # Embed values for short primitive lists so per-node UIs (e.g. the
        # tokenizer viz) can render chips without a separate REST round-trip.
        # The Inspector full-fidelity view still uses /api/execution/outputs.
        if len(value) <= 256:
            primitive_types = (str, int, float, bool, type(None))
            if all(isinstance(x, primitive_types) for x in value):
                out["values"] = list(value)
            elif all(
                isinstance(x, (list, tuple))
                and len(x) == 2
                and all(isinstance(y, (int, float)) for y in x)
                for x in value
            ):
                out["values"] = [list(x) for x in value]
        return out
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _models_dir_relative(value: str) -> str | None:
    """If *value* points to an existing file under ``MODELS_DIR``, return
    the relative path (POSIX-style) so the frontend can build a download URL.
    Returns ``None`` otherwise — keeps the check silent on any unexpected input.
    """
    try:
        p = Path(value).resolve()
        if not p.is_file():
            return None
        models_dir = settings.MODELS_DIR.resolve()
        if not p.is_relative_to(models_dir):
            return None
        return p.relative_to(models_dir).as_posix()
    except (OSError, ValueError):
        return None


def _summarize_outputs(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize all output ports of a node result."""
    summary = {}
    for key, val in result.items():
        if key.startswith("__"):
            continue
        summary[key] = _summarize_single(val)
    return summary


def declared_image_ports(nodes: list[Any]) -> dict[str, list[str]]:
    """Map ``node id -> output ports the node DECLARES as image media``.

    Resolved from the registry once per run, so the hot progress path only
    does a dict lookup. Nodes the registry does not know (presets, typos)
    and malformed entries are silently skipped: an unresolvable node simply
    has no declared media, which is the safe answer.
    """
    ports: dict[str, list[str]] = {}
    if not isinstance(nodes, list):
        return ports
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        if not node_id or not node_type:
            continue
        node_cls = registry.get(node_type)
        if node_cls is None:
            continue
        params = (node.get("data") or {}).get("params") or {}
        try:
            definitions = node_cls.define_outputs_dynamic(params)
        except Exception:  # pragma: no cover - defensive: bad third-party node
            logger.debug("define_outputs_dynamic failed for %s", node_type, exc_info=True)
            continue
        declared = [p.name for p in definitions if getattr(p, "media", None) == MEDIA_IMAGE]
        if declared:
            ports[node_id] = declared
    return ports


def build_node_output_entries(
    status: str,
    result: dict[str, Any] | None,
    image_ports: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build the typed ``outputs`` list for one ``node_status`` message.

    ``image_ports`` are the output port names this node *declared* as image
    media (see :func:`declared_image_ports`). Values on any other port stay
    plain data no matter what they look like -- there is deliberately no
    length or character-class inspection anywhere in this function.
    """
    if not result:
        return []

    # A "progress" status carries a live training event, nothing else.
    if status == "progress":
        return [{"output_kind": OUTPUT_KIND_PROGRESS, OUTPUT_KIND_PROGRESS: result}]

    if status not in _STATUSES_WITH_RESULT:
        return []

    entries: list[dict[str, Any]] = []

    # Log output (Print node etc.) is text, and says so.
    if "__log__" in result:
        entries.append(
            {"output_kind": OUTPUT_KIND_TEXT, OUTPUT_KIND_TEXT: str(result["__log__"])}
        )

    # Declared image ports. A declared port that produced nothing this run is
    # skipped rather than announced as an empty image.
    for port in image_ports:
        value = result.get(port)
        if not isinstance(value, str) or not value:
            continue
        entries.append(
            {
                "output_kind": OUTPUT_KIND_IMAGE,
                "port": port,
                OUTPUT_KIND_IMAGE: {
                    # MEDIA_IMAGE's contract (see node_base): base64 PNG.
                    "format": "png",
                    "encoding": "base64",
                    "data": value,
                },
            }
        )

    # Per-port summaries for edge inspection. Emitted on "cached" too so the
    # edge tooltip and Inspector still populate when a node is served from
    # ExecutionCache.
    entries.append(
        {
            "output_kind": OUTPUT_KIND_TENSOR_SUMMARY,
            OUTPUT_KIND_TENSOR_SUMMARY: _summarize_outputs(result),
        }
    )
    return entries
