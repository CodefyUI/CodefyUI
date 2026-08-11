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
state: :func:`declared_media_ports` asks the node registry what a node
declares. Nothing here touches a socket, a request or a database.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from ..config import settings
from .node_base import MEDIA_CHART, MEDIA_IMAGE, MEDIA_VIDEO
from .node_registry import registry

logger = logging.getLogger(__name__)

OUTPUT_KIND_TEXT = "text"
OUTPUT_KIND_IMAGE = "image"
OUTPUT_KIND_CHART = "chart"
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


def declared_media_ports(nodes: list[Any]) -> dict[str, dict[str, list[str]]]:
    """Map ``node id -> {media kind -> output ports declaring it}``.

    Resolved from the registry once per run, so the hot progress path only
    does a dict lookup. Nodes the registry does not know (presets, typos)
    and malformed entries are silently skipped: an unresolvable node simply
    has no declared media, which is the safe answer.

    The media kind is whatever string the port declared — this function does
    not know the vocabulary and never filters on it. That is what makes the
    "a node pack can add its own kind without a core release" promise in the
    module docstring true rather than aspirational (#130): a third-party pack
    declaring ``media="waveform"`` reaches the wire as
    ``{"output_kind": "waveform", ...}`` with no change here.
    """
    ports: dict[str, dict[str, list[str]]] = {}
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
        declared: dict[str, list[str]] = {}
        for port in definitions:
            kind = getattr(port, "media", None)
            if isinstance(kind, str) and kind:
                declared.setdefault(kind, []).append(port.name)
        if declared:
            ports[node_id] = declared
    return ports


def _image_payload(value: Any) -> dict[str, Any] | None:
    """Wrap a MEDIA_IMAGE port value in its container, or skip it."""
    if not isinstance(value, str) or not value:
        return None
    # MEDIA_IMAGE's contract (see node_base): base64 PNG.
    return {"format": "png", "encoding": "base64", "data": value}


def _object_payload(value: Any) -> dict[str, Any] | None:
    """Default envelope: ship a JSON-object port value through untouched.

    Used by MEDIA_CHART and by any kind core has never heard of. A value that
    is not a non-empty dict is skipped rather than shipped, so a declared port
    that produced nothing this run is silently absent instead of arriving as
    an empty picture — the same rule the image branch has always used.
    """
    if not isinstance(value, dict) or not value:
        return None
    return value


#: Keys of a MEDIA_VIDEO reference the wire forwards. A closed list rather
#: than passthrough so a node cannot smuggle an oversized payload (or the
#: bytes themselves) through a port declared as a reference.
_VIDEO_REFERENCE_KEYS = ("path", "url", "format", "fps", "frames", "width",
                         "height", "bytes")


def _video_payload(value: Any) -> dict[str, Any] | None:
    """Validate a MEDIA_VIDEO reference dict, or skip it.

    The contract (see node_base) is a pointer to a file under MEDIA_DIR, so
    the two things worth refusing here are a non-reference value and a path
    that is not relative — an absolute path in the event stream would leak
    the server's filesystem layout to every attached client and could never
    be fetched through ``/api/media`` anyway.
    """
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    fmt = value.get("format")
    if not isinstance(path, str) or not path or not isinstance(fmt, str):
        return None
    # Judged under BOTH path flavours, because the native Path is
    # platform-shaped in both directions: on Windows "/leak/a.mp4" has no
    # drive and counts as relative; on POSIX "C:/leak/a.mp4" is one odd
    # filename that is_absolute() waves through. A reference must be
    # relative and descend-only on every platform.
    posix, windows = PurePosixPath(path), PureWindowsPath(path)
    if (posix.is_absolute() or windows.is_absolute() or windows.drive
            or ".." in posix.parts or ".." in windows.parts):
        return None
    return {key: value[key] for key in _VIDEO_REFERENCE_KEYS if key in value}


#: Per-kind envelope builders. A kind absent from this map falls back to
#: :func:`_object_payload`, which is why adding a kind needs no entry here
#: unless its payload needs a container the node's raw value does not have.
_MEDIA_PAYLOADS: dict[str, Callable[[Any], dict[str, Any] | None]] = {
    MEDIA_IMAGE: _image_payload,
    MEDIA_CHART: _object_payload,
    MEDIA_VIDEO: _video_payload,
}

#: Keys an entry dict already uses for its own bookkeeping. Since an entry
#: stores its payload under a key NAMED BY THE KIND, a third-party pack
#: declaring ``media="port"`` would overwrite the port name, and one declaring
#: ``media="elided"`` would forge the marker ``run_store`` writes when a
#: payload exceeds the size cap — making an intact entry read as truncated, or
#: vice versa. Refused rather than silently renamed: a kind that collides is a
#: bug in the declaring pack, and quietly serving it under another name would
#: hide that from its author.
_RESERVED_ENTRY_KEYS = frozenset(
    {"output_kind", "port", "elided", "bytes", "cap_bytes"}
)


def build_node_output_entries(
    status: str,
    result: dict[str, Any] | None,
    media_ports: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build the typed ``outputs`` list for one ``node_status`` message.

    ``media_ports`` maps a media kind to the output port names this node
    *declared* for it (see :func:`declared_media_ports`). Values on any other
    port stay plain data no matter what they look like -- there is
    deliberately no length or character-class inspection anywhere in this
    function.
    """
    if not result:
        return []

    # A Sequence here is the pre-#130 signature (a bare list of image ports).
    # Left unguarded, iterating a string yields its characters and a list
    # yields port names used as media kinds -- either way the caller gets
    # plausible-looking nonsense instead of an error.
    if media_ports is not None and not isinstance(media_ports, Mapping):
        raise TypeError(
            "media_ports must be a mapping of media kind -> port names, e.g. "
            f"{{'image': ['plot']}}; got {type(media_ports).__name__}"
        )

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

    # Declared media ports, kind by kind. A declared port that produced
    # nothing this run is skipped rather than announced as an empty payload.
    for kind, port_names in (media_ports or {}).items():
        if kind in _RESERVED_ENTRY_KEYS:
            logger.warning(
                "skipping media kind %r: it collides with a reserved entry key "
                "(%s). Rename the kind in the declaring node pack.",
                kind, ", ".join(sorted(_RESERVED_ENTRY_KEYS)),
            )
            continue
        envelope = _MEDIA_PAYLOADS.get(kind, _object_payload)
        for port in port_names:
            payload = envelope(result.get(port))
            if payload is None:
                continue
            entries.append({"output_kind": kind, "port": port, kind: payload})

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
