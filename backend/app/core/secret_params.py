"""SECRET-typed parameter handling for saved graphs.

A node may declare a param as ``ParamType.SECRET`` (e.g. LLMChat's
``openai_api_key``). Such values live only in canvas / runtime state; they
must never be written to a saved graph file, a published snapshot, or an
exported JSON. This module is the single source of truth for:

- which params of a given node type are secret (registry lookup),
- scrubbing secret values out of a graph's nodes before persistence
  (defense-in-depth for the save endpoint), and
- detecting a leftover non-empty secret in an already-on-disk graph
  (publish pre-flight for hand-edited files).

Every function operates on the serialized node shape ``{"id", "type",
"data": {"params": {...}}}`` used by GraphData / the saved JSON, so the
save path (pydantic dump -> dict) and the publish path (json.loads dict)
share one implementation. Unknown node types (notes, presets, plugin
nodes not currently loaded) carry no known secret params and are left
untouched.
"""

from __future__ import annotations

from typing import Any, Iterable

from .node_base import ParamType
from .node_registry import registry


def secret_param_names(node_type: str) -> set[str]:
    """Names of the SECRET-typed params for ``node_type``.

    Empty set when the type is unknown to the registry (notes, presets,
    unloaded plugin nodes) or declares no secret params.
    """
    if not node_type:
        return set()
    node_cls = registry.get(node_type)
    if node_cls is None:
        return set()
    return {
        p.name
        for p in node_cls.define_params()
        if p.param_type == ParamType.SECRET
    }


def _params_of(node: dict[str, Any]) -> dict[str, Any] | None:
    data = node.get("data")
    if not isinstance(data, dict):
        return None
    params = data.get("params")
    return params if isinstance(params, dict) else None


def _internal_params_of(node: dict[str, Any]) -> dict[str, Any] | None:
    """The ``data.internalParams`` map of a preset node, keyed by internal
    node id -> {param name: value}. None when absent or malformed."""
    data = node.get("data")
    if not isinstance(data, dict):
        return None
    internal = data.get("internalParams")
    return internal if isinstance(internal, dict) else None


def _preset_secret_param_map(node_type: str) -> dict[str, set[str]]:
    """For a ``preset:<name>`` node type, map each internal node id to the
    set of its SECRET-typed param names.

    A preset node embeds per-inner-node overrides in ``data.internalParams``;
    those inner nodes are real registry types (e.g. an inner ``LLMChat``),
    so a hand-edited graph could bake an API key into
    ``internalParams["<inner_id>"]["openai_api_key"]`` — invisible to the
    plain ``data.params`` scrub. We resolve the preset through the preset
    registry (to learn each inner node's id + type) and the node registry
    (to learn which of its params are secret). Empty when the type is not a
    preset, the preset is unknown, or no inner node declares a secret.
    """
    if not node_type.startswith("preset:"):
        return {}
    # Lazy import: preset_registry pulls in schemas/node_registry; importing
    # it at module load would risk a cycle (graph_engine uses the same
    # lazy-import pattern for exactly this reason).
    from .preset_registry import preset_registry

    preset = preset_registry.get(node_type[len("preset:"):])
    if preset is None:
        return {}
    result: dict[str, set[str]] = {}
    for internal in preset.nodes:
        names = secret_param_names(internal.type)
        if names:
            result[internal.id] = names
    return result


def _is_nonempty_secret(value: Any) -> bool:
    """A secret value counts as "present" for lint/scrub purposes when it is
    anything other than the empty string or ``None`` (the scrubbed form)."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    # A non-string truthy value (should not happen for a text key, but a
    # hand-edited file could contain one) is still a leaked secret.
    return bool(value)


def scrub_graph_secrets(nodes: Iterable[dict[str, Any]]) -> int:
    """Blank every SECRET-typed param value in ``nodes`` (in place).

    Returns the number of values changed. Only params that are both
    declared secret AND currently non-empty are rewritten to ``""`` — a
    node with no secret params, or an unknown type, is untouched.
    """
    changed = 0
    for node in nodes:
        node_type = node.get("type", "")
        # Regular node: blank its own declared SECRET params.
        names = secret_param_names(node_type)
        if names:
            params = _params_of(node)
            if params is not None:
                for name in names:
                    if name in params and _is_nonempty_secret(params[name]):
                        params[name] = ""
                        changed += 1
        # Preset node: blank SECRET params embedded per inner node in
        # data.internalParams (secret_param_names is empty for a preset:*
        # type, so the block above no-ops for it — no double counting).
        preset_secrets = _preset_secret_param_map(node_type)
        if preset_secrets:
            internal_params = _internal_params_of(node)
            if internal_params is not None:
                for internal_id, secret_names in preset_secrets.items():
                    inner = internal_params.get(internal_id)
                    if not isinstance(inner, dict):
                        continue
                    for name in secret_names:
                        if name in inner and _is_nonempty_secret(inner[name]):
                            inner[name] = ""
                            changed += 1
    return changed


def find_secret_violations(
    nodes: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Report every non-empty SECRET-typed param still present in ``nodes``.

    Each entry is ``{"node_id": <id>, "param": <param name>}``. Used by the
    publish pre-flight to reject a hand-edited graph file that a client
    dropped into the graphs dir with a secret baked in.
    """
    violations: list[dict[str, str]] = []
    for node in nodes:
        node_type = node.get("type", "")
        node_id = str(node.get("id", ""))
        # Regular node: report its own declared SECRET params.
        names = secret_param_names(node_type)
        if names:
            params = _params_of(node)
            if params is not None:
                for name in sorted(names):
                    if _is_nonempty_secret(params.get(name)):
                        violations.append({"node_id": node_id, "param": name})
        # Preset node: report secrets baked into data.internalParams. The
        # param is reported as ``<inner_id>.<param>`` so the message names the
        # exact inner slot the client must clear.
        preset_secrets = _preset_secret_param_map(node_type)
        if preset_secrets:
            internal_params = _internal_params_of(node) or {}
            for internal_id in sorted(preset_secrets):
                inner = internal_params.get(internal_id)
                if not isinstance(inner, dict):
                    continue
                for name in sorted(preset_secrets[internal_id]):
                    if _is_nonempty_secret(inner.get(name)):
                        violations.append({
                            "node_id": node_id,
                            "param": f"{internal_id}.{name}",
                        })
    return violations
