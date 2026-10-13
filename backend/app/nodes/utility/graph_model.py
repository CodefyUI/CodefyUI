"""DAG-based model builder for the SequentialModel node.

Builds an nn.Module whose forward pass walks a topologically-sorted DAG of
layer nodes. Supports explicit Input/Output boundary nodes with named ports,
plain layers, and merge layers (Add, Concat, Multiply, Subtract, Mean, Stack).
"""
from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

import torch
import torch.nn as nn

from .sequential_node import _build_layer  # reuse existing builders

logger = logging.getLogger(__name__)


_MERGE_TYPES = {"Add", "Concat", "Multiply", "Subtract", "Mean", "Stack"}
# Merges whose result depends on operand order. The layer editor gives each
# of these one named target handle per operand ("a"/"b" for Subtract,
# "in0", "in1", ... for Concat and Stack), and the operands are taken in
# handle order. Add, Multiply and Mean are order-free and keep one handle.
_ORDERED_MERGE_TYPES = {"Subtract", "Concat", "Stack"}
_SUBTRACT_HANDLES = {"a": 0, "b": 1}
_INDEXED_HANDLE = re.compile(r"in(\d+)")
_BOUNDARY_TYPES = {"Input", "Output"}


class GraphModelModule(nn.Module):
    """nn.Module that executes a DAG of layers in topological order."""

    def __init__(
        self,
        layers: dict[str, nn.Module],
        merge_specs: dict[str, dict[str, Any]],
        topo_order: list[str],
        incoming: dict[str, list[tuple[str, str | None]]],
        input_node_id: str,
        output_node_id: str,
        input_ports: list[tuple[str, str]],
        output_ports: list[tuple[str, str]],
        output_incoming: list[tuple[str, str | None, str | None]],
        layer_types: dict[str, str] | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        # The spec's type name per layer id, for error messages. A wrapper
        # such as LSTMBlock reads better as the "LSTM" the user picked.
        self.layer_types = dict(layer_types or {})
        self.merge_specs = merge_specs
        self.topo_order = topo_order
        self.incoming = incoming
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.output_incoming = output_incoming

    def forward(self, *args, **kwargs):
        # Resolve named inputs
        if args and kwargs:
            raise ValueError("GraphModel.forward: pass either positional or keyword args, not both")
        if args:
            if len(args) != len(self.input_ports):
                raise ValueError(
                    f"GraphModel.forward: expected {len(self.input_ports)} positional args, got {len(args)}"
                )
            named = {port_name: arg for (_, port_name), arg in zip(self.input_ports, args)}
        else:
            for _, port_name in self.input_ports:
                if port_name not in kwargs:
                    raise ValueError(f"GraphModel.forward: missing required input '{port_name}'")
            expected = {port_name for _, port_name in self.input_ports}
            extra = set(kwargs.keys()) - expected
            if extra:
                raise ValueError(f"GraphModel.forward: unexpected input(s): {sorted(extra)}")
            named = {port_name: kwargs[port_name] for _, port_name in self.input_ports}

        values: dict[tuple[str, str | None], torch.Tensor] = {}
        for port_id, port_name in self.input_ports:
            values[(self.input_node_id, port_id)] = named[port_name]

        outputs: dict[str, torch.Tensor] = {}
        for node_id in self.topo_order:
            if node_id == self.input_node_id:
                continue
            if node_id == self.output_node_id:
                for port_id, port_name in self.output_ports:
                    incoming_edges = [(src, sh) for (src, sh, th) in self.output_incoming if th == port_id]
                    if len(incoming_edges) != 1:
                        raise ValueError(
                            f"Output port '{port_name}' must have exactly one incoming edge, got {len(incoming_edges)}"
                        )
                    src, sh = incoming_edges[0]
                    outputs[port_name] = values[(src, sh)]
                continue

            inputs = [values[(src, sh)] for (src, sh) in self.incoming.get(node_id, [])]

            if node_id in self.merge_specs:
                spec = self.merge_specs[node_id]
                try:
                    values[(node_id, None)] = _apply_merge(spec["type"], inputs, spec.get("params", {}))
                except Exception as exc:
                    _raise_naming_the_layer(exc, node_id, spec["type"], "")
            else:
                if len(inputs) != 1:
                    raise ValueError(f"Plain layer '{node_id}' expects 1 input, got {len(inputs)}")
                layer = self.layers[node_id]
                try:
                    values[(node_id, None)] = layer(inputs[0])
                except Exception as exc:
                    _raise_naming_the_layer(
                        exc, node_id,
                        # getattr: a model pickled before #703 has no table.
                        getattr(self, "layer_types", {}).get(node_id, type(layer).__name__),
                        layer.extra_repr(),
                    )

        if len(self.output_ports) == 1:
            return outputs[self.output_ports[0][1]]
        return outputs


def _raise_naming_the_layer(exc: Exception, node_id: str, layer_type: str, detail: str) -> None:
    """Re-raise *exc* worded to say which layer raised it (#703).

    torch's own message names the operation (``linear(): input and
    weight.T shapes cannot be multiplied``) but not the layer, which in a
    29-layer graph leaves the user guessing. The prefix carries the layer's
    id and type plus its ``extra_repr()`` -- the constructor arguments --
    so the mismatching ``in_features`` is on screen next to the shapes.
    The original exception stays attached as ``__cause__``.

    The exception keeps its class where the class can be rebuilt from a
    message, so anything that catches ``ValueError`` or ``RuntimeError``
    still does. An out-of-memory error is re-raised unchanged: the engine
    recognises it by type and message and gives it hints of its own.
    """
    from ...core.error_handling import is_out_of_memory

    if is_out_of_memory(exc):
        raise exc
    what = f"{layer_type}, {detail}" if detail else layer_type
    message = f'layer "{node_id}" ({what}): {exc}'
    try:
        renamed = type(exc)(message)
    except Exception:  # noqa: BLE001 - a class that needs other args
        renamed = RuntimeError(message)
    if str(renamed) != message:
        renamed = RuntimeError(message)
    raise renamed from exc


def _apply_merge(merge_type: str, inputs: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
    if merge_type == "Add":
        if len(inputs) < 2:
            raise ValueError("Add requires 2+ inputs")
        out = inputs[0]
        for t in inputs[1:]:
            out = out + t
        return out
    if merge_type == "Multiply":
        if len(inputs) < 2:
            raise ValueError("Multiply requires 2+ inputs")
        out = inputs[0]
        for t in inputs[1:]:
            out = out * t
        return out
    if merge_type == "Subtract":
        if len(inputs) != 2:
            raise ValueError("Subtract requires exactly 2 inputs")
        return inputs[0] - inputs[1]
    if merge_type == "Mean":
        if len(inputs) < 2:
            raise ValueError("Mean requires 2+ inputs")
        return torch.stack(inputs, dim=0).mean(dim=0)
    if merge_type == "Concat":
        dim = int(params.get("dim", 1))
        return torch.cat(inputs, dim=dim)
    if merge_type == "Stack":
        dim = int(params.get("dim", 1))
        return torch.stack(inputs, dim=dim)
    raise ValueError(f"Unknown merge type: {merge_type}")


def _operand_rank(merge_type: str, handle: Any) -> int | None:
    """Position of the operand wired to *handle*, or None for no named handle."""
    if not isinstance(handle, str):
        return None
    if merge_type == "Subtract":
        return _SUBTRACT_HANDLES.get(handle)
    match = _INDEXED_HANDLE.fullmatch(handle)
    return int(match.group(1)) if match else None


def _order_operands(
    node: dict[str, Any],
    edges_in: list[tuple[str, str | None, str | None]],
) -> list[tuple[str, str | None, str | None]]:
    """Sort a merge node's incoming edges by their target handle.

    Falls back to the edge-list order unless every incoming edge names a
    handle, so an old spec (no handles at all) builds exactly as before.
    """
    merge_type, node_id = node["type"], node["id"]
    ranks = [_operand_rank(merge_type, th) for (_src, _sh, th) in edges_in]
    if not edges_in or any(r is None for r in ranks):
        return edges_in
    seen: dict[int, str | None] = {}
    for (_src, _sh, th), rank in zip(edges_in, ranks):
        if rank in seen:
            raise ValueError(
                f'{merge_type} layer "{node_id}": two edges go into input "{th}". '
                "Connect one branch per input and delete the extra edge."
            )
        seen[rank] = th
    order = sorted(range(len(edges_in)), key=lambda i: ranks[i])
    return [edges_in[i] for i in order]


def build_graph_model(spec: dict[str, Any]) -> GraphModelModule:
    """Build a GraphModelModule from a v2 spec dict."""
    if spec.get("version") != 2:
        raise ValueError(f"Unsupported graph spec version: {spec.get('version')}")

    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])

    nodes_by_id = {n["id"]: n for n in nodes}

    input_nodes = [n for n in nodes if n["type"] == "Input"]
    output_nodes = [n for n in nodes if n["type"] == "Output"]
    if len(input_nodes) != 1:
        raise ValueError(f"Graph must have exactly one Input node, found {len(input_nodes)}")
    if len(output_nodes) != 1:
        raise ValueError(f"Graph must have exactly one Output node, found {len(output_nodes)}")

    input_node = input_nodes[0]
    output_node = output_nodes[0]

    input_ports_raw = input_node.get("ports", [])
    output_ports_raw = output_node.get("ports", [])
    if not input_ports_raw:
        raise ValueError("Input node must have at least one port")
    if not output_ports_raw:
        raise ValueError("Output node must have at least one port")

    in_names = [p["name"] for p in input_ports_raw]
    if len(set(in_names)) != len(in_names):
        raise ValueError("Input port names must be unique")
    out_names = [p["name"] for p in output_ports_raw]
    if len(set(out_names)) != len(out_names):
        raise ValueError("Output port names must be unique")

    input_ports = [(p["id"], p["name"]) for p in input_ports_raw]
    output_ports = [(p["id"], p["name"]) for p in output_ports_raw]

    # incoming[node_id] = list of (src_id, src_handle, target_handle)
    incoming: dict[str, list[tuple[str, str | None, str | None]]] = {n["id"]: [] for n in nodes}
    outgoing: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in nodes_by_id or tgt not in nodes_by_id:
            raise ValueError(f"Edge references unknown node: {src} -> {tgt}")
        incoming[tgt].append((src, e.get("sourceHandle"), e.get("targetHandle")))
        outgoing[src].append(tgt)

    # Order-sensitive merges take their operands in target-handle order. A
    # spec saved before the handles existed carries no targetHandle on these
    # edges and keeps the edge-list order it always had.
    for n in nodes:
        if n["type"] in _ORDERED_MERGE_TYPES:
            incoming[n["id"]] = _order_operands(n, incoming[n["id"]])

    # Build layers + merge specs
    layers: dict[str, nn.Module] = {}
    merge_specs: dict[str, dict[str, Any]] = {}
    for n in nodes:
        t = n["type"]
        if t in _BOUNDARY_TYPES:
            continue
        if t in _MERGE_TYPES:
            merge_specs[n["id"]] = {"type": t, "params": n.get("params", {})}
            continue
        cfg = {"type": t, **n.get("params", {})}
        layers[n["id"]] = _build_layer(cfg)

    # Topological sort (Kahn's algorithm)
    in_degree = {nid: len(incoming[nid]) for nid in nodes_by_id}
    queue = deque(nid for nid, d in in_degree.items() if d == 0)
    topo_order: list[str] = []
    while queue:
        nid = queue.popleft()
        topo_order.append(nid)
        for tgt in outgoing[nid]:
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)
    if len(topo_order) != len(nodes_by_id):
        raise ValueError("Graph contains a cycle")

    # Validation: each Output port has exactly 1 incoming, each Input port used 1+
    out_port_ids = {p["id"] for p in output_ports_raw}
    in_port_ids = {p["id"] for p in input_ports_raw}
    out_port_use: dict[str, int] = {pid: 0 for pid in out_port_ids}
    in_port_use: dict[str, int] = {pid: 0 for pid in in_port_ids}
    for e in edges:
        if e["target"] == output_node["id"]:
            th = e.get("targetHandle")
            if th not in out_port_ids:
                raise ValueError(f"Edge into Output references unknown port id: {th}")
            out_port_use[th] += 1
        if e["source"] == input_node["id"]:
            sh = e.get("sourceHandle")
            if sh not in in_port_ids:
                raise ValueError(f"Edge from Input references unknown port id: {sh}")
            in_port_use[sh] += 1
    for pid, count in out_port_use.items():
        if count != 1:
            name = next(p["name"] for p in output_ports_raw if p["id"] == pid)
            raise ValueError(f"Output port '{name}' must have exactly one incoming edge, got {count}")
    for pid, count in in_port_use.items():
        if count < 1:
            name = next(p["name"] for p in input_ports_raw if p["id"] == pid)
            raise ValueError(f"Input port '{name}' is unused")

    # Plain layers must have exactly 1 incoming
    for nid, edges_in in incoming.items():
        n = nodes_by_id[nid]
        if n["type"] in _BOUNDARY_TYPES or n["type"] in _MERGE_TYPES:
            continue
        if len(edges_in) != 1:
            raise ValueError(f"Plain layer '{nid}' ({n['type']}) must have exactly 1 incoming edge, got {len(edges_in)}")

    incoming_for_module: dict[str, list[tuple[str, str | None]]] = {
        nid: [(src, sh) for (src, sh, _th) in lst] for nid, lst in incoming.items()
    }

    model = GraphModelModule(
        layers=layers,
        merge_specs=merge_specs,
        topo_order=topo_order,
        incoming=incoming_for_module,
        input_node_id=input_node["id"],
        output_node_id=output_node["id"],
        input_ports=input_ports,
        output_ports=output_ports,
        output_incoming=incoming[output_node["id"]],
        layer_types={nid: nodes_by_id[nid]["type"] for nid in layers},
    )
    return model
