from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class MapNode(BaseNode):
    NODE_NAME = "Map"
    CATEGORY = "Data Flow"
    DESCRIPTION = (
        "Apply a preset (subgraph) to each element in a list. "
        "Returns a list of results. Functional-style batch processing."
    )

    # `items` is a whole collection consumed one element at a time. Aligning
    # the port would copy every element onto the device before the first body
    # node runs -- peak residency goes from one item to all of them, which is
    # the opposite of what iterating it was for. Each element is still
    # aligned, individually, by the `invoke_node` call on the body node.
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="items", data_type=DataType.LIST, description="List of items to process"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="results", data_type=DataType.LIST, description="List of processed results"),
            PortDefinition(name="count", data_type=DataType.SCALAR, description="Number of items processed"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="subgraph",
                param_type=ParamType.STRING,
                default="",
                description="Name of the preset/subgraph to apply to each item",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        from ...core.graph_engine import invoke_node, topological_sort
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            stop_checker,
        )
        from ...core.node_registry import registry as node_registry
        from ...core.preset_registry import preset_registry

        items = inputs.get("items", [])
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"Map expects a list input, got {type(items).__name__}")

        subgraph_name = params.get("subgraph", "")
        if not subgraph_name:
            raise ValueError("subgraph parameter is required")

        preset = preset_registry.get(subgraph_name)
        if not preset:
            raise ValueError(f"Subgraph '{subgraph_name}' not found")
        if not preset.exposed_inputs:
            raise ValueError(f"Subgraph '{subgraph_name}' has no exposed inputs")
        if not preset.exposed_outputs:
            raise ValueError(f"Subgraph '{subgraph_name}' has no exposed outputs")

        in_port = preset.exposed_inputs[0]
        out_port = preset.exposed_outputs[0]

        nodes_list = [
            {"id": n.id, "type": n.type, "data": {"params": dict(n.params)}}
            for n in preset.nodes
        ]
        edges_list = [
            {
                "source": e.source,
                "target": e.target,
                "sourceHandle": e.sourceHandle,
                "targetHandle": e.targetHandle,
            }
            for e in preset.edges
        ]
        order = topological_sort(nodes_list, edges_list)
        node_map = {n["id"]: n for n in nodes_list}

        incoming: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for edge in edges_list:
            incoming[edge["target"]].append(
                (edge["source"], edge["sourceHandle"], edge["targetHandle"])
            )

        # #122: the outer loop runs a WHOLE subgraph per item, so its cost is
        # len(items) x the subgraph -- easily the longest loop in the engine.
        # The check is per item rather than per inner node: one subgraph pass
        # is the unit of work that leaves consistent state behind.
        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        stopped_at_item: int | None = None

        # The Map's own id, as the engine stamped it on this context. Used
        # to qualify the body's node ids below; "map" only when Map is being
        # driven by something that keeps no per-node identity (a bare
        # ``execute`` in a test).
        outer_node_id = getattr(context, "current_node_id", "") or "map"

        results = []
        for i, item in enumerate(items):
            if should_stop():
                stopped_at_item = i
                break
            node_outputs: dict[str, dict[str, Any]] = {}

            for node_id in order:
                node_def = node_map[node_id]
                node_cls = node_registry.get(node_def["type"])
                if not node_cls:
                    raise ValueError(f"Unknown node type in subgraph: {node_def['type']}")

                node_inputs: dict[str, Any] = {}
                for src_id, src_handle, tgt_handle in incoming.get(node_id, []):
                    if src_id in node_outputs and src_handle in node_outputs[src_id]:
                        node_inputs[tgt_handle] = node_outputs[src_id][src_handle]

                if node_id == in_port.internal_node:
                    node_inputs[in_port.internal_port] = item

                instance = node_cls()

                # #196: the body used to run on ``context=None`` -- the last
                # node-execute call site in the repo that did not go through
                # ``invoke_node``. It cost every inner node its seed stream
                # (``seed_pipeline`` hands back the raw pipeline when there
                # is no context, so augmentation inside a Map was neither
                # reproducible nor isolated, with no error and no warning),
                # its cancellation flag, its metric/artifact sink and its
                # device.
                #
                # A per-inner-node COPY, never the shared object: the same
                # rule ``graph_engine`` follows since #253. Every
                # collaborator on the context -- the stop event, the outbox,
                # ``node_state_store`` -- stays the SAME object; only the
                # scalar id differs. It also means the caller's
                # ``current_node_id`` is untouched, so there is nothing to
                # restore after the loop.
                #
                # The id is QUALIFIED with the Map's own, using the ``a__b``
                # separator preset expansion already uses
                # (``graph_engine._expand_preset``). Preset internals are
                # named ``node_0``, ``node_1``, ... so a bare inner id is
                # unique only within one body, and three things key off this
                # field: ``seeded_for_node``'s ``transform:<id>`` seed label
                # (two transform nodes in one body must draw different
                # streams), ``StatefulModuleMixin``'s ``(graph_id,
                # current_node_id, structure_hash)`` (two same-shaped
                # stateful nodes must not share weights -- across two Map
                # instances as well as within one body), and the node id on
                # every metric/warning/artifact the body emits.
                node_context = context
                if context is not None:
                    node_context = copy.copy(context)
                    node_context.current_node_id = f"{outer_node_id}__{node_id}"

                result = invoke_node(
                    instance,
                    node_inputs,
                    node_def.get("data", {}).get("params", {}),
                    # Deliberately NOT the Map's own callback. The engine
                    # binds that to the MAP node's id, so an inner training
                    # loop's per-batch frames would arrive on the canvas as
                    # the Map node's progress and race the per-item frames
                    # emitted below. The body gets the context (state) but
                    # not the outer node's progress channel (presentation).
                    progress_callback=None,
                    context=node_context,
                )
                node_outputs[node_id] = result

            out = node_outputs.get(out_port.internal_node, {}).get(out_port.internal_port)
            if out is None:
                raise ValueError(f"Subgraph did not produce output for item {i}")
            results.append(out)
            logger.info("Map item %d/%d complete", i + 1, len(items))
            throttle.emit({"event": EVENT_BATCH, "batch": i + 1,
                           "total_batches": len(items)})

        result: dict[str, Any] = {"results": results,
                                  "count": float(len(results))}
        if stopped_at_item is not None:
            result.update(interrupted_result(batch=stopped_at_item,
                                             total_items=len(items)))
        return result
