from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..schemas.models import (
    ExposedParamSchema,
    ExposedPortSchema,
    InternalEdgeSchema,
    InternalNodeSchema,
    ParamDefinitionSchema,
    PresetDefinition,
)
from .node_registry import NodeRegistry

logger = logging.getLogger(__name__)


class PresetRegistry:
    def __init__(self) -> None:
        self._presets: dict[str, PresetDefinition] = {}

    @property
    def presets(self) -> dict[str, PresetDefinition]:
        return dict(self._presets)

    def discover(self, directory: Path, node_registry: NodeRegistry) -> int:
        count = 0
        if not directory.exists():
            return count
        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                preset = self._load_and_resolve(raw, node_registry)
                self._presets[preset.preset_name] = preset
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", path.name, e)
        return count

    def get(self, name: str) -> PresetDefinition | None:
        return self._presets.get(name)

    def all(self) -> list[PresetDefinition]:
        return list(self._presets.values())

    def clear(self) -> None:
        self._presets.clear()

    def _load_and_resolve(self, raw: dict[str, Any], node_registry: NodeRegistry) -> PresetDefinition:
        nodes = [InternalNodeSchema(**n) for n in raw["nodes"]]
        edges = [InternalEdgeSchema(**e) for e in raw["edges"]]

        # Validate internal node types exist
        for node in nodes:
            if not node_registry.get(node.type):
                raise ValueError(f"Internal node type '{node.type}' not found in registry")

        # Resolve exposed input port data_types
        exposed_inputs = []
        for port_raw in raw.get("exposed_inputs", []):
            port = ExposedPortSchema(**port_raw)
            if not port.data_type:
                port.data_type = self._resolve_port_type(
                    port.internal_node, port.internal_port, "input", nodes, node_registry
                )
            exposed_inputs.append(port)

        # Resolve exposed output port data_types
        exposed_outputs = []
        for port_raw in raw.get("exposed_outputs", []):
            port = ExposedPortSchema(**port_raw)
            if not port.data_type:
                port.data_type = self._resolve_port_type(
                    port.internal_node, port.internal_port, "output", nodes, node_registry
                )
            exposed_outputs.append(port)

        # Resolve exposed params
        exposed_params = []
        for param_raw in raw.get("exposed_params", []):
            param = ExposedParamSchema(**param_raw)
            if param.param_def is None:
                param.param_def = self._resolve_param_def(
                    param.internal_node, param.param_name, nodes, node_registry
                )
            exposed_params.append(param)

        return PresetDefinition(
            preset_name=raw["preset_name"],
            category=raw.get("category", "Preset"),
            description=raw.get("description", ""),
            tags=raw.get("tags", []),
            nodes=nodes,
            edges=edges,
            exposed_inputs=exposed_inputs,
            exposed_outputs=exposed_outputs,
            exposed_params=exposed_params,
        )

    def _resolve_port_type(
        self,
        internal_node_id: str,
        port_name: str,
        direction: str,
        nodes: list[InternalNodeSchema],
        node_registry: NodeRegistry,
    ) -> str:
        node = next((n for n in nodes if n.id == internal_node_id), None)
        if not node:
            return "ANY"
        cls = node_registry.get(node.type)
        if not cls:
            return "ANY"
        # #196: the DYNAMIC form. `InternalNodeSchema` carries the stored
        # params, so a port that only exists at this instance's port count
        # (ComposeTransform's step_3.., PythonScript's in3../out3..) resolves
        # to its real data type instead of silently falling through to "ANY"
        # and colouring the preset's handle grey.
        ports = (
            cls.define_inputs_dynamic(node.params) if direction == "input"
            else cls.define_outputs_dynamic(node.params)
        )
        port = next((p for p in ports if p.name == port_name), None)
        return port.data_type.value if port else "ANY"

    def _resolve_param_def(
        self,
        internal_node_id: str,
        param_name: str,
        nodes: list[InternalNodeSchema],
        node_registry: NodeRegistry,
    ) -> ParamDefinitionSchema | None:
        node = next((n for n in nodes if n.id == internal_node_id), None)
        if not node:
            return None
        cls = node_registry.get(node.type)
        if not cls:
            return None
        for p in cls.define_params():
            if p.name == param_name:
                return ParamDefinitionSchema(
                    name=p.name,
                    param_type=p.param_type.value,
                    default=p.default,
                    description=p.description,
                    options=p.options,
                    min_value=p.min_value,
                    max_value=p.max_value,
                    # Forwarded so an exposed preset param behaves like the
                    # inner one it stands for. ``visible_when`` was dropped
                    # here before #134, which quietly made a conditional
                    # param unconditional the moment it was exposed.
                    visible_when=p.visible_when,
                    advanced=p.advanced,
                    # Same reason: an exposed SELECT that lost its pack
                    # gating would offer options this machine cannot load.
                    option_packs=p.option_packs,
                )
        return None


preset_registry = PresetRegistry()
