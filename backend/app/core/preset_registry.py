from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from ..schemas.models import (
    ExposedParamSchema,
    ExposedPortSchema,
    InternalEdgeSchema,
    InternalNodeSchema,
    ParamDefinitionSchema,
    PresetDefinition,
)
from .device_utils import device_options
from .node_registry import NodeRegistry
from .validation_issues import ValidationIssue, validation_issue

logger = logging.getLogger(__name__)

#: Node types that run a preset they NAME in a param, rather than one placed
#: as a card, and that param. Map is the only one: its ``subgraph`` setting
#: names a preset that it looks up in this registry when it runs
#: (``map_node.py``) -- the installed presets, never a graph's own
#: ``presets[]``.
PRESET_NAME_PARAMS: dict[str, str] = {"Map": "subgraph"}


class NamedPreset(NamedTuple):
    """One node that runs a preset by name (:meth:`PresetRegistry.named_presets`)."""

    #: The node whose setting holds the name: one of the graph's own, or a
    #: node of a preset found before it, as ``<map>__<inner>``.
    node_id: str
    #: The setting that holds the name.
    param: str
    name: str
    #: What the registry has under the name; None when it has nothing.
    definition: PresetDefinition | None
    #: The one whose preset this node is in; None for one of the graph's own.
    inside: NamedPreset | None

    def refusal(self) -> ValidationIssue:
        """What validation and an export say when the registry has no such
        preset: the code a card naming a missing preset gets, so a client
        says both in the same words."""
        return validation_issue(
            "unknown_preset",
            f"Unknown preset: {self.name} (named by the '{self.param}' "
            f"setting of node {self.node_id})",
            node_id=self.node_id, preset=self.name,
        )


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
                self.load_file(path, node_registry)
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", path.name, e)
        return count

    def load_file(self, path: Path, node_registry: NodeRegistry) -> PresetDefinition:
        """Load one preset file and register it under its name.

        Raises when the file cannot be read or resolved; :meth:`discover`
        logs and skips such a file. ``POST /api/presets/create`` loads only
        the file it wrote, so the other presets keep the names they won.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        preset = self._load_and_resolve(raw, node_registry)
        self.add(preset)
        return preset

    def add(self, preset: PresetDefinition) -> None:
        """Register *preset* under its name, in place of any preset of that
        name. :meth:`load_file` registers what it read this way. A definition
        with no file is registered with this alone: the copy of a preset that
        an exported script carries (``runtime.use_preset_copies``).
        """
        self._presets[preset.preset_name] = preset

    def get(self, name: str) -> PresetDefinition | None:
        return self._presets.get(name)

    def named_presets(self, nodes: Iterable[Mapping[str, Any]]) -> Iterator[NamedPreset]:
        """Each node that runs a preset by name, with what this registry has
        under that name.

        *nodes* are graph nodes (``id``, ``type``, ``data.params``). A node
        runs a preset by name when its type is in :data:`PRESET_NAME_PARAMS`
        and that param holds a name; an empty one names nothing, and Map
        refuses it when it runs. The graph's own nodes come first, in the
        order given, then the nodes of each preset found, read once each: a
        Map in a preset names the next one, under the id ``<map>__<inner>``
        Map gives its body's nodes when it runs.
        """
        pending: deque[NamedPreset] = deque()

        def queue(node_id: str, node_type: Any, params: Any,
                  inside: NamedPreset | None) -> None:
            param = (
                PRESET_NAME_PARAMS.get(node_type)
                if isinstance(node_type, str) else None
            )
            if param is None or not isinstance(params, Mapping):
                return
            name = params.get(param)
            if isinstance(name, str) and name:
                pending.append(
                    NamedPreset(node_id, param, name, self.get(name), inside))

        for node in nodes:
            data = node.get("data")
            queue(
                str(node.get("id", "")),
                node.get("type"),
                data.get("params") if isinstance(data, Mapping) else None,
                None,
            )
        read: set[str] = set()
        while pending:
            named = pending.popleft()
            yield named
            if named.definition is None or named.name in read:
                continue
            read.add(named.name)
            for inner in named.definition.nodes:
                queue(f"{named.node_id}__{inner.id}", inner.type, inner.params, named)

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

        # A field the file leaves out takes the model's default, which is
        # also what a graph's own presets[] entry gets (#541).
        return PresetDefinition.model_validate({
            **raw,
            "nodes": nodes,
            "edges": edges,
            "exposed_inputs": exposed_inputs,
            "exposed_outputs": exposed_outputs,
            "exposed_params": exposed_params,
        })

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
                    # A device SELECT shares the machine-narrowed list the
                    # node API serves, so an exposed one offers the same
                    # devices the inner param does.
                    options=device_options(p.name, p.options),
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
