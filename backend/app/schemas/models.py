from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class PortDefinitionSchema(BaseModel):
    name: str
    data_type: str
    description: str = ""
    optional: bool = False


class ParamDefinitionSchema(BaseModel):
    name: str
    param_type: str
    default: Any = None
    description: str = ""
    options: list[str] = []
    min_value: float | None = None
    max_value: float | None = None
    # Conditional visibility — forwarded verbatim to the frontend.
    # See ParamDefinition.visible_when for semantics.
    visible_when: dict[str, Any] | None = None


class NodeDefinition(BaseModel):
    node_name: str
    category: str
    description: str
    inputs: list[PortDefinitionSchema]
    outputs: list[PortDefinitionSchema]
    params: list[ParamDefinitionSchema]
    # "builtin" | "custom" | "plugin:<id>" — populated from the class's __module__
    # so the frontend can badge plugin-provided nodes and prompt for install
    # when a graph references a node from a plugin that isn't loaded.
    provider: str = "builtin"


class NodeData(BaseModel):
    id: str
    type: str
    position: dict[str, float] = {"x": 0, "y": 0}
    data: dict[str, Any] = {}


class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: str = ""
    targetHandle: str = ""
    type: Literal["data", "trigger"] = "data"


class GraphData(BaseModel):
    nodes: list[NodeData]
    edges: list[EdgeData]
    name: str = "Untitled"
    description: str = ""
    presets: list[PresetDefinition] = []


class GraphValidationResponse(BaseModel):
    valid: bool
    errors: list[str] = []


class NodeExecutionStatus(BaseModel):
    node_id: str
    status: str  # running | completed | error
    data: dict[str, Any] | None = None


# ── Graph-as-a-function schemas ─────────────────────────────────


class RunError(BaseModel):
    code: str
    message: str
    node_id: str | None = None
    details: list[Any] | None = None


class RunTiming(BaseModel):
    total_s: float


class RunEnvelope(BaseModel):
    """Response envelope for POST /api/graph/run/{name}.

    Every key is ALWAYS present (null when not applicable); HTTP status
    mirrors ``status``/``error.code``. Forward compatibility (also stated
    on the docs page): clients MUST ignore unknown envelope fields, and
    ``error.code`` is an open enum — treat unknown codes as generic
    errors. The ``job`` key is reserved by name for the future async mode.
    """

    status: Literal["ok", "error"]
    run_id: str
    graph: str
    device: str | None = None
    outputs: dict[str, Any] | None = None
    error: RunError | None = None
    timing: RunTiming | None = None


class ContractInputSchema(BaseModel):
    name: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


class ContractOutputSchema(BaseModel):
    name: str
    type: str = "ANY"
    description: str = ""


class GraphContractResponse(BaseModel):
    graph: str
    inputs: list[ContractInputSchema]
    outputs: list[ContractOutputSchema]
    problems: list[str] = []


# ── Preset schemas ──────────────────────────────────────────────

class InternalNodeSchema(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = {}


class InternalEdgeSchema(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class ExposedPortSchema(BaseModel):
    name: str
    internal_node: str
    internal_port: str
    data_type: str = ""
    description: str = ""


class ExposedParamSchema(BaseModel):
    internal_node: str
    param_name: str
    display_name: str
    group: str = ""
    param_def: ParamDefinitionSchema | None = None


class PresetDefinition(BaseModel):
    preset_name: str
    category: str
    description: str
    tags: list[str] = []
    nodes: list[InternalNodeSchema]
    edges: list[InternalEdgeSchema]
    exposed_inputs: list[ExposedPortSchema]
    exposed_outputs: list[ExposedPortSchema]
    exposed_params: list[ExposedParamSchema]


class CreatePresetRequest(BaseModel):
    name: str
    description: str = ""
    category: str = "Custom"
    tags: list[str] = []
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


# Rebuild models that use forward references
GraphData.model_rebuild()
