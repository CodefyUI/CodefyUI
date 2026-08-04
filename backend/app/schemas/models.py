from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ``app.core.seeding`` imports nothing but the standard library, so taking
# the seed bound from its definition rather than restating it here costs no
# import-order risk -- and restating it is how the run path and the export
# path drift apart.
from ..core.seeding import MAX_SEED


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
    # Two-tier parameter UI — see ParamDefinition.advanced. Defaults to
    # False so a plugin built against an older host, whose params carry no
    # such flag, keeps every parameter in the basic tier.
    advanced: bool = False


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


class SegmentGroupData(BaseModel):
    """Teaching Inspector segment (head/tail node span). Mirrors the
    frontend ``SegmentGroup`` shape so save/load round-trips it verbatim
    instead of silently dropping it."""
    id: str
    headNodeId: str
    tailNodeId: str


class GraphData(BaseModel):
    nodes: list[NodeData]
    edges: list[EdgeData]
    name: str = "Untitled"
    description: str = ""
    presets: list[PresetDefinition] = []
    # Persisted so the Teaching Inspector's segment overlays survive a
    # save/load round-trip. Optional; older graph files simply omit it.
    segmentGroups: list[SegmentGroupData] = []


class GraphExportRequest(GraphData):
    """A graph plus the run settings an exported script has to carry.

    Separate from :class:`GraphData` because these are properties of a RUN,
    not of the saved graph: ``/save`` must not start writing them into graph
    files. Both are optional, so an older client (or a hand-rolled ``curl``)
    still exports, it just exports an unseeded script -- which is what every
    export did before core#136.
    """

    #: Canvas seed, baked in as the default for the generated ``--seed``.
    #: ``None`` means the canvas had no seed set.
    #:
    #: core#136 re-review, N-4. Bounded to the SAME range the run path
    #: enforces (``run_service._validate_options``). Without it a hand-rolled
    #: request could bake ``-1`` or ``2**86`` into ``GRAPH_SEED``, producing
    #: an export whose results the canvas would refuse to reproduce because
    #: it rejects that seed outright -- an export that disagrees with the
    #: graph it was exported from is worse than one that fails to build.
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    #: Canvas "deterministic kernels" toggle, default for
    #: ``--deterministic`` / ``--no-deterministic``.
    deterministic: bool = False


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
    """Response envelope for POST /api/graph/run/{name} AND
    POST /api/apps/{slug}/invoke.

    Every key is ALWAYS present (null when not applicable); HTTP status
    mirrors ``status``/``error.code``. ``app``/``version`` identify the
    published app and executed version on the invoke route; on the editor
    route both are always null. On invoke, ``graph`` and ``app`` are the
    slug on EVERY outcome, and ``version`` is null on errors before an
    app version was resolved. Forward compatibility (also stated on the
    docs page): clients MUST ignore unknown envelope fields, and
    ``error.code`` is an open enum — treat unknown codes as generic
    errors. The ``job`` key is reserved by name for the future async mode.
    """

    status: Literal["ok", "error"]
    run_id: str
    graph: str
    app: str | None = None
    version: int | None = None
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
