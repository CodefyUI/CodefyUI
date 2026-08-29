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
    # Package Center — see ParamDefinition.option_packs. Maps a SELECT
    # option to the optional pack it needs, so the editor can grey out
    # exactly those options instead of disabling the whole parameter.
    option_packs: dict[str, str] | None = None


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
    # Package Center — see BaseNode.REQUIRES_PACK. The optional pack this
    # node needs before it can run at all; None for everything the base
    # install already covers.
    requires_pack: str | None = None


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


class SubgraphPortData(BaseModel):
    """One boundary port of a subgraph (core#137).

    ``port`` is the handle name the INSTANCE node exposes on the canvas;
    ``innerNode``/``innerPort`` say which node inside the definition it
    stands for. Expansion rewrites every edge touching ``port`` into an
    edge touching the inner port, which is what makes a collapsed graph
    execute identically to the graph it was collapsed from.
    """
    port: str
    innerNode: str
    innerPort: str
    #: Declared for the canvas so the instance node can colour the handle.
    #: Never consulted by expansion — the inner node's own port definition
    #: is what validation type-checks against.
    data_type: str = ""


class SubgraphInterfaceData(BaseModel):
    """The boundary of a subgraph: what an instance node exposes."""
    inputs: list[SubgraphPortData] = []
    outputs: list[SubgraphPortData] = []
    #: Inner node ids an incoming TRIGGER edge fans out to. Collapse records
    #: exactly the inner nodes that were triggered before it ran, so the
    #: expanded graph has the same entry points as the original. Empty means
    #: "fan out to the inner roots" (see ``expand_subgraphs``).
    triggerTargets: list[str] = []


class SubgraphDefinition(BaseModel):
    """A reusable block of graph, referenced by ``subgraph:<id>`` nodes.

    Local to one graph file (v1): definitions travel in the graph's own
    ``subgraphs`` list rather than a server-side registry, so a graph is
    still one portable artifact. Every instance node with the same ``id``
    shares this one definition -- editing it changes all of them.
    """
    id: str
    name: str = ""
    description: str = ""
    nodes: list[NodeData] = []
    edges: list[EdgeData] = []
    interface: SubgraphInterfaceData = Field(
        default_factory=SubgraphInterfaceData
    )


class GraphData(BaseModel):
    nodes: list[NodeData]
    edges: list[EdgeData]
    name: str = "Untitled"
    description: str = ""
    presets: list[PresetDefinition] = []
    # Persisted so the Teaching Inspector's segment overlays survive a
    # save/load round-trip. Optional; older graph files simply omit it.
    segmentGroups: list[SegmentGroupData] = []
    # Subgraph definitions (core#137). Optional; older graph files omit it.
    subgraphs: list[SubgraphDefinition] = []


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
