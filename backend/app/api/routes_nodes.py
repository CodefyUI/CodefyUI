from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..core.node_registry import (
    node_definition,
    registry,
)
from ..core.plugin_validator import PluginValidationError
from ..core.script_policy import (
    MAX_SCRIPT_CHARS,
    TIER0_MODULES,
    defines_entry_point,
    validate_script_source,
)
from ..schemas import NodeDefinition

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


class ScriptValidationRequest(BaseModel):
    """One in-canvas script, as typed."""

    code: str = Field(default="", max_length=MAX_SCRIPT_CHARS * 2)


class ScriptValidationResult(BaseModel):
    """Whether the script may run, and why not.

    ``ok=False`` is a normal answer, not an HTTP error: the editor asks this
    while the user is still typing, and half-written code is the expected
    case rather than a failure of the request.
    """

    ok: bool
    error: str | None = None
    #: 1-based line the rejection points at, when the gate knows it.
    line: int | None = None
    #: Whether a module-level ``def run(...)`` exists yet.
    defines_run: bool = False
    #: The Tier-0 import allowlist, so the editor can show it without
    #: hard-coding a list that would drift from the server's.
    allowed_modules: list[str] = []


@router.post("/script/validate", response_model=ScriptValidationResult)
async def validate_script(payload: ScriptValidationRequest) -> ScriptValidationResult:
    """Check one PythonScript body against the Tier-0 policy (core#131).

    The node runs this same check before every execution -- this endpoint
    exists so the answer arrives while the code is being written instead of
    when the graph is run.
    """
    allowed = list(TIER0_MODULES)
    try:
        validate_script_source(payload.code)
    except PluginValidationError as exc:
        return ScriptValidationResult(
            ok=False,
            error=str(exc),
            line=exc.lineno,
            defines_run=defines_entry_point(payload.code),
            allowed_modules=allowed,
        )
    return ScriptValidationResult(
        ok=True,
        defines_run=defines_entry_point(payload.code),
        allowed_modules=allowed,
    )


@router.get("", response_model=list[NodeDefinition])
async def list_nodes():
    return [node_definition(name, cls) for name, cls in registry.nodes.items()]


@router.get("/{node_name}", response_model=NodeDefinition)
async def get_node(node_name: str):
    cls = registry.get(node_name)
    if not cls:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Node '{node_name}' not found")
    # If caller passed a bare name and got a fallback match, surface the
    # qualified form back so the frontend can show it.
    qualified = node_name
    if node_name not in registry._nodes:
        for k, v in registry._nodes.items():
            if v is cls:
                qualified = k
                break
    return node_definition(qualified, cls)
