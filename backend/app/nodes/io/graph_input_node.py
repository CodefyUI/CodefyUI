"""GraphInput node — declares a named input of this graph.

Together with GraphOutput this expresses the graph's transport-agnostic
function signature on the canvas: the CLI runner and Stage 2 published
apps consume the same contract. API callers supply the value via
``POST /api/graph/run/{name}``; canvas runs fall back to the ``default``
param.

Implementation note (the "hidden" ``value`` param): ``value`` is hidden
ONLY in the sense of being UNDECLARED in ``define_params`` — do NOT
declare it. Declaring it would render a text field: the config panel and
node card iterate declared defs only. The API injects it; the canvas
never sets it, so canvas runs fall back to ``default``.
"""

from __future__ import annotations

from typing import Any

from ...core.api_contract import coerce_input
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


def _load_canvas_image(path_value: Any) -> Any:
    """Load the canvas-only ``default`` image path the way ImageReader does.

    Only reached on canvas runs — the API path always injects ``value``
    (a base64 string) and never touches ``default``. Relative paths
    resolve against IMAGES_DIR so filenames picked from the uploaded-files
    dropdown work without the user typing a full path.
    """
    from pathlib import Path

    from PIL import Image
    from torchvision import transforms

    from ...config import settings

    path_str = str(path_value or "")
    if not path_str:
        raise ValueError(
            "GraphInput(type=image): set 'default' to a server-local image path "
            "for canvas runs (API callers send base64 instead)"
        )
    p = Path(path_str)
    if not p.is_absolute():
        p = settings.IMAGES_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"GraphInput default image not found: {path_str}")
    img = Image.open(p).convert("RGB")
    return transforms.ToTensor()(img)


class GraphInputNode(BaseNode):
    NODE_NAME = "GraphInput"
    CATEGORY = "IO"
    DESCRIPTION = (
        "Declares a named input of this graph — supplied by API callers via "
        "POST /api/graph/run, or by the 'default' param when run on the canvas. "
        "Wire a Start node into this node so it executes."
    )

    # Params include the injected value, so cache keys stay correct on
    # canvas runs (BaseNode default, restated here because it is load-bearing).
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="value",
                data_type=DataType.ANY,
                description=(
                    "Value supplied by the API caller (or 'default' when run on canvas)"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="name",
                param_type=ParamType.STRING,
                default="input",
                description=(
                    "Contract name of this input. Must match "
                    "^[a-zA-Z_][a-zA-Z0-9_]{0,63}$ (letters, digits, underscore; "
                    "not starting with a digit)."
                ),
            ),
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="string",
                description="JSON type API callers must send for this input.",
                options=["string", "number", "integer", "boolean", "json", "image"],
            ),
            ParamDefinition(
                name="required",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "When false, API callers may omit this input and 'default' "
                    "applies. Image inputs must stay required."
                ),
            ),
            ParamDefinition(
                name="default",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "ALWAYS the canvas test value; an API-side fallback ONLY when "
                    "required=false. Parsed per 'type' (string parsing applies here "
                    "only). For type=image: a server-local file path used only by "
                    "canvas runs."
                ),
            ),
            ParamDefinition(
                name="description",
                param_type=ParamType.STRING,
                default="",
                description="Shown in the contract for self-documenting APIs.",
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
        declared_type = str(params.get("type", "string"))
        if "value" in params:
            # API path: routes_graph_run injected the RAW JSON value; the
            # coercion (including base64 image decode) happens exactly once,
            # here. One helper serves both paths, so canvas and API behave
            # identically.
            return {"value": coerce_input(params["value"], declared_type)}
        # Canvas path (or omitted optional API input): fall back to `default`.
        raw_default = params.get("default", "")
        if declared_type == "image":
            return {"value": _load_canvas_image(raw_default)}
        return {"value": coerce_input(raw_default, declared_type, from_string=True)}
