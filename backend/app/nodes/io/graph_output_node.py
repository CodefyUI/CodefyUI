"""GraphOutput node — declares a named output of this graph.

Together with GraphInput this expresses the graph's transport-agnostic
function signature on the canvas. The run endpoint reads this node's
``value`` port from the engine result and returns it to API callers under
the declared name.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class GraphOutputNode(BaseNode):
    NODE_NAME = "GraphOutput"
    CATEGORY = "IO"
    DESCRIPTION = (
        "Declares a named output of this graph — returned to API callers by "
        "POST /api/graph/run under this node's 'name'. Connect the value you "
        "want the graph to return."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="value",
                data_type=DataType.ANY,
                description="The value to return under this output's name",
                optional=False,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="name",
                param_type=ParamType.STRING,
                default="output",
                description=(
                    "Contract name of this output. Must match "
                    "^[a-zA-Z_][a-zA-Z0-9_]{0,63}$ (letters, digits, underscore; "
                    "not starting with a digit)."
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
        # Pass-through: the run endpoint reads this port from the engine
        # result; returning the value also makes it visible in the
        # Inspector / edge tooltip on canvas runs for free.
        return {"value": inputs.get("value")}
