"""ExampleNode — a starting point. Replace it with your plugin's real node.

Demonstrates the BaseNode contract: one STRING param, one STRING output, and a
pure-Python ``execute`` (no torch). Add inputs / params / a real computation as
your plugin needs; the node shows up in the editor palette under ``CATEGORY``.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class ExampleNode(BaseNode):
    NODE_NAME = "Example"
    CATEGORY = "{{plugin_name}}"
    DESCRIPTION = "Greets the name in the `name` param"
    DETAILS = (
        "The `greeting` output is `Hello, <name>!`, and a blank `name` falls back "
        "to `world`. DESCRIPTION is the one-line summary the palette row shows, "
        "cut off past 56 characters. DETAILS is this longer text, which only the "
        "config panel and the Docs tab show; leave it empty when the summary "
        "already says it all."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="greeting",
                data_type=DataType.STRING,
                description="The composed greeting string.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="name",
                param_type=ParamType.STRING,
                default="world",
                description="Who to greet.",
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
        name = str(params.get("name", "world")).strip() or "world"
        return {"greeting": f"Hello, {name}!"}
