from typing import Any

from ...core.node_base import BaseNode, DataType, PortDefinition


class SwitchNode(BaseNode):
    NODE_NAME = "Switch"
    CATEGORY = "Data Flow"
    DESCRIPTION = "Forwards one of up to four inputs by selector index"
    DETAILS = (
        "Every input is evaluated before the selector picks one; this is data "
        "flow, not a branch that skips work. The index is 0-based, and one that is "
        "out of range or not connected falls back to input_0."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="selector", data_type=DataType.SCALAR, description="Index of input to select (0-based)"),
            PortDefinition(name="input_0", data_type=DataType.ANY, description="Option 0 (default)"),
            PortDefinition(name="input_1", data_type=DataType.ANY, description="Option 1", optional=True),
            PortDefinition(name="input_2", data_type=DataType.ANY, description="Option 2", optional=True),
            PortDefinition(name="input_3", data_type=DataType.ANY, description="Option 3", optional=True),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.ANY, description="The selected input value"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        selector = inputs.get("selector", 0)
        if hasattr(selector, "item"):
            selector = selector.item()
        idx = int(selector)

        key = f"input_{idx}"
        if key not in inputs or inputs[key] is None:
            # Fall back to input_0 if selected index is not connected
            key = "input_0"

        return {"output": inputs.get(key)}
