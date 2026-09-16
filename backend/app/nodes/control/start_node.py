"""Start node -- marks an execution entry point.

A `StartNode` has no inputs and one `trigger` output. It does no work at
runtime; its only purpose is to declare that the connected component
containing it is "live" and should be executed. Connect the trigger output
to a data-root node (e.g. Dataset) to mark that node as an entry point.
"""

from typing import Any

from app.core.node_base import BaseNode, DataType, PortDefinition


class StartNode(BaseNode):
    NODE_NAME = "Start"
    CATEGORY = "Control"
    DESCRIPTION = "Entry point; the node its trigger reaches runs first"
    DETAILS = (
        "It is the 'When Flag Clicked' block of a CodefyUI graph: the node does no "
        "work at runtime and passes no data. The trigger link tells the engine "
        "which part of the graph to execute, so it usually goes to a data root "
        "such as a Dataset node."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="trigger",
                data_type=DataType.TRIGGER,
                description="Execution trigger marker (carries no data)",
            ),
        ]

    @classmethod
    def define_params(cls) -> list:
        return []

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        return {}
