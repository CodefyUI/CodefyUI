"""ComposeTransform — merge several chains into one, in port order."""

from __future__ import annotations

from typing import Any

from ....core.node_base import (
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ._base import CHAIN_OUTPUT_DESC, TransformStepNode, compose

#: Bounds on the ``steps`` param. Two is the smallest count that composes
#: anything; eight is past any chain a canvas stays readable at.
MIN_STEPS = 2
MAX_STEPS = 8
DEFAULT_STEPS = 2


def _step_count(params: dict[str, Any] | None) -> int:
    """How many input ports this instance has. Clamps rather than raises.

    Called by the palette, the validator and the renderer with whatever the
    graph JSON happens to hold, which after a hand edit is not necessarily
    an integer at all.
    """
    try:
        count = int((params or {}).get("steps", DEFAULT_STEPS))
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    return max(MIN_STEPS, min(MAX_STEPS, count))


class ComposeTransformNode(TransformStepNode):
    NODE_NAME = "ComposeTransform"
    DESCRIPTION = (
        "Join several transform chains into one, in port order: step_1 runs "
        "first. Chaining node to node already composes, so reach for this "
        "when two chains were built separately and one pipeline has to run "
        "both."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return cls.define_inputs_dynamic({"steps": DEFAULT_STEPS})

    @classmethod
    def define_inputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        return [
            PortDefinition(
                name=f"step_{index}",
                data_type=DataType.TRANSFORM,
                description=f"Chain to run at position {index}",
                # Every port is optional so a gap is a gap and not a run
                # failure: wiring 1 and 3 of a 3-port node composes those
                # two in that order.
                optional=True,
            )
            for index in range(1, _step_count(params) + 1)
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="transform",
                data_type=DataType.TRANSFORM,
                description=CHAIN_OUTPUT_DESC,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="steps",
                param_type=ParamType.INT,
                default=DEFAULT_STEPS,
                description="How many chains to join",
                min_value=MIN_STEPS,
                max_value=MAX_STEPS,
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
        ordered = [
            inputs.get(f"step_{index}")
            for index in range(1, _step_count(params) + 1)
        ]
        return {"transform": compose(*ordered)}
