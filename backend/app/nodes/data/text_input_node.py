"""TextInput node — free-text entry point for LLM workflows.

Mirrors :class:`TensorInputNode`'s role for tensors: a no-input-port source
that emits a STRING value typed into an inline textarea on the node body.
Useful for tokenizer comparisons (one TextInput → multiple Tokenizers with
different families) and as the seed for future RAG pipelines.
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


class TextInputNode(BaseNode):
    NODE_NAME = "TextInput"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Free-text entry point. Type a string into the inline textarea and "
        "feed it into a Tokenizer, WordVector, or any other STRING-typed "
        "input port."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="The text value typed into this node.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="value",
                param_type=ParamType.STRING,
                default="The quick brown fox jumps over the lazy dog.",
                description=(
                    "Multi-line text. Drag the bottom-right corner of the "
                    "textarea on the node body to resize."
                ),
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
        return {"text": str(params.get("value", ""))}
