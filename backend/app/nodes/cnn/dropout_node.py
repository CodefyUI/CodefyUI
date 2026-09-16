from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DropoutNode(BaseNode):
    NODE_NAME = "Dropout"
    CATEGORY = "CNN"
    DESCRIPTION = "Zeros each element at random, scaling the rest up"
    DETAILS = (
        "Backed by nn.Dropout, which scales the surviving elements by $1/(1-p)$ so "
        "the mean is preserved. As a standalone node it is rebuilt in training "
        "mode on every run, so it drops even during evaluation; inside "
        "SequentialModel it follows the model's train/eval state."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Tensor with dropout applied"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="p",
                param_type=ParamType.FLOAT,
                default=0.5,
                description="Probability of an element to be zeroed",
                min_value=0.0,
                max_value=1.0,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        p = params.get("p", 0.5)

        dropout = nn.Dropout(p=p)
        output = dropout(tensor)
        return {"tensor": output}
