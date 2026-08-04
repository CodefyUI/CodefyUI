"""Transform — apply a pipeline to any dataset.

Two ways in, and the wired one always wins:

* wire a chain built from the transform nodes into ``transform``, which is
  the composable route and the only one that can express augmentation;
* leave it unwired and use the three params, which is the shape this node
  had before core#136 and which every saved graph still relies on.

``Dataset`` and ``ImageFolderDataset`` take their own ``train_transform`` /
``eval_transform`` inputs, so this node is for the datasets that do not --
``HuggingFaceDataset``, ``KaggleDataset``, ``CSVReader`` and anything a
custom node produces.
"""

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from .transforms._base import attach_transform, compose


class TransformNode(BaseNode):
    NODE_NAME = "Transform"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Apply a transform pipeline to a dataset. Wire a chain into "
        "'transform' for anything beyond the three built-in steps; the "
        "params below are ignored when it is wired."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset to apply transforms to"),
            PortDefinition(
                name="transform",
                data_type=DataType.TRANSFORM,
                description=(
                    "Pipeline to install. Replaces the params below when "
                    "wired."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset with updated transforms"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="resize", param_type=ParamType.INT, default=0, description="Resize dimension (0 means no resize). Ignored when a transform chain is wired.", min_value=0),
            ParamDefinition(name="normalize", param_type=ParamType.BOOL, default=True, description="Apply normalization (mean=0.5, std=0.5). Ignored when a transform chain is wired; NormalizeTransform has the dataset presets."),
            ParamDefinition(name="to_tensor", param_type=ParamType.BOOL, default=True, description="Convert PIL images to tensors. Ignored when a transform chain is wired."),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        from torchvision import transforms

        dataset = inputs["dataset"]
        wired = inputs.get("transform")
        if wired is not None:
            return {"dataset": attach_transform(dataset, wired, context)}

        resize = params.get("resize", 0)
        normalize = params.get("normalize", True)
        to_tensor = params.get("to_tensor", True)

        transform_list = []

        if resize > 0:
            transform_list.append(transforms.Resize((resize, resize)))

        if to_tensor:
            transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))

        if transform_list:
            # Through ``attach_transform`` like the wired branch, so the two
            # paths agree on everything downstream of "which steps". Nothing
            # here is random, so the seeding wrapper is never installed.
            attach_transform(dataset, compose(transform_list), context)

        return {"dataset": dataset}
