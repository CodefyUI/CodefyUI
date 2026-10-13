from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DatasetBatchNode(BaseNode):
    NODE_NAME = "DatasetBatch"
    CATEGORY = "Data"
    DESCRIPTION = "Take one batch from a dataset: images and their labels"
    DETAILS = (
        "Images come out as (N, C, H, W). Labels are (N,) class ids for a "
        "classification dataset, or (N, H, W) per-pixel masks for a segmentation "
        "one. `start_index` wraps around the end of the dataset."
    )

    # Reading a batch draws from the dataset's seeded augmentation stream, a
    # side effect its output does not carry (case 3 of ``BaseNode.cacheable``).
    # A cache hit would skip those draws, so every later reader of that
    # dataset would draw from a different place than on a fresh run (#603).
    # The cost: ``batch_size`` samples re-read per Run, and whatever this
    # feeds re-runs too, because opting out propagates downstream.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Source dataset (for example MNIST from a Dataset node).",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="images",
                data_type=DataType.TENSOR,
                description="Image batch tensor of shape (N, C, H, W).",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.TENSOR,
                description=(
                    "Matching labels: (N,) class ids for a classification dataset, "
                    "(N, H, W) per-pixel masks for a segmentation dataset."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=1,
                description="Number of samples to take (the N of the outputs).",
                min_value=1,
            ),
            ParamDefinition(
                name="start_index",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Index of the first sample to take. Change it to look at other "
                    "samples. A batch that runs past the last sample wraps to the "
                    "start: start 8, batch 4 on 10 samples takes 8, 9, 0, 1."
                ),
                min_value=0,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        dataset = inputs["dataset"]
        batch_size = int(params.get("batch_size", 1))
        start = int(params.get("start_index", 0))

        n = len(dataset)
        if n == 0:
            raise ValueError("Dataset is empty.")

        images: list[Any] = []
        labels: list[Any] = []
        label_is_tensor = False
        for offset in range(batch_size):
            sample, label = dataset[(start + offset) % n]
            if not torch.is_tensor(sample):
                sample = torch.as_tensor(sample)
            images.append(sample)
            # Scalar label (classification) -> int; tensor label with spatial
            # dims (e.g. a segmentation mask (H, W)) -> keep as a tensor.
            if torch.is_tensor(label) and label.ndim > 0:
                label_is_tensor = True
                labels.append(label)
            else:
                labels.append(int(label))

        images_tensor = torch.stack(images, dim=0)
        if label_is_tensor:
            labels_tensor = torch.stack(labels, dim=0)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {"images": images_tensor, "labels": labels_tensor}
