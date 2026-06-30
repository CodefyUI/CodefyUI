from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DatasetBatchNode(BaseNode):
    NODE_NAME = "DatasetBatch"
    CATEGORY = "Data"
    DESCRIPTION = (
        "從資料集（如 Dataset 節點載入的 MNIST）取出一個批次，"
        "輸出影像張量 (N, C, H, W) 與對應標籤。"
        "用來把資料集直接餵進手搭的網路做一次前向傳遞、觀察每層 shape。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="來源資料集（例如 Dataset 節點輸出的 MNIST）。",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="images",
                data_type=DataType.TENSOR,
                description="影像批次張量，形狀 (N, C, H, W)。",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.TENSOR,
                description="對應的標籤：分類資料集是 (N,) 類別；分割資料集是 (N, H, W) 的逐像素遮罩。",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=1,
                description="一次取幾筆樣本（決定輸出的 N）。",
                min_value=1,
            ),
            ParamDefinition(
                name="start_index",
                param_type=ParamType.INT,
                default=0,
                description="從資料集的第幾筆開始取（方便換不同樣本觀察）。",
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
