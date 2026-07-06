from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DatasetNode(BaseNode):
    NODE_NAME = "Dataset"
    CATEGORY = "Data"
    DESCRIPTION = "Load a standard dataset (MNIST, CIFAR10, or FashionMNIST)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Loaded dataset"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="name",
                param_type=ParamType.SELECT,
                default="MNIST",
                description="Dataset to load",
                options=["MNIST", "CIFAR10", "FashionMNIST"],
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.SELECT,
                default="train",
                description="Data split",
                options=["train", "test"],
            ),
            ParamDefinition(
                name="data_dir",
                param_type=ParamType.STRING,
                default="./data",
                description="Directory to download/store the dataset",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from torchvision import datasets, transforms

        name = params.get("name", "MNIST")
        split = params.get("split", "train")
        data_dir = params.get("data_dir", "./data")

        from pathlib import Path

        from ...config import settings

        if settings.PROJECT_DIR is not None and not Path(data_dir).is_absolute():
            # torchvision downloads land in the project, not the install CWD
            # (spec 7.2). kagglehub / HF caches stay machine-global.
            data_dir = str(settings.PROJECT_DIR / "assets" / "data")

        is_train = split == "train"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        dataset_map = {
            "MNIST": datasets.MNIST,
            "CIFAR10": datasets.CIFAR10,
            "FashionMNIST": datasets.FashionMNIST,
        }

        dataset_cls = dataset_map.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unsupported dataset: {name}")

        dataset = dataset_cls(
            root=data_dir,
            train=is_train,
            download=True,
            transform=transform,
        )

        return {"dataset": dataset}
