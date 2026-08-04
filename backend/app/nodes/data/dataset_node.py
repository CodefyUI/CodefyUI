from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from .transforms._base import (
    compose,
    seeded_for_node,
    select_split_transform,
)

#: Datasets whose constructor takes ``split="train"|"test"|...`` instead of
#: ``train=True|False``. torchvision is not consistent about this and the
#: difference is a TypeError, not a wrong result, so it is enumerated rather
#: than probed.
SPLIT_KWARG_DATASETS = frozenset({"SVHN", "STL10"})

DATASET_NAMES = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "STL10",
]


class DatasetNode(BaseNode):
    NODE_NAME = "Dataset"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Load a standard vision dataset. Wire a transform chain into "
        "train_transform / eval_transform to control preprocessing and "
        "augmentation; without one it applies ToTensor and Normalize(0.5)."
    )

    # Reads (and on first use downloads) the dataset under `data_dir`. The
    # cache key hashes the directory name, not its contents, so a cached
    # dataset would survive the files there changing.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="train_transform",
                data_type=DataType.TRANSFORM,
                description=(
                    "Pipeline for the training split -- this is where "
                    "augmentation belongs. Ignored when split is 'test', "
                    "with a warning in the log."
                ),
                optional=True,
            ),
            PortDefinition(
                name="eval_transform",
                data_type=DataType.TRANSFORM,
                description=(
                    "Pipeline for the test split, and the fallback for the "
                    "training split when train_transform is unwired. Keep "
                    "it free of randomness."
                ),
                optional=True,
            ),
        ]

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
                options=DATASET_NAMES,
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

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
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

        # One rule, shared with ImageFolderDataset, which also warns when a
        # wired train_transform is about to be dropped for a test split.
        wired = select_split_transform(inputs, split, node_name="Dataset")

        transform = wired if wired is not None else compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        dataset_map = {
            "MNIST": datasets.MNIST,
            "CIFAR10": datasets.CIFAR10,
            "CIFAR100": datasets.CIFAR100,
            "FashionMNIST": datasets.FashionMNIST,
            "SVHN": datasets.SVHN,
            "STL10": datasets.STL10,
        }

        dataset_cls = dataset_map.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unsupported dataset: {name}")

        kwargs: dict[str, Any] = {
            "root": data_dir,
            "download": True,
            # Through ``seeded_for_node`` rather than raw: an augmenting
            # chain gets the reproducibility wrapper here, which is the one
            # place that decision is made for a dataset this node builds.
            "transform": seeded_for_node(transform, context),
        }
        if name in SPLIT_KWARG_DATASETS:
            kwargs["split"] = "train" if is_train else "test"
        else:
            kwargs["train"] = is_train

        return {"dataset": dataset_cls(**kwargs)}
