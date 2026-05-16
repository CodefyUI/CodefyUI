"""HuggingFaceDataset node — load image-classification datasets from HF Hub."""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class HuggingFaceDatasetNode(BaseNode):
    NODE_NAME = "HuggingFaceDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Load an image-classification dataset from HuggingFace Hub"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset wrapped as a torch.utils.data.Dataset",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dataset_name",
                param_type=ParamType.STRING,
                default="ylecun/mnist",
                description="HuggingFace Hub repo id (e.g. cifar10, uoft-cs/cifar100)",
            ),
            ParamDefinition(
                name="subset",
                param_type=ParamType.STRING,
                default="",
                description="Config name for multi-config datasets (empty = no subset)",
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.STRING,
                default="train",
                description="Split: train/test/validation, or HF slice syntax (train[:1000])",
            ),
            ParamDefinition(
                name="image_column",
                param_type=ParamType.STRING,
                default="image",
                description="Column name for the image feature",
            ),
            ParamDefinition(
                name="label_column",
                param_type=ParamType.STRING,
                default="label",
                description="Column name for the integer label",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override HF cache directory (empty = HF default)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "HuggingFaceDataset requires the 'datasets' package. "
                "Install with: pip install datasets "
                "(or `uv sync --extra ml` from backend/)"
            ) from e

        from torchvision import transforms as T

        from ._hf_adapter import HFTorchImageDataset

        dataset_name = params.get("dataset_name", "ylecun/mnist")
        subset = params.get("subset", "") or None
        split = params.get("split", "train") or "train"
        image_column = params.get("image_column", "image")
        label_column = params.get("label_column", "label")
        cache_dir = params.get("cache_dir", "") or None

        try:
            # trust_remote_code=False blocks the HF dataset-script RCE class:
            # without this, ``datasets`` < 2.16 silently executes a script.py
            # bundled in the remote dataset repo on first load. We err on the
            # side of refusing rather than silently running attacker code,
            # even though newer versions default to False.
            ds = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
        except Exception as e:
            err_name = type(e).__name__
            err_msg = str(e)
            looks_like_auth = (
                "GatedRepoError" in err_name
                or "RepositoryNotFoundError" in err_name
                or "401" in err_msg
                or "unauthorized" in err_msg.lower()
            )
            if looks_like_auth:
                raise RuntimeError(
                    "HuggingFace authentication required to load "
                    f"'{dataset_name}'. Set the HF_TOKEN environment "
                    "variable to a token with read access. "
                    "See https://huggingface.co/docs/hub/security-tokens"
                ) from e
            raise

        available = list(ds.features.keys()) if hasattr(ds, "features") else None
        if available is not None:
            if image_column not in available:
                raise RuntimeError(
                    f"Column '{image_column}' not found in dataset "
                    f"'{dataset_name}'. Available columns: {available}"
                )
            if label_column not in available:
                raise RuntimeError(
                    f"Column '{label_column}' not found in dataset "
                    f"'{dataset_name}'. Available columns: {available}"
                )

        wrapped = HFTorchImageDataset(
            ds,
            image_column=image_column,
            label_column=label_column,
            transform=T.ToTensor(),
        )
        return {"dataset": wrapped}
