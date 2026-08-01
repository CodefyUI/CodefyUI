"""KaggleDataset node — download and load a Kaggle dataset as an ImageFolder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


def _kaggle_credentials_present() -> bool:
    """True if either env-var creds or ~/.kaggle/kaggle.json is present."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return False


class KaggleDatasetNode(BaseNode):
    NODE_NAME = "KaggleDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Download a Kaggle dataset and load it as an ImageFolder"

    # Hits the network, the kagglehub cache on disk, and KAGGLE_* credentials
    # in the environment. None of that is visible to the cache key, so a hit
    # would pin the graph to whatever the first run happened to download.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset loaded as torchvision ImageFolder",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dataset_slug",
                param_type=ParamType.STRING,
                default="",
                description="Kaggle dataset slug in 'owner/name' form",
            ),
            ParamDefinition(
                name="subdir",
                param_type=ParamType.STRING,
                default="",
                description="Relative path inside the downloaded dataset where ImageFolder structure begins",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override kagglehub cache directory (empty = default)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        try:
            import kagglehub
        except ImportError as e:
            raise RuntimeError(
                "KaggleDataset requires the 'kagglehub' package. "
                "Install with: pip install kagglehub "
                "(or `uv sync --extra ml` from backend/)"
            ) from e

        if not _kaggle_credentials_present():
            raise RuntimeError(
                "Kaggle authentication required. Set KAGGLE_USERNAME and "
                "KAGGLE_KEY environment variables, or place kaggle.json at "
                "~/.kaggle/kaggle.json. See https://www.kaggle.com/docs/api"
            )

        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder

        dataset_slug = params.get("dataset_slug", "")
        subdir = params.get("subdir", "")
        cache_dir = params.get("cache_dir", "")

        previous_cache = os.environ.get("KAGGLEHUB_CACHE")
        try:
            if cache_dir:
                os.environ["KAGGLEHUB_CACHE"] = cache_dir

            download_path = kagglehub.dataset_download(dataset_slug)
        finally:
            if previous_cache is None:
                os.environ.pop("KAGGLEHUB_CACHE", None)
            else:
                os.environ["KAGGLEHUB_CACHE"] = previous_cache
        root = os.path.join(download_path, subdir) if subdir else download_path

        class_dirs = [
            entry for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        ]
        if not class_dirs:
            found = sorted(os.listdir(root))[:10]
            raise RuntimeError(
                f"Path '{root}' does not contain class subdirectories. "
                f"Set 'subdir' to point at the folder that holds your "
                f"per-class subfolders. Found at this path: {found}"
            )

        return {"dataset": ImageFolder(root, transform=T.ToTensor())}
