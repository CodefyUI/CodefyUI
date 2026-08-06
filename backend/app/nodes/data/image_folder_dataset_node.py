"""ImageFolderDataset — your own images, one folder per class.

The bridge between the built-in academic datasets and a real project. Point
it at a directory laid out as

    <path>/<split>/<class name>/<image files>

and torchvision infers the classes from the folder names, sorted, so the
label of ``cat`` is stable across machines and runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from .transforms._base import (
    compose,
    seeded_for_node,
    select_split_transform,
)

#: Names people actually use for the evaluation folder. Only consulted to
#: make the "did you mean" hint on a missing split useful.
COMMON_SPLITS = ("train", "val", "valid", "validation", "test")


def resolve_dataset_root(path: str) -> Path:
    """Absolute directory for *path*.

    A relative path is taken as relative to the data root -- the directory
    holding ``models/`` and ``images/``, which in project mode is the
    project's ``assets/``. That is where uploads land, so ``my-dataset``
    means the obvious thing without anyone typing an absolute path.

    An absolute path is honoured as given. Reading a directory the user
    named is not an escalation: ``Dataset``'s ``data_dir`` and
    ``KaggleDataset``'s cache already do exactly that, and CodefyUI runs as
    the user, on the user's machine. Contrast ``resolve_checkpoint_path``,
    which is confined to the data root because it WRITES.

    ``~`` is expanded first. Without that it is not special at all, so
    ``~/datasets/x`` resolved to a literal ``<data root>/~/datasets/x`` --
    a directory that does not exist, reported as a missing path with no hint
    that the tilde was the problem.
    """
    from ...config import settings

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = settings.MODELS_DIR.parent / candidate
    return candidate.resolve()


class ImageFolderDatasetNode(BaseNode):
    NODE_NAME = "ImageFolderDataset"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Load your own images from one folder per class. Labels come from "
        "the folder names, sorted alphabetically."
    )

    # #144: cacheable again -- cache_fingerprint() below folds an aggregate
    # fingerprint (file count, total size, latest mtime) of the resolved
    # split directory's contents into the cache key, so files changing
    # under it busts the cache instead of the key surviving unchanged
    # because only `path`/`split` are hashed.
    cacheable = True

    @classmethod
    def _resolve_root(cls, params: dict[str, Any]) -> Path | None:
        raw_path = str(params.get("path", "") or "").strip()
        if not raw_path:
            return None
        split = str(params.get("split", "train") or "train")
        base = resolve_dataset_root(raw_path)
        return base if split == "(none)" else base / split

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import directory_fingerprint

        try:
            root = cls._resolve_root(params)
        except Exception:
            return None
        if root is None:
            return None
        return directory_fingerprint(root)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="train_transform",
                data_type=DataType.TRANSFORM,
                description=(
                    "Pipeline for the training split, and the one to use "
                    "when split is '(none)'. This is where augmentation "
                    "belongs. Ignored on val/test, with a warning."
                ),
                optional=True,
            ),
            PortDefinition(
                name="eval_transform",
                data_type=DataType.TRANSFORM,
                description=(
                    "Pipeline for val/test, and the fallback whenever "
                    "train_transform is unwired."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset loaded as a torchvision ImageFolder",
            ),
            PortDefinition(
                name="classes",
                data_type=DataType.LIST,
                description=(
                    "Class names in label order, so label 0 is classes[0]"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Directory holding the splits. Relative paths resolve "
                    "against the data folder that also holds models/ and "
                    "images/."
                ),
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.SELECT,
                default="train",
                description=(
                    "Sub-directory of path to load. Pick '(none)' when the "
                    "class folders sit directly under path with no split "
                    "level; there is then no split to tell the two "
                    "transform ports apart, so whichever is wired is used "
                    "and train_transform wins if both are."
                ),
                options=["train", "val", "test", "(none)"],
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
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        raw_path = str(params.get("path", "") or "").strip()
        if not raw_path:
            raise ValueError(
                "ImageFolderDataset needs a path. Give the directory that "
                "holds your class folders (or the splits containing them).")

        split = str(params.get("split", "train") or "train")
        base = resolve_dataset_root(raw_path)
        root = base if split == "(none)" else base / split

        if not root.is_dir():
            raise ValueError(_missing_root_message(base, root, split))

        class_dirs = sorted(
            entry.name for entry in root.iterdir() if entry.is_dir())
        if not class_dirs:
            raise ValueError(
                f"'{root}' has no sub-directories, so there are no classes "
                f"to read. ImageFolder expects one folder per class; if the "
                f"images sit directly in this folder, they need to be moved "
                f"into a folder named after their class.")

        wired = select_split_transform(
            inputs, split, node_name="ImageFolderDataset")

        transform = wired if wired is not None else compose([
            transforms.ToTensor(),
        ])

        dataset = ImageFolder(str(root),
                              transform=seeded_for_node(transform, context))

        return {"dataset": dataset, "classes": list(dataset.classes)}


def _missing_root_message(base: Path, root: Path, split: str) -> str:
    """Say which of the two levels is wrong, not merely that one is."""
    if not base.is_dir():
        return (
            f"'{base}' is not a directory. Check the path parameter; a "
            f"relative path resolves against the data folder next to "
            f"models/ and images/.")
    present = sorted(entry.name for entry in base.iterdir() if entry.is_dir())
    looks_like_splits = [name for name in present if name in COMMON_SPLITS]
    hint = (
        f" It does contain {looks_like_splits}."
        if looks_like_splits
        else f" Sub-directories of '{base}': {present[:10]}. If those are "
             f"your classes rather than splits, set split to '(none)'."
    )
    return f"'{base}' has no '{split}' sub-directory.{hint}"
