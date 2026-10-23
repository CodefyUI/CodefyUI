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
    DESCRIPTION = "Load your own images from one folder per class"
    DETAILS = (
        "Labels follow the folder names in alphabetical order, and the `classes` "
        "output lists them in that same order. `split` picks which sub-directory "
        "of `path` to read; choose '(none)' when the class folders sit directly "
        "under `path`. The class list is the sorted union of the class folders "
        "in every split folder under `path` (train, val, valid, validation, "
        "test), so a split missing a class still gives each class the index "
        "the training split uses; the run log names the missing folders. A "
        "wired `path` replaces the path param and orders this node after "
        "whatever produced it."
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
            if root is None:
                return None
            split = str(params.get("split", "train") or "train")
            # The class numbering also reads the sibling splits' folders.
            classes = tuple(_class_union(root.parent, split)) if split != "(none)" else ()
        except Exception:
            return None
        return (directory_fingerprint(root), classes)

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
            PortDefinition(
                name="path",
                data_type=DataType.STRING,
                description=(
                    "Directory holding the splits (overrides the path param "
                    "when wired), resolved the same way."
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

        # The wired path wins, as VideoLoad's does; the edge also orders
        # this node after the one that wrote the folder.
        raw_path = str(inputs.get("path") or params.get("path", "") or "").strip()
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

        result: dict[str, Any] = {}
        classes = class_dirs
        if split != "(none)":
            classes = _class_union(base, split)
            missing = [c for c in classes if c not in class_dirs]
            if missing:
                from ...core.advisories import emit_advisory

                result["__log__"] = emit_advisory(
                    f"ImageFolderDataset: '{root}' has no folder for class(es) "
                    f"{missing}, which other split folders under '{base}' have. "
                    f"The classes are numbered over every split ({classes}), so "
                    f"each class keeps the index the training split gives it.",
                    kind="image_folder_missing_class",
                    prefix="[WARN] ",
                    context=context,
                )

        from ._class_list_image_folder import ClassListImageFolder

        dataset = ClassListImageFolder(
            str(root), classes, transform=seeded_for_node(transform, context))

        _verify_sample_images(dataset.samples)

        result.update({"dataset": dataset, "classes": list(dataset.classes)})
        return result


def _class_dirs(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir() if entry.is_dir())


def _class_union(base: Path, split: str) -> list[str]:
    """Sorted union of the class folders in *split* and its sibling splits.

    A val or test folder often lacks a rare class. Numbering only the folders
    it has would shift every later class down by one, so a model trained
    with ``cherry = 2`` would be scored against ``cherry = 1`` with no
    error. Siblings are the directories of *base* named like a split
    (:data:`COMMON_SPLITS`); anything else under *base* is left alone.
    """
    names = set(_class_dirs(base / split))
    for entry in base.iterdir():
        if entry.is_dir() and entry.name != split and entry.name in COMMON_SPLITS:
            names.update(_class_dirs(entry))
    return sorted(names)


#: How many images :func:`_verify_sample_images` opens. Enough to catch a
#: truncated export or a folder of renamed files -- the two ways this goes
#: wrong in practice, both of which affect many files, not one -- while
#: staying a fixed cost that does not grow with the dataset.
VERIFY_SAMPLE_SIZE = 32


def _verify_sample_images(samples: list[tuple[str, int]]) -> None:
    """Open a spread sample of *samples*, raising on the first unreadable one.

    The node checks the directory tree carefully and, until #197, never
    checked a single file. ``ImageFolder`` accepts anything with a listed
    extension, so a truncated, zero-byte or merely renamed ``.png`` builds
    fine here and raises ``PIL.UnidentifiedImageError`` from inside a
    DataLoader WORKER, potentially minutes into training, naming nothing
    the user can act on -- the exact class of error the node's other
    messages go out of their way to avoid.

    An EVEN spread from the first file to the last, not the first N:
    ``samples`` is ordered by class, so the first N would only ever inspect
    the alphabetically first class and a dataset whose last class is the
    broken one would sail through. A sample rather than everything: full
    validation would read the whole dataset at build time, which is the cost
    this node exists to defer.

    ``verify()`` and not ``load()``: it reads structure (a PNG's chunk CRCs,
    a JPEG's markers) without decoding pixels, which is what makes 32 files
    free. It also invalidates the handle, hence the fresh ``open`` per file.
    """
    from PIL import Image

    if not samples:
        return
    count = min(VERIFY_SAMPLE_SIZE, len(samples))
    last = len(samples) - 1
    indices = sorted({i * last // max(1, count - 1) for i in range(count)})
    for path, _label in (samples[i] for i in indices):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001 - re-raised with the path
            raise ValueError(
                f"'{path}' is not a readable image ({type(exc).__name__}: "
                f"{exc}). ImageFolder takes any file with an image "
                f"extension, so a truncated download, a zero-byte file or "
                f"something renamed to .png gets this far and then fails "
                f"inside a data-loading worker mid-training. Remove or "
                f"re-export the file and run again."
            ) from exc


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
