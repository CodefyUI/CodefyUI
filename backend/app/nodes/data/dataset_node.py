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

    # #144: cacheable -- cache_fingerprint() below folds a per-file
    # identity fingerprint of THIS dataset's own files into the cache key,
    # so a change on disk busts the cache instead of the key surviving
    # unchanged because only the directory name is hashed. This is the
    # headline case #144 names: a CIFAR10/MNIST-rooted graph no longer
    # re-parses the dataset every run. Scoped to the dataset's own
    # directory rather than all of `data_dir` since #259 -- see
    # `_owned_paths` for what that covers and why.
    cacheable = True

    @staticmethod
    def _resolve_data_dir(data_dir: str) -> str:
        from pathlib import Path

        from ...config import settings

        if settings.PROJECT_DIR is not None and not Path(data_dir).is_absolute():
            # torchvision downloads land in the project, not the install CWD
            # (spec 7.2). kagglehub / HF caches stay machine-global.
            return str(settings.PROJECT_DIR / "assets" / "data")
        return data_dir

    @classmethod
    def _owned_paths(cls, name: str, split: str, root: str) -> list[Any] | None:
        """Where *name*'s files live under *root*, or None if unknown (#259).

        Read off torchvision's OWN class attributes rather than a table of
        directory names copied into this file, so a rename upstream is a
        changed answer here rather than a fingerprint that quietly stops
        covering anything:

        * ``base_folder`` is a class attribute on ``CIFAR10``
          (``cifar-10-batches-py``), ``CIFAR100`` and ``STL10``;
        * the ``MNIST`` family (``MNIST``, ``FashionMNIST``) has no
          ``base_folder`` -- both ``raw_folder`` and ``processed_folder``
          are ``root/<class name>/...``, so the class name IS the
          directory;
        * ``SVHN`` puts one flat ``.mat`` per split directly in ``root``,
          named by ``split_list[split][1]``.

        Returns a list of paths (directories, files, or both). ``None``
        means "no idea", and the caller falls back to walking the whole
        tree -- over-invalidation costs a re-read, never a stale answer, so
        an unrecognised name degrades to exactly the old behaviour.

        Scoping is per DATASET, not per split, for the families whose two
        splits share one directory: torchvision's MNIST loader requires all
        four idx files to be present before it will read either split, so
        splitting finer would fingerprint less than the loader actually
        reads. ``SVHN`` is per split because it downloads only the split
        asked for. #259's point is that an UNRELATED tree stored alongside
        must stop busting this key, and either granularity achieves that.
        """
        from pathlib import Path

        from torchvision import datasets

        dataset_cls = getattr(datasets, name, None)
        if dataset_cls is None:
            return None
        root_path = Path(root)

        base_folder = getattr(dataset_cls, "base_folder", None)
        if isinstance(base_folder, str) and base_folder:
            return [root_path / base_folder]

        if isinstance(dataset_cls, type) and issubclass(dataset_cls, datasets.MNIST):
            return [root_path / dataset_cls.__name__]

        split_list = getattr(dataset_cls, "split_list", None)
        if isinstance(split_list, dict):
            entry = split_list.get("train" if split == "train" else "test")
            if entry:
                return [root_path / str(entry[1])]

        return None

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        # #259. This used to walk the whole of `data_dir` recursively, which
        # is safe (over-invalidation costs a re-read, never serves a stale
        # answer) and wrong in two directions: in project mode every
        # relative `data_dir` collapses to the same
        # `PROJECT_DIR/assets/data` (see `_resolve_data_dir`), so writing an
        # unrelated dataset -- or a model file, which is where this actually
        # bit -- busts this node's key, and a large shared tree pays a full
        # recursive stat every run whether or not THIS dataset changed. That
        # is squarely against #144's own performance goal.
        #
        # It also had a second, undocumented job: `ModelSaver` may only
        # write under `MODELS_DIR`, which in the default layout sits INSIDE
        # this tree, so the saver's write dirtied the dataset's key on every
        # run and the resulting miss propagated down to `TrainingLoop`. That
        # accident was the only thing keeping the shipped saver graphs off
        # #253, which is why #259 was blocked on it. #253 is fixed at the
        # source now (`TrainingLoop` and `SequentialModel` are non-cacheable),
        # so the accident is no longer load-bearing -- pinned by
        # `test_cache_dataset_fingerprint_scope.py`, which runs the shipped
        # graphs three times against one cache and counts real execute()
        # calls rather than trusting status.
        from ...core.cache_fingerprint import directory_fingerprint, paths_fingerprint

        data_dir = str(params.get("data_dir", "./data") or "./data")
        name = str(params.get("name", "MNIST") or "MNIST")
        split = str(params.get("split", "train") or "train")
        try:
            resolved = cls._resolve_data_dir(data_dir)
            owned = cls._owned_paths(name, split, resolved)
        except Exception:
            # Includes a torchvision import failure. A fingerprint answers
            # "did the input change", never "is the input valid"; the
            # node's own execute() raises the real error.
            return None
        if owned is None:
            return directory_fingerprint(resolved)
        if len(owned) == 1 and owned[0].is_dir():
            return directory_fingerprint(owned[0])
        # A file list, or a directory that does not exist yet: both are
        # fingerprinted by identity, and a path that starts or stops
        # existing between runs still changes the answer.
        return paths_fingerprint(owned)

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
        data_dir = self._resolve_data_dir(params.get("data_dir", "./data"))

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
