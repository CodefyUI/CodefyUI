"""ImageFolderDataset (core#136) over a fixture directory tree.

What is actually worth asserting about a thin wrapper around torchvision's
``ImageFolder`` is the part torchvision does NOT do: resolving the path,
picking the split sub-directory, choosing between the two transform inputs,
and failing with a message that says which of the two directory levels is
wrong. A "path not found" on a two-level layout is the difference between a
five-second fix and a puzzled bug report.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from app.core.execution_context import ExecutionContext
from app.core.node_base import DataType
from app.nodes.data.image_folder_dataset_node import (
    ImageFolderDatasetNode,
    resolve_dataset_root,
)
from app.nodes.data.transforms._base import SeededAugmentation
from app.nodes.data.transforms.random_horizontal_flip_node import (
    RandomHorizontalFlipNode,
)
from app.nodes.data.transforms.resize_transform_node import ResizeTransformNode
from app.nodes.data.transforms.to_tensor_transform_node import (
    ToTensorTransformNode,
)


def _write_images(directory: Path, count: int, size: int = 8) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (size, size), (index * 20 % 256, 40, 60)).save(
            directory / f"{index}.png")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """``<root>/{train,val}/{cat,dog}/*.png`` with different counts per class."""
    root = tmp_path / "pets"
    _write_images(root / "train" / "cat", 3)
    _write_images(root / "train" / "dog", 2)
    _write_images(root / "val" / "cat", 1)
    _write_images(root / "val" / "dog", 1)
    return root


def _run(params, inputs=None, context=None):
    return ImageFolderDatasetNode().execute(
        inputs or {}, params, context=context)


# ── metadata ─────────────────────────────────────────────────────────────


def test_node_metadata():
    assert ImageFolderDatasetNode.NODE_NAME == "ImageFolderDataset"
    assert ImageFolderDatasetNode.CATEGORY == "Data"
    # #144: cacheable again -- cache_fingerprint() folds a directory content
    # fingerprint into the cache key instead. See test_cache_content_fingerprint.py.
    assert ImageFolderDatasetNode.cacheable is True
    outputs = ImageFolderDatasetNode.define_outputs()
    assert [p.name for p in outputs] == ["dataset", "classes"]
    inputs = ImageFolderDatasetNode.define_inputs()
    assert [p.name for p in inputs] == ["train_transform", "eval_transform"]
    assert all(p.data_type is DataType.TRANSFORM for p in inputs)
    assert all(p.optional for p in inputs)


# ── classes and splits ───────────────────────────────────────────────────


def test_classes_come_from_the_folder_names_in_sorted_order(tree: Path):
    result = _run({"path": str(tree), "split": "train"})
    assert result["classes"] == ["cat", "dog"]
    assert result["dataset"].classes == ["cat", "dog"]


def test_each_split_sees_only_its_own_images(tree: Path):
    train = _run({"path": str(tree), "split": "train"})["dataset"]
    val = _run({"path": str(tree), "split": "val"})["dataset"]
    assert len(train) == 5
    assert len(val) == 2


def test_labels_line_up_with_the_classes_list(tree: Path):
    result = _run({"path": str(tree), "split": "train"})
    dataset, classes = result["dataset"], result["classes"]
    labels = sorted(label for _, label in dataset.samples)
    # 3 cats (label 0) then 2 dogs (label 1), i.e. classes[label] is the
    # folder the file came from.
    assert labels == [0, 0, 0, 1, 1]
    assert classes[0] == "cat" and classes[1] == "dog"


def test_none_split_reads_the_classes_directly_under_path(tree: Path):
    result = _run({"path": str(tree / "train"), "split": "(none)"})
    assert result["classes"] == ["cat", "dog"]
    assert len(result["dataset"]) == 5


# ── transforms ───────────────────────────────────────────────────────────


def test_without_a_wired_transform_it_still_yields_tensors(tree: Path):
    dataset = _run({"path": str(tree), "split": "train"})["dataset"]
    sample, label = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 8, 8)
    assert label == 0


def test_the_train_split_takes_train_transform(tree: Path):
    resize = ResizeTransformNode().execute({}, {"size": 4})["transform"]
    chain = ToTensorTransformNode().execute(
        {"transform": resize}, {})["transform"]
    dataset = _run({"path": str(tree), "split": "train"},
                   {"train_transform": chain})["dataset"]
    assert dataset[0][0].shape == (3, 4, 4)


def test_a_non_train_split_ignores_train_transform(tree: Path):
    """Augmenting the evaluation set would make every measurement different."""
    resize = ResizeTransformNode().execute({}, {"size": 4})["transform"]
    chain = ToTensorTransformNode().execute(
        {"transform": resize}, {})["transform"]
    dataset = _run({"path": str(tree), "split": "val"},
                   {"train_transform": chain})["dataset"]
    assert dataset[0][0].shape == (3, 8, 8)


def test_eval_transform_is_the_fallback_for_the_train_split(tree: Path):
    resize = ResizeTransformNode().execute({}, {"size": 4})["transform"]
    chain = ToTensorTransformNode().execute(
        {"transform": resize}, {})["transform"]
    dataset = _run({"path": str(tree), "split": "train"},
                   {"eval_transform": chain})["dataset"]
    assert dataset[0][0].shape == (3, 4, 4)


def test_train_transform_wins_when_both_are_wired(tree: Path):
    train = ToTensorTransformNode().execute(
        {"transform": ResizeTransformNode().execute(
            {}, {"size": 4})["transform"]}, {})["transform"]
    evaluation = ToTensorTransformNode().execute(
        {"transform": ResizeTransformNode().execute(
            {}, {"size": 6})["transform"]}, {})["transform"]
    dataset = _run({"path": str(tree), "split": "train"},
                   {"train_transform": train,
                    "eval_transform": evaluation})["dataset"]
    assert dataset[0][0].shape == (3, 4, 4)


def test_the_none_split_honours_train_transform(tree: Path):
    """core#136 review, M-3.

    ``(none)`` is what the node's own help text and the augmentation guide
    tell you to pick for a flat ``<path>/<class>/<images>`` layout -- which
    is the layout a small custom dataset actually has. The old rule
    (``is_train = split == "train"``) therefore threw away the whole
    ``train_transform`` chain for exactly the users who wired it, with no
    error, no warning, and a plausible loss curve.
    """
    resize = ResizeTransformNode().execute({}, {"size": 4})["transform"]
    chain = ToTensorTransformNode().execute(
        {"transform": resize}, {})["transform"]
    dataset = _run({"path": str(tree / "train"), "split": "(none)"},
                   {"train_transform": chain})["dataset"]
    assert dataset[0][0].shape == (3, 4, 4)


def test_the_none_split_still_prefers_train_transform_over_eval(tree: Path):
    """Both wired and no split to tell them apart: train_transform wins."""
    train = ToTensorTransformNode().execute(
        {"transform": ResizeTransformNode().execute(
            {}, {"size": 4})["transform"]}, {})["transform"]
    evaluation = ToTensorTransformNode().execute(
        {"transform": ResizeTransformNode().execute(
            {}, {"size": 6})["transform"]}, {})["transform"]
    dataset = _run({"path": str(tree / "train"), "split": "(none)"},
                   {"train_transform": train,
                    "eval_transform": evaluation})["dataset"]
    assert dataset[0][0].shape == (3, 4, 4)


def test_the_none_split_still_falls_back_to_eval_transform(tree: Path):
    evaluation = ToTensorTransformNode().execute(
        {"transform": ResizeTransformNode().execute(
            {}, {"size": 6})["transform"]}, {})["transform"]
    dataset = _run({"path": str(tree / "train"), "split": "(none)"},
                   {"eval_transform": evaluation})["dataset"]
    assert dataset[0][0].shape == (3, 6, 6)


def test_an_augmenting_chain_survives_the_none_split(tmp_path: Path):
    """The behavioural half of M-3, not just the shape.

    A flip chain wired to ``train_transform`` at ``(none)`` used to produce
    zero flips over the whole dataset because the chain was never installed.

    Builds its own LEFT-RIGHT ASYMMETRIC image rather than using the shared
    fixture: the fixture's images are a single flat colour, so a horizontal
    flip is a no-op on them and this assertion would hold with or without
    the fix.
    """
    flat = tmp_path / "flat" / "cat"
    flat.mkdir(parents=True)
    pixels = torch.zeros(4, 4, 3, dtype=torch.uint8)
    pixels[:, 0, :] = 255  # left column white, right column black
    Image.fromarray(pixels.numpy(), mode="RGB").save(flat / "0.png")

    chain = RandomHorizontalFlipNode().execute({}, {"p": 1.0})["transform"]
    chain = ToTensorTransformNode().execute(
        {"transform": chain}, {})["transform"]
    plain = ToTensorTransformNode().execute({}, {})["transform"]

    upright = _run({"path": str(tmp_path / "flat"), "split": "(none)"},
                   {"eval_transform": plain})["dataset"][0][0]
    assert not torch.equal(upright, torch.flip(upright, dims=[2])), \
        "the fixture image must be asymmetric or this test pins nothing"

    flipped = _run({"path": str(tmp_path / "flat"), "split": "(none)"},
                   {"train_transform": chain})["dataset"][0][0]
    assert torch.equal(flipped, torch.flip(upright, dims=[2]))


def test_a_dropped_train_transform_is_warned_about(tree: Path, caplog):
    """Silence is what made M-3 cost a whole experiment; a test split warns."""
    chain = ToTensorTransformNode().execute({}, {})["transform"]
    with caplog.at_level("WARNING"):
        _run({"path": str(tree), "split": "val"}, {"train_transform": chain})
    assert any("train_transform" in record.getMessage()
               and "ImageFolderDataset" in record.getMessage()
               for record in caplog.records), caplog.text


def test_an_honoured_train_transform_warns_about_nothing(tree: Path, caplog):
    chain = ToTensorTransformNode().execute({}, {})["transform"]
    with caplog.at_level("WARNING"):
        _run({"path": str(tree), "split": "train"}, {"train_transform": chain})
        _run({"path": str(tree / "train"), "split": "(none)"},
             {"train_transform": chain})
    assert caplog.records == []


def test_a_random_chain_is_seeded_like_every_other_dataset(tree: Path):
    chain = RandomHorizontalFlipNode().execute({}, {"p": 0.5})["transform"]
    chain = ToTensorTransformNode().execute(
        {"transform": chain}, {})["transform"]
    context = ExecutionContext(seed=99)
    context.current_node_id = "folder1"
    dataset = _run({"path": str(tree), "split": "train"},
                   {"train_transform": chain}, context=context)["dataset"]
    assert isinstance(dataset.transform, SeededAugmentation)


# ── path resolution ──────────────────────────────────────────────────────


def test_a_relative_path_resolves_against_the_data_root(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    assert resolve_dataset_root("pets") == (tmp_path / "data" / "pets").resolve()


def test_an_absolute_path_is_used_as_given(tree: Path):
    assert resolve_dataset_root(str(tree)) == tree.resolve()


def test_a_tilde_expands_to_the_home_directory(monkeypatch, tmp_path):
    """Otherwise ``~`` is not special and lands under the data root verbatim."""
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    resolved = resolve_dataset_root("~/datasets/pets")
    assert resolved == (tmp_path / "home" / "datasets" / "pets").resolve()
    assert "~" not in str(resolved)


# ── the failure messages ─────────────────────────────────────────────────


def test_an_empty_path_is_rejected_before_anything_touches_the_disk():
    with pytest.raises(ValueError, match="needs a path"):
        _run({"path": "   ", "split": "train"})


def test_a_missing_base_directory_names_the_base(tmp_path: Path):
    with pytest.raises(ValueError, match="is not a directory"):
        _run({"path": str(tmp_path / "nope"), "split": "train"})


def test_a_missing_split_says_which_splits_do_exist(tree: Path):
    with pytest.raises(ValueError) as excinfo:
        _run({"path": str(tree), "split": "test"})
    message = str(excinfo.value)
    assert "no 'test' sub-directory" in message
    assert "train" in message and "val" in message


def test_class_folders_mistaken_for_splits_get_the_none_hint(tmp_path: Path):
    root = tmp_path / "flat"
    _write_images(root / "cat", 1)
    _write_images(root / "dog", 1)
    with pytest.raises(ValueError) as excinfo:
        _run({"path": str(root), "split": "train"})
    assert "'(none)'" in str(excinfo.value)


def test_a_directory_with_no_class_folders_says_so(tmp_path: Path):
    root = tmp_path / "loose"
    _write_images(root / "train", 2)  # images, but no class level
    with pytest.raises(ValueError, match="no sub-directories"):
        _run({"path": str(root), "split": "train"})


# ── the FILES, not only the tree (#197) ──────────────────────────────────
#
# The node validated the directory tree carefully and never opened a single
# file. ImageFolder accepts anything with a listed extension, so a
# truncated, zero-byte or merely renamed .png built fine and then raised
# PIL.UnidentifiedImageError from inside a DataLoader worker, potentially
# minutes into training, naming nothing the user could act on.


def test_a_zero_byte_image_is_named_at_build_time(tree: Path):
    (tree / "train" / "cat" / "broken.png").write_bytes(b"")
    with pytest.raises(ValueError) as excinfo:
        _run({"path": str(tree), "split": "train"})
    assert "broken.png" in str(excinfo.value)


def test_a_renamed_non_image_is_named_at_build_time(tree: Path):
    (tree / "train" / "dog" / "notes.png").write_text("this is not a png")
    with pytest.raises(ValueError) as excinfo:
        _run({"path": str(tree), "split": "train"})
    assert "notes.png" in str(excinfo.value)


def test_a_truncated_image_is_named_at_build_time(tree: Path):
    """The one PIL's ``open`` alone would miss: the header parses, so only
    ``verify()`` (chunk CRCs / markers) catches it."""
    good = tree / "train" / "cat" / "0.png"
    truncated = tree / "train" / "cat" / "cut.png"
    truncated.write_bytes(good.read_bytes()[:-40])
    with pytest.raises(ValueError) as excinfo:
        _run({"path": str(tree), "split": "train"})
    assert "cut.png" in str(excinfo.value)


def test_a_healthy_tree_is_not_rejected(tree: Path):
    """The check must not have made an ordinary dataset unloadable."""
    assert len(_run({"path": str(tree), "split": "train"})["dataset"]) == 5


def test_the_check_spreads_across_classes_rather_than_taking_the_first_n(
    monkeypatch, tmp_path: Path,
):
    """``samples`` is ordered by class, so a first-N sample would only ever
    inspect the alphabetically first one and miss every later class."""
    from app.nodes.data import image_folder_dataset_node as node_module

    monkeypatch.setattr(node_module, "VERIFY_SAMPLE_SIZE", 4)
    root = tmp_path / "wide"
    _write_images(root / "aaa", 20)
    _write_images(root / "zzz", 20)
    (root / "zzz" / "broken.png").write_bytes(b"")

    with pytest.raises(ValueError) as excinfo:
        _run({"path": str(root), "split": "(none)"})
    assert "broken.png" in str(excinfo.value)


def test_the_check_costs_a_bounded_number_of_opens(monkeypatch, tmp_path: Path):
    """A fixed cost, not one that grows with the dataset -- reading every
    file at build time is exactly what this node exists to defer."""
    from app.nodes.data import image_folder_dataset_node as node_module

    root = tmp_path / "big"
    _write_images(root / "cat", 100)
    opened: list[str] = []
    real_open = Image.open

    def _counting_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", _counting_open)
    _run({"path": str(root), "split": "(none)"})
    assert 0 < len(opened) <= node_module.VERIFY_SAMPLE_SIZE
