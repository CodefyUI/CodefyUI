"""Unit tests for HuggingFaceDataset and KaggleDataset nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeHFDataset:
    """Minimal stand-in for huggingface `datasets.Dataset`.

    Implements the subset of the API our node uses: __len__, __getitem__,
    and a `features` attribute that exposes column names as dict keys.
    """

    def __init__(self, rows: list[dict[str, Any]], features: dict[str, Any]):
        self._rows = rows
        self.features = features

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


def _make_image_folder_layout(
    root: Path,
    classes: dict[str, int] | None = None,
) -> None:
    """Create a tiny ImageFolder layout under `root`.

    Default layout: two classes ('cats', 'dogs') with one 4x4 PNG each.
    """
    if classes is None:
        classes = {"cats": 1, "dogs": 1}
    for class_name, count in classes.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (4, 4), (i * 30 % 255, 0, 0))
            img.save(class_dir / f"{class_name}_{i}.png")


def _make_two_row_hf_image_dataset() -> FakeHFDataset:
    return FakeHFDataset(
        rows=[
            {"image": Image.new("RGB", (8, 8), (255, 0, 0)), "label": 0},
            {"image": Image.new("RGB", (8, 8), (0, 255, 0)), "label": 1},
        ],
        features={"image": None, "label": None},
    )


# ---------------------------------------------------------------------------
# Smoke test (proves the file is collected and helpers work)
# ---------------------------------------------------------------------------


def test_helpers_smoke(tmp_path):
    fake = _make_two_row_hf_image_dataset()
    assert len(fake) == 2
    assert fake[0]["label"] == 0

    _make_image_folder_layout(tmp_path)
    assert (tmp_path / "cats").is_dir()
    assert (tmp_path / "dogs").is_dir()
    assert any(p.suffix == ".png" for p in (tmp_path / "cats").iterdir())


# ---------------------------------------------------------------------------
# HFTorchImageDataset adapter
# ---------------------------------------------------------------------------


def test_hf_adapter_returns_pil_when_no_transform():
    from app.nodes.data._hf_adapter import HFTorchImageDataset

    fake = _make_two_row_hf_image_dataset()
    ds = HFTorchImageDataset(fake, image_column="image", label_column="label")

    assert len(ds) == 2
    img, label = ds[0]
    assert isinstance(img, Image.Image)
    assert label == 0
    assert ds[1][1] == 1


def test_hf_adapter_applies_transform_in_constructor():
    import torch
    from torchvision import transforms

    from app.nodes.data._hf_adapter import HFTorchImageDataset

    fake = _make_two_row_hf_image_dataset()
    ds = HFTorchImageDataset(
        fake,
        image_column="image",
        label_column="label",
        transform=transforms.ToTensor(),
    )

    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 8, 8)
    assert label == 0


def test_hf_adapter_transform_is_mutable_after_init():
    """TransformNode interop: setting `dataset.transform` later must take effect."""
    from app.nodes.data._hf_adapter import HFTorchImageDataset

    fake = _make_two_row_hf_image_dataset()
    ds = HFTorchImageDataset(fake, image_column="image", label_column="label")

    sentinel = object()
    ds.transform = lambda _img: sentinel
    img, _ = ds[0]
    assert img is sentinel


# ---------------------------------------------------------------------------
# HuggingFaceDatasetNode
# ---------------------------------------------------------------------------


def _hf_node_default_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "dataset_name": "fake/dataset",
        "subset": "",
        "split": "train",
        "image_column": "image",
        "label_column": "label",
        "cache_dir": "",
    }
    base.update(overrides)
    return base


def test_hf_dataset_node_happy_path(monkeypatch):
    import torch
    import datasets as hf_datasets

    fake = _make_two_row_hf_image_dataset()

    def fake_load_dataset(name, subset=None, split=None, cache_dir=None, **kwargs):
        assert name == "fake/dataset"
        assert subset is None
        assert split == "train"
        assert cache_dir is None
        # M-3 fix: HF node must pass trust_remote_code=False to prevent the
        # dataset-script RCE class.
        assert kwargs.get("trust_remote_code") is False
        return fake

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    result = node.execute(inputs={}, params=_hf_node_default_params())

    ds = result["dataset"]
    assert len(ds) == 2
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)  # default ToTensor() applied by node
    assert img.shape == (3, 8, 8)
    assert label == 0


def test_hf_dataset_node_invalid_image_column_lists_available(monkeypatch):
    import datasets as hf_datasets

    fake = FakeHFDataset(
        rows=[{"img": Image.new("RGB", (8, 8)), "label": 0}],
        features={"img": None, "label": None},
    )
    monkeypatch.setattr(hf_datasets, "load_dataset", lambda *a, **k: fake)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(
            inputs={},
            params=_hf_node_default_params(image_column="image"),  # dataset has 'img'
        )

    msg = str(exc_info.value)
    assert "image" in msg
    assert "img" in msg  # available columns are listed
    assert "label" in msg


def test_hf_dataset_node_invalid_label_column_lists_available(monkeypatch):
    import datasets as hf_datasets

    fake = FakeHFDataset(
        rows=[{"image": Image.new("RGB", (8, 8)), "target": 0}],
        features={"image": None, "target": None},
    )
    monkeypatch.setattr(hf_datasets, "load_dataset", lambda *a, **k: fake)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(
            inputs={},
            params=_hf_node_default_params(label_column="label"),  # dataset has 'target'
        )

    msg = str(exc_info.value)
    assert "label" in msg
    assert "target" in msg


def test_hf_dataset_node_missing_package(monkeypatch):
    import builtins
    import sys

    # Force any future `import datasets` to fail.
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("No module named 'datasets'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_hf_node_default_params())

    msg = str(exc_info.value)
    assert "datasets" in msg
    assert "pip install" in msg


def test_hf_dataset_node_gated_repo_maps_to_token_message(monkeypatch):
    import datasets as hf_datasets

    class GatedRepoError(Exception):
        """Stand-in mimicking huggingface_hub.errors.GatedRepoError."""

    def fake_load_dataset(*args, **kwargs):
        raise GatedRepoError("Access to model X is gated")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(
            inputs={},
            params=_hf_node_default_params(dataset_name="private/gated"),
        )

    msg = str(exc_info.value)
    assert "HF_TOKEN" in msg
    assert "authentication" in msg.lower() or "auth" in msg.lower()


def test_hf_dataset_node_401_error_maps_to_token_message(monkeypatch):
    import datasets as hf_datasets

    def fake_load_dataset(*args, **kwargs):
        raise Exception("401 Client Error: Unauthorized")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_hf_node_default_params())

    msg = str(exc_info.value)
    assert "HF_TOKEN" in msg


def test_hf_dataset_node_other_errors_pass_through(monkeypatch):
    """Non-auth errors must NOT be silently rewritten as auth errors."""
    import datasets as hf_datasets

    def fake_load_dataset(*args, **kwargs):
        raise ValueError("dataset name is malformed")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(ValueError, match="malformed"):
        node.execute(inputs={}, params=_hf_node_default_params())


# ---------------------------------------------------------------------------
# KaggleDatasetNode
# ---------------------------------------------------------------------------


def _kaggle_node_default_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "dataset_slug": "fake/dataset",
        "subdir": "",
        "cache_dir": "",
    }
    base.update(overrides)
    return base


def test_kaggle_dataset_node_happy_path(monkeypatch, tmp_path):
    import torch
    import kagglehub

    _make_image_folder_layout(tmp_path)
    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))

    # Auth pre-check requires either env vars or kaggle.json
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    result = node.execute(inputs={}, params=_kaggle_node_default_params())

    ds = result["dataset"]
    assert len(ds) == 2  # cats + dogs, one image each
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert label in (0, 1)


def test_kaggle_dataset_node_uses_subdir(monkeypatch, tmp_path):
    import torch
    import kagglehub

    # Layout: tmp_path/train/<class>/<image>.png
    _make_image_folder_layout(tmp_path / "train")
    # Distractor at the top level (no class structure)
    (tmp_path / "README.md").write_text("not an image folder")

    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    result = node.execute(
        inputs={},
        params=_kaggle_node_default_params(subdir="train"),
    )

    ds = result["dataset"]
    assert len(ds) == 2
    img, _label = ds[0]
    assert isinstance(img, torch.Tensor)


def test_kaggle_dataset_node_missing_package(monkeypatch):
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "kagglehub", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kagglehub" or name.startswith("kagglehub."):
            raise ImportError("No module named 'kagglehub'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_kaggle_node_default_params())

    msg = str(exc_info.value)
    assert "kagglehub" in msg
    assert "pip install" in msg


def test_kaggle_dataset_node_missing_auth(monkeypatch, tmp_path):
    import kagglehub

    # Make sure neither env vars nor ~/.kaggle/kaggle.json exist
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Even if dataset_download were called, fail loudly so the test catches it
    def should_not_be_called(slug):
        raise AssertionError("dataset_download was called despite missing auth")

    monkeypatch.setattr(kagglehub, "dataset_download", should_not_be_called)

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_kaggle_node_default_params())

    msg = str(exc_info.value)
    assert "Kaggle" in msg
    assert ("KAGGLE_USERNAME" in msg) or ("kaggle.json" in msg)


def test_kaggle_dataset_node_flat_directory_raises_with_subdir_hint(monkeypatch, tmp_path):
    import kagglehub

    # Flat: just files, no class subdirs
    (tmp_path / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "image2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "metadata.csv").write_text("id,label\n")

    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_kaggle_node_default_params())

    msg = str(exc_info.value)
    assert "subdir" in msg.lower()
    # The error should surface what was actually in the directory
    assert "image1.png" in msg or "metadata.csv" in msg


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_new_nodes_are_registered(registry_with_nodes):
    """Auto-discovery (in conftest fixture) should pick up both new nodes."""
    nodes = registry_with_nodes.nodes
    assert "HuggingFaceDataset" in nodes
    assert "KaggleDataset" in nodes

    hf_cls = nodes["HuggingFaceDataset"]
    kaggle_cls = nodes["KaggleDataset"]

    assert hf_cls.CATEGORY == "Data"
    assert kaggle_cls.CATEGORY == "Data"

    hf_param_names = {p.name for p in hf_cls.define_params()}
    assert {"dataset_name", "split", "image_column", "label_column"} <= hf_param_names

    kaggle_param_names = {p.name for p in kaggle_cls.define_params()}
    assert {"dataset_slug", "subdir", "cache_dir"} <= kaggle_param_names


def test_kaggle_dataset_node_cache_dir_does_not_leak_across_calls(monkeypatch, tmp_path):
    """KAGGLEHUB_CACHE env var must be restored after execute() returns."""
    import kagglehub

    _make_image_folder_layout(tmp_path)
    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    # Establish a baseline value for KAGGLEHUB_CACHE
    monkeypatch.setenv("KAGGLEHUB_CACHE", "/baseline/cache")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    node.execute(
        inputs={},
        params=_kaggle_node_default_params(cache_dir="/temporary/override"),
    )

    # After execute() the env var must be restored to the baseline,
    # not stuck on the temporary override.
    assert os.environ.get("KAGGLEHUB_CACHE") == "/baseline/cache"


def test_hf_dataset_node_empty_split_falls_back_to_train(monkeypatch):
    """An empty `split` param must default to 'train', not be passed through."""
    import datasets as hf_datasets

    fake = _make_two_row_hf_image_dataset()
    captured = {}

    def fake_load_dataset(name, subset=None, split=None, cache_dir=None, **kwargs):
        captured["split"] = split
        return fake

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    node.execute(inputs={}, params=_hf_node_default_params(split=""))

    assert captured["split"] == "train"
