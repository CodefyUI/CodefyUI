"""Tests for ImageBatchReaderNode."""

from __future__ import annotations

import pytest
from PIL import Image

from app.config import settings
from app.nodes.io.image_batch_reader_node import ImageBatchReaderNode


def test_node_metadata():
    assert ImageBatchReaderNode.NODE_NAME == "ImageBatchReader"
    assert ImageBatchReaderNode.CATEGORY == "IO"


def test_empty_directory_raises():
    with pytest.raises(ValueError, match="Directory"):
        ImageBatchReaderNode().execute({}, {"directory": ""})


def test_nonexistent_dir_raises(monkeypatch, tmp_path):
    # Isolated under tmp_path the same way as the writer tests below
    # (review follow-up on #151): this one creates nothing, so it cannot
    # actually clobber a concurrent run, but pointing it at the real
    # backend/data/ instead read as half-applied isolation next to the
    # three that are. MODELS_DIR still has to be patched (not just
    # tmp_path used directly) for the same reason as the others: the
    # node's own path-escape check only accepts a directory under
    # MODELS_DIR.parent.
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    target_dir = settings.MODELS_DIR.parent / "_missing_dir_for_test"
    with pytest.raises(FileNotFoundError):
        ImageBatchReaderNode().execute({}, {"directory": str(target_dir)})


def test_reads_batch_of_images(monkeypatch, tmp_path):
    # A fixed name under the real backend/data/ (#151) let two concurrent
    # pytest runs race to create/write/delete the same directory. Point
    # MODELS_DIR under tmp_path -- unique per test process -- the same way
    # test_image_folder_dataset_node.py and test_tensorboard.py already do;
    # the node itself confines `directory` under MODELS_DIR.parent, so this
    # also keeps the node's own path-escape check satisfied.
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    test_dir = settings.MODELS_DIR.parent / "_test_img_batch"
    test_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        img = Image.new("RGB", (32, 32), (i * 50, 0, 0))
        img.save(test_dir / f"img_{i}.png")
    res = ImageBatchReaderNode().execute(
        {},
        {"directory": str(test_dir), "pattern": "*.png", "resize": 16, "max_images": 0, "mode": "RGB"},
    )
    assert res["images"].shape == (3, 3, 16, 16)
    assert res["count"] == 3


def test_max_images_limits_count(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    test_dir = settings.MODELS_DIR.parent / "_test_max_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        Image.new("RGB", (16, 16)).save(test_dir / f"img_{i}.png")
    res = ImageBatchReaderNode().execute(
        {},
        {"directory": str(test_dir), "pattern": "*.png", "resize": 8, "max_images": 2, "mode": "RGB"},
    )
    assert res["count"] == 2


def test_no_matching_files_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    test_dir = settings.MODELS_DIR.parent / "_test_no_match"
    test_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No images"):
        ImageBatchReaderNode().execute(
            {},
            {"directory": str(test_dir), "pattern": "*.png", "resize": 16},
        )
