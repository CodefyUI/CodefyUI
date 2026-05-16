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


def test_nonexistent_dir_raises(tmp_path):
    # Use a directory under data root for the path validation to pass
    target_dir = settings.MODELS_DIR.parent / "_missing_dir_for_test"
    with pytest.raises(FileNotFoundError):
        ImageBatchReaderNode().execute({}, {"directory": str(target_dir)})


def test_reads_batch_of_images():
    test_dir = settings.MODELS_DIR.parent / "_test_img_batch"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(3):
            img = Image.new("RGB", (32, 32), (i * 50, 0, 0))
            img.save(test_dir / f"img_{i}.png")
        res = ImageBatchReaderNode().execute(
            {},
            {"directory": str(test_dir), "pattern": "*.png", "resize": 16, "max_images": 0, "mode": "RGB"},
        )
        assert res["images"].shape == (3, 3, 16, 16)
        assert res["count"] == 3
    finally:
        for f in test_dir.glob("*"):
            f.unlink()
        test_dir.rmdir()


def test_max_images_limits_count():
    test_dir = settings.MODELS_DIR.parent / "_test_max_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(5):
            Image.new("RGB", (16, 16)).save(test_dir / f"img_{i}.png")
        res = ImageBatchReaderNode().execute(
            {},
            {"directory": str(test_dir), "pattern": "*.png", "resize": 8, "max_images": 2, "mode": "RGB"},
        )
        assert res["count"] == 2
    finally:
        for f in test_dir.glob("*"):
            f.unlink()
        test_dir.rmdir()


def test_no_matching_files_raises():
    test_dir = settings.MODELS_DIR.parent / "_test_no_match"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(ValueError, match="No images"):
            ImageBatchReaderNode().execute(
                {},
                {"directory": str(test_dir), "pattern": "*.png", "resize": 16},
            )
    finally:
        test_dir.rmdir()
