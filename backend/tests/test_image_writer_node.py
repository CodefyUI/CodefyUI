"""Tests for ImageWriterNode."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from app.config import settings
from app.nodes.io.image_writer_node import ImageWriterNode


def test_node_metadata():
    assert ImageWriterNode.NODE_NAME == "ImageWriter"
    assert ImageWriterNode.CATEGORY == "IO"


def test_writes_image_to_output_dir():
    # Use a relative path so it goes under settings data root
    image = torch.zeros(3, 16, 16)
    res = ImageWriterNode().execute(
        {"image": image},
        {"path": "test_writer_output.png", "format": "PNG"},
    )
    path = res["path"]
    assert path.endswith(".png")
    # Cleanup
    from pathlib import Path
    Path(path).unlink(missing_ok=True)


def test_format_changes_extension():
    image = torch.zeros(3, 4, 4)
    res = ImageWriterNode().execute(
        {"image": image},
        {"path": "test_writer_output.png", "format": "JPEG"},
    )
    assert res["path"].endswith(".jpg")
    from pathlib import Path
    Path(res["path"]).unlink(missing_ok=True)


def test_batched_input_writes_first_image():
    """4D input (N, C, H, W) — first image of batch is saved."""
    image = torch.zeros(2, 3, 8, 8)
    res = ImageWriterNode().execute(
        {"image": image},
        {"path": "test_writer_batch.png"},
    )
    from pathlib import Path
    Path(res["path"]).unlink(missing_ok=True)


def test_absolute_path_outside_data_dir_raises(tmp_path):
    image = torch.zeros(3, 4, 4)
    bad_path = str(tmp_path / "should_fail.png")
    with pytest.raises(ValueError, match="within the project"):
        ImageWriterNode().execute({"image": image}, {"path": bad_path})
