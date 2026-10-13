"""Tests for ImageWriterNode."""

from __future__ import annotations

import pytest
import torch

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


# -- 2026-10-13 audit: dtype, range and extension ----------------------------


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "data" / "codefyui.db")
    return tmp_path / "data"


def _read_back(path):
    import numpy as np
    from PIL import Image

    return torch.from_numpy(np.array(Image.open(path))).permute(2, 0, 1)


def test_uint8_image_round_trips_exactly(data_root):
    """It used to raise "result type Float can't be cast to ... Byte"."""
    torch.manual_seed(0)
    image = torch.randint(0, 256, (3, 5, 7), dtype=torch.uint8)
    res = ImageWriterNode().execute({"image": image}, {"path": "u8.png", "format": "PNG"})
    assert torch.equal(_read_back(res["path"]), image)
    assert "__log__" not in res


def test_negative_values_write_a_run_log_warning(data_root):
    warnings = []

    class Ctx:
        def log_warning(self, kind, note):
            warnings.append(kind)

    image = torch.linspace(-1, 1, 3 * 4 * 4).view(3, 4, 4)
    res = ImageWriterNode().execute(
        {"image": image}, {"path": "neg.png", "format": "PNG"}, context=Ctx())
    assert res["__log__"].startswith("[WARN] ImageWriter: the image's values span [-1, 1]")
    assert "(x + 1) / 2" in res["__log__"]
    assert warnings == ["image_writer_negative_values"]


def test_float_image_in_range_writes_no_log_line(data_root):
    res = ImageWriterNode().execute(
        {"image": torch.rand(3, 4, 4)}, {"path": "ok.png", "format": "PNG"})
    assert "__log__" not in res


@pytest.mark.parametrize("name,fmt", [("photo.jpeg", "JPEG"), ("scan.tif", "TIFF")])
def test_alternate_spelling_of_the_format_extension_is_kept(data_root, name, fmt):
    res = ImageWriterNode().execute({"image": torch.rand(3, 4, 4)}, {"path": name, "format": fmt})
    assert res["path"].endswith(name)
