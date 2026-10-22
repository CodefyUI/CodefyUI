"""Tests for ImageWriterNode."""

from __future__ import annotations

from pathlib import Path

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


# -- tensor input, batches and the directory output ---------------------------


def test_image_input_takes_a_tensor_and_an_image_source():
    from app.core.node_base import DataType
    from app.core.type_system import is_compatible

    (port,) = ImageWriterNode.define_inputs()
    assert port.data_type is DataType.TENSOR
    assert is_compatible(DataType.TENSOR, port.data_type)
    assert is_compatible(DataType.IMAGE, port.data_type)
    assert [o.name for o in ImageWriterNode.define_outputs()] == ["path", "paths", "directory"]


def test_a_batch_writes_one_indexed_file_per_image(data_root):
    torch.manual_seed(0)
    batch = torch.rand(3, 3, 4, 5)
    res = ImageWriterNode().execute({"image": batch}, {"path": "set/shape.png", "format": "PNG"})
    out = data_root / "output" / "set"
    assert res["directory"] == str(out.resolve())
    assert res["paths"] == [str((out / f"shape_{i}.png").resolve()) for i in range(3)]
    assert res["path"] == res["paths"][0]
    for i, written in enumerate(res["paths"]):
        assert torch.equal(_read_back(written), (batch[i] * 255 + 0.5).clamp(0, 255).to(torch.uint8))


def test_the_index_is_zero_padded_so_files_sort_in_order(data_root):
    res = ImageWriterNode().execute({"image": torch.zeros(11, 1, 2, 2)}, {"path": "pad.png"})
    names = [Path(p).name for p in res["paths"]]
    assert names[0] == "pad_00.png" and names[-1] == "pad_10.png"
    assert names == sorted(names)


def test_a_uint8_batch_round_trips_exactly(data_root):
    torch.manual_seed(1)
    batch = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
    res = ImageWriterNode().execute({"image": batch}, {"path": "u8b.png"})
    for i, written in enumerate(res["paths"]):
        assert torch.equal(_read_back(written), batch[i])


def test_a_batch_of_one_keeps_the_plain_name(data_root):
    res = ImageWriterNode().execute({"image": torch.zeros(1, 3, 4, 4)}, {"path": "one.png"})
    assert Path(res["path"]).name == "one.png"
    assert res["paths"] == [res["path"]]
    assert res["directory"] == str(Path(res["path"]).parent)


def test_a_negative_batch_logs_one_warning(data_root):
    res = ImageWriterNode().execute({"image": torch.full((2, 3, 2, 2), -0.5)}, {"path": "n.png"})
    assert res["__log__"].count("[WARN] ImageWriter:") == 1


def test_an_indexed_name_is_checked_against_protected_storage(data_root, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DB_PATH", data_root / "output" / "db_1.png")
    with pytest.raises(ValueError, match="own storage"):
        ImageWriterNode().execute({"image": torch.zeros(2, 3, 2, 2)}, {"path": "db.png"})
