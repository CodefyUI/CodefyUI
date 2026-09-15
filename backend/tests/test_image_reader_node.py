"""Tests for ImageReaderNode."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from app.nodes.io.image_reader_node import ImageReaderNode


def test_node_metadata():
    assert ImageReaderNode.NODE_NAME == "ImageReader"
    assert ImageReaderNode.CATEGORY == "IO"
    out_names = [p.name for p in ImageReaderNode.define_outputs()]
    assert "image" in out_names
    assert "tensor" in out_names


def test_empty_path_raises():
    with pytest.raises(ValueError, match="path"):
        ImageReaderNode().execute({}, {"path": ""})


def test_nonexistent_path_raises(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        ImageReaderNode().execute({}, {"path": str(missing)})


def test_reads_rgb_image(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (32, 24), (10, 20, 30))
    img.save(img_path)
    res = ImageReaderNode().execute({}, {"path": str(img_path), "mode": "RGB", "resize": 0})
    assert res["image"].shape == (3, 24, 32)
    assert res["image"].dtype == torch.float32
    assert (res["image"] >= 0).all()
    assert (res["image"] <= 1).all()


def test_grayscale_mode(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (16, 16))
    img.save(img_path)
    res = ImageReaderNode().execute({}, {"path": str(img_path), "mode": "L", "resize": 0})
    assert res["image"].shape == (1, 16, 16)


def test_resize_param_resizes_image(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)
    res = ImageReaderNode().execute({}, {"path": str(img_path), "mode": "RGB", "resize": 32})
    assert res["image"].shape == (3, 32, 32)


def test_image_and_tensor_outputs_match(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (8, 8), (50, 100, 150))
    img.save(img_path)
    res = ImageReaderNode().execute({}, {"path": str(img_path)})
    assert torch.equal(res["image"], res["tensor"])


def test_relative_path_prefers_images_dir(tmp_path, monkeypatch):
    """A bare filename present in the upload store resolves there, not to cwd."""
    from app.config import settings

    upload_store = tmp_path / "images"
    upload_store.mkdir()
    Image.new("RGB", (8, 4), (1, 2, 3)).save(upload_store / "shared.png")

    workdir = tmp_path / "work"
    workdir.mkdir()
    Image.new("RGB", (64, 32), (9, 9, 9)).save(workdir / "shared.png")

    monkeypatch.setattr(settings, "IMAGES_DIR", upload_store)
    monkeypatch.chdir(workdir)
    res = ImageReaderNode().execute({}, {"path": "shared.png", "mode": "RGB", "resize": 0})
    assert res["tensor"].shape == (3, 4, 8)


def test_relative_path_falls_back_to_cwd(tmp_path, monkeypatch):
    """Not in the upload store -> read it relative to the working directory.

    This is what lets a judging sandbox run a graph whose image sits beside
    the submission with no CODEFYUI_IMAGES_DIR set, matching what CSVReader
    has always done for DATA_FILE params.
    """
    from app.config import settings

    upload_store = tmp_path / "images"
    upload_store.mkdir()

    workdir = tmp_path / "work"
    workdir.mkdir()
    Image.new("RGB", (64, 32), (9, 9, 9)).save(workdir / "local_only.png")

    monkeypatch.setattr(settings, "IMAGES_DIR", upload_store)
    monkeypatch.chdir(workdir)
    res = ImageReaderNode().execute({}, {"path": "local_only.png", "mode": "RGB", "resize": 0})
    assert res["tensor"].shape == (3, 32, 64)


def test_relative_path_missing_everywhere_still_raises(tmp_path, monkeypatch):
    from app.config import settings

    upload_store = tmp_path / "images"
    upload_store.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()

    monkeypatch.setattr(settings, "IMAGES_DIR", upload_store)
    monkeypatch.chdir(workdir)
    with pytest.raises(FileNotFoundError, match="nowhere.png"):
        ImageReaderNode().execute({}, {"path": "nowhere.png", "mode": "RGB", "resize": 0})
