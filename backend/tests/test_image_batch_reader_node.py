"""Tests for ImageBatchReaderNode."""

from __future__ import annotations

import pytest
from PIL import Image

from app.config import settings
from app.nodes.io.image_batch_reader_node import ImageBatchReaderNode


def test_node_metadata():
    assert ImageBatchReaderNode.NODE_NAME == "ImageBatchReader"
    assert ImageBatchReaderNode.CATEGORY == "IO"
    assert [p.name for p in ImageBatchReaderNode.define_inputs()] == ["transform", "directory"]
    assert all(p.optional for p in ImageBatchReaderNode.define_inputs())


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


# -- 2026-10-13 audit: brace patterns ----------------------------------------


def test_expand_braces():
    from app.nodes.io.image_batch_reader_node import expand_braces

    assert expand_braces("*.png") == ["*.png"]
    assert expand_braces("*.{png,jpg}") == ["*.png", "*.jpg"]
    assert expand_braces("{a,b}_{1,2}.png") == ["a_1.png", "a_2.png", "b_1.png", "b_2.png"]
    assert expand_braces("*.{jp{g,eg},png}") == ["*.jpg", "*.jpeg", "*.png"]
    assert expand_braces("*.{png") == ["*.{png"]


def test_brace_pattern_matches_every_alternative(monkeypatch, tmp_path):
    """The documented *.{png,jpg} used to find "No images" at all."""
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    test_dir = settings.MODELS_DIR.parent / "imgs"
    test_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (8, 8), (i * 60, 0, 0)).save(test_dir / f"img_{i}.png")
    Image.new("RGB", (8, 8)).save(test_dir / "photo.jpg")
    Image.new("RGB", (8, 8)).save(test_dir / "skip.bmp")
    params = {"directory": str(test_dir), "pattern": "*.{png,jpg}", "resize": 8,
              "max_images": 0, "mode": "RGB"}

    res = ImageBatchReaderNode().execute({}, params)
    assert res["count"] == 4
    # The cache fingerprint covers the same files the loader reads.
    names = [p.name for p in ImageBatchReaderNode._selected_paths(params)]
    assert names == ["img_0.png", "img_1.png", "img_2.png", "photo.jpg"]



# -- a wired directory -----------------------------------------------------


def _three_images(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        Image.new("RGB", (8, 8), (i * 60, 0, 0)).save(directory / f"img_{i}.png")


def test_a_wired_directory_overrides_the_param(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    wired = settings.MODELS_DIR.parent / "wired"
    _three_images(wired)
    res = ImageBatchReaderNode().execute(
        {"directory": str(wired)},
        {"directory": str(settings.MODELS_DIR.parent / "not-here"), "resize": 4},
    )
    assert res["count"] == 3


def test_a_wired_relative_directory_resolves_under_the_data_root(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    _three_images(settings.MODELS_DIR.parent / "rel")
    res = ImageBatchReaderNode().execute({"directory": "rel"}, {"resize": 4})
    assert res["images"].shape == (3, 3, 4, 4)


def test_a_wired_directory_outside_the_data_root_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    outside = tmp_path / "outside"
    _three_images(outside)
    with pytest.raises(ValueError, match="within the project data"):
        ImageBatchReaderNode().execute({"directory": str(outside)}, {"resize": 4})
    with pytest.raises(ValueError, match="within the project data"):
        ImageBatchReaderNode().execute({"directory": "../outside"}, {"resize": 4})


def test_writer_then_reader_round_trip_in_one_graph(monkeypatch, tmp_path):
    """The directory edge orders the read after the write; before it, the
    reader ran first and found no directory."""
    import asyncio

    import torch

    from app.core.graph_engine import execute_graph

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "data" / "codefyui.db")
    nodes = [
        {"id": "s", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {"id": "t", "type": "TensorCreate", "position": {"x": 0, "y": 0},
         "data": {"params": {"fill": "rand", "shape": "4,3,6,6"}}},
        {"id": "w", "type": "ImageWriter", "position": {"x": 0, "y": 0},
         "data": {"params": {"path": "rt/img.png", "format": "PNG"}}},
        {"id": "r", "type": "ImageBatchReader", "position": {"x": 0, "y": 0},
         "data": {"params": {"directory": "", "pattern": "*.png", "resize": 6}}},
    ]
    edges = [
        {"id": "e0", "source": "s", "sourceHandle": "trigger", "target": "t", "type": "trigger"},
        {"id": "e1", "source": "t", "sourceHandle": "tensor", "target": "w", "targetHandle": "image"},
        {"id": "e2", "source": "w", "sourceHandle": "directory", "target": "r", "targetHandle": "directory"},
    ]
    results = asyncio.run(execute_graph(nodes, edges))
    assert results["r"]["count"] == 4
    assert torch.allclose(results["r"]["images"], results["t"]["tensor"], atol=1 / 255)
