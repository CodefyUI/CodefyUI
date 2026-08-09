"""Tests for the image files API and ImageReaderNode path resolution."""

import io

import pytest
from PIL import Image


def _png_bytes(size=(4, 4), color=(255, 0, 0)) -> bytes:
    """Build a tiny valid PNG payload."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def images_dir(tmp_path, monkeypatch):
    """Redirect settings.IMAGES_DIR at a temp dir for each test."""
    d = tmp_path / "images"
    d.mkdir()
    monkeypatch.setattr("app.config.settings.IMAGES_DIR", d)
    return d


@pytest.mark.asyncio
async def test_upload_and_download_roundtrip(test_client, images_dir):
    payload = _png_bytes()
    resp = await test_client.post(
        "/api/images/upload",
        files={"file": ("cat.png", payload, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] == "cat.png"

    resp = await test_client.get("/api/images/download/cat.png")
    assert resp.status_code == 200
    assert resp.content == payload
    assert "cat.png" in resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_upload_rejects_non_image_extension(test_client, images_dir):
    resp = await test_client.post(
        "/api/images/upload",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_download_missing_returns_404(test_client, images_dir):
    resp = await test_client.get("/api/images/download/missing.png")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_path_traversal_rejected(test_client, images_dir):
    resp = await test_client.get("/api/images/download/..%2Fsecret.png")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_and_delete(test_client, images_dir):
    (images_dir / "a.png").write_bytes(_png_bytes())
    (images_dir / "b.jpg").write_bytes(b"\xff\xd8\xff\xe0JPEG")  # minimal-ish
    (images_dir / "c.txt").write_bytes(b"ignored")

    resp = await test_client.get("/api/images")
    assert resp.status_code == 200
    names = {item["filename"] for item in resp.json()}
    assert names == {"a.png", "b.jpg"}  # .txt ignored

    resp = await test_client.delete("/api/images/a.png")
    assert resp.status_code == 200
    assert not (images_dir / "a.png").exists()


def test_image_reader_resolves_relative_path_against_images_dir(images_dir):
    """ImageReaderNode should treat a bare filename as living under IMAGES_DIR."""
    from app.nodes.io.image_reader_node import ImageReaderNode

    (images_dir / "pic.png").write_bytes(_png_bytes())

    node = ImageReaderNode()
    result = node.execute({}, {"path": "pic.png", "mode": "RGB", "resize": 0})
    assert result["image"].shape == (3, 4, 4)


def test_image_reader_absolute_path_still_works(images_dir, tmp_path):
    """Absolute paths outside IMAGES_DIR should still load (backward compat)."""
    from app.nodes.io.image_reader_node import ImageReaderNode

    outside = tmp_path / "outside.png"
    outside.write_bytes(_png_bytes())

    node = ImageReaderNode()
    result = node.execute({}, {"path": str(outside), "mode": "RGB", "resize": 0})
    assert result["image"].shape == (3, 4, 4)


def test_image_reader_resize_produces_square_from_square_input(images_dir):
    """A square image with resize=28 should yield exactly (C, 28, 28)."""
    from app.nodes.io.image_reader_node import ImageReaderNode

    (images_dir / "square.png").write_bytes(_png_bytes(size=(64, 64)))

    node = ImageReaderNode()
    result = node.execute({}, {"path": "square.png", "mode": "L", "resize": 28})
    assert result["image"].shape == (1, 28, 28)


def test_image_reader_resize_produces_square_from_nonsquare_input(images_dir):
    """A non-square image with resize=28 should also yield (C, 28, 28).

    This is the regression test for the inference example bug — the old
    Resize(int) only resized the shorter side, so a 100x500 image would end
    up as (1, 140, 28) and break downstream conv layers expecting 28x28.
    """
    from app.nodes.io.image_reader_node import ImageReaderNode

    (images_dir / "tall.png").write_bytes(_png_bytes(size=(100, 500)))
    (images_dir / "wide.png").write_bytes(_png_bytes(size=(500, 100)))

    node = ImageReaderNode()
    tall = node.execute({}, {"path": "tall.png", "mode": "L", "resize": 28})
    wide = node.execute({}, {"path": "wide.png", "mode": "L", "resize": 28})
    assert tall["image"].shape == (1, 28, 28)
    assert wide["image"].shape == (1, 28, 28)


def test_image_reader_inference_pipeline_shape_compatible_with_mnist_cnn(images_dir):
    """End-to-end shape check matching the InferenceCNN-MNIST example.

    Mirrors the example pipeline: ImageReader(L, 28) -> Unsqueeze(0) ->
    Conv2d(1, 32, 3, padding=1). Catches the regression where the inference
    example failed because the image had the wrong number of channels.
    """
    import torch.nn as nn

    from app.nodes.io.image_reader_node import ImageReaderNode

    (images_dir / "digit.png").write_bytes(_png_bytes(size=(64, 80)))

    node = ImageReaderNode()
    result = node.execute({}, {"path": "digit.png", "mode": "L", "resize": 28})
    img = result["image"]  # (1, 28, 28)
    batched = img.unsqueeze(0)  # (1, 1, 28, 28)

    conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    out = conv(batched)
    assert out.shape == (1, 32, 28, 28)
