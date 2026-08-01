"""Tests for the model files API (list, upload, download, delete)."""

import pytest

from app.core.output_entries import _summarize_single


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Redirect settings.MODELS_DIR at a temp dir for each test."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr("app.config.settings.MODELS_DIR", d)
    return d


@pytest.mark.asyncio
async def test_upload_and_download_roundtrip(test_client, models_dir):
    payload = b"\x00\x01\x02PYTORCH_WEIGHTS"
    resp = await test_client.post(
        "/api/models/upload",
        files={"file": ("roundtrip.pt", payload, "application/octet-stream")},
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] == "roundtrip.pt"

    resp = await test_client.get("/api/models/download/roundtrip.pt")
    assert resp.status_code == 200
    assert resp.content == payload
    assert resp.headers["content-type"] == "application/octet-stream"
    # FileResponse sets Content-Disposition with the filename
    assert "roundtrip.pt" in resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_download_missing_file_returns_404(test_client, models_dir):
    resp = await test_client.get("/api/models/download/does_not_exist.pt")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_path_traversal_rejected(test_client, models_dir, tmp_path):
    # A file outside models_dir that we should never be able to reach
    secret = tmp_path / "secret.pt"
    secret.write_bytes(b"secret")

    resp = await test_client.get("/api/models/download/..%2Fsecret.pt")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_download_non_model_extension_rejected(test_client, models_dir):
    (models_dir / "readme.txt").write_bytes(b"not a model")
    resp = await test_client.get("/api/models/download/readme.txt")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_download_nested_subdir_path(test_client, models_dir):
    nested = models_dir / "runs" / "exp1"
    nested.mkdir(parents=True)
    payload = b"NESTED_WEIGHTS"
    (nested / "model.pt").write_bytes(payload)

    resp = await test_client.get("/api/models/download/runs/exp1/model.pt")
    assert resp.status_code == 200
    assert resp.content == payload


def test_summarize_string_under_models_dir_gets_download_path(models_dir):
    target = models_dir / "out.pt"
    target.write_bytes(b"xx")
    summary = _summarize_single(str(target))
    assert summary["type"] == "string"
    assert summary["download_path"] == "out.pt"


def test_summarize_string_nested_under_models_dir(models_dir):
    nested = models_dir / "runs" / "exp1"
    nested.mkdir(parents=True)
    (nested / "model.pt").write_bytes(b"yy")
    summary = _summarize_single(str(nested / "model.pt"))
    assert summary["download_path"] == "runs/exp1/model.pt"


def test_summarize_string_outside_models_dir_has_no_download_path(models_dir, tmp_path):
    outside = tmp_path / "other.pt"
    outside.write_bytes(b"zz")
    summary = _summarize_single(str(outside))
    assert "download_path" not in summary


def test_summarize_string_nonfile_value_has_no_download_path(models_dir):
    summary = _summarize_single("just a label, not a path")
    assert "download_path" not in summary


@pytest.mark.asyncio
async def test_list_and_delete(test_client, models_dir):
    (models_dir / "a.pt").write_bytes(b"aa")
    (models_dir / "b.safetensors").write_bytes(b"bb")

    resp = await test_client.get("/api/models")
    assert resp.status_code == 200
    names = {item["filename"] for item in resp.json()}
    assert names == {"a.pt", "b.safetensors"}

    resp = await test_client.delete("/api/models/a.pt")
    assert resp.status_code == 200
    assert not (models_dir / "a.pt").exists()
