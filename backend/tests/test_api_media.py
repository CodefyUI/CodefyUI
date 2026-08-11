"""The /api/media route: inline playback semantics (#310).

What distinguishes this route from the download routes is the Content-Type —
a browser will only PLAY a clip served with its real mime, so that header is
the contract under test, alongside the same traversal guard every file route
carries.
"""

import pytest


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    d = tmp_path / "media"
    d.mkdir()
    monkeypatch.setattr("app.config.settings.MEDIA_DIR", d)
    return d


@pytest.mark.asyncio
async def test_serves_gif_inline_with_real_mime(test_client, media_dir):
    payload = b"GIF89a-not-really-but-served-verbatim"
    (media_dir / "clip.gif").write_bytes(payload)

    resp = await test_client.get("/api/media/clip.gif")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/gif"
    assert resp.content == payload
    # inline playback, not a download
    assert "attachment" not in resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_serves_mp4_from_a_subfolder(test_client, media_dir):
    sub = media_dir / "rollouts"
    sub.mkdir()
    (sub / "run1.mp4").write_bytes(b"\x00\x00\x00 ftypisom")

    resp = await test_client.get("/api/media/rollouts/run1.mp4")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"


@pytest.mark.asyncio
async def test_missing_file_is_404(test_client, media_dir):
    resp = await test_client.get("/api/media/absent.mp4")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_path_traversal_is_refused(test_client, media_dir):
    resp = await test_client.get("/api/media/..%2Fsecret.mp4")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_unplayable_extension_is_refused(test_client, media_dir):
    (media_dir / "notes.txt").write_bytes(b"text")
    resp = await test_client.get("/api/media/notes.txt")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_walks_subfolders_with_posix_names(test_client, media_dir):
    (media_dir / "a.gif").write_bytes(b"x")
    sub = media_dir / "deep"
    sub.mkdir()
    (sub / "b.mp4").write_bytes(b"y")
    (media_dir / "ignored.txt").write_bytes(b"z")

    resp = await test_client.get("/api/media")
    assert resp.status_code == 200
    names = {item["filename"] for item in resp.json()}
    assert names == {"a.gif", "deep/b.mp4"}
