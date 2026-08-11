"""VideoWrite / VideoLoad and the MEDIA_VIDEO wire contract (#310).

The gif tests are the zero-dependency path and always run; mp4 tests skip
cleanly on machines without an ffmpeg binary — CI must stay green either way.
"""

import base64
import io

import pytest
import torch
from PIL import Image

from app.core.output_entries import build_node_output_entries
from app.core.video_io import ffmpeg_path, frames_to_uint8_thwc
from app.nodes.io.video_load_node import VideoLoadNode
from app.nodes.io.video_write_node import VideoWriteNode

needs_ffmpeg = pytest.mark.skipif(
    ffmpeg_path() is None, reason="no ffmpeg binary on PATH")


@pytest.fixture
def media_dirs(tmp_path, monkeypatch):
    """Point MEDIA_DIR (and the data root around it) at a temp tree."""
    data = tmp_path / "data"
    models = data / "models"
    media = data / "media"
    models.mkdir(parents=True)
    media.mkdir(parents=True)
    monkeypatch.setattr("app.config.settings.MODELS_DIR", models)
    monkeypatch.setattr("app.config.settings.MEDIA_DIR", media)
    monkeypatch.setattr("app.config.settings.DB_PATH", data / "codefyui.db")
    return media


def _frames(t=6, h=32, w=48) -> torch.Tensor:
    """(T, 3, H, W) float ramp — distinct per frame so striding is testable."""
    base = torch.linspace(0.0, 1.0, t).view(t, 1, 1, 1)
    return base.expand(t, 3, h, w).clone()


# ── frames_to_uint8_thwc ────────────────────────────────────────────────


def test_layout_tchw_becomes_thwc_uint8():
    out = frames_to_uint8_thwc(_frames())
    assert out.shape == (6, 32, 48, 3)
    assert out.dtype == torch.uint8
    assert out[0].float().mean() == 0.0
    assert out[-1].float().mean() == 255.0


def test_layout_thwc_uint8_passes_through():
    raw = torch.randint(0, 256, (4, 20, 30, 3), dtype=torch.uint8)
    out = frames_to_uint8_thwc(raw)
    assert out.shape == (4, 20, 30, 3)
    assert torch.equal(out, raw)


def test_grayscale_channel_expands_to_rgb():
    out = frames_to_uint8_thwc(torch.rand(3, 1, 16, 16))
    assert out.shape == (3, 16, 16, 3)
    assert torch.equal(out[..., 0], out[..., 1])


def test_non_4d_frames_are_refused():
    with pytest.raises(ValueError, match="4D"):
        frames_to_uint8_thwc(torch.rand(3, 16, 16))


# ── VideoWrite (gif path — no dependency) ───────────────────────────────


def test_video_write_gif_writes_file_and_reference(media_dirs):
    node = VideoWriteNode()
    result = node.execute(
        {"frames": _frames()},
        {"filename": "clips/test", "format": "gif", "fps": 5.0, "resize": 0},
    )
    ref = result["video"]
    assert ref["format"] == "gif"
    assert ref["path"] == "clips/test.gif"
    assert ref["url"] == "/api/media/clips/test.gif"
    assert ref["frames"] == 6
    assert ref["width"] == 48 and ref["height"] == 32
    assert (media_dirs / "clips" / "test.gif").exists()
    assert ref["bytes"] == (media_dirs / "clips" / "test.gif").stat().st_size
    # the preview is a decodable PNG of the middle frame
    png = base64.b64decode(result["preview"])
    assert Image.open(io.BytesIO(png)).size == (48, 32)


def test_video_write_resize_scales_height_keeping_aspect(media_dirs):
    node = VideoWriteNode()
    result = node.execute(
        {"frames": _frames(h=32, w=48)},
        {"filename": "small", "format": "gif", "fps": 5.0, "resize": 64},
    )
    assert result["video"]["height"] == 64
    assert result["video"]["width"] == 96


def test_video_write_refuses_escaping_media_dir(media_dirs):
    node = VideoWriteNode()
    with pytest.raises(ValueError, match="media directory"):
        node.execute(
            {"frames": _frames()},
            {"filename": "../escape", "format": "gif", "fps": 5.0},
        )


def test_video_write_refuses_empty_frames(media_dirs):
    with pytest.raises(ValueError, match="zero frames"):
        VideoWriteNode().execute(
            {"frames": torch.zeros(0, 3, 8, 8)},
            {"filename": "empty", "format": "gif", "fps": 5.0},
        )


# ── VideoLoad (gif path) ────────────────────────────────────────────────


def test_gif_roundtrip_preserves_count_shape_and_fps(media_dirs):
    VideoWriteNode().execute(
        {"frames": _frames()},
        {"filename": "loop", "format": "gif", "fps": 10.0},
    )
    result = VideoLoadNode().execute({}, {"path": "loop.gif", "max_frames": 0, "stride": 1})
    frames = result["frames"]
    assert frames.shape == (6, 3, 32, 48)
    assert frames.dtype == torch.float32
    assert 0.0 <= frames.min() and frames.max() <= 1.0
    assert result["num_frames"] == 6
    assert result["fps"] == pytest.approx(10.0, rel=0.05)


def test_video_load_stride_and_max_frames(media_dirs):
    VideoWriteNode().execute(
        {"frames": _frames(t=8)},
        {"filename": "strided", "format": "gif", "fps": 10.0},
    )
    result = VideoLoadNode().execute(
        {}, {"path": "strided.gif", "max_frames": 3, "stride": 2})
    assert result["num_frames"] == 3
    assert result["fps"] == pytest.approx(5.0, rel=0.05)


def test_video_load_input_port_overrides_param(media_dirs):
    write = VideoWriteNode().execute(
        {"frames": _frames()},
        {"filename": "wired", "format": "gif", "fps": 5.0},
    )
    result = VideoLoadNode().execute(
        {"path": write["path"]}, {"path": "does-not-exist.gif"})
    assert result["num_frames"] == 6


def test_video_load_missing_file_is_a_clear_error(media_dirs):
    with pytest.raises(FileNotFoundError, match="not found"):
        VideoLoadNode().execute({}, {"path": "nope.gif"})


# ── mp4 path (needs the ffmpeg binary) ──────────────────────────────────


@needs_ffmpeg
def test_mp4_roundtrip(media_dirs):
    write = VideoWriteNode().execute(
        {"frames": _frames(t=10, h=32, w=48)},
        {"filename": "clip", "format": "mp4", "fps": 10.0},
    )
    assert write["video"]["format"] == "mp4"
    assert (media_dirs / "clip.mp4").exists()
    result = VideoLoadNode().execute({}, {"path": "clip.mp4"})
    assert result["num_frames"] == 10
    assert result["frames"].shape == (10, 3, 32, 48)
    assert result["fps"] == pytest.approx(10.0, rel=0.05)


@needs_ffmpeg
def test_mp4_odd_dimensions_are_cropped_even(media_dirs):
    write = VideoWriteNode().execute(
        {"frames": _frames(t=4, h=33, w=49)},
        {"filename": "odd", "format": "mp4", "fps": 5.0},
    )
    assert write["video"]["height"] == 32
    assert write["video"]["width"] == 48


def test_format_auto_picks_a_working_encoder(media_dirs):
    result = VideoWriteNode().execute(
        {"frames": _frames()},
        {"filename": "auto", "format": "auto", "fps": 5.0},
    )
    expected = "mp4" if ffmpeg_path() else "gif"
    assert result["video"]["format"] == expected


# ── MEDIA_VIDEO wire contract ───────────────────────────────────────────


def _entries_for(result):
    return build_node_output_entries(
        "completed", result, {"video": ["video"], "image": ["preview"]})


def test_video_entry_reaches_the_wire(media_dirs):
    result = VideoWriteNode().execute(
        {"frames": _frames()},
        {"filename": "wire", "format": "gif", "fps": 5.0},
    )
    entries = _entries_for(result)
    video = [e for e in entries if e["output_kind"] == "video"]
    assert len(video) == 1
    assert video[0]["port"] == "video"
    assert video[0]["video"]["url"] == "/api/media/wire.gif"
    image = [e for e in entries if e["output_kind"] == "image"]
    assert len(image) == 1 and image[0]["port"] == "preview"


def test_video_payload_refuses_absolute_paths_and_junk():
    from app.core.output_entries import _video_payload

    assert _video_payload("not a dict") is None
    assert _video_payload({"format": "mp4"}) is None  # no path
    assert _video_payload({"path": "a.mp4"}) is None  # no format
    # rooted/escaping paths refused on EVERY platform, both path flavours
    assert _video_payload({"path": "C:/leak/a.mp4", "format": "mp4"}) is None
    assert _video_payload({"path": "C:leak.mp4", "format": "mp4"}) is None
    assert _video_payload({"path": "/leak/a.mp4", "format": "mp4"}) is None
    assert _video_payload({"path": "\\\\srv\\share\\a.mp4", "format": "mp4"}) is None
    assert _video_payload({"path": "../up.mp4", "format": "mp4"}) is None
    assert _video_payload({"path": "a/../../up.mp4", "format": "mp4"}) is None
    kept = _video_payload(
        {"path": "ok.mp4", "format": "mp4", "url": "/api/media/ok.mp4",
         "junk": "dropped", "fps": 10.0})
    assert kept == {"path": "ok.mp4", "format": "mp4",
                    "url": "/api/media/ok.mp4", "fps": 10.0}
