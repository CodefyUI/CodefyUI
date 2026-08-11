"""VideoWriteNode — turn a frames tensor into a playable clip (#310).

The canvas's first video producer: rollout recordings, diffusion sampling
chains, augmentation previews — anything shaped ``(T, C, H, W)`` or
``(T, H, W, C)`` becomes an mp4 (ffmpeg binary on PATH) or a gif (Pillow,
zero dependencies), written under ``settings.MEDIA_DIR`` and served inline
by ``/api/media``. The ``video`` output is a small REFERENCE dict — the
event stream's 128 KB cap is two orders of magnitude under a clip, so the
bytes never ride the wire (see MEDIA_VIDEO in node_base).
"""

import logging
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    MEDIA_IMAGE,
    MEDIA_VIDEO,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

logger = logging.getLogger(__name__)

#: Longest side of the base64 PNG preview frame. The preview rides the
#: event stream, so it must stay far under the 128 KB payload cap even
#: when sharing the budget with the video entry and a tensor summary.
_PREVIEW_MAX_SIDE = 256


class VideoWriteNode(BaseNode):
    NODE_NAME = "VideoWrite"
    CATEGORY = "IO"
    DESCRIPTION = (
        "Encode a frames tensor (T,C,H,W) or (T,H,W,C) into a playable "
        "video under the media directory - mp4 via the ffmpeg binary when "
        "one is on PATH, gif via Pillow with no dependency at all - and "
        "emit a reference the editor plays inline."
    )

    # The write to disk IS this node's output (the ImageWriter rule, #143):
    # a cache hit would hand back a reference whose file may no longer
    # exist, so the file must be (re)written every run.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="frames",
                data_type=DataType.TENSOR,
                description=(
                    "Video frames: (T,C,H,W) or (T,H,W,C), float [0,1] or "
                    "uint8, gray or RGB"
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="video",
                data_type=DataType.ANY,
                description="Playable-clip reference (path, url, format, fps, frames)",
                media=MEDIA_VIDEO,
            ),
            PortDefinition(
                name="path",
                data_type=DataType.STRING,
                description="Absolute path of the written file",
            ),
            PortDefinition(
                name="preview",
                data_type=DataType.STRING,
                description="Middle frame as a PNG, so the clip has a face "
                            "even where video is not rendered yet",
                media=MEDIA_IMAGE,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="filename",
                param_type=ParamType.STRING,
                default="clip",
                description=(
                    "Name under the media directory (subfolders allowed); "
                    "the extension follows the chosen format. Same name "
                    "overwrites - vary it per run to keep several clips."
                ),
            ),
            ParamDefinition(
                name="format",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "mp4", "gif"],
                description=(
                    "auto: mp4 when an ffmpeg binary is on PATH, else gif. "
                    "gif always works (Pillow); mp4 needs ffmpeg installed."
                ),
            ),
            ParamDefinition(
                name="fps",
                param_type=ParamType.FLOAT,
                default=10.0,
                min_value=0.5,
                max_value=60.0,
                description="Playback frames per second",
            ),
            ParamDefinition(
                name="resize",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Output height in pixels, aspect kept, nearest-neighbor "
                    "(0 = native). Small research renders (96px) are easier "
                    "to watch at 2-3x."
                ),
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import base64
        import io

        import torch
        from PIL import Image

        from ...config import settings
        from ...core.data_paths import resolve_data_path
        from ...core.video_io import ffmpeg_path, frames_to_uint8_thwc, write_gif, write_mp4

        frames = frames_to_uint8_thwc(inputs["frames"])
        if frames.shape[0] == 0:
            raise ValueError("VideoWrite: frames tensor has zero frames")

        fps = float(params.get("fps", 10.0) or 10.0)
        resize = max(0, int(params.get("resize", 0) or 0))
        if resize and resize != frames.shape[1]:
            scale = resize / frames.shape[1]
            width = max(2, round(frames.shape[2] * scale))
            resized = torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2).float(),
                size=(resize, width), mode="nearest")
            frames = resized.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

        fmt = str(params.get("format", "auto") or "auto")
        if fmt == "auto":
            fmt = "mp4" if ffmpeg_path() else "gif"
        if fmt == "mp4":
            # write_mp4 crops odd dimensions for yuv420p; do it HERE so the
            # reference dict below describes the file actually written.
            height = frames.shape[1] - (frames.shape[1] % 2)
            width = frames.shape[2] - (frames.shape[2] % 2)
            frames = frames[:, :height, :width, :]

        media_dir = settings.MEDIA_DIR.resolve()
        media_dir.mkdir(parents=True, exist_ok=True)
        filename = str(params.get("filename", "clip") or "clip")
        p = resolve_data_path(filename, base=media_dir)
        expected_ext = "." + fmt
        if p.suffix.lower() != expected_ext:
            # The path written must be the path checked (the ImageWriter
            # extension-rewrite rule): re-validate after forcing the suffix.
            p = resolve_data_path(p.with_suffix(expected_ext), base=media_dir)
        # resolve_data_path guarantees the data ROOT; the /api/media URL
        # contract additionally needs the file under MEDIA_DIR itself.
        if not p.is_relative_to(media_dir):
            raise ValueError(
                "VideoWrite: filename must stay under the media directory "
                f"({media_dir}), got {p}")
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "gif":
            write_gif(frames, p, fps)
        else:
            write_mp4(frames, p, fps)

        preview = frames[frames.shape[0] // 2]
        image = Image.fromarray(preview.numpy())
        if max(image.size) > _PREVIEW_MAX_SIDE:
            image.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE))
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        relative = p.relative_to(media_dir).as_posix()
        size = p.stat().st_size
        reference = {
            "path": relative,
            "url": f"/api/media/{relative}",
            "format": fmt,
            "fps": fps,
            "frames": int(frames.shape[0]),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "bytes": size,
        }
        logger.info("Wrote %s (%d frames, %d bytes)", p, frames.shape[0], size)
        return {
            "video": reference,
            "path": str(p),
            "preview": base64.b64encode(buf.getvalue()).decode("ascii"),
            "__log__": (
                f"Wrote {relative} ({fmt}, {frames.shape[0]} frames at "
                f"{fps:g} fps, {size:,} bytes)."
            ),
        }
