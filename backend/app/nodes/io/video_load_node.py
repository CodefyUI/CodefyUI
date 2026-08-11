"""VideoLoadNode — decode a video file into a frames tensor (#310).

The ingestion half of video support: rollout clips written by VideoWrite,
demonstration recordings, any mp4/webm/gif on disk become ``(T, 3, H, W)``
float [0,1] for whatever the graph does next (dataset building, frame
analysis, re-encoding a subsample). GIF decodes through Pillow with no
dependency; mp4/webm shell out to the ffmpeg binary (see core.video_io for
why there is no Python video library here).
"""

import logging
from pathlib import Path
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class VideoLoadNode(BaseNode):
    NODE_NAME = "VideoLoad"
    CATEGORY = "IO"
    DESCRIPTION = (
        "Decode a video file (mp4/webm via the ffmpeg binary, gif via "
        "Pillow) into a frames tensor (T,3,H,W) float [0,1], with fps and "
        "frame count. Relative paths resolve under the media directory, "
        "where VideoWrite puts its clips."
    )

    # The output mirrors a file that can change or vanish between runs, and
    # nothing fingerprints its content -- a cache hit could describe a video
    # that is no longer on disk (the ImageWriter/#143 shape of the rule,
    # from the reading side).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="path",
                data_type=DataType.STRING,
                description="Video file path (overrides the path param when wired)",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="frames",
                data_type=DataType.TENSOR,
                description="Decoded frames (T, 3, H, W), float [0,1]",
            ),
            PortDefinition(
                name="fps",
                data_type=DataType.SCALAR,
                description="Frames per second after stride",
            ),
            PortDefinition(
                name="num_frames",
                data_type=DataType.SCALAR,
                description="Number of decoded frames",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="clip.mp4",
                description=(
                    "Video file: absolute, or relative to the media "
                    "directory (VideoWrite's output lands there)"
                ),
            ),
            ParamDefinition(
                name="max_frames",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Stop after this many decoded frames (0 = all). An "
                    "uncapped long video is materialized whole in memory."
                ),
            ),
            ParamDefinition(
                name="stride",
                param_type=ParamType.INT,
                default=1,
                min_value=1,
                description="Keep every Nth frame (fps output scales down to match)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from ...config import settings
        from ...core.video_io import read_gif, read_video_ffmpeg

        raw = inputs.get("path") or params.get("path", "")
        if not raw or not str(raw).strip():
            raise ValueError("VideoLoad: no path given")
        path = Path(str(raw))
        if not path.is_absolute():
            path = settings.MEDIA_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"VideoLoad: file not found: {path}")

        max_frames = max(0, int(params.get("max_frames", 0) or 0))
        stride = max(1, int(params.get("stride", 1) or 1))

        if path.suffix.lower() == ".gif":
            frames, fps = read_gif(path, max_frames=max_frames, stride=stride)
        else:
            frames, fps = read_video_ffmpeg(path, max_frames=max_frames, stride=stride)

        logger.info("Decoded %s: %d frames at %.2f fps", path, frames.shape[0], fps)
        return {
            "frames": frames,
            "fps": float(fps),
            "num_frames": int(frames.shape[0]),
            "__log__": (
                f"Decoded {path.name}: {frames.shape[0]} frames "
                f"({frames.shape[3]}x{frames.shape[2]}) at {fps:.2f} fps."
            ),
        }
