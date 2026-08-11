"""Video encode/decode helpers shared by VideoWrite / VideoLoad (#310).

There is deliberately no Python video library behind this module. torchvision
removed its video API in 0.26 (pointing at torchcodec), and pulling in PyAV or
imageio-ffmpeg for one feature is against the house rule that turning a
feature on should cost an install nothing (the tensorboard precedent). So:

- GIF is the ZERO-DEPENDENCY path, both directions, via Pillow — always
  available, plays in every ``<img>`` tag.
- MP4 uses the ``ffmpeg`` BINARY over a rawvideo pipe when one is on PATH —
  no Python dependency, and the error when it is missing names the gif
  fallback. ffmpeg is the one binary every machine doing video work already
  has; we shell out to it rather than vendor a decoder.

Frames move through this module in one canonical shape: ``(T, H, W, 3)``
uint8, produced by :func:`frames_to_uint8_thwc` from whatever a graph hands
us — ``(T, C, H, W)`` or ``(T, H, W, C)``, float [0,1] or uint8, gray or RGB.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

#: Read/write timeout for one ffmpeg invocation. Encoding a research-scale
#: clip (a few hundred small frames) is sub-second; this is a hang guard,
#: not a budget.
_FFMPEG_TIMEOUT_S = 120


def ffmpeg_path() -> str | None:
    """Absolute path of the ffmpeg binary, or None when not installed."""
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def frames_to_uint8_thwc(frames: Any) -> Any:
    """Normalize a frames tensor to ``(T, H, W, 3)`` uint8.

    Accepts ``(T, C, H, W)`` or ``(T, H, W, C)`` with C in {1, 3}, float
    (clamped to [0,1]) or uint8. Layout is decided by which axis holds a
    channel count; when BOTH axis 1 and axis 3 could be channels (e.g.
    ``(T, 3, H, 3)``) the torch-conventional ``(T, C, H, W)`` wins.
    """
    import torch

    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be a tensor, got {type(frames).__name__}")
    if frames.dim() != 4:
        raise ValueError(
            f"frames must be 4D (T, C, H, W) or (T, H, W, C), got shape "
            f"{tuple(frames.shape)}")
    t = frames.detach().cpu()
    if t.shape[1] in (1, 3):
        t = t.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
    elif t.shape[3] not in (1, 3):
        raise ValueError(
            f"neither axis 1 nor axis 3 is a channel count in shape "
            f"{tuple(frames.shape)}")
    if t.shape[3] == 1:
        t = t.expand(-1, -1, -1, 3)
    if t.dtype != torch.uint8:
        t = (t.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return t.contiguous()


def write_gif(frames_thwc: Any, path: Path, fps: float) -> None:
    from PIL import Image

    duration_ms = max(20, round(1000.0 / max(0.1, fps)))
    images = [Image.fromarray(frame.numpy()) for frame in frames_thwc]
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=duration_ms, loop=0)


def write_mp4(frames_thwc: Any, path: Path, fps: float) -> None:
    """Encode via ffmpeg's rawvideo stdin pipe (h264, yuv420p, faststart)."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError(
            "MP4 output needs the ffmpeg binary on PATH and none was found. "
            "Install ffmpeg, or set format=gif — the gif path has no "
            "external dependency.")
    # yuv420p subsamples chroma 2x2, so h264 refuses odd dimensions; a 1px
    # crop is invisible at these sizes and keeps the caller's frames intact.
    height = frames_thwc.shape[1] - (frames_thwc.shape[1] % 2)
    width = frames_thwc.shape[2] - (frames_thwc.shape[2] % 2)
    frames = frames_thwc[:, :height, :width, :]
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.run(
        cmd, input=frames.numpy().tobytes(),
        capture_output=True, timeout=_FFMPEG_TIMEOUT_S)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {tail}")


def read_gif(path: Path, max_frames: int = 0, stride: int = 1) -> tuple[Any, float]:
    """Decode a GIF -> ``((T, 3, H, W) float [0,1], fps)`` via Pillow."""
    import numpy as np
    import torch
    from PIL import Image, ImageSequence

    stride = max(1, stride)
    frames: list[Any] = []
    durations: list[int] = []
    with Image.open(path) as image:
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            durations.append(int(frame.info.get("duration", 0)) or 100)
            if index % stride != 0:
                continue
            frames.append(np.asarray(frame.convert("RGB")))
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"no frames decoded from {path.name}")
    stacked = torch.from_numpy(np.stack(frames))  # (T, H, W, 3) uint8
    fps = 1000.0 / (sum(durations) / len(durations))
    # Same convention as read_video_ffmpeg: fps describes the RETURNED
    # sequence, so striding scales it down and playback duration holds.
    return stacked.permute(0, 3, 1, 2).float() / 255.0, fps / stride


def _probe(path: Path) -> tuple[int, int, float]:
    """(width, height, fps) of the first video stream, via ffprobe."""
    import json

    ffprobe = ffprobe_path()
    if ffprobe is None:
        raise RuntimeError(
            "Reading this container needs the ffprobe binary on PATH "
            "(ships with ffmpeg) and none was found. Install ffmpeg, or "
            "use gif files — the gif path has no external dependency.")
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_S)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise RuntimeError(f"ffprobe failed on {path.name}: {tail}")
    streams = json.loads(proc.stdout.decode("utf-8", "replace")).get("streams") or []
    if not streams:
        raise ValueError(f"{path.name} has no video stream")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    try:
        fps = float(Fraction(stream.get("r_frame_rate", "10/1")))
    except (ValueError, ZeroDivisionError):
        fps = 10.0
    return width, height, fps


def read_video_ffmpeg(
    path: Path, max_frames: int = 0, stride: int = 1,
) -> tuple[Any, float]:
    """Decode mp4/webm -> ``((T, 3, H, W) float [0,1], fps)`` via ffmpeg.

    Streams rawvideo from ffmpeg's stdout frame by frame and stops as soon
    as ``max_frames`` are kept, so a long video with a cap never buffers
    whole; without a cap the full clip is materialized in memory — that is
    the caller's stated intent.
    """
    import numpy as np
    import torch

    width, height, fps = _probe(path)
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:  # ffprobe found but ffmpeg missing: broken install
        raise RuntimeError("ffprobe is on PATH but ffmpeg is not")
    stride = max(1, stride)
    cmd = [
        ffmpeg, "-loglevel", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = width * height * 3
    kept: list[Any] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        index = 0
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            if index % stride == 0:
                kept.append(np.frombuffer(chunk, dtype=np.uint8).reshape(
                    height, width, 3).copy())
                if max_frames and len(kept) >= max_frames:
                    proc.terminate()
                    break
            index += 1
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read()
        proc.stderr.close()
        returncode = proc.wait(timeout=_FFMPEG_TIMEOUT_S)
    if not kept:
        tail = stderr.decode("utf-8", "replace").strip()[-300:]
        raise ValueError(
            f"no frames decoded from {path.name}"
            + (f" (ffmpeg exit {returncode}: {tail})" if returncode else ""))
    stacked = torch.from_numpy(np.stack(kept))
    return stacked.permute(0, 3, 1, 2).float() / 255.0, fps / stride
