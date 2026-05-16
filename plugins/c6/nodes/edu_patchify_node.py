"""EduPatchifyNode — split an image into patches for ViT-style processing.

Supports textbook lesson **C6-2 (ViT)**: turn a ``[B, C, H, W]`` image into a
sequence of patch tokens ``[B, N_patches, C × P × P]`` so the rest of the
graph (linear projection, position embedding, transformer encoder) treats
each patch like a word.

The default settings reproduce the original ViT recipe: square non-overlapping
patches whose side is ``patch_size`` pixels. ``H`` and ``W`` must each be
divisible by ``patch_size``.
"""

from __future__ import annotations

from typing import Any

import torch

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduPatchifyNode(BaseNode):
    NODE_NAME = "EduPatchify"
    CATEGORY = "Transformer"
    DESCRIPTION = (
        "Split a [B, C, H, W] image into a sequence of P×P patches: "
        "[B, N_patches, C·P·P]. The unfold → permute → flatten chain that "
        "turns 'an image' into 'a sequence of tokens' is exposed step by step."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="image",
                data_type=DataType.TENSOR,
                description="Image batch of shape [B, C, H, W]. H and W must be divisible by patch_size.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tokens",
                data_type=DataType.TENSOR,
                description="Patch tokens, shape [B, N_patches, C·P·P] when flatten=True, else [B, N_patches, C, P, P].",
            ),
            PortDefinition(
                name="grid",
                data_type=DataType.TENSOR,
                description="Patch grid sizes as a 2-tensor [grid_h, grid_w].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="patch_size",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Side length of one square patch in pixels.",
            ),
            ParamDefinition(
                name="flatten",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "If true (the ViT default) each patch is flattened to a "
                    "1D vector. Set false to keep the [C, P, P] structure for "
                    "lessons that visualise individual patches."
                ),
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x = inputs.get("image")
        if x is None:
            raise ValueError("EduPatchify requires an `image` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        # Accept [C,H,W] by promoting to [1,C,H,W]; reject anything else.
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(
                f"EduPatchify expects [B, C, H, W] (or [C, H, W]); got shape {tuple(x.shape)}."
            )

        B, C, H, W = x.shape
        patch_size = int(params.get("patch_size", 8))
        flatten = bool(params.get("flatten", True))
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"EduPatchify: H={H}, W={W} must both be divisible by patch_size={patch_size}."
            )

        grid_h = H // patch_size
        grid_w = W // patch_size
        n_patches = grid_h * grid_w

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        if recorder is not None:
            recorder.record(
                "input",
                "Start: an image batch.",
                scalars={
                    "B": float(B), "C": float(C), "H": float(H), "W": float(W),
                    "patch_size": float(patch_size),
                    "grid_h": float(grid_h), "grid_w": float(grid_w),
                    "n_patches": float(n_patches),
                },
                image=x,
            )

        # unfold(H, patch_size) creates a sliding window — with stride=patch_size
        # the windows are non-overlapping, giving us exactly grid_h windows.
        unfolded_h = x.unfold(2, patch_size, patch_size)
        # shape after H unfold: [B, C, grid_h, W, patch_size]
        unfolded = unfolded_h.unfold(3, patch_size, patch_size)
        # shape now: [B, C, grid_h, grid_w, patch_size, patch_size]
        if recorder is not None:
            recorder.record(
                "unfold",
                "Slide a P×P window across H and W with stride P → "
                "shape [B, C, grid_h, grid_w, P, P].",
                unfolded=unfolded,
            )

        # Re-arrange channels-after-patch-position so each patch is contiguous.
        permuted = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
        # shape: [B, grid_h, grid_w, C, P, P]
        if recorder is not None:
            recorder.record(
                "permute",
                "Move grid axes in front of channels — patches are now contiguous.",
                permuted=permuted,
            )

        if flatten:
            tokens = permuted.view(B, n_patches, C * patch_size * patch_size)
            if recorder is not None:
                recorder.record(
                    "flatten",
                    "Flatten each patch (C·P·P numbers) into a single token vector.",
                    tokens=tokens,
                )
        else:
            tokens = permuted.view(B, n_patches, C, patch_size, patch_size)
            if recorder is not None:
                recorder.record(
                    "reshape",
                    "Keep the [C, P, P] shape per patch (useful for visualization).",
                    tokens=tokens,
                )

        grid = torch.tensor([grid_h, grid_w], dtype=torch.long)
        result: dict[str, Any] = {"tokens": tokens, "grid": grid}
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
