"""PreferenceDatasetNode -- synthetic preference pairs, with a shortcut planted in them.

RLHF's stage 2 needs pairs: a prompt, two answers, and which one a human
picked. There was no node producing them, so the chapter example faked the
whole stage with a ``TensorInput``.

Each item is a feature vector standing in for "what a response looks like".
Its **true quality** is a weighted sum over the first ``signal_dims``
coordinates -- spread thin on purpose, so no single feature gives it away. A
pair is two items, and the label is simply which one has the higher true
quality. That much would make a well-behaved dataset.

The point of this node is the part that is not well-behaved. One coordinate is
a **shortcut**: in the training split it is manufactured to track true quality
closely, and in the holdout split it is pure noise. Nothing about the two
splits differs otherwise.

That asymmetry is what makes reward hacking reproducible instead of anecdotal.
A reward model can fit the label two ways: read the diffuse true signal, or
read the one loud coordinate. The shortcut is a single strong feature, so
gradient descent reaches it first, and the model arrives at a perfect training
score having learned the wrong thing.

The measurement that shows this is the **gap**, not a curve. At default size
the model fits the training split to 1.000 either way, so the training number
is blind by construction -- which is the lesson. Holding everything else
fixed and turning ``shortcut_strength`` from 0 to 3 moves holdout accuracy
from ~0.94 to ~0.77 while training accuracy stays pinned at 1.000. Measured
across five seeds the two holdout bands do not overlap.

Deliberately not claimed: a train/holdout divergence *over epochs*. At this
scale the fit is essentially immediate, there is no "learns it right, then
learns it wrong" phase to watch, and the peak-to-final drop is smaller than
the seed-to-seed spread. Real RLHF grows that curve; 512 synthetic pairs do
not. ``shortcut_strength = 0`` is the control that attributes the gap to the
shortcut rather than to ordinary overfitting.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class PreferenceDatasetNode(BaseNode):
    NODE_NAME = "PreferenceDataset"
    CATEGORY = "RL"
    DESCRIPTION = (
        "Synthetic RLHF preference pairs: two feature vectors and which one is better. "
        "True quality is spread thinly over several coordinates; one coordinate is a "
        "SHORTCUT that tracks quality in the training split and is pure noise in the "
        "holdout. A reward model finds the loud shortcut before the diffuse signal, so it "
        "scores a PERFECT 1.000 on the training pairs while holdout accuracy drops -- "
        "reward hacking that only the held-out split can see. Set shortcut_strength to 0 "
        "for the control."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="train_w", data_type=DataType.TENSOR, description="[n_pairs, feature_dim] the preferred response of each training pair."),
            PortDefinition(name="train_l", data_type=DataType.TENSOR, description="[n_pairs, feature_dim] the rejected response of each training pair."),
            PortDefinition(name="holdout_w", data_type=DataType.TENSOR, description="[holdout_pairs, feature_dim] preferred, held-out split."),
            PortDefinition(name="holdout_l", data_type=DataType.TENSOR, description="[holdout_pairs, feature_dim] rejected, held-out split."),
            PortDefinition(name="feature_dim", data_type=DataType.SCALAR, description="Feature width -- set the reward model's input_dim to this."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="n_pairs", param_type=ParamType.INT, default=512, min_value=1, description="Training pairs."),
            ParamDefinition(name="holdout_pairs", param_type=ParamType.INT, default=256, min_value=1, description="Held-out pairs, used only to measure."),
            ParamDefinition(name="feature_dim", param_type=ParamType.INT, default=16, min_value=2, description="Feature width of one response."),
            ParamDefinition(
                name="signal_dims",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="How many coordinates carry the true quality. More = more diffuse = harder than the shortcut.",
            ),
            ParamDefinition(
                name="shortcut_strength",
                param_type=ParamType.FLOAT,
                default=3.0,
                min_value=0.0,
                description=(
                    "How loud the shortcut coordinate is in the TRAINING split. 0 removes it "
                    "entirely -- the control run, where holdout accuracy stays high."
                ),
            ),
            ParamDefinition(name="seed", param_type=ParamType.INT, default=0, description="Reproducibility."),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        n = int(params.get("n_pairs", 512))
        m = int(params.get("holdout_pairs", 256))
        d = int(params.get("feature_dim", 16))
        k = int(params.get("signal_dims", 8))
        strength = float(params.get("shortcut_strength", 3.0))
        gen = torch.Generator().manual_seed(int(params.get("seed", 0)))

        if k >= d:
            raise ValueError(
                f"PreferenceDataset: signal_dims ({k}) must be smaller than feature_dim ({d}) "
                "-- the last coordinate is reserved for the shortcut."
            )

        # One fixed direction defines "quality" for both splits, so the two are
        # the same problem and any gap between them is about the shortcut only.
        weights = torch.randn(k, generator=gen)

        def make(count: int, planted: bool):
            a = torch.randn(count, d, generator=gen)
            b = torch.randn(count, d, generator=gen)
            qa = a[:, :k] @ weights
            qb = b[:, :k] @ weights
            # The shortcut coordinate: quality plus a little noise in the
            # training split, independent noise in the holdout.
            if planted and strength > 0:
                a[:, -1] = strength * qa + torch.randn(count, generator=gen) * 0.1
                b[:, -1] = strength * qb + torch.randn(count, generator=gen) * 0.1
            else:
                a[:, -1] = torch.randn(count, generator=gen)
                b[:, -1] = torch.randn(count, generator=gen)
            better_is_a = qa > qb
            winner = torch.where(better_is_a.unsqueeze(1), a, b)
            loser = torch.where(better_is_a.unsqueeze(1), b, a)
            return winner, loser

        train_w, train_l = make(n, planted=True)
        holdout_w, holdout_l = make(m, planted=False)

        return {
            "train_w": train_w,
            "train_l": train_l,
            "holdout_w": holdout_w,
            "holdout_l": holdout_l,
            "feature_dim": d,
        }
