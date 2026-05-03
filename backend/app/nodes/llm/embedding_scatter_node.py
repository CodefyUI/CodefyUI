"""EmbeddingScatterNode — project high-dimensional embeddings down to 2D.

Used to *see* the geometry of the embedding space: words that cluster in the
same region share semantic features. PCA (linear, deterministic, fast) and
t-SNE (non-linear, stochastic, better at preserving local neighbourhoods)
are both wired in. UMAP can be added later via an optional dep group.

Output shape is always ``[N, 2]`` regardless of method — the inline scatter
visualization in ``EmbeddingScatterVizNode`` reads it directly off the
WebSocket output summary so no REST round-trip is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder


class EmbeddingScatterNode(BaseNode):
    NODE_NAME = "EmbeddingScatter"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Project an [N, D] embedding tensor down to [N, 2] for plotting. "
        "PCA finds the directions of maximum variance (linear, "
        "deterministic). t-SNE non-linearly preserves which points were "
        "neighbors in the original space, which often produces tighter "
        "semantic clusters at the cost of a stochastic layout."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embeddings",
                data_type=DataType.TENSOR,
                description="Float tensor of shape [N, D]. Typically the output of `WordVector`.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Optional list of length N — labels passed through to the output.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="points_2d",
                data_type=DataType.TENSOR,
                description="Float32 tensor of shape [N, 2] — projected coordinates, normalised to [-1, 1].",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Same labels passed through, so downstream nodes can match points to words.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="method",
                param_type=ParamType.SELECT,
                default="PCA",
                options=["PCA", "t-SNE"],
                description="PCA: linear, deterministic, fast. t-SNE: non-linear, preserves local neighbourhoods.",
            ),
            ParamDefinition(
                name="perplexity",
                param_type=ParamType.FLOAT,
                default=5.0,
                min_value=2.0,
                max_value=50.0,
                description="t-SNE only — neighbourhood size used for the local affinity model.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Random seed for stochastic methods (t-SNE). Same seed → same layout.",
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
        embeddings = inputs.get("embeddings")
        labels = inputs.get("labels") or []

        method = params.get("method", "PCA")
        perplexity = float(params.get("perplexity", 5.0))
        seed = int(params.get("seed", 42))

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        # Normalise input to a 2D float numpy array.
        arr = self._coerce_embeddings(embeddings)
        n, d = arr.shape

        if recorder is not None:
            recorder.record(
                "input",
                f"Received {n} vectors of dimension {d}.",
                scalars={"N": float(n), "D": float(d)},
            )

        actual_method = method
        # t-SNE needs perplexity < N - 1 and N reasonably > 4. Fall back to PCA
        # rather than crashing — surface the swap as a step trace note.
        if method == "t-SNE" and n < 5:
            actual_method = "PCA"
            if recorder is not None:
                recorder.record(
                    "fallback",
                    f"t-SNE requires N≥5; got N={n}. Falling back to PCA.",
                    scalars={"requested_perplexity": perplexity},
                )

        if actual_method == "PCA":
            from sklearn.decomposition import PCA

            if n == 0 or d == 0:
                projected = np.zeros((n, 2), dtype=np.float32)
            else:
                k = min(2, n, d)
                pca = PCA(n_components=k, random_state=seed)
                fit = pca.fit_transform(arr)
                if k < 2:
                    pad = np.zeros((n, 2 - k), dtype=np.float32)
                    fit = np.concatenate([fit, pad], axis=1)
                projected = fit.astype(np.float32, copy=False)
                if recorder is not None:
                    recorder.record(
                        "pca",
                        "Project onto the top-2 principal components — directions of maximum variance.",
                        scalars={"explained_variance_pc1": float(pca.explained_variance_ratio_[0])},
                    )
        else:  # t-SNE
            from sklearn.manifold import TSNE

            effective_perplexity = max(2.0, min(perplexity, n - 1))
            tsne = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=seed,
                init="pca",
                learning_rate="auto",
            )
            projected = tsne.fit_transform(arr).astype(np.float32, copy=False)
            if recorder is not None:
                recorder.record(
                    "tsne",
                    f"Optimise a 2D layout (perplexity={effective_perplexity:g}) that preserves local neighbourhoods.",
                    scalars={"perplexity": effective_perplexity},
                )

        # Normalise to [-1, 1] so the inline scatter doesn't have to handle
        # arbitrary scales. Centred on the mean to make analogy demonstrations
        # visually balanced.
        if n > 0:
            centred = projected - projected.mean(axis=0, keepdims=True)
            scale = np.max(np.abs(centred))
            if scale > 0:
                centred = centred / scale
            projected = centred.astype(np.float32, copy=False)
            if recorder is not None:
                recorder.record(
                    "normalize",
                    "Centre on origin and rescale so max |coord| = 1.",
                )

        tensor = torch.from_numpy(projected)
        result: dict[str, Any] = {
            "points_2d": tensor,
            "labels": list(labels)[: tensor.shape[0]],
        }
        if recorder is not None:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _coerce_embeddings(value: Any) -> np.ndarray:
        if value is None:
            return np.zeros((0, 0), dtype=np.float32)
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"EmbeddingScatter expects a 2D embeddings tensor; got shape {arr.shape}"
            )
        return arr.astype(np.float32, copy=False)
