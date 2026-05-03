"""CosineSimilarityNode — score every key vector against every query vector.

This is the kernel of vector retrieval (and the engine behind the canonical
``king − man + woman ≈ queen`` analogy demo): for each query row, it returns
the cosine similarity to every key row, plus the indices and labels of the
top-k most similar keys.

When the inputs are already L2-normalised (``WordVector(normalize=True)``),
the result is just ``Q @ K.T``. When they aren't, we normalise on the fly
so users never silently get a dot product instead of a similarity.
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder


class CosineSimilarityNode(BaseNode):
    NODE_NAME = "CosineSimilarity"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Compute cosine similarity between every query and every key vector. "
        "For unit-length inputs this is just the dot product; otherwise we "
        "normalise on the fly. Outputs the full similarity matrix plus the "
        "top-k indices and labels per query — the same kernel a vector "
        "retriever uses inside RAG."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="queries",
                data_type=DataType.TENSOR,
                description="Query tensor of shape [Q, D] (or [D] for a single query).",
            ),
            PortDefinition(
                name="keys",
                data_type=DataType.TENSOR,
                description="Key tensor of shape [K, D].",
            ),
            PortDefinition(
                name="key_labels",
                data_type=DataType.LIST,
                description="Optional list of K labels — used to populate `top_k_labels`.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="similarity",
                data_type=DataType.TENSOR,
                description="Float32 tensor of shape [Q, K] — entry [i, j] is cos(query_i, key_j).",
            ),
            PortDefinition(
                name="top_k_indices",
                data_type=DataType.LIST,
                description="List of length Q; each entry is a list of `top_k` integer key indices, highest similarity first.",
            ),
            PortDefinition(
                name="top_k_labels",
                data_type=DataType.LIST,
                description="List of length Q; each entry is the corresponding key labels (empty when `key_labels` not provided).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="top_k",
                param_type=ParamType.INT,
                default=5,
                min_value=1,
                description="How many nearest keys to surface per query.",
            ),
            ParamDefinition(
                name="exclude_self_words",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Comma-separated list of labels to exclude from the top-k. "
                    "Useful for analogy demos: setting 'king,man,woman' makes "
                    "the top-1 surface 'queen' instead of the input words."
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
        q_in = inputs.get("queries")
        k_in = inputs.get("keys")
        labels = list(inputs.get("key_labels") or [])

        if q_in is None or k_in is None:
            raise ValueError("CosineSimilarity requires both `queries` and `keys` inputs.")

        Q = self._coerce_2d(q_in)
        K = self._coerce_2d(k_in)
        if Q.shape[1] != K.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: queries D={Q.shape[1]}, keys D={K.shape[1]}"
            )

        top_k = max(1, int(params.get("top_k", 5)))
        excludes = {
            w.strip().lower()
            for w in str(params.get("exclude_self_words", "")).split(",")
            if w.strip()
        }

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        if recorder is not None:
            recorder.record(
                "input",
                f"Q={Q.shape[0]} queries × K={K.shape[0]} keys × D={Q.shape[1]}.",
                scalars={"Q": float(Q.shape[0]), "K": float(K.shape[0]), "D": float(Q.shape[1])},
            )

        # Normalise both sides defensively. WordVector(normalize=True) already
        # returns unit rows, but cosine on raw embeddings would silently be a
        # dot product.
        q_norm = torch.nn.functional.normalize(Q, dim=1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(K, dim=1, eps=1e-12)
        sim = q_norm @ k_norm.T  # [Q, K]

        if recorder is not None:
            recorder.record(
                "similarity",
                "sim[i, j] = q_i · k_j / (||q_i|| · ||k_j||).",
                similarity=sim,
            )

        # Build per-query top-k while honouring exclude_self_words.
        sim_for_topk = sim.clone()
        if excludes and labels:
            for j, label in enumerate(labels):
                if label.lower() in excludes:
                    sim_for_topk[:, j] = -float("inf")

        effective_k = min(top_k, K.shape[0])
        top_indices: list[list[int]] = []
        top_labels: list[list[str]] = []
        if Q.shape[0] > 0 and effective_k > 0:
            _, idx = torch.topk(sim_for_topk, k=effective_k, dim=1)
            for row in idx.tolist():
                top_indices.append(row)
                top_labels.append([labels[j] for j in row] if labels else [])

        if recorder is not None and labels:
            recorder.record(
                "top_k",
                f"Surface the top-{effective_k} keys per query"
                + (f" (excluding {sorted(excludes)})" if excludes else "")
                + ".",
                scalars={"k": float(effective_k), "excluded": float(len(excludes))},
            )

        result: dict[str, Any] = {
            "similarity": sim,
            "top_k_indices": top_indices,
            "top_k_labels": top_labels,
        }
        if recorder is not None:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _coerce_2d(value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        if value.dtype != torch.float32:
            value = value.float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError(
                f"CosineSimilarity expects 1D or 2D tensors; got shape {tuple(value.shape)}"
            )
        return value
