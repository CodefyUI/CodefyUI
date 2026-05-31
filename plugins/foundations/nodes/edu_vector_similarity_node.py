"""EduVectorSimilarityNode — dot-product / cosine similarity, step by step.

Supports textbook lesson **I1-3 (向量相似度)**: given a query vector (or a
batch of them) and a set of key vectors, compute the pairwise similarity that
underpins retrieval, attention, and nearest-neighbour search.

Instead of one opaque "Similarity" node, this expands the computation into the
named pieces a student needs to see:

    raw_dot[i, j]   = q_i · k_j                       # inner products
    qn[i]           = ||q_i||                          # query L2 norms
    kn[j]           = ||k_j||                          # key   L2 norms
    cosine[i, j]    = raw_dot[i, j] / (qn[i] · kn[j])  # length-normalised
    dot[i, j]       = raw_dot[i, j]                    # unnormalised

The final ``similarity`` matrix is ``[N, M]`` and is ready to be rendered as a
heatmap (compare with ``EduSelfAttention``'s ``weights`` output). A 1-D query
``[D]`` is treated as a single row, so the result is ``[1, M]`` (i.e. N = 1).
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

_EPS = 1e-12


class EduVectorSimilarityNode(BaseNode):
    NODE_NAME = "Edu-VectorSimilarity"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Pairwise query/key similarity exposed as: raw dot products q·k → L2 "
        "norms ||q|| and ||k|| → cosine = dot / (||q||·||k||) (or raw dot). "
        "Outputs the [N, M] similarity matrix for direct heatmap visualisation, "
        "and records each intermediate in verbose mode."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="query",
                data_type=DataType.TENSOR,
                description="Query vector(s), shape [D] or [N, D]. A 1-D query becomes N=1.",
            ),
            PortDefinition(
                name="keys",
                data_type=DataType.TENSOR,
                description="Key vectors, shape [M, D]. Each row is one key to score against.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="similarity",
                data_type=DataType.TENSOR,
                description="Pairwise similarity, shape [N, M] (a 1-D query yields [1, M]). Ready for a heatmap.",
            ),
            PortDefinition(
                name="query_norms",
                data_type=DataType.TENSOR,
                description="L2 norm of each query row, shape [N]. Returned for both metrics.",
            ),
            PortDefinition(
                name="key_norms",
                data_type=DataType.TENSOR,
                description="L2 norm of each key row, shape [M]. Returned for both metrics.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="metric",
                param_type=ParamType.SELECT,
                default="cosine",
                options=["cosine", "dot"],
                description=(
                    "'cosine' divides the dot product by the product of the two "
                    "vectors' lengths (scale-invariant, in [-1, 1]); 'dot' returns "
                    "the raw inner product (grows with vector magnitude)."
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
        query = inputs.get("query")
        keys = inputs.get("keys")
        if query is None or keys is None:
            raise ValueError(
                "EduVectorSimilarity requires `query` and `keys` inputs."
            )

        if not isinstance(query, torch.Tensor):
            query = torch.as_tensor(query, dtype=torch.float32)
        if not isinstance(keys, torch.Tensor):
            keys = torch.as_tensor(keys, dtype=torch.float32)
        query = query.float()
        keys = keys.float()

        if keys.ndim != 2:
            raise ValueError(
                f"EduVectorSimilarity expects `keys` of shape [M, D]; got {tuple(keys.shape)}."
            )

        # Reshape query to [N, D]; a 1-D [D] query becomes a single row (N=1).
        if query.ndim == 1:
            query = query.unsqueeze(0)
        elif query.ndim != 2:
            raise ValueError(
                f"EduVectorSimilarity expects `query` of shape [D] or [N, D]; got {tuple(query.shape)}."
            )

        n, d_q = query.shape
        m, d_k = keys.shape
        if d_q != d_k:
            raise ValueError(
                f"EduVectorSimilarity: query feature dim D={d_q} does not match keys D={d_k}."
            )

        metric = str(params.get("metric", "cosine"))
        if metric not in ("cosine", "dot"):
            raise ValueError(
                f"EduVectorSimilarity: unknown metric {metric!r}; expected 'cosine' or 'dot'."
            )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        # raw_dot[i, j] = q_i · k_j  →  [N, M]
        raw_dot = query @ keys.transpose(0, 1)
        if recorder is not None:
            recorder.record(
                "dot_products",
                "raw_dot[i, j] = q_i · k_j (every query against every key).",
                scalars={"N": float(n), "M": float(m), "D": float(d_q)},
                query=query, keys=keys, raw_dot=raw_dot,
            )

        # L2 norms of each query / key row.
        query_norms = torch.linalg.vector_norm(query, dim=-1)  # [N]
        key_norms = torch.linalg.vector_norm(keys, dim=-1)      # [M]
        if recorder is not None:
            recorder.record(
                "norms",
                "L2 norms: query_norms[i] = ||q_i||, key_norms[j] = ||k_j||.",
                scalars={"D": float(d_q), "N": float(n), "M": float(m)},
                query_norms=query_norms, key_norms=key_norms,
            )

        if metric == "cosine":
            denom = query_norms.unsqueeze(1) * key_norms.unsqueeze(0) + _EPS
            similarity = raw_dot / denom
        else:  # "dot"
            similarity = raw_dot
        if recorder is not None:
            recorder.record(
                "similarity",
                (
                    "cosine = raw_dot / (||q_i|| · ||k_j||)"
                    if metric == "cosine"
                    else "dot = raw_dot (unnormalised inner product)"
                ),
                scalars={"metric": 1.0 if metric == "cosine" else 0.0},
                similarity=similarity,
            )

        result: dict[str, Any] = {
            "similarity": similarity,
            "query_norms": query_norms,
            "key_norms": key_norms,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
