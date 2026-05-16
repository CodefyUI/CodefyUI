"""EduTokenEmbeddingNode — toy token-to-vector lookup for the attention demos.

The production-style ``Embedding`` node (Utility category) wraps
``nn.Embedding`` with vocab_size=10000, embed_dim=256 — far too large to
visualise, and it expects a LongTensor input rather than the LIST that
``Tokenizer`` emits.

This educational variant:

* Takes ``tokens:LIST`` (strings) **or** ``token_ids:LIST`` (ints) directly,
  no manual coercion needed in the graph.
* Defaults to embed_dim=8, vocab_size=32 — small enough that every value in
  the table can be inspected at a glance.
* Two modes:

  - ``hash``: stable Python ``hash(token) % vocab_size`` for strings or
    ``id % vocab_size`` for ints. Same token always maps to the same row.
    Doesn't require knowing the full vocabulary up front.
  - ``ordinal``: assigns row IDs by first appearance order in this run.
    Reports the assignment via the ``vocab`` output so students can see
    "cat got id 0, dog got id 1, …".

Both modes deterministically build an embedding table seeded by ``seed`` so
runs are reproducible. Real training of these embeddings is out of scope —
use the Utility/Embedding node when you want a learnable lookup table.
"""

from __future__ import annotations

import math
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


def _stable_hash(s: str, mod: int) -> int:
    """Deterministic hash that doesn't depend on PYTHONHASHSEED.

    Built-in ``hash()`` is randomised per process, so two graph runs would
    map "cat" to different rows. We hash the bytes via a tiny FNV-1a so the
    mapping is stable across processes.
    """
    h = 0x811C9DC5  # FNV offset basis (32-bit)
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, keep 32-bit
    return h % mod


class EduTokenEmbeddingNode(BaseNode):
    NODE_NAME = "EduTokenEmbedding"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Toy token-to-vector lookup. Accepts `tokens:LIST` or `token_ids:LIST` "
        "and emits [seq, embed_dim] embeddings. `hash` mode maps any token "
        "deterministically; `ordinal` mode assigns row IDs in first-appearance "
        "order so the vocab is visible. Use the Utility/Embedding node for a "
        "trainable, production-sized embedding table."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tokens",
                data_type=DataType.LIST,
                description="List of token strings. Optional if `token_ids` is connected.",
                optional=True,
            ),
            PortDefinition(
                name="token_ids",
                data_type=DataType.LIST,
                description="List of integer token IDs (e.g. from Tokenizer). Optional if `tokens` is connected.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embeddings",
                data_type=DataType.TENSOR,
                description="Float32 tensor of shape [seq, embed_dim].",
            ),
            PortDefinition(
                name="vocab",
                data_type=DataType.LIST,
                description="Unique tokens used in this run, in row-id order (ordinal mode) or as observed (hash mode).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="embed_dim",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Dimension of each embedding vector. Keep small (≤16) for inline visualisation.",
            ),
            ParamDefinition(
                name="vocab_size",
                param_type=ParamType.INT,
                default=32,
                min_value=1,
                description="Size of the embedding table. `hash` mode wraps any token id mod this size.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the embedding-table initialisation. Same seed → same vectors.",
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="hash",
                options=["hash", "ordinal"],
                description=(
                    "hash: stable hash(token) mod vocab_size. "
                    "ordinal: assign row IDs by first-appearance order in this run."
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
        tokens = inputs.get("tokens")
        token_ids = inputs.get("token_ids")
        if tokens is None and token_ids is None:
            raise ValueError("EduTokenEmbedding requires either `tokens` or `token_ids` input.")

        embed_dim = max(1, int(params.get("embed_dim", 8)))
        vocab_size = max(1, int(params.get("vocab_size", 32)))
        seed = int(params.get("seed", 42))
        mode = str(params.get("mode", "hash"))

        # Build the seeded embedding table once per call. Local generator so
        # we don't perturb the global RNG.
        gen = torch.Generator()
        gen.manual_seed(seed)
        table = torch.randn(vocab_size, embed_dim, generator=gen, dtype=torch.float32) * (
            1.0 / math.sqrt(embed_dim)
        )

        # Resolve each input position to a row id and a display label.
        ids: list[int] = []
        vocab_out: list[str] = []
        ordinal_map: dict[str, int] = {}

        seq_iter = self._normalise_seq(tokens, token_ids)
        for label, raw_id in seq_iter:
            if mode == "hash":
                if raw_id is not None:
                    row = int(raw_id) % vocab_size
                else:
                    row = _stable_hash(label, vocab_size)
                ids.append(row)
                if label not in vocab_out:
                    vocab_out.append(label)
            elif mode == "ordinal":
                if label in ordinal_map:
                    row = ordinal_map[label]
                else:
                    row = len(ordinal_map) % vocab_size
                    ordinal_map[label] = row
                    vocab_out.append(label)
                ids.append(row)
            else:
                raise ValueError(f"Unknown EduTokenEmbedding mode: {mode!r}")

        if not ids:
            embeddings = torch.zeros((0, embed_dim), dtype=torch.float32)
        else:
            embeddings = table[torch.tensor(ids, dtype=torch.long)]

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "input_tokens",
                f"Resolve {len(ids)} input position(s) using mode={mode!r}.",
                scalars={
                    "seq_len": float(len(ids)),
                    "embed_dim": float(embed_dim),
                    "vocab_size": float(vocab_size),
                },
            )
            recorder.record(
                "embedding_table",
                f"Seeded random table of shape ({vocab_size}, {embed_dim}).",
                table=table,
            )
            recorder.record(
                "lookup",
                "Gather rows: embeddings[i] = table[row_id(token_i)].",
                embeddings=embeddings,
            )
            return {"embeddings": embeddings, "vocab": vocab_out, "__steps__": recorder.steps}

        return {"embeddings": embeddings, "vocab": vocab_out}

    @staticmethod
    def _normalise_seq(tokens: Any, token_ids: Any):
        """Yield (label_str, raw_id_or_None) pairs in input order.

        Prefers ``tokens`` for labelling (more readable) but uses ``token_ids``
        for the lookup id when only the numeric form is available.
        """
        if tokens is not None and token_ids is not None:
            t_list = list(tokens)
            i_list = list(token_ids)
            n = min(len(t_list), len(i_list))
            for i in range(n):
                yield str(t_list[i]), int(i_list[i])
            return
        if tokens is not None:
            for t in tokens:
                yield str(t), None
            return
        # token_ids only — synthesize a label from the id.
        for i in token_ids or []:
            yield f"<id:{int(i)}>", int(i)
