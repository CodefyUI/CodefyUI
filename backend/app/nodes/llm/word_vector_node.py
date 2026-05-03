"""WordVectorNode — look up pre-trained vectors for input words/tokens.

Supports two flavours of "word embedding":

* ``demo-16d`` — a hand-crafted toy vocabulary (~60 words, 16 dimensions)
  shipped inline so the canonical analogy demo
  (``king − man + woman ≈ queen``) works fully offline. Constructed dimensions
  (royalty / divinity / masculinity / femininity / animal classes / motion /
  vehicles / food / weather) are deliberately interpretable.

* ``glove-50d`` / ``glove-100d`` — real GloVe vectors over a top-10k English
  vocabulary, lazy-downloaded from a GitHub Release asset on first use via
  :mod:`app.core.asset_cache`. When the URL hasn't been published yet the
  node raises a friendly error pointing the user at ``demo-16d``.

A third backend (``minilm-sentence-384d`` via the optional ``llm-sentence``
extra) is reserved for a follow-up PR — the SELECT entry exists so the UI
plumbing is in place, but it isn't loadable from this PR.
"""

from __future__ import annotations

from functools import lru_cache
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
from ._demo_vectors import DEMO_VECTORS, DIM as DEMO_DIM


# Future GloVe asset specs (not published yet — first attempt raises a friendly
# error pointing users at the offline ``demo-16d`` backend).
_GLOVE_PLACEHOLDER_URL = "https://github.com/treeleaves30760/CodefyUI/releases/download/llm-assets-v0/{name}"


@lru_cache(maxsize=8)
def _load_backend(backend: str) -> tuple[list[str], np.ndarray]:
    """Return ``(vocab, matrix)`` where ``matrix.shape == (V, D)``.

    The lru_cache makes repeated runs of the same backend instant; the heavy
    one-time cost (asset download or numpy construction) happens once per
    Python process.
    """
    if backend == "demo-16d":
        vocab = sorted(DEMO_VECTORS.keys())
        matrix = np.array([DEMO_VECTORS[w] for w in vocab], dtype=np.float32)
        return vocab, matrix

    if backend in ("glove-50d", "glove-100d"):
        # The real wiring would call asset_cache.resolve(...) and parse the
        # downloaded .npz here. The asset hasn't been published yet, so we
        # surface a clear error instead of an opaque 404.
        raise NotImplementedError(
            f"Backend {backend!r} requires the GloVe asset bundle, which is "
            f"not yet published as a GitHub Release. Use 'demo-16d' for the "
            f"offline analogy demo. (See {_GLOVE_PLACEHOLDER_URL.format(name=backend)})"
        )

    if backend == "minilm-sentence-384d":
        raise NotImplementedError(
            "Sentence-transformer backend is gated behind the optional "
            "[llm-sentence] dependency group and lands in a follow-up PR. "
            "Use 'demo-16d' or 'glove-50d' (when published)."
        )

    raise ValueError(f"Unknown WordVector backend: {backend!r}")


class WordVectorNode(BaseNode):
    NODE_NAME = "WordVector"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Look up a pre-trained vector for each input word. Pre-trained "
        "embeddings place semantically related words near each other, so "
        "$king - man + woman \\approx queen$. The default `demo-16d` "
        "backend ships with the install; `glove-*` backends lazy-download "
        "real GloVe vectors on first use."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tokens",
                data_type=DataType.LIST,
                description="List of words/tokens to look up. Optional — falls back to the `words` param.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embeddings",
                data_type=DataType.TENSOR,
                description="Float32 tensor of shape [N, D] — one vector per recognised word.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="The words that were actually recognised (out-of-vocabulary words are dropped).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="backend",
                param_type=ParamType.SELECT,
                default="demo-16d",
                options=[
                    "demo-16d",
                    "glove-50d",
                    "glove-100d",
                    "minilm-sentence-384d",
                ],
                description=(
                    "Vector source. demo-16d is a hand-crafted toy vocab that "
                    "ships offline; glove-* downloads real GloVe vectors on "
                    "first use; minilm-sentence-384d requires the [llm-sentence] extra."
                ),
            ),
            ParamDefinition(
                name="words",
                param_type=ParamType.STRING,
                default="king queen man woman cat dog",
                description="Whitespace- or comma-separated list of words. Used when no `tokens` input is connected.",
            ),
            ParamDefinition(
                name="normalize",
                param_type=ParamType.BOOL,
                default=False,
                description="L2-normalise each vector. Required for cosine similarity to behave like dot product downstream.",
            ),
            ParamDefinition(
                name="keep_oov",
                param_type=ParamType.BOOL,
                default=False,
                description="Emit a zero vector for out-of-vocabulary words instead of dropping them.",
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
        words = self._coerce_words(inputs.get("tokens"), params.get("words", ""))
        backend = params.get("backend", "demo-16d")
        normalize = bool(params.get("normalize", False))
        keep_oov = bool(params.get("keep_oov", False))

        vocab, matrix = _load_backend(backend)
        vocab_index = {w: i for i, w in enumerate(vocab)}
        D = matrix.shape[1]

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        if recorder is not None:
            recorder.record(
                "input_words",
                f"{len(words)} input word(s); backend={backend} (D={D}, V={len(vocab)}).",
                scalars={"input_count": float(len(words)), "vocab_size": float(len(vocab))},
            )

        rows: list[np.ndarray] = []
        labels: list[str] = []
        oov: list[str] = []
        for w in words:
            key = w.lower().strip()
            if not key:
                continue
            idx = vocab_index.get(key)
            if idx is not None:
                rows.append(matrix[idx])
                labels.append(key)
            elif keep_oov:
                rows.append(np.zeros(D, dtype=np.float32))
                labels.append(key)
                oov.append(key)
            else:
                oov.append(key)

        if not rows:
            tensor = torch.zeros((0, D), dtype=torch.float32)
        else:
            arr = np.stack(rows).astype(np.float32, copy=False)
            if normalize:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # leave zero rows untouched
                arr = arr / norms
            tensor = torch.from_numpy(arr)

        if recorder is not None:
            recorder.record(
                "lookup",
                f"Resolved {len(labels)} of {len(words)} words against backend vocab; {len(oov)} OOV.",
                scalars={
                    "matched": float(len(labels)),
                    "oov": float(len(oov)),
                    "dim": float(D),
                },
                embeddings=tensor,
            )
            if normalize:
                recorder.record(
                    "normalize",
                    "L2-normalise each row so dot products give cosine similarity downstream.",
                    embeddings=tensor,
                )

        result: dict[str, Any] = {"embeddings": tensor, "labels": labels}
        if recorder is not None:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _coerce_words(input_value: Any, fallback: str) -> list[str]:
        if input_value is None:
            text = str(fallback)
        elif isinstance(input_value, list):
            return [str(x) for x in input_value]
        elif isinstance(input_value, str):
            text = input_value
        else:
            text = str(input_value)
        # Tolerant split: comma- or whitespace-separated.
        return [w for w in text.replace(",", " ").split() if w]
