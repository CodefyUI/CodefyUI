"""WordVectorNode -- one vector per input word, from a table or an encoder.

Three kinds of backend sit behind one SELECT, and the difference between
them is the lesson this node teaches:

``demo-16d``
    A hand-built toy vocabulary in 16 interpretable dimensions (royalty /
    divinity / masculinity / femininity / animal classes / motion / vehicles
    / food / weather). It ships inline, needs no download, and the canonical
    ``king - man + woman = queen`` analogy is EXACT on it -- because the
    vectors were written so that it would be.

``glove-50d``
    The real 400,000-word GloVe table, out of the ``word-vectors`` pack. The
    same analogy is only approximate here, and that gap is the point.

the four sentence-transformer repos
    A modern encoder from the ``sentence-embeddings`` pack, run over one
    word at a time. Messier still for single words -- these models are
    trained on sentences -- and what real retrieval systems actually use.

Two things this module deliberately does not do.

**It never downloads.** Both real backends read what the Package Center
already fetched and raise ``PackMissingError``, naming the pack, when it is
not there; ``core.packs.require_pack`` is where that promise is written
down. The ``option_packs`` map below is the editor's half of it: an option
whose pack is missing is greyed out before anybody presses Run.

**It stays importable without any of it.** The registry imports every node
module at startup to build the palette, so the GloVe reader, the encoder
loader and ``app.core.packs`` are all imported inside the functions that
need them -- an install that has downloaded nothing still gets a palette.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch

from ...core.loop_control import (
    ProgressThrottle,
    interrupted_result,
    stop_checker,
)
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder
from . import _packs_bridge
from ._demo_vectors import DEMO_VECTORS
from ._glove import GLOVE_PACK
from ._sentence_models import SENTENCE_MODELS, option_packs_for_models

#: How many words go through the encoder in one forward pass. Stop is
#: answered between batches, so this is also the longest a click waits;
#: 64 keeps a whole teaching vocabulary to a single pass while a document's
#: worth of words still checks in several times.
SENTENCE_BATCH_SIZE = 64

#: Backend names that saved graphs from earlier previews still carry, and
#: what to say about each. ``ValueError`` rather than ``PackMissingError``
#: on purpose: nothing a learner can install fixes a name that no longer
#: exists, and offering them a download button for it would be a dead end.
#: So each message names the option they should pick instead.
_RETIRED_BACKENDS: dict[str, str] = {
    "glove-100d": (
        "Backend 'glove-100d' was retired; the word-vectors pack ships "
        "glove-50d. Set backend to 'glove-50d'."),
    "minilm-sentence-384d": (
        "Backend 'minilm-sentence-384d' was renamed; set backend to "
        "'sentence-transformers/all-MiniLM-L6-v2'."),
}


@lru_cache(maxsize=8)
def _load_backend(backend: str) -> tuple[list[str], np.ndarray]:
    """Return ``(vocab, matrix)`` for a TABLE backend; ``matrix`` is [V, D].

    Cached, so the second run of a session pays nothing: building the demo
    matrix is milliseconds, reading the converted GloVe npz is a second or
    two of disk, and neither answer can change while the process lives.

    The sentence backends are not here. They have no vocabulary to look a
    word up in, and their weights are held by ``_sentence_models``' own
    bounded cache rather than by this unbounded-in-bytes one.
    """
    if backend == "demo-16d":
        vocab = sorted(DEMO_VECTORS.keys())
        matrix = np.array([DEMO_VECTORS[w] for w in vocab], dtype=np.float32)
        return vocab, matrix

    if backend == "glove-50d":
        # Function-level, like every pack-backed loader in this package. Not
        # for import cost -- ``._glove`` is already imported above for its
        # pack id and is cheap either way; the 83 MB npz read is deferred by
        # the CALL. It raises ``PackMissingError`` naming the pack when the
        # download is not there, and nothing here catches it: the editor
        # reads the ``(pack=<id>)`` suffix to offer the download, and a
        # wrapped error would cost the learner that button.
        from ._glove import load_glove_50d

        return load_glove_50d()

    retired = _RETIRED_BACKENDS.get(backend)
    if retired is not None:
        raise ValueError(retired)

    raise ValueError(f"Unknown WordVector backend: {backend!r}")


class WordVectorNode(BaseNode):
    NODE_NAME = "WordVector"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Look up a vector for each input word. Pre-trained embeddings place "
        "related words near each other, so $king - man + woman \\approx "
        "queen$. demo-16d is a hand-built 59-word toy vocabulary that ships "
        "offline and makes the analogy exact; glove-50d is the real "
        "400k-word GloVe table (word-vectors pack), where the analogy is "
        "approximate; the sentence-transformer backends (sentence-embeddings "
        "pack) run each word through a modern encoder, which is messier "
        "still for single words but is what real retrieval systems use."
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
                # demo-16d first because it is the one that always works.
                # The rest are in catalog order, which is the order the
                # Package Center lists the downloads in.
                options=["demo-16d", "glove-50d", *SENTENCE_MODELS],
                # demo-16d has no entry: an option with no pack is one every
                # install can pick, which is exactly what "ships offline"
                # means. Built through the bridge and through
                # ``option_packs_for_models`` so the ``pack:item`` spelling
                # is written down once instead of retyped here.
                option_packs={
                    "glove-50d": _packs_bridge.requirement(
                        GLOVE_PACK, "glove-50d"),
                    **option_packs_for_models(),
                },
                description=(
                    "Vector source. Options greyed out need a pack from "
                    "Package Center; a run never downloads."
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
                description=(
                    "Emit a zero vector for out-of-vocabulary words instead "
                    "of dropping them. Only meaningful for the table "
                    "backends (demo-16d, glove-50d); sentence models embed "
                    "every word."
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
        words = self._coerce_words(inputs.get("tokens"), params.get("words", ""))
        backend = params.get("backend", "demo-16d")
        normalize = bool(params.get("normalize", False))
        keep_oov = bool(params.get("keep_oov", False))

        # Lower-cased and stripped once, here, because both paths want the
        # same thing from an input port that may carry anything: a table
        # lookup is case-insensitive, and an encoder handed "  " would embed
        # a blank and hand back a row nothing names.
        keys = [key for key in (str(w).lower().strip() for w in words) if key]

        oov: list[str] = []
        vocab_size: int | None = None
        stopped_at: int | None = None

        if backend in SENTENCE_MODELS:
            arr, labels, stopped_at = self._encode(
                backend, keys, normalize=normalize,
                progress_callback=progress_callback, context=context)
            summary = (
                f"Embedded {len(labels)} of {len(words)} word(s) with "
                f"{backend}; an encoder has no vocabulary, so nothing is OOV.")
        else:
            vocab, matrix = _load_backend(backend)
            vocab_size = len(vocab)
            arr, labels, oov = self._look_up(
                vocab, matrix, keys, normalize=normalize, keep_oov=keep_oov)
            summary = (
                f"Resolved {len(labels)} of {len(words)} words against "
                f"backend vocab; {len(oov)} OOV.")

        dim = int(arr.shape[1])
        tensor = torch.from_numpy(arr)

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        if recorder is not None:
            # No ``vocab_size`` scalar for an encoder rather than a zero: a
            # backend with no vocabulary has no size, and 0 would be plotted
            # as one.
            scalars = {"input_count": float(len(words))}
            if vocab_size is not None:
                scalars["vocab_size"] = float(vocab_size)
            recorder.record(
                "input_words",
                f"{len(words)} input word(s); backend={backend} (D={dim}, "
                f"V={vocab_size if vocab_size is not None else 'n/a'}).",
                scalars=scalars,
            )
            recorder.record(
                "lookup",
                summary,
                scalars={
                    "matched": float(len(labels)),
                    "oov": float(len(oov)),
                    "dim": float(dim),
                },
                embeddings=tensor,
            )
            if normalize:
                recorder.record(
                    "normalize",
                    "L2-normalise each row so dot products give cosine similarity downstream.",
                    embeddings=tensor,
                )

        # Embeddings come from numpy on CPU; move them to the global run device
        # so downstream tensor ops (Add, CosineSimilarity) stay on one device.
        from ...core.device_utils import context_device, to_device
        tensor = to_device(tensor, context_device(context))

        result: dict[str, Any] = {"embeddings": tensor, "labels": labels}
        if recorder is not None:
            result["__steps__"] = recorder.steps
        if stopped_at is not None:
            # The partial rows are the output, not a consolation prize: the
            # engine reads this marker as "interrupted" rather than "failed"
            # precisely so what was computed survives the click.
            result.update(interrupted_result(
                batch=stopped_at, embedded=len(labels), total_words=len(keys)))
        return result

    @staticmethod
    def _look_up(
        vocab: list[str],
        matrix: np.ndarray,
        keys: list[str],
        *,
        normalize: bool,
        keep_oov: bool,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Table lookup: ``(rows, labels, oov)``, rows already normalised.

        One path for both tables. An empty result still carries the table's
        width, because a downstream node reading ``[0, 50]`` knows what it
        was handed and one reading ``[0, 0]`` does not.
        """
        index = {word: row for row, word in enumerate(vocab)}
        dim = int(matrix.shape[1])

        rows: list[np.ndarray] = []
        labels: list[str] = []
        oov: list[str] = []
        for key in keys:
            row = index.get(key)
            if row is not None:
                rows.append(matrix[row])
                labels.append(key)
            elif keep_oov:
                rows.append(np.zeros(dim, dtype=np.float32))
                labels.append(key)
                oov.append(key)
            else:
                oov.append(key)

        if not rows:
            return np.zeros((0, dim), dtype=np.float32), labels, oov

        arr = np.stack(rows).astype(np.float32, copy=False)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # leave zero rows untouched
            arr = arr / norms
        return arr, labels, oov

    @staticmethod
    def _encode(
        backend: str,
        keys: list[str],
        *,
        normalize: bool,
        progress_callback: Any | None,
        context: Any,
    ) -> tuple[np.ndarray, list[str], int | None]:
        """Run every word through a sentence encoder: ``(rows, labels,
        stopped_at)``.

        ``keep_oov`` is not a parameter here, and that is the whole
        difference from the table path: an encoder produces a vector for any
        string at all, so there is no vocabulary to fall out of and no zero
        row to stand in for a miss.

        The labels are cut to the rows that actually came back. A stopped
        encode returns what it had -- and one stopped before its first batch
        returns ``(0, 0)``, having never run a forward pass to learn the
        width from -- while ``embeddings`` and ``labels`` are read side by
        side downstream (``CosineSimilarity`` takes ``keys`` and
        ``key_labels``), where one extra name would shift every row's label
        by one.
        """
        # ``._sentence_models`` is already imported above for its registry,
        # so these two names cost nothing here; they are pulled in beside
        # the code that uses them. The import that MUST stay deferred is
        # ``sentence_transformers`` (transformers, torch, half a second of
        # startup), and that one lives inside ``load_sentence_model`` -- so
        # an install with no pack still gets a palette.
        from ...core.device_utils import resolve_node_device
        from ._sentence_models import encode_in_batches, load_sentence_model

        # No per-node device param: this node is a lookup, not a training
        # sink, so it follows the run's global device the way its output
        # tensor does.
        device = resolve_node_device(None, context)
        model = load_sentence_model(backend, device)

        arr, stopped_at = encode_in_batches(
            model,
            keys,
            batch_size=SENTENCE_BATCH_SIZE,
            normalize=normalize,
            prefix="",
            progress=(ProgressThrottle(progress_callback)
                      if progress_callback else None),
            should_stop=stop_checker(context),
        )
        return arr, keys[:int(arr.shape[0])], stopped_at

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
