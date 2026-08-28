"""TextEmbeddingNode -- one dense vector per text, from a real encoder.

The node every RAG chain is built around, and the smallest honest statement
of what an embedding is FOR: two texts that mean the same thing come back as
two vectors pointing the same way, so "find the passage that answers this
question" becomes a dot product. Embed the documents once, embed the
question, compare.

It is deliberately a sibling of ``WordVector`` rather than a generalisation
of it. WordVector answers "what does one WORD mean", has three backends and
two of them are lookup tables; this one answers "what does one TEXT mean",
and a table cannot answer that at all -- there is no row for a sentence
nobody has written before. What they share is the encoder, and that lives in
``._sentence_models``: one registry, one gate, one bounded cache, so a graph
running both nodes against the same model loads it once.

Three decisions worth stating.

**Two text inputs, and they never combine.** ``texts`` is the list a chunker
produces; ``text`` is the string a learner types or a ``TextInput`` carries.
A graph wiring both has said two different things about what to embed, and
silently picking one would drop half of what was connected -- so it raises,
before anything is loaded.

**Labels are cut to the rows, always.** ``embeddings`` and ``labels`` are
read side by side downstream (``CosineSimilarity`` takes ``keys`` and
``key_labels``; ``EmbeddingScatter`` takes ``labels``), where one extra name
shifts every row's label by one. A stopped encode returns the rows it had --
and one stopped before its first batch returns a ``(0, 0)`` tensor, having
never run a forward pass to learn the width from -- so the labels are sliced
by the ROW COUNT and not by the number of texts that went in.

**The run never downloads.** ``load_sentence_model`` raises
``PackMissingError`` naming the pack, and nothing here catches it: the
editor reads the ``(pack=<id>)`` suffix back to offer the download, and a
wrapped error would cost the learner that button. ``REQUIRES_PACK`` is the
editor's half of the same promise -- the node is greyed out in the palette
before anybody presses Run.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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
from ._sentence_models import (
    DEFAULT_SENTENCE_MODEL,
    SENTENCE_MODELS,
    SENTENCE_PACK,
    encode_in_batches,
    load_sentence_model,
    option_packs_for_models,
)

logger = logging.getLogger(__name__)


def _integer(params: dict[str, Any], name: str, default: int, *,
             minimum: int) -> int:
    """A whole-number param, clamped to *minimum*, defaulted on null/empty.

    The idiom this replaces is ``int(params.get(name, d) or d)``, which reads
    FALSINESS as "not set" -- and 0 is a legal, meaningful value for
    ``max_seq_length`` (it means "leave the model's own cap alone"). Sharing
    one helper across all three keeps that trap from being reintroduced in
    the one place it bites.

    Clamped rather than refused: the editor already bounds these with
    min/max, so an out-of-range value arrived from a hand-edited graph or an
    exported script, and embedding in batches of one is a better answer than
    refusing to embed at all. A value that is not a number at all is a
    different thing and does raise, because there is nothing to clamp.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TextEmbedding: {name} must be a whole number, got {raw!r}."
        ) from exc
    return max(minimum, value)


class TextEmbeddingNode(BaseNode):
    NODE_NAME = "TextEmbedding"
    CATEGORY = "LLM"
    # The whole node, not just some options: every backend it has comes out
    # of this pack, so an install without it cannot run the node at all.
    REQUIRES_PACK = SENTENCE_PACK
    DESCRIPTION = (
        "Turn each text into one dense vector with a pre-trained "
        "sentence-transformer, so texts that mean the same thing end up "
        "close together (cosine near 1). This is the encoder behind semantic "
        "search and RAG: embed your documents once, embed the question, and "
        "compare. Needs the sentence-embeddings pack from Package Center; "
        "the four bundled models are small (22M-118M parameters) and run on "
        "CPU."
    )

    # A function of its inputs: the same texts, model and params give the
    # same rows every time. Nothing is streamed that a cache hit would
    # swallow (the contrast is TextGenerate, whose streamed tokens are the
    # point of running it), and the encode is the expensive part of any RAG
    # graph whose downstream nodes get re-run.
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="texts",
                data_type=DataType.LIST,
                description=(
                    "List of strings, one vector each, e.g. "
                    "TextChunker.chunks."
                ),
                optional=True,
            ),
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description=(
                    "A single string, or several separated by newlines when "
                    "split_lines is on; wire a TextInput here."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embeddings",
                data_type=DataType.TENSOR,
                description=(
                    "[N, D] float32, one row per input text, on the run "
                    "device"
                ),
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description=(
                    "The input texts, shortened to label_chars for display; "
                    "wire to CosineSimilarity.key_labels or "
                    "EmbeddingScatter.labels"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="model",
                param_type=ParamType.SELECT,
                default=DEFAULT_SENTENCE_MODEL,
                # Catalog order, which is the order Package Center lists the
                # downloads in. The map is built through
                # ``option_packs_for_models`` so the ``pack:item`` spelling
                # is written down once instead of retyped here.
                options=list(SENTENCE_MODELS),
                option_packs=option_packs_for_models(),
                description=(
                    "all-MiniLM-L6-v2: smallest, English. "
                    "paraphrase-multilingual-MiniLM-L12-v2: 50+ languages "
                    "incl. Traditional Chinese, no prefixes needed. "
                    "bge-small-zh-v1.5: Chinese-focused. multilingual-e5-"
                    "small: strongest for retrieval; expects 'query: ' / "
                    "'passage: ' prefixes (see prefix)."
                ),
            ),
            ParamDefinition(
                name="text",
                param_type=ParamType.STRING,
                default="Machine learning finds patterns in data.",
                description="Fallback text when no input is connected.",
            ),
            ParamDefinition(
                name="split_lines",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Treat each non-empty line of the text input as a "
                    "separate text. Turn off when a single multi-line "
                    "document should become one vector."
                ),
            ),
            ParamDefinition(
                name="prefix",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Prepended to every text before encoding. "
                    "multilingual-e5 was trained with 'query: ' for "
                    "questions and 'passage: ' for documents; other models "
                    "ignore this."
                ),
            ),
            ParamDefinition(
                name="normalize",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "L2-normalise so dot product equals cosine similarity "
                    "downstream."
                ),
            ),
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=32,
                min_value=1,
                max_value=512,
                description=(
                    "Texts per forward pass. Only speed and memory."
                ),
            ),
            ParamDefinition(
                name="max_seq_length",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                max_value=8192,
                description=(
                    "Token cap per text (0 = the model's own default: 128 "
                    "for paraphrase-multilingual, 256 for all-MiniLM, 512 "
                    "for bge/e5). Longer texts are truncated, so size chunks "
                    "accordingly."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="label_chars",
                param_type=ParamType.INT,
                default=48,
                min_value=8,
                max_value=200,
                description=(
                    "How many characters of each text to keep in labels."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description=(
                    "Where to encode (auto follows the global device)."
                ),
                advanced=True,
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
        # Function-level, like every device-aware node in this package: the
        # registry imports this module at startup to build the palette, and
        # the palette must not pay for anything a run pays for.
        from ...core.device_utils import (
            context_device,
            resolve_node_device,
            to_device,
        )

        # First, and before anything is loaded: a wiring mistake must read as
        # a wiring mistake, not as a missing download.
        texts = self._resolve_texts(inputs, params)

        repo = str(params.get("model") or DEFAULT_SENTENCE_MODEL)
        prefix = str(params.get("prefix") or "")
        normalize = bool(params.get("normalize", True))
        batch_size = _integer(params, "batch_size", 32, minimum=1)
        max_seq_length = _integer(params, "max_seq_length", 0, minimum=0)
        label_chars = _integer(params, "label_chars", 48, minimum=1)

        device = resolve_node_device(params.get("device"), context)
        started = time.monotonic()
        model = load_sentence_model(
            repo, device, max_seq_length=max_seq_length)

        rows_array, stopped_at = encode_in_batches(
            model,
            texts,
            batch_size=batch_size,
            normalize=normalize,
            prefix=prefix,
            progress=(ProgressThrottle(progress_callback)
                      if progress_callback else None),
            should_stop=stop_checker(context),
        )
        elapsed = time.monotonic() - started

        rows = int(rows_array.shape[0])
        dim = int(rows_array.shape[1])
        tensor = torch.from_numpy(rows_array)
        # Sliced by the ROW count, never by ``len(texts)`` -- see the module
        # docstring. Truncation is for display only; the model saw the whole
        # text.
        labels = [text[:label_chars] for text in texts[:rows]]

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        if recorder is not None:
            mean_chars = sum(len(text) for text in texts) / len(texts)
            recorder.record(
                "input_texts",
                f"{len(texts)} text(s) to embed with {repo}"
                + (f", each prefixed with {prefix!r}" if prefix else "")
                + ".",
                scalars={"count": float(len(texts)),
                         "mean_chars": float(mean_chars)},
            )
            recorder.record(
                "encode",
                f"One forward pass per batch of {batch_size}: {rows} row(s) "
                f"of {dim} numbers.",
                scalars={"dim": float(dim)},
                embeddings=tensor,
            )
            if normalize:
                recorder.record(
                    "normalize",
                    "L2-normalise each row so a dot product downstream is "
                    "cosine similarity.",
                    embeddings=tensor,
                )

        # The encode ran on ``device`` (which a pinned node may have chosen
        # for itself) and came back through numpy, i.e. on the CPU. The
        # OUTPUT goes to the run's global device, so whatever is wired after
        # this node -- CosineSimilarity, EmbeddingScatter, an Add -- meets it
        # on the same device as every other tensor in the run.
        tensor = to_device(tensor, context_device(context))

        note = (f"embedded {rows} texts with {repo} (D={dim}) in "
                f"{elapsed:.1f}s on {device}")
        result: dict[str, Any] = {
            "embeddings": tensor,
            "labels": labels,
            # The one result key the canvas Log tab renders; dunder keys are
            # filtered out of recorded outputs and port summaries.
            "__log__": note,
        }
        if recorder is not None:
            result["__steps__"] = recorder.steps
        if stopped_at is not None:
            # The partial rows are the output, not a consolation prize: the
            # engine reads this marker as "interrupted" rather than "failed"
            # precisely so what was computed survives the click.
            result.update(interrupted_result(batch=stopped_at, texts=rows))
        logger.info("TextEmbedding: %s", note)
        return result

    @classmethod
    def _resolve_texts(cls, inputs: dict[str, Any],
                       params: dict[str, Any]) -> list[str]:
        """The texts to embed, from whichever of the three sources is live.

        Precedence is connection over parameter, and the two INPUTS do not
        combine -- see the module docstring. An empty result raises rather
        than returning nothing: a node that produced a ``[0, 0]`` tensor
        here would fail further downstream, at a node that never saw the
        empty ``text`` box.
        """
        listed = inputs.get("texts")
        single = inputs.get("text")

        if listed is not None and single is not None:
            raise ValueError(
                "TextEmbedding: connect either texts or text, not both.")

        if listed is not None:
            if isinstance(listed, str):
                # ``list("hello")`` is five one-character texts, and every
                # layer below would accept that: five rows come back, the
                # shapes are right, nothing raises. Same guard and the same
                # reason as ``encode_in_batches``.
                raise ValueError(
                    "TextEmbedding: the `texts` input carries a single "
                    "string, not a list of strings. Wire it into the `text` "
                    "input instead.")
            texts = [cls._element_text(item) for item in listed]
        else:
            raw = single if single is not None else params.get("text", "")
            texts = cls._split(str(raw), bool(params.get("split_lines", True)))

        if not texts:
            raise ValueError(
                "TextEmbedding has nothing to embed: connect texts or text, "
                "or set the text param.")
        return texts

    @staticmethod
    def _element_text(item: Any) -> str:
        """One element of the ``texts`` list, as the string to embed.

        A dict is the shape a retrieval chain passes chunks around in -- the
        text plus its source, its page, its offsets -- so the ``text`` field
        is read out of it. Stringifying the whole dict instead would embed
        the words "source" and "page" into every vector, which is a wrong
        answer that looks exactly like a right one.
        """
        if isinstance(item, dict):
            if "text" not in item:
                raise ValueError(
                    "TextEmbedding: a dict on the `texts` input must carry a "
                    "'text' key holding the string to embed; this one has "
                    f"{sorted(str(key) for key in item)}.")
            item = item["text"]
        return str(item)

    @staticmethod
    def _split(text: str, split_lines: bool) -> list[str]:
        """A string as one text per non-empty line, or as one whole text.

        Blank lines are dropped rather than embedded: an empty string has no
        meaning to encode, and the row it produced would sit in the output
        under an empty label.
        """
        if split_lines:
            return [line.strip() for line in text.splitlines() if line.strip()]
        stripped = text.strip()
        return [stripped] if stripped else []
