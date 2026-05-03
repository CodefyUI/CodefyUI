"""Tokenizer node — text → tokens via tiktoken (BPE) or HF tokenizers (WordPiece/SentencePiece).

Used to teach how text becomes the integer-valued sequences that LLMs consume.
The same text typed into different ``family`` settings produces visibly different
breakdowns — perfect for showing why GPT, BERT, and Llama "see" text differently.
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
from ...core.step_trace import StepRecorder

# Process-level cache of loaded encoders. Tokenizers are deterministic by family,
# so a single instance can be safely shared across graphs/nodes/threads. We avoid
# NodeStateStore here because (a) tokenizers are not nn.Modules, and (b) keying
# by family alone gives better reuse than (graph_id, node_id, hash).
_ENCODER_CACHE: dict[str, tuple[str, Any]] = {}

TIKTOKEN_FAMILIES = {"cl100k_base", "o200k_base", "p50k_base", "gpt2"}

# HF tokenizer family display name → repo id on the HuggingFace Hub. Each repo
# must publish a tokenizer.json that the standalone `tokenizers` library can load
# without pulling in `transformers`.
HF_FAMILIES: dict[str, str] = {
    "bert-base-uncased": "bert-base-uncased",
    "Llama-3": "Xenova/llama-3-tokenizer",
    "T5": "google/t5-v1_1-base",
}


def _load_encoder(family: str) -> tuple[str, Any]:
    """Return ``(kind, encoder)`` where kind is "tiktoken" or "hf"."""
    cached = _ENCODER_CACHE.get(family)
    if cached is not None:
        return cached

    if family in TIKTOKEN_FAMILIES:
        import tiktoken

        enc = tiktoken.get_encoding(family)
        _ENCODER_CACHE[family] = ("tiktoken", enc)
        return _ENCODER_CACHE[family]

    if family in HF_FAMILIES:
        from tokenizers import Tokenizer

        repo = HF_FAMILIES[family]
        tok = Tokenizer.from_pretrained(repo)
        _ENCODER_CACHE[family] = ("hf", tok)
        return _ENCODER_CACHE[family]

    raise ValueError(f"Unknown tokenizer family: {family!r}")


class TokenizerNode(BaseNode):
    NODE_NAME = "Tokenizer"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Split text into the integer tokens an LLM consumes. Different families use "
        "different algorithms — BPE (GPT), WordPiece (BERT), SentencePiece (Llama, T5) — "
        "so the same input produces visibly different breakdowns."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="Input text. Optional — falls back to the `text` param when not connected.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tokens",
                data_type=DataType.LIST,
                description="List of token strings (one per token).",
            ),
            PortDefinition(
                name="token_ids",
                data_type=DataType.LIST,
                description="List of integer token IDs.",
            ),
            PortDefinition(
                name="offsets",
                data_type=DataType.LIST,
                description="List of [start, end] character offsets per token in the input text.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="family",
                param_type=ParamType.SELECT,
                default="cl100k_base",
                options=[
                    "cl100k_base",
                    "o200k_base",
                    "p50k_base",
                    "gpt2",
                    "bert-base-uncased",
                    "Llama-3",
                    "T5",
                ],
                description=(
                    "Tokenizer family. tiktoken serves cl100k/o200k/p50k/gpt2 fully offline; "
                    "the rest download a tokenizer.json from HuggingFace on first use."
                ),
            ),
            ParamDefinition(
                name="text",
                param_type=ParamType.STRING,
                default="The quick brown fox jumps over the lazy dog.",
                description="Text to tokenize. Used when no `text` input is connected.",
            ),
            ParamDefinition(
                name="show_special_tokens",
                param_type=ParamType.BOOL,
                default=False,
                description="Include the tokenizer's special tokens (BOS/EOS/CLS/SEP/...) in the output.",
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
        text = self._coerce_text(inputs.get("text"), params.get("text", ""))
        family = params.get("family", "cl100k_base")
        show_special = bool(params.get("show_special_tokens", False))

        kind, enc = _load_encoder(family)

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        if recorder is not None:
            recorder.record(
                "input_text",
                f"Raw input ({len(text)} characters).",
                scalars={"chars": float(len(text))},
            )

        if kind == "tiktoken":
            tokens, token_ids, offsets = self._tokenize_tiktoken(enc, text)
        else:
            tokens, token_ids, offsets = self._tokenize_hf(enc, text, show_special)

        if recorder is not None:
            recorder.record(
                "tokenize",
                f"Apply {kind}/{family} → {len(tokens)} tokens.",
                scalars={
                    "num_tokens": float(len(tokens)),
                    "unique_ids": float(len(set(token_ids))),
                },
            )

        result: dict[str, Any] = {
            "tokens": tokens,
            "token_ids": token_ids,
            "offsets": [list(o) for o in offsets],
        }
        if recorder is not None:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _coerce_text(input_value: Any, fallback: str) -> str:
        if input_value is None:
            return str(fallback)
        if isinstance(input_value, list):
            # Batch passthrough: tokenize the first item. Future RAG nodes will
            # iterate batches explicitly; this keeps the teaching demo focused.
            return "" if not input_value else str(input_value[0])
        return str(input_value)

    @staticmethod
    def _tokenize_tiktoken(
        enc: Any, text: str
    ) -> tuple[list[str], list[int], list[tuple[int, int]]]:
        # ``disallowed_special=()`` allows special-token literals to be encoded
        # as plain text instead of raising — necessary for arbitrary user input.
        ids = list(enc.encode(text, disallowed_special=()))
        tokens = [enc.decode([i]) for i in ids]
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            tok_len = len(tok)
            offsets.append((cursor, cursor + tok_len))
            cursor += tok_len
        return tokens, ids, offsets

    @staticmethod
    def _tokenize_hf(
        enc: Any, text: str, show_special: bool
    ) -> tuple[list[str], list[int], list[tuple[int, int]]]:
        encoding = enc.encode(text, add_special_tokens=show_special)
        tokens = list(encoding.tokens)
        ids = list(encoding.ids)
        offsets = [(int(s), int(e)) for s, e in encoding.offsets]
        return tokens, ids, offsets
