"""LMTokenizer node (#290) -- one reusable tokenizer object for the LM stack.

``Tokenizer`` already turns a string into tokens, but it returns the RESULT of
tokenizing one piece of text. Training a language model needs the tokenizer
ITSELF: ``LMTokenizedDataset`` calls it once per corpus row, and Task 3's
``TextGenerate`` / ``Perplexity`` call it again at generation time. So this node
outputs a small object with four members and nothing else:

======================  ====================================================
``encode(text)``        ``list[int]`` -- the token ids of *text*
``decode(ids)``         ``str`` -- ids back to text
``eos_id``              ``int`` -- the end-of-document token
``vocab_size``          ``int`` -- how many ids exist
======================  ====================================================

That is a DUCK-TYPED contract on a ``DataType.ANY`` port, not a new wire type
(see the ``lr_scheduler`` ports for the same precedent): a ``TOKENIZER`` type
would cost five frontend files and two CSS blocks to teach the canvas a colour,
and buy nothing the four members above do not already pin. Every consumer
checks the four members and names this node in its error message.

**Why the eos id is derived rather than tabulated.** gpt2 numbers
``<|endoftext|>`` 50256 and cl100k_base numbers the same literal 100257. A
per-family table of those numbers is a table that goes stale the first time
tiktoken adds an encoding, so the id comes from the encoding's own
``eot_token``.

**Offline behaviour.** tiktoken downloads each encoding's BPE ranks once and
caches them on disk (``TIKTOKEN_CACHE_DIR``), so the first use of an encoding
needs the network and every later one does not. A failure to load is reported
as exactly that rather than as an opaque ``ConnectionError``.
"""

from __future__ import annotations

from typing import Any, Iterable

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from .tokenizer_node import _load_encoder

#: The tiktoken BPE encodings this node offers, ordered small vocab first.
#: A strict subset of ``tokenizer_node.TIKTOKEN_FAMILIES`` -- the shared
#: ``_load_encoder`` also serves HuggingFace ``tokenizers`` objects, whose
#: ``encode`` returns an Encoding rather than ids and which have no
#: ``eot_token``, so passing this param straight through would hand the graph
#: an object that fails the contract above at the first ``encode``.
LM_ENCODINGS = ["gpt2", "p50k_base", "cl100k_base", "o200k_base"]


class TiktokenLMTokenizer:
    """The four-member tokenizer contract, backed by one tiktoken encoding.

    Module scope, not a closure inside the node, so ``torch.save`` and the
    Python export can name the class (#283).

    ``name`` is the fifth, OPTIONAL member: ``LMTokenizedDataset`` folds it
    into its disk-cache key so the same corpus tokenized by gpt2 and by
    cl100k_base cannot collide. It is not part of the contract consumers may
    require, only of the one they may read.
    """

    __slots__ = ("name", "vocab_size", "eos_id", "_encoder")

    def __init__(self, name: str, encoder: Any) -> None:
        self.name = name
        self._encoder = encoder
        self.vocab_size = int(encoder.n_vocab)
        self.eos_id = int(encoder.eot_token)

    def __repr__(self) -> str:
        return (f"TiktokenLMTokenizer(name={self.name!r}, "
                f"vocab_size={self.vocab_size}, eos_id={self.eos_id})")

    def encode(self, text: str) -> list[int]:
        """Token ids of *text*. Never raises on the text's content.

        ``disallowed_special=()`` mirrors ``TokenizerNode``: tiktoken's default
        RAISES when the input contains a special-token literal such as
        ``<|endoftext|>``, and a corpus row is arbitrary text that may well
        contain one. Encoding it as ordinary characters is the only reading
        that lets a real corpus through.
        """
        return list(self._encoder.encode(text, disallowed_special=()))

    def decode(self, ids: Iterable[int]) -> str:
        """*ids* back to text. ``list()`` because tiktoken wants a sequence and
        callers legitimately hold a tensor slice or a generator."""
        return self._encoder.decode([int(i) for i in ids])


class LMTokenizerNode(BaseNode):
    NODE_NAME = "LMTokenizer"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "The tokenizer itself, as a reusable object: wire it into "
        "LMTokenizedDataset to turn a text corpus into training blocks, and "
        "into the generation nodes so they speak the same token ids the model "
        "was trained on. gpt2's 50257-token vocabulary is the usual starting "
        "point. Each encoding downloads its BPE table once, then works "
        "offline."
    )

    # Cacheable, and correctly so: the output is a stateless wrapper around a
    # process-wide encoder, built from one param. Nothing downstream mutates
    # it -- the four members are read, never written -- so replaying the
    # recorded handle describes it exactly. See ``BaseNode.cacheable`` for the
    # four shapes that are not.

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tokenizer",
                data_type=DataType.ANY,
                description=(
                    "The tokenizer object: encode(text) -> list[int], "
                    "decode(ids) -> str, plus eos_id and vocab_size. Wire it "
                    "to LMTokenizedDataset and to the generation nodes."
                ),
            ),
            PortDefinition(
                name="vocab_size",
                data_type=DataType.SCALAR,
                description=(
                    "How many distinct token ids this encoding has. Set "
                    "CausalLMModel's vocab_size to the same number."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="encoding",
                param_type=ParamType.SELECT,
                default="gpt2",
                options=list(LM_ENCODINGS),
                description=(
                    "Which BPE vocabulary to use. gpt2 (50257 tokens) is the "
                    "cheapest to train against; cl100k_base (GPT-3.5/4) and "
                    "o200k_base (GPT-4o) pack more text into the same number "
                    "of tokens but need a much wider output layer."
                ),
            ),
        ]

    def execute(
        self, inputs: dict[str, Any], params: dict[str, Any],
    ) -> dict[str, Any]:
        encoding = str(params.get("encoding", "gpt2") or "gpt2")
        if encoding not in LM_ENCODINGS:
            raise ValueError(
                f"LMTokenizer: unknown encoding {encoding!r}; set the "
                f"`encoding` param to one of {LM_ENCODINGS}.")

        try:
            _kind, encoder = _load_encoder(encoding)
        except Exception as exc:  # noqa: BLE001 - re-raised with the fix
            # Everything reaching here is a load failure, and the useful thing
            # to say is the same in every case: the BPE table is not on this
            # machine yet and fetching it needs the network. Naming the cache
            # directory matters for the air-gapped classroom case -- a table
            # copied in from another machine works.
            raise RuntimeError(
                f"LMTokenizer could not load the {encoding!r} BPE table: "
                f"{type(exc).__name__}: {exc}. tiktoken downloads it on first "
                f"use, so this machine is offline (or behind a proxy) and has "
                f"no cached copy. Run the node once with network access, or "
                f"point TIKTOKEN_CACHE_DIR at a directory holding the table."
            ) from exc

        tokenizer = TiktokenLMTokenizer(encoding, encoder)
        return {"tokenizer": tokenizer, "vocab_size": tokenizer.vocab_size}
