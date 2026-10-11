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

**Character and byte encodings (#692).** ``byte`` maps each UTF-8 byte to
its value (ids 0-255) and adds an end-of-document id 256, so it needs no
download and no corpus. ``char`` builds its vocabulary from the text on the
``corpus`` input: one id per distinct character, in code-point order, then
an end-of-document id and an unknown-character id for prompt characters the
corpus never contained. Both keep the classic character-level GPT exercise
to a vocabulary of a few dozen to a few hundred ids.

**Offline behaviour.** tiktoken downloads each encoding's BPE ranks once and
caches them on disk (``TIKTOKEN_CACHE_DIR``), so the first use of an encoding
needs the network and every later one does not. A failure to load is reported
as exactly that rather than as an opaque ``ConnectionError``.
"""

from __future__ import annotations

import hashlib
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

#: Encodings built in this module rather than loaded from tiktoken.
LOCAL_ENCODINGS = ["char", "byte"]


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


class ByteLMTokenizer:
    """UTF-8 bytes as ids 0-255, plus ``eos_id`` 256.

    Decoding replaces an incomplete or invalid UTF-8 sequence with U+FFFD, so
    a sampled id stream always decodes to a string.
    """

    __slots__ = ("name", "vocab_size", "eos_id")

    def __init__(self) -> None:
        self.name = "byte"
        self.eos_id = 256
        self.vocab_size = 257

    def __repr__(self) -> str:
        return "ByteLMTokenizer(vocab_size=257, eos_id=256)"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        data = bytes(i for i in (int(i) for i in ids) if 0 <= i < 256)
        return data.decode("utf-8", errors="replace")


class CharLMTokenizer:
    """One id per character of a fixed alphabet, plus end-of-document and
    unknown-character ids.

    ``name`` carries a digest of the alphabet: ``LMTokenizedDataset`` keys its
    disk cache on it, and two corpora with different alphabets give the same
    character different ids.
    """

    __slots__ = ("name", "vocab_size", "eos_id", "unk_id", "chars", "_index")

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars = "".join(sorted(set(chars)))
        self._index = {ch: i for i, ch in enumerate(self.chars)}
        self.eos_id = len(self.chars)
        self.unk_id = len(self.chars) + 1
        self.vocab_size = len(self.chars) + 2
        digest = hashlib.sha256(self.chars.encode("utf-8")).hexdigest()[:16]
        self.name = f"char:{digest}"

    def __repr__(self) -> str:
        return (f"CharLMTokenizer(chars={len(self.chars)}, "
                f"vocab_size={self.vocab_size}, eos_id={self.eos_id})")

    def encode(self, text: str) -> list[int]:
        """Ids of *text*; a character outside the alphabet becomes ``unk_id``."""
        index = self._index
        unk = self.unk_id
        return [index.get(ch, unk) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Text of *ids*; ``eos_id`` decodes to nothing, ``unk_id`` and
        out-of-range ids to U+FFFD."""
        chars = self.chars
        out: list[str] = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(chars):
                out.append(chars[i])
            elif i != self.eos_id:
                out.append("\ufffd")
        return "".join(out)


def _corpus_characters(corpus: Any) -> set[str]:
    """Every distinct character in *corpus*: a string, or rows of text."""
    if isinstance(corpus, str):
        return set(corpus)
    seen: set[str] = set()
    for index in range(len(corpus)):
        seen.update(str(corpus[index]))
    return seen


class LMTokenizerNode(BaseNode):
    NODE_NAME = "LMTokenizer"
    CATEGORY = "LLM"
    DESCRIPTION = "A reusable tokenizer: text to tokens and back"
    DETAILS = (
        "Wire it into LMTokenizedDataset to pack a corpus, and into the generation "
        "nodes so they speak the same ids the model was trained on. gpt2's "
        "50257-token vocabulary is the cheapest of the BPE tables; cl100k_base and "
        "o200k_base pack more text per token but need a wider output layer. Each "
        "BPE encoding downloads its table once, then works offline. byte uses the "
        "256 UTF-8 byte values plus an end-of-text id (257 ids). char builds its "
        "vocabulary from the text on the corpus input: one id per distinct "
        "character, plus end-of-text and unknown-character ids."
    )

    # Cacheable, and correctly so: the output is a stateless wrapper around a
    # process-wide encoder, built from one param. Nothing downstream mutates
    # it -- the four members are read, never written -- so replaying the
    # recorded handle describes it exactly. See ``BaseNode.cacheable`` for the
    # four shapes that are not. The char encoding reads the ``corpus`` input,
    # and the engine does not cache a node downstream of a non-cacheable one
    # such as TextCorpusDataset, so its vocabulary is rebuilt every run.

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="corpus",
                data_type=DataType.DATASET,
                description=(
                    "Rows of text the char encoding takes its alphabet from "
                    "(e.g. TextCorpusDataset). Required for char; the other "
                    "encodings ignore it."
                ),
                optional=True,
            ),
        ]

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
                options=[*LM_ENCODINGS, *LOCAL_ENCODINGS],
                description=(
                    "Which vocabulary to use. gpt2 (50257 tokens) is the "
                    "smallest BPE table; cl100k_base (GPT-3.5/4) and "
                    "o200k_base (GPT-4o) pack more text into the same number "
                    "of tokens but need a much wider output layer. byte: 257 "
                    "ids (256 byte values + end-of-text). char: one id per "
                    "character of the corpus input, + end-of-text and unknown."
                ),
            ),
        ]

    def execute(
        self, inputs: dict[str, Any], params: dict[str, Any],
    ) -> dict[str, Any]:
        encoding = str(params.get("encoding", "gpt2") or "gpt2")
        if encoding == "byte":
            byte_tokenizer = ByteLMTokenizer()
            return {"tokenizer": byte_tokenizer,
                    "vocab_size": byte_tokenizer.vocab_size}
        if encoding == "char":
            corpus = inputs.get("corpus")
            if corpus is None:
                raise ValueError(
                    "LMTokenizer: the char encoding builds its vocabulary "
                    "from text, and the `corpus` input is not connected. Wire "
                    "the TextCorpusDataset you train on into `corpus`.")
            chars = _corpus_characters(corpus)
            if not chars:
                raise ValueError(
                    "LMTokenizer: the `corpus` input has no characters, so "
                    "the char encoding has no vocabulary.")
            char_tokenizer = CharLMTokenizer(chars)
            return {
                "tokenizer": char_tokenizer,
                "vocab_size": char_tokenizer.vocab_size,
                "__log__": (
                    f"char vocabulary: {len(char_tokenizer.chars)} characters "
                    f"+ end-of-text + unknown = {char_tokenizer.vocab_size} ids."
                ),
            }
        if encoding not in LM_ENCODINGS:
            raise ValueError(
                f"LMTokenizer: unknown encoding {encoding!r}; set the "
                f"`encoding` param to one of {[*LM_ENCODINGS, *LOCAL_ENCODINGS]}.")

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
