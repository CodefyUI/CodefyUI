"""LMTokenizerNode — 給訓練管線用的、可重複使用的 tokenizer 物件。

示範用的 `Tokenizer` 節點把一句話切給你看（輸出 LIST）；訓練管線需要的是
「一個可以帶著走的 tokenizer」：LMTokenizedDataset 用它把整個語料編碼打包、
文字生成節點用它編碼 prompt／解碼輸出。這顆輸出一個輕量 handle（契約：
``encode`` / ``encode_batch`` / ``decode`` / ``eos_id`` / ``vocab_size`` /
``encoding_name``），底層是 tiktoken 的 BPE。

輸出的 port 型別是 ``ANY``（和 ``LRScheduler.scheduler`` 一樣的先例）：新增
DataType 需要配套的前端 edge 驗證／顏色變更，v1 先不動核心型別系統，契約
寫在描述裡。
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

#: tiktoken encodings this node serves. gpt2 (50257, eos <|endoftext|> = 50256)
#: matches CausalLMModel's default vocab_size.
LM_TOKENIZER_ENCODINGS = ["gpt2", "p50k_base", "cl100k_base", "o200k_base"]


class LMTokenizerHandle:
    """Reusable tokenizer facade over one tiktoken encoding.

    Module-scope and pickle-friendly: state is just the encoding NAME; the
    actual encoder is reloaded lazily after unpickling, so a graph value
    that ends up inside a full_model pickle (#283) or a spawned DataLoader
    worker never tries to serialize the tiktoken internals.
    """

    def __init__(self, encoding_name: str) -> None:
        self.encoding_name = encoding_name
        self._enc: Any = None

    # -- pickling ----------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        return {"encoding_name": self.encoding_name}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.encoding_name = state["encoding_name"]
        self._enc = None

    # -- encoder -----------------------------------------------------------
    def _encoder(self) -> Any:
        if self._enc is None:
            import tiktoken

            self._enc = tiktoken.get_encoding(self.encoding_name)
        return self._enc

    # -- contract ----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        # encode_ordinary: special-token literals in the corpus are data,
        # not control tokens.
        return list(self._encoder().encode_ordinary(text))

    def encode_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        encoded = self._encoder().encode_ordinary_batch(texts, num_threads=num_threads)
        return [list(ids) for ids in encoded]

    def decode(self, ids: list[int]) -> str:
        return self._encoder().decode(list(ids))

    @property
    def eos_id(self) -> int:
        return int(self._encoder().eot_token)

    @property
    def vocab_size(self) -> int:
        return int(self._encoder().n_vocab)

    def __repr__(self) -> str:  # keeps node-output summaries readable
        return f"LMTokenizerHandle({self.encoding_name!r})"


class LMTokenizerNode(BaseNode):
    NODE_NAME = "LMTokenizer"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Produce a reusable tokenizer object (tiktoken BPE) for the LM "
        "training pipeline: LMTokenizedDataset uses it to encode and pack a "
        "corpus, and text generation uses it to encode prompts and decode "
        "output. The tokenizer output exposes encode/encode_batch/decode, "
        "eos_id and vocab_size. gpt2 = vocab 50257 with eos <|endoftext|> "
        "(50256), matching CausalLMModel's default vocab_size."
    )

    # tiktoken downloads and caches its BPE ranks on first use of an
    # encoding — external state the cache key cannot see (same reasoning as
    # HuggingFaceDataset). After that first fetch everything is offline.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tokenizer",
                data_type=DataType.ANY,
                description="Tokenizer handle: encode(text)->ids, encode_batch, decode(ids)->text, eos_id, vocab_size",
            ),
            PortDefinition(
                name="vocab_size",
                data_type=DataType.SCALAR,
                description="Vocabulary size of the chosen encoding (wire into CausalLMModel.vocab_size checks)",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="encoding",
                param_type=ParamType.SELECT,
                default="gpt2",
                options=list(LM_TOKENIZER_ENCODINGS),
                description=(
                    "tiktoken encoding. gpt2 = 50257 tokens (GPT-2 vocabulary); "
                    "cl100k/o200k are the larger GPT-4-era vocabularies. "
                    "Downloaded once, then cached offline."
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
        encoding = str(params.get("encoding", "gpt2"))
        if encoding not in LM_TOKENIZER_ENCODINGS:
            raise ValueError(
                f"Unknown tokenizer encoding {encoding!r}; expected one of "
                f"{LM_TOKENIZER_ENCODINGS}"
            )
        handle = LMTokenizerHandle(encoding)
        try:
            vocab_size = handle.vocab_size
        except Exception as error:  # first-use download failed / offline
            raise RuntimeError(
                f"Could not load tiktoken encoding {encoding!r}. The first "
                "use of an encoding downloads its BPE ranks once; check "
                "network access, then retry."
            ) from error
        return {"tokenizer": handle, "vocab_size": int(vocab_size)}
