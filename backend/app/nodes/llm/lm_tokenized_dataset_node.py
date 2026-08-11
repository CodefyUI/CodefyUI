"""LMTokenizedDatasetNode — 把文字語料編碼、串接、切成固定長度的訓練塊。

標準的 LM 預訓練資料形狀：每列文字 → token ids（每列後面接一個 EOS）→
全部串成一條長流 → 切成 ``seq_len + 1`` 的塊；訓練時 ``input = 塊[:-1]``、
``labels = 塊[1:]``（右移一格的 next-token 目標）。輸出的 DATASET 直接接
既有 `DataLoader`（預設 collate 疊成 ``(B, seq_len)`` int64 批次）餵給
TrainingLoop。

編碼上億 token 要花好幾分鐘 CPU，所以打包結果會以內容指紋為鍵存進資料根目錄
的 ``lm_token_cache/``；同一份語料＋同一組參數的重跑（以及實驗的重複執行）
直接從磁碟載回，不再重新編碼。
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

#: Rows per encode_batch call — small enough to stay responsive to Stop,
#: large enough that tiktoken's thread pool is fed.
_CHUNK_ROWS = 512


class LMTokenizedDatasetNode(BaseNode):
    NODE_NAME = "LMTokenizedDataset"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Tokenize a text corpus and pack it into fixed-length next-token "
        "blocks — the standard LM pretraining layout. Joins rows with EOS, "
        "cuts the stream into seq_len+1 blocks, and yields (input_ids, "
        "labels) pairs where labels are inputs shifted by one. Wire "
        "TextCorpusDataset + LMTokenizer in, and the output DATASET into "
        "DataLoader -> TrainingLoop. Packed blocks are disk-cached by "
        "content fingerprint so reruns skip retokenization."
    )

    # Writes the packed-block cache file — a side effect the return value
    # does not carry (rule 3 of the cacheable contract), and its input is a
    # live DATASET handle a fingerprint cannot describe.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Rows of raw text (TextCorpusDataset output)",
            ),
            PortDefinition(
                name="tokenizer",
                data_type=DataType.ANY,
                description="Tokenizer handle from LMTokenizer (encode_batch / eos_id)",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Packed blocks yielding (input_ids int64[seq_len], labels int64[seq_len])",
            ),
            PortDefinition(
                name="total_tokens",
                data_type=DataType.SCALAR,
                description="Tokens in the packed stream (after any max_tokens cap)",
            ),
            PortDefinition(
                name="num_blocks",
                data_type=DataType.SCALAR,
                description="Number of seq_len-sized training blocks",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="seq_len",
                param_type=ParamType.INT,
                default=1024,
                min_value=16,
                max_value=8192,
                description="Block length in tokens (must not exceed the model's max_seq_len)",
            ),
            ParamDefinition(
                name="append_eos",
                param_type=ParamType.BOOL,
                default=True,
                description="Append the tokenizer's EOS between documents so the model learns document boundaries",
            ),
            ParamDefinition(
                name="max_tokens",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Cap the packed stream at N tokens (0 = all) — budget knob for pilot runs",
            ),
            ParamDefinition(
                name="cache",
                param_type=ParamType.BOOL,
                default=True,
                description="Cache packed blocks on disk (lm_token_cache/ under the data directory)",
                advanced=True,
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override the cache directory (relative paths resolve inside the data directory)",
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
        import hashlib
        import json

        import torch

        from ...core.data_paths import data_root, resolve_data_path
        from ...core.loop_control import (
            ProgressThrottle,
            interrupted_result,
            stop_checker,
        )
        from ._lm_data import PackedLMDataset

        corpus = inputs.get("dataset")
        tokenizer = inputs.get("tokenizer")
        if corpus is None:
            raise ValueError("LMTokenizedDataset requires a `dataset` input (raw text rows).")
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "eos_id"):
            raise ValueError(
                "LMTokenizedDataset requires a `tokenizer` input from the "
                "LMTokenizer node (an object with encode/encode_batch/eos_id)."
            )

        seq_len = max(16, min(8192, int(params.get("seq_len", 1024))))
        append_eos = bool(params.get("append_eos", True))
        max_tokens = max(0, int(params.get("max_tokens", 0) or 0))
        use_cache = bool(params.get("cache", True))
        cache_dir_param = str(params.get("cache_dir", "") or "")

        num_rows = len(corpus)
        if num_rows == 0:
            raise RuntimeError("The text corpus has no rows to tokenize.")

        # ── content fingerprint ─────────────────────────────────────────
        # The corpus is a live handle, so the fingerprint samples it:
        # row count plus hashes of the first/last rows. A same-name corpus
        # with edited middle rows in between CAN alias — the cache doc
        # states this, and `cache=false` opts out entirely.
        encoding_name = str(getattr(tokenizer, "encoding_name", type(tokenizer).__name__))
        sample_indices = list(range(min(3, num_rows))) + [
            index for index in (num_rows - 2, num_rows - 1) if index >= 3
        ]
        samples = [
            hashlib.sha1(str(corpus[index]).encode("utf-8", "replace")).hexdigest()[:16]
            for index in sample_indices
        ]
        fingerprint = hashlib.sha256(json.dumps({
            "v": 1,
            "encoding": encoding_name,
            "seq_len": seq_len,
            "append_eos": append_eos,
            "max_tokens": max_tokens,
            "num_rows": num_rows,
            "samples": samples,
        }, sort_keys=True).encode("utf-8")).hexdigest()[:24]

        cache_path = None
        if use_cache:
            base = data_root() / "lm_token_cache"
            target = cache_dir_param or "."
            cache_base = resolve_data_path(target, base=base)
            cache_path = cache_base / f"lmpack-{fingerprint}.pt"
            if cache_path.is_file():
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                blocks = payload["blocks"]
                total_tokens = int(payload["total_tokens"])
                dataset = PackedLMDataset(blocks)
                if progress_callback:
                    progress_callback({
                        "event": "cache_hit",
                        "num_blocks": len(dataset),
                        "total_tokens": total_tokens,
                    })
                return {
                    "dataset": dataset,
                    "total_tokens": total_tokens,
                    "num_blocks": len(dataset),
                }

        # ── tokenize + pack ─────────────────────────────────────────────
        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        eos = [int(tokenizer.eos_id)] if append_eos else []
        pieces: list[torch.Tensor] = []
        total_tokens = 0
        rows_done = 0
        stopped_at_row: int | None = None

        for start in range(0, num_rows, _CHUNK_ROWS):
            if should_stop():
                stopped_at_row = start
                break
            stop = min(start + _CHUNK_ROWS, num_rows)
            texts = [str(corpus[index]) for index in range(start, stop)]
            if hasattr(tokenizer, "encode_batch"):
                encoded = tokenizer.encode_batch(texts)
            else:
                encoded = [tokenizer.encode(text) for text in texts]
            flat: list[int] = []
            for ids in encoded:
                flat.extend(ids)
                flat.extend(eos)
            if flat:
                pieces.append(torch.tensor(flat, dtype=torch.int32))
                total_tokens += len(flat)
            rows_done = stop
            throttle.emit({
                "event": "tokenize",
                "rows": rows_done,
                "total_rows": num_rows,
                "tokens": total_tokens,
            })
            if max_tokens and total_tokens >= max_tokens:
                break

        if stopped_at_row is not None:
            return {
                "dataset": None,
                "total_tokens": total_tokens,
                "num_blocks": 0,
                **interrupted_result(row=stopped_at_row),
            }

        stream = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int32)
        if max_tokens:
            stream = stream[:max_tokens]
        total_tokens = int(stream.numel())
        block = seq_len + 1
        usable = (total_tokens // block) * block
        if usable == 0:
            raise RuntimeError(
                f"Corpus too small: {total_tokens} tokens cannot fill one "
                f"block of seq_len+1 = {block}. Lower seq_len or provide "
                "more text."
            )
        blocks = stream[:usable].view(-1, block).clone()
        dataset = PackedLMDataset(blocks)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp")
            torch.save({"blocks": blocks, "total_tokens": total_tokens}, temp_path)
            temp_path.replace(cache_path)

        return {
            "dataset": dataset,
            "total_tokens": total_tokens,
            "num_blocks": len(dataset),
        }
