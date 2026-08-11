"""DataMixDatasetNode — 依權重、可重現地混合多個文字語料（#300）。

資料混合與課程研究的入口：TinyStories 配多少比例的 wikitext？先簡單後困難
的排序有沒有差？這顆吃 2–6 個 TextCorpusDataset 的輸出，依權重以種子化的
順序交錯（interleave）或依序串接（concat）成一個新的文字資料集，接
LMTokenizedDataset 之後就是可研究的混合預訓練資料。

決定的是「列的順序」而不是列的內容：混合結果只存 (來源, 列號) 索引、
逐列惰性讀取，混兩個 HF 語料不會把文字實體化進記憶體。
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
    resolve_count_param,
)

_MIN_SOURCES = 2
_MAX_SOURCES = 6
#: Draw source picks in batches: one multinomial per row would put a python
#: loop around a kernel launch for every row of a million-row corpus.
_DRAW_CHUNK = 8192


class DataMixDatasetNode(BaseNode):
    NODE_NAME = "DataMixDataset"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Mix 2-6 text corpora into one dataset of raw text rows — by "
        "weighted, seeded interleaving (rows drawn proportionally, without "
        "replacement, deterministic per seed) or ordered concatenation "
        "(corpus_1 fully, then corpus_2, ... — a curriculum). Feed "
        "TextCorpusDataset outputs in and the result into "
        "LMTokenizedDataset to study data mixtures."
    )

    # Consumes live DATASET handles a fingerprint cannot describe (the
    # cacheable contract's rule 2, from the consumer side).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return cls.define_inputs_dynamic(None)

    @classmethod
    def define_inputs_dynamic(
        cls, params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        count = resolve_count_param(
            params, "sources",
            default=_MIN_SOURCES, minimum=_MIN_SOURCES, maximum=_MAX_SOURCES)
        return [
            PortDefinition(
                name=f"corpus_{index + 1}",
                data_type=DataType.DATASET,
                description=f"Text corpus {index + 1} (rows of raw text)",
            )
            for index in range(count)
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Mixed rows of raw text (lazy; order fixed by the seed)",
            ),
            PortDefinition(
                name="num_rows",
                data_type=DataType.SCALAR,
                description="Total rows in the mixture",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="sources",
                param_type=ParamType.INT,
                default=_MIN_SOURCES,
                min_value=_MIN_SOURCES,
                max_value=_MAX_SOURCES,
                description="How many corpus input ports this node has",
            ),
            ParamDefinition(
                name="weights",
                param_type=ParamType.STRING,
                default="0.5, 0.5",
                description=(
                    "Comma-separated draw weights, one per source "
                    "(normalized; interleave mode only). A source that "
                    "empties stops being drawn and the rest renormalize — "
                    "the tail of the mixture is whatever corpora remain."
                ),
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="interleave",
                options=["interleave", "concat"],
                description=(
                    "interleave: seeded proportional draws without "
                    "replacement. concat: corpus_1 fully, then corpus_2, "
                    "... — an ordered curriculum."
                ),
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Interleave order seed — the same seed and inputs reproduce the same mixture",
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
        import torch

        from ..data._hf_adapter import MixedTextDataset

        count = resolve_count_param(
            params, "sources",
            default=_MIN_SOURCES, minimum=_MIN_SOURCES, maximum=_MAX_SOURCES)
        sources: list[Any] = []
        for index in range(count):
            corpus = inputs.get(f"corpus_{index + 1}")
            if corpus is None:
                raise ValueError(
                    f"DataMixDataset: corpus_{index + 1} is not connected "
                    f"(sources={count}).")
            sources.append(corpus)
        lengths = [len(source) for source in sources]
        if any(length == 0 for length in lengths):
            empty = [i + 1 for i, length in enumerate(lengths) if length == 0]
            raise RuntimeError(
                f"DataMixDataset: corpus_{empty[0]} has no rows.")

        mode = str(params.get("mode", "interleave") or "interleave")
        seed = max(0, int(params.get("seed", 0) or 0))

        if mode == "concat":
            index_pairs = [
                (source_index, row)
                for source_index in range(count)
                for row in range(lengths[source_index])
            ]
        else:
            weights = self._parse_weights(
                str(params.get("weights", "") or ""), count)
            index_pairs = self._interleave(lengths, weights, seed, torch)

        dataset = MixedTextDataset(sources, index_pairs)
        breakdown = ", ".join(
            f"corpus_{i + 1}: {lengths[i]:,}" for i in range(count))
        return {
            "dataset": dataset,
            "num_rows": len(dataset),
            "__log__": (
                f"Mixed {len(dataset):,} rows ({mode}, seed {seed}) from "
                f"{breakdown}."
            ),
        }

    @staticmethod
    def _parse_weights(raw: str, count: int) -> list[float]:
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
        if not pieces:
            # Absent/blank (a hand-built graph without the param): equal
            # weights, the only default that works for every source count.
            return [1.0 / count] * count
        if len(pieces) != count:
            raise ValueError(
                f"DataMixDataset: weights has {len(pieces)} values but "
                f"sources={count}; give one weight per corpus, e.g. "
                f"\"{', '.join(['1'] * count)}\".")
        try:
            values = [float(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError(
                f"DataMixDataset: weights must be numbers, got {raw!r}."
            ) from exc
        if any(value <= 0 for value in values):
            raise ValueError(
                "DataMixDataset: every weight must be positive — a source "
                "with weight 0 should simply not be wired.")
        total = sum(values)
        return [value / total for value in values]

    @staticmethod
    def _interleave(
        lengths: list[int], weights: list[float], seed: int, torch: Any,
    ) -> list[tuple[int, int]]:
        """Proportional draws without replacement, deterministic per seed.

        Draws come in chunks (one multinomial call for thousands of picks);
        when a source empties mid-chunk its remaining picks are discarded
        and the probabilities renormalize over what is left — so the tail
        of the mixture is drawn from the corpora that still have rows,
        exactly as the weights param documents.
        """
        generator = torch.Generator().manual_seed(seed)
        remaining = list(lengths)
        next_row = [0] * len(lengths)
        pairs: list[tuple[int, int]] = []
        total = sum(lengths)
        while len(pairs) < total:
            active = [i for i in range(len(lengths)) if remaining[i] > 0]
            if len(active) == 1:
                source = active[0]
                pairs.extend(
                    (source, row)
                    for row in range(next_row[source], lengths[source]))
                break
            probabilities = torch.tensor(
                [weights[i] if remaining[i] > 0 else 0.0
                 for i in range(len(lengths))],
                dtype=torch.float64)
            probabilities = probabilities / probabilities.sum()
            draws = torch.multinomial(
                probabilities,
                min(_DRAW_CHUNK, total - len(pairs)),
                replacement=True,
                generator=generator,
            )
            for source in draws.tolist():
                if remaining[source] == 0:
                    # This source emptied earlier in the chunk; the rest of
                    # the chunk is stale against the new probabilities.
                    break
                pairs.append((source, next_row[source]))
                next_row[source] += 1
                remaining[source] -= 1
        return pairs
