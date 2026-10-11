"""DataMixDatasetNode — 依權重、可重現地混合多個文字語料（#300）。

資料混合與課程研究的入口：TinyStories 配多少比例的 wikitext？先簡單後困難
的排序有沒有差？這顆吃 2–6 個 TextCorpusDataset 的輸出，依權重決定各來源
的列數（ratio）、依權重以種子化的順序交錯（interleave）或依序串接
（concat）成一個新的文字資料集，接 LMTokenizedDataset 之後就是可研究的混合
預訓練資料。

interleave 與 concat 只決定「列的順序」，每個語料的每一列都恰好出現一次；
ratio 決定各來源的列數，必要時重複或略過列（#695）。混合結果只存
(來源, 列號) 索引、逐列惰性讀取，混兩個 HF 語料不會把文字實體化進記憶體。
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
    DESCRIPTION = "Mix 2-6 text corpora: by ratio, weighted order or concat"
    DETAILS = (
        "ratio makes the weights the share of each corpus in the output: "
        "total_rows rows (0 = the sum of the corpus sizes), split by the weights; "
        "a corpus asked for more rows than it has repeats rows, one asked for "
        "fewer contributes a seeded subset, and the rows are shuffled by the seed. "
        "interleave uses every row of every corpus exactly once, so the output "
        "share is the corpus sizes; the weights only set how the corpora are "
        "spread through the order (a corpus that empties stops being drawn and "
        "the rest renormalise). concat runs corpus_1 to the end, then corpus_2. "
        "The mixture stores only (source, row) indices and reads rows lazily. "
        "Feed TextCorpusDataset outputs in and the result into LMTokenizedDataset."
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
                # Empty, not "0.5, 0.5": a left-out param runs as its default,
                # and two weights refuse every mix of three or more sources.
                default="",
                description=(
                    "Comma-separated weights, one per source (normalized; "
                    "ignored by concat). Empty: equal weights. ratio: the "
                    "share of each source in the output. interleave: only "
                    "the order; every row is used once, so the output share "
                    "is the corpus sizes."
                ),
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="interleave",
                options=["interleave", "ratio", "concat"],
                description=(
                    "ratio: total_rows rows split by the weights, repeating "
                    "or subsampling rows as needed, in seeded order. "
                    "interleave: every row once, in a seeded order the "
                    "weights spread. concat: corpus_1 fully, then corpus_2, "
                    "... — an ordered curriculum."
                ),
            ),
            ParamDefinition(
                name="total_rows",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                visible_when={"mode": "ratio"},
                description=(
                    "ratio mode only: rows in the output. 0 = the sum of the "
                    "corpus sizes."
                ),
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Mixture seed (interleave order, ratio row picks and order) — the same seed and inputs reproduce the same mixture",
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
        elif mode == "ratio":
            weights = self._parse_weights(
                str(params.get("weights", "") or ""), count)
            total_rows = max(0, int(params.get("total_rows", 0) or 0))
            index_pairs = self._ratio(
                lengths, weights, total_rows or sum(lengths), seed, torch)
        else:
            weights = self._parse_weights(
                str(params.get("weights", "") or ""), count)
            index_pairs = self._interleave(lengths, weights, seed, torch)

        dataset = MixedTextDataset(sources, index_pairs)
        if mode == "ratio":
            used = [0] * count
            for source_index, _row in index_pairs:
                used[source_index] += 1
            breakdown = ", ".join(
                f"corpus_{i + 1}: {used[i]:,} of {lengths[i]:,}"
                for i in range(count))
        else:
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
    def _ratio(
        lengths: list[int], weights: list[float], total: int, seed: int,
        torch: Any,
    ) -> list[tuple[int, int]]:
        """``total`` rows whose per-source counts follow ``weights``.

        Each source's quota is ``total * weight`` rounded by largest
        remainder, so the counts sum to ``total`` and match the weights as
        closely as whole rows allow. A source's rows come from seeded
        permutations of the whole corpus, one full pass after another: no
        row repeats before every row of that source has been used once.
        The combined rows are then shuffled by the same seed.
        """
        generator = torch.Generator().manual_seed(seed)
        exact = [total * weight for weight in weights]
        quotas = [int(value) for value in exact]
        by_remainder = sorted(
            range(len(weights)), key=lambda i: exact[i] - quotas[i],
            reverse=True)
        for i in by_remainder[: total - sum(quotas)]:
            quotas[i] += 1
        pairs: list[tuple[int, int]] = []
        for source, (length, quota) in enumerate(zip(lengths, quotas)):
            picked = 0
            while picked < quota:
                take = min(length, quota - picked)
                rows = torch.randperm(length, generator=generator)[:take]
                pairs.extend((source, row) for row in rows.tolist())
                picked += take
        order = torch.randperm(len(pairs), generator=generator).tolist()
        return [pairs[i] for i in order]

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
