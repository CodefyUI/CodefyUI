"""SyntheticSequenceNode — tiny *sequence* dataset for recurrent models.

The Data category had three synthetic generators before this one and every
one of them is spatial or tabular: ``SyntheticDataset`` makes 2D points,
``SyntheticSegmentation`` an image plus a mask, ``SyntheticShapes`` an image.
Nothing produced a **sequence**, so the RNN category's nodes (``LSTM``,
``GRU``, ``RNNCell``) had no zero-download dataset to train against — you
could run one forward pass on a hand-typed ``TensorInput`` and that was it.

This node fills that hole with the standard *memory* benchmark for recurrent
nets: a long sequence of distractor tokens with the answer hidden at one end,
and a label that can only be produced by carrying information across the
whole sequence.

    kind="recall_first"   answer sits at position 0, then T-1 distractors
    kind="recall_last"    T-1 distractors, then the answer at position T-1

The pair is the point. ``recall_first`` forces a dependency of length T and
is where a plain RNN's gradient dies (vanishing gradient); ``recall_last``
has a dependency of length 1 and is easy for anything. Same generator, same
shapes, same difficulty of the *task* — only the distance changes. Flipping
that one dropdown, holding everything else fixed, isolates distance as the
variable, which is the experiment C4-1 describes in prose.

Vocabulary layout is deliberately simple so the numbers stay readable::

    0 .. n_classes-1                  answer tokens (also the class labels)
    n_classes .. n_classes+n_distract-1   distractor tokens

so ``num_embeddings`` on a downstream ``Embedding`` must be at least
``n_classes + n_distractors``; ``vocab_size`` reports exactly that.

Each sample is ``(sequence (T,) int64, label int)`` — the shape
``nn.Embedding`` wants and the shape ``CrossEntropyLoss`` wants, so the
dataset drops straight into DataLoader → TrainingLoop with no adapter.
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


class _SyntheticSequenceDataset:
    """In-memory (sequence, label) pairs. Materialised once in ``__init__``.

    Generating up front rather than per ``__getitem__`` keeps the dataset
    deterministic under a DataLoader with ``shuffle=True`` and any worker
    count: the seed fixes the whole tensor, not a per-call RNG that workers
    would each fork differently.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        n_classes: int,
        n_distractors: int,
        kind: str,
        seed: int,
    ) -> None:
        import torch

        gen = torch.Generator().manual_seed(int(seed))

        # Distractors fill every position; the answer then overwrites one end.
        # Drawing the full block in one call (rather than per sample) is both
        # faster and keeps the seed → data mapping independent of n_samples
        # ordering.
        seqs = torch.randint(
            n_classes,
            n_classes + n_distractors,
            (n_samples, seq_len),
            generator=gen,
            dtype=torch.long,
        )
        labels = torch.randint(0, n_classes, (n_samples,), generator=gen, dtype=torch.long)

        answer_pos = 0 if kind == "recall_first" else seq_len - 1
        seqs[:, answer_pos] = labels

        self.sequences = seqs
        self.labels = labels
        self.answer_pos = answer_pos

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int):
        return self.sequences[idx], int(self.labels[idx])


class SyntheticSequenceNode(BaseNode):
    NODE_NAME = "SyntheticSequence"
    CATEGORY = "Data"
    DESCRIPTION = (
        "產生一個小型「序列記憶」資料集（CPU 友善、免下載）：每筆是一條長度 seq_len 的整數序列，"
        "答案藏在序列的開頭（recall_first）或結尾（recall_last），其餘位置全是隨機干擾 Token；"
        "標籤就是那個答案。recall_first 需要把記憶從第 1 步一路帶到最後一步，是檢驗梯度消失的"
        "標準任務；recall_last 的依賴距離只有 1，任何模型都學得起來。輸出資料集，可接 DataLoader → "
        "TrainingLoop，模型端用 Embedding → LSTM/GRU/RNN → SelectIndex → Linear。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="序列資料集：每筆是 (序列 (seq_len,) int64、標籤 int)。",
            ),
            PortDefinition(
                name="vocab_size",
                data_type=DataType.SCALAR,
                description=(
                    "這份資料用到的 Token 種類數（= n_classes + n_distractors）。"
                    "下游 Embedding 的 num_embeddings 至少要設成這個數。"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kind",
                param_type=ParamType.SELECT,
                default="recall_first",
                options=["recall_first", "recall_last"],
                description=(
                    "recall_first：答案在第 1 個位置，模型必須把它記到最後一步（依賴距離 = seq_len）。"
                    "recall_last：答案在最後一個位置（依賴距離 = 1）。"
                    "兩者其餘完全相同，只差『距離』——這是梯度消失對照實驗的唯一變因。"
                ),
            ),
            ParamDefinition(
                name="seq_len",
                param_type=ParamType.INT,
                default=60,
                description="每條序列的長度 T。調大會拉長依賴距離，純 RNN 會先撐不住。",
            ),
            ParamDefinition(
                name="n_samples",
                param_type=ParamType.INT,
                default=2000,
                description="產生幾筆序列。",
            ),
            ParamDefinition(
                name="n_classes",
                param_type=ParamType.INT,
                default=10,
                description=(
                    "答案有幾種（也就是分類的類別數）。Token 0 到 n_classes-1 是答案 Token，"
                    "同時也是標籤。亂猜的準確率是 1/n_classes。"
                ),
            ),
            ParamDefinition(
                name="n_distractors",
                param_type=ParamType.INT,
                default=10,
                description=(
                    "干擾 Token 有幾種。它們佔用 n_classes 之後的編號，本身不帶任何資訊，"
                    "唯一的作用是把答案和輸出隔開。"
                ),
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="隨機種子。訓練集與測試集請用不同的 seed。",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        kind = str(params.get("kind", "recall_first"))
        if kind not in ("recall_first", "recall_last"):
            raise ValueError(
                f"SyntheticSequence: unknown kind {kind!r}; expected 'recall_first' or 'recall_last'."
            )

        seq_len = int(params.get("seq_len", 60))
        n_samples = int(params.get("n_samples", 2000))
        n_classes = int(params.get("n_classes", 10))
        n_distractors = int(params.get("n_distractors", 10))
        seed = int(params.get("seed", 0))

        # Fail here, on the node the user can see, rather than deep inside
        # torch.randint with a message about `low >= high`.
        if seq_len < 2:
            raise ValueError(f"SyntheticSequence: seq_len must be >= 2, got {seq_len}.")
        if n_samples < 1:
            raise ValueError(f"SyntheticSequence: n_samples must be >= 1, got {n_samples}.")
        if n_classes < 2:
            raise ValueError(f"SyntheticSequence: n_classes must be >= 2, got {n_classes}.")
        if n_distractors < 1:
            raise ValueError(
                f"SyntheticSequence: n_distractors must be >= 1, got {n_distractors}."
            )

        dataset = _SyntheticSequenceDataset(
            n_samples=n_samples,
            seq_len=seq_len,
            n_classes=n_classes,
            n_distractors=n_distractors,
            kind=kind,
            seed=seed,
        )

        return {"dataset": dataset, "vocab_size": n_classes + n_distractors}
