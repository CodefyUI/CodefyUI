"""LMCrossEntropyLossNode — 語言模型用的 next-token cross-entropy。

`Loss` 節點給的 ``nn.CrossEntropyLoss`` 期望 logits ``(N, C)`` 或
``(N, C, d1, ...)``，但 causal LM 的 forward 吐 ``(B, T, vocab)``、標籤是
``(B, T)`` — 形狀直接餵會炸。這顆輸出一個會自己攤平的 LOSS_FN：
``loss(logits (B,T,V), targets (B,T)) = mean CE over B*T``，含 ``ignore_index``
與 label smoothing，接上既有 TrainingLoop（含它的 val-loss 路徑與 bf16
autocast）就能訓練語言模型。

損失類別本身住在 ``_lm_modules.py``（module-scope、可被 full_model pickle，
#283），這裡只在 ``execute`` 內延遲 import，讓 registry 掃描不用付 torch 的
啟動成本。
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


class LMCrossEntropyLossNode(BaseNode):
    NODE_NAME = "LMCrossEntropyLoss"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Next-token cross-entropy loss for language models: accepts logits "
        "(B,T,vocab) against targets (B,T) by flattening internally — the "
        "shapes CausalLMModel and LMTokenizedDataset produce. Supports "
        "ignore_index and label smoothing. Use this instead of the generic "
        "Loss node's CrossEntropyLoss for LM training."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="loss_fn",
                data_type=DataType.LOSS_FN,
                description="Callable loss(logits (B,T,V) or (N,V), targets) -> scalar mean CE",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="ignore_index",
                param_type=ParamType.INT,
                default=-100,
                description="Target positions with this value contribute no loss (padding convention)",
            ),
            ParamDefinition(
                name="label_smoothing",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=0.3,
                advanced=True,
                description="Label smoothing factor (0 = off)",
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
        from ._lm_modules import LMCrossEntropy

        loss_fn = LMCrossEntropy(
            ignore_index=int(params.get("ignore_index", -100)),
            label_smoothing=float(params.get("label_smoothing", 0.0)),
        )
        return {"loss_fn": loss_fn}
