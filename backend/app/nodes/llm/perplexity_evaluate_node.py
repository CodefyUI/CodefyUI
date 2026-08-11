"""PerplexityEvaluateNode — 語言模型的驗證損失與困惑度。

`EvaluateModel` 是 argmax 準確率（分類專用），對語言模型沒有意義。這顆吃
訓練好的 causal LM 和打包好的 `(input_ids, labels)` 驗證資料，跑一遍計算
token 加權的平均 cross-entropy 與 perplexity（``exp(val_loss)``）— LM 訓練
的標準評估數字。標籤為 -100 的位置不計入。
"""

from __future__ import annotations

from typing import Any

from ...core.amp import PRECISIONS
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class PerplexityEvaluateNode(BaseNode):
    NODE_NAME = "PerplexityEvaluate"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Evaluate a causal language model on packed (input_ids, labels) "
        "blocks: token-weighted mean cross-entropy (val_loss) and perplexity "
        "= exp(val_loss). Positions labeled -100 are ignored. Wire "
        "TrainingLoop.model and a validation LMTokenizedDataset in; "
        "perplexity is per-token and only comparable on the same dataset "
        "and tokenizer."
    )

    # Same reasoning as EvaluateModel (#254): the measurement depends on the
    # model's live WEIGHTS (which the cache key cannot see), and the node
    # logs val_loss/perplexity metric points — a side effect a cache replay
    # would silently drop.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL,
                           description="Trained causal LM (input_ids (B,T) -> logits (B,T,V))"),
            PortDefinition(name="dataset", data_type=DataType.DATASET,
                           description="Packed (input_ids, labels) blocks (LMTokenizedDataset output)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="val_loss", data_type=DataType.SCALAR,
                           description="Token-weighted mean cross-entropy"),
            PortDefinition(name="perplexity", data_type=DataType.SCALAR,
                           description="exp(val_loss)"),
            PortDefinition(name="tokens", data_type=DataType.SCALAR,
                           description="Number of tokens scored (ignore_index positions excluded)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="batch_size", param_type=ParamType.INT, default=8,
                min_value=1, max_value=256,
                description="Evaluation batch size (speed/memory only; the result is identical)",
            ),
            ParamDefinition(
                name="max_batches", param_type=ParamType.INT, default=0,
                min_value=0,
                description="Evaluate at most N batches (0 = the whole dataset)",
            ),
            ParamDefinition(
                name="device", param_type=ParamType.SELECT, default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description="Device to evaluate on ('auto' follows the global device)",
            ),
            ParamDefinition(
                name="precision", param_type=ParamType.SELECT, default="bf16",
                options=list(PRECISIONS), advanced=True,
                description=(
                    "Autocast precision for the forward pass. bf16 halves "
                    "activation memory on Ampere+ and is the LM-eval "
                    "default; a device that cannot honour it falls back to "
                    "fp32 and says so. fp32 is the number to publish."
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
        import math

        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        from ...core.amp import AmpPolicy
        from ...core.device_utils import resolve_node_device
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            loader_length,
            stop_checker,
        )

        model = inputs.get("model")
        dataset = inputs.get("dataset")
        if model is None:
            raise ValueError("PerplexityEvaluate requires a `model` input.")
        if dataset is None:
            raise ValueError("PerplexityEvaluate requires a `dataset` input.")

        batch_size = max(1, int(params.get("batch_size", 8)))
        max_batches = max(0, int(params.get("max_batches", 0) or 0))
        device = resolve_node_device(params.get("device"), context)
        policy = AmpPolicy.for_device(params.get("precision", "bf16"), device)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = model.to(device)
        model.eval()

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        total_batches = loader_length(loader)
        if max_batches and total_batches is not None:
            total_batches = min(total_batches, max_batches)
        stopped_at_batch: int | None = None

        loss_sum = 0.0
        token_count = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches and batch_index >= max_batches:
                    break
                if should_stop():
                    stopped_at_batch = batch_index
                    break
                input_ids = batch[0].to(device)
                targets = batch[1].to(device)
                with policy.autocast():
                    logits = model(input_ids)
                # Sum-reduced CE in fp32 so the running total is exact;
                # ignore_index positions contribute neither loss nor count.
                flat_logits = logits.reshape(-1, logits.shape[-1]).float()
                flat_targets = targets.reshape(-1)
                loss_sum += float(F.cross_entropy(
                    flat_logits, flat_targets,
                    ignore_index=-100, reduction="sum",
                ).item())
                token_count += int((flat_targets != -100).sum().item())
                running = loss_sum / token_count if token_count else 0.0
                throttle.emit({
                    "event": EVENT_BATCH,
                    "batch": batch_index + 1,
                    "total_batches": total_batches,
                    "val_loss": round(running, 6),
                })

        if token_count == 0:
            raise RuntimeError(
                "PerplexityEvaluate scored zero tokens — the dataset is "
                "empty or every label is ignore_index (-100)."
            )
        val_loss = loss_sum / token_count
        # exp() overflows float64 beyond ~709; an untrained model's CE can
        # legitimately sit near ln(vocab) ~ 11, but a broken run can hand us
        # anything. Cap the exponent and say nothing — the val_loss output
        # carries the exact number either way.
        perplexity = math.exp(min(val_loss, 700.0))

        result: dict[str, Any] = {
            "val_loss": float(val_loss),
            "perplexity": float(perplexity),
            "tokens": int(token_count),
        }
        if stopped_at_batch is not None:
            result.update(interrupted_result(batch=stopped_at_batch))
        elif context is not None:
            context.log_metric("val_loss", float(val_loss), 1)
            context.log_metric("perplexity", float(perplexity), 1)
        return result
