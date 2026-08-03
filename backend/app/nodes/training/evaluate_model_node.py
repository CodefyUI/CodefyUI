"""EvaluateModelNode — 算一個訓練好的分類模型在某個 dataset 上的準確率。

通用訓練流原本只到「訓練」（Dataset → DataLoader → Optimizer → Loss → TrainingLoop），
缺一塊「評估」：把訓練好的 model 拿來在測試集上跑一遍、看分對幾成。這顆補上那塊。

吃一個 `model`（任何吃 batch、吐 [B, C] logits 的分類網路）和一個 `dataset`
（如 Dataset 節點載入的 MNIST 測試集），內部建一個 DataLoader 跑完整個資料集，
對每筆取 argmax 當預測、跟標籤比，輸出準確率。對應教材 I2-4：訓練完 MNIST MLP 後
用它看驗證準確率有沒有到約 98%。
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


class EvaluateModelNode(BaseNode):
    NODE_NAME = "EvaluateModel"
    CATEGORY = "Training"
    DESCRIPTION = (
        "算訓練好的分類模型在一個 dataset 上的準確率。吃 model + dataset，內部建 "
        "DataLoader 跑完整個資料集、對每筆取 argmax 跟標籤比，輸出 accuracy / correct / total。"
        "補上通用訓練流缺的「評估」那一塊（對應 I2-4 看 MNIST 測試準確率）。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="訓練好的分類模型（吃 batch、吐 [B, C] logits）。"),
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="要評估的資料集（如 Dataset 節點的 MNIST test）。"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="accuracy", data_type=DataType.SCALAR, description="準確率，落在 [0, 1]。"),
            PortDefinition(name="correct", data_type=DataType.SCALAR, description="分對的筆數。"),
            PortDefinition(name="total", data_type=DataType.SCALAR, description="總筆數。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=256,
                min_value=1,
                description="評估時每批跑幾筆（不影響結果，只影響速度/記憶體）。",
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="cpu",
                options=["cpu", "cuda"],
                description="跑在 cpu 還是 cuda。",
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
        from torch.utils.data import DataLoader

        from ...core.device_utils import resolve_device
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
            raise ValueError("EvaluateModel requires a `model` input.")
        if dataset is None:
            raise ValueError("EvaluateModel requires a `dataset` input.")

        batch_size = max(1, int(params.get("batch_size", 256)))
        # Through ``resolve_device`` since #135 rather than an inline
        # ``device == "cuda"`` equality check: that check let ``cuda:1``
        # past the availability guard entirely, and an out-of-range index
        # reached ``.to()`` unvalidated.
        device = resolve_device(str(params.get("device", "cpu")))

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = model.to(device)
        model.eval()

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        total_batches = loader_length(loader)
        stopped_at_batch: int | None = None

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                # #122: a full test set is a long loop too, and an
                # uninterruptible evaluation is what makes Stop feel broken
                # right after the training it follows finally stopped.
                if should_stop():
                    stopped_at_batch = batch_index
                    break
                x, y = batch[0], batch[1]
                x = x.to(device)
                y = torch.as_tensor(y).to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
                throttle.emit({
                    "event": EVENT_BATCH,
                    "batch": batch_index + 1,
                    "total_batches": total_batches,
                    "accuracy": round(float(correct) / float(total), 6) if total else 0.0,
                })

        accuracy = float(correct) / float(total) if total else 0.0
        result: dict[str, Any] = {"accuracy": accuracy, "correct": int(correct),
                                  "total": int(total)}
        if stopped_at_batch is not None:
            # The partial counts are still returned -- "0.97 over the first
            # 40% of the set" beats nothing -- but the result says so, and
            # an incomplete pass is deliberately NOT filed as a measured
            # accuracy.
            result.update(interrupted_result(batch=stopped_at_batch))
        elif context is not None:
            context.log_metric("eval_accuracy", accuracy, 1)
        return result
