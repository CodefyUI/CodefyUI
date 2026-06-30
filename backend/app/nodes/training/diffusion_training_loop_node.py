"""DiffusionTrainingLoopNode — train a noise-predicting U-Net (DDPM).

The generic TrainingLoop does supervised ``loss(model(x), y)``; diffusion
training is different: for each image we pick a random timestep, add that
much noise, and ask the model to predict the noise we added. This node
packages that loop so a small diffusion model can be trained on CPU, then
sampled with ``DDPMSampler``.

CRITICAL: the noise schedule here (``schedule`` / ``num_timesteps`` /
``beta_start`` / ``beta_end``) must MATCH the ``DDPMSampler`` used to sample
afterwards, or generation will fail. The defaults (linear, T=160,
beta_end=0.05) are chosen so the schedule both fully noises the image
(alpha_bar_T ~ 0) and stays numerically stable in the reverse loop.
"""

import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class DiffusionTrainingLoopNode(BaseNode):
    NODE_NAME = "DiffusionTrainingLoop"
    CATEGORY = "Diffusion"
    cacheable = False
    DESCRIPTION = (
        "訓練一個會去雜訊的 U-Net（DDPM 訓練）。每一步：從資料取一張乾淨圖、隨機挑一個時間步 t、"
        "加上對應強度的雜訊、讓模型預測『剛剛加了什麼雜訊』，用 MSE 比對更新權重。訓練好的 model "
        "接 DDPMSampler 就能從純雜訊生成新圖。"
        "注意：這裡的雜訊排程（schedule / num_timesteps / beta_start / beta_end）必須和之後取樣的 "
        "DDPMSampler 設成一樣，否則生成會壞掉。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="會去雜訊的 U-Net（如 DiffusionUNet），forward 吃 (x, t) 吐預測雜訊。"),
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="乾淨影像資料集（如 SyntheticShapes），每筆 (影像, 標籤)，標籤會被忽略。"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="訓練好的去雜訊模型。"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="每個 epoch 的平均訓練 loss。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="epochs", param_type=ParamType.INT, default=200, min_value=1, description="訓練幾輪。"),
            ParamDefinition(name="batch_size", param_type=ParamType.INT, default=32, min_value=1, description="每批幾張圖。"),
            ParamDefinition(name="lr", param_type=ParamType.FLOAT, default=0.002, min_value=0.0, description="學習率（Adam）。"),
            ParamDefinition(
                name="num_timesteps", param_type=ParamType.INT, default=160, min_value=2,
                description="擴散總步數 T。要和取樣的 DDPMSampler 的 num_steps 一致。",
            ),
            ParamDefinition(
                name="schedule", param_type=ParamType.SELECT, default="linear", options=["linear", "cosine"],
                description="雜訊排程。要和 DDPMSampler 一致。linear + beta_end=0.05 在小 T 下既能噪滿又穩定。",
            ),
            ParamDefinition(name="beta_start", param_type=ParamType.FLOAT, default=0.0001, min_value=0.0, description="linear 排程起始 beta（要和 DDPMSampler 一致）。"),
            ParamDefinition(name="beta_end", param_type=ParamType.FLOAT, default=0.05, min_value=0.0, description="linear 排程結束 beta（要和 DDPMSampler 一致）。"),
            ParamDefinition(name="device", param_type=ParamType.SELECT, default="cpu", options=["cpu", "cuda"], description="訓練裝置。"),
            ParamDefinition(name="seed", param_type=ParamType.INT, default=0, description="亂數種子（決定每步挑的時間步與加的雜訊）。"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], progress_callback: Any | None = None) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        from ..diffusion.ddpm_sampler_node import _cosine_betas, _linear_betas

        model = inputs.get("model")
        dataset = inputs.get("dataset")
        if model is None:
            raise ValueError("DiffusionTrainingLoop requires a `model` input.")
        if dataset is None:
            raise ValueError("DiffusionTrainingLoop requires a `dataset` input.")

        epochs = int(params.get("epochs", 200))
        batch_size = max(1, int(params.get("batch_size", 32)))
        lr = float(params.get("lr", 0.002))
        T = int(params.get("num_timesteps", 160))
        schedule = str(params.get("schedule", "linear"))
        beta_start = float(params.get("beta_start", 0.0001))
        beta_end = float(params.get("beta_end", 0.05))
        device = str(params.get("device", "cpu"))
        seed = int(params.get("seed", 0))
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        torch.manual_seed(seed)
        if schedule == "cosine":
            betas = _cosine_betas(T)
        else:
            betas = _linear_betas(T, beta_start, beta_end)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0).to(device)

        model = model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            running, batches = 0.0, 0
            for batch in loader:
                x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                x0 = x0.to(device).float()
                b = x0.shape[0]
                t = torch.randint(0, T, (b,), device=device)
                eps = torch.randn_like(x0)
                ab = alpha_bars[t].view(b, *([1] * (x0.dim() - 1)))
                x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
                eps_hat = model(x_t, t)
                loss = ((eps_hat - eps) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
                batches += 1
            avg = running / max(batches, 1)
            epoch_losses.append(avg)
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                logger.info("Diffusion epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg)
            if progress_callback:
                progress_callback({"event": "epoch", "epoch": epoch + 1, "total_epochs": epochs,
                                   "loss": round(avg, 6), "losses": [round(l, 6) for l in epoch_losses]})

        losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32)
        return {"model": model, "losses": losses_tensor}
