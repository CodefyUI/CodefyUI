"""DecisionBoundaryNode — 畫一個已訓練分類器的 2D 決策邊界（base64 PNG）。

給 **I2** 用：把整個 2D 平面鋪成密格、每一點都丟給分類器問「這裡會被分成哪一
類」，染色後就看到分類器的「形狀偏好」——KNN 的邊界彎彎曲曲貼著資料、線性 /
邏輯斯是一條直線。這正是「準確率數字一樣、但邊界形狀不同」唯一看得出來的地方。

它**吃 `Classifier` 節點吐出的 `model`**（接在分類器後面），所以換 Classifier 的
kind、邊界就跟著換——真正做到「只改一個下拉選單」。另外吃 `x_train` / `y_train`
用來決定畫布範圍、並把訓練點疊在邊界上。
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


class DecisionBoundaryNode(BaseNode):
    NODE_NAME = "DecisionBoundary"
    CATEGORY = "Utility"
    DESCRIPTION = (
        "畫已訓練分類器的 2D 決策邊界：吃 Classifier 的 model，把平面鋪成密格、每點"
        "問分類器分到哪一類，染色成區域，再疊上訓練點。接在 Classifier 後面，換 kind "
        "邊界就跟著換。看「同一份資料、不同分類器」的邊界形狀差異。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Classifier 節點吐出的 model（訓練好的分類器）。"),
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="訓練特徵 [N, 2]（決定畫布範圍、疊點；只支援 2D）。"),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="訓練標籤（長度 N，給疊點上色）。"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="image", data_type=DataType.STRING, description="base64 編碼的 PNG 決策邊界圖。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="grid_steps",
                param_type=ParamType.INT,
                default=200,
                min_value=20,
                max_value=500,
                description="每軸鋪幾格（越大邊界越平滑、越慢）。",
            ),
            ParamDefinition(name="title", param_type=ParamType.STRING, default="", description="圖標題。"),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        model = inputs.get("model")
        x = inputs.get("x_train")
        y = inputs.get("y_train")
        if model is None or not hasattr(model, "predict"):
            raise ValueError(
                "DecisionBoundary needs a `model` input — connect a Classifier node's `model` output."
            )
        if x is None or y is None:
            raise ValueError("DecisionBoundary requires `x_train` and `y_train` inputs.")

        if hasattr(x, "detach"):
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=float)
        x_np = np.atleast_2d(x_np)
        if x_np.ndim != 2 or x_np.shape[1] != 2:
            raise ValueError(
                f"DecisionBoundary only supports 2D features [N, 2]; got {tuple(x_np.shape)}."
            )
        labels = y.tolist() if hasattr(y, "tolist") else list(y)
        labels = [str(l) for l in labels]

        steps = max(20, min(500, int(params.get("grid_steps", 200))))
        title = str(params.get("title", ""))

        classes = sorted(set(labels))
        idx = {c: i for i, c in enumerate(classes)}

        pad_x = 0.5 + 0.05 * (x_np[:, 0].max() - x_np[:, 0].min())
        pad_y = 0.5 + 0.05 * (x_np[:, 1].max() - x_np[:, 1].min())
        x_min, x_max = x_np[:, 0].min() - pad_x, x_np[:, 0].max() + pad_x
        y_min, y_max = x_np[:, 1].min() - pad_y, x_np[:, 1].max() + pad_y
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, steps),
            np.linspace(y_min, y_max, steps),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        grid_pred = model.predict(grid)
        zz = np.array([idx.get(str(p), 0) for p in grid_pred]).reshape(xx.shape)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(classes))]
        region_cmap = ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.contourf(xx, yy, zz, levels=np.arange(-0.5, len(classes), 1.0), cmap=region_cmap, alpha=0.3)
        arr = np.asarray(labels)
        for i, c in enumerate(classes):
            mask = arr == c
            ax.scatter(x_np[mask, 0], x_np[mask, 1], s=18, color=colors[i], edgecolor="white", linewidth=0.4, label=c)
        ax.legend(title="class", loc="best", fontsize=8)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if title:
            ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return {"image": base64.b64encode(buf.read()).decode("utf-8")}
