"""ScatterPlot2DNode — 把 2D 點雲依類別上色畫成散點圖（base64 PNG）。

給 **I2** 用：`SyntheticDataset` 輸出 `(N, 2)` 特徵 + 標籤，接這顆就能讓學生「看到
資料分布」——兩類點落在平面哪裡、能不能用一條直線分開。輸出與 `Visualize` 一致，
是一張 base64 編碼的 PNG，可直接在節點上預覽。
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


class ScatterPlot2DNode(BaseNode):
    NODE_NAME = "ScatterPlot2D"
    CATEGORY = "Utility"
    DESCRIPTION = (
        "把 (N, 2) 的 2D 點雲畫成散點圖，依 labels 給每一類不同顏色，輸出 base64 PNG。"
        "給分類資料看分布用：接在 SyntheticDataset 之後就能看到各類點落在平面哪裡。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="points", data_type=DataType.TENSOR, description="2D 點，shape [N, 2]。"),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="每點的類別標籤（長度 N）；沒接就全部畫成同一色。",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="image", data_type=DataType.STRING, description="base64 編碼的 PNG 散點圖。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
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
        import numpy as np

        pts = inputs.get("points")
        if pts is None:
            raise ValueError("ScatterPlot2D requires a `points` input of shape [N, 2].")
        if hasattr(pts, "detach"):
            pts = pts.detach().cpu().numpy()
        else:
            pts = np.asarray(pts, dtype=float)
        pts = np.atleast_2d(pts)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"ScatterPlot2D expects points of shape [N, 2]; got {tuple(pts.shape)}."
            )

        labels = inputs.get("labels")
        title = str(params.get("title", ""))

        fig, ax = plt.subplots(figsize=(6, 6))
        if labels is None:
            ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.8)
        else:
            labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
            labels_s = [str(l) for l in labels_list]
            classes = sorted(set(labels_s))
            cmap = plt.get_cmap("tab10")
            arr = np.asarray(labels_s)
            for i, c in enumerate(classes):
                mask = arr == c
                ax.scatter(pts[mask, 0], pts[mask, 1], s=18, alpha=0.8, color=cmap(i % 10), label=c)
            ax.legend(title="class", loc="best", fontsize=8)

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        if title:
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return {"image": base64.b64encode(buf.read()).decode("utf-8")}
