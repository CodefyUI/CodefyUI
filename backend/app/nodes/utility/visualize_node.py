from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class VisualizeNode(BaseNode):
    NODE_NAME = "Visualize"
    CATEGORY = "Utility"
    DESCRIPTION = "Generate a matplotlib plot of data (tensor, losses, etc.) as a base64-encoded PNG"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="data", data_type=DataType.ANY, description="Data to visualize (tensor, list, or numpy array)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="image", data_type=DataType.STRING, description="Base64-encoded PNG image string"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="title", param_type=ParamType.STRING, default="", description="Plot title"),
            ParamDefinition(
                name="plot_type",
                param_type=ParamType.SELECT,
                default="line",
                description="Type of plot to generate",
                options=["line", "histogram", "heatmap", "image"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        data = inputs["data"]
        title = params.get("title", "")
        plot_type = params.get("plot_type", "line")

        # Convert to numpy if it is a torch tensor
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == "line":
            if data.ndim == 1:
                ax.plot(data)
            else:
                # Plot each row or flatten for multi-dim
                flat = data.flatten()
                ax.plot(flat)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")

        elif plot_type == "histogram":
            flat = data.flatten()
            ax.hist(flat, bins=50, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        elif plot_type == "heatmap":
            if data.ndim == 1:
                # Reshape 1D into a square-ish 2D grid for display
                side = int(np.ceil(np.sqrt(len(data))))
                padded = np.zeros(side * side)
                padded[: len(data)] = data
                data = padded.reshape(side, side)
            elif data.ndim > 2:
                # Take first 2D slice
                data = data.reshape(-1, data.shape[-1])
            im = ax.imshow(data, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax)

        elif plot_type == "image":
            # Drop a leading batch dim of 1, e.g. (1,C,H,W) -> (C,H,W)
            if data.ndim == 4 and data.shape[0] == 1:
                data = data[0]
            # Render an image tensor directly: (C,H,W), (H,W,C), or (H,W)
            if data.ndim == 3 and data.shape[0] in (1, 3, 4):
                # (C,H,W) -> (H,W,C)
                data = np.transpose(data, (1, 2, 0))
            if data.ndim == 3 and data.shape[2] == 1:
                data = data.squeeze(2)
            # Clamp to [0,1] for float, or leave as-is for uint8
            if data.dtype in (np.float32, np.float64):
                data = np.clip(data, 0.0, 1.0)
            ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
            ax.axis("off")

        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        if title:
            ax.set_title(title)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"image": image_base64}
