from typing import Any

from ...core.node_base import (
    MEDIA_IMAGE,
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


#: The most rows a 2-D ``[K, N]`` line input may have to be drawn as K
#: separate series. Past it a legend stops being readable, so the data is
#: flattened into one curve as before, with a note in the log.
MAX_LINE_SERIES = 12


def _tile_images(imgs: Any, np: Any) -> Any:
    """Lay a batch of (N, H, W) or (N, H, W, C) images out on one grid."""
    n = imgs.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    ih, iw = imgs.shape[1], imgs.shape[2]
    pad = 1
    fill = float(imgs.min())
    size = (rows * ih + (rows + 1) * pad, cols * iw + (cols + 1) * pad)
    if imgs.ndim == 4:
        size = (*size, imgs.shape[-1])
    grid = np.full(size, fill, dtype=imgs.dtype)
    for idx in range(n):
        r, cc = idx // cols, idx % cols
        y0, x0 = pad + r * (ih + pad), pad + cc * (iw + pad)
        grid[y0:y0 + ih, x0:x0 + iw] = imgs[idx]
    return grid


class VisualizeNode(BaseNode):
    NODE_NAME = "Visualize"
    CATEGORY = "Utility"
    DESCRIPTION = "Plot data as a line, histogram, heatmap or image"
    DETAILS = (
        "The plot is rendered with matplotlib. Line draws a 2-D [K, N] input "
        f"as K series with a legend when K is at most {MAX_LINE_SERIES} "
        "(series 0 to series K-1, one per row), and flattens anything else "
        "into one curve, as histogram does. Heatmap folds a 1D input into a square "
        "grid, and image tiles a batch of (N, C, H, W) or (N, H, W) into a grid."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="data",
                data_type=DataType.ANY,
                description=(
                    "Data to visualize (tensor, list, or numpy array). For a "
                    f"line plot, each row of a [K, N] tensor with K <= "
                    f"{MAX_LINE_SERIES} is its own series"
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="image",
                data_type=DataType.STRING,
                description="Base64-encoded PNG image string",
                media=MEDIA_IMAGE,
            ),
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
            data = data.detach().cpu()
            # numpy has no bfloat16, and matplotlib clips only float32/64:
            # widen the half-precision floats before converting.
            if data.is_floating_point() and data.element_size() < 4:
                data = data.float()
            data = data.numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        fig, ax = plt.subplots(figsize=(8, 6))
        log_lines: list[str] = []

        if plot_type == "line":
            rows = data.shape[0] if data.ndim == 2 else 0
            if data.ndim == 2 and 1 < rows <= MAX_LINE_SERIES and data.shape[1] > 1:
                # [K, N]: one series per row, e.g. stacked loss curves.
                for k in range(rows):
                    ax.plot(data[k], label=f"series {k}")
                ax.legend()
            elif data.ndim == 1:
                ax.plot(data)
            else:
                # [N, 1], [1, N] and everything else flatten to one curve.
                if data.ndim == 2 and rows > MAX_LINE_SERIES and data.shape[1] > 1:
                    log_lines.append(
                        f"Visualize: a [{rows}, {data.shape[1]}] tensor has more "
                        f"than {MAX_LINE_SERIES} rows, so it is drawn as one "
                        f"flattened curve; pass at most {MAX_LINE_SERIES} rows "
                        f"to draw one series per row."
                    )
                ax.plot(data.flatten())
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
            # A batch of images (N,C,H,W): N==1 -> drop batch dim; N>1 -> tile into a grid.
            # A 3-D (N,H,W) batch of grayscale images (MNIST's layout) tiles
            # the same way; it is told apart from (C,H,W) and (H,W,C) by
            # neither end being a channel count.
            if data.ndim == 4:
                if data.shape[0] == 1:
                    data = data[0]
                else:
                    c = data.shape[1]
                    if c in (1, 3, 4):
                        imgs = np.transpose(data, (0, 2, 3, 1))  # (N,H,W,C)
                        if imgs.shape[-1] == 1:
                            imgs = imgs[..., 0]                  # (N,H,W)
                    else:
                        imgs = data[:, 0]                        # take first channel
                    data = _tile_images(imgs, np)
            elif (
                data.ndim == 3
                and data.shape[0] not in (1, 3, 4)
                and data.shape[2] not in (1, 3, 4)
            ):
                data = _tile_images(data, np)
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

        result: dict[str, Any] = {"image": image_base64}
        if log_lines:
            result["__log__"] = "\n".join(log_lines)
        return result
