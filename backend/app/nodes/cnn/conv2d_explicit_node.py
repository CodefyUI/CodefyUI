from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class Conv2dExplicitNode(BaseNode):
    NODE_NAME = "Conv2dExplicit"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "2D convolution with an *explicit* kernel supplied through an input "
        "port — no learnable weights, no random init. The same kernel is "
        "applied to every input channel via grouped (depthwise) convolution, "
        "so a (N, C, H, W) input produces a (N, C, H, W) output with channels "
        "untouched. Pair with Conv2dKernel to drop in classical filters "
        "(Laplacian / Sharpen / Prewitt-X) or any hand-edited matrix."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input tensor (N, C, H, W)",
            ),
            PortDefinition(
                name="kernel",
                data_type=DataType.TENSOR,
                description="Kernel tensor — accepted shapes: (k, k), (1, 1, k, k), or (C, 1, k, k)",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Convolved output (N, C, H_out, W_out)",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="stride",
                param_type=ParamType.INT,
                default=1,
                description="Convolution stride",
                min_value=1,
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=1,
                description="Zero-padding on each side of the spatial dims",
                min_value=0,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        tensor = inputs["tensor"]
        kernel = inputs["kernel"]

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Conv2dExplicit `tensor` input must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
        if not isinstance(kernel, torch.Tensor):
            raise ValueError(
                f"Conv2dExplicit `kernel` input must be a torch.Tensor, "
                f"got {type(kernel).__name__}"
            )
        if tensor.dim() != 4:
            raise ValueError(
                f"Conv2dExplicit `tensor` must be 4D (N, C, H, W); "
                f"got shape {list(tensor.shape)}. Use an Unsqueeze on dim=0 "
                f"if you have a (C, H, W) tensor."
            )

        c_in = tensor.size(1)

        # Normalise the kernel into (C, 1, k, k) for grouped (depthwise)
        # conv. The three supported source shapes:
        #   - (k, k)        — bare 2D matrix, broadcast to every channel
        #   - (1, 1, k, k)  — already conv-ready, broadcast to every channel
        #   - (C, 1, k, k)  — one kernel per channel (advanced)
        if kernel.dim() == 2:
            k_h, k_w = kernel.shape
            if k_h != k_w:
                raise ValueError(f"Kernel must be square; got {list(kernel.shape)}")
            base = kernel.reshape(1, 1, k_h, k_w)
            weight = base.expand(c_in, 1, k_h, k_w).contiguous()
        elif kernel.dim() == 4:
            kc_out, kc_in, k_h, k_w = kernel.shape
            if k_h != k_w:
                raise ValueError(f"Kernel must be square; got {list(kernel.shape)}")
            if kc_in != 1:
                raise ValueError(
                    f"Kernel inner channel must be 1 for depthwise conv; "
                    f"got shape {list(kernel.shape)}"
                )
            if kc_out == 1:
                weight = kernel.expand(c_in, 1, k_h, k_w).contiguous()
            elif kc_out == c_in:
                weight = kernel.contiguous()
            else:
                raise ValueError(
                    f"Kernel out-channel must be 1 (broadcast) or C={c_in} "
                    f"(one per channel); got shape {list(kernel.shape)}"
                )
        else:
            raise ValueError(
                f"Kernel must be 2D (k, k) or 4D (out, 1, k, k); "
                f"got shape {list(kernel.shape)}"
            )

        stride = max(1, int(params.get("stride", 1) or 1))
        padding = max(0, int(params.get("padding", 1) or 0))

        weight = weight.to(dtype=tensor.dtype, device=tensor.device)
        output = F.conv2d(
            tensor,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            groups=c_in,
        )
        return {"tensor": output}
