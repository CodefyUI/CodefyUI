"""TensorStatsNode -- one node that says what a tensor is.

Shape, element count, min, max, mean, sum and standard deviation, each on
its own port, plus a one-block ``report`` that also goes to the Execution
Log. The tensor itself passes straight through, so the node can sit inline
on any edge without rewiring what comes after it.

Why this exists: checking a tensor on the canvas used to take a ``Mean``
wired into a ``Print``, then deleting both before saving -- and that is
what a competition problem's "reference result" section had to tell every
student to do. Now the statement says "a TensorStats after MaxPool2d shows
mean=0.076823" and that is the whole check. The scalar ports exist so a
judge can compare a signature (say, sum and max) without shipping the
full tensor, and so a downstream ``GraphOutput`` can carry one number.

Standard deviation is the population form (``unbiased=False``). A
summary node reports the data it was given; it does not estimate a
population the data was sampled from, and the biased form is the one a
student reproduces by hand from the definition.
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class TensorStatsNode(BaseNode):
    NODE_NAME = "TensorStats"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Shape, min, max, mean, sum and std of a tensor"
    DETAILS = (
        "Each statistic has its own port, and `report` also goes to the "
        "Execution Log. The input passes through unchanged, so the node can be dropped onto "
        "any edge. Statistics cover the whole tensor; integer and boolean "
        "input is promoted to float32 first. `std` is the population value "
        "(divides by N, not N-1). `precision` only affects the text in `report`; "
        "the scalar ports carry the full-precision numbers."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Tensor to summarise"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="The input, unchanged"),
            PortDefinition(name="shape", data_type=DataType.STRING, description="Shape as text, e.g. (1, 2, 48, 48)"),
            PortDefinition(name="numel", data_type=DataType.SCALAR, description="Number of elements"),
            PortDefinition(name="min", data_type=DataType.SCALAR, description="Smallest element"),
            PortDefinition(name="max", data_type=DataType.SCALAR, description="Largest element"),
            PortDefinition(name="mean", data_type=DataType.SCALAR, description="Mean of all elements"),
            PortDefinition(name="sum", data_type=DataType.SCALAR, description="Sum of all elements"),
            PortDefinition(name="std", data_type=DataType.SCALAR, description="Population standard deviation (unbiased=False)"),
            PortDefinition(name="report", data_type=DataType.STRING, description="All of the above as one block of text"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="precision",
                param_type=ParamType.INT,
                default=6,
                min_value=0,
                max_value=12,
                description="Decimal places used in `report` (the scalar ports are not rounded)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        value = inputs.get("tensor")
        if value is None:
            raise ValueError("TensorStats requires a `tensor` input.")

        # Lists and plain numbers arrive from Python-side nodes; accept them
        # rather than make the student add a converter for a summary.
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.numel() == 0:
            raise ValueError(
                f"TensorStats: the tensor is empty (shape {tuple(tensor.shape)}); "
                "min/max/mean are undefined."
            )

        # Integer and boolean tensors have no mean/std in torch; promote them.
        # A floating tensor keeps its own dtype so the reported mean is the
        # same number a Mean node on the same edge would produce.
        stats_src = tensor.detach()
        if not stats_src.is_floating_point():
            stats_src = stats_src.to(torch.float32)

        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        t_min = float(stats_src.min())
        t_max = float(stats_src.max())
        t_mean = float(stats_src.mean())
        t_sum = float(stats_src.sum())
        t_std = float(stats_src.std(unbiased=False)) if numel > 1 else 0.0

        try:
            precision = int(params.get("precision", 6))
        except (TypeError, ValueError):
            precision = 6
        precision = max(0, min(12, precision))

        def fmt(x: float) -> str:
            return f"{x:.{precision}f}"

        report = (
            f"shape={shape} numel={numel}\n"
            f"min={fmt(t_min)} max={fmt(t_max)} mean={fmt(t_mean)} "
            f"sum={fmt(t_sum)} std={fmt(t_std)}"
        )

        return {
            "tensor": value,
            "shape": str(shape),
            "numel": numel,
            "min": t_min,
            "max": t_max,
            "mean": t_mean,
            "sum": t_sum,
            "std": t_std,
            "report": report,
            # The only result key the canvas Log tab renders: a student sees
            # the numbers on Run, with nothing else wired in.
            "__log__": report,
        }
