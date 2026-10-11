"""RegressionMetricsNode — MSE, MAE and R² between predictions and targets.

The regression counterpart of ``Accuracy`` (#712). It closes a regression
pipeline, classical or neural:

    TrainTestSplit → LinearRegression → RegressionMetrics
    TrainTestSplit → TensorDataset → … → TrainingLoop → Inference → RegressionMetrics

and records the three numbers as run metrics, so a sweep can rank by any
of them (#711).
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    PortDefinition,
)


def _as_column(values: Any, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    else:
        try:
            tensor = torch.tensor([float(v) for v in values], dtype=torch.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RegressionMetrics: `{name}` must be numbers ({exc})."
            ) from exc
    return tensor.to(torch.float64).reshape(-1)


class RegressionMetricsNode(BaseNode):
    NODE_NAME = "RegressionMetrics"
    CATEGORY = "Classical"
    DESCRIPTION = "Predictions vs true values: MSE, MAE and R²"
    DETAILS = (
        "MSE is the mean squared error, MAE the mean absolute error, and R² is "
        "1 - SS_res / SS_tot, where SS_tot is the spread of the targets around "
        "their mean: 1 is a perfect fit, 0 is no better than predicting the "
        "mean, and below 0 is worse. Predictions and targets are flattened, so "
        "[N] and [N, 1] compare element by element; they must hold the same "
        "number of values. R² is NaN when every target is the same. The three "
        "values are also recorded as the run metrics `mse`, `mae` and `r2`, so "
        "a sweep can rank by them."
    )

    # The metrics go out through ``context.log_metric`` as well as the ports,
    # and a cache hit skips execute(), so they would be missing on a rerun.
    cacheable = False
    # Host-side arithmetic on what may be a long target list.
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.TENSOR,
                description="Predicted values, [N] or [N, 1].",
            ),
            PortDefinition(
                name="targets",
                data_type=DataType.ANY,
                description="True values: a LIST of numbers or a TENSOR.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="mse", data_type=DataType.SCALAR, description="Mean squared error."),
            PortDefinition(name="mae", data_type=DataType.SCALAR, description="Mean absolute error."),
            PortDefinition(name="r2", data_type=DataType.SCALAR, description="Coefficient of determination."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        preds = inputs.get("predictions")
        targets = inputs.get("targets")
        if preds is None or targets is None:
            raise ValueError("RegressionMetrics requires `predictions` and `targets` inputs.")
        y_hat = _as_column(preds, "predictions")
        y = _as_column(targets, "targets")
        if y_hat.numel() != y.numel():
            raise ValueError(
                f"RegressionMetrics: {y_hat.numel()} predictions vs "
                f"{y.numel()} targets; they must be the same length."
            )
        if y.numel() == 0:
            raise ValueError("RegressionMetrics: no values to compare.")

        error = y_hat - y
        mse = float((error ** 2).mean())
        mae = float(error.abs().mean())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float((error ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        if context is not None:
            context.log_metric("mse", mse, 0)
            context.log_metric("mae", mae, 0)
            context.log_metric("r2", r2, 0)
        return {"mse": mse, "mae": mae, "r2": r2}
