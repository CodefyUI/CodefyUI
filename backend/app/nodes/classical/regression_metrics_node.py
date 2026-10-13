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


def _as_matrix(values: Any, name: str) -> torch.Tensor:
    """Return ``values`` as a float64 ``[N, T]`` matrix (T = 1 for one output).

    ``[N]`` and ``[N, 1]`` both become ``[N, 1]``; trailing dims beyond the
    first are flattened into the output axis.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    else:
        try:
            tensor = torch.tensor(values, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"RegressionMetrics: `{name}` must be numbers ({exc})."
            ) from exc
    tensor = tensor.to(torch.float64)
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    if tensor.dim() == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(tensor.shape[0], -1)


class RegressionMetricsNode(BaseNode):
    NODE_NAME = "RegressionMetrics"
    CATEGORY = "Classical"
    DESCRIPTION = "Predictions vs true values: MSE, MAE and R²"
    DETAILS = (
        "MSE is the mean squared error, MAE the mean absolute error, and R² is "
        "1 - SS_res / SS_tot, where SS_tot is the spread of the targets around "
        "their mean: 1 is a perfect fit, 0 is no better than predicting the "
        "mean, and below 0 is worse. [N] and [N, 1] compare element by element. "
        "With several outputs ([N, T] predictions and targets of the same "
        "shape), R² is computed per output column and averaged, so outputs on "
        "different scales count equally; MSE and MAE average over every value. "
        "R² is NaN when every target in a column is the same. The three "
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
                description="Predicted values, [N], [N, 1] or [N, T].",
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
        y_hat = _as_matrix(preds, "predictions")
        y = _as_matrix(targets, "targets")
        if y_hat.numel() != y.numel():
            raise ValueError(
                f"RegressionMetrics: {y_hat.numel()} predictions vs "
                f"{y.numel()} targets; they must be the same length."
            )
        if y_hat.shape != y.shape:
            raise ValueError(
                f"RegressionMetrics: `predictions` has shape {list(y_hat.shape)} "
                f"but `targets` has shape {list(y.shape)}. Give both one row per "
                "sample and one column per output, e.g. [N] or [N, 1] for a "
                "single output and [N, T] for T outputs."
            )
        if y.numel() == 0:
            raise ValueError("RegressionMetrics: no values to compare.")

        error = y_hat - y
        mse = float((error ** 2).mean())
        mae = float(error.abs().mean())
        # R² per output column, then the unweighted mean (sklearn's
        # ``multioutput="uniform_average"``). Pooling every column into one
        # SS_tot would let the spread *between* columns of different scales
        # count as explained variance.
        ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        ss_res = (error ** 2).sum(dim=0)
        if bool((ss_tot > 0).all()):
            r2 = float((1.0 - ss_res / ss_tot).mean())
        else:
            r2 = float("nan")

        if context is not None:
            context.log_metric("mse", mse, 0)
            context.log_metric("mae", mae, 0)
            context.log_metric("r2", r2, 0)
        return {"mse": mse, "mae": mae, "r2": r2}
