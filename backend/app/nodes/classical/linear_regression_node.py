"""LinearRegressionNode — production OLS via ``sklearn.linear_model``.

Closed-form ordinary least squares. The math (``β = (XᵀX)⁻¹ Xᵀy``) is one
line; sklearn handles the numerics (SVD when ``X`` is rank-deficient, etc.).
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class LinearRegressionNode(BaseNode):
    NODE_NAME = "LinearRegression"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Ordinary-least-squares linear regression (sklearn). Closed-form fit, "
        "no iteration. Outputs coefficients, intercept, and predictions for a "
        "query set. The first line of every regression curriculum."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.TENSOR, description="Training targets [N] or [N, T]."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features [M, F]."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.TENSOR, description="Predicted targets for x_query."),
            PortDefinition(name="coef", data_type=DataType.TENSOR, description="Fitted coefficients (one per feature, possibly per target)."),
            PortDefinition(name="intercept", data_type=DataType.SCALAR, description="Fitted intercept (scalar for 1-D targets)."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="fit_intercept",
                param_type=ParamType.BOOL,
                default=True,
                description="If False, the regression line passes through the origin.",
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
        from sklearn.linear_model import LinearRegression

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("LinearRegression requires `x_train`, `y_train`, and `x_query` inputs.")

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(y_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)

        x_train_np = x_train.detach().cpu().float().numpy()
        x_query_np = x_query.detach().cpu().float().numpy()
        y_train_np = y_train.detach().cpu().float().numpy()

        if x_train_np.shape[0] != y_train_np.shape[0]:
            raise ValueError(
                f"LinearRegression: features and targets length mismatch — "
                f"{x_train_np.shape[0]} rows vs {y_train_np.shape[0]} targets."
            )

        fit_intercept = bool(params.get("fit_intercept", True))
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(x_train_np, y_train_np)
        preds_np = model.predict(x_query_np)

        intercept = model.intercept_
        if hasattr(intercept, "shape") and intercept.shape:
            intercept_out = torch.from_numpy(intercept).float()
        else:
            intercept_out = float(intercept)

        return {
            "predictions": torch.from_numpy(preds_np).float(),
            "coef": torch.from_numpy(model.coef_).float(),
            "intercept": intercept_out,
        }
