"""EduLinearRegressionNode — y = Xw + b, fit by closed form or gradient descent.

Two solver paths so the textbook can show both:

* ``closed_form`` — solve the normal equation
  ``w = (XᵀX + λI)⁻¹ Xᵀ y`` directly. Exact, fast, the answer that
  every other method approaches asymptotically.
* ``gradient_descent`` — iteratively update
  ``w ← w − lr · ∇L(w)`` for ``epochs`` steps. Slower but introduces
  the same machinery used by every neural network in the rest of the
  curriculum.

The ``regularization`` param applies L2 (ridge) penalty in both modes:
``L = ||Xw + b − y||² + λ||w||²``. Gradient descent operates on the
augmented loss; closed form solves the regularised normal equation.

Bias is handled by augmenting X with a column of ones internally — the
external ``weights`` output stays in feature dimensions and the bias is
returned separately for clarity.
"""

from __future__ import annotations

from typing import Any

import torch

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduLinearRegressionNode(BaseNode):
    NODE_NAME = "EduLinearRegression"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Linear regression $y = Xw + b$, solved either via the closed-form "
        "normal equation $w = (X^T X + \\lambda I)^{-1} X^T y$ or via "
        "gradient descent. L2 regularisation supported in both modes."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.TENSOR, description="Training targets [N] or [N, 1]."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features [M, F] to predict."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.TENSOR, description="Predicted y values [M]."),
            PortDefinition(name="weights", data_type=DataType.TENSOR, description="Learned weight vector [F]."),
            PortDefinition(name="bias", data_type=DataType.TENSOR, description="Learned scalar bias."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="method",
                param_type=ParamType.SELECT,
                default="closed_form",
                options=["closed_form", "gradient_descent"],
                description="closed_form solves the normal equation; gradient_descent runs iterative updates.",
            ),
            ParamDefinition(
                name="lr",
                param_type=ParamType.FLOAT,
                default=0.01,
                description="Step size for gradient descent. Ignored for closed_form.",
            ),
            ParamDefinition(
                name="epochs",
                param_type=ParamType.INT,
                default=100,
                description="Number of GD iterations. Ignored for closed_form.",
            ),
            ParamDefinition(
                name="regularization",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="L2 (ridge) regularisation strength λ.",
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
        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError(
                "EduLinearRegression requires `x_train`, `y_train`, `x_query` inputs."
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(y_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train = x_train.float()
        y_train = y_train.float().squeeze()
        x_query = x_query.float()

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"EduLinearRegression: features and targets length mismatch — "
                f"{x_train.shape[0]} rows vs {y_train.shape[0]} labels."
            )

        method = str(params.get("method", "closed_form"))
        lam = float(params.get("regularization", 0.0))

        n_features = x_train.shape[1]
        if method == "closed_form":
            # Augment with bias column.
            ones = torch.ones(x_train.shape[0], 1)
            X_aug = torch.cat([x_train, ones], dim=1)
            # Ridge-aware: λ on weights only, not the bias term — standard convention.
            reg = lam * torch.eye(n_features + 1)
            reg[-1, -1] = 0.0  # don't regularise bias
            try:
                theta = torch.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
            except torch.linalg.LinAlgError:
                # Fallback: use lstsq when the matrix is singular.
                theta = torch.linalg.lstsq(X_aug.T @ X_aug + reg, X_aug.T @ y_train).solution
            weights = theta[:n_features]
            bias = theta[n_features]
        elif method == "gradient_descent":
            lr = float(params.get("lr", 0.01))
            epochs = max(1, int(params.get("epochs", 100)))
            weights = torch.zeros(n_features)
            bias = torch.zeros(())
            n = x_train.shape[0]
            for _ in range(epochs):
                preds = x_train @ weights + bias
                err = preds - y_train
                grad_w = (2 / n) * x_train.T @ err + 2 * lam * weights
                grad_b = (2 / n) * err.sum()
                weights = weights - lr * grad_w
                bias = bias - lr * grad_b
        else:
            raise ValueError(f"EduLinearRegression: unknown method {method!r}")

        predictions = x_query @ weights + bias

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "fit",
                f"Solve via {method}.",
                weights=weights, bias=bias,
                scalars={"lambda": lam, "n_features": float(n_features)},
            )
            recorder.record(
                "predict",
                "y_pred = X_query @ w + b",
                predictions=predictions,
            )
            return {
                "predictions": predictions,
                "weights": weights,
                "bias": bias,
                "__steps__": recorder.steps,
            }

        return {"predictions": predictions, "weights": weights, "bias": bias}
