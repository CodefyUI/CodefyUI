"""EduLogisticRegressionNode — softmax classifier fitted by gradient descent.

The natural multi-class generalisation of binary logistic regression:

    logits[i, c] = X[i] @ W[:, c] + b[c]
    P(class=c | x_i) = exp(logits[i, c]) / Σ_k exp(logits[i, k])

Loss is cross-entropy with optional L2 weight decay on W (not on b):

    L = − Σ_i log P(y_i | x_i) + λ ||W||_F²

Gradient descent updates W and b together; for binary problems this
collapses to two-column softmax which is mathematically equivalent to
the sigmoid form. We always use softmax so the same code handles both.

Outputs:
- ``predictions`` — predicted *string* labels (matches the input label
  type, so a downstream node sees the same vocabulary).
- ``probabilities`` — full softmax distribution per query, [M, C].
- ``weights`` — [F, C].
- ``bias`` — [C].
- ``classes`` — the unique labels in fit order; column k of weights
  corresponds to ``classes[k]``.
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


class EduLogisticRegressionNode(BaseNode):
    NODE_NAME = "Edu-LogisticRegression"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Softmax classifier fitted by gradient descent on cross-entropy loss. "
        "Handles binary and multi-class with the same code. Optional L2 "
        "regularisation on weights (not bias). Predictions are string labels "
        "matching the training label vocabulary."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="Training labels (length N)."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features [M, F] to classify."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="Predicted labels per query."),
            PortDefinition(name="probabilities", data_type=DataType.TENSOR, description="Softmax probabilities [M, C]."),
            PortDefinition(name="weights", data_type=DataType.TENSOR, description="Learned weight matrix [F, C]."),
            PortDefinition(name="bias", data_type=DataType.TENSOR, description="Learned bias vector [C]."),
            PortDefinition(name="classes", data_type=DataType.LIST, description="Unique class labels in column order."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="lr", param_type=ParamType.FLOAT, default=0.1, description="Gradient-descent step size."),
            ParamDefinition(name="epochs", param_type=ParamType.INT, default=200, description="Training iterations."),
            ParamDefinition(
                name="regularization",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="L2 weight decay strength λ.",
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
                "EduLogisticRegression requires `x_train`, `y_train`, `x_query` inputs."
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train = x_train.float()
        x_query = x_query.float()
        labels = list(y_train) if not isinstance(y_train, torch.Tensor) else y_train.tolist()
        labels = [str(v) for v in labels]

        if x_train.shape[0] != len(labels):
            raise ValueError(
                f"EduLogisticRegression: features/labels length mismatch — "
                f"{x_train.shape[0]} rows vs {len(labels)} labels."
            )

        # Build a class index from the unique labels.
        classes = sorted(set(labels))
        if len(classes) < 2:
            raise ValueError(
                f"EduLogisticRegression: need at least 2 classes, got {classes!r}."
            )
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = torch.tensor([class_to_idx[v] for v in labels], dtype=torch.long)

        n_features = x_train.shape[1]
        n_classes = len(classes)
        lr = float(params.get("lr", 0.1))
        epochs = max(1, int(params.get("epochs", 200)))
        lam = float(params.get("regularization", 0.0))

        weights = torch.zeros(n_features, n_classes)
        bias = torch.zeros(n_classes)
        n = x_train.shape[0]

        for _ in range(epochs):
            logits = x_train @ weights + bias  # [N, C]
            # Numerically-stable softmax: subtract per-row max before exp.
            shifted = logits - logits.max(dim=1, keepdim=True).values
            exp = torch.exp(shifted)
            probs = exp / exp.sum(dim=1, keepdim=True)
            # Cross-entropy gradient wrt logits: probs - one_hot(y_idx)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, y_idx.unsqueeze(1), 1.0)
            err = probs - one_hot  # [N, C]
            grad_w = (1 / n) * x_train.T @ err + 2 * lam * weights
            grad_b = (1 / n) * err.sum(dim=0)
            weights = weights - lr * grad_w
            bias = bias - lr * grad_b

        # Predict on query.
        query_logits = x_query @ weights + bias
        shifted = query_logits - query_logits.max(dim=1, keepdim=True).values
        exp = torch.exp(shifted)
        query_probs = exp / exp.sum(dim=1, keepdim=True)
        pred_idx = query_probs.argmax(dim=1).tolist()
        predictions = [classes[i] for i in pred_idx]

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "fit",
                f"Trained {epochs} epochs at lr={lr}; classes={classes}.",
                weights=weights, bias=bias,
                scalars={"lr": lr, "epochs": float(epochs), "lambda": lam},
            )
            recorder.record(
                "predict",
                "Softmax over query logits; argmax → predicted class.",
                logits=query_logits, probabilities=query_probs,
            )
            return {
                "predictions": predictions,
                "probabilities": query_probs,
                "weights": weights,
                "bias": bias,
                "classes": classes,
                "__steps__": recorder.steps,
            }

        return {
            "predictions": predictions,
            "probabilities": query_probs,
            "weights": weights,
            "bias": bias,
            "classes": classes,
        }
