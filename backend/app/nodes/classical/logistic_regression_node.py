"""LogisticRegressionNode — production softmax classifier (sklearn).

Drop-in replacement for :class:`EduLogisticRegressionNode`. Same inputs and
outputs; sklearn's solver picks LBFGS by default and handles multinomial
softmax automatically.

Use ``EduLogisticRegression`` to teach the gradient-descent loop and the
softmax math; switch to this node when the dataset is real.
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


class LogisticRegressionNode(BaseNode):
    NODE_NAME = "LogisticRegression"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Multinomial logistic regression (sklearn). Fits a softmax classifier "
        "with L2 / L1 regularisation; same I/O shape as EduLogisticRegression "
        "for plug-and-play swapping."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="Training labels (length N)."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features [M, F]."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="Predicted label per query."),
            PortDefinition(name="probabilities", data_type=DataType.TENSOR, description="Softmax probabilities [M, C]."),
            PortDefinition(name="classes", data_type=DataType.LIST, description="Class labels in column order."),
            PortDefinition(name="coef", data_type=DataType.TENSOR, description="Learned coefficients [C, F]."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="C",
                param_type=ParamType.FLOAT,
                default=1.0,
                description="Inverse regularisation strength (smaller = more regularised).",
            ),
            ParamDefinition(
                name="max_iter",
                param_type=ParamType.INT,
                default=200,
                min_value=1,
                description="Maximum solver iterations.",
            ),
            ParamDefinition(
                name="penalty",
                param_type=ParamType.SELECT,
                default="l2",
                options=["l2", "l1", "none"],
                description="Regularisation type. l1 needs the liblinear/saga solver; sklearn picks one for you.",
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
        from sklearn.linear_model import LogisticRegression

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("LogisticRegression requires `x_train`, `y_train`, and `x_query` inputs.")

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)

        x_train_np = x_train.detach().cpu().float().numpy()
        x_query_np = x_query.detach().cpu().float().numpy()

        labels = y_train.tolist() if isinstance(y_train, torch.Tensor) else list(y_train)
        labels = [str(v) for v in labels]
        if x_train_np.shape[0] != len(labels):
            raise ValueError(
                f"LogisticRegression: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )
        if len(set(labels)) < 2:
            raise ValueError("LogisticRegression: need at least 2 classes in y_train.")

        C = float(params.get("C", 1.0))
        max_iter = max(1, int(params.get("max_iter", 200)))
        penalty = str(params.get("penalty", "l2"))
        # sklearn 1.4+ accepts None instead of 'none' for the penalty string.
        penalty_arg: str | None = None if penalty == "none" else penalty
        # liblinear is the only solver that supports L1 + binary; saga handles
        # L1 + multiclass. lbfgs handles L2 + multinomial. Letting sklearn pick
        # would otherwise raise on L1.
        solver = "saga" if penalty == "l1" else "lbfgs"

        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            penalty=penalty_arg,
            solver=solver,
        )
        model.fit(x_train_np, labels)

        preds = model.predict(x_query_np).tolist()
        proba = model.predict_proba(x_query_np)
        classes = [str(c) for c in model.classes_.tolist()]

        return {
            "predictions": [str(p) for p in preds],
            "probabilities": torch.from_numpy(proba).float(),
            "classes": classes,
            "coef": torch.from_numpy(model.coef_).float(),
        }
