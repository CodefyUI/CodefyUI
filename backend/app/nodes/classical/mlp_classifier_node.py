"""MLPClassifierNode — production multi-layer perceptron classifier (sklearn).

This is the high-level wrapper that gives C2-4 / C2-5 a clean teaching
pipeline without forcing students to wire 6 nodes for a 2-layer MLP. For
the inside-the-box view (linear → activation → linear → softmax exposed
step-by-step) the textbook uses the production Linear/ReLU/CrossEntropy
nodes; for "I just want a model that solves circles" the assignment is
``SyntheticDataset → TrainTestSplit → MLPClassifier → Accuracy``.

The two-track design mirrors KNN/EduKNN and LinearRegression/EduLinear:
production node for "ship the result", Edu nodes for "watch the steps".
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


class MLPClassifierNode(BaseNode):
    NODE_NAME = "MLPClassifier"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Feed-forward neural network classifier (sklearn). One or more "
        "hidden layers, ReLU/tanh activations, Adam optimiser. Drop-in "
        "replacement for the linear classifiers — same I/O — so the "
        "concentric-circles failure → MLP rescue narrative needs only a "
        "node-type swap."
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
            PortDefinition(name="train_loss", data_type=DataType.SCALAR, description="Final training loss."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="hidden_sizes",
                param_type=ParamType.STRING,
                default="16,16",
                description="Comma-separated hidden layer sizes. '16,16' = two layers of 16 neurons.",
            ),
            ParamDefinition(
                name="activation",
                param_type=ParamType.SELECT,
                default="relu",
                options=["relu", "tanh", "logistic", "identity"],
                description=(
                    "Hidden-layer activation. 'identity' makes the whole network linear — "
                    "use it to demonstrate why activation functions are necessary."
                ),
            ),
            ParamDefinition(
                name="max_iter",
                param_type=ParamType.INT,
                default=500,
                min_value=1,
                description="Maximum training iterations (epochs over the full dataset).",
            ),
            ParamDefinition(
                name="learning_rate_init",
                param_type=ParamType.FLOAT,
                default=0.01,
                min_value=0.0,
                description="Initial learning rate for Adam.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Random seed for reproducibility.",
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
        from sklearn.neural_network import MLPClassifier

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("MLPClassifier requires `x_train`, `y_train`, and `x_query` inputs.")

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
                f"MLPClassifier: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        hidden_raw = str(params.get("hidden_sizes", "16,16"))
        hidden_sizes = tuple(int(s.strip()) for s in hidden_raw.split(",") if s.strip())
        if not hidden_sizes:
            raise ValueError("MLPClassifier: hidden_sizes must list at least one layer width.")

        activation = str(params.get("activation", "relu"))
        max_iter = int(params.get("max_iter", 500))
        lr = float(params.get("learning_rate_init", 0.01))
        seed = int(params.get("seed", 42))

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation=activation,
            solver="adam",
            learning_rate_init=lr,
            max_iter=max_iter,
            random_state=seed,
        )
        clf.fit(x_train_np, labels)
        preds = clf.predict(x_query_np).tolist()
        probs = torch.from_numpy(clf.predict_proba(x_query_np)).float()
        classes = [str(c) for c in clf.classes_.tolist()]
        train_loss = float(clf.loss_)

        return {
            "predictions": [str(p) for p in preds],
            "probabilities": probs,
            "classes": classes,
            "train_loss": train_loss,
        }
