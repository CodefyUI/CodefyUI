"""SVMClassifierNode — Support Vector Classifier via ``sklearn.svm.SVC``.

Picks the maximum-margin hyperplane between classes; the kernel trick lets it
draw nonlinear decision boundaries in input space (rbf, polynomial, sigmoid).

Outputs the support vectors so downstream viz can highlight which training
points actually determine the boundary — the geometric heart of SVMs.
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


class SVMClassifierNode(BaseNode):
    NODE_NAME = "SVMClassifier"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Support Vector Classifier (sklearn). Maximum-margin hyperplane, "
        "with kernel trick for nonlinear boundaries. Exposes support vectors "
        "for visualisation — those are the points that define the boundary."
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
            PortDefinition(name="support_vectors", data_type=DataType.TENSOR, description="Training points on or near the margin [n_sv, F]."),
            PortDefinition(name="classes", data_type=DataType.LIST, description="Class labels in sorted order."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="C",
                param_type=ParamType.FLOAT,
                default=1.0,
                description="Penalty strength; smaller C = wider margin and more violations tolerated.",
            ),
            ParamDefinition(
                name="kernel",
                param_type=ParamType.SELECT,
                default="rbf",
                options=["linear", "rbf", "poly", "sigmoid"],
                description="Kernel function: linear hyperplane, gaussian RBF, polynomial, or sigmoid.",
            ),
            ParamDefinition(
                name="gamma",
                param_type=ParamType.STRING,
                default="scale",
                description="Kernel coefficient (rbf/poly/sigmoid). 'scale' uses 1/(F·var(X)); 'auto' uses 1/F; or pass a number as a string.",
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
        from sklearn.svm import SVC

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("SVMClassifier requires `x_train`, `y_train`, and `x_query` inputs.")

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
                f"SVMClassifier: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        C = float(params.get("C", 1.0))
        kernel = str(params.get("kernel", "rbf"))
        gamma_raw = params.get("gamma", "scale")
        gamma: float | str
        try:
            gamma = float(gamma_raw)
        except (TypeError, ValueError):
            gamma = str(gamma_raw)

        clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=False)
        clf.fit(x_train_np, labels)
        preds = clf.predict(x_query_np).tolist()
        classes = [str(c) for c in clf.classes_.tolist()]

        return {
            "predictions": [str(p) for p in preds],
            "support_vectors": torch.from_numpy(clf.support_vectors_).float(),
            "classes": classes,
        }
