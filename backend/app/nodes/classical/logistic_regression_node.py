"""LogisticRegressionNode — production softmax classifier (sklearn).

Drop-in replacement for :class:`EduLogisticRegressionNode`. Same inputs and
outputs; sklearn's solver picks LBFGS by default and handles multinomial
softmax automatically.

Use ``EduLogisticRegression`` to teach the gradient-descent loop and the
softmax math; switch to this node when the dataset is real.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import torch
from packaging.version import Version

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


def _penalty_kwargs(penalty: str, C: float, sklearn_version: str) -> dict[str, Any]:
    """The LogisticRegression arguments that fit ``penalty`` on this scikit-learn.

    scikit-learn 1.8 deprecated ``penalty`` (removal planned for 1.10) in
    favour of ``l1_ratio`` and ``C``. Before 1.8, ``l1_ratio`` only counts with
    penalty='elasticnet', so ``l1_ratio=1`` there quietly fits an L2 model.
    Hence one spelling per version range; both fit the same model.
    """
    if penalty not in ("l2", "l1", "none"):
        raise ValueError(
            f"LogisticRegression: penalty must be l2, l1 or none, got {penalty!r}."
        )
    # Checked here for every option: from 1.8, "none" hands scikit-learn
    # C=inf, so its own check would never see this C. `not >` rejects NaN too.
    if not C > 0:
        raise ValueError(f"LogisticRegression: C must be greater than 0, got {C}.")
    # saga fits L1 for any number of classes; lbfgs fits L2 and the
    # unpenalised model, and refuses L1.
    solver = "saga" if penalty == "l1" else "lbfgs"
    if Version(sklearn_version).release < (1, 8):
        return {"C": C, "penalty": None if penalty == "none" else penalty, "solver": solver}
    if penalty == "none":
        # An infinite C is the unpenalised fit. The node's C goes unused, as
        # it does with penalty=None.
        return {"C": math.inf, "solver": solver}
    return {"C": C, "l1_ratio": 1.0 if penalty == "l1" else 0.0, "solver": solver}


class LogisticRegressionNode(BaseNode):
    NODE_NAME = "LogisticRegression"
    CATEGORY = "Classical"
    DESCRIPTION = "Softmax classifier: a label and class probabilities"
    DETAILS = (
        "sklearn's LogisticRegression. C is the inverse regularisation strength, "
        "so a smaller C regularises harder; penalty may be l2, l1 or none, and the "
        "solver is picked to match. y_train needs at least two classes, and coef "
        "carries one row per class, or a single row when there are exactly two. "
        "Its ports match Edu-LogisticRegression in the foundations plugin, so the "
        "two swap without rewiring."
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
                description="Regularisation type. l1 uses the saga solver, l2 and none use lbfgs.",
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
        import sklearn
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

        model = LogisticRegression(
            max_iter=max_iter,
            **_penalty_kwargs(penalty, C, sklearn.__version__),
        )
        if penalty == "none":
            # scikit-learn warns that C is ignored when penalty=None comes with
            # a C other than 1.0, and 1.8.x does so for C=inf too, the newer
            # spelling of "none". The node never uses C for "none", so drop
            # just that message, and only around this fit: the filter list is
            # process-wide, and nodes run in worker threads.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=re.escape("Setting penalty=None will ignore the C"),
                    category=UserWarning,
                )
                model.fit(x_train_np, labels)
        else:
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
