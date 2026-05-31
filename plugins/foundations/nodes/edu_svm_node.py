"""EduSVMNode — hand-written linear soft-margin SVM trained by gradient descent.

Textbook lesson **I2-2 (linear soft-margin SVM)**: instead of handing the
problem to ``sklearn.svm.LinearSVC``, expose every moving part of the
sub-gradient descent that minimises the primal hinge-loss objective:

    scores            = X @ w + b
    functional_margin = y_pm1 * scores
    hinge             = max(0, 1 - functional_margin)
    loss              = ½‖w‖² + C · mean(hinge)

A point is a **support vector** when its functional margin is < 1 (it sits on
or inside the margin, or is misclassified) — exactly the points that
contribute a non-zero sub-gradient. The sub-gradient of the objective is

    violating = (functional_margin < 1)
    dw        = w − C · mean(violating · y_pm1 · x)   (mean over the batch)
    db        =     − C · mean(violating · y_pm1)

and one descent step is ``w ← w − lr·dw``, ``b ← b − lr·db``.

Labels are mapped to ±1 internally (the lower class value → −1, the higher
→ +1) and predictions are mapped back to the original label space so a
downstream node sees the same two label values it fed in.

Outputs expose the learned ``weights``/``bias``, the boolean
``support_vectors`` mask over the training set, and the raw
``decision_values`` ``w·x_query + b`` so a visualisation can draw the
separating hyperplane, the ±1 margins, and highlight the support vectors.
In verbose mode the per-epoch hinge loss and weight updates are recorded
(capped to ~8 evenly-spaced snapshots) so students can watch the margin
tighten over training.
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

# Cap the number of per-epoch snapshots recorded in verbose mode so a
# 10k-epoch run doesn't emit 10k steps into the Teaching Inspector.
_MAX_EPOCH_SNAPSHOTS = 8


class EduSVMNode(BaseNode):
    NODE_NAME = "Edu-SVM"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Hand-written linear soft-margin SVM trained by sub-gradient descent on "
        "the hinge loss ½‖w‖² + C·mean(max(0, 1 − y·(w·x+b))). Binary labels are "
        "mapped to ±1 internally and predictions returned in the original label "
        "space. Exposes weights, bias, the support-vector mask (margin < 1), and "
        "the raw decision values w·x+b for visualising the hyperplane and margins."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_train",
                data_type=DataType.TENSOR,
                description="Training features [N, D].",
            ),
            PortDefinition(
                name="y_train",
                data_type=DataType.TENSOR,
                description=(
                    "Training labels [N] with EXACTLY two distinct values. The "
                    "lower value maps to −1, the higher to +1 internally."
                ),
            ),
            PortDefinition(
                name="x_query",
                data_type=DataType.TENSOR,
                description="Query features [M, D] to classify.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.TENSOR,
                description="Predicted label per query [M], in the original label space.",
            ),
            PortDefinition(
                name="weights",
                data_type=DataType.TENSOR,
                description="Learned weight vector [D].",
            ),
            PortDefinition(
                name="bias",
                data_type=DataType.TENSOR,
                description="Learned scalar bias.",
            ),
            PortDefinition(
                name="support_vectors",
                data_type=DataType.TENSOR,
                description=(
                    "Boolean mask [N]; True where the train margin y·(w·x+b) < 1 "
                    "(support vectors). Display-only."
                ),
            ),
            PortDefinition(
                name="decision_values",
                data_type=DataType.TENSOR,
                description="Raw decision function w·x_query + b, shape [M]. Display-only.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="C",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.0,
                description=(
                    "Soft-margin penalty. Larger C punishes margin violations "
                    "harder (narrower margin, fewer mistakes); smaller C tolerates "
                    "more violations for a wider margin."
                ),
            ),
            ParamDefinition(
                name="lr",
                param_type=ParamType.FLOAT,
                default=0.01,
                min_value=0.0,
                description="Sub-gradient descent step size.",
            ),
            ParamDefinition(
                name="epochs",
                param_type=ParamType.INT,
                default=100,
                min_value=1,
                description="Number of full-batch sub-gradient descent steps.",
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
                "EduSVM requires `x_train`, `y_train`, and `x_query` inputs."
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(y_train)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train = x_train.float()
        x_query = x_query.float()
        y_train = y_train.flatten()

        # --- Shape validation -------------------------------------------------
        if x_train.ndim != 2:
            raise ValueError(
                f"EduSVM: x_train must be 2-D [N, D]; got shape {tuple(x_train.shape)}."
            )
        if x_query.ndim != 2:
            raise ValueError(
                f"EduSVM: x_query must be 2-D [M, D]; got shape {tuple(x_query.shape)}."
            )
        n_samples, n_features = x_train.shape
        if y_train.shape[0] != n_samples:
            raise ValueError(
                f"EduSVM: x_train and y_train length mismatch — "
                f"{n_samples} rows vs {y_train.shape[0]} labels."
            )
        if x_query.shape[1] != n_features:
            raise ValueError(
                f"EduSVM: x_query has {x_query.shape[1]} features but x_train has "
                f"{n_features}. Feature dimensions must match."
            )

        # --- Param validation -------------------------------------------------
        C = float(params.get("C", 1.0))
        lr = float(params.get("lr", 0.01))
        epochs = int(params.get("epochs", 100))
        if C < 0:
            raise ValueError(f"EduSVM: C must be >= 0; got {C}.")
        if lr < 0:
            raise ValueError(f"EduSVM: lr must be >= 0; got {lr}.")
        if epochs < 1:
            raise ValueError(f"EduSVM: epochs must be >= 1; got {epochs}.")

        # --- Label mapping to ±1 ---------------------------------------------
        # Exactly two distinct classes; lower value -> -1, higher value -> +1.
        unique_classes = torch.unique(y_train)
        if unique_classes.numel() != 2:
            raise ValueError(
                f"EduSVM: y_train must have exactly 2 distinct classes; got "
                f"{unique_classes.numel()} ({unique_classes.tolist()})."
            )
        neg_class, pos_class = unique_classes[0], unique_classes[1]  # sorted ascending
        # y_pm1[i] = +1 if y_train[i] == pos_class else -1
        y_pm1 = torch.where(
            y_train == pos_class,
            torch.ones(n_samples),
            -torch.ones(n_samples),
        )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        # --- Train: full-batch sub-gradient descent on the hinge objective ----
        w = torch.zeros(n_features)
        b = torch.zeros(())

        if recorder is not None:
            recorder.record(
                "init",
                "Initialise w = 0 (shape [D]) and b = 0; objective is "
                "½‖w‖² + C·mean(max(0, 1 − y·(w·x+b))).",
                scalars={
                    "C": C,
                    "lr": lr,
                    "epochs": float(epochs),
                    "n_features": float(n_features),
                    "n_samples": float(n_samples),
                },
                w=w,
            )

        # Pick at most _MAX_EPOCH_SNAPSHOTS evenly-spaced epoch indices to record.
        snapshot_epochs: set[int] = set()
        if recorder is not None:
            n_snap = min(_MAX_EPOCH_SNAPSHOTS, epochs)
            # Evenly spaced across [1, epochs], always including the last epoch.
            for s in range(n_snap):
                idx = round((s + 1) * epochs / n_snap)
                snapshot_epochs.add(max(1, min(epochs, idx)))

        for epoch in range(1, epochs + 1):
            scores = x_train @ w + b                  # [N]
            functional_margins = y_pm1 * scores       # [N]
            hinge = torch.clamp(1.0 - functional_margins, min=0.0)
            violating = functional_margins < 1.0      # [N] bool

            # Sub-gradient of ½‖w‖² + C·mean(hinge):
            #   dw = w − C · mean(violating · y_pm1 · x)   (mean over batch)
            #   db =     − C · mean(violating · y_pm1)
            viol_y = violating.float() * y_pm1        # [N]
            dw = w - C * (viol_y.unsqueeze(1) * x_train).mean(dim=0)
            db = -C * viol_y.mean()

            w = w - lr * dw
            b = b - lr * db

            if progress_callback is not None:
                try:
                    progress_callback(epoch / epochs)
                except Exception:
                    pass

            if recorder is not None and epoch in snapshot_epochs:
                loss = 0.5 * (w @ w) + C * hinge.mean()
                num_sv = int(violating.sum().item())
                recorder.record(
                    f"epoch_{epoch}",
                    "Sub-gradient step: dw = w − C·mean(violating·y·x); "
                    "db = −C·mean(violating·y); then w ← w − lr·dw, b ← b − lr·db.",
                    scalars={
                        "epoch": float(epoch),
                        "loss": float(loss.item()),
                        "num_support_vectors": float(num_sv),
                    },
                    w=w,
                )

        # --- Final support-vector mask over the training set ------------------
        final_scores = x_train @ w + b
        final_margins = y_pm1 * final_scores
        support_vectors = final_margins < 1.0  # bool [N]

        # --- Predict on the query set ----------------------------------------
        decision_values = x_query @ w + b      # [M], display-only
        # sign(decision) mapped back to original labels: >= 0 -> pos_class.
        predictions = torch.where(
            decision_values >= 0,
            pos_class.to(decision_values.dtype),
            neg_class.to(decision_values.dtype),
        )

        if recorder is not None:
            recorder.record(
                "decision",
                "Final hyperplane: predictions = sign(w·x_query + b) mapped back "
                "to the original labels; support vectors are train points with "
                "margin y·(w·x+b) < 1.",
                scalars={
                    "bias": float(b.item()),
                    "weight_norm": float(w.norm().item()),
                    "num_support_vectors": float(int(support_vectors.sum().item())),
                },
                weights=w,
                support_vectors=support_vectors,
                decision_values=decision_values,
            )

        result: dict[str, Any] = {
            "predictions": predictions,
            "weights": w,
            "bias": b,
            "support_vectors": support_vectors,
            "decision_values": decision_values,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
