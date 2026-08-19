"""Bradley-Terry: the loss on its own, and the training loop that uses it.

Two nodes, because the chapter needs them apart.

``BradleyTerryLoss`` is pure arithmetic over two reward tensors:

    P(w > l) = sigmoid( r(w) - r(l) )        loss = -log P

Having it as its own node is what lets a lesson demonstrate the property that
makes the whole approach work: **only the difference matters**. Add 100 to both
scores and the preference probability does not move a digit. That is why
RLHF can get away with never asking a human for an absolute number -- the data
only ever contained an ordering, and an ordering is all the model ever learns.
It also means two different reward models' scores are not comparable; each one's
zero is wherever training happened to leave it.

``BradleyTerryTrain`` runs the fit and, crucially, measures **two** numbers:
accuracy on the pairs it is training on, and accuracy on a held-out split it
never sees. The training number alone cannot show reward hacking -- against a
planted shortcut it reads a perfect 1.000 either way. Only the gap to the
held-out number moves. C5-3 lists "keep a human validation set" as the first
defence against hacking, and this node is that practice wired in rather than
described.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class BradleyTerryLossNode(BaseNode):
    NODE_NAME = "BradleyTerryLoss"
    CATEGORY = "RL"
    DESCRIPTION = (
        "Turn two reward scores into a preference: P(w>l) = sigmoid(r_w - r_l), "
        "loss = -log P. ONLY THE DIFFERENCE MATTERS -- add any constant to both scores "
        "and the probability is unchanged, which is why RLHF never needs absolute human "
        "scores and why two reward models' outputs cannot be compared."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="reward_w", data_type=DataType.TENSOR, description="[N] score of the preferred response."),
            PortDefinition(name="reward_l", data_type=DataType.TENSOR, description="[N] score of the rejected response."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="loss", data_type=DataType.TENSOR, description="Scalar mean of -log sigmoid(r_w - r_l)."),
            PortDefinition(name="preference_prob", data_type=DataType.TENSOR, description="[N] sigmoid(r_w - r_l): the model's P(w is better)."),
            PortDefinition(name="score_diff", data_type=DataType.TENSOR, description="[N] r_w - r_l, the only quantity that matters."),
            PortDefinition(name="accuracy", data_type=DataType.SCALAR, description="Fraction of pairs where r_w > r_l."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        rw = inputs["reward_w"]
        rl = inputs["reward_l"]
        if not isinstance(rw, torch.Tensor):
            rw = torch.as_tensor(rw, dtype=torch.float32)
        if not isinstance(rl, torch.Tensor):
            rl = torch.as_tensor(rl, dtype=torch.float32)
        rw, rl = rw.float().flatten(), rl.float().flatten()
        rl = rl.to(rw.device)
        if rw.shape != rl.shape:
            raise ValueError(
                f"BradleyTerryLoss: reward_w has {rw.shape[0]} entries but reward_l has "
                f"{rl.shape[0]} -- they are the two halves of the same pairs."
            )

        diff = rw - rl
        # softplus(-x) is -log(sigmoid(x)) without the overflow at large |x|.
        loss = F.softplus(-diff).mean()

        result: dict[str, Any] = {
            "loss": loss,
            "preference_prob": torch.sigmoid(diff),
            "score_diff": diff,
            "accuracy": float((diff > 0).float().mean()) if diff.numel() else 0.0,
        }

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder

            recorder = StepRecorder()
            recorder.record("scores", "The reward model's score for each half of the pair.", reward_w=rw, reward_l=rl)
            recorder.record("difference", "$r_w - r_l$. Shifting both scores leaves this untouched.", score_diff=diff)
            recorder.record("sigmoid", "$P(w \\succ l) = \\sigma(r_w - r_l)$.", preference_prob=torch.sigmoid(diff))
            result["__steps__"] = recorder.steps

        return result


class BradleyTerryTrainNode(BaseNode):
    NODE_NAME = "BradleyTerryTrain"
    CATEGORY = "RL"
    DESCRIPTION = (
        "Fit a reward model on preference pairs with the Bradley-Terry objective, measuring "
        "accuracy on BOTH the training pairs and a held-out split every epoch. The GAP "
        "between those two numbers is reward hacking: the model can score a perfect 1.000 on "
        "the pairs it trained on while having learned a shortcut instead of the preference, "
        "and only the held-out split reveals it."
    )

    #: Owns trained weights and has no required upstream that changes; a cache
    #: hit would hand back an already-fitted model and a flat curve.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="train_w", data_type=DataType.TENSOR, description="[N, D] preferred responses."),
            PortDefinition(name="train_l", data_type=DataType.TENSOR, description="[N, D] rejected responses."),
            PortDefinition(name="holdout_w", data_type=DataType.TENSOR, description="[M, D] preferred, held out.", optional=True),
            PortDefinition(name="holdout_l", data_type=DataType.TENSOR, description="[M, D] rejected, held out.", optional=True),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="The fitted reward model."),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="[epochs] training loss."),
            PortDefinition(name="train_accuracy", data_type=DataType.TENSOR, description="[epochs] accuracy on the training pairs."),
            PortDefinition(name="holdout_accuracy", data_type=DataType.TENSOR, description="[epochs] accuracy on the held-out pairs -- the only number that can see a shortcut."),
            PortDefinition(name="best_holdout_epoch", data_type=DataType.SCALAR, description="Epoch where holdout accuracy peaked."),
            PortDefinition(name="report", data_type=DataType.STRING, description="Per-epoch summary as text."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="epochs", param_type=ParamType.INT, default=60, min_value=1, description="Passes over the pairs. Enough to fit the training split fully."),
            ParamDefinition(name="hidden_dim", param_type=ParamType.INT, default=32, min_value=1, description="Reward head width."),
            ParamDefinition(name="lr", param_type=ParamType.FLOAT, default=0.01, description="Learning rate."),
            ParamDefinition(name="seed", param_type=ParamType.INT, default=0, description="Reproducibility."),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        tw, tl = inputs["train_w"].float(), inputs["train_l"].float()
        hw, hl = inputs.get("holdout_w"), inputs.get("holdout_l")
        if tw.shape != tl.shape:
            raise ValueError(
                f"BradleyTerryTrain: train_w {tuple(tw.shape)} and train_l {tuple(tl.shape)} "
                "must have the same shape -- they are paired."
            )

        epochs = int(params.get("epochs", 60))
        torch.manual_seed(int(params.get("seed", 0)))
        d = tw.shape[-1]
        model = nn.Sequential(
            nn.Linear(d, int(params.get("hidden_dim", 32))),
            nn.ReLU(),
            nn.Linear(int(params.get("hidden_dim", 32)), 1),
        )
        opt = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 0.01)))

        def acc(a, b):
            with torch.no_grad():
                return float(((model(a.float()) - model(b.float())).squeeze(-1) > 0).float().mean())

        losses, tr_acc, ho_acc, lines = [], [], [], []
        for ep in range(epochs):
            diff = (model(tw) - model(tl)).squeeze(-1)
            loss = F.softplus(-diff).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.detach()))
            tr_acc.append(acc(tw, tl))
            ho_acc.append(acc(hw, hl) if hw is not None and hl is not None else float("nan"))
            lines.append(f"epoch {ep + 1:3d}: loss {losses[-1]:.4f}  train {tr_acc[-1]:.3f}  holdout {ho_acc[-1]:.3f}")

        finite = [(i, a) for i, a in enumerate(ho_acc) if a == a]
        best = max(finite, key=lambda p: p[1])[0] + 1 if finite else 0

        return {
            "model": model,
            "losses": torch.tensor(losses),
            "train_accuracy": torch.tensor(tr_acc),
            "holdout_accuracy": torch.tensor(ho_acc),
            "best_holdout_epoch": best,
            "report": "\n".join(lines),
        }
