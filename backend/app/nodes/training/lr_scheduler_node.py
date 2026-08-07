from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LRSchedulerNode(BaseNode):
    NODE_NAME = "LRScheduler"
    CATEGORY = "Training"
    DESCRIPTION = "Create a learning rate scheduler for an optimizer"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer to schedule"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="scheduler", data_type=DataType.ANY, description="Configured LR scheduler"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="StepLR",
                description="Scheduler type",
                options=["StepLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "MultiStepLR", "OneCycleLR"],
            ),
            ParamDefinition(
                name="step_size",
                param_type=ParamType.INT,
                default=10,
                description=(
                    "StepLR: epochs between each LR drop. MultiStepLR: drops at "
                    "1x, 2x, 3x and 4x this value. TrainingLoop steps the "
                    "scheduler once per EPOCH, so this counts epochs, not batches."
                ),
            ),
            ParamDefinition(
                name="gamma",
                param_type=ParamType.FLOAT,
                default=0.1,
                description=(
                    "Decay factor for StepLR, MultiStepLR and ExponentialLR. "
                    "ReduceLROnPlateau reuses it as its `factor`."
                ),
            ),
            ParamDefinition(
                name="T_max",
                param_type=ParamType.INT,
                default=50,
                description=(
                    "CosineAnnealingLR: length of one cosine cycle, in epochs. "
                    "Normally set this EQUAL to TrainingLoop.epochs — the "
                    "scheduler steps once per epoch, so a smaller value ends the "
                    "run mid-cycle at a high LR and a larger one stops partway "
                    "down the curve, typically costing a few points of accuracy "
                    "with nothing to show that the schedule was the cause. "
                    "Deliberately NOT enforced: a truncated schedule is a valid "
                    "choice. CosineAnnealingWarmRestarts reuses this value as "
                    "`T_0`, the length of the FIRST restart cycle, where equality "
                    "with epochs would mean no restart ever happens."
                ),
            ),
            ParamDefinition(name="max_lr", param_type=ParamType.FLOAT, default=0.01, description="Max learning rate for OneCycleLR"),
            ParamDefinition(
                name="total_steps",
                param_type=ParamType.INT,
                default=1000,
                description=(
                    "OneCycleLR: length of the one-cycle schedule. TrainingLoop "
                    "steps the scheduler once per EPOCH, so set this to "
                    "TrainingLoop.epochs — NOT to the number of batches, which is "
                    "what OneCycleLR's own documentation means by a step. The "
                    "default of 1000 is far larger than any usual epoch count, so "
                    "leaving it alone traverses only the beginning of the cycle: "
                    "the LR warms up slightly and never anneals."
                ),
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.optim.lr_scheduler as lr_scheduler

        optimizer = inputs["optimizer"]
        sched_type = params.get("type", "StepLR")

        if sched_type == "StepLR":
            sched = lr_scheduler.StepLR(optimizer, step_size=params.get("step_size", 10), gamma=params.get("gamma", 0.1))
        elif sched_type == "CosineAnnealingLR":
            sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.get("T_max", 50))
        elif sched_type == "ExponentialLR":
            sched = lr_scheduler.ExponentialLR(optimizer, gamma=params.get("gamma", 0.1))
        elif sched_type == "ReduceLROnPlateau":
            sched = lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.get("gamma", 0.1))
        elif sched_type == "CosineAnnealingWarmRestarts":
            sched = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params.get("T_max", 50))
        elif sched_type == "MultiStepLR":
            milestones = [params.get("step_size", 10) * i for i in range(1, 5)]
            sched = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=params.get("gamma", 0.1))
        elif sched_type == "OneCycleLR":
            sched = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params.get("max_lr", 0.01),
                total_steps=params.get("total_steps", 1000),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

        return {"scheduler": sched}
