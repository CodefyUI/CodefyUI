from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LRSchedulerNode(BaseNode):
    NODE_NAME = "LRScheduler"
    CATEGORY = "Training"
    DESCRIPTION = "Create a learning rate scheduler for an optimizer"

    # #254. A scheduler is a live handle exactly like a model or an
    # optimizer: ``TrainingLoop`` steps it once per epoch, so its
    # ``last_epoch`` advances during the run and the recorded output stops
    # describing it. Replaying it hands the next run a scheduler that has
    # already finished its schedule. Measured, three runs against one
    # ExecutionCache with StepLR(step_size=1, gamma=0.1) over 2 epochs:
    # 1 / 0 / 0 real execute() calls, and TrainingLoop received
    # (last_epoch=0, lr=0.1), then (2, 0.001), then (4, 1e-05) -- run 3
    # trained four orders of magnitude below the learning rate on the node,
    # with nothing anywhere saying so. ``OneCycleLR`` is worse: it raises
    # "Tried to step N times" and takes the run down.
    #
    # Constructing a scheduler is microseconds, and it is not a #144 node
    # (it reads nothing off disk), so there is no cost to weigh against
    # that. The output port is typed ANY rather than MODEL/OPTIMIZER, which
    # is why ``test_cache_live_handle_nodes.py`` names this node explicitly
    # instead of catching it with the port-type rule.
    cacheable = False

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
                options=["StepLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "MultiStepLR", "OneCycleLR", "warmup_cosine", "warmup_linear", "constant_with_warmup"],
            ),
            ParamDefinition(
                name="warmup_steps",
                param_type=ParamType.INT,
                default=100,
                min_value=1,
                description=(
                    "warmup_cosine / warmup_linear / constant_with_warmup: "
                    "steps of linear LR ramp from ~0 to the optimizer's LR "
                    "before the decay phase. Denominated in SCHEDULER steps "
                    "— pair with TrainingLoop's scheduler_step="
                    "optimizer_step and set total_steps to the run's "
                    "max_steps so 'warm up for 100 steps then anneal over "
                    "1500' means exactly that (#297). At the default "
                    "scheduler_step=epoch a scheduler step is one EPOCH, so "
                    "the default 100 would still be ramping long after a "
                    "5-epoch run ended and the run would never train at the "
                    "LR you set (#308)."
                ),
                visible_when={"type": ["warmup_cosine", "warmup_linear", "constant_with_warmup"]},
            ),
            ParamDefinition(
                name="step_size",
                param_type=ParamType.INT,
                default=10,
                description=(
                    "StepLR: how long between each LR drop. MultiStepLR: "
                    "drops at 1x, 2x, 3x and 4x this value. The unit follows "
                    "TrainingLoop.scheduler_step (#308): EPOCHS at the "
                    "default `epoch` (not batches), OPTIMIZER STEPS at "
                    "`optimizer_step`. Either way it has to be smaller than "
                    "the run's length in that same unit, or the first drop "
                    "never arrives and the whole run trains at a constant LR."
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
                    "CosineAnnealingLR: length of one cosine cycle. The unit "
                    "follows TrainingLoop.scheduler_step (#308): at the "
                    "default `epoch` it is EPOCHS, so set it EQUAL to "
                    "TrainingLoop.epochs; at `optimizer_step` it is OPTIMIZER "
                    "STEPS, so set it to the run's step budget instead "
                    "(max_steps when set, otherwise epochs x batches per "
                    "epoch / accumulate_steps). Either way a smaller value "
                    "ends the cycle early and the cosine turns back UP, so "
                    "the tail of the run trains at a RISING LR; a larger one "
                    "stops partway down the curve at a still-high LR. Both "
                    "typically cost a few points of accuracy with nothing to "
                    "show that the schedule was the cause. Deliberately NOT "
                    "enforced: a truncated schedule is a valid choice. "
                    "CosineAnnealingWarmRestarts reuses this value as `T_0`, "
                    "the length of the FIRST restart cycle, where equality "
                    "with the run's length would mean no restart ever happens."
                ),
            ),
            ParamDefinition(name="max_lr", param_type=ParamType.FLOAT, default=0.01, description="Max learning rate for OneCycleLR"),
            ParamDefinition(
                name="total_steps",
                param_type=ParamType.INT,
                default=1000,
                description=(
                    "OneCycleLR: length of the one-cycle schedule. The "
                    "warmup_* families use it as their TOTAL length too "
                    "(warmup ramp + decay). The unit follows "
                    "TrainingLoop.scheduler_step (#308): at the default "
                    "`epoch` it is an EPOCH count, so set it to "
                    "TrainingLoop.epochs — NOT to the number of batches, "
                    "which is what OneCycleLR's own documentation means by a "
                    "step. At `optimizer_step` it IS that step count, so set "
                    "it to the run's step budget: max_steps when set, "
                    "otherwise epochs x batches per epoch / accumulate_steps. "
                    "The default of 1000 is far larger than any usual epoch "
                    "count, so leaving it alone in epoch mode traverses only "
                    "the beginning of the cycle: the LR warms up slightly and "
                    "never anneals."
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
        elif sched_type in ("warmup_cosine", "warmup_linear", "constant_with_warmup"):
            # The standard LM pretraining shapes (#297): a linear ramp from
            # ~0 to the optimizer's LR over warmup_steps, then cosine decay,
            # linear decay to ~0, or a constant hold — all denominated in
            # scheduler STEPS over total_steps. Meant for TrainingLoop's
            # scheduler_step=optimizer_step; at the default per-epoch
            # stepping these traverse one point per epoch like everything
            # else here.
            warmup = max(1, int(params.get("warmup_steps", 100) or 100))
            total = max(warmup + 1, int(params.get("total_steps", 1000) or 1000))
            remainder = total - warmup
            # start_factor cannot be exactly 0 (torch rejects it); 1e-8 of
            # the base LR is indistinguishable from cold.
            ramp = lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0,
                total_iters=warmup)
            if sched_type == "warmup_cosine":
                decay = lr_scheduler.CosineAnnealingLR(optimizer, T_max=remainder)
            elif sched_type == "warmup_linear":
                decay = lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=1e-8,
                    total_iters=remainder)
            else:  # constant_with_warmup
                decay = lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=0)
            sched = lr_scheduler.SequentialLR(
                optimizer, schedulers=[ramp, decay], milestones=[warmup])
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

        return {"scheduler": sched}
