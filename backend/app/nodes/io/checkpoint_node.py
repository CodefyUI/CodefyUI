"""Save / load a full training checkpoint.

The payload format, the path rules and the writer itself live in
``core.checkpoints`` since #122 -- the interrupt path of every long-running
training node writes the same files, and one format needs one definition.
Read that module's docstring for the key table; these two nodes are the
graph-facing wrapper around it.
"""

import logging
from typing import Any

from ...core.advisories import emit_advisory
from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)

#: Prefix on the canvas-log line this node's advisory carries. The Log tab
#: has no warning severity (``LogEntry.type`` is info/error/success), so the
#: marker in the text is the only thing distinguishing an advisory from a
#: ``Print`` node's output -- same device ``TrainingLoop`` uses (#252).
CHECKPOINT_NOTE_PREFIX = "[CheckpointLoader] "

#: ``kind`` on the ``WarningSignal`` the same note is sent as, which is what
#: a client would branch on. Stable token; the sentence is the detail.
SCHEDULER_STATE_DISCARDED_KIND = "checkpoint_scheduler_state_discarded"


class CheckpointSaverNode(BaseNode):
    NODE_NAME = "CheckpointSaver"
    CATEGORY = "IO"
    DESCRIPTION = "Save a full training checkpoint (model + optimizer + LR schedule + epoch + loss) for resuming training later"

    # The write to disk IS this node's output. A cache hit returns the
    # recorded {"path": ..., "model": ...} without calling execute() again,
    # which is correct for a pure node and wrong here: if the user deleted
    # the checkpoint (or it never ran with Rec off), a cache hit would hand
    # back a path that no longer points at anything (#143).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer state"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history", optional=True),
            PortDefinition(
                name="lr_scheduler",
                data_type=DataType.ANY,
                description="LR scheduler whose position in the schedule to store",
                optional=True,
            ),
            PortDefinition(
                name="grad_scaler_state",
                data_type=DataType.ANY,
                description=(
                    "fp16 loss-scale state to store. Wire "
                    "TrainingLoop.grad_scaler_state here; leave unconnected "
                    "for fp32 and bf16 runs"
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="path", data_type=DataType.STRING, description="Path to the saved checkpoint"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="Pass-through model"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="checkpoint.pt",
                description="Output checkpoint file path",
            ),
            ParamDefinition(
                name="epoch",
                param_type=ParamType.INT,
                default=0,
                description="Current epoch number to store in checkpoint",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from ...core.checkpoints import write_checkpoint

        model = inputs["model"]
        target = write_checkpoint(
            params.get("path", "checkpoint.pt"),
            model,
            inputs["optimizer"],
            epoch=params.get("epoch", 0),
            losses=inputs.get("losses"),
            lr_scheduler=inputs.get("lr_scheduler"),
            scaler_state=inputs.get("grad_scaler_state"),
        )
        return {"path": str(target), "model": model}


class CheckpointLoaderNode(BaseNode):
    NODE_NAME = "CheckpointLoader"
    CATEGORY = "IO"
    DESCRIPTION = "Load a training checkpoint to resume training (restores model + optimizer + LR schedule + epoch)"

    # #254. This node's product is a MUTATION of the model, optimizer and
    # scheduler it was handed -- ``load_state_dict`` writes in place -- and
    # a cache hit returns the recorded outputs without calling execute() at
    # all, so on a hit the restore simply does not happen while the node
    # reports success. Measured, fed by a cacheable model source and run
    # three times against one ExecutionCache: 1 / 0 / 0 real execute()
    # calls, and the model kept whatever weights the previous run's
    # training had left on it instead of the checkpoint's.
    #
    # This DOES undo the ``cacheable = True`` #144 gave it, and that is a
    # smaller loss than it sounds -- measured, not assumed. The engine
    # refuses to cache any node with a non-cacheable upstream, and both of
    # this node's required inputs trace back to a weight-owning model node,
    # every one of which is non-cacheable (``StatefulModuleMixin``, and
    # ``SequentialModel`` joined them in #253). So on the shipped shape
    # (SequentialModel -> Optimizer -> CheckpointLoader) it measured
    # 1 / 1 / 1 execute() calls across three runs BEFORE this change: the
    # hit #144 re-enabled was already unreachable there. The only graphs
    # where it WAS reachable are the ones where it is wrong. #144's
    # fingerprint mechanism itself is untouched and still does its job for
    # the reader nodes it was built for -- Dataset, CSVReader, ImageReader
    # and the rest.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model architecture to load weights into"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer to restore state into"),
            PortDefinition(
                name="lr_scheduler",
                data_type=DataType.ANY,
                description="LR scheduler to restore its position in the schedule into",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model with restored weights"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer with restored state"),
            PortDefinition(name="epoch", data_type=DataType.SCALAR, description="Epoch number from checkpoint (wire to TrainingLoop.start_epoch)"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history from checkpoint"),
            PortDefinition(name="lr_scheduler", data_type=DataType.ANY, description="LR scheduler with restored state (None if none was wired in)"),
            PortDefinition(
                name="grad_scaler_state",
                data_type=DataType.ANY,
                description=(
                    "fp16 loss-scale state from the checkpoint, or None. "
                    "Wire to TrainingLoop.grad_scaler_state to resume an "
                    "fp16 run at the scale it reached"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.MODEL_FILE,
                default="",
                description="Path to the checkpoint file",
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                description="Device to load onto ('auto' follows the global device)",
                options=["auto", "cpu", "cuda", "mps"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        from ...core.checkpoints import resolve_checkpoint_path
        from ...core.device_utils import is_mps_device, resolve_node_device, to_device

        model = inputs["model"]
        optimizer = inputs["optimizer"]
        lr_scheduler = inputs.get("lr_scheduler")
        path = params.get("path", "checkpoint.pt")
        device = resolve_node_device(params.get("device"), context)

        # Same rules as the saver: relative to MODELS_DIR, never outside the
        # project data directory.
        try:
            p = resolve_checkpoint_path(path)
        except ValueError as exc:
            # The reason is carried through rather than flattened to the
            # containment message: since #224 there are two ways to fail
            # this rule (outside the data directory, or naming CodefyUI's
            # own storage) and they need different fixes from the user.
            raise ValueError(f"Checkpoint file path rejected: {exc}") from None

        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        # MPS can't receive float64 via map_location, so stage doubles on CPU
        # and let to_device downcast them on the way to the device.
        load_device = "cpu" if is_mps_device(device) else device
        checkpoint = torch.load(str(p), map_location=load_device, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = to_device(model, device)

        # Re-bind optimizer to device-mapped parameters before restoring state
        for param_group in optimizer.param_groups:
            param_group["params"] = list(model.parameters())

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Move optimizer state tensors (momentum buffers, etc.) to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = to_device(v, device)

        # Honest base_lrs (#149) -- a DEFENSIVE invariant, not a fix for a
        # reachable divergence: `initial_lrs` is derived, at save time, from
        # the same live optimizer.param_groups that optimizer_state_dict
        # also serialises, so for any checkpoint this codebase can actually
        # produce the two never disagree, and restoring this key is
        # provably a no-op against today's torch (verified against the
        # installed 2.11.0+cu128: `update_group` replaces each live param
        # group with the SAVED group's dict wholesale, which already
        # carries `initial_lr` through whenever the checkpoint has it).
        # What this DOES buy: CodefyUI's own contract no longer depends on
        # optimizer.state_dict()/load_state_dict() choosing to carry that
        # key through -- it is read from param_groups directly and
        # restored explicitly, so a future torch (or a plugin's custom
        # Optimizer) that stops doing so surfaces as a visible difference
        # here rather than a silent LR corruption. See the "Honest
        # base_lrs" section of core.checkpoints for the full reasoning.
        # Done BEFORE the scheduler section below on purpose -- "before any
        # scheduler is constructed".
        initial_lrs = checkpoint.get("initial_lrs")
        if initial_lrs is not None:
            if len(initial_lrs) != len(optimizer.param_groups):
                logger.warning(
                    "Checkpoint %s holds %d initial_lr value(s) but the "
                    "optimizer has %d param group(s); restoring only the "
                    "overlap.", p, len(initial_lrs), len(optimizer.param_groups),
                )
            for param_group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                param_group["initial_lr"] = initial_lr

        # LR schedule (#118). Restored AFTER the optimizer on purpose: the
        # optimizer state dict carries the learning rate that was live when the
        # checkpoint was taken, and ``LRScheduler.load_state_dict`` does not
        # touch ``param_groups``, so this order leaves both consistent.
        #
        # Every branch below is a no-op-with-a-log rather than an error, so a
        # checkpoint written before this key existed still loads (and so does a
        # newer checkpoint read by a graph that has no scheduler wired in).
        scheduler_state = checkpoint.get("scheduler_state_dict")
        scheduler_note: str | None = None
        if lr_scheduler is not None and scheduler_state is not None:
            saved_class = checkpoint.get("scheduler_class")
            live_class = type(lr_scheduler).__name__
            if saved_class and saved_class != live_class:
                logger.warning(
                    "Checkpoint holds %s state but a %s is wired in; leaving the "
                    "scheduler untouched rather than restoring incompatible state.",
                    saved_class, live_class,
                )
            else:
                lr_scheduler.load_state_dict(scheduler_state)
                logger.info(
                    "Restored %s at last_epoch=%s",
                    live_class, getattr(lr_scheduler, "last_epoch", "?"),
                )
        elif lr_scheduler is not None:
            logger.info(
                "Checkpoint %s holds no scheduler state (written before it was "
                "recorded); the LR schedule will be fast-forwarded from start_epoch "
                "instead.", p,
            )
        elif scheduler_state is not None:
            # #149. This was a logger.info line, which is to say invisible to
            # anyone watching the canvas -- and it is the branch that loses
            # information the checkpoint actually contains. With no scheduler
            # wired in there is nothing to restore the state INTO, so the
            # schedule is instead reconstructed downstream by replaying
            # ``start_epoch`` steps from ``base_lrs``. For a closed-form
            # schedule (StepLR, CosineAnnealingLR, ...) that replay is exact,
            # which is why this never showed up as a wrong number. For a
            # metric-driven one it cannot be: measured on a
            # ``ReduceLROnPlateau`` resumed from an 8-epoch checkpoint whose
            # last 5 epochs were a plateau, the restored-state path came back
            # with best=0.8 / num_bad_epochs=5 and this path with best=inf /
            # num_bad_epochs=0 -- the decay that was one epoch away postponed
            # indefinitely, silently.
            scheduler_note = emit_advisory(
                f"The checkpoint {p.name} stores the position of a "
                f"{checkpoint.get('scheduler_class') or 'learning-rate'} "
                f"schedule, but nothing is wired into this loader's "
                f"lr_scheduler input, so that stored position is being "
                f"discarded. Wire LRScheduler.scheduler into "
                f"CheckpointLoader.lr_scheduler (and into "
                f"CheckpointSaver.lr_scheduler when saving) to resume the "
                f"schedule exactly. Without it TrainingLoop rebuilds the "
                f"schedule by replaying start_epoch steps, which matches the "
                f"original run for StepLR/CosineAnnealingLR and cannot for "
                f"ReduceLROnPlateau, whose plateau history is not recoverable "
                f"from an epoch number.",
                kind=SCHEDULER_STATE_DISCARDED_KIND,
                prefix=CHECKPOINT_NOTE_PREFIX,
                context=context,
                logger=logger,
            )

        epoch = checkpoint.get("epoch", 0)
        losses = checkpoint.get("losses", torch.tensor([]))

        logger.info("Loaded checkpoint from %s (epoch=%d)", p, epoch)

        result: dict[str, Any] = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "losses": losses,
            "lr_scheduler": lr_scheduler,
            # Absent from every checkpoint written before #135 and from
            # every fp32/bf16 run since, so None is the ordinary answer
            # rather than a failure -- ``TrainingLoop`` treats it as "start
            # from a fresh loss scale".
            "grad_scaler_state": checkpoint.get("scaler_state_dict"),
        }
        if scheduler_note is not None:
            # ``__log__`` is the only result key the canvas Log tab renders,
            # and dunder keys are filtered out of recorded outputs and port
            # summaries, so this adds a log line and nothing else.
            result["__log__"] = scheduler_note
        return result
