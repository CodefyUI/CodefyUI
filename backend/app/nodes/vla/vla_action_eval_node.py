"""VLAActionEval node (#312) -- open-loop action error on held-out demos.

The fast, low-variance half of VLA evaluation: how close are the policy's
predicted chunks to the expert's on states it never trained on? One number
(mean squared error over action chunks), deterministic per seed, seconds
to compute -- the complement to VLARollout's closed-loop success rate,
which is the real thing but noisy and minutes-scale.

The gap between the two is itself informative: low action MSE with low
closed-loop success is the signature of compounding error (the policy is
fine on-distribution and lost off it), which is exactly the effect
demo_noise and execute_k exist to manage.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class VLAActionEvalNode(BaseNode):
    NODE_NAME = "VLAActionEval"
    CATEGORY = "VLA"
    DESCRIPTION = (
        "Open-loop evaluation: mean squared error between the policy's "
        "predicted action chunks and the expert's, over a held-out demos "
        "dataset (PushWorldDemos' holdout output). Fast and deterministic "
        "per seed - the complement to VLARollout's closed-loop success "
        "rate. A low MSE beside a low success rate is the compounding-"
        "error signature."
    )

    # Consumes live handles (model, dataset); the number describes THIS
    # run's weights.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Trained VLAModel (needs predict_action)",
            ),
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description=(
                    "Held-out BC samples ((image, tokens, chunk), chunk) - "
                    "PushWorldDemos' holdout output"
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="action_mse",
                data_type=DataType.SCALAR,
                description="Mean squared error over evaluated action chunks",
            ),
            PortDefinition(
                name="evaluated",
                data_type=DataType.SCALAR,
                description="Samples actually evaluated",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="max_samples",
                param_type=ParamType.INT,
                default=2048,
                min_value=1,
                max_value=100000,
                description="Evaluate at most this many samples (0 caps nothing)",
            ),
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=256,
                min_value=1,
                max_value=4096,
                description="Inference batch size",
                advanced=True,
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Seeds the flow head's sampling noise so the number is "
                    "reproducible (regression ignores it)"
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description="auto follows the run's device",
                advanced=True,
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
        from ...core.device_utils import resolve_node_device
        from ...core.loop_control import ProgressThrottle, interrupted_result, stop_checker

        model = inputs["model"]
        dataset = inputs["dataset"]
        if not hasattr(model, "predict_action"):
            raise TypeError(
                "VLAActionEval: the model input must be a VLAModel (it has "
                "no predict_action).")
        if len(dataset) == 0:
            raise ValueError(
                "VLAActionEval: the dataset is empty - wire PushWorldDemos' "
                "holdout output (and set holdout_episodes > 0).")

        max_samples = max(0, int(params.get("max_samples", 2048) or 0))
        batch_size = max(1, int(params.get("batch_size", 256) or 256))
        seed = max(0, int(params.get("seed", 0) or 0))
        device = resolve_node_device(params.get("device"), context)
        model_device = next(model.parameters()).device
        model.to(device)

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        target_count = (
            min(len(dataset), max_samples) if max_samples else len(dataset))

        torch.manual_seed(seed)
        error_sum = 0.0
        evaluated = 0
        stopped = False
        try:
            for batch, targets in loader:
                if should_stop():
                    stopped = True
                    break
                if evaluated >= target_count:
                    break
                images, tokens = batch[0], batch[1]
                take = min(images.shape[0], target_count - evaluated)
                images = images[:take].to(device)
                tokens = tokens[:take].to(device)
                expert = targets[:take].to(device)
                predicted = model.predict_action(images, tokens)
                error_sum += float(
                    (predicted.float() - expert.float()).pow(2).sum())
                evaluated += take
                throttle.emit({
                    "event": "progress",
                    "evaluated": evaluated,
                    "total_samples": target_count,
                })
        finally:
            model.to(model_device)

        if evaluated == 0:
            raise RuntimeError("VLAActionEval: stopped before any sample ran.")

        chunk_numel = dataset[0][1].numel()
        action_mse = error_sum / (evaluated * chunk_numel)
        if context is not None:
            context.log_metric("action_mse", action_mse, 0)
        note = f"action MSE {action_mse:.4f} over {evaluated:,} held-out samples."
        logger.info("VLAActionEval: %s", note)
        result: dict[str, Any] = {
            "action_mse": action_mse,
            "evaluated": evaluated,
            "__log__": note,
        }
        if stopped:
            result.update(interrupted_result(sample=evaluated))
        return result
