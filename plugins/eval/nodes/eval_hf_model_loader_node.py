"""EvalHFModelLoaderNode -- a Hugging Face model at one exact revision.

The requested revision is resolved to a commit sha (or, for a directory on
disk, a content hash) BEFORE loading, and the model that comes out is the
model at that sha. Both the request and the resolution are stamped on the
model and handed out on ``provenance``, which is how a prediction table can
name the exact weights it came from instead of a branch name that moves.
See ``_eval_hub`` for the resolution rules, the cache directory and offline
behaviour.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from . import _eval_hub as hub


class EvalHFModelLoaderNode(BaseNode):
    NODE_NAME = "Eval-HFModelLoader"
    CATEGORY = "LLM"
    DESCRIPTION = "A Hugging Face model pinned to its resolved commit"
    DETAILS = (
        "`revision` (a branch, tag or commit; default main) is resolved to a "
        "commit sha before loading: from the Hub when it is reachable, else "
        "from the snapshot already cached, and the source is recorded. "
        "HF_HUB_OFFLINE=1 is respected -- nothing is downloaded and an "
        "uncached model is an error. A local directory is identified by the "
        "sha256 of its files. Downloads go to CodefyUI's own cache under the "
        "user data directory. `task` auto reads the config's architecture: "
        "...ForSequenceClassification is classification, a causal LM is "
        "generation. Needs the transformers package, which installing this "
        "pack brings."
    )

    # Hands out a live model (#254), and reads the network.
    cacheable = False
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="The loaded model, in eval mode, carrying its provenance.",
            ),
            PortDefinition(
                name="provenance",
                data_type=DataType.ANY,
                description="model_id, requested and resolved revision, how it resolved, task, dtype, device.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="model_id",
                param_type=ParamType.STRING,
                default="",
                description="Hugging Face Hub id (org/name) or a local model directory.",
            ),
            ParamDefinition(
                name="revision",
                param_type=ParamType.STRING,
                default="main",
                description="Branch, tag or commit sha. Resolved to a commit sha before loading.",
            ),
            ParamDefinition(
                name="task",
                param_type=ParamType.SELECT,
                default="auto",
                options=list(hub.TASKS),
                description="classification (argmax over labels) or generation (causal LM); auto reads the config.",
            ),
            ParamDefinition(
                name="dtype",
                param_type=ParamType.SELECT,
                default="float32",
                options=list(hub.DTYPES),
                description="Precision to cast the weights to. auto keeps the checkpoint's own.",
                advanced=True,
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description="auto follows the run's device.",
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
        from app.core.device_utils import resolve_node_device

        node = self.NODE_NAME
        task = str(params.get("task", "auto") or "auto")
        if task not in hub.TASKS:
            raise ValueError(f"{node}: unknown task {task!r}; expected one of {list(hub.TASKS)}.")
        dtype = str(params.get("dtype", "float32") or "float32")
        if dtype not in hub.DTYPES:
            raise ValueError(f"{node}: unknown dtype {dtype!r}; expected one of {list(hub.DTYPES)}.")
        device = resolve_node_device(params.get("device"), context)

        try:
            resolved = hub.resolve_revision(
                str(params.get("model_id", "") or ""), str(params.get("revision", "") or "")
            )
            model, task = hub.load_model(resolved, task=task, dtype=dtype, device=device)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ValueError(f"{node}: {exc}") from exc

        provenance = {
            "model_id": resolved["model_id"],
            "requested_revision": resolved["requested_revision"],
            "model_revision": resolved["revision"],
            "revision_source": resolved["source"],
            "revision_note": resolved["note"],
            "task": task,
            "dtype": dtype,
            "device": device,
            "model_class": type(model).__name__,
        }
        hub.stamp(model, provenance)
        log = (
            f"{resolved['model_id']} @ {resolved['revision']} "
            f"({resolved['source']}; asked for {resolved['requested_revision']}) "
            f"-- {task}, {dtype}, {device}"
        )
        return {"model": model, "provenance": provenance, "__log__": log}
