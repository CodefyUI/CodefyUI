"""EvalHFProcessorLoaderNode -- the tokenizer (or processor) that matches a model.

A tokenizer from a different commit than its model is a silent mismatch:
the vocabulary or the special tokens can differ, and the model then reads
ids that mean something else. So with a model connected and ``model_id``
left empty, this loads the processor from the SAME repository at the SAME
resolved commit as the model -- the two are resolved together. A revision
typed here that resolves to a different commit of the model's own
repository is refused.
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

KINDS = ("tokenizer", "auto")


class EvalHFProcessorLoaderNode(BaseNode):
    NODE_NAME = "Eval-HFProcessorLoader"
    CATEGORY = "LLM"
    DESCRIPTION = "The tokenizer or processor matching a model's revision"
    DETAILS = (
        "Connect the model from Eval-HFModelLoader and leave `model_id` and "
        "`revision` empty: the processor is then loaded from the model's own "
        "repository at the commit the model resolved to. A different "
        "`model_id` loads that repository instead, resolved the same way. A "
        "`revision` of the model's repository that resolves to another commit "
        "is refused. `kind` tokenizer loads AutoTokenizer; auto loads "
        "AutoProcessor (image and multimodal processors)."
    )

    cacheable = False
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Model from Eval-HFModelLoader, whose repository and commit to follow.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="processor",
                data_type=DataType.ANY,
                description="The tokenizer or processor, carrying its provenance.",
            ),
            PortDefinition(
                name="provenance",
                data_type=DataType.ANY,
                description="processor id, requested and resolved revision, and how it resolved.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="model_id",
                param_type=ParamType.STRING,
                default="",
                description="Empty follows the connected model. Otherwise a Hub id or local directory.",
            ),
            ParamDefinition(
                name="revision",
                param_type=ParamType.STRING,
                default="",
                description="Empty follows the connected model's commit (or main).",
            ),
            ParamDefinition(
                name="kind",
                param_type=ParamType.SELECT,
                default="tokenizer",
                options=list(KINDS),
                description="tokenizer: AutoTokenizer. auto: AutoProcessor.",
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
        node = self.NODE_NAME
        kind = str(params.get("kind", "tokenizer") or "tokenizer")
        if kind not in KINDS:
            raise ValueError(f"{node}: unknown kind {kind!r}; expected one of {list(KINDS)}.")
        model_prov = hub.provenance_of(inputs.get("model")) if inputs.get("model") is not None else {}
        model_id = str(params.get("model_id", "") or "").strip()
        revision = str(params.get("revision", "") or "").strip()
        follows_model = False
        if not model_id:
            if not model_prov:
                raise ValueError(
                    f"{node}: set `model_id`, or connect a model from "
                    "Eval-HFModelLoader to load its matching processor."
                )
            model_id = str(model_prov["model_id"])
            follows_model = True
        same_repo = bool(model_prov) and model_id == model_prov.get("model_id")
        if same_repo and not revision:
            revision = str(model_prov.get("model_revision") or "")
            follows_model = True

        try:
            resolved = hub.resolve_revision(model_id, revision)
            if (
                same_repo
                and resolved["source"] != "local"
                and resolved["revision"] != model_prov.get("model_revision")
            ):
                raise ValueError(
                    f"{model_id}: revision {revision!r} resolves to "
                    f"{resolved['revision']}, but the connected model was loaded "
                    f"at {model_prov.get('model_revision')}. Leave `revision` "
                    "empty to follow the model."
                )
            processor = hub.load_processor(resolved, kind=kind)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ValueError(f"{node}: {exc}") from exc

        provenance = {
            "processor_id": resolved["model_id"],
            "requested_revision": resolved["requested_revision"],
            "processor_revision": resolved["revision"],
            "revision_source": resolved["source"],
            "revision_note": resolved["note"],
            "kind": kind,
            "follows_model": follows_model,
            "processor_class": type(processor).__name__,
        }
        try:
            hub.stamp(processor, provenance)
        except (AttributeError, TypeError):
            # A processor class that refuses new attributes still works; its
            # provenance then travels on the `provenance` port only.
            pass
        log = (
            f"{resolved['model_id']} @ {resolved['revision']} ({resolved['source']}) "
            f"-- {type(processor).__name__}"
            + (" -- follows the connected model" if follows_model else "")
        )
        return {"processor": processor, "provenance": provenance, "__log__": log}
