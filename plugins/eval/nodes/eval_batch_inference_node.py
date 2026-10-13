"""EvalBatchInferenceNode -- a model over an eval set, as a prediction table.

The table is the pack's record-table contract with the fixed prediction
columns (``_eval_core.PREDICTION_COLUMNS``) and a ``meta`` that says where
every row came from: the model and processor at their resolved revisions,
the effective inference and generation configuration, the seed, and the
dataset's identity. ``meta.reproducibility`` says whether those together pin
the result down, and lists what is missing when they do not.

Progress is reported per batch; a stop request is honoured between batches,
and the table that comes back still holds every item -- the finished ones
``ok``, the rest ``cancelled`` -- with the node marked interrupted. One item
that raises is marked ``error`` and the others carry on (see
``_eval_infer``).
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
from ._eval_core import (
    NORMALIZE_MODES,
    PREDICTION_COLUMNS,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    format_records,
    input_hash,
    is_pinned_revision,
    json_safe,
    make_table,
    normalize_answer,
    parse_label_map,
    reproducibility,
    require_dataset,
)
from ._eval_infer import (
    MAX_TOPK,
    ClassificationPredictor,
    GenerationPredictor,
    run_batches,
)

TASKS = ("auto", "classification", "generation")


class EvalBatchInferenceNode(BaseNode):
    NODE_NAME = "Eval-BatchInference"
    CATEGORY = "LLM"
    DESCRIPTION = "A model over an eval set, batched, as a prediction table"
    DETAILS = (
        "classification takes the argmax of a sequence-classification model's "
        "logits and names it from the model's id2label; generation decodes "
        "max_new_tokens from a causal LM, greedily or by sampling with a local "
        "generator seeded per item from `seed`. Each row records id, input, "
        "input_hash, raw_answer, the normalised answer, label, status (ok, "
        "error or cancelled) and error. Stop between batches keeps finished "
        "rows as ok and marks the rest cancelled; an item that raises is "
        "marked error and the others carry on. `label_map` renames raw "
        "answers (LABEL_0=negative) before normalising. The table's meta "
        "records both revisions, the effective configuration, the seed and "
        "the dataset's sha256."
    )

    # Hands back partial results on Stop, and its model input is a live
    # handle: a cache hit would replay a table the model may no longer give.
    cacheable = False
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Model from Eval-HFModelLoader.",
            ),
            PortDefinition(
                name="processor",
                data_type=DataType.ANY,
                description="Tokenizer from Eval-HFProcessorLoader.",
            ),
            PortDefinition(
                name="dataset",
                data_type=DataType.ANY,
                description="Eval dataset from Eval-JSONLDataset or Eval-PromptTemplate.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.ANY,
                description="Prediction table: one row per item, with status and provenance.",
            ),
            PortDefinition(
                name="answers",
                data_type=DataType.LIST,
                description="Normalised answers of the ok, labelled rows, in table order.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Labels of the same rows, aligned with `answers`.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="task",
                param_type=ParamType.SELECT,
                default="auto",
                options=list(TASKS),
                description="auto follows the task Eval-HFModelLoader recorded.",
            ),
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                max_value=1024,
                description="Items per batch; progress and Stop are checked between batches.",
            ),
            ParamDefinition(
                name="normalize",
                param_type=ParamType.SELECT,
                default="strip_lower",
                options=list(NORMALIZE_MODES),
                description="How raw answers and labels are normalised before comparison.",
            ),
            ParamDefinition(
                name="label_map",
                param_type=ParamType.STRING,
                default="",
                description="Rename raw answers first, e.g. LABEL_0=negative, LABEL_1=positive.",
            ),
            ParamDefinition(
                name="max_new_tokens",
                param_type=ParamType.INT,
                default=32,
                min_value=1,
                max_value=4096,
                description="Tokens to generate per item (generation only).",
                visible_when={"task": ["auto", "generation"]},
            ),
            ParamDefinition(
                name="do_sample",
                param_type=ParamType.BOOL,
                default=False,
                description="Sample instead of greedy decoding (generation only).",
                advanced=True,
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.01,
                max_value=10.0,
                description="Sampling temperature (generation with do_sample).",
                advanced=True,
            ),
            ParamDefinition(
                name="top_p",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.01,
                max_value=1.0,
                description="Nucleus sampling mass (generation with do_sample).",
                advanced=True,
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Seed for sampling; each item draws from its own generator.",
                advanced=True,
            ),
            ParamDefinition(
                name="max_length",
                param_type=ParamType.INT,
                default=256,
                min_value=8,
                max_value=8192,
                description="Inputs are truncated to this many tokens (classification).",
                advanced=True,
            ),
            ParamDefinition(
                name="top_k",
                param_type=ParamType.INT,
                default=MAX_TOPK,
                min_value=1,
                max_value=MAX_TOPK,
                description="Ranked labels kept per row for Eval-TopKAccuracy (classification).",
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
        from app.core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            stop_checker,
        )

        node = self.NODE_NAME
        model = inputs.get("model")
        processor = inputs.get("processor")
        if model is None:
            raise ValueError(f"{node} requires a `model` input.")
        if processor is None:
            raise ValueError(f"{node} requires a `processor` input.")
        dataset = require_dataset(inputs.get("dataset"), node=node)
        items = list(dataset["items"])

        model_prov = hub.provenance_of(model)
        proc_prov = hub.provenance_of(processor)
        task = str(params.get("task", "auto") or "auto")
        if task not in TASKS:
            raise ValueError(f"{node}: unknown task {task!r}; expected one of {list(TASKS)}.")
        if task == "auto":
            task = str(model_prov.get("task") or "classification")
        _check_revisions_agree(node, model_prov, proc_prov)

        normalize = str(params.get("normalize", "strip_lower") or "strip_lower")
        if normalize not in NORMALIZE_MODES:
            raise ValueError(
                f"{node}: unknown normalize {normalize!r}; expected one of {list(NORMALIZE_MODES)}."
            )
        label_map = parse_label_map(params.get("label_map", ""), node=node)
        batch_size = max(1, int(params.get("batch_size", 8) or 8))
        seed = int(params.get("seed", 0) or 0)

        try:
            model.eval()
        except AttributeError:
            pass

        generation: dict[str, Any] | None = None
        if task == "classification":
            predictor = ClassificationPredictor(
                model, processor,
                max_length=int(params.get("max_length", 256) or 256),
                top_k=int(params.get("top_k", MAX_TOPK) or MAX_TOPK),
            )
            predict_batch = predictor.predict_batch
            predict_one = predictor.predict_one
            extras = ["score", "topk"]
            inference = {"task": task, "batch_size": batch_size,
                         "max_length": predictor.max_length, "top_k": predictor.top_k}
        else:
            temperature = float(params.get("temperature", 1.0) or 1.0)
            top_p = float(params.get("top_p", 1.0) or 1.0)
            if not 0.0 < top_p <= 1.0:
                raise ValueError(f"{node}: `top_p` must be in (0, 1]; got {top_p}.")
            predictor = GenerationPredictor(
                model, processor,
                max_new_tokens=int(params.get("max_new_tokens", 32) or 32),
                do_sample=bool(params.get("do_sample", False)),
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            predict_batch = None
            predict_one = predictor.predict_one
            extras = ["new_tokens"]
            generation = predictor.config()
            inference = {"task": task, "batch_size": batch_size}

        throttle = ProgressThrottle(progress_callback)

        def on_batch(batch: int, total_batches: int, done: int) -> None:
            throttle.emit({
                "event": EVENT_BATCH,
                "batch": batch,
                "total_batches": total_batches,
                "completed": done,
                "total": len(items),
            })

        outcomes, stopped_at = run_batches(
            items,
            batch_size=batch_size,
            predict_batch=predict_batch,
            predict_one=predict_one,
            should_stop=stop_checker(context),
            on_batch=on_batch,
        )

        rows = []
        answers: list[str] = []
        labels: list[str] = []
        counts = {STATUS_OK: 0, STATUS_ERROR: 0, STATUS_CANCELLED: 0}
        for item, outcome in zip(items, outcomes):
            status = outcome["status"]
            counts[status] += 1
            text = item.get("input")
            label = item.get("label")
            norm_label = normalize_answer(label, normalize) if label is not None else None
            answer = (
                normalize_answer(outcome["raw_answer"], normalize, label_map)
                if status == STATUS_OK else None
            )
            extra_values = []
            for name in extras:
                value = outcome["extras"].get(name)
                if name == "topk" and isinstance(value, list):
                    value = [normalize_answer(v, normalize, label_map) for v in value]
                extra_values.append(value)
            rows.append([
                str(item["id"]),
                text if isinstance(text, str) else None,
                input_hash(text) if isinstance(text, str) else None,
                outcome["raw_answer"],
                answer,
                norm_label,
                status,
                outcome["error"],
                *extra_values,
            ])
            if status == STATUS_OK and norm_label is not None:
                answers.append(answer)
                labels.append(norm_label)

        dataset_meta = dict(dataset.get("meta") or {})
        producer = {
            "node": node,
            "model_id": model_prov.get("model_id"),
            "requested_revision": model_prov.get("requested_revision"),
            "model_revision": model_prov.get("model_revision"),
            "revision_source": model_prov.get("revision_source"),
            "processor_id": proc_prov.get("processor_id"),
            "processor_revision": proc_prov.get("processor_revision"),
            "processor_revision_source": proc_prov.get("revision_source"),
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "dtype": model_prov.get("dtype"),
            "device": model_prov.get("device"),
            "normalize": normalize,
            "label_map": label_map,
        }
        meta = json_safe({
            "producer": producer,
            "inference": inference,
            "generation": generation,
            "seed": seed,
            "dataset": {
                "path": dataset_meta.get("path"),
                "sha256": dataset_meta.get("sha256"),
                "rows": dataset_meta.get("rows"),
                "used": dataset_meta.get("used"),
                "template": dataset_meta.get("template"),
            },
            "counts": counts,
            "interrupted_at_batch": stopped_at,
            "reproducibility": reproducibility(producer, dataset_meta, generation),
        })
        table = make_table([*PREDICTION_COLUMNS, *extras], rows, meta)

        model_name = producer["model_id"] or type(model).__name__
        revision = str(producer["model_revision"] or "unpinned")
        summary = (
            f"{len(items)} item(s): {counts[STATUS_OK]} ok, {counts[STATUS_ERROR]} error, "
            f"{counts[STATUS_CANCELLED]} cancelled -- {model_name} @ {revision}"
        )
        if not meta["reproducibility"]["reproducible"]:
            summary += "\nnot reproducible from this record: " + "; ".join(
                meta["reproducibility"]["reasons"])
        preview_cols = ["id", "answer", "label", "status"]
        index = [list(table["columns"]).index(c) for c in preview_cols]
        preview = format_records(preview_cols, [[r[i] for i in index] for r in rows])
        result: dict[str, Any] = {
            "predictions": table,
            "answers": answers,
            "labels": labels,
            "__log__": f"{summary}\n{preview}",
        }
        if stopped_at is not None:
            result.update(interrupted_result(batch=stopped_at, completed=counts[STATUS_OK]))
        return result


def _check_revisions_agree(node: str, model_prov: dict[str, Any], proc_prov: dict[str, Any]) -> None:
    """Refuse a processor from another commit of the model's own repository."""
    if not model_prov or not proc_prov:
        return
    if model_prov.get("model_id") != proc_prov.get("processor_id"):
        return
    a, b = model_prov.get("model_revision"), proc_prov.get("processor_revision")
    if is_pinned_revision(a) and is_pinned_revision(b) and a != b:
        raise ValueError(
            f"{node}: the model is {model_prov.get('model_id')} @ {a} but the processor "
            f"is the same repository @ {b}. Load the processor from the model "
            "(Eval-HFProcessorLoader with `revision` empty) so the two match."
        )
