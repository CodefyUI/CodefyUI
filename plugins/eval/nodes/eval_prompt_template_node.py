"""EvalPromptTemplateNode -- render each item's input through a template.

``{input}`` is the item's current input; any other ``{name}`` is that field
of the item's original JSONL record. The rendered text becomes the item's
new input, so it is what BatchInference feeds the model and what the
prediction table's ``input`` / ``input_hash`` record -- two runs with
different templates are, correctly, two runs over different inputs.

The template's text and hash join the dataset's identity. A field that would
hand the model its own answer -- the label field -- is refused, because a
template that leaks the label turns an evaluation into a copying exercise
whose score looks excellent.
"""

from __future__ import annotations

import string
from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._eval_core import make_dataset, require_dataset, sha256_text


def template_fields(template: str, *, node: str) -> list[str]:
    """The plain field names *template* uses; anything fancier is refused.

    ``{input.__class__}`` and ``{fields[0]}`` are str.format's attribute and
    index lookups, which reach into the Python objects behind a value. A
    prompt has no use for either.
    """
    names: list[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{node}: the template is malformed ({exc}).") from exc
    for _literal, field, _spec, _conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier():
            raise ValueError(
                f"{node}: `{{{field}}}` is not a plain field name; use names "
                "like {input} or {question}."
            )
        names.append(field)
    return names


class EvalPromptTemplateNode(BaseNode):
    NODE_NAME = "Eval-PromptTemplate"
    CATEGORY = "Data"
    DESCRIPTION = "Rewrite each eval item's input through a text template"
    DETAILS = (
        "`{input}` is the item's input; `{name}` is that field of the original "
        "JSONL record. Write `{{` and `}}` for literal braces. The rendered "
        "text replaces the input, and the template's text and sha256 join the "
        "dataset's identity. A template that uses the label field is refused: "
        "it would show the model the answer."
    )

    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.ANY,
                description="Eval dataset from Eval-JSONLDataset.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.ANY,
                description="The same items with the rendered text as their input.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="template",
                param_type=ParamType.STRING,
                default="{input}",
                description="Template text. {input} is the item's input; {field} any record field.",
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
        dataset = require_dataset(inputs.get("dataset"), node=node)
        template = str(params.get("template", "{input}") or "")
        if not template.strip():
            raise ValueError(f"{node}: the template is empty.")
        fields = template_fields(template, node=node)
        meta = dict(dataset.get("meta") or {})
        label_field = str((meta.get("fields") or {}).get("label") or "label")
        if label_field in fields or "label" in fields:
            raise ValueError(
                f"{node}: the template uses {{{label_field}}}, the label field -- "
                "that would show the model the answer it is being scored on."
            )

        rendered_items = []
        for item in dataset["items"]:
            values = dict(item.get("fields") or {})
            values["input"] = item["input"]
            values["id"] = item["id"]
            missing = [f for f in fields if f not in values]
            if missing:
                raise ValueError(
                    f"{node}: item {item['id']!r} has no field(s) {missing} "
                    f"(it has {sorted(values)})."
                )
            try:
                text = template.format_map({f: values[f] for f in fields})
            except (KeyError, IndexError, ValueError, AttributeError) as exc:
                raise ValueError(
                    f"{node}: the template cannot be rendered for item "
                    f"{item['id']!r} ({type(exc).__name__}: {exc})."
                ) from exc
            rendered_items.append({**item, "input": text, "source_input": item["input"]})

        templates = list(meta.get("templates") or [])
        templates.append({"text": template, "sha256": sha256_text(template)})
        meta["templates"] = templates
        meta["template"] = templates[-1]
        return {"dataset": make_dataset(rendered_items, meta)}
