"""FreezeParameters: switch ``requires_grad`` off (or on) for part of a model.

Transfer learning and fine-tuning both start by deciding which weights stay
fixed. Doing that in a PythonScript hides the one decision that matters from
anybody reading the graph, so it gets a node: pick parameters by name
prefix, freeze or unfreeze them, and see the counts in the report.

The model is changed in place and passed on, the same object, so a node
downstream (Optimizer, TrainingLoop) sees the new flags. Optimizers skip a
parameter whose gradient stays ``None``, so freezing after the Optimizer was
built still keeps those weights fixed; building it afterwards is cleaner.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


def _prefixes(match: Any) -> list[str]:
    """The comma-separated prefixes of *match*, stripped, empty ones dropped."""
    if not isinstance(match, str):
        return []
    return [part.strip() for part in match.split(",") if part.strip()]


def select_parameters(names: list[str], match: Any, invert: bool) -> list[str]:
    """The parameter names a FreezeParameters node acts on.

    An empty ``match`` selects every parameter. Otherwise a name is selected
    when it starts with one of the prefixes; ``invert`` selects the others,
    which is how "freeze everything except the head" is written.
    """
    prefixes = _prefixes(match)
    if not prefixes:
        hits = set(names)
    else:
        hits = {name for name in names if any(name.startswith(prefix) for prefix in prefixes)}
    return [name for name in names if (name not in hits) == bool(invert)]


class FreezeParametersNode(BaseNode):
    NODE_NAME = "FreezeParameters"
    CATEGORY = "Training"
    DESCRIPTION = "Freeze or unfreeze parameters by name prefix"
    DETAILS = (
        "Sets requires_grad on the parameters whose names start with one of the "
        "comma-separated prefixes in match (empty matches all), or on the others "
        "when invert is on: freeze with match set to the head's prefix and invert "
        "on to train only the head. The model is changed in place and passed on. "
        "The report counts the parameter tensors and parameters that are now "
        "frozen and trainable, and names the prefixes that matched nothing."
    )

    # The node mutates the model it receives; a cache hit would hand back a
    # model whose flags were set by an earlier run.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model to change"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL,
                           description="The same model, with requires_grad set"),
            PortDefinition(name="report", data_type=DataType.STRING,
                           description="Frozen and trainable counts"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="mode", param_type=ParamType.SELECT, default="freeze",
                options=["freeze", "unfreeze"],
                description="Turn requires_grad off (freeze) or on (unfreeze)",
            ),
            ParamDefinition(
                name="match", param_type=ParamType.STRING, default="",
                description="Comma-separated parameter-name prefixes; empty means all",
            ),
            ParamDefinition(
                name="invert", param_type=ParamType.BOOL, default=False,
                description="Act on the parameters that do NOT match",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        model = inputs["model"]
        if not hasattr(model, "named_parameters"):
            raise ValueError("FreezeParameters needs a model (an nn.Module) on its model input")
        named = list(model.named_parameters())
        names = [name for name, _ in named]
        match = params.get("match", "")
        invert = bool(params.get("invert", False))
        selected = set(select_parameters(names, match, invert))
        trainable = params.get("mode", "freeze") == "unfreeze"
        for name, parameter in named:
            if name in selected:
                parameter.requires_grad_(trainable)

        frozen = [(name, p) for name, p in named if not p.requires_grad]
        live = [(name, p) for name, p in named if p.requires_grad]
        unmatched = [prefix for prefix in _prefixes(match)
                     if not any(name.startswith(prefix) for name in names)]
        report = (
            f"frozen: {len(frozen)} tensors, {sum(p.numel() for _, p in frozen):,} parameters; "
            f"trainable: {len(live)} tensors, {sum(p.numel() for _, p in live):,} parameters"
        )
        if unmatched:
            report += f"; no parameter starts with: {', '.join(unmatched)}"
        return {"model": model, "report": report}
