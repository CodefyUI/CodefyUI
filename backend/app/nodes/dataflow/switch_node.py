"""Switch -- forward one of several inputs, picked by an index (a MUX).

Several sources wired into one input used to merge by edge order: the last
edge whose source produced a value won, and nothing on the canvas showed
which one that was (#562). A Switch is the explicit way to choose between
sources, so its choice must never be a guess either: an index that names no
wired input is an error, where it used to fall back to ``input_0`` silently.
"""

from __future__ import annotations

import math
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
    resolve_count_param,
)

#: Bounds on the ``inputs`` param. The default is the four ports Switch had
#: before the param existed, so a saved graph without it, which the editor
#: completes with this default, keeps every port it was wired to.
MIN_INPUTS = 2
MAX_INPUTS = 16
DEFAULT_INPUTS = 4

SELECTOR = "selector"
OUTPUT = "output"


def input_count(params: dict[str, Any] | None) -> int:
    """How many ``input_N`` ports this instance has. Clamps rather than raises."""
    return resolve_count_param(
        params, "inputs",
        default=DEFAULT_INPUTS, minimum=MIN_INPUTS, maximum=MAX_INPUTS,
    )


def input_name(index: int) -> str:
    return f"input_{index}"


def coerce_selector(raw: Any) -> int:
    """The index *raw* names. Raises ``ValueError`` for anything else.

    A one-element tensor and an integral float (``1.0``) are accepted, since
    a SCALAR port carries either. ``1.7`` is refused: truncating it picked
    ``input_1`` without a word, which is the guess this node exists to stop.
    """
    value = raw
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Switch: selector must be a single number, got a value of shape "
                f"{tuple(getattr(raw, 'shape', ()))}"
            ) from exc
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Switch: selector must be an integer index, got {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"Switch: selector must be an integer index, got {value!r}")
        return int(value)
    if isinstance(value, int):
        return value
    raise ValueError(f"Switch: selector must be an integer index, got {value!r}")


class SwitchNode(BaseNode):
    NODE_NAME = "Switch"
    CATEGORY = "Data Flow"
    DESCRIPTION = "Forwards one of its inputs, picked by a 0-based index"
    DETAILS = (
        "The selector param picks the input; wiring the selector port overrides it "
        "with a value computed during the run. Wiring the last empty input adds "
        "another. An index that is out of range or names an unwired input stops the "
        "run with an error. Every wired input is computed before the Switch runs, "
        "including the ones it does not pick."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return cls.define_inputs_dynamic(None)

    @classmethod
    def define_inputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        # The options come first and the selector last: bypass forwards the
        # FIRST declared input compatible with the output (``resolve_bypass``),
        # and a bypassed Switch must pass input_0 on, not its index.
        return [
            *(
                PortDefinition(
                    name=input_name(index),
                    data_type=DataType.ANY,
                    description=f"Option {index}",
                    # Which input must be wired depends on the selector, so
                    # none is required by itself; validation and the run
                    # check the one the selector names.
                    optional=True,
                )
                for index in range(input_count(params))
            ),
            PortDefinition(
                name=SELECTOR,
                data_type=DataType.SCALAR,
                description="Index of the input to forward; overrides the selector param when wired",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name=OUTPUT,
                data_type=DataType.ANY,
                description="The selected input; its type is the type of the wired inputs",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name=SELECTOR,
                param_type=ParamType.INT,
                default=0,
                description="Index of the input to forward (0-based); the selector port overrides it",
                min_value=0,
                max_value=MAX_INPUTS - 1,
            ),
            ParamDefinition(
                name="inputs",
                param_type=ParamType.INT,
                default=DEFAULT_INPUTS,
                description="How many inputs to choose between",
                min_value=MIN_INPUTS,
                max_value=MAX_INPUTS,
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
        count = input_count(params)
        if SELECTOR in inputs:
            index = coerce_selector(inputs[SELECTOR])
            source = "selector port"
        else:
            index = coerce_selector((params or {}).get(SELECTOR, 0))
            source = "selector param"
        if not 0 <= index < count:
            raise ValueError(
                f"Switch: {source} is {index}, but this Switch has inputs 0 to {count - 1}"
            )
        key = input_name(index)
        if key not in inputs:
            raise ValueError(
                f"Switch: {source} is {index}, but {key} has no value "
                "(it is not connected, or its source produced nothing)"
            )
        return {OUTPUT: inputs[key]}


# ── Graph-level rules ─────────────────────────────────────────────────────
#
# A Switch's port list cannot say two things the graph decides: which input
# its param selects, and what type its output carries. ``validate_graph``
# asks these, so the editor's check, Run and Export refuse the same graphs.


def _is_switch(node_cls: Any) -> bool:
    return isinstance(node_cls, type) and issubclass(node_cls, SwitchNode)


def _params_of(node: dict) -> dict[str, Any]:
    data = node.get("data")
    params = data.get("params") if isinstance(data, dict) else None
    return params if isinstance(params, dict) else {}


def switch_output_types(
    nodes: list[dict],
    edges: list[dict],
    registry: Any,
    opaque_node_ids: set[str] = frozenset(),
) -> dict[str, DataType]:
    """The type each Switch's output carries, by Switch node id.

    That is the one concrete type its wired inputs carry. A Switch whose
    inputs are all ``ANY`` (or unwired, or disagree -- reported on its own by
    :func:`switch_graph_errors`) maps to ``ANY``, which every port accepts.
    A Switch fed by another Switch takes that one's type.
    """
    node_map = {n["id"]: n for n in nodes}
    feeds: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        if edge.get("type", "data") != "data":
            continue
        handle = edge.get("targetHandle", "") or ""
        if handle.startswith("input_"):
            feeds.setdefault(edge["target"], []).append(
                (edge["source"], edge.get("sourceHandle", "") or "")
            )

    resolved: dict[str, DataType] = {}

    def concrete_types(switch_id: str, visiting: set[str]) -> set[DataType]:
        found: set[DataType] = set()
        for src_id, src_port in feeds.get(switch_id, []):
            found.add(source_type(src_id, src_port, visiting))
        found.discard(DataType.ANY)
        return found

    def source_type(src_id: str, src_port: str, visiting: set[str]) -> DataType:
        src = node_map.get(src_id)
        if src is None or src_id in opaque_node_ids:
            return DataType.ANY
        src_cls = registry.get(src.get("type", ""))
        if src_cls is None:
            return DataType.ANY
        if _is_switch(src_cls) and src_port == OUTPUT:
            return of(src_id, visiting)
        for port in src_cls.define_outputs_dynamic(_params_of(src)):
            if port.name == src_port:
                return port.data_type
        return DataType.ANY

    def of(switch_id: str, visiting: set[str]) -> DataType:
        if switch_id in resolved:
            return resolved[switch_id]
        if switch_id in visiting:
            # A loop; the cycle check names it.
            return DataType.ANY
        types = concrete_types(switch_id, visiting | {switch_id})
        result = next(iter(types)) if len(types) == 1 else DataType.ANY
        resolved[switch_id] = result
        return result

    for node in nodes:
        if node["id"] in opaque_node_ids:
            continue
        if _is_switch(registry.get(node.get("type", ""))):
            of(node["id"], set())
    return resolved


def switch_graph_errors(
    nodes: list[dict],
    edges: list[dict],
    registry: Any,
    opaque_node_ids: set[str] = frozenset(),
) -> list[str]:
    """What is wrong with each Switch that its ports alone cannot show.

    - Its wired inputs carry different concrete types, so the type of its
      output depends on the selector.
    - Its selector comes from the param (the port is unwired) and names an
      input it does not have, or one that is not wired. A selector from the
      port is only known during the run, which checks it then.
    """
    from ...core.validation_issues import validation_issue

    node_map = {n["id"]: n for n in nodes}
    wired: dict[str, set[str]] = {}
    for edge in edges:
        if edge.get("type", "data") != "data":
            continue
        wired.setdefault(edge["target"], set()).add(edge.get("targetHandle", "") or "")

    errors: list[str] = []
    for node in nodes:
        node_id = node["id"]
        if node_id in opaque_node_ids:
            continue
        if not _is_switch(registry.get(node.get("type", ""))):
            continue
        params = _params_of(node)

        types: set[DataType] = set()
        for edge in edges:
            if edge.get("type", "data") != "data" or edge["target"] != node_id:
                continue
            if not (edge.get("targetHandle", "") or "").startswith("input_"):
                continue
            src = node_map.get(edge["source"])
            if src is None or src["id"] in opaque_node_ids:
                continue
            src_cls = registry.get(src.get("type", ""))
            if src_cls is None:
                continue
            src_port = edge.get("sourceHandle", "") or ""
            if _is_switch(src_cls) and src_port == OUTPUT:
                continue  # its own type is checked at that Switch
            for port in src_cls.define_outputs_dynamic(_params_of(src)):
                if port.name == src_port and port.data_type is not DataType.ANY:
                    types.add(port.data_type)
        if len(types) > 1:
            names = ", ".join(sorted(t.value for t in types))
            errors.append(validation_issue(
                "switch_input_types_differ",
                (
                    f"Switch {node_id}: its inputs carry different types ({names}); "
                    "a Switch forwards one type, whichever input it selects"
                ),
                node_id=node_id, types=names,
            ))

        if SELECTOR in wired.get(node_id, set()):
            continue
        try:
            index = coerce_selector(params.get(SELECTOR, 0))
        except ValueError:
            # Not a number: the parameter range check already says so.
            continue
        count = input_count(params)
        if not 0 <= index < count:
            errors.append(validation_issue(
                "switch_selector_out_of_range",
                (
                    f"Switch {node_id}: selector is {index}, but it has inputs "
                    f"0 to {count - 1}"
                ),
                node_id=node_id, value=index, max=count - 1,
            ))
        elif input_name(index) not in wired.get(node_id, set()):
            errors.append(validation_issue(
                "switch_selected_unwired",
                (
                    f"Switch {node_id}: selector is {index}, but "
                    f"{input_name(index)} is not connected"
                ),
                node_id=node_id, value=index, port=input_name(index),
            ))
    return errors
