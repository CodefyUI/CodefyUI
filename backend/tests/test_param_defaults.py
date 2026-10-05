"""A param the graph leaves out runs as its declared default, on every path.

The editor fills in every param a node declares when it loads a graph (#570),
but a graph reaches the backend by other roads too: ``cdui run``,
``POST /api/graph/run`` and published apps, a preset's internal nodes (which
carry only the params their author set), the Map node's inner calls, and an
exported script whose ``params = {...}`` someone trimmed. Each node used to
read a missing param with its own ``params.get(name, fallback)``, and dozens
of those fallbacks disagree with the declared default -- GridWorldEnv ran with
no traps instead of its default ``"1,1"`` -- so the canvas and the CLI could
run one file two different ways.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from app.core.execution_context import ExecutionContext
from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


def _fill(node_cls: Any, params: Any) -> Any:
    from app.core.param_defaults import fill_missing_params

    return fill_missing_params(node_cls, params)


def _declared(node_cls: Any) -> dict[str, Any]:
    return {p.name: p.default for p in node_cls.define_params()}


# ── fill_missing_params ──────────────────────────────────────────────────

#: One list object handed out on every ``define_params()`` call, the way a
#: module-level constant would be -- so a default that is not copied would
#: be shared between every node that reads it.
_SHARED_GRID = [[1.0, 2.0]]


class _Spec:
    """Stands in for a node class: the helper reads only ``define_params``."""

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="count", param_type=ParamType.INT, default=3),
            ParamDefinition(name="label", param_type=ParamType.STRING, default="x"),
            ParamDefinition(name="flag", param_type=ParamType.BOOL, default=True),
            ParamDefinition(
                name="grid", param_type=ParamType.TENSOR_GRID, default=_SHARED_GRID),
            ParamDefinition(
                name="unset", param_type=ParamType.TENSOR_GRID, default=None),
        ]


class _CannotDescribeItself:
    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        raise RuntimeError("broken plugin")


class _Uncopyable:
    def __deepcopy__(self, memo):
        raise TypeError("cannot be copied")


_UNCOPYABLE = _Uncopyable()


class _DeclaresAnUncopyableDefault:
    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="handle", param_type=ParamType.STRING, default=_UNCOPYABLE),
            ParamDefinition(name="count", param_type=ParamType.INT, default=3),
        ]


def test_fills_every_declared_param_the_params_leave_out():
    assert _fill(_Spec, {"count": 5}) == {
        "count": 5, "label": "x", "flag": True, "grid": [[1.0, 2.0]],
        "unset": None,
    }


def test_keeps_every_value_the_params_carry_falsy_ones_included():
    params = {"count": 0, "label": "", "flag": False, "unset": None}
    assert _fill(_Spec, params) == {**params, "grid": [[1.0, 2.0]]}


def test_complete_params_come_back_as_the_same_dict():
    params = {"count": 1, "label": "y", "flag": False, "grid": [], "unset": None}
    assert _fill(_Spec, params) is params


def test_the_params_handed_in_are_never_changed():
    params = {"count": 1}
    filled = _fill(_Spec, params)
    assert params == {"count": 1}
    assert filled is not params


def test_a_list_default_is_a_copy_not_the_declared_object():
    first = _fill(_Spec, {})
    first["grid"][0][0] = 99.0
    assert _fill(_Spec, {})["grid"] == [[1.0, 2.0]]
    assert _SHARED_GRID == [[1.0, 2.0]]


def test_a_default_that_cannot_be_copied_is_handed_over_as_it_is():
    assert _fill(_DeclaresAnUncopyableDefault, {}) == {"handle": _UNCOPYABLE, "count": 3}
    assert _fill(_DeclaresAnUncopyableDefault, {})["handle"] is _UNCOPYABLE


def test_a_node_that_cannot_describe_itself_keeps_its_params():
    params = {"anything": 1}
    assert _fill(_CannotDescribeItself, params) is params


# ── the engine ───────────────────────────────────────────────────────────


def _node(node_id: str, node_type: str, **params: Any) -> dict:
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": {"params": params}}


def _trigger(target: str) -> dict:
    return {"id": f"t-{target}", "source": "start", "target": target,
            "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"}


def _data(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}->{target}", "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle,
            "type": "data"}


def _context() -> ExecutionContext:
    return ExecutionContext(device="cpu", weights_persistent=False, graph_id="t9")


async def _run(nodes: list[dict], edges: list[dict], **kwargs: Any) -> dict:
    from app.core.graph_engine import execute_graph

    return await execute_graph(nodes, edges, context=_context(), **kwargs)


def _gridworld(**params: Any) -> tuple[list[dict], list[dict]]:
    return [_node("start", "Start"), _node("gw", "GridWorldEnv", **params)], [
        _trigger("gw")]


def _columns(**selector_params: Any) -> tuple[list[dict], list[dict]]:
    nodes = [
        _node("start", "Start"),
        _node("src", "TensorInput", shape="2,3", value_mode="arange"),
        _node("sel", "ColumnSelector", **selector_params),
    ]
    return nodes, [_trigger("src"), _data("src", "tensor", "sel", "tensor")]


def _gridworld_traps() -> str:
    from app.nodes.rl.gridworld_env_node import GridWorldEnvNode

    traps = _declared(GridWorldEnvNode)["traps"]
    # GridWorldEnv reads a missing ``traps`` as "" -- no traps. The test can
    # only tell filling from not filling while the default says otherwise.
    assert traps.strip(), "pick a param whose default differs from its fallback"
    return traps


@pytest.mark.asyncio
async def test_gridworld_with_traps_left_out_runs_its_declared_traps():
    left_out = await _run(*_gridworld(size=4))
    spelled = await _run(*_gridworld(size=4, traps=_gridworld_traps()))

    assert spelled["gw"]["env"].traps  # not two empty sets agreeing
    assert left_out["gw"]["env"].traps == spelled["gw"]["env"].traps
    assert left_out["gw"]["layout"] == spelled["gw"]["layout"]


@pytest.mark.asyncio
async def test_column_selector_with_indices_left_out_runs_its_declared_indices():
    from app.nodes.data.column_selector_node import ColumnSelectorNode

    indices = _declared(ColumnSelectorNode)["indices"]
    left_out = await _run(*_columns())
    spelled = await _run(*_columns(indices=indices))

    assert spelled["sel"]["tensor"].shape[-1] > 0
    assert torch.equal(left_out["sel"]["tensor"], spelled["sel"]["tensor"])


@pytest.mark.asyncio
async def test_a_left_out_param_and_its_spelled_out_default_share_a_cache_entry():
    """The cache key is computed from the params the node RUNS with.

    Filling only inside ``invoke_node`` would hand the node the default but
    key the cache on the trimmed params -- and a ``cache_fingerprint`` that
    reads a path param would describe a file the node never opens.
    """
    from app.core.cache import ExecutionCache

    cache = ExecutionCache()
    statuses: list[tuple[str, str]] = []

    async def on_progress(node_id: str, status: str, data: Any) -> None:
        statuses.append((node_id, status))

    from app.nodes.data.column_selector_node import ColumnSelectorNode

    await _run(*_columns(), cache=cache, on_progress=on_progress)
    statuses.clear()
    await _run(*_columns(indices=_declared(ColumnSelectorNode)["indices"]),
               cache=cache, on_progress=on_progress)

    assert ("sel", "cached") in statuses, statuses


# ── invoke_node: a hand-edited export, the Map node's inner calls ────────


class _Echo(BaseNode):
    """Hands back the params it was given, so a test can see what arrived."""

    NODE_NAME = "_ParamDefaultsEcho"
    CATEGORY = "Test"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="params", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="count", param_type=ParamType.INT, default=3),
            ParamDefinition(name="label", param_type=ParamType.STRING, default="x"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"params": dict(params)}


def test_invoke_node_hands_execute_every_declared_param():
    from app.core.graph_engine import invoke_node

    trimmed = {"label": "kept"}
    seen = invoke_node(_Echo(), {}, trimmed)["params"]

    assert seen == {"count": 3, "label": "kept"}
    # An exported script's literal, or a preset's own params: never edited.
    assert trimmed == {"label": "kept"}


def test_invoke_node_runs_a_real_node_with_its_declared_default():
    from app.core.graph_engine import invoke_node
    from app.nodes.rl.gridworld_env_node import GridWorldEnvNode

    left_out = invoke_node(GridWorldEnvNode(), {}, {"size": 4})
    spelled = invoke_node(
        GridWorldEnvNode(), {}, {"size": 4, "traps": _gridworld_traps()})

    assert left_out["layout"] == spelled["layout"]


# ── prepare_executable_graph ─────────────────────────────────────────────


def test_preparing_a_run_fills_params_on_copies_of_the_callers_nodes():
    from app.core.graph_engine import prepare_executable_graph
    from app.nodes.rl.gridworld_env_node import GridWorldEnvNode

    nodes, edges = _gridworld(size=4)
    before = copy.deepcopy(nodes)

    prepared, _edges, _containers = prepare_executable_graph(
        nodes, edges, fill_defaults=True)

    assert nodes == before
    gridworld = next(node for node in prepared if node["id"] == "gw")
    assert gridworld["data"]["params"] == {**_declared(GridWorldEnvNode), "size": 4}


def test_preparing_without_the_flag_keeps_the_params_the_graph_carries():
    """What the exporter asks for: its script lists exactly these."""
    from app.core.graph_engine import prepare_executable_graph

    prepared, _edges, _containers = prepare_executable_graph(*_gridworld(size=4))

    gridworld = next(node for node in prepared if node["id"] == "gw")
    assert gridworld["data"]["params"] == {"size": 4}


def test_an_unknown_node_type_still_reaches_validation_by_name():
    from app.core.graph_engine import GraphValidationError, prepare_executable_graph

    nodes = [_node("start", "Start"), _node("x", "NoSuchNodeType", a=1)]
    with pytest.raises(GraphValidationError, match="Unknown node type: NoSuchNodeType"):
        prepare_executable_graph(nodes, [_trigger("x")], fill_defaults=True)


# ── Export as Python ─────────────────────────────────────────────────────


def _exported_params(script: str, node_id: str) -> dict:
    """The ``params`` literal of one node function, read without running it."""
    for statement in ast.parse(script).body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_call"
            and call.args[1].value == node_id
            for call in ast.walk(statement)
        ):
            continue
        for inner in statement.body:
            if (
                isinstance(inner, ast.Assign)
                and isinstance(inner.targets[0], ast.Name)
                and inner.targets[0].id == "params"
            ):
                return ast.literal_eval(inner.value)
    raise AssertionError(f"no params assignment for node {node_id!r}")


def _run_exported(script: str) -> dict:
    """Run a generated script's ``run_graph`` in this process.

    The registry this process already holds stands in for the script's own
    runtime discovery, so nothing is re-registered mid-suite.
    """
    module: dict[str, Any] = {"__name__": "exported_graph"}
    exec(compile(script, "<generated>", "exec"), module)
    module["_RT"] = module["_load_runtime"](None)
    context = ExecutionContext(
        seed=module.get("GRAPH_SEED"),
        deterministic=bool(module.get("GRAPH_DETERMINISTIC", False)),
    )
    return module["run_graph"](context, {})


def test_the_export_lists_only_the_params_the_graph_carries():
    """The script's ``params = {...}`` is the graph's, not a filled copy.

    A grader reads these literals out of the file, so the export keeps
    listing exactly what the graph holds; the defaults are filled in when
    the script RUNS.
    """
    from app.core.codegen import generate_python

    script = generate_python(*_gridworld(size=4), name="gridworld")

    assert _exported_params(script, "gw") == {"size": 4}


@pytest.mark.asyncio
async def test_the_exported_script_runs_a_left_out_param_at_its_declared_default():
    from app.core.codegen import generate_python

    script = generate_python(*_gridworld(size=4), name="gridworld")
    exported = _run_exported(script)
    canvas = await _run(*_gridworld(size=4, traps=_gridworld_traps()))

    assert exported["gw"]["layout"] == canvas["gw"]["layout"]


# ── every declared default, linted ───────────────────────────────────────


@pytest.fixture(scope="module")
def every_node_class(tmp_path_factory) -> dict[str, Any]:
    """Core nodes and every pack bundled under ``plugins/``, in a registry of
    their own, so the suite's shared registry is not changed by this file."""
    from app.config import settings
    from app.core import plugin_loader
    from app.core.node_registry import NodeRegistry

    private = NodeRegistry()
    private.discover(settings.NODES_DIR, "app.nodes")
    root = plugin_loader.plugins_builtin_root()
    packs = sorted(
        pack.name for pack in root.iterdir()
        if (pack / plugin_loader.MANIFEST_FILENAME).is_file()
    )
    plugin_loader.discover_plugin_nodes(
        private, root, tmp_path_factory.mktemp("no-user-plugins"),
        {"schema": 1, "plugins": {
            pack: {"source_kind": "builtin", "source": pack, "enabled": True}
            for pack in packs
        }},
    )
    nodes = dict(private.nodes)
    for pack in packs:
        assert any(name.startswith(f"{pack}:") for name in nodes), pack
    return nodes


def _default_problem(definition: ParamDefinition) -> str | None:
    """Why *definition*'s default is not a value of its own type, or None."""
    value, kind = definition.default, definition.param_type
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == ParamType.INT:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif kind == ParamType.FLOAT:
        valid = is_number
    elif kind == ParamType.BOOL:
        valid = isinstance(value, bool)
    elif kind == ParamType.SELECT:
        valid = not definition.options or value in definition.options
    elif kind == ParamType.TENSOR_GRID:
        valid = value is None or isinstance(value, list)
    else:  # STRING, CODE, SECRET and the three file pickers
        valid = isinstance(value, str)
    if not valid:
        return f"default {value!r} is not a valid {kind}"
    if is_number and definition.min_value is not None and value < definition.min_value:
        return f"default {value!r} is below min_value {definition.min_value}"
    if is_number and definition.max_value is not None and value > definition.max_value:
        return f"default {value!r} is above max_value {definition.max_value}"
    return None


def test_every_declared_default_is_a_valid_value_of_its_own_type(every_node_class):
    """A left-out param now RUNS as its default, so the default must be a
    value the param accepts -- the canvas would refuse it typed by hand."""
    problems = [
        f"{name}.{definition.name}: {problem}"
        for name, node_cls in sorted(every_node_class.items())
        for definition in node_cls.define_params()
        if (problem := _default_problem(definition)) is not None
    ]
    assert not problems, "\n".join(problems)


def test_dynamic_ports_are_the_same_with_and_without_the_defaults(every_node_class):
    """The exporter validates the graph as it carries its params and the
    engine validates it filled; both must see the same ports."""

    def ports(found: list[PortDefinition]) -> list[tuple]:
        return [(port.name, str(port.data_type), port.optional) for port in found]

    drift = []
    for name, node_cls in sorted(every_node_class.items()):
        defaults = _declared(node_cls)
        for method in ("define_inputs_dynamic", "define_outputs_dynamic"):
            bare = ports(getattr(node_cls, method)({}))
            filled = ports(getattr(node_cls, method)(defaults))
            if bare != filled:
                drift.append(f"{name}.{method}: {bare} != {filled}")
    assert not drift, "\n".join(drift)


# ── DataMixDataset ───────────────────────────────────────────────────────


class _Rows(Dataset):
    """A corpus: rows of raw text."""

    def __init__(self, rows: list[str]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> str:
        return self._rows[index]


def test_data_mix_with_weights_left_out_draws_from_any_number_of_sources():
    """The default must work for every source count: "0.5, 0.5" did not."""
    from app.core.graph_engine import invoke_node
    from app.nodes.llm.data_mix_dataset_node import DataMixDatasetNode

    assert _declared(DataMixDatasetNode)["weights"] == ""

    corpora = {
        f"corpus_{index + 1}": _Rows([f"{name}{row}" for row in range(4)])
        for index, name in enumerate("abc")
    }
    result = invoke_node(DataMixDatasetNode(), corpora, {"sources": 3})

    rows = [result["dataset"][index] for index in range(result["num_rows"])]
    assert sorted(rows) == sorted(f"{name}{row}" for name in "abc" for row in range(4))


# ── a file loader with no file chosen ────────────────────────────────────


def _model_and_optimizer():
    model = torch.nn.Linear(2, 2)
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _no_file_message(node_name: str, what: str) -> str:
    return re.escape(
        f"{node_name} has no file selected. Pick one from the `path` dropdown, "
        f"or use the upload button next to it to add a {what}.")


@pytest.mark.parametrize("params", [{"path": ""}, {"path": "  "}, {}],
                         ids=["empty", "blank", "left out"])
def test_model_loader_with_no_file_selected_says_so(params):
    from app.core.graph_engine import invoke_node
    from app.nodes.io.model_loader_node import ModelLoaderNode

    model, _optimizer = _model_and_optimizer()
    with pytest.raises(ValueError, match=_no_file_message("ModelLoader", "weights file")):
        invoke_node(ModelLoaderNode(), {"model": model},
                    {"load_mode": "state_dict", **params})


@pytest.mark.parametrize("params", [{"path": ""}, {"path": "  "}, {}],
                         ids=["empty", "blank", "left out"])
def test_checkpoint_loader_with_no_file_selected_says_so(params):
    from app.core.graph_engine import invoke_node
    from app.nodes.io.checkpoint_node import CheckpointLoaderNode

    model, optimizer = _model_and_optimizer()
    with pytest.raises(
            ValueError, match=_no_file_message("CheckpointLoader", "checkpoint file")):
        invoke_node(CheckpointLoaderNode(), {"model": model, "optimizer": optimizer},
                    params)
