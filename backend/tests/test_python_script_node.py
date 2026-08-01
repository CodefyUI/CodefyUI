"""Tests for PythonScriptNode -- in-canvas Python with an AST-gated policy.

Covers the four things that make this node different from every other
builtin: the Tier-0 import policy, the ``run(inputs, params)`` contract, the
line numbers a failing script reports, and the fact that the code is an
ordinary param (so the execution cache sees an edit for free).
"""

from __future__ import annotations

import pytest

from app.core.cache import ExecutionCache
from app.core.graph_engine import validate_graph
from app.core.node_base import DataType, ParamType
from app.core.plugin_validator import PluginValidationError
from app.core.script_policy import (
    ESCAPE_HATCH_HINT,
    SCRIPT_FILENAME,
    TIER0_MODULES,
    validate_script_source,
)
from app.nodes.utility.python_script_node import (
    MAX_PORTS,
    PythonScriptNode,
    resolve_port_count,
    resolve_port_types,
)


def _run(code: str, inputs: dict | None = None, **params):
    params = {"code": code, **params}
    return PythonScriptNode().execute(inputs or {}, params)


# ── Node metadata / schema ───────────────────────────────────────────────


def test_node_metadata():
    assert PythonScriptNode.NODE_NAME == "PythonScript"
    assert PythonScriptNode.CATEGORY == "Utility"
    # The code is a plain param, so ExecutionCache keys it for free.
    assert PythonScriptNode.cacheable is True


def test_static_schema_matches_default_params():
    """The palette template must show what a freshly-dragged node has."""
    defaults = {p.name: p.default for p in PythonScriptNode.define_params()}
    static_in = [p.name for p in PythonScriptNode.define_inputs()]
    static_out = [p.name for p in PythonScriptNode.define_outputs()]
    assert static_in == [p.name for p in PythonScriptNode.define_inputs_dynamic(defaults)]
    assert static_out == [p.name for p in PythonScriptNode.define_outputs_dynamic(defaults)]


def test_code_param_is_declared_as_code_type_with_a_runnable_default():
    params = {p.name: p for p in PythonScriptNode.define_params()}
    assert params["code"].param_type == ParamType.CODE
    # The shipped template runs as-is against an unconnected node.
    result = PythonScriptNode().execute({}, {"code": params["code"].default})
    assert "out1" in result


def test_port_count_params_are_bounded():
    params = {p.name: p for p in PythonScriptNode.define_params()}
    for name in ("input_ports", "output_ports"):
        assert params[name].param_type == ParamType.INT
        assert params[name].min_value == 1
        assert params[name].max_value == MAX_PORTS


# ── Dynamic ports ────────────────────────────────────────────────────────


def test_dynamic_inputs_follow_input_ports_param():
    names = [p.name for p in PythonScriptNode.define_inputs_dynamic({"input_ports": 3})]
    assert names == ["in1", "in2", "in3"]


def test_dynamic_outputs_follow_output_ports_param():
    names = [p.name for p in PythonScriptNode.define_outputs_dynamic({"output_ports": 2})]
    assert names == ["out1", "out2"]


def test_dynamic_ports_tolerate_none_and_garbage():
    for params in (None, {}, {"input_ports": "many", "output_ports": None}):
        assert [p.name for p in PythonScriptNode.define_inputs_dynamic(params)] == ["in1"]
        assert [p.name for p in PythonScriptNode.define_outputs_dynamic(params)] == ["out1"]


def test_port_count_is_clamped_to_the_declared_range():
    assert resolve_port_count({"input_ports": 0}, "input_ports") == 1
    assert resolve_port_count({"input_ports": 99}, "input_ports") == MAX_PORTS


def test_input_ports_are_optional_so_a_partly_wired_script_still_validates():
    # A script decides for itself which of its ports it needs; requiring all
    # of them would make the port-count knob a trap.
    assert all(p.optional for p in PythonScriptNode.define_inputs_dynamic({"input_ports": 4}))


def test_per_port_data_types_come_from_the_types_param():
    ports = PythonScriptNode.define_inputs_dynamic(
        {"input_ports": 3, "input_types": "TENSOR,STRING,SCALAR"}
    )
    assert [p.data_type for p in ports] == [
        DataType.TENSOR,
        DataType.STRING,
        DataType.SCALAR,
    ]


def test_port_types_default_tensor_in_any_out():
    assert PythonScriptNode.define_inputs_dynamic({})[0].data_type == DataType.TENSOR
    assert PythonScriptNode.define_outputs_dynamic({})[0].data_type == DataType.ANY


def test_short_type_list_repeats_its_last_entry():
    """Bumping the port count must not leave ports untyped."""
    types = resolve_port_types("TENSOR,STRING", 4, DataType.ANY)
    assert types == [DataType.TENSOR, DataType.STRING, DataType.STRING, DataType.STRING]


def test_unknown_type_names_fall_back_to_the_default():
    assert resolve_port_types("NONSENSE", 1, DataType.ANY) == [DataType.ANY]
    assert resolve_port_types("", 2, DataType.TENSOR) == [DataType.TENSOR] * 2


def test_trigger_is_never_a_selectable_port_type():
    """TRIGGER only connects to TRIGGER; letting a script declare it would
    produce a data port nothing can legally feed."""
    assert resolve_port_types("TRIGGER", 1, DataType.ANY) == [DataType.ANY]


# ── The run(inputs, params) contract ─────────────────────────────────────


def test_dict_return_maps_keys_to_output_ports():
    result = _run(
        "def run(inputs, params):\n"
        "    return {'out1': inputs['in1'] + 1, 'out2': 'two'}\n",
        {"in1": 41},
        output_ports=2,
    )
    assert result["out1"] == 42
    assert result["out2"] == "two"


def test_non_dict_return_maps_to_out1():
    assert _run("def run(inputs, params):\n    return 7\n")["out1"] == 7


def test_none_return_maps_to_out1():
    result = _run("def run(inputs, params):\n    return None\n")
    assert result["out1"] is None


def test_params_are_visible_to_the_script():
    result = _run(
        "def run(inputs, params):\n    return params['output_ports']\n",
        output_ports=3,
    )
    assert result["out1"] == 3


def test_script_cannot_mutate_the_node_params():
    """The engine reuses the params dict for the cache key -- hand out a copy."""
    params = {"code": "def run(inputs, params):\n    params['code'] = 'gone'\n    return 1\n"}
    PythonScriptNode().execute({}, params)
    assert params["code"].startswith("def run")


def test_unconnected_ports_are_simply_absent_from_inputs():
    result = _run(
        "def run(inputs, params):\n    return sorted(inputs)\n",
        {"in1": 1},
        input_ports=3,
    )
    assert result["out1"] == ["in1"]


def test_undeclared_output_keys_are_dropped_and_reported():
    result = _run(
        "def run(inputs, params):\n    return {'out1': 1, 'debug': 2}\n",
    )
    assert result["out1"] == 1
    assert "debug" not in result
    assert "debug" in result["__log__"]


def test_missing_run_function_is_a_clear_error():
    with pytest.raises(RuntimeError, match="def run"):
        _run("x = 1\n")


def test_run_must_be_callable():
    with pytest.raises(RuntimeError, match="def run"):
        _run("run = 5\n")


# ── Namespace / stdout ───────────────────────────────────────────────────


def test_allowlisted_modules_are_pre_bound_in_the_namespace():
    result = _run("def run(inputs, params):\n    return math.floor(2.7)\n")
    assert result["out1"] == 2


def test_allowlisted_modules_can_also_be_imported_with_an_alias():
    result = _run(
        "import numpy as np\n\n"
        "def run(inputs, params):\n"
        "    return float(np.mean([1.0, 3.0]))\n"
    )
    assert result["out1"] == 2.0


def test_print_output_is_captured_into_the_execution_log():
    result = _run("def run(inputs, params):\n    print('hello', 1)\n    return 0\n")
    assert result["__log__"].splitlines()[0] == "hello 1"


def test_print_is_captured_at_import_time_too():
    result = _run("print('top level')\n\ndef run(inputs, params):\n    return 0\n")
    assert "top level" in result["__log__"]


def test_a_silent_script_reports_no_log_entry():
    result = _run("def run(inputs, params):\n    return 0\n")
    assert "__log__" not in result


def test_captured_output_is_truncated_rather_than_flooding_the_log():
    result = _run(
        "def run(inputs, params):\n"
        "    for i in range(20000):\n"
        "        print('x' * 40)\n"
        "    return 0\n"
    )
    assert len(result["__log__"]) < 200_000
    assert "truncated" in result["__log__"]


def test_builtins_are_restricted_at_runtime_too():
    """Defense in depth: the AST gate rejects these at save time, so runtime
    lookups only matter if a graph reaches the engine some other way."""
    with pytest.raises(RuntimeError) as exc:
        PythonScriptNode()._invoke(  # noqa: SLF001 - probing the runtime gate
            "def run(inputs, params):\n    return eval('1+1')\n", {}, {}
        )
    assert "eval" in str(exc.value)


def test_torch_is_available_for_tensor_statistics():
    torch = pytest.importorskip("torch")
    x = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    result = _run(
        "def run(inputs, params):\n"
        "    x = inputs['in1']\n"
        "    flat = x.reshape(x.shape[0], x.shape[1], -1)\n"
        "    return {'out1': flat.mean(dim=(0, 2))}\n",
        {"in1": x},
    )
    assert result["out1"].shape == (3,)


# ── Import policy (Tier 0 allowlist) ─────────────────────────────────────


@pytest.mark.parametrize("module", sorted(TIER0_MODULES))
def test_allowlisted_imports_are_accepted(module):
    validate_script_source(f"import {module}\n\ndef run(inputs, params):\n    return 1\n")


@pytest.mark.parametrize(
    "statement",
    [
        "import requests",
        "import os",
        "import sys",
        "import socket",
        "import pathlib",
        "import subprocess",
        "import pandas",  # not dangerous, just not on the allowlist
        "from os import path",
        "from urllib.request import urlopen",
        "import os.path",
        "from . import sibling",
        "from .. import parent",
    ],
)
def test_non_allowlisted_imports_are_rejected(statement):
    with pytest.raises(PluginValidationError) as exc:
        validate_script_source(f"{statement}\n\ndef run(inputs, params):\n    return 1\n")
    assert "not allowed" in str(exc.value)


def test_rejection_points_at_the_custom_node_and_plugin_escape_hatches():
    with pytest.raises(PluginValidationError) as exc:
        validate_script_source("import requests\n")
    message = str(exc.value)
    assert ESCAPE_HATCH_HINT in message
    assert "custom node" in message.lower()
    assert "plugin" in message.lower()


def test_submodules_of_allowlisted_packages_are_importable():
    validate_script_source("import torch.nn.functional as F\nfrom collections import Counter\n")


@pytest.mark.parametrize(
    "snippet",
    [
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', '<s>', 'eval')",
        "__import__('os')",
        "open('/etc/passwd')",
        "globals()",
        "breakpoint()",
        "().__class__.__bases__[0].__subclasses__()",
        "run.__globals__['__builtins__']",
        "getattr(__builtins__, 'exec')",
        "object.__subclasses__()",
        "torch.load('x.pt')",
    ],
)
def test_escape_primitives_are_rejected(snippet):
    with pytest.raises(PluginValidationError):
        validate_script_source(f"def run(inputs, params):\n    return {snippet}\n")


def test_dunder_attribute_access_is_rejected():
    with pytest.raises(PluginValidationError, match="__class__"):
        validate_script_source("def run(inputs, params):\n    return inputs.__class__\n")


def test_syntax_errors_report_the_offending_line():
    with pytest.raises(PluginValidationError) as exc:
        validate_script_source("def run(inputs, params)\n    return 1\n")
    assert "line 1" in str(exc.value)


def test_plain_statistics_scripts_pass_the_gate():
    validate_script_source(
        "import statistics\n\n"
        "def run(inputs, params):\n"
        "    values = list(inputs['in1'])\n"
        "    return {'out1': statistics.mean(values), 'out2': statistics.pstdev(values)}\n"
    )


def test_execution_rejects_a_policy_violation_before_running_anything():
    with pytest.raises(PluginValidationError):
        _run("import requests\n\ndef run(inputs, params):\n    return 1\n")


def test_policy_violation_never_executes_module_level_code(tmp_path):
    marker = tmp_path / "written.txt"
    with pytest.raises(PluginValidationError):
        _run(
            f"open({str(marker)!r}, 'w').write('x')\n\n"
            "def run(inputs, params):\n    return 1\n"
        )
    assert not marker.exists()


# ── Failure reporting ────────────────────────────────────────────────────


def test_runtime_error_reports_the_line_inside_the_script():
    code = (
        "def run(inputs, params):\n"
        "    a = 1\n"
        "    b = 0\n"
        "    return a / b\n"
    )
    with pytest.raises(RuntimeError) as exc:
        _run(code)
    message = str(exc.value)
    assert "line 4" in message
    assert "ZeroDivisionError" in message
    assert "division by zero" in message


def test_error_line_points_at_the_deepest_script_frame():
    code = (
        "def helper(x):\n"
        "    return x['missing']\n"
        "\n"
        "def run(inputs, params):\n"
        "    return helper({})\n"
    )
    with pytest.raises(RuntimeError) as exc:
        _run(code)
    assert "line 2" in str(exc.value)


def test_error_inside_a_library_call_points_at_the_calling_line():
    code = (
        "import statistics\n"
        "\n"
        "def run(inputs, params):\n"
        "    return statistics.mean([])\n"
    )
    with pytest.raises(RuntimeError) as exc:
        _run(code)
    assert "line 4" in str(exc.value)


def test_module_level_failure_is_reported_with_its_line():
    with pytest.raises(RuntimeError) as exc:
        _run("x = 1\ny = x['nope']\n\ndef run(inputs, params):\n    return 1\n")
    assert "line 2" in str(exc.value)


def test_the_script_filename_shows_up_in_reports_not_a_temp_path():
    with pytest.raises(RuntimeError) as exc:
        _run("def run(inputs, params):\n    raise ValueError('boom')\n")
    assert SCRIPT_FILENAME in str(exc.value) or "PythonScript" in str(exc.value)
    assert "boom" in str(exc.value)


def test_captured_output_survives_a_failure(caplog):
    """Whatever the script printed before it died is still worth seeing."""
    with pytest.raises(RuntimeError) as exc:
        _run("def run(inputs, params):\n    print('before')\n    raise ValueError('x')\n")
    assert "before" in str(exc.value)


# ── Cache participation ──────────────────────────────────────────────────


def test_cache_key_changes_when_the_code_changes():
    base = "def run(inputs, params):\n    return 1\n"
    edited = "def run(inputs, params):\n    return 2\n"
    key_a = ExecutionCache.compute_key("PythonScript", {"code": base}, [])
    key_b = ExecutionCache.compute_key("PythonScript", {"code": edited}, [])
    assert key_a != key_b


def test_cache_key_is_stable_for_identical_code():
    code = "def run(inputs, params):\n    return 1\n"
    assert ExecutionCache.compute_key(
        "PythonScript", {"code": code, "output_ports": 1}, []
    ) == ExecutionCache.compute_key(
        "PythonScript", {"output_ports": 1, "code": code}, []
    )


# ── Validator integration ────────────────────────────────────────────────


def _script_graph(source_handle: str, target_handle: str, **params) -> tuple[list, list]:
    nodes = [
        {"id": "s", "type": "Start", "data": {"params": {}}},
        {"id": "t", "type": "TensorInput", "data": {"params": {"shape": "6", "value_mode": "zeros"}}},
        {"id": "py", "type": "PythonScript", "data": {"params": params}},
        {"id": "p", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        {"id": "e1", "source": "s", "target": "t", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e2", "source": "t", "target": "py", "sourceHandle": "tensor", "targetHandle": target_handle},
        {"id": "e3", "source": "py", "target": "p", "sourceHandle": source_handle, "targetHandle": "value"},
    ]
    return nodes, edges


def test_validator_accepts_ports_within_the_configured_count():
    nodes, edges = _script_graph("out2", "in2", input_ports=2, output_ports=2)
    errors = validate_graph(nodes, edges)
    assert not errors, errors


def test_validator_rejects_an_output_port_above_the_configured_count():
    nodes, edges = _script_graph("out3", "in1", input_ports=1, output_ports=2)
    errors = validate_graph(nodes, edges)
    assert any("Invalid output port 'out3'" in e for e in errors), errors


def test_validator_rejects_an_input_port_above_the_configured_count():
    nodes, edges = _script_graph("out1", "in3", input_ports=2, output_ports=1)
    errors = validate_graph(nodes, edges)
    assert any("Invalid input port 'in3'" in e for e in errors), errors


def test_validator_type_checks_the_declared_input_type():
    nodes, edges = _script_graph(
        "out1", "in1", input_ports=1, output_ports=1, input_types="STRING"
    )
    errors = validate_graph(nodes, edges)
    assert any("Type mismatch" in e for e in errors), errors


def test_a_typed_tensor_output_feeds_a_tensor_consumer():
    """Acceptance: the script's out1 declared TENSOR wires into a tensor input."""
    nodes = [
        {"id": "s", "type": "Start", "data": {"params": {}}},
        {"id": "t", "type": "TensorInput", "data": {"params": {"shape": "2,3", "value_mode": "zeros"}}},
        {
            "id": "py",
            "type": "PythonScript",
            "data": {"params": {"output_types": "TENSOR", "code": "def run(i, p):\n    return 1\n"}},
        },
        {"id": "f", "type": "Flatten", "data": {"params": {}}},
    ]
    edges = [
        {"id": "e1", "source": "s", "target": "t", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e2", "source": "t", "target": "py", "sourceHandle": "tensor", "targetHandle": "in1"},
        {"id": "e3", "source": "py", "target": "f", "sourceHandle": "out1", "targetHandle": "tensor"},
    ]
    assert validate_graph(nodes, edges) == []
