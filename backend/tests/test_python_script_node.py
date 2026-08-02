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
    SCRIPT_PROXY_DENIED_ATTRS,
    TIER0_GATEWAY_MODULE_ATTRS,
    TIER0_MODULES,
    TIER0_RCE_ATTR_LEAVES,
    TIER0_SAFE_LOAD_RECEIVERS,
    ScriptPolicyError,
    validate_script_source,
)
from app.nodes.utility.python_script_node import (
    MAX_PORTS,
    PythonScriptNode,
    _OutputCapture,
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


# ── Escape-matrix regressions (review round 1) ───────────────────────────
#
# Every probe below was a WORKING escape (or a false rejection) against the
# first cut of this node, verified end-to-end through ``execute``. They are
# kept as executable probes rather than validator unit tests so a future
# refactor cannot pass by loosening only the layer they happen to hit.


def _escapes(code: str) -> bool:
    """True when *code* runs to completion -- i.e. the gate let it through."""
    try:
        PythonScriptNode().execute({}, {"code": code})
        return True
    except (PluginValidationError, RuntimeError):
        return False


@pytest.mark.parametrize(
    ("label", "code"),
    [
        (
            "module machinery by bare name",
            "def run(inputs, params):\n"
            "    m = __loader__.load_module('nt')\n"
            "    return getattr(m, 'getcwd')()\n",
        ),
        (
            "builtins aliased then subscripted",
            "def run(inputs, params):\n"
            "    b = __builtins__\n"
            "    return b['__loader__'].load_module('nt').getcwd()\n",
        ),
        (
            "__spec__ by bare name",
            "def run(inputs, params):\n    return __spec__.loader\n",
        ),
        (
            "site builtin left in the namespace",
            "def run(inputs, params):\n    return copyright._Printer__filenames\n",
        ),
        (
            "numpy writes a file",
            "import numpy\n\ndef run(inputs, params):\n"
            "    numpy.savetxt('probe.txt', numpy.zeros(3))\n    return 1\n",
        ),
        (
            "numpy reads a file",
            "import numpy\n\ndef run(inputs, params):\n"
            "    return numpy.fromfile('probe.txt')\n",
        ),
        (
            "torch.hub downloads and executes",
            "import torch\n\ndef run(inputs, params):\n    return torch.hub.load\n",
        ),
        (
            "cpp_extension compiles and executes",
            "import torch\n\ndef run(inputs, params):\n"
            "    return torch.utils.cpp_extension.load_inline\n",
        ),
        (
            "denied door imported as a bare name",
            "from torch.utils.cpp_extension import load_inline\n\n"
            "def run(inputs, params):\n    return load_inline\n",
        ),
        (
            "denied numpy leaf imported by name",
            "from numpy import savetxt\n\ndef run(inputs, params):\n    return savetxt\n",
        ),
        (
            "denied submodule imported directly",
            "import torch.hub\n\ndef run(inputs, params):\n    return 1\n",
        ),
        (
            "literal getattr around the attribute rule",
            "import numpy\n\ndef run(inputs, params):\n"
            "    return getattr(numpy, 'savetxt')\n",
        ),
        (
            "aliased torch.load",
            "import torch as t\n\ndef run(inputs, params):\n    return t.load('x.pt')\n",
        ),
        (
            "torch.load down an attribute chain",
            "import torch\n\ndef run(inputs, params):\n"
            "    return torch.serialization.load('x.pt')\n",
        ),
        (
            "class walk",
            "def run(inputs, params):\n"
            "    return ().__class__.__bases__[0].__subclasses__()\n",
        ),
        (
            "open at module level",
            "fh = open('x')\n\ndef run(inputs, params):\n    return 1\n",
        ),
        # ── review round 2: the frame walk ───────────────────────────
        #
        # A caught exception carries a traceback, a traceback carries the
        # frame it was raised in, and that frame's ``f_back`` is ``_invoke``
        # -- whose module globals hold ``importlib`` and ``builtins``. The
        # restricted ``__builtins__`` the script executes with is beside the
        # point once someone else's globals are in hand. All three of these
        # returned live host objects: ``os.system``, ``os.getcwd()`` (the
        # backend path) and file bytes read off disk.
        (
            "traceback walk to the caller's globals",
            "def run(inputs, params):\n"
            "    try:\n"
            "        raise ValueError()\n"
            "    except ValueError as e:\n"
            "        g = e.__traceback__.tb_frame.f_back.f_globals\n"
            "        return getattr(g['importlib'].import_module('os'), 'system')\n",
        ),
        (
            "traceback walk calling os.getcwd()",
            "def run(inputs, params):\n"
            "    try:\n"
            "        raise ValueError()\n"
            "    except ValueError as e:\n"
            "        g = e.__traceback__.tb_frame.f_back.f_globals\n"
            "        return getattr(g['importlib'].import_module('os'), 'getcwd')()\n",
        ),
        (
            "traceback walk to builtins.open",
            "def run(inputs, params):\n"
            "    try:\n"
            "        raise ValueError()\n"
            "    except ValueError as e:\n"
            "        g = e.__traceback__.tb_frame.f_back.f_globals\n"
            "        return getattr(g['builtins'], 'open')('pyproject.toml').read()\n",
        ),
        (
            "generator frame instead of a traceback",
            "def run(inputs, params):\n"
            "    def gen():\n"
            "        yield 1\n"
            "    return sorted(gen().gi_frame.f_globals)\n",
        ),
        (
            "generator code object",
            "def run(inputs, params):\n"
            "    def gen():\n"
            "        yield 1\n"
            "    return str(gen().gi_code)\n",
        ),
        (
            "frame attributes spelled with a literal getattr",
            "def run(inputs, params):\n"
            "    try:\n"
            "        raise ValueError()\n"
            "    except ValueError as e:\n"
            "        return getattr(getattr(e, '__traceback__'), 'tb_frame')\n",
        ),
        # ── review round 2: an allowlisted module as a gateway ───────
        #
        # The import allowlist means nothing if an allowed module hands the
        # blocked one over by name. Every one of these returned the real
        # host module; ``torch.os.getcwd()`` returned the backend path.
        (
            "os through torch",
            "import torch\n\ndef run(inputs, params):\n    return torch.os.getcwd()\n",
        ),
        (
            "sys through torch",
            "import torch\n\ndef run(inputs, params):\n    return torch.sys.executable\n",
        ),
        (
            "pickle through torch.serialization",
            "import torch\n\ndef run(inputs, params):\n"
            "    return str(torch.serialization.pickle)\n",
        ),
        (
            "sys through json",
            "import json\n\ndef run(inputs, params):\n    return str(json.codecs.sys)\n",
        ),
        (
            "subprocess through numpy.f2py",
            "import numpy\n\ndef run(inputs, params):\n"
            "    return str(numpy.f2py.subprocess)\n",
        ),
        (
            "importlib through torch.cuda",
            "import torch\n\ndef run(inputs, params):\n"
            "    return torch.cuda.importlib.import_module('os').getcwd()\n",
        ),
        (
            "multiprocessing under torch's own alias",
            "import torch\n\ndef run(inputs, params):\n"
            "    return str(torch.cuda.tunable.mp)\n",
        ),
        (
            "gateway reached with a literal getattr",
            "import torch\n\ndef run(inputs, params):\n"
            "    return getattr(torch, 'os').getcwd()\n",
        ),
        (
            "gateway reached through a local alias",
            "import torch\n\ndef run(inputs, params):\n"
            "    x = torch\n    return x.os.getcwd()\n",
        ),
        (
            "gateway reached through an unresolvable receiver",
            "import torch\n\ndef run(inputs, params):\n"
            "    return (lambda: torch)().os.getcwd()\n",
        ),
    ],
)
def test_escape_probe_fails_closed(label, code):
    assert not _escapes(code), f"escape still open: {label}"


def _gate_rejects(code: str) -> bool:
    """True when the GATE refuses *code* -- not merely when running it fails.

    ``_escapes`` above counts any ``RuntimeError`` as a block, which is right
    for a probe whose payload cannot exist without escaping. It is wrong for
    the ``.load`` probes: ``torch.load('x.pt')`` on a missing file raises
    ``FileNotFoundError``, so the lenient helper reports "blocked" for code
    that sailed through the gate and called the real pickle loader. Every
    receiver-laundering bypass below was invisible for exactly that reason.
    """
    try:
        PythonScriptNode().execute({}, {"code": code})
        return False
    except PluginValidationError:
        return True
    except Exception:  # noqa: BLE001 - ran, therefore not gated
        return False


@pytest.mark.parametrize(
    ("label", "code"),
    [
        (
            "receiver laundered through a local name",
            "import torch\n\ndef run(inputs, params):\n"
            "    b = torch\n    return b.load('x.pt')\n",
        ),
        (
            "receiver laundered through a literal getattr",
            "import torch\n\ndef run(inputs, params):\n"
            "    return getattr(torch, 'load')('x.pt')\n",
        ),
        (
            "receiver laundered through a lambda",
            "import torch\n\ndef run(inputs, params):\n"
            "    return (lambda: torch)().load('x.pt')\n",
        ),
        (
            "receiver laundered through a subscript",
            "import torch\n\ndef run(inputs, params):\n    return (torch,)[0].load('x.pt')\n",
        ),
        (
            "numpy laundered through a local name",
            "import numpy\n\ndef run(inputs, params):\n"
            "    n = numpy\n    return n.load('x.npy')\n",
        ),
        (
            "pre-bound module, no import statement at all",
            "def run(inputs, params):\n    t = torch\n    return t.load('x.pt')\n",
        ),
        (
            "alias of an alias",
            "import torch as t\n\ndef run(inputs, params):\n"
            "    u = t\n    return u.loads(b'')\n",
        ),
        # The spelled-out forms, kept here too so the strict helper covers
        # the whole rule rather than only the shapes that were broken.
        (
            "spelled-out torch.load",
            "import torch\n\ndef run(inputs, params):\n    return torch.load('x.pt')\n",
        ),
        (
            "torch.load down an attribute chain",
            "import torch\n\ndef run(inputs, params):\n"
            "    return torch.serialization.load('x.pt')\n",
        ),
    ],
)
def test_the_gate_itself_rejects_a_laundered_load(label, code):
    """Every one of these reached the real ``torch.load`` / ``numpy.load``.

    They were reported as blocked only because the file they named did not
    exist -- the guard resolved import aliases and nothing else, so any
    receiver it could not tie back to an import walked through.
    """
    assert _gate_rejects(code), f"gate still lets this through: {label}"


def test_a_laundered_load_names_what_it_objected_to():
    with pytest.raises(PluginValidationError) as exc:
        validate_script_source(
            "import torch\n\ndef run(inputs, params):\n    b = torch\n    return b.load('x')\n"
        )
    message = str(exc.value)
    assert "'b' (torch)" in message
    assert "json" in message  # says what IS permitted, not only what is not


def test_json_is_the_only_safe_load_receiver():
    """Guard the constant itself: widening it is a security decision, not a
    tidy-up, and it should be visible in a diff."""
    assert TIER0_SAFE_LOAD_RECEIVERS == ("json",)


def test_an_unresolvable_receiver_fails_closed_rather_than_open():
    """The old rule returned early on a receiver it could not resolve, which
    is precisely what the lambda and subscript forms manufacture."""
    assert _gate_rejects(
        "import torch\n\ndef run(inputs, params):\n    return [torch][0].load('x')\n"
    )


def test_frame_attributes_are_refused_whatever_the_receiver():
    from app.core.plugin_validator import frame_introspection_attrs

    for attr in sorted(frame_introspection_attrs()):
        with pytest.raises(PluginValidationError, match="frame attribute"):
            validate_script_source(f"def run(inputs, params):\n    return inputs.{attr}\n")
        with pytest.raises(PluginValidationError, match="frame attribute"):
            validate_script_source(
                f"def run(inputs, params):\n    return getattr(inputs, {attr!r})\n"
            )


def test_traceback_is_a_forbidden_dunder():
    """``__traceback__`` is the doorway to the whole frame chain and was the
    one dunder the set was missing."""
    from app.core.plugin_validator import forbidden_dunders

    assert "__traceback__" in forbidden_dunders()
    with pytest.raises(PluginValidationError, match="__traceback__"):
        validate_script_source(
            "def run(inputs, params):\n"
            "    try:\n"
            "        raise ValueError()\n"
            "    except ValueError as e:\n"
            "        return e.__traceback__\n"
        )


def test_blocked_modules_are_refused_as_attributes_too():
    """An allowlisted module handing over a blocked one by name defeats the
    import allowlist without writing an import."""
    for name in ("os", "sys", "subprocess", "importlib", "pickle", "ctypes", "socket"):
        assert name in TIER0_GATEWAY_MODULE_ATTRS
        with pytest.raises(PluginValidationError):
            validate_script_source(
                f"import torch\n\ndef run(inputs, params):\n    return torch.{name}\n"
            )


def test_the_gateway_rule_leaves_torch_signal_alone():
    """``torch.signal`` is torch's own DSP namespace, not the stdlib module
    of that name -- a name-keyed rule has to say so explicitly."""
    assert "signal" not in TIER0_GATEWAY_MODULE_ATTRS
    # The gate claim holds whatever torch is installed; ``torch.signal``
    # itself only exists from 2.1, and pyproject's floor is 2.0.
    validate_script_source(
        "import torch\n\ndef run(inputs, params):\n    return torch.signal.windows\n"
    )
    import torch

    if not hasattr(torch, "signal"):  # pragma: no cover - torch < 2.1
        pytest.skip("torch.signal arrived in 2.1")
    result = _run(
        "import torch\n\ndef run(inputs, params):\n"
        "    return float(torch.signal.windows.hann(8).sum())\n"
    )
    assert result["out1"] > 0


def test_the_gateway_rule_leaves_ordinary_attribute_names_alone():
    """``code`` is exempt as a word: this node's own parameter is called
    ``code``, and the module of that name is unreachable regardless."""
    assert "code" not in TIER0_GATEWAY_MODULE_ATTRS
    result = _run(
        "class Result:\n"
        "    def __init__(self):\n"
        "        self.code = 7\n"
        "\n"
        "\n"
        "def run(inputs, params):\n"
        "    return Result().code\n"
    )
    assert result["out1"] == 7


def test_a_local_named_like_a_frame_attribute_is_still_legal():
    """The frame rule is an ATTRIBUTE rule; binding the same word as a local
    reaches no frame and must not be swept up."""
    result = _run(
        "def run(inputs, params):\n    f_code = 3\n    gi_frame = 4\n    return f_code + gi_frame\n"
    )
    assert result["out1"] == 7


# ── The shared walker: what tier 0 tightens, and what it must not ────────
#
# ``validate_python_source`` also gates uploaded custom nodes and installed
# plugin packs, which are checked as a BLOCKLIST because the user chose the
# file. Tier 0's extra rules ride on arguments, so they must not leak into
# that path -- and the two rules that DO apply to both (frame walking, a
# laundered pickle receiver) are deliberate, so they are asserted rather
# than left to be discovered as a regression.


def _plugin_mode(code: str) -> None:
    """Validate as an installed plugin: no tier-0 arguments at all."""
    from app.core.plugin_validator import validate_python_source

    validate_python_source(code, "plugin.py")


def test_a_plugin_may_still_load_from_a_receiver_the_walker_cannot_resolve():
    """The strict receiver rule is tier-0 only. A plugin calling
    ``self.backend.load(path)`` is ordinary code, and the user installed it."""
    _plugin_mode("def go(things):\n    return things[0].load('x')\n")
    _plugin_mode("class C:\n    def load(self, p):\n        return p\n")


def test_a_plugin_may_still_name_a_blocked_module_as_an_attribute():
    """The gateway rule is tier-0 only; a plugin that legitimately imports
    ``os`` reaches ``os.path`` through attributes all day."""
    _plugin_mode("import json\n\ndef go(cfg):\n    return cfg.subprocess\n")


def test_a_plugin_may_still_call_json_load():
    _plugin_mode("import json\n\ndef go(fh):\n    return json.load(fh)\n")


def test_the_frame_walk_is_closed_for_plugins_too():
    """Deliberate widening: reading another frame's globals is a sandbox
    escape wherever it is written, and no shipped plugin or custom node in
    this repo touches these names."""
    for snippet in (
        "e.__traceback__",
        "gen().gi_frame",
        "frame.f_globals",
        "tb.tb_frame",
    ):
        with pytest.raises(PluginValidationError):
            _plugin_mode(f"def go(e, gen, frame, tb):\n    return {snippet}\n")


def test_a_laundered_pickle_receiver_is_closed_for_plugins_too():
    """``getattr(torch, 'load')(p)`` is the pickle call with the attribute
    access spelled out of existence; the blocklist should see through it."""
    with pytest.raises(PluginValidationError):
        _plugin_mode("import torch\n\ndef go(p):\n    return getattr(torch, 'load')(p)\n")
    with pytest.raises(PluginValidationError):
        _plugin_mode("import torch\n\ndef go(p):\n    b = torch\n    return b.load(p)\n")


def test_ordinary_torch_and_numpy_namespaces_still_resolve():
    """The gateway rule adds 25 denied attribute names; assert the library
    surface a statistics script actually uses survived it."""
    result = _run(
        "import numpy as np\n"
        "import torch\n"
        "\n"
        "\n"
        "def run(inputs, params):\n"
        "    a = float(np.linalg.norm(np.ones(4)))\n"
        "    b = float(torch.nn.functional.relu(torch.ones(3)).sum())\n"
        "    return a + b\n"
    )
    assert result["out1"] == pytest.approx(5.0)


def test_the_namespace_is_an_allowlist_not_a_blocklist():
    """The ``__loader__`` escape existed because the namespace was built by
    subtraction from ``vars(builtins)``. Assert the shape, not just the
    symptom: a future CPython builtin must not appear here by default."""
    exposed = PythonScriptNode._script_builtins(_OutputCapture())  # noqa: SLF001
    for leaked in ("__loader__", "__spec__", "__package__", "copyright", "open", "eval"):
        assert leaked not in exposed
    for needed in ("len", "range", "sorted", "isinstance", "getattr", "ValueError"):
        assert needed in exposed
    # BaseException-only types stay out: a script must not raise something
    # the engine's ``except Exception`` cannot catch.
    assert "SystemExit" not in exposed
    assert "KeyboardInterrupt" not in exposed


def test_ordinary_python_still_works_inside_the_allowlisted_namespace():
    result = _run(
        "import numpy as np\n"
        "\n"
        "\n"
        "class Summary:\n"
        "    def __init__(self, values):\n"
        "        self.values = values\n"
        "\n"
        "    def mean(self):\n"
        "        try:\n"
        "            return float(np.mean(self.values))\n"
        "        except ValueError:\n"
        "            return 0.0\n"
        "\n"
        "\n"
        "def run(inputs, params):\n"
        "    return Summary([1.0, 2.0, 3.0]).mean()\n"
    )
    assert result["out1"] == 2.0


def test_json_parsing_is_allowed_because_json_is_tier_zero():
    """The pickle heuristic used to condemn ``.loads`` on ANY receiver, so
    the node told users to write a custom node to parse JSON."""
    result = _run(
        'import json\n\ndef run(inputs, params):\n'
        '    return json.dumps(json.loads(\'{"a": 1}\'))\n'
    )
    assert result["out1"] == '{"a": 1}'


def test_async_run_is_rejected_at_the_gate():
    with pytest.raises(PluginValidationError, match="async def run"):
        _run("async def run(inputs, params):\n    return 1\n")


def test_a_coroutine_run_is_refused_at_runtime_too():
    """The gate cannot see a ``run`` rebound after definition; the port must
    still never carry an un-awaited coroutine."""
    code = (
        "async def _work(inputs, params):\n"
        "    return 1\n"
        "\n"
        "run = _work\n"
    )
    with pytest.raises(RuntimeError, match="coroutine"):
        PythonScriptNode()._invoke(code, {}, {})  # noqa: SLF001


def test_should_stop_is_exposed_so_a_long_loop_can_bail_out():
    from app.core.execution_context import ExecutionContext

    context = ExecutionContext()
    context.cancel()
    result = PythonScriptNode().execute(
        {},
        {
            "code": "def run(inputs, params):\n"
            "    for i in range(1000):\n"
            "        if should_stop():\n"
            "            return i\n"
            "    return -1\n"
        },
        context=context,
    )
    assert result["out1"] == 0


def test_should_stop_answers_false_without_a_context():
    result = _run("def run(inputs, params):\n    return should_stop()\n")
    assert result["out1"] is False


def test_the_policy_message_does_not_promise_files_are_unreachable():
    """numpy can write files; a hint that says otherwise is a promise the
    gate cannot keep, printed exactly when trust is being decided."""
    lowered = ESCAPE_HATCH_HINT.lower()
    assert "libraries" in lowered
    assert "for file, network or process access" not in lowered
    assert "libraries" in PythonScriptNode.DESCRIPTION.lower()
    assert "guardrail, not a sandbox" in PythonScriptNode.DESCRIPTION.lower()


# ═════════════════════════════════════════════════════════════════════════
# Review round 3: the runtime module proxy is the boundary
# ═════════════════════════════════════════════════════════════════════════
#
# Rounds 1 and 2 each closed an escape by adding names to a list, and each
# time the next reviewer found a name that was not on it:
#
#     collections._sys                    -> the real sys, under an alias
#     collections._sys.modules['os']      -> every imported module, by key
#     statistics.random._os.getcwd()      -> os, two private hops away
#     json.codecs.builtins.eval           -> builtins, via an unlisted module
#     f = osmod.system; f(cmd)            -> no Call-keyed rule can fire
#     f = torch.load;   f(path)           -> ditto for the pickle rule
#
# The list was never the problem. The namespace handed the script the host's
# REAL module objects, and a real module hands over whatever it holds. The
# fix is structural: the namespace holds proxies that judge the RESOLVED
# OBJECT -- is this a module, and is it on the list -- so an alias, a
# subscript, a local binding and next year's library reshuffle are the same
# rule. The probes below are driven through ``execute``, and a probe is
# closed whether the GATE or the PROXY refused it: both are locks on the
# same door and the test asserts the door is shut, not which lock held.


def _refused(code: str, inputs: dict | None = None) -> str:
    """``"GATE"``, ``"PROXY"``, ``"NAMEERROR"`` or ``"RAN"`` for *code*."""
    try:
        PythonScriptNode().execute(inputs or {}, {"code": code})
        return "RAN"
    except PluginValidationError:
        return "GATE"
    except ScriptPolicyError:  # pragma: no cover - wrapped by the node
        return "PROXY"
    except RuntimeError as exc:
        text = str(exc)
        if "ScriptPolicyError" in text:
            return "PROXY"
        if "NameError" in text:
            return "NAMEERROR"
        return "RAN"


ROUND_THREE_PROBES = [
    (
        "the real sys under a private alias",
        "def run(inputs, params):\n    return str(collections._sys)\n",
    ),
    (
        "sys.modules subscript reaches every imported module",
        "def run(inputs, params):\n"
        "    s = collections._sys\n"
        "    osmod = s.modules['os']\n"
        "    return osmod.getcwd()\n",
    ),
    (
        "os.system bound to a local, so no Call-keyed rule fires",
        "def run(inputs, params):\n"
        "    s = collections._sys\n"
        "    osmod = s.modules['os']\n"
        "    f = osmod.system\n"
        "    return f('cmd /c ver')\n",
    ),
    (
        "os two private hops from statistics",
        "def run(inputs, params):\n    return statistics.random._os.getcwd()\n",
    ),
    (
        "subprocess.run through sys.modules",
        "def run(inputs, params):\n"
        "    sp = collections._sys.modules['subprocess']\n"
        "    return str(sp.run(['cmd', '/c', 'ver'], capture_output=True))\n",
    ),
    (
        "builtins.eval through sys.modules",
        "def run(inputs, params):\n"
        "    b = collections._sys.modules['builtins']\n"
        "    f = b.eval\n"
        "    return f('6*7')\n",
    ),
    (
        "builtins through an unlisted module (json.codecs)",
        "def run(inputs, params):\n"
        "    b = json.codecs.builtins\n"
        "    f = b.eval\n"
        "    return f('6*7')\n",
    ),
    (
        "the pickle loader bound to a local instead of called",
        "def run(inputs, params):\n    f = torch.load\n    return str(f)\n",
    ),
    (
        "a private submodule of an allowed module",
        "def run(inputs, params):\n    return str(re._parser)\n",
    ),
    (
        "a private leaf of an allowed module",
        "def run(inputs, params):\n    return str(statistics._sum)\n",
    ),
    (
        "collections.abc as the private hop",
        "def run(inputs, params):\n    return str(collections.abc._sys)\n",
    ),
    (
        "module poisoning by assignment",
        "def run(inputs, params):\n"
        "    def evil(*a, **k):\n        return 'poisoned'\n"
        "    torch.zeros = evil\n"
        "    return 1\n",
    ),
    (
        "module poisoning through setattr",
        "def run(inputs, params):\n"
        "    def evil(*a, **k):\n        return 'poisoned'\n"
        "    setattr(torch, 'zeros', evil)\n"
        "    return 1\n",
    ),
    (
        "deleting a library attribute",
        "def run(inputs, params):\n    del torch.zeros\n    return 1\n",
    ),
    (
        "subscripting a module",
        "def run(inputs, params):\n    m = torch\n    return str(m['os'])\n",
    ),
    (
        "calling a module",
        "def run(inputs, params):\n    m = torch\n    return m()\n",
    ),
    (
        "an import statement binds the proxy too",
        "import collections\n\ndef run(inputs, params):\n"
        "    c = collections\n    return str(c._sys)\n",
    ),
    (
        "torch.load with weights_only=True (tier 0 has no file access)",
        "import torch\n\ndef run(inputs, params):\n"
        "    return torch.load('x.pt', weights_only=True)\n",
    ),
    # ── review round 4: a capability that is not a module ────────────
    #
    # The proxy's module rule asks "may I hand over this MODULE". A class is
    # not a module, so pathlib.Path -- arbitrary file read AND write -- came
    # through four different spellings, and an INSTANCE of it through a
    # fifth. Proven live by writing a file, including into the backend's own
    # source tree, where a later import would execute it.
    (
        "pathlib.Path re-exported by numpy.f2py",
        "def run(inputs, params):\n"
        "    P = numpy.f2py.crackfortran.Path\n"
        "    return P('.').resolve().name\n",
    ),
    (
        "pathlib.Path re-exported by numpy.f2py.f2py2e.rules",
        "def run(inputs, params):\n"
        "    P = numpy.f2py.f2py2e.rules.Path\n"
        "    return P('.').resolve().name\n",
    ),
    (
        "pathlib.Path re-exported by torch.fx",
        "def run(inputs, params):\n"
        "    P = torch.fx.graph_module.Path\n"
        "    return P('.').resolve().name\n",
    ),
    (
        "pathlib.Path re-exported by torch.package",
        "def run(inputs, params):\n"
        "    P = torch.package.package_exporter.Path\n"
        "    return P('.').resolve().name\n",
    ),
    (
        "a Path INSTANCE, not the class",
        "def run(inputs, params):\n"
        "    root = numpy.testing.NUMPY_ROOT\n"
        "    return str(root.parent)\n",
    ),
    (
        "writing a file through a re-exported Path",
        "def run(inputs, params):\n"
        "    P = torch.fx.graph_module.Path\n"
        "    P('round4_probe.txt').write_text('pwned')\n"
        "    return 1\n",
    ),
    (
        "importlib's find_spec re-exported by torch.cuda.amp",
        "def run(inputs, params):\n"
        "    f = torch.cuda.amp.common.find_spec\n"
        "    return str(f('os'))\n",
    ),
    # NOTE: ``numpy.random.bit_generator.RLock`` is deliberately NOT a probe
    # here. It surfaced in the sweep, and closing it is the reason the C
    # implementations of blocked modules count as their wrapper -- but which
    # object numpy re-exports varies by version (``threading.RLock`` here,
    # ``_thread.RLock`` with ``__module__`` unset on CI's numpy), and a lock
    # is not a capability. Asserting it would be asserting something the rule
    # does not guarantee. The rule itself is tested against stable stdlib
    # objects in ``test_the_c_implementation_of_a_blocked_module...``.
    (
        "numpy's own temp-directory helper",
        "def run(inputs, params):\n"
        "    with numpy.testing.tempdir() as d:\n"
        "        return str(d)\n",
    ),
]


@pytest.mark.parametrize(("label", "code"), ROUND_THREE_PROBES)
def test_round_three_probes_fail_closed(label, code):
    """Every one of these ran to completion at ``f4a9185``; six of them were
    full RCE (``os.system``, ``subprocess.run``, ``builtins.eval``)."""
    assert _refused(code) in ("GATE", "PROXY"), f"escape still open: {label}"


def test_the_boundary_is_the_resolved_object_not_the_attribute_name():
    """The point of the whole round, asserted without naming a real escape.

    A module bound onto an allowed one under a name no blocklist contains --
    which is exactly what ``collections._sys`` and ``torch.cuda.tunable.mp``
    are -- must be refused because of WHAT IT IS.
    """
    import math
    import os

    from app.core.script_proxy import module_proxy

    proxy = module_proxy(math)
    try:
        math.perfectlyordinaryname = os  # a name no list will ever hold
        with pytest.raises(ScriptPolicyError, match="not on the Tier-0 list"):
            proxy.perfectlyordinaryname
    finally:
        del math.perfectlyordinaryname


def test_a_submodule_of_an_allowed_package_is_wrapped_not_handed_over():
    from app.core.script_proxy import RestrictedModule, module_proxy

    import numpy

    linalg = module_proxy(numpy).linalg
    assert isinstance(linalg, RestrictedModule)
    assert not isinstance(linalg, type(numpy))


def test_plain_values_are_returned_unwrapped_so_tensor_work_is_unaffected():
    """The performance contract: proxy the module surface, not the data.

    A returned tensor must be a tensor -- wrapping it would put a Python
    ``__getattribute__`` in front of every ``.mean()`` in every script.
    """
    import torch

    from app.core.script_proxy import RestrictedModule, module_proxy

    proxy = module_proxy(torch)
    tensor = proxy.zeros(3)
    assert isinstance(tensor, torch.Tensor)
    assert not isinstance(tensor, RestrictedModule)
    assert not isinstance(proxy.pi, RestrictedModule)


def test_a_missing_attribute_still_raises_attributeerror():
    """``getattr(mod, 'x', default)`` and ``hasattr`` must keep working; only
    a POLICY refusal is a ScriptPolicyError."""
    import math

    from app.core.script_proxy import module_proxy

    proxy = module_proxy(math)
    with pytest.raises(AttributeError):
        proxy.definitely_not_a_math_function
    assert getattr(proxy, "definitely_not_a_math_function", 7) == 7


def test_the_proxy_keeps_its_own_state_out_of_reach():
    """A proxy that stored the real module in an ordinary attribute would
    hand it back to the first script that guessed the name."""
    import math

    from app.core.script_proxy import module_proxy

    proxy = module_proxy(math)
    for name in ("_target", "_label", "_cache", "__dict__", "__class__"):
        with pytest.raises((ScriptPolicyError, AttributeError)):
            getattr(proxy, name)


def test_the_proxy_holds_no_instance_state_at_all():
    """Found while probing this very fix, and live for an afternoon.

    The first cut kept the wrapped module in ``__slots__``. A slot is a
    descriptor on the CLASS, and ``type`` is an allowed builtin, so
    ``type(collections)._target.__get__(collections)`` returned the real
    module -- a proxy that leaked what it wrapped. State now lives off the
    object entirely: there is no descriptor to fetch and no ``__dict__`` to
    read, so ``object.__getattribute__`` has nothing to hand over either.
    """
    import math

    from app.core.script_proxy import RestrictedModule, module_proxy

    proxy = module_proxy(math)
    assert RestrictedModule.__slots__ == ()
    for name in ("_target", "_label", "_cache", "__dict__"):
        with pytest.raises(AttributeError):
            object.__getattribute__(proxy, name)
    # ...and no state on the class either, which is where a slot lives.
    for name in ("_target", "_label", "_cache"):
        with pytest.raises(AttributeError):
            getattr(RestrictedModule, name)
    # The escape as a script would write it, refused for the structural
    # reason rather than by a rule about the word "_target".
    with pytest.raises(RuntimeError) as exc:
        PythonScriptNode().execute(
            {},
            {"code": "def run(inputs, params):\n"
                     "    d = type(collections)._target\n"
                     "    m = d.__get__(collections)\n"
                     "    return m._sys.modules['os'].getcwd()\n"},
        )
    assert "no attribute '_target'" in str(exc.value)


def test_the_namespace_hands_out_proxies_not_modules():
    import types

    from app.core.script_proxy import RestrictedModule, tier0_module_namespace

    namespace = tier0_module_namespace()
    assert set(namespace) <= set(TIER0_MODULES)
    for name, value in namespace.items():
        assert isinstance(value, RestrictedModule), name
        assert not isinstance(value, types.ModuleType), name


def test_no_module_outside_the_allowlist_is_reachable_through_the_proxies():
    """The replacement for round 2's "a depth-5 name walk finds nothing".

    That claim was a statement about the names in one version of numpy and
    torch, and it was false at ``f4a9185``. This one is a statement about the
    RULE: walk the proxy surface following every attribute the policy allows
    and every module that comes back is, by construction, on the list. The
    walk is a check that the construction holds, not the thing that makes it
    true.
    """
    import types

    from app.core.script_proxy import (
        RestrictedModule,
        tier0_module_namespace,
        unwrap,
    )

    allowed = set(TIER0_MODULES)
    seen: set[int] = set()
    leaked: list[str] = []
    visited = 0

    def walk(proxy, path: str, depth: int) -> None:
        nonlocal visited
        if depth > 3:
            return
        target = unwrap(proxy)
        if id(target) in seen:
            return
        seen.add(id(target))
        visited += 1
        for name in dir(target):
            try:
                value = getattr(proxy, name)
            except Exception:  # noqa: BLE001 - refused or simply absent
                continue
            if isinstance(value, RestrictedModule):
                inner = unwrap(value)
                if inner.__name__.split(".")[0] not in allowed:
                    leaked.append(f"{path}.{name} -> {inner.__name__}")
                walk(value, f"{path}.{name}", depth + 1)
            elif isinstance(value, types.ModuleType):
                leaked.append(f"{path}.{name} -> UNWRAPPED {value.__name__}")

    for root, proxy in tier0_module_namespace().items():
        walk(proxy, root, 1)

    assert visited > 50, "the walk should reach a real slice of the surface"
    assert leaked == [], f"reachable outside the allowlist: {leaked[:10]}"


def test_the_capability_rule_is_keyed_on_the_type_not_the_name():
    """The round-4 fix, asserted without naming a real re-export.

    ``pathlib.Path`` reached the script under four different attribute names
    and once as an instance. A name rule would have needed all five; asking
    what the VALUE IS needs one, and covers subclasses and aliases nobody has
    written yet.
    """
    import math
    import pathlib

    from app.core.script_proxy import module_proxy

    class MyPath(pathlib.PurePosixPath):
        """A subclass, defined in a module that IS allowlisted."""

    proxy = module_proxy(math)
    try:
        math.harmless_looking_name = pathlib.Path
        with pytest.raises(ScriptPolicyError, match="reads and writes files"):
            proxy.harmless_looking_name

        math.another_name = pathlib.Path(".")  # an instance, not the class
        with pytest.raises(ScriptPolicyError, match="reads and writes files"):
            proxy.another_name

        math.subclassed = MyPath  # __module__ is this test file, not pathlib
        with pytest.raises(ScriptPolicyError, match="reads and writes files"):
            proxy.subclassed
    finally:
        for attr in ("harmless_looking_name", "another_name", "subclassed"):
            delattr(math, attr)


def test_a_value_defined_by_a_blocked_module_is_refused_however_it_is_named():
    """The companion rule: ``torch.cuda.amp.common.find_spec`` is
    ``importlib``'s, and ``numpy.random.bit_generator.RLock`` is
    ``threading``'s. Being re-exported by an allowlisted library does not
    change what a function does."""
    import math
    import shutil
    import subprocess

    from app.core.script_proxy import module_proxy

    proxy = module_proxy(math)
    try:
        math.zz_one = shutil.rmtree
        with pytest.raises(ScriptPolicyError, match="'shutil' module"):
            proxy.zz_one
        math.zz_two = subprocess.Popen
        with pytest.raises(ScriptPolicyError, match="'subprocess' module"):
            proxy.zz_two
    finally:
        del math.zz_one
        del math.zz_two


def test_the_c_implementation_of_a_blocked_module_counts_as_that_module():
    """A blocked module is usually a thin wrapper, and what it re-exports
    declares the implementation: ``os.getcwd.__module__`` is ``nt``, and CI
    found ``numpy.random.bit_generator.RLock`` resolving to ``threading`` on
    one numpy and ``_thread`` on another. Blocking only the wrapper would make
    the verdict depend on which one a library imported from."""
    import _thread
    import io as io_module
    import math
    import os

    from app.core.script_proxy import module_proxy

    proxy = module_proxy(math)
    for name, value in (
        ("zz_getcwd", os.getcwd),
        ("zz_system", os.system),
        ("zz_fileio", io_module.FileIO),
        ("zz_rlock", _thread.RLock),
    ):
        setattr(math, name, value)
        try:
            with pytest.raises(ScriptPolicyError):
                getattr(proxy, name)
        finally:
            delattr(math, name)


def test_a_removed_builtin_is_refused_by_identity_not_by_name():
    """The namespace allowlist stops a script *writing* ``open``. It says
    nothing about a library handing the same object over under another name,
    and identity is the only check that does not care what that name is."""
    import builtins
    import math

    from app.core.script_proxy import module_proxy

    proxy = module_proxy(math)
    for banned in (builtins.open, builtins.eval, builtins.exec, builtins.compile):
        math.zz_banned = banned
        try:
            with pytest.raises(ScriptPolicyError, match="builtin"):
                proxy.zz_banned
        finally:
            del math.zz_banned
    # ...and an ordinary builtin-implemented library function is untouched.
    assert proxy.floor(1.7) == 1


def test_a_policy_check_never_crashes_on_an_odd_module_attribute():
    """``__module__`` is not always a string -- on some C-defined classes it
    is a slot descriptor, which CI hit on a numpy this machine did not have
    (``'member_descriptor' object has no attribute 'split'``). A guardrail
    that raises AttributeError from inside itself is worse than the hole."""
    import math

    from app.core.script_proxy import module_proxy

    class Weird:
        pass

    Weird.__module__ = object()  # not a str, as a C slot descriptor is not

    proxy = module_proxy(math)
    math.zz_weird = Weird
    try:
        assert proxy.zz_weird is Weird
    finally:
        del math.zz_weird


def test_no_file_capability_is_reachable_through_the_proxies():
    """Walk the proxy surface and assert no value that reads or writes files
    comes back. The sweep is not vacuous: with the capability check disabled
    it finds seven, four of which a review proved by writing a file."""
    import io as io_module
    import mmap
    import pathlib
    import types

    from app.core.plugin_validator import dangerous_modules
    from app.core.script_proxy import (
        RestrictedModule,
        tier0_module_namespace,
        unwrap,
    )

    caps = (pathlib.PurePath, io_module.IOBase, mmap.mmap)
    blocked = dangerous_modules()
    seen: set[int] = set()
    leaks: list[str] = []
    inspected = 0

    def defining_root(value):
        owner = (
            value
            if isinstance(
                value,
                (type, types.FunctionType, types.BuiltinFunctionType,
                 types.MethodType),
            )
            else type(value)
        )
        module = getattr(owner, "__module__", "")
        # Not always a str: some C-defined classes expose a slot descriptor
        # here, which is what broke this sweep on CI's numpy.
        return module.split(".")[0] if isinstance(module, str) else ""

    def walk(proxy, path: str, depth: int) -> None:
        nonlocal inspected
        if depth > 3:
            return
        target = unwrap(proxy)
        if id(target) in seen:
            return
        seen.add(id(target))
        for name in dir(target):
            try:
                value = getattr(proxy, name)
            except Exception:  # noqa: BLE001 - refused or simply absent
                continue
            if isinstance(value, RestrictedModule):
                walk(value, f"{path}.{name}", depth + 1)
                continue
            inspected += 1
            try:
                is_cap = (
                    isinstance(value, type) and issubclass(value, caps)
                ) or isinstance(value, caps)
            except TypeError:  # pragma: no cover - exotic metaclass
                is_cap = False
            if is_cap or defining_root(value) in blocked:
                leaks.append(f"{path}.{name} -> {value!r}"[:120])

    for root, proxy in tier0_module_namespace().items():
        walk(proxy, root, 1)

    assert inspected > 500, "the sweep should reach a real slice of the surface"
    assert leaks == [], f"file capability reachable: {leaks[:10]}"


def test_the_gate_and_the_proxy_agree_on_which_attributes_are_closed():
    """Two locks, one list. A door closed in the editor but open at run time
    (or the reverse) is how a policy drifts into decoration."""
    from app.core.script_proxy import _DENIED_ATTRS  # noqa: SLF001

    assert set(SCRIPT_PROXY_DENIED_ATTRS) == set(_DENIED_ATTRS)
    for name in ("hub", "cpp_extension", "savetxt", "os", "sys", "system", "popen"):
        assert name in SCRIPT_PROXY_DENIED_ATTRS


def test_the_rce_leaves_are_refused_as_attributes_not_only_as_calls():
    """``f = obj.system`` then ``f(cmd)`` has no Call node with a dangerous
    leaf. Tier 0 refuses the attribute; the shared blocklist still does not,
    because a plugin with a ``self.system`` field is ordinary code."""
    assert "system" in TIER0_RCE_ATTR_LEAVES
    with pytest.raises(PluginValidationError, match="system"):
        validate_script_source("def run(inputs, params):\n    return inputs.system\n")
    _plugin_mode("def go(cfg):\n    return cfg.system\n")


def test_a_library_s_private_attributes_are_refused_at_the_gate_too():
    """The proxy is the boundary, but the editor should say so while typing:
    ``collections._sys`` is a red line, not a surprise ten minutes later."""
    for snippet in ("collections._sys", "statistics.random._os", "re._parser"):
        with pytest.raises(PluginValidationError, match="private attribute"):
            validate_script_source(f"def run(inputs, params):\n    return {snippet}\n")


def test_the_private_attribute_rule_only_applies_to_library_receivers():
    """``self._cache`` is ordinary Python and must stay legal."""
    result = _run(
        "class Box:\n"
        "    def __init__(self):\n"
        "        self._items = [1, 2, 3]\n"
        "\n"
        "    def total(self):\n"
        "        return sum(self._items)\n"
        "\n"
        "\n"
        "def run(inputs, params):\n    return Box().total()\n"
    )
    assert result["out1"] == 6
    # ...and the rule is tier-0 only: an installed plugin reads its own
    # library internals all day.
    _plugin_mode("import json\n\ndef go():\n    return json._default_encoder\n")


def test_assigning_to_a_library_is_refused_but_ordinary_attributes_are_not():
    with pytest.raises(PluginValidationError, match="shared with the host"):
        validate_script_source(
            "import torch\n\ndef run(inputs, params):\n    torch.zeros = None\n"
        )
    result = _run(
        "class Box:\n    pass\n\n\n"
        "def run(inputs, params):\n    b = Box()\n    b.value = 5\n    return b.value\n"
    )
    assert result["out1"] == 5
    # Tier-0 only: a plugin patching a library attribute is its business.
    _plugin_mode("import json\n\ndef go():\n    json.x = 1\n")


def test_reading_a_pickle_loader_is_refused_even_without_calling_it():
    """The Call-keyed rule is evaded by one assignment; the attribute rule
    is not. ``json`` stays exempt, spelled out or aliased."""
    with pytest.raises(PluginValidationError, match="binding the function"):
        validate_script_source(
            "import torch\n\ndef run(inputs, params):\n    f = torch.load\n    return f\n"
        )
    validate_script_source(
        'import json\n\ndef run(inputs, params):\n    f = json.loads\n    return f("[]")\n'
    )
    validate_script_source(
        'import json as j\n\ndef run(inputs, params):\n    return j.loads("[]")\n'
    )


def test_weights_only_is_no_longer_a_tier0_escape_hatch():
    """A deliberate behaviour change, and the reason for it.

    The proxy hands out ATTRIBUTES; it cannot see a keyword argument, so
    ``torch.load`` reachable at all is ``f = torch.load; f(p)`` reachable.
    Tier 0 has no file access by design (``open`` is denied), so the two
    layers agree on "no" rather than the gate promising what the runtime
    refuses. Checkpoint loading belongs in a custom node.
    """
    with pytest.raises(PluginValidationError):
        validate_script_source(
            "import torch\n\ndef run(inputs, params):\n"
            "    return torch.load('x.pt', weights_only=True)\n"
        )
    # Installed plugins keep the escape hatch: they are files the user chose.
    _plugin_mode("import torch\n\ndef go(p):\n    return torch.load(p, weights_only=True)\n")


MUST_PASS_SCRIPTS = [
    (
        "json round-trip",
        'import json\n\ndef run(inputs, params):\n'
        '    return json.dumps(json.loads(\'{"a": 1}\'))\n',
    ),
    (
        "json through an alias",
        'import json as j\n\ndef run(inputs, params):\n    return j.loads("[1,2]")\n',
    ),
    (
        "numpy.linalg",
        "import numpy as np\n\ndef run(inputs, params):\n"
        "    return float(np.linalg.norm(np.ones(4)))\n",
    ),
    (
        "torch.nn.functional",
        "import torch\n\ndef run(inputs, params):\n"
        "    return float(torch.nn.functional.relu(torch.ones(3)).sum())\n",
    ),
    (
        "torch.nn.functional through an import alias",
        "import torch.nn.functional as F\nimport torch\n\ndef run(inputs, params):\n"
        "    return float(F.softmax(torch.ones(3), dim=0).sum())\n",
    ),
    (
        "from torch import nn",
        "from torch import nn\n\ndef run(inputs, params):\n"
        "    return int(nn.Linear(2, 2).weight.numel())\n",
    ),
    (
        "import numpy.linalg as la",
        "import numpy.linalg as la\nimport numpy as np\n\ndef run(inputs, params):\n"
        "    return float(la.norm(np.ones(9)))\n",
    ),
    (
        "from collections import Counter",
        "from collections import Counter\n\ndef run(inputs, params):\n"
        "    return {'out1': dict(Counter('aab'))}\n",
    ),
    (
        "collections.Counter",
        "import collections\n\ndef run(inputs, params):\n"
        "    return {'out1': dict(collections.Counter('aab'))}\n",
    ),
    (
        "statistics",
        "import statistics\n\ndef run(inputs, params):\n"
        "    return statistics.mean([1, 2, 3])\n",
    ),
    (
        "numpy.random",
        "import numpy as np\n\ndef run(inputs, params):\n"
        "    return float(np.random.default_rng(0).normal(size=4).mean())\n",
    ),
    (
        "math + itertools",
        "import math\nimport itertools\n\ndef run(inputs, params):\n"
        "    return math.sqrt(sum(1 for _ in itertools.repeat(0, 9)))\n",
    ),
    (
        "re",
        "import re\n\ndef run(inputs, params):\n"
        "    return bool(re.match(r'a+', 'aaa'))\n",
    ),
    (
        "a user class with helper methods",
        "import numpy as np\n\n\nclass S:\n"
        "    def __init__(self, v):\n        self.v = v\n\n"
        "    def mean(self):\n        return float(np.mean(self.v))\n\n\n"
        "def run(inputs, params):\n    return S([1.0, 2.0, 3.0]).mean()\n",
    ),
    (
        "a local named like a frame attribute",
        "def run(inputs, params):\n    f_code = 3\n    gi_frame = 4\n"
        "    return f_code + gi_frame\n",
    ),
    (
        "an attribute named .code",
        "class R:\n    def __init__(self):\n        self.code = 7\n\n\n"
        "def run(inputs, params):\n    return R().code\n",
    ),
    # ── the five recipes the docs ship ──────────────────────────────
    (
        "doc recipe: per-channel mean/std",
        "import torch\n\n\ndef run(inputs, params):\n"
        "    x = inputs['in1']\n"
        "    flat = x.reshape(x.shape[0], x.shape[1], -1)\n"
        "    mean = flat.mean(dim=(0, 2))\n"
        "    std = flat.std(dim=(0, 2))\n"
        "    print('channels:', mean.numel())\n"
        "    return {'out1': torch.stack([mean, std])}\n",
    ),
    (
        "doc recipe: class balance",
        "import collections\n\n\ndef run(inputs, params):\n"
        "    labels = inputs['in1']\n"
        "    counts = collections.Counter(int(v) for v in labels.reshape(-1).tolist())\n"
        "    total = sum(counts.values())\n"
        "    return {'out1': {k: v / total for k, v in sorted(counts.items())}}\n",
    ),
    (
        "doc recipe: robust summary",
        "import numpy as np\nimport torch\n\n\ndef run(inputs, params):\n"
        "    x = inputs['in1'].detach().cpu().numpy().reshape(-1)\n"
        "    q1, med, q3 = np.percentile(x, [25, 50, 75])\n"
        "    return {'out1': torch.tensor([float(q1), float(med), float(q3)])}\n",
    ),
    (
        "doc recipe: tensor comparison",
        "import torch\n\n\ndef run(inputs, params):\n"
        "    a = inputs['in1']\n    b = inputs['in2']\n"
        "    return {'out1': float((a - b).abs().max())}\n",
    ),
    (
        "doc recipe: creating on the run device",
        "import torch\n\n\ndef run(inputs, params):\n"
        "    return {'out1': torch.zeros(4, device=device)}\n",
    ),
]


@pytest.mark.parametrize(("label", "code"), MUST_PASS_SCRIPTS)
def test_the_must_pass_set_still_runs_under_the_proxy(label, code):
    """A boundary that breaks the library surface is not a fix, it is an
    outage. Every script here ran before the proxy and must run after it."""
    import torch

    inputs = {
        "in1": torch.arange(24.0).reshape(2, 3, 4),
        "in2": torch.ones(2, 3, 4),
    }
    result = PythonScriptNode().execute(inputs, {"code": code})
    assert "out1" in result, label


def test_a_refused_attribute_reports_a_line_number_like_any_other_failure():
    """A runtime refusal must land in the Execution Log the way a
    ZeroDivisionError does -- policy text, and the user's line."""
    with pytest.raises(RuntimeError) as exc:
        PythonScriptNode().execute(
            {},
            {"code": "def run(inputs, params):\n"
                     "    b = json.codecs.builtins\n"
                     "    f = b.eval\n"
                     "    return f('6*7')\n"},
        )
    message = str(exc.value)
    assert "line 2" in message
    assert "ScriptPolicyError" in message
    assert "not on the Tier-0 list" in message
    assert "Custom Nodes" in message  # the escape-hatch hint rides along
