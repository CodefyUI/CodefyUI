"""PythonScript nodes in shipped examples make their tensors on the run's device.

#808: ``Papers/Weight-Decay-Regression`` built ``torch.linspace(-1, 1, 200)``
inside a PythonScript and stacked it with the node's inputs. The engine moves
a node's inputs onto the run's device, so on a GPU run the inputs were on
``cuda:0`` and the grid on the CPU, and ``torch.cat`` refused:
"Expected all tensors to be on the same device". On the CPU every tensor is on
the same device, so running the graph in CI (which has no GPU) cannot catch it.
Reading the script can, and that is all this file does.

The rule: a script that reads ``inputs`` and builds a tensor from a shape or
a range (``torch.linspace``, ``torch.zeros``, ...) passes ``device=``. The
script's ``device`` global names the run's device for exactly this. Two cases
are left alone on purpose:

* A script that reads no inputs (a data generator). Its outputs reach their
  consumers through the engine, which moves them onto each consumer's device.
* ``torch.tensor(...)`` / ``torch.as_tensor(...)`` built from Python data, the
  usual way a script hands a list of numbers to a chart. Requiring a device
  there would flag correct scripts far more often than it would catch a bug.

The check is static and cheap. The real-GPU half is
``scripts/device_smoke.py cuda``, which runs the #808 examples on the device.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_GRAPHS = sorted(
    [*(_REPO_ROOT / "examples").rglob("graph.json"),
     *(_REPO_ROOT / "plugins").glob("*/examples/**/graph.json")]
)
assert _GRAPHS, "example script-device suite discovered no graphs"

_IDS = [p.relative_to(_REPO_ROOT).as_posix() for p in _GRAPHS]

# torch factories that build a tensor from a shape or a range, not from data.
# They land on the CPU unless told otherwise.
_SHAPE_FACTORIES = frozenset({
    "arange", "empty", "eye", "full", "linspace", "logspace", "normal",
    "ones", "rand", "randint", "randn", "randperm", "zeros",
})


def _python_scripts(payload: dict):
    """(node id, code) for every PythonScript node, at any nesting depth."""
    stack: list = [payload]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            if obj.get("type") == "PythonScript":
                code = ((obj.get("data") or {}).get("params") or {}).get("code")
                if isinstance(code, str):
                    yield obj.get("id"), code
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)


def _factories_off_the_run_device(code: str) -> list[str]:
    """Shape/range factory calls with no ``device=``, in a script that reads
    ``inputs``; empty for a script that reads none."""
    tree = ast.parse(code)
    if not any(isinstance(n, ast.Name) and n.id == "inputs" for n in ast.walk(tree)):
        return []
    return [
        f"line {n.lineno}: torch.{n.func.attr}(...)"
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "torch"
        and n.func.attr in _SHAPE_FACTORIES
        and not any(k.arg == "device" for k in n.keywords)
    ]


@pytest.mark.parametrize("graph_path", _GRAPHS, ids=_IDS)
def test_python_scripts_make_tensors_on_the_run_device(graph_path: Path):
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    offenders = {
        node_id: found
        for node_id, code in _python_scripts(payload)
        if (found := _factories_off_the_run_device(code))
    }

    assert not offenders, (
        f"{graph_path.name}: PythonScript nodes read their inputs, which are "
        f"on the run's device, and also build tensors without device=: "
        f"{offenders}. Pass device=device (the script's global for the run's "
        f"device), or a GPU run mixes CPU and GPU tensors (#808)."
    )


_BEFORE_808 = (
    "import torch\n"
    "def run(inputs, params):\n"
    "    grid = torch.linspace(-1, 1, 200)\n"
    "    return {'out1': torch.stack([grid, inputs['in1']], 1)}\n"
)


def test_the_rule_flags_the_808_script():
    assert _factories_off_the_run_device(_BEFORE_808) == [
        "line 3: torch.linspace(...)"]


def test_the_rule_passes_the_808_fix():
    fixed = _BEFORE_808.replace("200)", "200, device=device)")
    assert _factories_off_the_run_device(fixed) == []


def test_the_rule_leaves_a_generator_and_data_conversion_alone():
    generator = (
        "import torch\n"
        "def run(inputs, params):\n"
        "    return {'out1': torch.linspace(-1, 1, 200)}\n"
    )
    chart = (
        "import torch\n"
        "def run(inputs, params):\n"
        "    return {'out1': torch.tensor(inputs['in1'].tolist())}\n"
    )
    assert _factories_off_the_run_device(generator) == []
    assert _factories_off_the_run_device(chart) == []


def test_the_rule_finds_the_808_example_scripts():
    """Guards the walk itself: if it stopped finding PythonScript nodes, every
    parametrized case above would pass on nothing."""
    graph = _REPO_ROOT / "examples/Papers/Weight-Decay-Regression/graph.json"
    ids = {i for i, _ in _python_scripts(json.loads(graph.read_text(encoding="utf-8")))}
    assert {"gen", "fits"} <= ids
