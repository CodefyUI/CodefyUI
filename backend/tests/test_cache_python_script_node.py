"""Regression tests for #223 - PythonScript must not skip its side effects
on a cache hit.

A cache hit returns the RECORDED outputs dict without calling execute()
again. For a pure node that is invisible; for a node that did something
besides return a value, the something silently does not happen the second
time. #143 fixed exactly this for ImageWriter / ModelSaver / CheckpointSaver
by marking them ``cacheable = False``. PythonScript was left cacheable on
the reasoning that ``code`` is an ordinary param, so editing the script
already busts the key -- true, and beside the point: an UNEDITED script over
UNCHANGED inputs is precisely the case where the second run gets a hit and
its side effects are dropped.

What makes this node different from those three is that its type cannot tell
you whether it writes; only the source can, and only sometimes. These tests
pin the conservative answer by exercising the two side-effect routes that
are open to any script and cannot be closed:

* Something reachable through an input port. The ports are typed ANY, so a
  custom node or plugin can hand the script a logger, a writer, an open
  handle -- anything -- and ordinary attribute access on it is invisible to
  the tier-0 policy, which bounds which LIBRARIES a script may import, not
  what the objects it is GIVEN can do. ``_FileSinkNode`` below is the
  smallest honest stand-in for that whole class.
* In-place mutation of an input. ``inputs`` is a shallow copy, so the values
  are the upstream node's own objects. ``t.add_(1)`` is ordinary numerics
  that no AST rule could ever flag, and it changes state every downstream
  node shares.

The second one is also the argument against deriving cacheability from the
source AST (option 2 in #223): the check would have to separate "pure
transform" from "in-place mutation" in arbitrary Python, and a static
approximation that is ever wrong in the permissive direction reintroduces
this exact bug in a form nobody can see.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.core.cache import ExecutionCache
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.nodes.utility.python_script_node import PythonScriptNode


class _FileSinkNode(BaseNode):
    """Hands the script an object with a ``write`` method that appends to a
    real file.

    Deliberately not a builtin writer node: the point of #223 is that
    PythonScript's side-effect surface is whatever arrives on its input
    ports, which the node class cannot know. A test-only node makes that
    concrete without depending on which production node happens to emit a
    writable object this month.
    """

    NODE_NAME = "_FileSink223"
    CATEGORY = "Test"
    DESCRIPTION = "Appendable file sink, for the #223 cache regression test"

    #: Set by the test before the run; class-level so the node needs no
    #: params (params would land in the cache key and confuse the repro).
    target: Path | None = None

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="sink", data_type=DataType.ANY)]

    def execute(self, inputs, params):
        class _Sink:
            def __init__(self, path: Path) -> None:
                self._path = path

            def write(self, text: str) -> None:
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(text)

        assert _FileSinkNode.target is not None
        return {"sink": _Sink(_FileSinkNode.target)}


class _ZeroTensorNode(BaseNode):
    """A pure, cacheable tensor source.

    Its output is what the mutating script mutates, and -- because
    ExecutionCache stores results by reference -- the cache hands back the
    same object on the second run, so the mutation accumulates exactly as it
    would for a real upstream node.
    """

    NODE_NAME = "_ZeroTensor223"
    CATEGORY = "Test"
    DESCRIPTION = "Deterministic zero tensor, for the #223 cache regression test"

    #: Every tensor this node has produced, so the test can inspect the one
    #: object both runs shared.
    produced: list[torch.Tensor] = []

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="tensor", data_type=DataType.TENSOR)]

    def execute(self, inputs, params):
        tensor = torch.zeros(3)
        _ZeroTensorNode.produced.append(tensor)
        return {"tensor": tensor}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes["_FileSink223"] = _FileSinkNode
    registry._nodes["_ZeroTensor223"] = _ZeroTensorNode
    _ZeroTensorNode.produced = []
    yield
    registry._nodes.pop("_FileSink223", None)
    registry._nodes.pop("_ZeroTensor223", None)
    _FileSinkNode.target = None
    _ZeroTensorNode.produced = []


def _graph(source_type: str, source_handle: str, code: str, input_type: str):
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": source_type, "data": {"params": {}}},
        {
            "id": "py",
            "type": "PythonScript",
            "data": {
                "params": {
                    "code": code,
                    "input_types": input_type,
                    "output_types": "ANY",
                }
            },
        },
    ]
    edges = [
        {
            "id": "et", "source": "start", "target": "src",
            "sourceHandle": "trigger", "type": "trigger",
        },
        {
            "id": "e1", "source": "src", "target": "py",
            "sourceHandle": source_handle, "targetHandle": "in1",
        },
    ]
    return nodes, edges


def test_python_script_is_not_cacheable() -> None:
    """Pins #223's answer so it cannot drift back.

    ``code`` being a cache-keyed param covers the EDITED script. It says
    nothing about the unedited one, which is where a hit happens and where
    the side effects go missing.
    """
    assert PythonScriptNode.cacheable is False, (
        "PythonScript runs code only the user has seen, and a cache hit "
        "returns its recorded outputs without running it. Any side effect "
        "the script has -- an input mutated in place, process-global torch "
        "or numpy state, anything reached through an ANY-typed input port -- "
        "is then silently skipped, so the node must opt out with "
        "`cacheable = False` (#223, same class as #143)."
    )


def test_the_node_says_it_always_re_runs() -> None:
    """#223's acceptance: a user who notices the node never shows `cached`
    should be able to read why in the node itself, not guess at a bug.
    """
    assert "never cached" in PythonScriptNode.DESCRIPTION


@pytest.mark.asyncio
async def test_a_writing_script_writes_on_every_run(tmp_path: Path) -> None:
    """The #223 repro. A pure, cacheable upstream feeds a script that
    writes, so the whole graph is cacheable end to end. Run, delete the file
    the way a user would, run again against the same cache: the file must
    come back.
    """
    target = tmp_path / "written_by_the_script.txt"
    _FileSinkNode.target = target

    code = (
        "def run(inputs, params):\n"
        "    inputs['in1'].write('ran\\n')\n"
        "    return {'out1': 1}\n"
    )
    nodes, edges = _graph("_FileSink223", "sink", code, "ANY")

    cache = ExecutionCache()
    statuses: dict[str, str] = {}

    async def track(node_id, status, data):
        statuses[node_id] = status

    await execute_graph(nodes, edges, on_progress=track, cache=cache)
    assert statuses["py"] == "completed"
    assert target.exists(), "first run must write the file"

    target.unlink()
    assert not target.exists()

    await execute_graph(nodes, edges, on_progress=track, cache=cache)
    assert target.exists(), (
        "the file the script writes was not rewritten on the second run -- "
        f"PythonScript was served from cache (status {statuses['py']!r}) and "
        "its write was skipped"
    )
    # Same guard as the #143 test (see 866c23d): PythonScript is
    # cacheable=False unconditionally now, so its own "completed" status on
    # the second run reads identically whether this fix engaged or caching
    # broke globally and every node always re-runs. src is pure, cacheable,
    # has no upstream and unchanged params, so it MUST be a hit here -- if it
    # is not, this test proves nothing about PythonScript specifically.
    assert statuses["src"] == "cached", (
        "the pure upstream must still hit the cache on the second run -- "
        "otherwise this test cannot distinguish '#223 fixed' from 'caching "
        f"stopped working entirely' (status was {statuses['src']!r})"
    )


@pytest.mark.asyncio
async def test_a_script_that_mutates_its_input_mutates_it_on_every_run() -> None:
    """The side-effect route no static analysis can see.

    ``t.add_(1)`` is indistinguishable from a pure transform in the AST, and
    no security policy will ever refuse it -- it is the numerics this node
    exists for. It still changes an object the upstream node owns and every
    downstream node shares, so skipping it on a cache hit leaves the graph
    holding a value that does not match what a fresh run would produce.
    """
    code = (
        "def run(inputs, params):\n"
        "    t = inputs['in1']\n"
        "    t.add_(1)\n"
        "    return {'out1': float(t.sum())}\n"
    )
    nodes, edges = _graph("_ZeroTensor223", "tensor", code, "TENSOR")

    cache = ExecutionCache()
    statuses: dict[str, str] = {}

    async def track(node_id, status, data):
        statuses[node_id] = status

    await execute_graph(nodes, edges, on_progress=track, cache=cache)
    await execute_graph(nodes, edges, on_progress=track, cache=cache)

    # The upstream ran once and was cached; ExecutionCache stores by
    # reference, so both runs saw the same tensor object.
    assert statuses["src"] == "cached"
    assert len(_ZeroTensorNode.produced) == 1
    shared = _ZeroTensorNode.produced[0]

    assert shared.tolist() == [2.0, 2.0, 2.0], (
        "the script's in-place mutation ran only once across two runs -- "
        "the second run was served from cache, so the tensor every "
        f"downstream node sees is stale (it holds {shared.tolist()})"
    )
