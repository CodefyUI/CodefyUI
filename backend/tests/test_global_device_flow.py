"""Global device setting: the /api/system/devices endpoint, the
ExecutionContext.device flow into tensor-source nodes, and the
StatefulModuleMixin device move, and the engine-level input alignment that
makes the two meet.

CPU is the baseline and the default. An accelerator is opt-in, and
everything below has to keep holding when someone opts in.

Accelerator-specific assertions are gated on availability so the file still
runs on CPU-only CI. That gating is load-bearing rather than tidy: a device
mismatch is UNREPRESENTABLE on a single-device machine, so the alignment
guards at the bottom can only fail where a second device exists. That is
exactly how the two failures they pin got out.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.stateful_module import StatefulModuleMixin
from app.core.node_base import BaseNode

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not mps_available, reason="MPS not available")


# ── /api/system/devices endpoint ────────────────────────────────────

@pytest.mark.asyncio
async def test_devices_endpoint(test_client):
    resp = await test_client.get("/api/system/devices")
    assert resp.status_code == 200
    data = resp.json()
    assert "default" in data and "devices" in data
    values = {d["value"] for d in data["devices"]}
    assert "cpu" in values
    assert data["default"] in values


# ── ExecutionContext.device flows into a tensor-source node ──────────

def _tensor_create_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {
            "id": "tc",
            "type": "TensorCreate",
            "data": {"params": {"shape": "2,3", "fill": "randn"}},
        },
    ]
    edges = [
        {"id": "t", "source": "start", "target": "tc", "sourceHandle": "trigger", "type": "trigger"},
    ]
    return nodes, edges


@pytest.mark.asyncio
async def test_tensor_create_runs_on_context_device_cpu():
    nodes, edges = _tensor_create_graph()
    outs = await execute_graph(nodes, edges, context=ExecutionContext(device="cpu"))
    assert outs["tc"]["tensor"].device.type == "cpu"


@requires_mps
@pytest.mark.asyncio
async def test_tensor_create_runs_on_context_device_mps():
    nodes, edges = _tensor_create_graph()
    outs = await execute_graph(nodes, edges, context=ExecutionContext(device="mps"))
    assert outs["tc"]["tensor"].device.type == "mps"


@pytest.mark.asyncio
async def test_default_context_keeps_tensors_on_cpu():
    # No context (CLI-style) → tensors stay on CPU.
    nodes, edges = _tensor_create_graph()
    outs = await execute_graph(nodes, edges)
    assert outs["tc"]["tensor"].device.type == "cpu"


# ── StatefulModuleMixin moves the module to context.device ──────────

class _LinearNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "_TestLinear"
    CATEGORY = "Test"
    DESCRIPTION = "test"
    structural_params = ("n",)

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return []

    def build_module(self, params):
        return nn.Linear(params.get("n", 4), 2)

    def execute(self, inputs, params, *, context=None):
        return {"module": self.get_or_build_module(context, params)}


def test_get_or_build_module_moves_to_context_device_cpu():
    node = _LinearNode()
    mod = node.get_or_build_module(ExecutionContext(device="cpu"), {"n": 4})
    assert next(mod.parameters()).device.type == "cpu"


@requires_mps
def test_get_or_build_module_moves_to_context_device_mps():
    node = _LinearNode()
    mod = node.get_or_build_module(ExecutionContext(device="mps"), {"n": 4})
    assert next(mod.parameters()).device.type == "mps"


def test_get_or_build_module_no_context_builds_on_cpu():
    node = _LinearNode()
    mod = node.get_or_build_module(None, {"n": 4})
    assert next(mod.parameters()).device.type == "cpu"


# ── The engine aligns a node's inputs to the device it runs on ───────
#
# The bug this closes, twice over, from the server log of a CUDA box:
#
#   conv2d_node.py:54  RuntimeError: Input type (torch.FloatTensor) and weight
#                      type (torch.cuda.FloatTensor) should be the same
#   graph_model.py:98  RuntimeError: Expected all tensors to be on the same
#                      device, but got mat1 is on cpu, different from other
#                      tensors on cuda:0
#
# Two halves of one sentence in ExecutionContext.device's docstring: "tensor-
# source nodes create on it AND StatefulModuleMixin moves layer modules to
# it". The module half was enforced centrally and always held. The tensor half
# was left to each node, and ten of the fourteen nodes that create tensors
# from nothing never learned it. Alignment now happens in ``invoke_node``, so
# it is a property of the engine rather than of each author's memory.

from app.core.device_utils import align_tensors, node_target_device  # noqa: E402
from app.core.graph_engine import invoke_node  # noqa: E402
from app.core.node_base import ParamDefinition, ParamType  # noqa: E402

cuda_available = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

#: A second device to move things to, when the machine has one at all.
SECOND_DEVICE = "cuda" if cuda_available else ("mps" if mps_available else None)
requires_two_devices = pytest.mark.skipif(
    SECOND_DEVICE is None,
    reason="needs a non-CPU device: a mismatch cannot exist on one device",
)


class _SpyNode(BaseNode):
    """Records where its inputs arrived, so a test can assert on it."""

    NODE_NAME = "_TestSpy"
    CATEGORY = "Test"
    DESCRIPTION = "test"

    def __init__(self):
        self.seen = None

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return []

    def execute(self, inputs, params, *, context=None):
        self.seen = inputs
        return {}


class _PinnedSpyNode(_SpyNode):
    """A node with its own ``device`` param, which must win over the run's."""

    NODE_NAME = "_TestPinnedSpy"

    @classmethod
    def define_params(cls):
        return [ParamDefinition(name="device", param_type=ParamType.SELECT, default="auto")]


def test_align_tensors_moves_tensors_and_leaves_modules_alone():
    module = nn.Linear(2, 2)
    tensor = torch.zeros(2, 2)
    out = align_tensors({"m": module, "t": tensor}, "cpu")
    # The module is the SAME object, not a relocated copy: `Module.to()` is
    # in-place, so moving one handed across a wire would flip weights out
    # from under whichever node owns it.
    assert out["m"] is module
    assert out["t"].device.type == "cpu"


def test_align_tensors_walks_nested_collections():
    payload = {"batch": [torch.zeros(1), (torch.ones(1), torch.ones(1))]}
    out = align_tensors(payload, "cpu")
    assert out["batch"][0].device.type == "cpu"
    assert isinstance(out["batch"][1], tuple)
    assert out["batch"][1][0].device.type == "cpu"


def test_align_tensors_passes_non_tensor_leaves_through():
    class _Dataset:
        pass

    ds = _Dataset()
    out = align_tensors({"ds": ds, "n": 3, "s": "x", "none": None}, "cpu")
    # A dataset, a DataLoader or an environment is not a tensor, and walking
    # one would drag a lazily-loaded dataset into memory.
    assert out["ds"] is ds
    assert out == {"ds": ds, "n": 3, "s": "x", "none": None}


def test_node_target_device_follows_the_run_then_the_node():
    ctx = ExecutionContext(device="cpu")
    assert node_target_device({}, ctx) == "cpu"
    assert node_target_device({"device": "auto"}, ctx) == "cpu"
    assert node_target_device({"device": "cpu"}, ctx) == "cpu"
    assert node_target_device(None, ctx) == "cpu"


def test_invoke_node_leaves_inputs_alone_without_a_context():
    node = _SpyNode()
    t = torch.zeros(2)
    invoke_node(node, {"tensor": t}, {}, context=None)
    # The CLI runner and a direct execute() call pass no context; nothing to
    # align against, so nothing is touched.
    assert node.seen["tensor"] is t


@requires_two_devices
def test_invoke_node_aligns_a_stray_cpu_input_onto_the_run_device():
    node = _SpyNode()
    invoke_node(
        node,
        {"tensor": torch.zeros(2, 2)},   # a source node that forgot to move it
        {},
        context=ExecutionContext(device=SECOND_DEVICE),
    )
    assert node.seen["tensor"].device.type == SECOND_DEVICE


@requires_two_devices
def test_invoke_node_does_not_relocate_a_module_it_was_handed():
    node = _SpyNode()
    module = nn.Linear(2, 2)
    invoke_node(
        node, {"model": module}, {}, context=ExecutionContext(device=SECOND_DEVICE),
    )
    assert node.seen["model"] is module
    assert next(node.seen["model"].parameters()).device.type == "cpu"


@requires_two_devices
def test_a_node_pinned_to_cpu_gets_cpu_inputs_on_an_accelerated_run():
    node = _PinnedSpyNode()
    tensor = torch.zeros(2, 2).to(SECOND_DEVICE)
    invoke_node(
        node,
        {"tensor": tensor},
        {"device": "cpu"},
        context=ExecutionContext(device=SECOND_DEVICE),
    )
    # A graph may pin one node to the CPU; alignment follows the NODE, not
    # the run, or the pin would be silently undone.
    assert node.seen["tensor"].device.type == "cpu"


@requires_cuda
def test_a_cpu_tensor_reaching_conv2d_no_longer_dies_on_cuda():
    """The exact call that failed in the log, with the exact node.

    ``Conv2d`` is a ``StatefulModuleMixin`` node, so its weights are moved to
    the run device. Hand it a tensor that some upstream node built without
    consulting that device -- ten of the fourteen tensor sources did exactly
    that -- and on a CUDA box ``conv(tensor)`` raised "Input type
    (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the
    same". This is that call, one layer below the graph engine so nothing but
    the alignment stands between the fixture and the crash.
    """
    from app.nodes.cnn.conv2d_node import Conv2dNode

    out = invoke_node(
        Conv2dNode(),
        {"tensor": torch.randn(1, 1, 8, 8)},        # built on the CPU
        {"in_channels": 1, "out_channels": 2},
        context=ExecutionContext(device="cuda"),    # module goes to cuda
    )
    assert out["tensor"].device.type == "cuda"
