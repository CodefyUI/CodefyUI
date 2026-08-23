"""Global device setting: the /api/system/devices endpoint, the
ExecutionContext.device flow into tensor-source nodes, and the
StatefulModuleMixin device move, and the engine-level input alignment that
makes the two meet.

CPU is the baseline and the default. An accelerator is opt-in, and
everything below has to keep holding when someone opts in.

The alignment guards at the bottom relocate onto ``meta``, a real torch
device present on every machine, so they FAIL on a CPU-only runner when the
feature is broken. Gating them on real hardware instead was how the first
version of this file passed while doing nothing: CI is ubuntu and windows,
so every such test skipped, and the ungated ones aligned a CPU tensor to
"cpu", where ``.to()`` returns self.

The few assertions that need arithmetic to actually happen -- ``meta``
allocates no storage -- stay gated on a real accelerator.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, namedtuple
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, ParamDefinition, ParamType
from app.core.stateful_module import StatefulModuleMixin

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

    @classmethod
    def define_params(cls):
        return [ParamDefinition(
            name="device", param_type=ParamType.SELECT, default="auto")]

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

from app.core.device_utils import (  # noqa: E402
    MAX_ALIGN_DEPTH,
    align_tensors,
    node_target_device,
)
from app.core.graph_engine import invoke_node  # noqa: E402

cuda_available = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

#: A second device to move things to -- on EVERY machine, including CPU-only
#: CI. `meta` is a real torch device with real device identity and no
#: allocation, so `t.to("meta").device.type` is "meta" on a bare ubuntu
#: runner. That matters more here than the tidiness of using the accelerator
#: when one is present: gating these on real hardware is what let the first
#: version of this file pass while the feature did nothing. Replacing
#: `align_tensors` with the identity function left the whole suite green,
#: because every ungated assertion aligned a CPU tensor to "cpu" -- where
#: `.to()` returns self and pass-through is indistinguishable from working.
#:
#: A node with no `device` param resolves through `context_device`, which
#: returns `context.device` verbatim, so "meta" reaches `align_tensors`
#: without needing `resolve_device` to know the name.
SECOND_DEVICE = "meta"

#: For the handful of assertions that need arithmetic to actually run (`meta`
#: allocates nothing, so a conv on it produces shapes but no values).
REAL_ACCELERATOR = "cuda" if cuda_available else ("mps" if mps_available else None)
requires_two_devices = pytest.mark.skipif(
    REAL_ACCELERATOR is None,
    reason="needs a real second device: this one computes, not just relocates",
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
    out = align_tensors({"m": module, "t": tensor}, SECOND_DEVICE)
    # The module is the SAME object, not a relocated copy: `Module.to()` is
    # in-place, so moving one handed across a wire would flip weights out
    # from under whichever node owns it.
    assert out["m"] is module
    assert next(module.parameters()).device.type == "cpu"
    assert out["t"].device.type == SECOND_DEVICE


def test_align_tensors_walks_nested_collections():
    payload = {"batch": [torch.zeros(1), (torch.ones(1), torch.ones(1))]}
    out = align_tensors(payload, SECOND_DEVICE)
    assert out["batch"][0].device.type == SECOND_DEVICE
    assert isinstance(out["batch"][1], tuple)
    assert out["batch"][1][0].device.type == SECOND_DEVICE


def test_align_tensors_returns_the_same_object_when_nothing_moved():
    """No-op alignment must not rebuild containers.

    ``python_script_node`` documents in-place mutation of an input --
    ``inputs["lst"].append(x)`` -- as a side-effect route downstream nodes
    share, and the engine's cacheability reasoning rests on it. Rebuilding a
    container whose contents are already in place would break that on
    CPU-only runs, where alignment has no business changing anything.
    """
    payload = {"lst": [torch.zeros(1)], "t": torch.zeros(1)}
    out = align_tensors(payload, "cpu")
    assert out is payload
    assert out["lst"] is payload["lst"]
    assert out["t"] is payload["t"]


def test_align_tensors_rebuilds_a_namedtuple_field_by_field():
    """``type(obj)(moved)`` is silently WRONG for an arity-1 namedtuple.

    It packs the whole iterable into the first field instead of raising, so
    the ``except TypeError`` fallback never fires and the field's type
    changes from Tensor to list with nothing to see in the log.
    """
    single = namedtuple("_Single", "x")
    out = align_tensors(single(torch.zeros(2)), SECOND_DEVICE)
    assert isinstance(out, single)
    assert isinstance(out.x, torch.Tensor)
    assert out.x.device.type == SECOND_DEVICE

    pair = namedtuple("_Pair", "a b")
    out2 = align_tensors(pair(torch.zeros(1), torch.ones(1)), SECOND_DEVICE)
    assert isinstance(out2, pair)
    assert out2.a.device.type == SECOND_DEVICE and out2.b.device.type == SECOND_DEVICE


def test_align_tensors_keeps_a_mapping_subclass_and_its_state():
    """A state_dict is an OrderedDict carrying ``_metadata``.

    ``load_state_dict`` reads that attribute for versioned loading, and
    ``dict(moved)`` drops both the type and the attribute.
    """
    sd = nn.Linear(2, 2).state_dict()
    out = align_tensors(sd, SECOND_DEVICE)
    assert isinstance(out, OrderedDict)
    assert hasattr(out, "_metadata")
    assert all(v.device.type == SECOND_DEVICE for v in out.values())

    dd = defaultdict(list)
    dd["w"] = torch.zeros(1)
    out2 = align_tensors(dd, SECOND_DEVICE)
    assert isinstance(out2, defaultdict) and out2.default_factory is list


def test_align_tensors_aligns_inside_a_set():
    out = align_tensors({torch.zeros(1)}, SECOND_DEVICE)
    assert isinstance(out, set)
    assert {t.device.type for t in out} == {SECOND_DEVICE}


def test_align_tensors_keeps_a_parameter_optimisable():
    """Cross-device ``.to()`` drops Parameter-ness and leaf-ness.

    ``SGD([t])`` then raises "can't optimize a non-leaf Tensor", and a
    ``.grad`` that never populates is the quieter half of the same bug.
    """
    p = nn.Parameter(torch.zeros(2))
    out = align_tensors(p, SECOND_DEVICE)
    assert isinstance(out, nn.Parameter)
    assert out.is_leaf and out.requires_grad

    leaf = torch.zeros(2, requires_grad=True)
    assert align_tensors(leaf, SECOND_DEVICE).is_leaf


def test_align_tensors_survives_a_self_referential_container():
    """Unbounded recursion here surfaces as "the node failed".

    ``invoke_node`` runs inside the node's own try, so a RecursionError from
    a cycle would be reported against a node that computed fine -- the same
    reasoning that made ``memory_budget._walk`` bound its descent.
    """
    cyclic: list = [torch.zeros(1)]
    cyclic.append(cyclic)
    out = align_tensors(cyclic, SECOND_DEVICE)
    assert out[0].device.type == SECOND_DEVICE

    deep: Any = torch.zeros(1)
    for _ in range(MAX_ALIGN_DEPTH + 5):
        deep = [deep]
    align_tensors(deep, SECOND_DEVICE)  # must not raise


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


def test_node_target_device_ignores_a_device_key_the_node_never_declared():
    """``params`` is client-supplied; only a DECLARED param may steer this."""
    ctx = ExecutionContext(device=SECOND_DEVICE)
    # _SpyNode declares no params, so this key is not its device param and
    # the run's device stands.
    assert node_target_device({"device": "cpu"}, ctx, _SpyNode()) == SECOND_DEVICE
    # _PinnedSpyNode does declare one, so the same key is honoured.
    assert node_target_device({"device": "cpu"}, ctx, _PinnedSpyNode()) == "cpu"
    # With no node to ask, the key is taken at face value, as before.
    assert node_target_device({"device": "cpu"}, ctx) == "cpu"


def test_node_target_device_ignores_a_value_that_is_not_a_device():
    """A ``device`` param meaning a serial port must not pin the run to CPU.

    ``resolve_device`` answers an unrecognised string with "cpu" and a
    warning naming the wrong subject, which would silently pull every input
    off the accelerator.
    """
    ctx = ExecutionContext(device=SECOND_DEVICE)
    assert node_target_device({"device": "/dev/ttyUSB0"}, ctx) == SECOND_DEVICE
    assert node_target_device({"device": "auto:balanced"}, ctx) == SECOND_DEVICE
    # A device-shaped value that is merely unavailable still degrades.
    assert node_target_device({"device": "cpu"}, ctx) == "cpu"


def test_the_module_half_and_the_tensor_half_use_one_rule():
    """A pinned stateful node must get weights and inputs on the same device.

    ``get_or_build_module`` used ``context.device`` while ``invoke_node``
    used the node's pin, so the two halves of one guarantee came apart
    exactly where a graph asked for something specific.
    """
    node = _LinearNode()
    mod = node.get_or_build_module(
        ExecutionContext(device=SECOND_DEVICE), {"n": 4, "device": "cpu"})
    assert next(mod.parameters()).device.type == "cpu"


def test_a_node_can_opt_out_of_alignment():
    """Host-only nodes hand their input to numpy, where a GPU tensor raises."""
    class _HostOnly(_SpyNode):
        align_inputs = False

    node = _HostOnly()
    t = torch.zeros(2)
    invoke_node(node, {"tensor": t}, {}, context=ExecutionContext(device=SECOND_DEVICE))
    assert node.seen["tensor"] is t


def test_invoke_node_leaves_inputs_alone_without_a_context():
    node = _SpyNode()
    t = torch.zeros(2)
    invoke_node(node, {"tensor": t}, {}, context=None)
    # The CLI runner and a direct execute() call pass no context; nothing to
    # align against, so nothing is touched.
    assert node.seen["tensor"] is t


def test_invoke_node_aligns_a_stray_cpu_input_onto_the_run_device():
    node = _SpyNode()
    invoke_node(
        node,
        {"tensor": torch.zeros(2, 2)},   # a source node that forgot to move it
        {},
        context=ExecutionContext(device=SECOND_DEVICE),
    )
    assert node.seen["tensor"].device.type == SECOND_DEVICE


def test_invoke_node_does_not_relocate_a_module_it_was_handed():
    node = _SpyNode()
    module = nn.Linear(2, 2)
    invoke_node(
        node, {"model": module}, {}, context=ExecutionContext(device=SECOND_DEVICE),
    )
    assert node.seen["model"] is module
    assert next(node.seen["model"].parameters()).device.type == "cpu"


def test_a_node_pinned_to_cpu_gets_cpu_inputs_on_an_accelerated_run():
    node = _PinnedSpyNode()
    tensor = torch.zeros(2, 2)
    invoke_node(
        node,
        {"tensor": tensor},
        {"device": "cpu"},
        context=ExecutionContext(device=SECOND_DEVICE),
    )
    # A graph may pin one node to the CPU; alignment follows the NODE, not
    # the run, or the pin would be silently undone. Without the pin this
    # same call relocates the tensor -- the test above does exactly that.
    assert node.seen["tensor"] is tensor
    assert node.seen["tensor"].device.type == "cpu"


@requires_two_devices
def test_a_stray_cpu_input_computes_against_an_aligned_module():
    """The `meta` guards prove relocation; this one proves the arithmetic.

    A module the engine put on the accelerator, a tensor an upstream source
    built without consulting the device, and a forward pass that raised
    "Expected all tensors to be on the same device" until alignment moved
    into the engine.
    """
    class _Forward(_SpyNode, StatefulModuleMixin):
        NODE_NAME = "_TestForward"

        def build_module(self, params):
            return nn.Linear(4, 2)

        def execute(self, inputs, params, *, context=None):
            module = self.get_or_build_module(context, params)
            return {"tensor": module(inputs["tensor"])}

    out = invoke_node(
        _Forward(),
        {"tensor": torch.randn(3, 4)},          # built on the CPU
        {},
        context=ExecutionContext(device=REAL_ACCELERATOR),
    )
    assert out["tensor"].device.type == REAL_ACCELERATOR


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
