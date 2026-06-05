"""Global device setting: the /api/system/devices endpoint, the
ExecutionContext.device flow into tensor-source nodes, and the
StatefulModuleMixin device move.

MPS-specific assertions are gated on availability so the file still runs on
Linux CI (where it exercises the CPU paths).
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
