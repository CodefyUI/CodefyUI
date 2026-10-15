"""A node ``device`` param accepts the per-card values the UI offers (#802).

Every node-level ``device`` param declares the static
``DEVICE_PARAM_OPTIONS`` (auto, cpu, cuda, mps). The node API serves the
machine's list through ``device_options``, which adds ``cuda:0..N`` on a
multi-card box, so the config panel offers ``cuda:1``. The #698 SELECT check
in ``validate_graph`` compared against the static list only, so the value
the UI offered was refused at run and at export. These tests pin the fix and
its limits: per-card forms pass, anything else still fails as before.

No graphics card needed: ``torch.cuda`` is monkeypatched the way
``test_device_index.py`` does it.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.core.device_utils import get_available_devices
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph, validate_graph
from app.nodes.io.model_saver_node import ModelSaverNode


@pytest.fixture
def four_cards(monkeypatch):
    """Pretend this machine has four CUDA devices, numbered 0 to 3."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda index=0: f"Fake GPU {index}")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    get_available_devices.cache_clear()
    yield
    get_available_devices.cache_clear()


@pytest.fixture
def no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    get_available_devices.cache_clear()
    yield
    get_available_devices.cache_clear()


def _node(node_id: str, node_type: str, **params) -> dict:
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": {"params": params}}


def _trigger(source: str, target: str) -> dict:
    return {"id": f"t-{source}-{target}", "source": source, "target": target,
            "sourceHandle": "trigger", "targetHandle": "__trigger",
            "type": "trigger"}


def _loader_graph(device: str, **params) -> tuple[list[dict], list[dict]]:
    """Start -> ModelLoader(device=...), the graph from the issue."""
    nodes = [_node("start", "Start"),
             _node("load", "ModelLoader", device=device, **params)]
    return nodes, [_trigger("start", "load")]


def _option_issues(errors: list) -> list:
    return [e for e in errors if getattr(e, "code", None) == "param_not_an_option"]


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1", "cuda:12", "mps:0"])
def test_a_per_card_device_passes_validation(device):
    nodes, edges = _loader_graph(device)
    assert _option_issues(validate_graph(nodes, edges)) == []


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps"])
def test_the_declared_options_still_pass(device):
    nodes, edges = _loader_graph(device)
    assert _option_issues(validate_graph(nodes, edges)) == []


@pytest.mark.parametrize("device", [
    "cuda:x", "gpu", "cuda:", "cuda:0:1", "CUDA:1", " cuda:1", "cuda: 1",
    "mps:a", "tpu:0",
])
def test_anything_else_is_still_refused_with_the_same_message(device):
    nodes, edges = _loader_graph(device)
    (issue,) = _option_issues(validate_graph(nodes, edges))
    assert issue == (
        f"Parameter 'device' on node load (ModelLoader): value {device!r} "
        "is not one of its options (auto, cpu, cuda, mps)"
    )


def test_a_non_string_device_value_is_refused_not_raised():
    nodes, edges = _loader_graph(1)  # type: ignore[arg-type]
    assert len(_option_issues(validate_graph(nodes, edges))) == 1


def test_a_device_shaped_value_on_another_select_is_still_refused():
    """Only a ``device`` param with the device vocabulary is widened."""
    nodes = [_node("start", "Start"),
             _node("load", "ModelLoader", load_mode="cuda:1")]
    (issue,) = _option_issues(validate_graph(nodes, [_trigger("start", "load")]))
    assert issue.params["param"] == "load_mode"


def test_export_of_a_per_card_device_succeeds_and_compiles(four_cards):
    """The issue's reproduction: export with ``device="cuda:1"`` on four cards."""
    from app.core.codegen import generate_python

    nodes, edges = _loader_graph("cuda:1")
    script = generate_python(nodes, edges, name="per-card")
    compile(script, "<generated>", "exec")
    assert "'device': 'cuda:1'" in script


async def test_a_run_without_cuda_falls_back_like_resolve_device(
        no_accelerator, caplog):
    """Validation passes; the node then degrades to CPU with resolve_device's warning."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = "_device_param_validation_802.pt"
    try:
        ModelSaverNode().execute(
            {"model": nn.Linear(4, 2)},
            {"path": name, "save_mode": "full_model", "format": "pytorch"},
        )
        nodes, edges = _loader_graph("cuda:1", path=name, load_mode="full_model")
        assert validate_graph(nodes, edges) == []

        with caplog.at_level(logging.WARNING):
            results = await execute_graph(nodes, edges, context=ExecutionContext(
                device="cpu", weights_persistent=False, graph_id="issue-802"))
        model = results["load"]["model"]
        assert next(model.parameters()).device.type == "cpu"
        assert "CUDA not available, falling back to CPU" in caplog.text
    finally:
        (settings.MODELS_DIR / name).unlink(missing_ok=True)
