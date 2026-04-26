"""Tests for /api/execution/outputs REST endpoints."""

import pytest
import torch

from app.core.run_output_store import RunOutputStore
from app.main import app


@pytest.fixture(autouse=True)
def _ensure_store():
    """Make sure the app has a fresh store for each test (bypasses lifespan)."""
    app.state.run_output_store = RunOutputStore(max_runs=5)
    yield
    # Don't clear — subsequent tests may depend on lifespan-installed store


@pytest.mark.asyncio
async def test_get_tensor_full(test_client):
    store = app.state.run_output_store
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    await store.put("r1", "n1", "out", t)

    resp = await test_client.get("/api/execution/outputs/r1/n1/out")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "tensor"
    assert body["full_shape"] == [2, 3]
    assert body["sliced_shape"] == [2, 3]
    assert body["values"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.mark.asyncio
async def test_get_tensor_with_slice(test_client):
    store = app.state.run_output_store
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    await store.put("r1", "n1", "out", t)

    resp = await test_client.get("/api/execution/outputs/r1/n1/out?slice=0,:,:")
    assert resp.status_code == 200
    body = resp.json()
    assert body["full_shape"] == [2, 3, 4]
    assert body["sliced_shape"] == [3, 4]
    assert body["slice"] == "0,:,:"


@pytest.mark.asyncio
async def test_get_unknown_run_returns_404(test_client):
    resp = await test_client.get("/api/execution/outputs/missing/n1/out")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_unknown_node_returns_404(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "out", torch.zeros(2))

    resp = await test_client.get("/api/execution/outputs/r1/otherNode/out")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_bad_slice_returns_400(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "out", torch.zeros(2, 2))

    resp = await test_client.get("/api/execution/outputs/r1/n1/out?slice=abc")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_too_large_returns_413(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "out", torch.zeros(100, 100))  # 10000 elements

    resp = await test_client.get("/api/execution/outputs/r1/n1/out?max_elements=100")
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_steps_index_returns_metas_in_order(test_client):
    """`__steps_index` collects all `__step__N__meta` entries for a node."""
    store = app.state.run_output_store
    # Out of order on purpose to verify sorting by index.
    await store.put("r-steps", "att", "__step__1__meta", {
        "name": "softmax", "description": "norm", "scalars": {}, "tensor_keys": ["weights"],
    })
    await store.put("r-steps", "att", "__step__0__meta", {
        "name": "qkv", "description": "project", "scalars": {"d": 8.0}, "tensor_keys": ["Q", "K", "V"],
    })
    # Tensor entries should be ignored by the index endpoint.
    await store.put("r-steps", "att", "__step__0__Q", torch.zeros(2, 8))

    resp = await test_client.get("/api/execution/outputs/r-steps/att/__steps_index")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["index"] == 0
    assert body[0]["name"] == "qkv"
    assert body[0]["scalars"] == {"d": 8.0}
    assert body[0]["tensor_keys"] == ["Q", "K", "V"]
    assert body[1]["index"] == 1
    assert body[1]["name"] == "softmax"


@pytest.mark.asyncio
async def test_steps_index_returns_empty_list_for_node_without_steps(test_client):
    store = app.state.run_output_store
    await store.put("r-empty", "n1", "out", torch.zeros(2))
    resp = await test_client.get("/api/execution/outputs/r-empty/n1/__steps_index")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_steps_index_unknown_run_returns_404(test_client):
    resp = await test_client.get("/api/execution/outputs/missing-run/n1/__steps_index")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_grad_index_lists_port_and_weight_grads(test_client):
    """`__grad_index` enumerates per-port and per-weight gradients with health."""
    store = app.state.run_output_store
    await store.put("r-grad", "conv", "tensor__grad", torch.zeros(2, 2))
    await store.put("r-grad", "conv", "tensor__grad__meta", {
        "status": "healthy", "norm": 0.5, "mean": 0.1, "max": 0.3,
    })
    await store.put("r-grad", "conv", "__weight_grad__weight", torch.zeros(4, 1, 3, 3))
    await store.put("r-grad", "conv", "__weight_grad__weight__meta", {
        "status": "vanishing", "norm": 1e-9, "mean": 1e-10, "max": 1e-9,
    })
    await store.put("r-grad", "conv", "__weight_grad__bias", torch.zeros(4))
    # No __meta for bias — entry should still appear with health=None.

    resp = await test_client.get("/api/execution/outputs/r-grad/conv/__grad_index")
    assert resp.status_code == 200
    body = resp.json()
    by_kind_port = {(e["kind"], e["port"]): e for e in body}

    assert ("port", "tensor") in by_kind_port
    assert by_kind_port[("port", "tensor")]["health"]["status"] == "healthy"

    assert ("weight", "weight") in by_kind_port
    assert by_kind_port[("weight", "weight")]["health"]["status"] == "vanishing"

    assert ("weight", "bias") in by_kind_port
    assert by_kind_port[("weight", "bias")]["health"] is None


@pytest.mark.asyncio
async def test_grad_index_empty_for_node_with_no_grads(test_client):
    store = app.state.run_output_store
    await store.put("r-empty-grad", "conv", "tensor", torch.zeros(2))
    resp = await test_client.get("/api/execution/outputs/r-empty-grad/conv/__grad_index")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_run_outputs(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "a", torch.zeros(2, 2))
    await store.put("r1", "n1", "b", torch.ones(3))
    await store.put("r1", "n2", "c", 42)

    resp = await test_client.get("/api/execution/outputs/r1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    types = {(item["node_id"], item["port"]): item for item in body}
    assert types[("n1", "a")]["type"] == "tensor"
    assert types[("n1", "a")]["full_shape"] == [2, 2]
    assert types[("n1", "b")]["type"] == "tensor"
    assert types[("n2", "c")]["type"] == "scalar"


@pytest.mark.asyncio
async def test_list_unknown_run_returns_404(test_client):
    resp = await test_client.get("/api/execution/outputs/missing")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_run(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "out", torch.zeros(1))

    resp = await test_client.delete("/api/execution/outputs/r1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp = await test_client.get("/api/execution/outputs/r1/n1/out")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_unknown_returns_404(test_client):
    resp = await test_client.delete("/api/execution/outputs/missing")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_scalar_output(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "out", 3.14)

    resp = await test_client.get("/api/execution/outputs/r1/n1/out")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "scalar"
    assert body["value"] == 3.14


@pytest.mark.asyncio
async def test_string_output(test_client):
    store = app.state.run_output_store
    await store.put("r1", "n1", "out", "hello")

    resp = await test_client.get("/api/execution/outputs/r1/n1/out")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "string"
    assert body["value"] == "hello"
