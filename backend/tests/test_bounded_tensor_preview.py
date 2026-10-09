"""#640: a captured tensor too big to send whole previews as a bounded slice.

The Inspector used to retry a 413 with a fixed ``0,:,:``, which still overflows
an image batch and cannot index a rank-1 or rank-2 tensor. ``?preview=true``
lets the server, which knows the shape, pick the slice. These run real Torch
tensors through the real app and route, at the client's preview limit.

``fixtures/bounded_preview_contract.json`` is GENERATED here (set
``UPDATE_CONTRACT_FIXTURES=1``) and read by
``frontend/src/components/InspectorPanel/boundedPreview.contract.test.tsx``,
so the frontend is tested against the responses this route actually sends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from app.api.routes_execution_outputs import _parse_slice
from app.core.run_output_store import RunOutputStore
from app.main import app

#: The limit the Inspector asks previews for (``PREVIEW_MAX_ELEMENTS``).
LIMIT = 65536
FIXTURE = Path(__file__).parent / "fixtures" / "bounded_preview_contract.json"


@pytest.fixture(autouse=True)
def _ensure_store():
    app.state.run_output_store = RunOutputStore(max_runs=5)
    yield


async def _preview(test_client, shape, *, limit=LIMIT, dtype=torch.float32):
    tensor = torch.arange(int(torch.Size(shape).numel()), dtype=dtype).reshape(shape)
    await app.state.run_output_store.put("r1", "n1", "out", tensor)
    resp = await test_client.get(
        f"/api/execution/outputs/r1/n1/out?preview=true&max_elements={limit}"
    )
    return tensor, resp


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "expected_slice", "expected_shape"),
    [
        # The four shapes in the issue.
        ([16, 3, 224, 224], "0,0", [224, 224]),
        ([2, 8, 3, 224, 224], "0,0,0", [224, 224]),
        ([4, 70000], "0,0:65536", [65536]),
        ([100000], "0:65536", [65536]),
        # Each rank, keeping as many leading rows as fit.
        ([70000, 4], "0:16384", [16384, 4]),
        ([3, 300, 300], "0,0:218", [218, 300]),
        ([2, 1, 100, 1024], "0,0,0:64", [64, 1024]),
        ([2, 2, 20, 64, 64], "0,0,0:16", [16, 64, 64]),
        # Pinned to index 0 where one entry is all that fits.
        ([2, 4, 128, 128], "0", [4, 128, 128]),
        # A last dimension wider than the limit, at rank 3, 4 and 5.
        ([2, 3, 70000], "0,0,0:65536", [65536]),
        ([2, 2, 2, 70000], "0,0,0,0:65536", [65536]),
        ([2, 1, 1, 1, 70000], "0,0,0,0,0:65536", [65536]),
    ],
)
async def test_an_oversized_tensor_previews_a_bounded_slice(
    test_client, shape, expected_slice, expected_shape,
):
    tensor, resp = await _preview(test_client, shape)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "tensor"
    assert body["full_shape"] == shape
    assert body["slice"] == expected_slice
    assert body["sliced_shape"] == expected_shape
    assert body["truncated"] is True
    shown = tensor[_parse_slice(expected_slice)]
    assert shown.numel() <= LIMIT
    # The values are the slice's own, from the front of the tensor.
    assert body["values"] == shown.tolist()
    assert body["min"] == float(shown.min())
    assert body["max"] == float(shown.max())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", [[16, 3, 224, 224], [2, 8, 3, 224, 224], [4, 70000], [100000]],
)
async def test_the_old_fixed_retry_fails_on_the_issue_shapes(test_client, shape):
    """The ``0,:,:`` retry this replaces: a second 413, or a 400."""
    tensor = torch.zeros(shape)
    await app.state.run_output_store.put("r1", "n1", "out", tensor)
    resp = await test_client.get(
        f"/api/execution/outputs/r1/n1/out?slice=0,:,:&max_elements={LIMIT}"
    )
    assert resp.status_code in (400, 413)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", [[], [5], [3, 4], [2, 3, 4], [2, 2, 3, 4], [1, 2, 2, 3, 4]])
async def test_a_tensor_within_the_limit_is_sent_whole(test_client, shape):
    tensor, resp = await _preview(test_client, shape)

    assert resp.status_code == 200
    body = resp.json()
    assert body["full_shape"] == shape
    assert body["sliced_shape"] == shape
    assert body["slice"] == ""
    assert body["truncated"] is False
    assert body["values"] == tensor.tolist()


@pytest.mark.asyncio
async def test_an_integer_tensor_keeps_its_exact_values(test_client):
    tensor, resp = await _preview(test_client, [100000], dtype=torch.int64)
    body = resp.json()
    assert body["values"] == list(range(LIMIT))
    assert body["min"] == 0
    assert body["max"] == LIMIT - 1


@pytest.mark.asyncio
async def test_scalars_are_unchanged_by_preview(test_client):
    await app.state.run_output_store.put("r1", "n1", "out", 3.5)
    resp = await test_client.get("/api/execution/outputs/r1/n1/out?preview=true")
    assert resp.json()["type"] == "scalar"
    assert resp.json()["value"] == 3.5


@pytest.mark.asyncio
async def test_preview_and_an_explicit_slice_are_exclusive(test_client):
    await app.state.run_output_store.put("r1", "n1", "out", torch.zeros(2, 2))
    resp = await test_client.get(
        "/api/execution/outputs/r1/n1/out?preview=true&slice=0,:"
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_the_query_form_previews_an_inner_node(test_client):
    """A node inside a block is read through ``/value?node_id=`` (#621)."""
    await app.state.run_output_store.put("r1", "blk/mul", "out", torch.zeros(100000))
    resp = await test_client.get(
        "/api/execution/outputs/r1/value",
        params={"node_id": "blk/mul", "port": "out", "preview": "true",
                "max_elements": LIMIT},
    )
    assert resp.status_code == 200
    assert resp.json()["slice"] == "0:65536"
    assert resp.json()["truncated"] is True


@pytest.mark.asyncio
async def test_the_frontend_contract_fixture_matches_the_route(test_client):
    """The refusal and the preview the frontend's contract test replays.

    Small numbers (limit 12) so the fixture stays readable; the walk is the
    same one the tests above run at the real limit.
    """
    tensor = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
    await app.state.run_output_store.put("run1", "a", "out", tensor)
    refused = await test_client.get("/api/execution/outputs/run1/a/out?max_elements=12")
    preview = await test_client.get(
        "/api/execution/outputs/run1/a/out?preview=true&max_elements=12"
    )
    assert refused.status_code == 413
    assert preview.status_code == 200
    live = {
        "refused": {"status": refused.status_code, "body": refused.json()},
        "preview": {"status": preview.status_code, "body": preview.json()},
    }
    if os.environ.get("UPDATE_CONTRACT_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(live, indent=2) + "\n", encoding="utf-8")
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == live
