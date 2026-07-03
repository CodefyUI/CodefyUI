"""Per-image pixel budget (spec Decision H2): enforced header-only in
_decode_image between Image.open and img.load(), so decode memory is
bounded on BOTH run routes.

Deliberately NOT run with warnings-as-errors: PIL emits
DecompressionBombWarning above ~89.5 MP, and above ~179 MP PIL's own
DecompressionBombError fires FIRST inside Image.open (front-stop) — in
that range tests match the CODE only, never the reason text.
"""

from __future__ import annotations

import base64
import io

import pytest

from app.config import Settings
from app.core import api_contract
from app.core.api_contract import InputCoercionError


def _png_base64(width: int, height: int, mode: str = "RGB") -> str:
    from PIL import Image

    img = Image.new(mode, (width, height),
                    color=0 if mode == "1" else (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bilevel_png_base64(width: int, height: int) -> str:
    """Small-bytes / huge-dimensions PNG: 1-bit mode keeps the in-test
    allocation at ~w*h/8 bytes and the file KB-scale."""
    return _png_base64(width, height, mode="1")


# ── unit level ───────────────────────────────────────────────────────────


def test_default_budget_env_overridable(monkeypatch):
    assert Settings().MAX_IMAGE_PIXELS == 25_000_000
    monkeypatch.setenv("CODEFYUI_MAX_IMAGE_PIXELS", "123")
    assert Settings().MAX_IMAGE_PIXELS == 123


def test_decode_image_rejects_over_budget_before_load(monkeypatch):
    monkeypatch.setattr("app.config.settings.MAX_IMAGE_PIXELS", 7)
    with pytest.raises(InputCoercionError) as exc_info:
        api_contract.coerce_input(_png_base64(4, 2), "image")
    # Our message names dimensions and product: "(WxH=N)".
    assert "image exceeds MAX_IMAGE_PIXELS (4x2=8)" in exc_info.value.reason


def test_decode_image_under_budget_still_decodes(monkeypatch):
    monkeypatch.setattr("app.config.settings.MAX_IMAGE_PIXELS", 9)
    tensor = api_contract.coerce_input(_png_base64(4, 2), "image")
    assert tuple(tensor.shape) == (3, 2, 4)   # (C, H, W)


def test_real_default_budget_rejects_100_megapixel_png():
    # No monkeypatch: a genuine 100 MP crafted PNG (success criterion 5)
    # against the real 25 MP default. The 10000x10000 bilevel PNG
    # allocates ~12.5 MB in-test and rejects BEFORE decode pays the
    # full-size cost. (PIL emits DecompressionBombWarning above ~89.5 MP
    # — expected, harmless, and exactly why this path never runs
    # warnings-as-errors.)
    with pytest.raises(InputCoercionError) as exc_info:
        api_contract.coerce_input(_bilevel_png_base64(10000, 10000), "image")
    assert "exceeds MAX_IMAGE_PIXELS (10000x10000=100000000)" \
        in exc_info.value.reason


# ── endpoint level: BOTH routes (decode is shared) ───────────────────────


def _image_graph(name: str = "img-graph") -> dict:
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 200, "y": 0},
             "data": {"params": {
                 "name": "photo", "type": "image", "required": True,
                 "default": "", "description": "",
             }}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 400, "y": 0},
             "data": {"params": {"name": "echo", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }


@pytest.fixture(autouse=True)
def _graphs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_editor_run_rejects_100mp_png_422(test_client):
    # Small-bytes / huge-dimensions PNG at the REAL default budget
    # (success criterion 5, editor route).
    resp = await test_client.post("/api/graph/save", json=_image_graph())
    assert resp.status_code == 200
    resp = await test_client.post(
        "/api/graph/run/img-graph",
        json={"inputs": {"photo": _bilevel_png_base64(10000, 10000)}},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    assert any("MAX_IMAGE_PIXELS" in d["reason"]
               for d in body["error"]["details"])


@pytest.mark.asyncio
async def test_invoke_rejects_100mp_png_422(test_client, app_db):
    # Same crafted PNG on the published-invoke route (decode is shared).
    key = (await test_client.post(
        "/api/keys", json={"name": "px"})).json()
    resp = await test_client.post("/api/graph/save",
                                  json=_image_graph("img-pub"))
    assert resp.status_code == 200
    resp = await test_client.post(
        "/api/apps/img-app/publish",
        json={"graph": "img-pub", "create": True},
    )
    assert resp.status_code == 200, resp.text
    resp = await test_client.post(
        "/api/apps/img-app/invoke",
        json={"inputs": {"photo": _bilevel_png_base64(10000, 10000)}},
        headers={"Authorization": f"Bearer {key['token']}"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_pil_front_stop_range_matches_code_never_reason(
    test_client, monkeypatch,
):
    # Above ~179 MP PIL's DecompressionBombError fires FIRST inside
    # Image.open — simulate it exactly like the Stage-1 regression test.
    # Clients and tests match on the CODE only in that range: the reason
    # text is PIL's, not ours.
    import PIL.Image

    resp = await test_client.post("/api/graph/save", json=_image_graph())
    assert resp.status_code == 200

    def _boom(*_args, **_kwargs):
        raise PIL.Image.DecompressionBombError("boom")

    monkeypatch.setattr(PIL.Image, "open", _boom)
    resp = await test_client.post(
        "/api/graph/run/img-graph",
        json={"inputs": {"photo": _png_base64(4, 2)}},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"
    # Deliberately NO assertion on the reason text (spec Section 6.3).
