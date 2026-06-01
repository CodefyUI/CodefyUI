"""SPA cache-header policy.

Regression guard for the "loads but the Run button does nothing" bug: a
browser that cached a stale ``index.html`` after an upgrade loads an OLD JS
bundle against the NEW backend. The old bundle predates the session-token
handshake, so every WebSocket / mutating request is rejected 403.

The contract that prevents this:
  * ``index.html`` is served ``no-cache`` so the browser always revalidates
    and picks up the current hashed bundle names.
  * hashed ``/assets/*`` files are content-addressed, so they're served
    ``immutable`` with a long max-age.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_pytestmark_reason = "frontend/dist not built; SPA routes are not registered"
pytestmark = pytest.mark.skipif(
    not (_DIST / "index.html").exists(), reason=_pytestmark_reason
)


def test_index_html_is_not_cached():
    with TestClient(app, base_url="http://localhost") as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        cc = resp.headers.get("cache-control", "")
        assert "no-cache" in cc or "no-store" in cc, cc


def test_spa_fallback_route_is_not_cached():
    # An arbitrary client-side route also serves index.html and must not cache.
    with TestClient(app, base_url="http://localhost") as client:
        resp = client.get("/some/deep/spa/route")
        assert resp.status_code == 200
        cc = resp.headers.get("cache-control", "")
        assert "no-cache" in cc or "no-store" in cc, cc


def test_hashed_assets_are_immutable():
    asset = next((_DIST / "assets").glob("*.js"), None)
    if asset is None:
        pytest.skip("no built JS asset to check")
    with TestClient(app, base_url="http://localhost") as client:
        resp = client.get(f"/assets/{asset.name}")
        assert resp.status_code == 200
        cc = resp.headers.get("cache-control", "")
        assert "immutable" in cc and "max-age=" in cc, cc
