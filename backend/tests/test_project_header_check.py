"""Phase 4A, R3: a write from a window left on another project is refused.

The check is one function in ``main.py`` (``_project_mismatch``, run by
``auth_guard`` before anything else), not code in each route. This module
enumerates every mutating route the app serves and asserts each one
answers 409 ``project_changed`` to a stale ``X-CodefyUI-Project``: a new
route is covered by construction, and if one ever is not, this goes red.
"""

from __future__ import annotations

import json
import re

from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.api import ws_execution
from app.config import settings
from app.core import projects
from app.core.auth import TOKEN_HEADER, TOKEN_QUERY_PARAM, session_token
from app.main import PROJECT_HEADER, app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
#: Writes the check leaves alone on purpose: switching projects itself.
EXEMPT_PREFIXES = ("/api/projects",)
_PARAM = re.compile(r"\{[^}]+\}")


def write_routes() -> list[tuple[str, str]]:
    found = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api/"):
            continue
        for method in route.methods & MUTATING:
            found.add((method, route.path))
    return sorted(found)


def _concrete(path: str) -> str:
    return _PARAM.sub("x", path)


def test_the_enumeration_finds_the_write_routes():
    routes = write_routes()
    # Sanity: the enumeration is not silently empty or partial.
    assert len(routes) > 80, len(routes)
    paths = {path for _, path in routes}
    for expected in ("/api/graph/save", "/api/runs", "/api/git/commit",
                     "/api/projects/open"):
        assert expected in paths, expected


async def test_every_write_route_checks_the_project():
    stale = "0" * 16
    assert stale != projects.project_id(settings.PROJECT_DIR)
    unprotected = []
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as http:
        for method, path in write_routes():
            if any(path == p or path.startswith(p + "/")
                   for p in EXEMPT_PREFIXES):
                continue
            resp = await http.request(
                method, _concrete(path),
                headers={TOKEN_HEADER: session_token(), PROJECT_HEADER: stale},
                content=b"{}")
            body = resp.json() if resp.headers.get(
                "content-type", "").startswith("application/json") else {}
            detail = body.get("detail") if isinstance(body, dict) else None
            if resp.status_code != 409 or not isinstance(detail, dict) \
                    or detail.get("code") != "project_changed":
                unprotected.append((method, path, resp.status_code))
    assert unprotected == []


async def test_the_check_runs_before_authentication_and_validation():
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as http:
        # No token, a body that would fail validation: still 409 first, so
        # a stale window learns the real reason.
        resp = await http.post("/api/graph/save", content=b"not json",
                               headers={PROJECT_HEADER: "0" * 16})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "project_changed"
    assert resp.json()["detail"]["details"]["current"] \
        == projects.project_id(settings.PROJECT_DIR)


async def test_a_matching_or_absent_header_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "GRAPHS_DIR", tmp_path)
    graph = {"name": "header-ok", "nodes": [], "edges": []}
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        current = projects.project_id(settings.PROJECT_DIR)
        resp = await http.post("/api/graph/save", json=graph,
                               headers={PROJECT_HEADER: current})
        assert resp.status_code == 200, resp.text
        resp = await http.post("/api/graph/save", json=graph)
        assert resp.status_code == 200, resp.text
        # Reads are never checked.
        resp = await http.get("/api/graph/list",
                              headers={PROJECT_HEADER: "0" * 16})
        assert resp.status_code == 200


def test_the_project_id_is_header_safe(tmp_path):
    odd = tmp_path / "專案 with spaces"
    odd.mkdir()
    pid = projects.project_id(odd)
    assert re.fullmatch(r"[0-9a-f]{16}", pid)
    assert projects.project_id(None) == "flat"
    assert projects.project_id(odd) == projects.project_id(str(odd) + "/")


# ── the execution socket ──────────────────────────────────────────────────


def _ws_client() -> AsyncClient:
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE_URL)


async def test_a_stale_window_cannot_run_over_the_socket():
    url = (f"/ws/execution?{TOKEN_QUERY_PARAM}={session_token()}"
           f"&project={'0' * 16}")
    async with _ws_client() as client, aconnect_ws(url, client) as ws:
        await ws.send_text(json.dumps({"action": "execute", "nodes": [],
                                       "edges": []}))
        message = json.loads(await ws.receive_text())
    assert message["type"] == "execution_error"
    assert message["code"] == "project_changed"


async def test_a_switch_is_broadcast_to_every_open_window():
    url = f"/ws/execution?{TOKEN_QUERY_PARAM}={session_token()}"
    async with _ws_client() as client, aconnect_ws(url, client) as first, \
            aconnect_ws(url, client) as second:
        # Both sockets are registered once their first frame round-trips.
        for ws in (first, second):
            await ws.send_text(json.dumps({"action": "clear_cache"}))
            assert json.loads(await ws.receive_text())["type"] == "cache_cleared"
        told = await ws_execution.broadcast_project_changed(
            {"type": "project_changed", "project": "/p", "project_id": "abc"})
        assert told == 2
        for ws in (first, second):
            message = json.loads(await ws.receive_text())
            assert message == {"type": "project_changed", "project": "/p",
                               "project_id": "abc"}
    assert not ws_execution._SOCKETS
