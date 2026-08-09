"""What an unhandled ``/api`` path answers once the SPA bundle is present.

#285. ``backend/app/main.py`` registers a catch-all that serves the built
frontend, and it does so ONLY when ``frontend/dist/index.html`` exists. That
single ``if`` split the project's behaviour in two:

  * a developer who has never run ``pnpm build`` — and CI, whose checkout
    never has — exercises the branch where the catch-all does not exist;
  * every real installation exercises the other one, because the no-Node
    release path ships ``frontend/dist``.

Four traversal assertions in ``test_api_data_files.py`` were red in the second
world and green in the first, deterministically, for as long as the catch-all
has existed. Nothing caught it because CI only ever ran the first.

These tests close that gap from the test side: :func:`app.main.mount_spa`
takes the app and the dist directory as arguments, so the production wiring
can be pointed at a ``tmp_path`` and asserted on in any checkout, built or
not. The companion half is a CI job that runs the whole suite against a real
``pnpm build`` — see ``.github/workflows/backend-test.yml``.

Nothing here touches the repository's own ``frontend/dist``. It is only ever
read, never created and never removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import compile_path

from app.main import NON_SPA_PREFIXES, mount_spa

#: The #285 payloads: multi-segment, so no ``{filename}`` route can match
#: them, which is exactly what makes them fall through to the catch-all.
TRAVERSAL_INPUTS = [
    "../../etc/passwd",
    "subdir/../../../secret.csv",
    "/etc/passwd",
    "a/../../../../../../tmp/x.csv",
]


def _verbatim(filename: str) -> str:
    """Encode *filename* so the router receives it character for character.

    Same helper, same reason as ``test_api_data_files.py``: httpx strips dot
    segments before sending, so an un-encoded ``..`` would never reach the
    route and the test would pass with the fix reverted.
    """
    return filename.replace(".", "%2e")


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    """A stand-in for a built frontend — never the repository's own."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>stub</title>")
    (tmp_path / "assets" / "index-stub.js").write_text("// stub\n")
    return tmp_path


@pytest.fixture
def client(dist: Path) -> TestClient:
    """An app shaped like production: API routes first, catch-all last.

    ``/api/files/{filename}`` is the real route from ``routes_data_files``,
    reproduced with the same converter (no ``:path``), because the whole bug
    is about what happens to a request that route CANNOT match.
    """
    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.delete("/api/files/{filename}")
    async def delete_file(filename: str):
        return {"deleted": filename}

    mount_spa(app, dist)
    return TestClient(app, base_url="http://localhost")


# ── the #285 regression ──────────────────────────────────────────────────


@pytest.mark.parametrize("filename", TRAVERSAL_INPUTS)
def test_delete_to_an_unmatched_api_path_is_404_not_405(client, filename):
    """The exact failure: 405 once ``dist/`` exists, 404 without it.

    Starlette does not stop at the first route whose PATH matches. A route
    that matches the path but not the method is recorded as a PARTIAL match
    and, if nothing FULL-matches, it is what answers — with 405. So a
    GET-only catch-all turned every unmatched multi-segment ``/api`` DELETE
    into "Method Not Allowed", which says the resource exists and the verb is
    wrong. Both halves of that are false.
    """
    resp = client.request("DELETE", f"/api/files/{_verbatim(filename)}")
    assert resp.status_code == 404, resp.text


def test_an_unmatched_api_get_is_404_rather_than_the_index_document(client):
    """A GET typo must not come back as 200 text/html.

    The frontend's ``fetch()`` calls parse JSON; an ``index.html`` body with
    a 200 on it turns "that endpoint does not exist" into a syntax error
    somewhere unrelated.
    """
    resp = client.get("/api/definitely-not-a-route")
    assert resp.status_code == 404, resp.text
    assert "text/html" not in resp.headers.get("content-type", "")


def test_the_ws_prefix_is_excluded_on_the_same_terms(client):
    """``/ws/`` is in the same tuple and gets the same answer.

    WebSocket routes only match a ``websocket`` scope, so a plain HTTP GET to
    one has no handler at all — and must not be handed the SPA document.
    """
    assert client.get("/ws/execution").status_code == 404


def test_a_real_route_with_the_wrong_method_still_answers_405(client):
    """The over-fix this pattern was chosen to avoid.

    An all-methods ``/api/{rest:path}`` 404 route registered ahead of the
    catch-all would also fix the test above — by FULL-matching every request
    to every ``/api`` path, including a POST to a real GET endpoint, whose
    honest answer is 405. Excluding the prefixes from the CATCH-ALL's pattern
    leaves the API layer's own routing intact.
    """
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/health").status_code == 405


# ── the SPA still behaves like a SPA ─────────────────────────────────────


def test_client_side_routes_still_serve_the_index_document(client):
    """The catch-all's actual job, unchanged."""
    resp = client.get("/some/deep/spa/route")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "no-cache" in resp.headers.get("cache-control", "")


def test_a_real_asset_is_served_immutable(client):
    assert client.get("/assets/index-stub.js").status_code == 200
    cc = client.get("/assets/index-stub.js").headers.get("cache-control", "")
    assert "immutable" in cc and "max-age=" in cc, cc


def test_traversal_out_of_the_dist_directory_is_still_refused(client, tmp_path):
    """The guard the catch-all already had, kept honest by a real target."""
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"hunter2")

    resp = client.get("/%2e%2e/secret%2etxt")
    assert resp.status_code == 400, resp.text
    assert b"hunter2" not in resp.content
    assert secret.exists()


# ── the exclusion lives in the pattern, not in the handler ───────────────


@pytest.mark.parametrize(
    "path",
    ["/api/files/x/y", "/api/health", "/ws/execution", "/api/"],
)
def test_the_route_pattern_itself_refuses_api_and_ws(path):
    """Where the exclusion has to live, asserted on the compiled regex.

    A ``startswith`` check inside the handler cannot help: the 405 is
    produced by the ROUTER, before any handler runs. The route has to not
    match at all, which means the exclusion belongs in the path pattern —
    and this is the assertion that notices if it moves back.
    """
    regex, _fmt, _convertors = compile_path("/{full_path:spa_path}")
    assert regex.match(path) is None, f"{path} still matches the SPA catch-all"


@pytest.mark.parametrize("path", ["/", "/graphs/42", "/settings", "/apidocs"])
def test_the_route_pattern_still_accepts_everything_else(path):
    """Including ``/apidocs``, which starts with ``api`` but is not ``api/``."""
    regex, _fmt, _convertors = compile_path("/{full_path:spa_path}")
    assert regex.match(path) is not None, f"{path} no longer reaches the SPA"


def test_both_halves_are_generated_from_one_tuple():
    """The handler's guard and the pattern cannot disagree.

    Two independent spellings of the same list is how one of them ends up
    stale. Both are derived from ``NON_SPA_PREFIXES``, so this pins the tuple
    rather than either copy of it.
    """
    assert NON_SPA_PREFIXES == ("api/", "ws/")
    regex, _fmt, _convertors = compile_path("/{full_path:spa_path}")
    for prefix in NON_SPA_PREFIXES:
        assert regex.match(f"/{prefix}anything") is None, prefix
