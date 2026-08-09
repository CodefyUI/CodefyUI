"""The one request-body ceiling: core.body_limit (core#265, core#242).

The headline test in this file is ``test_chunked_body_over_the_limit_is_refused``
and everything else supports it. Every guard this replaces compared
``Content-Length`` to a limit, and a chunked request declares no
``Content-Length`` — so the old guards were not merely missing from four
routes, they were skipped in full on all three routes that had them. A suite
that does not send a chunked body has not tested the bug.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from app.core.body_limit import (
    MULTIPART_ENVELOPE_ALLOWANCE,
    UPLOAD_PATHS,
    limit_for_path,
)

# No module-level asyncio mark: pyproject sets asyncio_mode = "auto", and a
# blanket mark would also land on the two synchronous unit tests below.


async def _chunks(total: int, chunk: int = 32) -> AsyncIterator[bytes]:
    """Yield *total* bytes in *chunk*-sized pieces.

    Handing httpx an async iterator (rather than a bytes object) is what makes
    it send ``Transfer-Encoding: chunked`` with no ``Content-Length`` at all.
    ``chunk`` stays deliberately small in the tests below so that no single
    piece is over the limit — only the RUNNING TOTAL is, which is the property
    a Content-Length comparison cannot have.
    """
    sent = 0
    while sent < total:
        n = min(chunk, total - sent)
        yield b"a" * n
        sent += n


def _graph() -> dict:
    return {
        "name": "cap-probe",
        "nodes": [
            {"id": "start", "type": "Start",
             "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        ],
        "edges": [],
    }


# ── the hole every previous guard had ────────────────────────────────────


async def test_chunked_body_over_the_limit_is_refused(test_client, monkeypatch):
    """A chunked request with NO Content-Length that exceeds the limit is
    refused. This is the case every guard this replaces missed.

    Note the shape of the send: 320 bytes in 32-byte pieces against a 100-byte
    ceiling. No individual piece is over the limit, so nothing but a running
    total can refuse this — and there is no Content-Length to consult even if
    something wanted to.
    """
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        "/api/graph/save",
        content=_chunks(320, chunk=32),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413, resp.text

    # Prove the request really did exercise the hole rather than quietly
    # arriving with a length header that an old-style guard could have read.
    sent = resp.request.headers
    assert "content-length" not in {k.lower() for k in sent}
    assert sent.get("transfer-encoding") == "chunked"


async def test_chunked_body_under_the_limit_still_succeeds(test_client):
    """The cap is a ceiling, not a gate: chunked transfer is not itself
    suspicious. Pinned because the cheapest way to pass the test above is to
    refuse every request that has no Content-Length."""
    resp = await test_client.post("/api/graph/save", json=_graph())
    assert resp.status_code == 200, resp.text

    body = json.dumps(_graph()).encode()

    async def _stream() -> AsyncIterator[bytes]:
        for i in range(0, len(body), 16):
            yield body[i:i + 16]

    resp = await test_client.post(
        "/api/graph/save",
        content=_stream(),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200, resp.text


async def test_chunked_body_is_refused_on_the_run_route_with_its_envelope(
    test_client, monkeypatch, tmp_path,
):
    """The chunked hole is closed on /api/graph/run too, and the 413 still
    arrives as the 9-key envelope that route promises on every response."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    assert (await test_client.post(
        "/api/graph/save", json=_graph())).status_code == 200
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)

    resp = await test_client.post(
        "/api/graph/run/cap-probe",
        content=_chunks(320, chunk=32),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert set(body.keys()) == {
        "status", "run_id", "graph", "app", "version",
        "device", "outputs", "error", "timing",
    }
    assert body["error"]["code"] == "payload_too_large"
    assert body["graph"] == "cap-probe"
    assert body["run_id"]


async def test_chunked_body_is_refused_on_submit(test_client, monkeypatch):
    """POST /api/runs keeps its {"detail": ...} shape, chunked or not."""
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        "/api/runs",
        content=_chunks(320, chunk=32),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413, resp.text
    assert "max 100" in resp.json()["detail"]


# ── the four routes that had no cap at all (core#265) ────────────────────


@pytest.mark.parametrize("path", [
    "/api/graph/save",
    "/api/graph/validate",
    "/api/graph/export",
    "/api/presets/create",
])
async def test_previously_uncapped_route_refuses_an_oversized_body(
    test_client, monkeypatch, path,
):
    """Each route that took a pydantic body with no size check of its own.

    The payload is deliberately not valid JSON for any of these schemas: a
    413 proves the refusal happens while the body is being READ, before
    FastAPI parses or validates it. A 422 here would mean the bytes were paid
    for first.
    """
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        path,
        content=b"{" + b"x" * 400,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413, f"{path}: {resp.text}"
    assert "max 100" in resp.json()["detail"]


@pytest.mark.parametrize("path", [
    "/api/graph/save",
    "/api/graph/validate",
    "/api/graph/export",
    "/api/presets/create",
])
async def test_previously_uncapped_route_refuses_a_chunked_oversized_body(
    test_client, monkeypatch, path,
):
    """The same four routes, with no Content-Length to lean on."""
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        path,
        content=_chunks(400, chunk=32),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413, f"{path}: {resp.text}"


# ── the hot path (the editor calls these on ordinary interaction) ────────


async def test_hot_path_requests_are_unaffected(test_client, tmp_path, monkeypatch):
    """/api/graph/save and /api/graph/validate at the PRODUCTION default.

    No monkeypatched ceiling anywhere in this test: an ordinary graph must
    still round-trip under the real 64 MB limit, because these two routes fire
    on ordinary canvas interaction and a cap that costs them anything is a cap
    that will be removed.
    """
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    save = await test_client.post("/api/graph/save", json=_graph())
    assert save.status_code == 200, save.text
    check = await test_client.post("/api/graph/validate", json=_graph())
    assert check.status_code == 200, check.text


# ── two limits, resolved per path ────────────────────────────────────────


async def test_a_declared_oversize_body_is_refused_without_reading_a_byte(
    monkeypatch,
):
    """The Content-Length short-circuit, pinned by the property that makes it
    worth having.

    The running total alone would already refuse this request — but only after
    reading up to the limit, which at the 64 MB default means buffering 64 MB
    to reject a body that announced itself as larger. When a client declares a
    length that is already over, the right number of bytes to read is zero.

    Asserted against the ASGI callable directly because that is where "was
    receive ever awaited" is observable at all; through an HTTP client both
    paths look identical.
    """
    from app.core.body_limit import BodySizeLimitMiddleware, RequestBodyTooLarge

    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b"a" * 400, "more_body": False}

    async def app(scope, recv, send):
        await recv()

    async def send(message):  # pragma: no cover - never reached
        raise AssertionError("no response should be sent")

    scope = {
        "type": "http",
        "path": "/api/graph/save",
        "method": "POST",
        "headers": [(b"content-length", b"400")],
    }
    with pytest.raises(RequestBodyTooLarge) as excinfo:
        await BodySizeLimitMiddleware(app)(scope, receive, send)

    assert reads == 0, "the body was read despite an over-limit Content-Length"
    assert excinfo.value.status_code == 413
    assert excinfo.value.declared == 400


@pytest.mark.parametrize("header", [
    b"not-a-number",   # unparseable
    b"-1",             # negative
    b"0",              # understated to nothing
    b"1 2",            # two values smuggled into one header
])
async def test_a_lying_content_length_cannot_widen_the_cap(monkeypatch, header):
    """The invariant that makes the header safe to consult at all.

    ``Content-Length`` is attacker-controlled, so the only sound way to read it
    is one that can cause an EARLIER refusal and never a later one. Every way
    of lying here costs the sender the short-circuit and nothing else, because
    the running total is unconditional. A guard that trusted the header — or
    that treated garbage as zero — would be walked past by sending one.
    """
    from app.core.body_limit import BodySizeLimitMiddleware, RequestBodyTooLarge

    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)

    async def receive():
        return {"type": "http.request", "body": b"a" * 400, "more_body": False}

    async def app(scope, recv, send):
        await recv()

    async def send(message):  # pragma: no cover - never reached
        raise AssertionError("no response should be sent")

    scope = {
        "type": "http",
        "path": "/api/graph/save",
        "method": "POST",
        "headers": [(b"content-length", header)],
    }
    with pytest.raises(RequestBodyTooLarge) as excinfo:
        await BodySizeLimitMiddleware(app)(scope, receive, send)
    # declared is None -> it was the running total that refused this, which is
    # the whole point: the header bought the sender nothing.
    assert excinfo.value.declared is None


def test_limit_for_path_distinguishes_uploads_from_graphs(monkeypatch):
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 111)
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 999)
    assert limit_for_path("/api/graph/save") == 111
    assert limit_for_path("/api/runs") == 111
    for upload in UPLOAD_PATHS:
        assert limit_for_path(upload) == 999 + MULTIPART_ENVELOPE_ALLOWANCE
    # Exact match only: a sibling path must not inherit the upload ceiling.
    assert limit_for_path("/api/files/upload-batch") == 111


def test_limits_are_read_per_request_not_captured_at_import(monkeypatch):
    """The middleware must not snapshot settings at construction, or an env
    override (and every test's monkeypatch) would be ignored."""
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 7)
    assert limit_for_path("/api/graph/save") == 7
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 8)
    assert limit_for_path("/api/graph/save") == 8


@pytest.fixture
def data_files_dir(tmp_path, monkeypatch):
    d = tmp_path / "files"
    d.mkdir()
    monkeypatch.setattr("app.config.settings.DATA_FILES_DIR", d)
    return d


async def test_upload_uses_the_upload_ceiling_not_the_graph_ceiling(
    test_client, data_files_dir, monkeypatch,
):
    """The point of having two limits.

    MAX_RUN_BODY_BYTES is pinned to 100 bytes here — a single global number
    would refuse this upload outright. It succeeds because the upload routes
    resolve to MAX_UPLOAD_SIZE instead.
    """
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 50_000)
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("probe.csv", b"c" * 20_000, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["size"] == 20_000


async def test_upload_at_exactly_the_documented_file_limit_succeeds(
    test_client, data_files_dir, monkeypatch,
):
    """The boundary case the envelope allowance exists for.

    MAX_UPLOAD_SIZE is documented as the largest FILE you may upload. What
    crosses the wire is a multipart body — boundary lines, a
    Content-Disposition header, the filename — so a body ceiling set to
    exactly the file limit would make a file of exactly the documented size
    un-uploadable. It must not.
    """
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 4096)
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("exact.csv", b"c" * 4096, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["size"] == 4096
    assert (data_files_dir / "exact.csv").stat().st_size == 4096


async def test_upload_one_byte_over_the_file_limit_is_refused(
    test_client, data_files_dir, monkeypatch,
):
    """Just past the documented file limit and still well inside the body
    ceiling: the ROUTE refuses it, and says so in its own words."""
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 4096)
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("over.csv", b"c" * 4097, "text/csv")},
    )
    assert resp.status_code == 413, resp.text
    assert resp.json()["detail"] == "File too large"
    assert not (data_files_dir / "over.csv").exists()


async def test_upload_past_the_body_ceiling_is_refused_before_it_is_read(
    test_client, data_files_dir, monkeypatch,
):
    """Far past the limit: the MIDDLEWARE refuses it, so the route never
    buffers it. This is core#242 — the old code read the whole thing first
    and nothing bounded a request larger still."""
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 4096)
    over = 4096 + MULTIPART_ENVELOPE_ALLOWANCE + 1024
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("huge.csv", b"c" * over, "text/csv")},
    )
    assert resp.status_code == 413, resp.text
    # The middleware's wording, not the route's — proof of which layer refused.
    assert resp.json()["detail"] != "File too large"
    assert "max" in resp.json()["detail"]
    assert not (data_files_dir / "huge.csv").exists()


_BOUNDARY = "cdui-test-boundary"


async def _multipart_stream(payload_bytes: int,
                            chunk: int = 8192) -> AsyncIterator[bytes]:
    """A REAL multipart body for one .csv field, yielded in pieces.

    Genuine multipart rather than filler: python-multipart parses
    incrementally and rejects garbage long before enough of it arrives, so a
    stream of nonsense would answer 400 and prove nothing about the ceiling.
    """
    head = (
        f"--{_BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="huge.csv"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode()
    tail = f"\r\n--{_BOUNDARY}--\r\n".encode()
    yield head
    sent = 0
    while sent < payload_bytes:
        n = min(chunk, payload_bytes - sent)
        yield b"c" * n
        sent += n
    yield tail


async def test_chunked_upload_past_the_body_ceiling_is_refused(
    test_client, data_files_dir, monkeypatch,
):
    """The upload routes get the streaming refusal too, not just the
    Content-Length one — a valid multipart body with no declared length is
    still cut off the moment the running total passes the ceiling."""
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 1024)
    over = 1024 + MULTIPART_ENVELOPE_ALLOWANCE + 4096
    resp = await test_client.post(
        "/api/files/upload",
        content=_multipart_stream(over),
        headers={"content-type":
                 f"multipart/form-data; boundary={_BOUNDARY}"},
    )
    assert resp.status_code == 413, resp.text
    assert "content-length" not in {k.lower() for k in resp.request.headers}
    assert not (data_files_dir / "huge.csv").exists()


async def test_a_chunked_upload_inside_the_ceiling_still_lands(
    test_client, data_files_dir, monkeypatch,
):
    """The same streaming path, under the ceiling: the file must arrive
    intact. Pinned so the test above cannot be satisfied by refusing every
    streamed upload."""
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 50_000)
    resp = await test_client.post(
        "/api/files/upload",
        content=_multipart_stream(20_000),
        headers={"content-type":
                 f"multipart/form-data; boundary={_BOUNDARY}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["size"] == 20_000
    assert (data_files_dir / "huge.csv").stat().st_size == 20_000


async def test_the_cap_is_not_swallowed_by_fastapis_body_parser(
    test_client, monkeypatch, data_files_dir,
):
    """A regression guard on why RequestBodyTooLarge subclasses HTTPException.

    FastAPI wraps body parsing in ``except Exception`` and rewrites whatever
    it catches into ``400 There was an error parsing the body`` — every
    exception except HTTPException, which it re-raises. Any other base class
    would turn this cap into a silent 400 on every route with a declared body,
    which is exactly what a multipart route reports for a malformed body. So
    the two outcomes must stay distinguishable: too-big is 413, malformed is
    400.
    """
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 1024)
    over = 1024 + MULTIPART_ENVELOPE_ALLOWANCE + 4096
    too_big = await test_client.post(
        "/api/files/upload",
        content=_multipart_stream(over),
        headers={"content-type":
                 f"multipart/form-data; boundary={_BOUNDARY}"},
    )
    assert too_big.status_code == 413, too_big.text
    assert "parsing the body" not in too_big.text


# ── composition with the auth stack (must not reorder anything) ──────────


def test_the_cap_sits_inside_both_guards_and_outside_cors():
    """Pin the middleware ORDER structurally, not just in a comment.

    Outermost first. The cap must stay inside both BaseHTTPMiddleware guards:
    an exception raised from an upstream ``receive`` does not survive that
    class's task-group plumbing, and an over-limit request ends up answering
    400 "There was an error parsing the body" instead of 413. Moving it out
    reddens 17 tests, but every one of those failures reads as a puzzle; this
    one names the rule.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware

    from app.core.body_limit import BodySizeLimitMiddleware
    from app.main import app

    layers = [
        (m.cls, getattr(m.kwargs.get("dispatch"), "__name__", None))
        for m in app.user_middleware
    ]
    assert layers == [
        (BaseHTTPMiddleware, "host_guard"),
        (BaseHTTPMiddleware, "auth_guard"),
        (BodySizeLimitMiddleware, None),
        (CORSMiddleware, None),
    ], layers


async def test_a_disallowed_host_still_wins_over_an_oversized_body(
    test_client, monkeypatch,
):
    """host_guard is OUTSIDE the cap and must stay there.

    An oversized body with a forged Host answers 421, not 413. Beyond
    ordering, this is the cheaper refusal: host_guard never reads the body, so
    nothing is counted and nothing is buffered.
    """
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        "/api/graph/save",
        content=b"x" * 400,
        headers={"content-type": "application/json", "host": "evil.example"},
    )
    assert resp.status_code == 421, resp.text


async def test_a_missing_session_token_still_wins_over_an_oversized_body(
    test_client, monkeypatch,
):
    """auth_guard is OUTSIDE the cap and must stay there: an unauthenticated
    oversized request answers 403, and is not told that it was also too big."""
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        "/api/graph/save",
        content=b"x" * 400,
        headers={"content-type": "application/json",
                 "x-codefyui-token": "wrong"},
    )
    assert resp.status_code == 403, resp.text


async def test_the_413_carries_cors_headers(test_client, monkeypatch):
    """The cap sits INSIDE CORSMiddleware so a cross-origin caller can
    actually read the status. Without the header the browser reports an
    opaque network error and the 413 is invisible."""
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 100)
    resp = await test_client.post(
        "/api/graph/save",
        content=b"x" * 400,
        headers={"content-type": "application/json",
                 "origin": "http://localhost:5173"},
    )
    assert resp.status_code == 413
    assert resp.headers.get("access-control-allow-origin") == \
        "http://localhost:5173"


async def test_a_body_nobody_reads_is_not_refused(test_client, monkeypatch):
    """The stated rule: the cap bounds what the application READS.

    POST /api/nodes/reload declares no body. An oversized one is never read,
    never buffered and therefore never refused. This is not an oversight — it
    is what keeps every route's own auth check ahead of the cap, including
    /api/apps/{slug}/invoke's 401, which fires before that route reads its
    body.
    """
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 10)
    resp = await test_client.post("/api/nodes/reload", content=b"x" * 400)
    assert resp.status_code == 200, resp.text


async def test_the_websocket_is_not_touched_by_the_http_cap(monkeypatch):
    """Non-HTTP scopes pass straight through, and this is deliberate.

    The cap is an HTTP body limit; a WebSocket has no request body to count.
    A message far over MAX_RUN_BODY_BYTES therefore still reaches the socket's
    own handler and gets that handler's answer — here, the malformed-JSON
    error — rather than an HTTP 413 the client could not receive anyway.

    What bounds a WS message is the transport: uvicorn's ``ws_max_size``
    (16 MB by default) is enforced by the ``websockets`` library while it
    assembles fragments, which is the same "count as it arrives" property
    this module gives HTTP. Making that ceiling an explicit CodefyUI setting
    rather than an inherited uvicorn default is filed as follow-up work.
    """
    from httpx import AsyncClient
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    from app.config import settings
    from app.core.auth import TOKEN_QUERY_PARAM, session_token
    from app.main import app

    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 10)
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=f"http://127.0.0.1:{settings.PORT}",
    ) as client:
        path = f"/ws/execution?{TOKEN_QUERY_PARAM}={session_token()}"
        async with aconnect_ws(path, client) as ws:
            await ws.send_text("z" * 4000)
            reply = json.loads(await ws.receive_text())
            assert reply["type"] == "error"
            assert "malformed" in reply["error"]
