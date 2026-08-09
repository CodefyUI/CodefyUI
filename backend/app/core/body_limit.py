"""One request-body ceiling for the whole HTTP surface, counted as bytes arrive.

Why this exists (core#265, core#242)
------------------------------------
Three routes used to cap their own bodies by reading ``Content-Length`` and
comparing it to ``MAX_RUN_BODY_BYTES`` before touching the body. Four more
routes (``/api/graph/save``, ``/api/graph/validate``, ``/api/graph/export``,
``/api/presets/create``) had no cap at all, and the four upload routes read the
entire file into memory and *then* compared it to ``MAX_UPLOAD_SIZE``.

The deeper problem was not the coverage gap, it was that a ``Content-Length``
comparison answers nothing about a **chunked** request: ``Transfer-Encoding:
chunked`` declares no length, so every one of those guards was skipped in full
by any client that chose to chunk. They were advisory, not enforcing.

Counting bytes as they arrive fixes both at once. There is exactly one place a
body can enter the application — the ASGI ``receive`` callable — so wrapping it
bounds every route, present and future, chunked or not.

Prior art, and why it is not used verbatim
------------------------------------------
Starlette 1.6 ships ``starlette.middleware.body_limit.RequestBodyLimitMiddleware``
and it is the right *design*: wrap ``receive``, keep a running total, short-
circuit on ``Content-Length`` when it is offered. Its design is adopted here.
Its implementation is not, for two measured reasons (both reproduced against
this app's middleware stack before deciding):

1. It also wraps ``send`` and, when ``Content-Length`` exceeds the limit, it
   **replaces whatever response the application produced** with a plain-text
   413 — including a 401. ``POST /api/apps/{slug}/invoke`` checks its API key
   before it reads the body precisely so a bad key answers 401; under that
   middleware an oversized request with a bad key answers 413 instead. Silently
   converting an auth failure into a size failure is not a trade this repo can
   make (see ``core/auth.py``'s threat model).
2. Its 413 body changes shape depending on the transport: ``text/plain`` when
   ``Content-Length`` was present (the ``send`` path above), JSON when the
   request was chunked (the ``receive`` path). Two shapes for one condition,
   and neither is the 9-key run envelope that ``/api/graph/run/{name}`` and
   ``/api/apps/{slug}/invoke`` promise on every response — a promise their
   per-app OpenAPI document hands to third-party client generators.

So the rule here is narrower and stated positively: **the limit bounds what the
application reads.** A body nobody reads is never buffered and is never
refused, which is what keeps every route's own auth and routing decisions
ahead of this one.

The two limits
--------------
``MAX_RUN_BODY_BYTES`` (64 MB) is a *graph* ceiling and applies to everything by
default. ``MAX_UPLOAD_SIZE`` (500 MB) is a *file* ceiling and applies to the
four upload routes. A single global number could only be one of those: 64 MB
everywhere breaks model uploads, 500 MB everywhere makes the graph limit
decorative. Both are read at request time, never captured at construction, so
an env override or a test's monkeypatch takes effect.

Note the upload ceiling is deliberately the file limit *plus* a small envelope
allowance. ``MAX_UPLOAD_SIZE`` is documented as the largest **file** you may
upload, but what crosses the wire is a multipart body: boundaries, a
``Content-Disposition`` header and the filename ride along with it. Capping the
body at exactly the file limit would make a file of exactly the documented size
un-uploadable. The routes keep their own ``len(content) > MAX_UPLOAD_SIZE``
check because it measures a different quantity — the file, not the body — and
it is the one that enforces the documented number exactly. This middleware is
what stops that check from having to buffer an unbounded request first.
"""

from __future__ import annotations

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..config import settings

# Routes whose body is a file upload rather than a graph, and which therefore
# answer to MAX_UPLOAD_SIZE. Exact paths, not prefixes: a frozenset lookup is
# the cheapest possible test on a hot path, and an exact match cannot be
# widened by accident the way a prefix can (a future `/api/files/uploads-list`
# would silently inherit the 500 MB ceiling under `startswith`).
UPLOAD_PATHS = frozenset({
    "/api/files/upload",
    "/api/images/upload",
    "/api/models/upload",
    "/api/custom-nodes/upload",
})

# Headroom over MAX_UPLOAD_SIZE for the multipart envelope around the file:
# boundary lines, the Content-Disposition header and the filename. Measured at
# a few hundred bytes for a single-file POST; 64 KB is generous enough that a
# pathological filename cannot eat it and small enough (0.013% of the 500 MB
# default) that it is not a limit of its own.
MULTIPART_ENVELOPE_ALLOWANCE = 64 * 1024


class RequestBodyTooLarge(HTTPException):
    """413, raised from the wrapped ``receive`` the instant a body outgrows
    its ceiling.

    An ``HTTPException`` subclass on purpose, and this is load-bearing rather
    than stylistic. FastAPI wraps body parsing in ``except Exception`` and
    converts anything it catches into ``400 There was an error parsing the
    body`` — except ``HTTPException``, which it re-raises untouched with the
    comment "If a middleware raises an HTTPException, it should be raised
    again" (``fastapi/routing.py``). Any other base class would have this cap
    silently answering 400 on every route that declares a pydantic body.

    Being an ``HTTPException`` is also what routes the error to Starlette's
    ``ExceptionMiddleware``, which sits *inside* CORS — so the 413 comes back
    with the CORS headers a browser needs in order to read it.
    """

    def __init__(self, *, limit: int, declared: int | None) -> None:
        self.limit = limit
        self.declared = declared
        if declared is None:
            # Streaming refusal: the total is unknown and unknowable — that is
            # the entire point of the chunked case. Report the ceiling only,
            # and never a running total, which would just be "the limit plus
            # one chunk" dressed up as a measurement.
            detail = f"request body exceeds max {limit} bytes"
        else:
            detail = f"request body is {declared} bytes (max {limit})"
        super().__init__(status_code=413, detail=detail)


def limit_for_path(path: str) -> int:
    """Body ceiling in bytes for *path*. Read at request time, deliberately."""
    if path in UPLOAD_PATHS:
        return settings.MAX_UPLOAD_SIZE + MULTIPART_ENVELOPE_ALLOWANCE
    return settings.MAX_RUN_BODY_BYTES


class BodySizeLimitMiddleware:
    """Pure-ASGI middleware bounding every HTTP request body.

    Pure ASGI rather than ``BaseHTTPMiddleware``: the job is to substitute one
    callable (``receive``) before the application reads it, which is precisely
    what the raw interface is for. ``BaseHTTPMiddleware`` would run the whole
    downstream app in a task group to achieve the same substitution, which is
    both more machinery and more failure modes than a wrapped coroutine.

    Non-HTTP scopes are passed straight through. WebSocket frames are a
    different transport and are bounded by uvicorn's ``ws_max_size`` (16 MB by
    default), enforced by the ``websockets`` library while it assembles
    fragments — the same "count as it arrives" property this class gives HTTP.

    Where it is registered matters and is documented at the call site in
    ``main.py``: it must sit INSIDE the ``BaseHTTPMiddleware`` guards, because
    an exception raised from an upstream ``receive`` does not survive that
    class's task-group plumbing and ends up rewritten as a 400.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        limit = limit_for_path(scope["path"])
        declared = _declared_length(scope)
        seen = 0

        async def receive_capped() -> Message:
            nonlocal seen
            # Cheap path first: when the client offered a length and it is
            # already over, refuse without reading a byte. Raised HERE and not
            # in __call__ because this middleware sits OUTSIDE
            # ExceptionMiddleware — an exception raised before the app is
            # entered escapes upward to ServerErrorMiddleware and becomes a
            # 500. Raised from inside receive it surfaces within the request,
            # where ExceptionMiddleware -- and so main.py's
            # RequestBodyTooLarge handler -- can render it.
            if declared is not None and declared > limit:
                raise RequestBodyTooLarge(limit=limit, declared=declared)
            message = await receive()
            # The enforcing path: a chunked request declares no length, so the
            # running total is the only thing that can refuse it.
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise RequestBodyTooLarge(limit=limit, declared=None)
            return message

        await self.app(scope, receive_capped, send)


def _declared_length(scope: Scope) -> int | None:
    """``Content-Length`` as an int, or None when absent or unparseable.

    A malformed header reads as "no declared length" rather than as zero:
    treating garbage as a measurement is how a cap gets talked past.

    The invariant that makes this header safe to consult at all: it can only
    ever cause an EARLIER refusal, never a later one. The running total is
    unconditional, so every way of lying here — a negative value, a zero, a
    duplicated header, an unparseable one — costs the sender the short-circuit
    and nothing else. Nothing read from this header can widen the limit.
    """
    for key, value in scope.get("headers", ()):
        if key == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
