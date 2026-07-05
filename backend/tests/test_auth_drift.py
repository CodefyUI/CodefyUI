"""Auth-drift guarantee (spec Decision E): every route under the
auth_guard-exempt prefixes declares exactly ONE of the three Stage-2 auth
dependencies. The middleware exemption alone is NEVER trusted — a future
bare route matching an exempt prefix silently skips the middleware, and
only this test catches it arriving without route-level auth.

Two complementary layers, both deliberately restricted to STABLE public
surfaces (FastAPI 0.139 stopped flattening included routers into
``app.routes``, which silently emptied a private-internals walk):

1. Structural: every route on the two exempt routers declares exactly one
   auth dependency, read from ``route.dependant`` on OUR own
   ``APIRouter.routes`` (verified present on 0.135 and 0.139 alike — the
   0.139 change only removed the app-level flattening). ``dependant``
   is required here because parameter-style ``Depends`` (e.g. invoke's
   non-raising key dependency) never appear in ``route.dependencies``.
2. Behavioral app-level net: enumerate the LIVE app's paths via
   ``app.openapi()`` (public API, flattens included routers on every
   version) and hit every exempt path/method anonymously — each one must
   reject with 401/403. A FUTURE router mounted under ``/api/apps`` or
   ``/api/keys`` whose routes forget route-level auth shows up here as a
   non-401/403 anonymous response, with no test update needed.
"""

from __future__ import annotations

from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.api import routes_apps, routes_keys
from app.config import settings
from app.core.api_keys import (
    require_api_key,
    require_api_key_or_session,
    require_session_token,
)
from app.main import _AUTH_EXEMPT_PREFIXES, _prefix_exempt, app

_AUTH_DEPS = {require_api_key, require_api_key_or_session,
              require_session_token}


def test_exempt_prefixes_are_the_spec_pair():
    assert _AUTH_EXEMPT_PREFIXES == ("/api/apps", "/api/keys")


def test_prefix_matching_is_exact_or_slash():
    assert _prefix_exempt("/api/apps")
    assert _prefix_exempt("/api/apps/some-slug/invoke")
    assert _prefix_exempt("/api/keys")
    assert _prefix_exempt("/api/keys/3/revoke")
    assert not _prefix_exempt("/api/appsfoo")   # sibling: no false match
    assert not _prefix_exempt("/api/keysx")
    assert not _prefix_exempt("/api/app")


def _exempt_router_api_routes() -> list[APIRoute]:
    """All APIRoutes on the two exempt routers — OUR objects, so the
    ``routes`` attribute is stable regardless of how FastAPI represents
    them inside ``app.routes`` after ``include_router``."""
    return [
        route
        for router in (routes_apps.router, routes_keys.router)
        for route in router.routes
        if isinstance(route, APIRoute)
    ]


def test_every_exempt_route_declares_exactly_one_auth_dependency():
    routes = _exempt_router_api_routes()
    # Sanity: an empty walk would make the loop below vacuously pass —
    # fail loudly instead if the traversal itself is broken.
    assert len(routes) >= 10, "router walk is broken (expected the full apps+keys surface)"
    for route in routes:
        calls = [
            d.call for d in route.dependant.dependencies
            if d.call in _AUTH_DEPS
        ]
        assert len(calls) == 1, (
            f"{sorted(route.methods)} {route.path} declares "
            f"{len(calls)} auth dependencies — every route under an "
            "exempt prefix MUST declare exactly one of "
            "require_session_token / require_api_key / "
            "require_api_key_or_session"
        )


def _exempt_openapi_operations() -> list[tuple[str, str]]:
    """Every (path, method) the LIVE app serves under an exempt prefix,
    enumerated through ``app.openapi()`` so included routers are covered
    on every FastAPI version."""
    return [
        (path, method)
        for path, ops in app.openapi()["paths"].items()
        for method in ops
        if _prefix_exempt(path)
    ]


async def test_every_exempt_route_rejects_anonymous_requests():
    operations = _exempt_openapi_operations()
    assert len(operations) >= 10, "openapi walk is broken"
    assert any(p.startswith("/api/apps") for p, _ in operations)
    assert any(p.startswith("/api/keys") for p, _ in operations)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{settings.PORT}"
    ) as client:
        for path, method in operations:
            concrete = path.replace("{slug}", "drift-probe")
            for param in ("{key_id}", "{version}", "{run_id}"):
                concrete = concrete.replace(param, "1")
            response = await client.request(method.upper(), concrete)
            assert response.status_code in (401, 403), (
                f"{method.upper()} {concrete} answered anonymous request "
                f"with {response.status_code} — every route under an "
                "exempt prefix must reject missing credentials with "
                "401/403 (did a new route forget its auth dependency?)"
            )


def test_exempt_router_prefixes_are_actually_exempt():
    # Belt-and-braces: each currently-mounted exempt router's prefix
    # really is covered by the tuple.
    for router in (routes_apps.router, routes_keys.router):
        assert _prefix_exempt(router.prefix), router.prefix
