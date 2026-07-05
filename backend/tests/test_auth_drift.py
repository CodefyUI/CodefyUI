"""Auth-drift guarantee (spec Decision E): every route under the
auth_guard-exempt prefixes declares exactly ONE of the three Stage-2 auth
dependencies. The middleware exemption alone is NEVER trusted — a future
bare route matching an exempt prefix silently skips the middleware, and
only this test catches it arriving without route-level auth.

Walks ``app.routes`` directly (not a hardcoded router list) so a FUTURE
router mounted under an exempt prefix is covered automatically — the
whole point of this test is to not depend on anyone remembering to add
it here.
"""

from __future__ import annotations

from app.api import routes_apps, routes_keys
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


def _exempt_app_routes() -> list:
    """Every route object in the LIVE app whose path falls under an
    auth-exempt prefix. Walking ``app.routes`` (rather than a fixed list
    of routers) means a future router mounted under ``/api/apps`` or
    ``/api/keys`` is covered automatically, with no test update needed.
    Filtered to ``dependant``-bearing routes (real FastAPI ``APIRoute``s)
    so static mounts / the SPA catch-all can never be mistaken for one.
    """
    return [
        route for route in app.routes
        if hasattr(route, "dependant") and _prefix_exempt(route.path)
    ]


def test_every_exempt_route_declares_exactly_one_auth_dependency():
    routes = _exempt_app_routes()
    # Sanity: an empty walk would make the loop below vacuously pass —
    # fail loudly instead if the traversal itself is broken.
    assert routes, "no exempt routes found — app.routes walk is broken"
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


def test_exempt_router_prefixes_are_actually_exempt():
    # Belt-and-braces: each currently-mounted exempt router's prefix
    # really is covered by the tuple.
    for router in (routes_apps.router, routes_keys.router):
        assert _prefix_exempt(router.prefix), router.prefix
