import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

# Windows reads MIME types from the registry, which other installers (VS, IIS,
# antivirus, etc.) routinely clobber — `.js` often ends up as `text/plain`,
# which browsers refuse to execute under `<script type="module">` strict MIME
# checks, leaving the SPA blank. Force the correct types in-process so we
# never depend on whatever the host registry happens to say.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .api import (
    routes_custom_nodes,
    routes_examples,
    routes_execution_outputs,
    routes_execution_state,
    routes_graph,
    routes_images,
    routes_models,
    routes_nodes,
    routes_plugins,
    routes_presets,
    ws_execution,
)
from .config import settings
from .core.auth import (
    TOKEN_HEADER,
    constant_time_equals,
    host_is_allowed,
    init_allowed_hosts,
    session_token,
    write_token_file,
)
from .core.logging_config import setup_logging
from .core.node_registry import registry
from .core.node_state_store import NodeStateStore
from .core.plugin_loader import (
    MANIFEST_FILENAME,
    install_plugin_finder,
    iter_plugin_dirs,
    load_lockfile,
    rediscover_all,
)
from .core.preset_registry import preset_registry
from .core.run_output_store import RunOutputStore

logger = logging.getLogger(__name__)


# Mutating methods that require a valid session token. GET/HEAD/OPTIONS are
# unauthenticated reads (the spa_fallback path-traversal fix elsewhere prevents
# those from leaking arbitrary files), so they only need the Host check.
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Paths that intentionally skip auth: the bootstrap endpoint (frontend uses
# it to *get* the token), and Starlette's docs/openapi which are read-only
# anyway. The Host header check still applies.
_AUTH_EXEMPT_PATHS = frozenset({
    "/api/auth/bootstrap",
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(
        level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
        json_format=settings.LOG_JSON,
    )

    # Populate Host whitelist and persist the session token before any handler
    # can fire. Frontend bootstrap reads the token via /api/auth/bootstrap; CLI
    # tools (e.g. `cdui plugin install` → POST /api/plugins/reload) read it
    # from the file.
    init_allowed_hosts(settings.HOST, settings.PORT, extra=[
        urlparse(o).netloc for o in settings.CORS_ORIGINS
    ])
    token_path = write_token_file()
    logger.info("Session token written to %s", token_path)

    # Discover built-in nodes
    count = registry.discover(settings.NODES_DIR, "app.nodes")
    logger.info("Discovered %d built-in nodes", count)

    # Discover custom nodes
    custom_count = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    logger.info("Discovered %d custom nodes", custom_count)

    # Discover plugin nodes (per-user installed packs + built-in chapter packs)
    lockfile = load_lockfile()
    pairs = install_plugin_finder(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, lockfile
    )
    plugin_count = 0
    for nodes_dir, pkg_name in pairs:
        plugin_count += registry.discover(nodes_dir, pkg_name)
    logger.info(
        "Discovered %d plugin nodes from %d active plugin(s)", plugin_count, len(pairs)
    )

    for name in sorted(registry.nodes.keys()):
        logger.debug("  - %s (%s)", name, registry.nodes[name].CATEGORY)

    # Discover presets (built-in + per-plugin)
    preset_count = preset_registry.discover(settings.PRESETS_DIR, registry)
    for _plugin_id, plugin_dir in iter_plugin_dirs(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, lockfile
    ):
        preset_count += preset_registry.discover(plugin_dir / "presets", registry)
    logger.info("Discovered %d presets", preset_count)
    for name in sorted(preset_registry.presets.keys()):
        logger.debug("  * %s", name)

    # Mount each installed plugin's assets/ dir so the frontend can fetch
    # plugin-shipped CSVs / images at /plugins/<id>/assets/<file>.
    for plugin_id, plugin_dir in iter_plugin_dirs(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, lockfile
    ):
        assets = plugin_dir / "assets"
        if assets.is_dir():
            app.mount(
                f"/plugins/{plugin_id}/assets",
                StaticFiles(directory=assets),
                name=f"plugin_{plugin_id}_assets",
            )

    # In-memory store for captured per-run node outputs (Teaching Inspector)
    app.state.run_output_store = RunOutputStore(max_runs=20)

    # Persistent ``nn.Module`` instances per (graph, node, structure-hash).
    # Lifetime: server process. Survives Run clicks; lost on restart.
    app.state.node_state_store = NodeStateStore(max_modules=200)

    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


# ── Security middleware ───────────────────────────────────────────────
#
# Order matters: Starlette applies the *last-added* middleware *first*. We
# want the request to flow as:
#
#   incoming → host_guard → auth_guard → CORS → route handler
#
# Because middleware adds in reverse order, we add CORS first (innermost),
# auth second, host third (outermost).

# Innermost: CORS preflight + response headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", TOKEN_HEADER],
    expose_headers=[],
)


@app.middleware("http")
async def auth_guard(request: Request, call_next):
    """Require the session token for any mutating request under /api/.

    GET requests are allowed through (the static-file routes and read-only
    endpoints don't change anything on disk), with the explicit exemption
    that the bootstrap endpoint is also unauthenticated — that's where the
    frontend obtains the token in the first place.
    """
    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)
    if request.method not in _MUTATING_METHODS:
        return await call_next(request)
    if not path.startswith("/api/"):
        # WebSocket upgrades and static-file mounts handle their own auth.
        return await call_next(request)

    provided = request.headers.get(TOKEN_HEADER)
    if not constant_time_equals(provided, session_token()):
        return JSONResponse(
            status_code=403,
            content={"detail": f"Missing or invalid {TOKEN_HEADER} header"},
        )
    return await call_next(request)


@app.middleware("http")
async def host_guard(request: Request, call_next):
    """Reject requests whose ``Host`` header isn't in our whitelist.

    This is the layer that closes DNS-rebinding attacks: a browser tricked
    into resolving ``attacker.com`` to ``127.0.0.1`` still sends
    ``Host: attacker.com`` (the browser doesn't know about the rebinding).
    """
    host = request.headers.get("host", "")
    if not host_is_allowed(host):
        logger.warning("rejected request with Host=%r path=%s", host, request.url.path)
        return JSONResponse(
            status_code=421,
            content={"detail": "Misdirected Request (Host not allowed)"},
        )
    return await call_next(request)


# ── Routers ────────────────────────────────────────────────────────────
app.include_router(routes_nodes.router)
app.include_router(routes_examples.router)
app.include_router(routes_graph.router)
app.include_router(routes_presets.router)
app.include_router(routes_custom_nodes.router)
app.include_router(routes_plugins.router)
app.include_router(routes_models.router)
app.include_router(routes_images.router)
app.include_router(routes_execution_outputs.router)
app.include_router(routes_execution_state.router)
app.include_router(ws_execution.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "nodes_loaded": len(registry.nodes),
        "presets_loaded": len(preset_registry.presets),
    }


@app.get("/api/auth/bootstrap")
async def auth_bootstrap():
    """Hand the session token to the frontend.

    Reachable only when the Host header is whitelisted (the ``host_guard``
    middleware above rejects everything else). That stops DNS-rebinding and
    arbitrary-Origin browsers from grabbing the token, while keeping the
    legitimate same-origin / dev-Vite-proxy frontend working without any
    user-visible bootstrap step.
    """
    return {"token": session_token()}


@app.post("/api/nodes/reload")
async def reload_nodes():
    # Built-ins are immutable for the server lifetime — no point in paying
    # the reload tax. Custom nodes and plugins, however, may have been
    # edited on disk since the last load, so :func:`rediscover_all` force-
    # reloads them to pick up the changes. Plugin presets are also re-scanned.
    return rediscover_all(
        registry,
        preset_registry,
        nodes_dir=settings.NODES_DIR,
        custom_nodes_dir=settings.CUSTOM_NODES_DIR,
        presets_dir=settings.PRESETS_DIR,
        builtin_root=settings.PLUGINS_BUILTIN_DIR,
        user_root=settings.PLUGINS_USER_DIR,
    )


# Production mode: serve the pre-built frontend bundle. The catch-all is
# registered last so it never shadows /api/* or /ws/* routes (FastAPI matches
# in registration order). Skipped silently in dev when dist/ doesn't exist.
DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if (DIST_DIR / "index.html").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=DIST_DIR / "assets"),
        name="assets",
    )

    # Paths that should 404 instead of falling through to index.html so that
    # frontend fetch() errors stay distinguishable from "the SPA loaded".
    _NON_SPA_PREFIXES = ("api/", "ws/")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith(_NON_SPA_PREFIXES):
            raise HTTPException(status_code=404, detail="Not Found")
        # Defence against path traversal: resolve the candidate and confirm
        # it's still inside DIST_DIR. Browsers normalise ``..`` segments
        # before sending, but ``curl --path-as-is`` and other tools don't,
        # and a stray ``..`` would previously let local processes read any
        # file the server's UID could open.
        dist_resolved = DIST_DIR.resolve()
        candidate = (DIST_DIR / full_path).resolve()
        try:
            candidate.relative_to(dist_resolved)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
