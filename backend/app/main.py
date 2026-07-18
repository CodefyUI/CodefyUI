import asyncio
import logging
import mimetypes
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

# Windows reads MIME types from the registry, which other installers (VS, IIS,
# antivirus, etc.) routinely clobber — `.js` often ends up as `text/plain`,
# which browsers refuse to execute under `<script type="module">` strict MIME
# checks, leaving the SPA blank. Force the correct types in-process so we
# never depend on whatever the host registry happens to say.
if sys.platform == "win32":
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/svg+xml", ".svg")
    mimetypes.add_type("font/woff2", ".woff2")
    mimetypes.add_type("font/woff", ".woff")

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
    routes_apps,
    routes_custom_nodes,
    routes_examples,
    routes_execution_outputs,
    routes_execution_state,
    routes_graph,
    routes_graph_run,
    routes_images,
    routes_keys,
    routes_llm,
    routes_models,
    routes_nodes,
    routes_plugin_frontend,
    routes_plugins,
    routes_presets,
    routes_system,
    ws_execution,
)
from .config import settings
from .core.auth import (
    TOKEN_HEADER,
    allowed_hosts,
    constant_time_equals,
    host_is_allowed,
    init_allowed_hosts,
    local_interface_ips,
    session_token,
    write_token_file,
)
from .core.db import Database
from .core.logging_config import setup_logging
from .core.node_registry import registry
from .core.node_state_store import NodeStateStore
from .core import plugin_loader
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

# Router prefixes whose routes carry their OWN route-level auth dependency
# (exactly one each — enforced by tests/test_auth_drift.py). auth_guard
# skips them entirely; host_guard still applies to everything.
_AUTH_EXEMPT_PREFIXES = ("/api/apps", "/api/keys")


def _prefix_exempt(path: str) -> bool:
    """Exact-or-slash prefix match: '/api/apps' and '/api/apps/x' are
    exempt, '/api/appsfoo' is not. Footgun (spec Section 8): a future
    bare route at an exempt prefix matches ``path == p`` and is ALSO
    exempt — the drift test, not this middleware, is the guarantee.
    """
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in _AUTH_EXEMPT_PREFIXES
    )


def _extra_allowed_host_entries() -> list[str]:
    """Extra Host-whitelist entries beyond the bind address.

    ``EXTRA_ALLOWED_HOSTS`` (comma-separated str — see config.py) split
    and stripped; plus, when binding a wildcard (which init_allowed_hosts
    deliberately skips), each concrete interface IP as ``{ip}:{port}``.
    """
    entries = [
        entry.strip()
        for entry in settings.EXTRA_ALLOWED_HOSTS.split(",")
        if entry.strip()
    ]
    if settings.HOST in ("0.0.0.0", "::"):
        entries.extend(
            f"{ip}:{settings.PORT}" for ip in local_interface_ips()
        )
    return entries


def _has_port(entry: str) -> bool:
    """True when a whitelist *entry* carries an explicit port —
    ``host:port`` or ``[ipv6]:port`` — i.e. it is directly usable as the
    authority of a printable URL.

    Structural on purpose. Both a naive ``":" in entry`` check and a
    parse-based one (``ipaddress.ip_address``) mis-classify IPv6 forms:
    the former keeps the portless ``"::1"``, and the latter only excluded
    ``"::1:8000"`` by accident — four-digit ports happen to parse as a
    hextet while five-digit ports (10000-65535) do not, so the malformed
    line came back on high ports. Bracket structure is unambiguous:
    bracketed entries have a port iff ``"]:"`` appears; unbracketed
    entries (hostname/IPv4 only — ``init_allowed_hosts`` never suffixes a
    port onto unbracketed IPv6) have a port iff they contain exactly one
    colon.
    """
    if entry.startswith("["):
        return "]:" in entry
    return entry.count(":") == 1


def _reachable_urls() -> list[str]:
    """Sorted ``http://host:port`` lines worth printing at startup: every
    whitelisted entry that carries a real port (spec Section 9's startup
    transparency log)."""
    return sorted(
        f"http://{h}" for h in allowed_hosts() if _has_port(h)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(
        level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
        json_format=settings.LOG_JSON,
    )

    # Project .env: execution-time secrets only, os.environ.setdefault
    # semantics, loaded before node/plugin discovery. CODEFYUI_* config keys
    # here are IGNORED (settings already materialized at import) -- spec 7.3.
    if settings.PROJECT_DIR is not None:
        from .core.dotenv import load_dotenv_file
        env_applied = load_dotenv_file(settings.PROJECT_DIR / ".env")
        if env_applied:
            # Log the COUNT only, never the values.
            logger.info("Loaded %d value(s) from project .env", env_applied)

    # Populate Host whitelist and persist the session token before any handler
    # can fire. Frontend bootstrap reads the token via /api/auth/bootstrap; CLI
    # tools (e.g. `cdui plugin install` → POST /api/plugins/reload) read it
    # from the file.
    init_allowed_hosts(settings.HOST, settings.PORT, extra=[
        urlparse(o).netloc for o in settings.CORS_ORIGINS
    ] + _extra_allowed_host_entries())
    if settings.HOST not in ("127.0.0.1", "localhost", "::1"):
        # Startup transparency for non-loopback binds (spec Section 9):
        # print the effective whitelist and the reachable URLs. Anyone who
        # can reach the port controls the instance — the docs carry the
        # full framing; this log makes the exposure visible at start.
        logger.warning(
            "Serving on a non-loopback bind (%s:%s) — anyone who can "
            "reach this port controls the instance; use only on trusted "
            "networks.",
            settings.HOST, settings.PORT,
        )
        logger.info("Host whitelist: %s", ", ".join(sorted(allowed_hosts())))
        logger.info("Reachable at: %s", ", ".join(_reachable_urls()))
    token_path = write_token_file()
    logger.info("Session token written to %s", token_path)

    # Discover built-in nodes
    count = registry.discover(settings.NODES_DIR, "app.nodes")
    logger.info("Discovered %d built-in nodes", count)

    # Discover custom nodes
    custom_count = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    logger.info("Discovered %d custom nodes", custom_count)

    # Discover plugin nodes (per-user installed packs + built-in direction packs)
    lockfile = load_lockfile()
    pairs = install_plugin_finder(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), lockfile
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
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), lockfile
    ):
        preset_count += preset_registry.discover(plugin_dir / "presets", registry)
    logger.info("Discovered %d presets", preset_count)
    for name in sorted(preset_registry.presets.keys()):
        logger.debug("  * %s", name)

    # ── Project transparency (spec 7.4) ────────────────────────────────
    if settings.PROJECT_DIR is not None:
        from .core.project import (
            check_pin_issues,
            git_provenance,
            read_project_manifest,
        )
        commit, dirty = git_provenance(settings.PROJECT_DIR)
        if commit is None:
            logger.info("Project: %s (not a repo)", settings.PROJECT_DIR)
        else:
            logger.info("Project: %s (git %s%s)", settings.PROJECT_DIR,
                        commit[:7], " dirty" if dirty else "")
        # Shared stale-pin rule (issue #85): same classification the CLI's
        # `cdui project validate` consumes.
        issues = check_pin_issues(
            read_project_manifest(settings.PROJECT_DIR), lockfile)
        stale = sorted(i.plugin_id for i in issues if i.kind != "malformed")
        malformed = sorted(i.plugin_id for i in issues if i.kind == "malformed")
        if stale:
            # ONE warning; no auto-install at startup (spec 7.4).
            logger.warning(
                "Project plugin pins missing/mismatched: %s -- run "
                "`cdui project restore`", ", ".join(stale))
        if malformed:
            # Warn-and-skip: a non-table pin cannot be enforced or restored.
            logger.warning(
                "Project manifest has malformed plugin pins (skipped): %s -- "
                "each pin must be a table like "
                "{ url = \"...\", ref = \"...\", sha = \"...\" }",
                ", ".join(malformed))

    # Mount each installed plugin's assets/ dir so the frontend can fetch
    # plugin-shipped CSVs / images at /plugins/<id>/assets/<file>.
    for plugin_id, plugin_dir in iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(), lockfile
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

    # ── Stage-2 storage: published apps, API keys, run records ─────────
    # Routes access it via getattr(app.state, "db", None) and 503 when
    # absent (routes_execution_outputs precedent) — the lifespan does not
    # run under httpx ASGITransport, so tests set app.state.db directly.
    db = Database(settings.DB_PATH)
    await asyncio.to_thread(db.connect)
    app.state.db = db
    logger.info("SQLite storage ready at %s", settings.DB_PATH)
    # Startup retention prune (no-op at the default RUNS_RETENTION_DAYS=0;
    # prune_runs itself logs loudly when it deletes anything).
    await db.prune_runs(settings.RUNS_RETENTION_DAYS, force=True)
    # Per-slug invoke serialization (spec Decision I); entries are pruned
    # on app delete.
    app.state.app_locks = {}

    yield

    # Release the SQLite handle so `cdui stop` on Windows frees the DB and
    # its WAL sidecar files (spec Section 13, Windows file locking).
    db.close()
    app.state.db = None


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
    allow_headers=["Content-Type", TOKEN_HEADER, "Authorization"],
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
    if _prefix_exempt(path):
        # /api/apps + /api/keys routes each declare exactly one explicit
        # auth dependency (require_session_token / require_api_key /
        # require_api_key_or_session) — enforced by the drift test.
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
app.include_router(routes_graph_run.router)
app.include_router(routes_presets.router)
app.include_router(routes_custom_nodes.router)
app.include_router(routes_plugins.router)
app.include_router(routes_plugin_frontend.router)
app.include_router(routes_models.router)
app.include_router(routes_images.router)
app.include_router(routes_execution_outputs.router)
app.include_router(routes_execution_state.router)
app.include_router(routes_system.router)
app.include_router(routes_llm.router)
app.include_router(routes_apps.router)
app.include_router(routes_keys.router)
app.include_router(ws_execution.router)


@app.get("/api/health")
async def health():
    body = {
        "status": "ok",
        "nodes_loaded": len(registry.nodes),
        "presets_loaded": len(preset_registry.presets),
    }
    if settings.PROJECT_DIR is not None:
        # Additive (spec ID4), project mode ONLY: the refactor guard requires
        # non-project responses to stay byte-for-byte identical, so this key
        # is omitted entirely (not even null) when PROJECT_DIR is unset.
        # The frontend (Tasks 12/13) and the Task 15 publish CLI's mismatch
        # refusal both read this to detect project mode + identity.
        body["project"] = str(settings.PROJECT_DIR)
    return body


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
        builtin_root=plugin_loader.plugins_builtin_root(),
        user_root=plugin_loader.plugins_user_root(),
    )


# Production mode: serve the pre-built frontend bundle. The catch-all is
# registered last so it never shadows /api/* or /ws/* routes (FastAPI matches
# in registration order). Skipped silently in dev when dist/ doesn't exist.
DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if (DIST_DIR / "index.html").exists():
    # Vite emits content-hashed asset filenames (e.g. index-LKCMvfbh.js), so a
    # given URL's bytes never change — cache them aggressively & immutably.
    class _ImmutableStaticFiles(StaticFiles):
        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return resp

    app.mount(
        "/assets",
        _ImmutableStaticFiles(directory=DIST_DIR / "assets"),
        name="assets",
    )

    # index.html, by contrast, must NEVER be cached: it's the only file that
    # references the current hashed bundles by name. If a browser serves a
    # stale index.html after an upgrade, it loads an OLD bundle against the NEW
    # backend — and that old bundle predates the session-token handshake, so
    # every WebSocket / mutating request is rejected 403 ("loads but the Run
    # button does nothing"). `no-cache` forces revalidation on every load so
    # the document always matches the running server.
    _INDEX_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    def _index_response() -> FileResponse:
        return FileResponse(DIST_DIR / "index.html", headers=_INDEX_HEADERS)

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
        return _index_response()
