import asyncio
import logging
import mimetypes
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.convertors import Convertor, register_url_convertor

from .api import (
    routes_apps,
    routes_custom_nodes,
    routes_examples,
    routes_data_files,
    routes_execution_outputs,
    routes_execution_state,
    routes_git,
    routes_graph,
    routes_graph_run,
    routes_images,
    routes_keys,
    routes_llm,
    routes_media,
    routes_models,
    routes_nodes,
    routes_packs,
    routes_plugin_frontend,
    routes_plugins,
    routes_presets,
    routes_runs,
    routes_sweeps,
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
from .core.body_limit import BodySizeLimitMiddleware, RequestBodyTooLarge
from .core.cache import execution_cache_stats
from .core.db import Database
from .core.git.service import GitService
from .core.logging_config import setup_logging
from .core.node_registry import registry
from .core.node_state_store import NodeStateStore
from .core.packs import restart as pack_restart
from .core.packs.service import PackService
from .core.port_stats import PortStatsCache
from .core.version import get_version
from .core import plugin_loader
from .core.plugin_loader import (
    discover_plugin_nodes,
    iter_plugin_dirs,
    load_lockfile,
    rediscover_all,
)
from .core.preset_registry import preset_registry
from .core.run_output_store import RunOutputStore
from .core.run_service import RunService
from .core.run_store import RunStore
from .core.sweep_store import SweepStore

logger = logging.getLogger(__name__)

# Identifies THIS process, generated once at import. Reported by /api/health
# so the SPA can tell "the server is back" from "the server never went away":
# after a restart-mode pack install the frontend polls health, and a health
# that answers again is not proof a NEW process came up — only a boot_id it
# has not seen before is.
BOOT_ID = uuid4().hex


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
    plugin_count, pack_count = discover_plugin_nodes(
        registry,
        plugin_loader.plugins_builtin_root(),
        plugin_loader.plugins_user_root(),
        lockfile,
    )
    logger.info(
        "Discovered %d plugin nodes from %d active plugin(s)", plugin_count, pack_count
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

    # In-memory store for captured per-run node outputs (Teaching Inspector).
    # Bounded by runs AND by bytes (#135) — twenty runs of MNIST batches and
    # twenty runs of 4K feature maps are three orders of magnitude apart.
    app.state.run_output_store = RunOutputStore(
        max_runs=20,
        max_bytes=settings.RUN_OUTPUT_STORE_MAX_MB * 1024 * 1024,
    )

    # Memoised /stats payloads for those captures (#129). Bounded by BYTES —
    # a stat payload ranges from ~1.5 KB to ~50 KB, so an entry count would
    # be a limit in name only.
    app.state.port_stats_cache = PortStatsCache(
        max_bytes=settings.STATS_CACHE_MAX_BYTES
    )

    # Persistent ``nn.Module`` instances per (graph, node, structure-hash).
    # Lifetime: server process. Survives Run clicks; lost on restart.
    app.state.node_state_store = NodeStateStore(
        max_modules=200,
        max_bytes=settings.NODE_STATE_STORE_MAX_MB * 1024 * 1024,
    )

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

    # ── Run Service (#120): server-owned graph runs ────────────────────
    # Owns its own asyncio.Tasks, so it must be created after the DB and
    # drained before it closes (see the shutdown block below).
    # The per-event payload cap is NOT passed: RunService reads
    # settings.RUN_EVENT_PAYLOAD_CAP_BYTES itself, so there is one place the
    # ceiling comes from rather than a lifespan-only override that leaves
    # every other construction of the service on a different number.
    run_service = RunService(
        RunStore(db),
        output_store=app.state.run_output_store,
        retention_keep_last=settings.RUN_RETENTION_KEEP_LAST,
    )
    app.state.run_service = run_service
    # Sweeps (#140) read and write their own table through the same
    # Database. Built here, next to the run service, only so the routes can
    # 503 honestly when it is absent -- the lifespan does not run under
    # httpx's ASGITransport, and tests set app.state directly.
    app.state.sweep_store = SweepStore(db)
    # Order matters. Recovery FIRST: nothing resumes a `running` row after a
    # restart, and retention never deletes an active run — so an abandoned
    # one would keep its events and metrics forever. Retiring it is what
    # makes it prunable in the very next call. Both steps log their own
    # counts when they do anything.
    await run_service.recover_interrupted()
    # Then the SECRET sweep (#251): recovery has just made every row terminal,
    # which is the only kind this may rewrite, and pruning has not yet run —
    # so a row that is about to be deleted costs one wasted scrub rather than
    # a key surviving in a row that pruning was never going to reach.
    await run_service.scrub_stored_secrets()
    await run_service.prune_retention()

    # ── Package Center: one optional-pack install at a time ────────────
    # Same lifetime rules as the run service above: it owns an asyncio.Task
    # and a worker thread, so it is drained in the shutdown block below.
    #
    # A pending restart file still on disk at startup is the wreckage of a
    # restart-mode install that did not happen -- the process that claimed it
    # is gone -- and leaving it there would refuse every future one with "an
    # install is already pending". A claim whose process is still alive is
    # left exactly where it is: that is another server, mid-handshake.
    if pack_restart.clear_stale_pending():
        logger.info("Cleared a stale pending restart file (the install it "
                    "claimed never ran)")
    # `runs_active` is the veto on ending this process for an install: a
    # graph run dies with the server, and that is minutes or hours of
    # somebody's training thrown away. Passed as a closure so the service
    # asks the question without holding the application.
    pack_service = PackService(
        runs_active=lambda: pack_restart.runs_active(app))
    app.state.pack_service = pack_service

    # ── Source Control: the host's own git, in the open project ────────
    # Nothing to start and nothing to drain -- it owns no task and no
    # thread of its own, only a lock and the project directory, which it
    # reads through a closure rather than at startup: `--project` can be
    # set after this process has booted, and a service holding the value
    # from now would keep pointing at wherever it was then.
    app.state.git_service = GitService(
        project_dir=lambda: settings.PROJECT_DIR)

    yield

    # An install in flight when the server stops is cancelled and waited for
    # (bounded), so a pip subprocess is not left writing into site-packages
    # after the process that started it has gone.
    await pack_service.shutdown()
    app.state.pack_service = None
    app.state.git_service = None

    # Drain in-flight runs BEFORE the handle goes away. Their tasks are the
    # only database work in the process that nobody is awaiting, and
    # `Database.close` would otherwise race a run's final writes (#119).
    await run_service.shutdown()
    app.state.run_service = None

    # Release the SQLite handle so `cdui stop` on Windows frees the DB and
    # its WAL sidecar files (spec Section 13, Windows file locking).
    db.close()
    app.state.db = None


app = FastAPI(title=settings.APP_NAME, version=get_version(), lifespan=lifespan)


# ── Security middleware ───────────────────────────────────────────────
#
# Order matters: Starlette applies the *last-added* middleware *first*. We
# want the request to flow as:
#
#   incoming → host_guard → auth_guard → body_limit → CORS → route handler
#
# Because middleware adds in reverse order, we add CORS first (innermost),
# the body cap second, auth third, host fourth (outermost).

# Innermost: CORS preflight + response headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", TOKEN_HEADER, "Authorization"],
    expose_headers=[],
)

# Request-body ceiling, counted as the bytes arrive (core#265, core#242).
#
# DO NOT MOVE THIS OUTSIDE THE TWO GUARDS BELOW. The position is not a matter
# of taste and not merely about precedence — the cap stops working. Measured,
# by relocating it and running the suite: 17 tests fail and an over-limit
# request answers 400 "There was an error parsing the body" instead of 413.
#
# The reason is that auth_guard and host_guard are BaseHTTPMiddleware, which
# runs the downstream app in a task group and pumps `receive` through its own
# wrapper. An exception raised from an UPSTREAM `receive` does not survive that
# plumbing as itself; it surfaces to FastAPI's body parser as a generic
# failure, and FastAPI's `except Exception` rewrites it to a 400. Registered
# here, the cap's `receive` is the one the route reads from directly, so
# RequestBodyTooLarge reaches ExceptionMiddleware intact.
#
# Being inside the guards is also the behaviour we want on its own terms. An
# oversized request with a bad Host still answers 421 and one with no session
# token still answers 403, because both refuse without ever reading the body —
# and a body nobody reads is never counted. A size failure never masks an auth
# failure, and an unauthenticated caller is never handed a 413 that tells it
# the route exists.
#
# INSIDE CORS, finally: the 413 is rendered by ExceptionMiddleware (innermost
# of all), so it travels back out through CORSMiddleware and picks up the
# Access-Control-Allow-Origin header a cross-origin caller needs in order to
# read the status at all.
app.add_middleware(BodySizeLimitMiddleware)


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


@app.exception_handler(RequestBodyTooLarge)
async def body_too_large(request: Request, exc: RequestBodyTooLarge):
    """Render the body cap's 413 in the shape the calling route promises.

    One mechanism counts the bytes; this is the one place that decides how the
    refusal *reads*. Two routes answer with the 9-key run envelope on every
    single response, and that is not a local style choice — the per-app
    OpenAPI document served at ``GET /api/apps/{slug}/openapi.json`` types its
    ``default`` response as ``RunEnvelope``, so third-party clients generated
    from it decode a 413 as an envelope. Everything else in the API answers
    ``{"detail": ...}``, which is what the previous ``POST /api/runs`` cap
    returned and what FastAPI returns for every other HTTPException.

    Reached via Starlette's ``ExceptionMiddleware``: ``RequestBodyTooLarge``
    is an ``HTTPException`` subclass, and handler lookup walks the MRO, so
    this wins over the default HTTPException handler without displacing it
    for any other status.
    """
    params = request.path_params
    path = request.url.path
    if path.startswith("/api/graph/run/"):
        return routes_graph_run.error_response(
            413,
            run_id=uuid4().hex,
            graph=params.get("name", ""),
            code="payload_too_large",
            message=exc.detail,
        )
    if path.startswith("/api/apps/") and path.endswith("/invoke"):
        # version stays None: the cap fires before any version is resolved.
        slug = params.get("slug", "")
        return routes_graph_run.error_response(
            413,
            run_id=uuid4().hex,
            graph=slug,
            app=slug,
            code="payload_too_large",
            message=exc.detail,
        )
    return JSONResponse(status_code=413, content={"detail": exc.detail})


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
app.include_router(routes_media.router)
app.include_router(routes_data_files.router)
app.include_router(routes_execution_outputs.router)
app.include_router(routes_execution_state.router)
app.include_router(routes_runs.router)
app.include_router(routes_sweeps.router)
app.include_router(routes_packs.router)
app.include_router(routes_git.router)
app.include_router(routes_system.router)
app.include_router(routes_llm.router)
app.include_router(routes_apps.router)
app.include_router(routes_keys.router)
app.include_router(ws_execution.router)


async def _cache_usage() -> dict[str, dict[str, int]]:
    """What the three in-memory stores are holding right now (#135).

    Additive on /api/health because "is the server about to run out of
    memory" is a health question, and until this existed the only way to
    answer it was to attach a profiler. Each block reports current bytes
    against the configured budget, plus the count-based limit that still
    applies alongside it.

    Every store is optional here: the lifespan does not run under httpx's
    ASGITransport, so a test client reaches this endpoint with nothing on
    ``app.state``. A missing store is omitted rather than reported as zero,
    which would read as "empty" instead of "not running".
    """
    usage: dict[str, dict[str, int]] = {
        "execution_cache": execution_cache_stats(),
    }
    output_store = getattr(app.state, "run_output_store", None)
    if output_store is not None:
        usage["run_output_store"] = await output_store.stats()
    state_store = getattr(app.state, "node_state_store", None)
    if state_store is not None:
        usage["node_state_store"] = state_store.stats()
    return usage


@app.get("/api/health")
async def health():
    body = {
        "status": "ok",
        # Unconditional, unlike `project` below: this is a new capability
        # rather than a refactor, and the whole point is that a bug reporter
        # on any install can state their version. It is also the first thing
        # `cdui status` and the install check can assert against.
        "version": get_version(),
        # Which PROCESS answered. A client watching for a restart cannot tell
        # "the server came back" from "it never went down" by whether this
        # route answers; a changed boot_id is the only proof.
        "boot_id": BOOT_ID,
        "nodes_loaded": len(registry.nodes),
        "presets_loaded": len(preset_registry.presets),
        "caches": await _cache_usage(),
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


# Production mode: serve the pre-built frontend bundle. Skipped silently in
# dev when dist/ doesn't exist.
DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"

#: URL prefixes the SPA must never answer, so a frontend ``fetch()`` error
#: stays distinguishable from "the SPA loaded". Both halves of the exclusion
#: below are generated from this tuple, so they cannot drift apart.
NON_SPA_PREFIXES = ("api/", "ws/")


class _SpaPathConvertor(Convertor[str]):
    """``{full_path:spa_path}`` — everything ``:path`` matches, minus the API.

    #285. The catch-all used to be a plain ``{full_path:path}`` whose handler
    opened with a ``startswith`` check, on the reasoning that FastAPI matches
    in registration order and the catch-all is registered last. Registration
    order is not enough, because Starlette's router does not stop at the
    first route whose PATH matches — it records a path-matched/method-missed
    route as a PARTIAL match and keeps looking, and a partial match with no
    full match anywhere is answered **405 Method Not Allowed**.

    So ``DELETE /api/files/../../etc/passwd`` — no API route matches, because
    ``{filename}`` does not span ``/`` — reached this GET-only catch-all,
    matched it by path, missed it by method, and came back 405 instead of the
    404 that same request gets when ``dist/`` is absent. The handler's own
    check never ran: the 405 is produced by the router, before any handler.

    Excluding the prefixes in the PATTERN removes the route from
    consideration entirely, so an unhandled ``/api`` path is answered by the
    API layer in both builds — 404 when nothing matches, and still 405 when a
    real API route matched the path but not the method. That last case is the
    reason this is a lookahead in the catch-all rather than an all-methods
    ``/api/{rest:path}`` 404 route registered ahead of it: such a route would
    FULL-match every wrong-method request to a real endpoint and turn its
    honest 405 into a 404.
    """

    regex = "(?!" + "|".join(re.escape(prefix) for prefix in NON_SPA_PREFIXES) + ").*"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("spa_path", _SpaPathConvertor())


def mount_spa(target: FastAPI, dist_dir: Path) -> None:
    """Register the static-asset mount and the SPA catch-all on *target*.

    Takes the app and the directory as arguments rather than reading the
    module globals so the production wiring can be exercised against a
    throwaway ``dist`` — which is the other half of #285. This code used to be
    reachable only through the ``if (DIST_DIR / "index.html").exists()``
    branch below, so every test of it was either skipped or subtly wrong on a
    checkout whose frontend had never been built, and CI's checkout never is.
    """
    # Vite emits content-hashed asset filenames (e.g. index-LKCMvfbh.js), so a
    # given URL's bytes never change — cache them aggressively & immutably.
    class _ImmutableStaticFiles(StaticFiles):
        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return resp

    target.mount(
        "/assets",
        _ImmutableStaticFiles(directory=dist_dir / "assets"),
        name="assets",
    )

    # index.html, by contrast, must NEVER be cached: it's the only file that
    # references the current hashed bundles by name. If a browser serves a
    # stale index.html after an upgrade, it loads an OLD bundle against the NEW
    # backend — and that old bundle predates the session-token handshake, so
    # every WebSocket / mutating request is rejected 403 ("loads but the Run
    # button does nothing"). `no-cache` forces revalidation on every load so
    # the document always matches the running server.
    index_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    @target.get("/{full_path:spa_path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # Belt and braces: unreachable while the convertor above holds, and
        # kept because the failure it guards against is silent. If the pattern
        # ever stops excluding these, an /api typo would start answering 200
        # index.html — a frontend fetch() would then parse HTML as JSON and
        # report something unrelated, instead of the 404 it asked for.
        if full_path.startswith(NON_SPA_PREFIXES):
            raise HTTPException(status_code=404, detail="Not Found")
        # Defence against path traversal: resolve the candidate and confirm
        # it's still inside dist_dir. Browsers normalise ``..`` segments
        # before sending, but ``curl --path-as-is`` and other tools don't,
        # and a stray ``..`` would previously let local processes read any
        # file the server's UID could open.
        dist_resolved = dist_dir.resolve()
        candidate = (dist_dir / full_path).resolve()
        try:
            candidate.relative_to(dist_resolved)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist_dir / "index.html", headers=index_headers)


if (DIST_DIR / "index.html").exists():
    mount_spa(app, DIST_DIR)
