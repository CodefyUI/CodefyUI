import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    routes_presets,
    ws_execution,
)
from .config import settings
from .core.logging_config import setup_logging
from .core.node_registry import registry
from .core.node_state_store import NodeStateStore
from .core.preset_registry import preset_registry
from .core.run_output_store import RunOutputStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(
        level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
        json_format=settings.LOG_JSON,
    )

    # Discover built-in nodes
    count = registry.discover(settings.NODES_DIR, "app.nodes")
    logger.info("Discovered %d built-in nodes", count)

    # Discover custom nodes
    custom_count = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    logger.info("Discovered %d custom nodes", custom_count)

    for name in sorted(registry.nodes.keys()):
        logger.debug("  - %s (%s)", name, registry.nodes[name].CATEGORY)

    # Discover presets
    preset_count = preset_registry.discover(settings.PRESETS_DIR, registry)
    logger.info("Discovered %d presets", preset_count)
    for name in sorted(preset_registry.presets.keys()):
        logger.debug("  * %s", name)

    # In-memory store for captured per-run node outputs (Teaching Inspector)
    app.state.run_output_store = RunOutputStore(max_runs=20)

    # Persistent ``nn.Module`` instances per (graph, node, structure-hash).
    # Lifetime: server process. Survives Run clicks; lost on restart.
    app.state.node_state_store = NodeStateStore(max_modules=200)

    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(routes_nodes.router)
app.include_router(routes_examples.router)
app.include_router(routes_graph.router)
app.include_router(routes_presets.router)
app.include_router(routes_custom_nodes.router)
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


@app.post("/api/nodes/reload")
async def reload_nodes():
    registry.clear()
    count = registry.discover(settings.NODES_DIR, "app.nodes")
    custom_count = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    preset_registry.clear()
    preset_count = preset_registry.discover(settings.PRESETS_DIR, registry)
    return {
        "builtin": count,
        "custom": custom_count,
        "presets": preset_count,
        "total": count + custom_count,
    }


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
        candidate = DIST_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
