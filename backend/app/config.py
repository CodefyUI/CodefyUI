import os
from pathlib import Path
from platformdirs import user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings

# Let the handful of ops not yet implemented for Apple MPS fall back to CPU
# instead of hard-crashing the graph run. Set here (before torch is imported
# anywhere) so it applies to the server, the CLI runner, and tests alike.
# ``setdefault`` keeps an explicit external value (e.g. a test pinning "0" to
# assert native MPS support) authoritative.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _user_data_root() -> Path:
    """Resolve the per-user data root (lockfile, downloaded plugins, session token).

    Honors ``CODEFYUI_USER_DATA_DIR`` so a dev clone using
    ``scripts/dev.py`` can keep its lockfile in ``.codefyui_dev/`` inside
    the repo, isolated from the global ``cdui`` install at
    ``%LOCALAPPDATA%\\codefyui``. ``plugin_loader.plugins_user_root`` and
    ``auth._token_dir`` honor the same variable so plugin install, the
    server, and the hot-reload token all stay in sync.
    """
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    return Path(override) if override else Path(user_data_dir("codefyui", appauthor=False))


def _default_plugins_user_dir() -> Path:
    return _user_data_root() / "plugins"


class Settings(BaseSettings):
    APP_NAME: str = "CodefyUI"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB
    # Cap for POST /api/graph/run/{name} request bodies, checked against
    # Content-Length before the body is read (-> 413 payload_too_large).
    # 64 MB comfortably covers a handful of base64 image inputs.
    # Env-overridable as CODEFYUI_MAX_RUN_BODY_BYTES.
    MAX_RUN_BODY_BYTES: int = 64 * 1024 * 1024  # 64 MB

    # ── Stage-2 publish storage + limits (spec Section 8) ──────────────
    # All env-overridable via the CODEFYUI_ prefix like everything else.
    # SQLite file for published apps / API keys / run records; sibling of
    # GRAPHS_DIR so a dev clone and a global install stay isolated.
    DB_PATH: Path = Path(__file__).parent.parent / "data" / "codefyui.db"
    MAX_IMAGE_PIXELS: int = 25_000_000  # per-image decode budget (Decision H2)
    RUN_IO_CAP_BYTES: int = 64 * 1024   # per-field runs.inputs_json/outputs_json cap
    RUNS_RETENTION_DAYS: int = 0        # 0 = keep forever (default); >0 prunes loudly
    # Comma-separated extra Host-whitelist entries, e.g.
    # "192.168.1.20:8000,mybox:8000". A str, not list[str]: pydantic list
    # env vars demand JSON-in-env quote hell; split in init_allowed_hosts.
    EXTRA_ALLOWED_HOSTS: str = ""

    NODES_DIR: Path = Path(__file__).parent / "nodes"
    CUSTOM_NODES_DIR: Path = Path(__file__).parent / "custom_nodes"
    GRAPHS_DIR: Path = Path(__file__).parent.parent / "data" / "graphs"
    PRESETS_DIR: Path = Path(__file__).parent / "presets"
    MODELS_DIR: Path = Path(__file__).parent.parent / "data" / "models"
    IMAGES_DIR: Path = Path(__file__).parent.parent / "data" / "images"
    EXAMPLES_DIR: Path = Path(__file__).parent.parent.parent / "examples"

    # First-party plugin packs shipped with the repo (<REPO>/plugins/).
    PLUGINS_BUILTIN_DIR: Path = Path(__file__).parent.parent.parent / "plugins"
    # Third-party downloads + lockfile. Resolved per-instance so the
    # CODEFYUI_USER_DATA_DIR env var picked up by dev-mode tooling actually
    # flows through to the server. Without default_factory the snapshot
    # would freeze at module-import time, before scripts/dev.py sets the
    # env var.
    PLUGINS_USER_DIR: Path = Field(default_factory=_default_plugins_user_dir)

    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path | None = None
    LOG_JSON: bool = False

    MAX_PARALLEL_NODES: int = 4

    model_config = {"env_prefix": "CODEFYUI_"}


settings = Settings()
