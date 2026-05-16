from pathlib import Path
from platformdirs import user_data_dir
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "CodefyUI"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB

    NODES_DIR: Path = Path(__file__).parent / "nodes"
    CUSTOM_NODES_DIR: Path = Path(__file__).parent / "custom_nodes"
    GRAPHS_DIR: Path = Path(__file__).parent.parent / "data" / "graphs"
    PRESETS_DIR: Path = Path(__file__).parent / "presets"
    MODELS_DIR: Path = Path(__file__).parent.parent / "data" / "models"
    IMAGES_DIR: Path = Path(__file__).parent.parent / "data" / "images"
    EXAMPLES_DIR: Path = Path(__file__).parent.parent.parent / "examples"

    # First-party plugin packs shipped with the repo (<REPO>/plugins/).
    PLUGINS_BUILTIN_DIR: Path = Path(__file__).parent.parent.parent / "plugins"
    # Third-party downloads + lockfile (per-user, platformdirs).
    PLUGINS_USER_DIR: Path = Path(user_data_dir("codefyui", appauthor=False)) / "plugins"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path | None = None
    LOG_JSON: bool = False

    MAX_PARALLEL_NODES: int = 4

    model_config = {"env_prefix": "CODEFYUI_"}


settings = Settings()
