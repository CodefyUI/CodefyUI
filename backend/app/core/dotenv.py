"""Minimal stdlib .env loader for project directories (spec 7.3).

No python-dotenv dependency. Loaded ONCE at server start, BEFORE node/plugin
discovery, with os.environ.setdefault semantics (an already-set variable
wins). Values are execution-time secrets only and are NEVER logged. Because
the Settings singleton materializes at import (config.py), CODEFYUI_* CONFIG
keys placed here do not reconfigure the running server -- only execution-time
keys (LLM API keys, arbitrary secrets read via os.environ at execute time)
take effect.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Skips blanks/comments, tolerates a leading
    `export `, strips matching surrounding single/double quotes."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        result[key] = val
    return result


def load_dotenv_file(path: Path) -> int:
    """os.environ.setdefault each KEY=VALUE from *path*. Returns the number of
    NEW variables applied. Absent/unreadable file -> 0. Never logs values."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    applied = 0
    for key, val in parse_dotenv(text).items():
        if key not in os.environ:
            os.environ[key] = val
            applied += 1
    return applied
