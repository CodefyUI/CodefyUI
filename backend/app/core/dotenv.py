"""Minimal stdlib .env loader for project directories (spec 7.3).

No python-dotenv dependency. Loaded at server start, BEFORE node/plugin
discovery (and again when a project is opened at run time, through
:func:`apply_project_env`, which records what it applied so a switch can
take it back out), with os.environ.setdefault semantics (an already-set variable
wins). Values are execution-time secrets only and are NEVER logged. Because
the Settings singleton materializes at import (config.py), CODEFYUI_* CONFIG
keys placed here do not reconfigure the running server -- only execution-time
keys (LLM API keys, arbitrary secrets read via os.environ at execute time)
take effect.
"""

from __future__ import annotations

import fnmatch
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
        text = path.read_text(encoding="utf-8-sig")  # tolerates a leading UTF-8 BOM
    except (OSError, UnicodeDecodeError):
        return 0
    applied = 0
    for key, val in parse_dotenv(text).items():
        if key not in os.environ:
            os.environ[key] = val
            applied += 1
    return applied


# ── Run-time loading (phase 4A) ──────────────────────────────────────────
#
# A project can be opened, and switched away from, while the server runs.
# Its .env then has to come back out of the process when another project
# opens -- otherwise one project's API keys serve the next one's runs, and
# the next one's own value for the same key is ignored by setdefault. And
# opening a folder over the API must not let that folder's .env reconfigure
# the server: a repo can commit PYTHONPATH or LD_PRELOAD as easily as an API
# key. Hence: the keys a load applied are recorded and unloaded, and a run-
# time load skips the blocked names below.

#: Names a run-time load never applies (patterns, matched case-insensitively
#: on Windows, where environment names are case-insensitive, and exactly
#: elsewhere). They change how processes start, where code is imported from,
#: how git and ssh run, or this server's own configuration.
BLOCKED_PATTERNS: tuple[str, ...] = (
    "PATH", "PYTHON*", "LD_*", "DYLD_*", "GIT_*", "SSH_*", "CODEFYUI_*",
    "HOME", "USERPROFILE", "SHELL", "COMSPEC",
)

#: Names libraries read once, at import or first use, so a value applied by
#: a run-time switch has no effect until the server restarts. Reported back
#: to the user, never refused.
RESTART_ONLY_PATTERNS: tuple[str, ...] = (
    "CUDA_*", "HF_*", "HUGGINGFACE_*", "TRANSFORMERS_*", "TORCH_*", "PYTORCH_*",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM", "XDG_CACHE_HOME",
)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    candidate = name.upper() if os.name == "nt" else name
    return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns)


def is_blocked(name: str) -> bool:
    return _matches(name, BLOCKED_PATTERNS)


def needs_restart(name: str) -> bool:
    return _matches(name, RESTART_ONLY_PATTERNS)


def read_dotenv(path: Path) -> dict[str, str] | None:
    """The parsed file, or None when it is absent or unreadable."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    return parse_dotenv(text)


#: What the current project's .env put into os.environ: name -> value.
_APPLIED: dict[str, str] = {}


def applied_names() -> list[str]:
    return sorted(_APPLIED)


def apply_project_env(values: dict[str, str], *, enforce_blocklist: bool) -> dict[str, list[str]]:
    """Apply *values* the way the start-up load does (an existing variable
    wins) and remember what was applied, for :func:`unload_project_env`.

    ``enforce_blocklist`` is off only for a project the operator named at
    start-up (``--project``), which keeps today's behaviour for it.
    Returns the names ``applied``, ``blocked``, ``kept`` (already set in
    the environment) and ``restart`` (applied, but read only at start).
    Values are never returned or logged.
    """
    report: dict[str, list[str]] = {"applied": [], "blocked": [], "kept": [], "restart": []}
    for name, value in values.items():
        if enforce_blocklist and is_blocked(name):
            report["blocked"].append(name)
            continue
        if name in os.environ:
            report["kept"].append(name)
            continue
        os.environ[name] = value
        _APPLIED[name] = value
        report["applied"].append(name)
        if needs_restart(name):
            report["restart"].append(name)
    return {key: sorted(names) for key, names in report.items()}


def unload_project_env() -> list[str]:
    """Remove what :func:`apply_project_env` applied. A variable something
    else has changed since is left alone: it is no longer ours."""
    removed = []
    for name, value in list(_APPLIED.items()):
        if os.environ.get(name) == value:
            del os.environ[name]
            removed.append(name)
        del _APPLIED[name]
    return sorted(removed)
