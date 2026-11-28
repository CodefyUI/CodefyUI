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

#: Names a run-time load never applies (patterns, matched without regard to
#: case: ``https_proxy`` works as well as ``HTTPS_PROXY``, and Windows does
#: not tell the two apart at all). They change how processes start, where
#: code or packages come from, how git, ssh and shells run, where network
#: traffic goes and which certificates it trusts, or this server's own
#: configuration.
BLOCKED_PATTERNS: tuple[str, ...] = (
    # Process start-up, interpreters and the server itself.
    "PATH", "PYTHON*", "LD_*", "DYLD_*", "CODEFYUI_*",
    "NODE_OPTIONS", "NODE_PATH", "PERL5*", "RUBY*",
    "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS",
    # Package installs (a pack or plugin install fetches from these).
    "PIP_*", "UV_*", "CONDA_*",
    # git, ssh, shells and the programs they start.
    "GIT_*", "SSH_*", "HOME", "USERPROFILE", "SHELL", "COMSPEC",
    "BASH_ENV", "ENV", "IFS", "XDG_*",
    "EDITOR", "VISUAL", "PAGER", "BROWSER",
    "TMPDIR", "TEMP", "TMP",
    # Where traffic goes and which certificates it trusts.
    "*_PROXY", "SSL_CERT_*", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
)

#: Names no project's .env may set, not even the project the operator named
#: with ``--project`` / ``CODEFYUI_PROJECT_DIR`` (phase 4C): they choose
#: which code the interpreter and the dynamic loader run, for this server's
#: child processes (runs, workers, installs), and a cloned repository could
#: commit them as easily as anything else. Someone who means it exports
#: them in their own shell.
ALWAYS_BLOCKED_PATTERNS: tuple[str, ...] = ("PYTHON*", "LD_*", "DYLD_*")

#: Names libraries read once, at import or first use, so a value applied by
#: a run-time switch has no effect until the server restarts. Reported back
#: to the user, never refused.
RESTART_ONLY_PATTERNS: tuple[str, ...] = (
    "CUDA_*", "HF_*", "HUGGINGFACE_*", "TRANSFORMERS_*", "TORCH_*", "PYTORCH_*",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)

#: Names that choose a server to talk to. Allowed, but the consent prompt
#: points them out: ``OPENAI_BASE_URL=https://someone-else`` sends the
#: user's own key there with every request.
ENDPOINT_PATTERNS: tuple[str, ...] = (
    "*_URL", "*_BASE", "*_ENDPOINT", "*_HOST", "*_API_BASE",
)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    candidate = name.upper()
    return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns)


def is_blocked(name: str) -> bool:
    return _matches(name, BLOCKED_PATTERNS)


def is_always_blocked(name: str) -> bool:
    """Blocked even for the project named at start-up (``--project``)."""
    return _matches(name, ALWAYS_BLOCKED_PATTERNS)


def needs_restart(name: str) -> bool:
    return _matches(name, RESTART_ONLY_PATTERNS)


def is_endpoint(name: str) -> bool:
    """True when *name* looks like it picks a server (see ENDPOINT_PATTERNS)."""
    return _matches(name, ENDPOINT_PATTERNS)


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
    start-up (``--project``), which then skips only the names that choose
    which code runs (:data:`ALWAYS_BLOCKED_PATTERNS`).
    Returns the names ``applied``, ``blocked``, ``kept`` (already set in
    the environment) and ``restart`` (applied, but read only at start).
    Values are never returned or logged.
    """
    report: dict[str, list[str]] = {"applied": [], "blocked": [], "kept": [], "restart": []}
    for name, value in values.items():
        if is_blocked(name) if enforce_blocklist else is_always_blocked(name):
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
