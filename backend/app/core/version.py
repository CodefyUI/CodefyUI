"""The running CodefyUI version, resolved once.

There was no way for a user or a bug reporter to say which version they were
on: the FastAPI app declared none, ``/api/health`` returned none, and there
was no ``cdui --version``. The only runtime signal was
``frontend/dist/build-info.json``, which describes the frontend bundle rather
than the backend. For a tool handed to a class, that makes triage guesswork.

``importlib.metadata`` is the source of truth because it reads what is
actually installed, not what the checkout happens to say -- those differ
exactly when it matters, e.g. a `cdui update` that fetched a newer dist but
left the backend on an older commit.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path

PACKAGE_NAME = "codefyui-backend"

#: Reported when neither installed metadata nor pyproject.toml can be read.
#: Deliberately not a plausible-looking version -- a wrong number in a bug
#: report is worse than an obvious "unknown".
UNKNOWN_VERSION = "0.0.0+unknown"

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _version_from_pyproject() -> str | None:
    """Fallback for a source checkout that was never installed (`cdui dev`)."""
    try:
        text = _PYPROJECT.read_text(encoding="utf-8")
    except OSError:
        return None
    # Only the [project] table's own version -- a `[tool.*]` section further
    # down could carry an unrelated one.
    head = text.split("\n[", 1)[0] if text.startswith("[project]") else text
    match = _VERSION_RE.search(head)
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def get_version() -> str:
    try:
        return _package_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return _version_from_pyproject() or UNKNOWN_VERSION
