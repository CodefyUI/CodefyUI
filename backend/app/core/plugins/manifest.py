"""``cdui.plugin.toml``: reading one, and refusing the ones we will not run.

A manifest is HAND-WRITTEN, by a plugin author who cannot see what the
installer does with it, so every check here exists because a wrong shape once
failed in the wrong direction rather than because a schema wanted to be
complete. ``allowed_modules = "os"`` reaching ``frozenset("os")`` -- which is
``{"o", "s"}`` -- is the shape that named this module's rule: say what is
wrong with the file, at the point it is read, instead of granting nothing and
failing much later somewhere that blames the wrong line.

The reader (:func:`read_manifest`) and the accessors below never validate,
and :func:`validate_manifest` never reads a file. That split is what lets a
route show a manifest it has refused: the GUI wants to say "this plugin asks
for the network and its id is invalid", which needs the parsed table and the
refusal at the same time.

Nothing here formats a message for a human. ``ManifestError``'s text is
English and one line -- the CLI and the routes each have their own way of
telling the user, and neither wants the other's.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from app.core.plugin_loader import MANIFEST_FILENAME
from app.core.security_tiers import (
    CAPABILITIES,
    normalize_capabilities,
    unknown_capabilities,
)

from .errors import ManifestError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport -- same API.

#: A plugin id is a directory name, a URL segment and a Python-ish namespace
#: component all at once, so it is deliberately narrower than any of them.
PLUGIN_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

#: The one ``[plugin].schema_version`` this build knows how to install.
SUPPORTED_SCHEMA = 1


def read_manifest(plugin_root: Path) -> dict[str, Any]:
    """Parse ``<plugin_root>/cdui.plugin.toml``.

    Raises ``FileNotFoundError`` when there is no manifest and
    ``tomllib.TOMLDecodeError`` when there is one that is not TOML. Neither is
    turned into a :class:`~.errors.ManifestError`: "there is no file" and
    "the file is not TOML" are answers about the DISK, and a caller that
    catches them separately (the CLI does) can say something better than
    "invalid manifest".
    """
    p = plugin_root / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found at {p}")
    return tomllib.loads(p.read_text(encoding="utf-8"))


def validate_manifest(m: dict[str, Any]) -> None:
    """Refuse a manifest this build will not install. Returns None or raises.

    Only the parts something ACTS on: the ``[plugin]`` table (without which
    there is no id to install under), the schema version, the id itself, and
    ``[security]``. Everything else in a manifest is metadata that a wrong
    value only makes ugly.
    """
    plugin = m.get("plugin")
    if not isinstance(plugin, dict):
        raise ManifestError("Manifest is missing required [plugin] table.")
    schema_version = plugin.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA:
        raise ManifestError(
            f"Unsupported plugin schema_version: {schema_version!r}. "
            "Upgrade cdui or use an older plugin release."
        )
    plugin_id = plugin.get("id", "")
    if not PLUGIN_ID_RE.match(plugin_id):
        raise ManifestError(
            f"Invalid plugin id: {plugin_id!r}. "
            "Must match ^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
        )
    _validate_security_table(m.get("security"))


def _validate_security_table(security: Any) -> None:
    """Check ``[security]`` before anything acts on it.

    Both keys are lists of strings, and a wrong shape used to fail silently in
    the worst possible direction: ``allowed_modules = "os"`` reaches
    ``frozenset("os")``, which is ``{"o", "s"}``, which grants nothing and
    unlocks nothing while printing "Plugin requests non-default modules: o, s".
    A manifest is hand-written; say what is wrong with it.

    An unknown capability is an error rather than a no-op. It is either a typo
    or a manifest written against a newer CodefyUI, and granting nothing then
    failing at the import would blame the wrong line.
    """
    if security is None:
        return
    if not isinstance(security, dict):
        raise ManifestError("[security] must be a table.")
    for key in ("allowed_modules", "capabilities"):
        value = security.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ManifestError(
                f"[security].{key} must be a list of strings, got {value!r}."
            )
    unknown = unknown_capabilities(security.get("capabilities"))
    if unknown:
        raise ManifestError(
            f"Unknown capability in [security].capabilities: "
            f"{', '.join(unknown)}. This build knows: {', '.join(CAPABILITIES)}."
        )


def manifest_capabilities(manifest: dict[str, Any]) -> tuple[str, ...]:
    """The normalised ``[security].capabilities`` a manifest declares."""
    security = manifest.get("security")
    if not isinstance(security, dict):
        return ()
    return normalize_capabilities(security.get("capabilities"))


def manifest_allowed_modules(manifest: dict[str, Any]) -> list[str]:
    """The ``[security].allowed_modules`` a manifest asks to import.

    A list rather than a tuple because that is what the AST gate and the
    lockfile writer both want, and a fresh one each call because callers pass
    it straight into ``validate_plugin_dir``.

    Answers ``[]`` for every shape that is not a list of strings.
    :func:`validate_manifest` has already refused those loudly; this is the
    accessor a reader (``plugin list``, an inspect route) uses on a manifest
    nobody promised was valid, and there it must describe rather than raise.
    """
    security = manifest.get("security")
    if not isinstance(security, dict):
        return []
    value = security.get("allowed_modules")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def manifest_python_deps(manifest: dict[str, Any]) -> dict[str, str]:
    """The ``[python_deps]`` table -- ``{name: version constraint}``, ``{}``
    when the manifest declares none.

    Returned exactly as written, NOT vetted: a name or a constraint from here
    is still an untrusted string that only ``_build_dep_spec`` may turn into
    something ``uv pip install`` sees. Filtering the odd entries out here
    would be worse than useless -- it would silently drop the malformed spec
    that the installer currently refuses by name.
    """
    deps = manifest.get("python_deps")
    return deps if isinstance(deps, dict) else {}


def manifest_has_frontend(manifest: dict[str, Any]) -> bool:
    """Whether this plugin ships browser code (``[frontend].entry``).

    Worth its own question because the answer drives a warning rather than a
    file operation: frontend code runs in the user's browser inside the
    editor with full editor access, which is a different trust decision from
    anything the AST gate covers.
    """
    fe = manifest.get("frontend")
    return isinstance(fe, dict) and isinstance(fe.get("entry"), str) and bool(fe.get("entry"))
