"""AST-based pre-import validator for plugin and custom-node Python sources.

Blocks module-level imports and calls that would trivially break the node
sandbox model. Not a real sandbox — a determined attacker can still get
through — but enough to reject the obvious RCE shapes (``import os;
os.system(...)``).

Shared by ``/api/custom-nodes/upload`` (browser uploads) and the
``cdui plugin install`` CLI path.
"""

from __future__ import annotations

import ast
from typing import Iterable


class PluginValidationError(ValueError):
    """Raised when a Python source file fails AST validation."""


_DANGEROUS_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint",
    "globals", "locals", "getattr", "setattr", "delattr",
})

_DANGEROUS_MODULES = frozenset({
    "os", "subprocess", "shutil", "sys", "importlib",
    "ctypes", "socket", "http", "urllib", "requests",
    "pathlib", "tempfile", "signal", "pickle", "shelve",
    "code", "codeop", "compileall",
})


def dangerous_modules() -> frozenset[str]:
    """Public view of the default blocklist (mainly for tests and error messages)."""
    return _DANGEROUS_MODULES


def validate_python_source(
    content: bytes | str,
    filename: str = "<plugin>",
    *,
    allowed_modules: Iterable[str] | None = None,
) -> None:
    """Parse *content* and raise if it contains obviously dangerous patterns.

    *allowed_modules* widens the import whitelist for plugins that legitimately
    need one of the default-banned top-level modules (declared in their
    manifest under ``[security].allowed_modules`` and accepted by the user
    via ``--trust-author``). Dangerous builtin *calls* (``exec`` etc.) are
    never allowed.
    """
    allowed = frozenset(allowed_modules) if allowed_modules else frozenset()
    try:
        tree = ast.parse(content, filename=filename)
    except SyntaxError as e:
        raise PluginValidationError(f"Syntax error in {filename}: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _DANGEROUS_MODULES and top not in allowed:
                    raise PluginValidationError(
                        f"Importing '{alias.name}' is not allowed in {filename}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _DANGEROUS_MODULES and top not in allowed:
                    raise PluginValidationError(
                        f"Importing from '{node.module}' is not allowed in {filename}"
                    )
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _DANGEROUS_NAMES:
                # getattr/setattr/delattr are the common false-positives — the
                # attack shape that matters is *dynamic* attribute lookup
                # (`getattr(__builtins__, name_from_user)`). Pinned-literal
                # access like `getattr(context, "verbose", False)` is a
                # mainstream Python idiom and safe to allow.
                if (
                    name in ("getattr", "setattr", "delattr")
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    continue
                raise PluginValidationError(
                    f"Use of '{name}()' is not allowed in {filename}"
                )
