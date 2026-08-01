"""AST-based pre-import validator for plugin and custom-node Python sources.

Blocks the easy RCE shapes (``import os; os.system(...)``,
``getattr(__builtins__, "exec")(...)``, ``__class__.__bases__[0].__subclasses__()``
escape, ``torch.load`` with pickle, etc.) before we let the importer touch the
file. **Not** a sandbox — a determined attacker who controls the file contents
can still escape this with enough work; the goal is to make casual / drive-by
RCE non-trivial and to surface declarative-only plugins for the casual case.

Shared by ``/api/custom-nodes/upload`` (browser uploads), the
``cdui plugin install`` CLI path, and — through
:mod:`app.core.script_policy` — the in-canvas ``PythonScript`` node.

Two gate shapes, one walker
---------------------------
Files the user *installed* are checked against a **blocklist**: anything not
known-dangerous is fine, because the user chose the file. Code typed into
the canvas is checked against an **allowlist** (``import_allowlist``):
nothing but the named modules gets in, because nobody reviewed it. Both
shapes share this walker so a bypass found in one is fixed for both; only
the import rule differs.
"""

from __future__ import annotations

import ast
from typing import Iterable


class PluginValidationError(ValueError):
    """Raised when a Python source file fails AST validation.

    ``lineno`` is the 1-based line the offending construct sits on, when the
    walker knows it. The in-canvas editor turns it into a banner that points
    at a line; the upload and CLI paths ignore it and print the message.
    """

    def __init__(self, message: str, *, lineno: int | None = None) -> None:
        super().__init__(message)
        self.lineno = lineno


# Builtin names that allow direct code execution. We refuse any *call* whose
# resolved callable name lands in this set.
_DANGEROUS_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint",
    "globals", "locals", "getattr", "setattr", "delattr",
    "vars", "dir",
})

# Top-level module imports that bypass the rest of the gate. ``importlib``
# (any sub-module) lets you re-create ``import``; ``ctypes`` is direct memory
# access; ``socket`` / ``urllib`` / ``requests`` give network egress.
_DANGEROUS_MODULES = frozenset({
    "os", "subprocess", "shutil", "sys", "importlib",
    "ctypes", "socket", "http", "urllib", "requests",
    "pathlib", "tempfile", "signal", "pickle", "shelve",
    "code", "codeop", "compileall", "marshal", "dill",
    "runpy", "atexit", "asyncio", "multiprocessing", "threading",
})

# Attribute-access patterns that are RCE in disguise. We can't tell
# `numpy.load` apart from `np.load` purely by name (the import alias matters),
# but matching on the *attribute leaf* still catches the most common forms.
# Pure attribute names — no module prefix — to keep false negatives small.
_DANGEROUS_ATTR_LEAVES = frozenset({
    "system", "popen", "spawnl", "spawnv",       # os
    "loads", "load",                              # pickle / marshal — see _DANGEROUS_LOAD_PATTERNS for the qualified torch.load case
    "execfile", "compile_command",                # code / codeop
})

# Dunder names that should *never* appear inside a plugin: they're the
# universal Python sandbox-escape primitives. Includes the names used by:
#   * ``().__class__.__bases__[0].__subclasses__()[N](...)``  — class walk
#   * ``getattr(__builtins__, "exec")``                       — builtins escape
#   * ``func.__globals__["__builtins__"]``                    — globals escape
#   * ``some_code.__code__.co_consts``                         — bytecode peek
_FORBIDDEN_DUNDERS = frozenset({
    "__class__", "__bases__", "__base__", "__mro__", "__subclasses__",
    "__builtins__", "__globals__", "__import__", "__dict__",
    "__code__", "__closure__", "__func__", "__self__",
    "__getattribute__", "__getattr__", "__setattr__", "__delattr__",
    "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    "__init_subclass__", "__class_getitem__",
    "__loader__", "__spec__", "__package__",
})


def dangerous_modules() -> frozenset[str]:
    """Public view of the default blocklist (mainly for tests and error messages)."""
    return _DANGEROUS_MODULES


def forbidden_dunders() -> frozenset[str]:
    """Public view of the dunder blocklist (mainly for tests)."""
    return _FORBIDDEN_DUNDERS


def _resolve_call_name(func: ast.expr) -> str | None:
    """Return the leaf name of a callable expression, if recoverable.

    ``foo()`` → ``"foo"``; ``a.b.c()`` → ``"c"``; ``something[0]()`` → None.
    The leaf is what we compare against ``_DANGEROUS_NAMES`` and
    ``_DANGEROUS_ATTR_LEAVES``.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _check_getattr_arg_safety(call: ast.Call, denial_hint: str = "") -> None:
    """Disallow ``getattr(<dunder-like>, ...)`` even when arg 1 is a literal."""
    if not call.args:
        return
    first = call.args[0]
    # First arg as a bare ``Name`` matching a forbidden dunder (``__builtins__``
    # being the canonical bypass) — refuse.
    if isinstance(first, ast.Name) and first.id in _FORBIDDEN_DUNDERS:
        raise PluginValidationError(
            f"getattr() against forbidden name {first.id!r} is not allowed"
            f"{denial_hint}",
            lineno=call.lineno,
        )
    # Second arg as a *literal* string that itself names a forbidden dunder
    # (e.g. ``getattr(obj, "__class__")``) — refuse.
    if len(call.args) >= 2:
        second = call.args[1]
        if (
            isinstance(second, ast.Constant)
            and isinstance(second.value, str)
            and second.value in _FORBIDDEN_DUNDERS
        ):
            raise PluginValidationError(
                f"getattr() retrieving forbidden dunder "
                f"{second.value!r} is not allowed{denial_hint}",
                lineno=call.lineno,
            )


def validate_python_source(
    content: bytes | str,
    filename: str = "<plugin>",
    *,
    allowed_modules: Iterable[str] | None = None,
    import_allowlist: Iterable[str] | None = None,
    extra_denied_names: Iterable[str] | None = None,
    denial_hint: str = "",
) -> None:
    """Parse *content* and raise if it contains obviously dangerous patterns.

    *allowed_modules* widens the import blocklist for plugins that legitimately
    need one of the default-banned top-level modules (declared in their
    manifest under ``[security].allowed_modules`` and accepted by the user
    via ``--trust-author``). Dangerous builtin *calls* and dunder attribute
    access are never widened.

    *import_allowlist* switches the import rule from blocklist to allowlist:
    a top-level module outside the set is refused whatever it is, and
    relative imports (which name no module at all) are refused with it. This
    is the shape in-canvas scripts use — see :mod:`app.core.script_policy`.

    *extra_denied_names* adds to the denied *call* names for this call only
    (``open``, ``input``, ... for scripts). *denial_hint* is appended to
    every message raised here, so one caller can point users at the escape
    hatch its policy implies without every rule growing a special case.
    """
    allowed = frozenset(allowed_modules) if allowed_modules else frozenset()
    allowlist = frozenset(import_allowlist) if import_allowlist is not None else None
    denied_names = _DANGEROUS_NAMES | frozenset(extra_denied_names or ())

    def fail(message: str, node: ast.AST | None = None) -> PluginValidationError:
        return PluginValidationError(
            f"{message}{denial_hint}", lineno=getattr(node, "lineno", None)
        )

    def check_import(
        module: str | None,
        node: ast.AST,
        *,
        level: int = 0,
    ) -> None:
        """Apply whichever import rule this call selected."""
        if allowlist is not None:
            if level or not module:
                raise fail(f"Relative imports are not allowed in {filename}", node)
            if module.split(".")[0] not in allowlist:
                raise fail(f"Importing '{module}' is not allowed in {filename}", node)
            return
        if module and module.split(".")[0] in _DANGEROUS_MODULES and (
            module.split(".")[0] not in allowed
        ):
            raise fail(f"Importing '{module}' is not allowed in {filename}", node)

    try:
        tree = ast.parse(content, filename=filename)
    except SyntaxError as e:
        # ``e`` already renders as "msg (<file>, line N)"; naming the line
        # first makes it usable as a one-line editor banner.
        line = f" at line {e.lineno}" if e.lineno else ""
        # No policy hint here: half-typed code is not a policy question, and
        # reciting the import allowlist under "expected ':'" is noise.
        raise PluginValidationError(
            f"Syntax error{line} in {filename}: {e.msg}",
            lineno=e.lineno,
        ) from e

    for node in ast.walk(tree):
        # ── Import / ImportFrom ──────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_import(alias.name, node)
        elif isinstance(node, ast.ImportFrom):
            check_import(node.module, node, level=node.level or 0)

        # ── Attribute access (Foo.__class__, x.__bases__, etc.) ──────
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_DUNDERS:
                raise fail(
                    f"Access to attribute {node.attr!r} is not allowed in {filename}",
                    node,
                )

        # ── Subscript: ``__builtins__["exec"]`` form ─────────────────
        elif isinstance(node, ast.Subscript):
            target = node.value
            if isinstance(target, ast.Name) and target.id in _FORBIDDEN_DUNDERS:
                raise fail(
                    f"Subscript on {target.id!r} is not allowed in {filename}",
                    node,
                )

        # ── Calls ────────────────────────────────────────────────────
        elif isinstance(node, ast.Call):
            name = _resolve_call_name(node.func)
            if name is None:
                # The callable came from a runtime expression like
                # ``getattr(obj, "x")()`` — the inner getattr is itself a
                # Call node that ast.walk will visit, so we still gate on
                # it there. Nothing left to check at this level.
                continue
            if name in denied_names:
                if name in ("getattr", "setattr", "delattr"):
                    # Tighter version of the original exception: literal
                    # 2nd-arg string is still allowed, but only after we
                    # verify it isn't being used to retrieve a forbidden
                    # dunder or applied to a forbidden first-arg name.
                    _check_getattr_arg_safety(node, denial_hint)
                    if (
                        len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        continue
                raise fail(f"Use of {name!r}() is not allowed in {filename}", node)
            if isinstance(node.func, ast.Attribute):
                # ``a.b.load(...)`` and ``a.system(...)`` — match the leaf
                # against known-bad patterns. Most legitimate ML code that
                # legitimately calls ``torch.load`` does so with
                # ``weights_only=True`` keyword; we enforce that explicitly.
                if node.func.attr == "load":
                    _enforce_safe_load(node, filename, denial_hint)
                elif node.func.attr in _DANGEROUS_ATTR_LEAVES:
                    raise fail(
                        f"Call to '.{node.func.attr}(...)' is not allowed in {filename}",
                        node,
                    )


def _enforce_safe_load(call: ast.Call, filename: str, denial_hint: str = "") -> None:
    """Require ``weights_only=True`` for any ``X.load(...)`` call.

    Catches the common pickle-RCE pattern via ``torch.load(...)`` /
    ``numpy.load(allow_pickle=True)`` without having to track import aliases.
    Conservative: any ``.load()`` without an explicit safety kwarg is rejected.
    """
    # numpy.load with allow_pickle=True → reject
    for kw in call.keywords:
        if kw.arg == "allow_pickle" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            raise PluginValidationError(
                f"Call to '.load(allow_pickle=True)' is not allowed in {filename}; "
                f"it can execute arbitrary code from the source file{denial_hint}",
                lineno=call.lineno,
            )
    # torch.load → require explicit weights_only=True
    for kw in call.keywords:
        if kw.arg == "weights_only":
            if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return  # explicit safe call → OK
            raise PluginValidationError(
                f"Call to '.load(weights_only={ast.unparse(kw.value)})' is not allowed "
                f"in {filename}; only weights_only=True is permitted{denial_hint}",
                lineno=call.lineno,
            )
    # No weights_only / allow_pickle kwarg supplied. Allow safe stdlib-style
    # uses by checking the receiver: ``json.load``, ``yaml.safe_load``,
    # ``csv.load`` etc. are fine. ``torch.load`` / ``np.load`` / ``pickle.load``
    # are not.
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        receiver = func.value.id
        if receiver in {"torch", "np", "numpy", "pickle", "joblib", "dill"}:
            raise PluginValidationError(
                f"Bare {receiver}.load(...) is not allowed in {filename}; "
                f"pass weights_only=True explicitly to make the intent "
                f"obvious{denial_hint}",
                lineno=call.lineno,
            )
