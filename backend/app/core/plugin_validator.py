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

Three tiers on the blocklist side (core#133)
--------------------------------------------
Within the blocklist shape, a refused module now belongs to one of three
answers rather than one:

* **Tier 0** -- not on the blocklist at all. Nothing to declare.
* **Tier 1** -- on the blocklist, and unlocked by a capability the manifest
  declared and the user confirmed at install time (*capabilities*).
* **Tier 2** -- on the blocklist, unlocked only by ``--trust-author`` plus
  ``[security].allowed_modules`` (*allowed_modules*).

The tier vocabulary lives in :mod:`app.core.security_tiers`, which imports
nothing from here so the CLI can print a capability request without dragging
in the walker. The blocklist itself is untouched by the tiering: a capability
is an OVERLAY that says which refusals a declaration can lift, so the module
sets the in-canvas policy derives from :func:`dangerous_modules` are exactly
what they were.
"""

from __future__ import annotations

import ast
from typing import Iterable

from .security_tiers import (
    TIER0_PATH_HELPERS,
    TIER0_PATH_MODULE,
    TIER0_PATH_ROOT,
    capability_for_module,
    describe_capability,
    granted_modules,
)

_TIER0_PATH_HELPERS: frozenset[str] = frozenset(TIER0_PATH_HELPERS)


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
    # File IO that is not spelled ``open``. Every one of these reads or
    # writes a real path, and several also execute (``zipfile`` extracting
    # over a path, ``sqlite3`` loading an extension). They were missing, so a
    # library re-exporting one was a handover the value rules could not see.
    "zipfile", "gzip", "tarfile", "bz2", "lzma", "codecs",
    "sqlite3", "glob", "fileinput", "webbrowser",
    # ``os.path`` under its real names. These look like pure string helpers
    # and are not: each one does ``import os`` and ``import sys`` at module
    # level and leaves both bound as ordinary attributes, so
    # ``import ntpath; ntpath.sys.modules['subprocess'].run([...])`` is
    # arbitrary command execution through a module nobody thinks of as a
    # capability. That route predates core#133 -- these three were simply
    # never on the list -- and it is the same door the ``os.path`` import
    # exception has to keep shut.
    "ntpath", "posixpath", "genericpath",
    # core#183 -- ``nt`` (Windows) / ``posix`` (POSIX) are the raw C modules
    # ``os`` itself is built on. CPython's own ``Lib/os.py`` does
    # ``from nt import *`` (or ``from posix import *``), which is where
    # every ``os.remove``, ``os.environ`` and ``os.system`` originates, then
    # ``del``s the name -- so ``os.nt`` / ``os.posix`` never exist as an
    # attribute, and the only way back to the module is ``import nt`` /
    # ``import posix`` directly. Neither name was on this set, in
    # ``CAPABILITY_MODULES["process-env"]``, or in
    # ``TIER0_PURE_COMPUTE_MODULES`` -- never enumerated anywhere, in either
    # direction. ``import nt`` reached a real, writable ``nt.environ`` and a
    # real ``nt.remove(path)`` with zero capability declared; only
    # ``nt.system`` / ``nt.popen`` tripped anything, and only because
    # ``_DANGEROUS_ATTR_LEAVES`` below is receiver-independent and has no
    # idea what module it is looking at. Gated by ``process-env``, same as
    # ``os`` -- they ARE ``os``, under the name it briefly imports them by.
    "nt", "posix",
    # core#177 CI round 2 -- the enumeration test meeting a real Linux
    # interpreter surfaced 19-20 unclassified names (platform- and
    # version-dependent). Two earned an outright decision here rather than
    # deferral to ``ACCEPTED_UNGATED_MODULES`` in security_tiers.py, each
    # independently verified, not assumed to be dangerous by name alone:
    #
    # ``readline`` -- ``readline.add_history(s)`` accepts an arbitrary
    # string, and ``readline.write_history_file(path)`` writes the
    # accumulated history to an arbitrary path: an attacker-directed write of
    # attacker-directed content, verified directly (wrote a marked payload,
    # read the target file back and confirmed it landed). Gated by
    # ``filesystem``, the same bucket ``pathlib`` / ``tempfile`` / ``shutil``
    # already sit in -- it is, once you look past the "line editing" framing,
    # a file-writing library like the rest of that group.
    #
    # ``spwd`` -- reads the shadow password-hash database
    # (``spwd.getspnam()``). The module's own docstring says "You have to be
    # root to be able to use this module"; verified directly that this is
    # not reliably true: on a real system, a plain ``open("/etc/shadow")``
    # correctly raised ``PermissionError`` for an unprivileged account not in
    # the ``shadow`` group, but ``spwd.getspnam()`` from the SAME account
    # succeeded regardless -- NSS-mediated lookups can bypass the file's own
    # permission bits. No existing capability fits "read the shadow
    # database"; Tier 2 only, the same bucket as ``pickle`` / ``ctypes``.
    "readline", "spwd",
})

# Attribute-access patterns that are RCE in disguise whatever the receiver
# is: no legitimate caller reaches these leaves.
_DANGEROUS_ATTR_LEAVES = frozenset({
    "system", "popen", "spawnl", "spawnv",       # os
    "execfile", "compile_command",                # code / codeop
})

# Receivers whose ``.load`` / ``.loads`` deserialize by executing. Matched
# against the RESOLVED root of the attribute chain (see ``_import_roots``),
# so ``torch.hub.load``, ``t.load`` after ``import torch as t``, and a bare
# pre-bound ``numpy.load`` are all the same rule — and ``json.loads`` is not
# caught by it, which it used to be.
_UNPICKLING_RECEIVERS = frozenset({
    "torch", "numpy", "np", "pickle", "joblib", "dill", "marshal",
    "shelve", "pandas",
})

# Dunder names that should *never* appear inside a plugin. This is a list of
# KNOWN escape primitives, not a closed category — read it as "these specific
# doors are shut", never as "reflection is handled". Includes the names used by:
#   * ``().__class__.__bases__[0].__subclasses__()[N](...)``  — class walk
#   * ``getattr(__builtins__, "exec")``                       — builtins escape
#   * ``func.__globals__["__builtins__"]``                    — globals escape
#   * ``some_code.__code__.co_consts``                         — bytecode peek
#   * ``exc.__traceback__.tb_frame.f_back.f_globals``          — frame walk
_FORBIDDEN_DUNDERS = frozenset({
    "__class__", "__bases__", "__base__", "__mro__", "__subclasses__",
    "__builtins__", "__globals__", "__import__", "__dict__",
    "__code__", "__closure__", "__func__", "__self__",
    "__getattribute__", "__getattr__", "__setattr__", "__delattr__",
    "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    "__init_subclass__", "__class_getitem__",
    "__loader__", "__spec__", "__package__",
    "__traceback__",
})

# Frame-walking attributes. These carry no leading underscores, so none of the
# dunder rules above ever saw them, and every one of them is a step on a path
# from an ordinary object back to the *host's* module globals:
#
#     try:
#         raise ValueError()
#     except ValueError as e:
#         g = e.__traceback__.tb_frame.f_back.f_globals   # the caller's globals
#         g['importlib'].import_module('os').system(...)  # the real os module
#
#     it = mygen(); it.gi_frame.f_globals                 # same, via a generator
#
# The restricted ``__builtins__`` a script is executed with is irrelevant once
# a frame object is in hand: ``f_globals`` and ``f_builtins`` belong to whoever
# called you, and the node's own module imports ``builtins`` and ``importlib``.
# Blocked whatever the receiver is, exactly like the dunder set — nothing
# legitimately reads another frame's state.
_FRAME_INTROSPECTION_ATTRS = frozenset({
    "tb_frame", "tb_next",                            # traceback objects
    "f_back", "f_globals", "f_locals", "f_builtins",  # frame objects
    "f_code", "f_trace",
    "gi_frame", "gi_code",                            # generators
    "cr_frame", "cr_code",                            # coroutines
    "ag_frame", "ag_code",                            # async generators
})

# The module that DEFINES the names in ``_DANGEROUS_NAMES``.
#
# Those names are refused as CALLS, and until core#133 the rule matched a
# call's leaf whatever the receiver was -- so ``re.compile(r'\d+')``,
# ``torch.compile(model)`` and ``model.eval()`` were all refused as if they
# were the builtin. That is a false positive with no security value: an
# attribute-leaf ``compile`` is whatever the receiver defines, and the receiver
# that defines the real one is ``builtins``.
#
# So the call rule now asks WHOSE ``eval`` this is, and only this module's
# answer is refused. ``__builtins__`` is here for symmetry; the bare name is
# already refused as a forbidden dunder, and the assignment-following in
# ``_import_roots`` means ``import builtins as b; b.eval(s)`` resolves back to
# the same verdict.
_BUILTINS_RECEIVERS = frozenset({"builtins", "__builtins__"})

# Names on the real ``os.path`` that ARE modules -- ``os``, ``sys``, ``stat``,
# ``genericpath`` on both platforms today.
#
# Computed rather than listed, because the whole failure this closes was a
# name screen: ``from os.path import genericpath`` cleared a blocklist-name
# check and handed back a module whose ``.os`` is the real thing. Asking the
# interpreter what is actually a module there is the same "what is it, not
# what is it called" move the runtime proxy makes, and it covers whatever a
# future CPython re-exports without an edit here.
_OS_PATH_MODULE_LEAVES: frozenset[str] = frozenset()


def _compute_os_path_module_leaves() -> frozenset[str]:
    import os.path
    import types as _types

    return frozenset(
        name
        for name in dir(os.path)
        if isinstance(getattr(os.path, name, None), _types.ModuleType)
    )


_OS_PATH_MODULE_LEAVES = _compute_os_path_module_leaves()


def dangerous_modules() -> frozenset[str]:
    """Public view of the default blocklist (mainly for tests and error messages)."""
    return _DANGEROUS_MODULES


def dangerous_attr_leaves() -> frozenset[str]:
    """Public view of the RCE attribute leaves (``system``, ``popen``, ...).

    Exposed so a stricter policy can refuse these as ATTRIBUTES rather than
    only as calls. The Call-keyed form here is evaded by one assignment --
    ``f = obj.system`` then ``f(cmd)`` -- and closing that for every caller
    would condemn any plugin with a ``self.system`` attribute, so the tier
    that wants it opts in through ``denied_attributes``.
    """
    return _DANGEROUS_ATTR_LEAVES


def forbidden_dunders() -> frozenset[str]:
    """Public view of the dunder blocklist (mainly for tests)."""
    return _FORBIDDEN_DUNDERS


def frame_introspection_attrs() -> frozenset[str]:
    """Public view of the frame-walking blocklist (mainly for tests)."""
    return _FRAME_INTROSPECTION_ATTRS


def _import_roots(tree: ast.AST) -> dict[str, str]:
    """Map each name that BINDS a module to the top-level module behind it.

    ``import numpy as np`` -> ``{"np": "numpy"}``; ``import torch.nn as nn``
    -> ``{"nn": "torch"}``; ``from torch import hub`` -> ``{"hub": "torch"}``.
    Without this, every receiver rule could only match code that spells the
    module out, and ``import torch as t; t.load(...)`` walked straight past
    the pickle gate.

    Plain assignments are followed for one hop as well, because an import
    alias was never the only way to rename a module: ``b = torch`` then
    ``b.load(x)`` is the same laundering with none of the import syntax, and
    it worked. This is a flat, order-insensitive approximation -- a name
    rebound twice resolves to whichever assignment ``ast.walk`` reached last.
    That is deliberate: the receiver rules treat an unresolved name as
    suspicious in allowlist mode, so a miss here costs a clearer message, not
    a hole.
    """
    roots: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c``.
                roots[alias.asname or top] = top
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            for alias in node.names:
                roots[alias.asname or alias.name] = top

    # Second pass: ``b = torch``, ``n = numpy``, ``h = torch.hub``. Runs after
    # the imports so an alias-of-an-alias resolves through them.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        source = _receiver_root(node.value)
        if source is None:
            continue
        resolved = roots.get(source, source)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id != resolved:
                roots[target.id] = resolved
    return roots


def _receiver_root(func: ast.expr) -> str | None:
    """Base name of an attribute chain: ``a.b.c`` -> ``"a"``, ``x[0].c`` -> None."""
    node = func
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


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


def _calls_the_builtin(func: ast.expr, roots: dict[str, str]) -> bool:
    """Whether a call to a ``_DANGEROUS_NAMES`` leaf is the BUILTIN of that name.

    A bare ``eval(s)`` is. So is ``builtins.eval(s)``, and so is
    ``import builtins as b; b.eval(s)`` -- ``_import_roots`` resolves the
    alias, and follows a plain assignment for one hop as well.

    ``re.compile``, ``torch.compile`` and ``model.eval()`` are not, and were
    refused anyway until core#133 because the rule matched the leaf and
    ignored the receiver (core#178, core#174). Nothing was gained by it: an
    attribute-leaf ``compile`` is whatever the receiver defines, and the only
    receiver that defines the real one is the module this checks for. The
    in-canvas tier leans on the same fact from the other side -- its runtime
    proxy refuses the builtin ``compile``/``eval``/``open`` by IDENTITY under
    any name, and ``builtins`` is on its denied-attribute list -- so the
    relaxation costs that tier nothing either.

    An unresolvable receiver (``things[0].compile()``) answers False: for an
    installed file the user chose the code, and for the script tier the object
    that would have to be hiding there cannot be reached.
    """
    if isinstance(func, ast.Name):
        return True
    if isinstance(func, ast.Attribute):
        base = _receiver_root(func)
        if base is None:
            return False
        return roots.get(base, base) in _BUILTINS_RECEIVERS
    return False


def _check_getattr_arg_safety(
    call: ast.Call,
    denial_hint: str = "",
    denied_attrs: frozenset[str] = frozenset(),
    *,
    roots: dict[str, str] | None = None,
    safe_load_receivers: frozenset[str] | None = None,
    denied_names: frozenset[str] = frozenset(),
    scoped_attrs: dict[str, frozenset[str]] | None = None,
) -> None:
    """Disallow ``getattr(<dunder-like>, ...)`` even when arg 1 is a literal.

    *denied_attrs* closes the spelling-it-differently route: a literal
    ``getattr(numpy, 'savetxt')`` reaches the same function the attribute
    rule refuses, and reads as a deliberate attempt to get around it. The
    frame-walking names and ``.load`` / ``.loads`` are closed here for the
    same reason -- every rule enforced on ``ast.Attribute`` has to be
    enforced here too, or the rule is decoration.
    """
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
        if isinstance(second, ast.Constant) and isinstance(second.value, str):
            if second.value in _FORBIDDEN_DUNDERS:
                raise PluginValidationError(
                    f"getattr() retrieving forbidden dunder "
                    f"{second.value!r} is not allowed{denial_hint}",
                    lineno=call.lineno,
                )
            if second.value in _FRAME_INTROSPECTION_ATTRS:
                raise PluginValidationError(
                    f"getattr() retrieving frame attribute {second.value!r} "
                    f"is not allowed{denial_hint}",
                    lineno=call.lineno,
                )
            if second.value in denied_attrs:
                raise PluginValidationError(
                    f"getattr() retrieving {second.value!r} is not allowed"
                    f"{denial_hint}",
                    lineno=call.lineno,
                )
            if second.value in ("load", "loads"):
                _check_getattr_load(
                    call,
                    second.value,
                    denial_hint,
                    roots=roots,
                    safe_load_receivers=safe_load_receivers,
                )
            receiver = _receiver_root(first)
            if second.value in denied_names:
                resolved = (roots or {}).get(receiver, receiver)
                if resolved in _BUILTINS_RECEIVERS:
                    raise PluginValidationError(
                        f"getattr() retrieving {second.value!r} from the "
                        f"'builtins' module is not allowed{denial_hint}",
                        lineno=call.lineno,
                    )
            if scoped_attrs:
                message = _receiver_scope_violation(
                    second.value, receiver, roots, scoped_attrs
                )
                if message is not None:
                    raise PluginValidationError(
                        f"getattr(): {message}{denial_hint}", lineno=call.lineno
                    )


def _check_getattr_load(
    call: ast.Call,
    leaf: str,
    denial_hint: str,
    *,
    roots: dict[str, str] | None,
    safe_load_receivers: frozenset[str] | None,
) -> None:
    """Apply the ``.load`` receiver rule to ``getattr(x, 'load')``.

    ``getattr(torch, 'load')(path)`` calls the same pickle-executing function
    as ``torch.load(path)`` and never writes an attribute access, so the
    Attribute branch never sees it. Resolve the receiver the same way
    :func:`_enforce_safe_load` does and apply the same verdict; the keyword
    escape hatch (``weights_only=True``) does not apply, because the kwargs
    land on the *outer* call, which is a Call-of-a-Call this walker cannot
    tie back to the getattr.
    """
    receiver = _receiver_root(call.args[0])
    resolved = (roots or {}).get(receiver, receiver) if receiver is not None else None
    if safe_load_receivers is not None:
        if resolved in safe_load_receivers:
            return
        raise PluginValidationError(
            f"getattr(..., {leaf!r}) is not allowed; under this policy "
            f"'.{leaf}(...)' is permitted only on "
            f"{', '.join(sorted(safe_load_receivers))}{denial_hint}",
            lineno=call.lineno,
        )
    if resolved in _UNPICKLING_RECEIVERS:
        raise PluginValidationError(
            f"getattr({receiver}, {leaf!r}) is not allowed; it retrieves a "
            f"function that executes code from the file it reads"
            f"{denial_hint}",
            lineno=call.lineno,
        )


def _receiver_scope_violation(
    attr: str,
    receiver: str | None,
    roots: dict[str, str] | None,
    scoped: dict[str, frozenset[str]],
) -> str | None:
    """Message when *attr* is reached on a receiver its scope does not allow.

    ``receiver_scoped_attrs`` is the "denied everywhere EXCEPT these roots"
    shape, and it exists because two attributes with the same spelling can be
    a Tier-0 module's headline function and a compiler entry point at the same
    time: ``re.compile`` is the former, ``torch.compile`` is the latter
    (TorchInductor generates C++/Triton and shells out to a real compiler,
    which is the same door round 5 of core#131 closed by refusing
    ``torch.utils.cpp_extension``).

    Fails closed on an unresolvable receiver, exactly like
    *safe_load_receivers*: ``(lambda: torch)().compile`` is what that is for.
    """
    allowed_roots = scoped.get(attr)
    if allowed_roots is None:
        return None
    resolved = (roots or {}).get(receiver, receiver) if receiver is not None else None
    if resolved in allowed_roots:
        return None
    shown = f"'{receiver}'" if receiver else "this expression"
    return (
        f"Access to '.{attr}' on {shown} is not allowed; under this policy "
        f"'.{attr}' is available only on "
        f"{', '.join(sorted(allowed_roots))}"
    )


def _is_tier0_path_import(
    module: str | None, names: Iterable[str], from_import: bool
) -> bool:
    """Whether this import is the ``os.path`` slice Tier 0 keeps.

    **One** form, and every leaf it binds must be on an explicit allowlist of
    pure string helpers::

        from os.path import join, basename, splitext    # the exception
        from os.path import expandvars                  # NOT -- reads os.environ
        from os.path import exists, getsize             # NOT -- real stat()
        from os.path import genericpath                 # NOT -- a module
        from os import path                             # NOT -- binds ntpath
        import os / import os.path                      # NOT -- both bind ``os``

    Two review rounds landed on this shape, and the way they landed is the
    point. The first cut allowed ``from os import path`` and screened leaves
    against the blocklist by NAME; that handed a zero-declaration plugin the
    real ``os`` and ``sys``, verified end to end. The second screened leaves
    by whether they ARE modules, which closed that -- and was still scoped to
    the escape that had been *demonstrated* rather than to the property that
    made it possible: **``os.path`` is a real module and its surface is not
    string manipulation.** ``expandvars`` reads ``os.environ`` (it returned a
    real API key), ``expanduser`` returns the home directory, and
    ``exists`` / ``isfile`` / ``getsize`` call ``stat()`` on any path given.
    None is a module, so no module screen could ever have seen them.

    Hence an allowlist -- :data:`app.core.security_tiers.TIER0_PATH_HELPERS`,
    every entry verified by CALLING it against the live ``os.path``. The
    module screen stays as a second lock (a name added to the allowlist that
    later becomes a module still fails) and because it is free.

    ``*`` is refused: ``ntpath.__all__`` includes ``exists``, ``expanduser``
    and ``getsize``, so a star import is exactly the leak this closes.

    ``ntpath``, ``posixpath`` and ``genericpath`` are on the blocklist now
    too, so the route that never needed this exception at all
    (``import ntpath; ntpath.os...``, live since long before core#133) is
    closed with it.
    """
    if not from_import or module != TIER0_PATH_MODULE:
        return False
    leaves = set(names)
    if not leaves or not leaves <= _TIER0_PATH_HELPERS:
        return False
    # Belt and braces: the allowlist is a list of names, and a name is not a
    # boundary. If a future CPython turns one of them into a module, the
    # structural screen still refuses it.
    return not (leaves & _OS_PATH_MODULE_LEAVES) and not (leaves & _DANGEROUS_MODULES)


def _capability_denial(module: str, root: str, filename: str) -> str:
    """The refusal, attributed to the tier that would unlock *root*.

    The old message was ``Importing 'requests' is not allowed`` and stopped
    there, which reads as a wall rather than as a door with a key. Every
    refusal now names the tier: a capability the manifest can ask for, or
    ``--trust-author`` for the roots no capability will ever cover.
    """
    capability = capability_for_module(root)
    if capability is None:
        return (
            f"Importing '{module}' is not allowed in {filename}: no capability "
            f"grants '{root}', because it can execute code or reach the "
            f"interpreter hosting this plugin. A plugin that genuinely needs "
            f"it has to be installed with --trust-author and list it in "
            f"[security].allowed_modules"
        )
    hint = ""
    if root == TIER0_PATH_ROOT:
        hint = (
            ". If you only need path helpers, "
            "'from os.path import join, basename' needs no capability at all"
        )
    return (
        f"Importing '{module}' is not allowed in {filename}: it requires "
        f"capability '{capability}' -- permission to "
        f"{describe_capability(capability)}. Declare it in the plugin "
        f"manifest as [security] capabilities = [\"{capability}\"]; "
        f"'cdui plugin install' then shows the request and asks the user to "
        f"confirm before anything is written{hint}"
    )


def validate_python_source(
    content: bytes | str,
    filename: str = "<plugin>",
    *,
    allowed_modules: Iterable[str] | None = None,
    capabilities: Iterable[str] | None = None,
    import_allowlist: Iterable[str] | None = None,
    extra_denied_names: Iterable[str] | None = None,
    denied_attributes: Iterable[str] | None = None,
    safe_load_receivers: Iterable[str] | None = None,
    receiver_scoped_attrs: dict[str, Iterable[str]] | None = None,
    library_roots: Iterable[str] | None = None,
    denial_hint: str = "",
) -> None:
    """Parse *content* and raise if it contains obviously dangerous patterns.

    *allowed_modules* widens the import blocklist for plugins that legitimately
    need one of the default-banned top-level modules (declared in their
    manifest under ``[security].allowed_modules`` and accepted by the user
    via ``--trust-author``). Dangerous builtin *calls* and dunder attribute
    access are never widened.

    *capabilities* is core#133's middle answer: the capability names a
    manifest declared under ``[security].capabilities`` and the user confirmed
    at install time. Each one unlocks the module GROUP
    :data:`app.core.security_tiers.CAPABILITY_MODULES` maps it to and nothing
    else -- ``"network"`` does not bring ``subprocess`` with it, and no
    capability lifts the pickle, dunder or RCE-leaf rules. Absent (the default,
    and what a pre-#133 lockfile means) is the empty set, which is exactly the
    behaviour every caller had before.

    *import_allowlist* switches the import rule from blocklist to allowlist:
    a module whose FULL DOTTED NAME is outside the set is refused whatever it
    is, and relative imports (which name no module at all) are refused with
    it. Paths, not roots — ``torch.utils`` and ``numpy.f2py`` have an
    allowlisted root and each one held a working escape. This is the shape
    in-canvas scripts use — see :mod:`app.core.script_policy`.

    *extra_denied_names* adds to the denied *call* names for this call only
    (``open``, ``input``, ... for scripts). *denied_attributes* refuses whole
    attribute names on top of the dunder set — the caller's list of "this
    library door is closed" leaves (``torch.hub``, ``numpy.savetxt``).

    *safe_load_receivers* flips the pickle rule from blocklist to allowlist
    the same way *import_allowlist* flips the import rule: ``.load(...)`` /
    ``.loads(...)`` is refused on every receiver EXCEPT the named ones. The
    blocklist form only ever matched a receiver it could resolve to a known
    unpickler, which made it launderable in four different ways
    (``b = torch; b.load(x)``, ``getattr(torch, 'load')(x)``,
    ``(lambda: torch)().load(x)``, ``things[0].load(x)``). Callers that pass
    it should expect the cost: a script's *own* ``obj.load()`` helper is
    refused too, exactly as its own ``obj.save()`` already is under
    *denied_attributes*.

    *receiver_scoped_attrs* is the same inversion applied per ATTRIBUTE NAME:
    ``{"compile": ("re",)}`` refuses ``.compile`` on everything except a
    receiver resolving to ``re``. It is what lets one spelling be a Tier-0
    module's headline function (``re.compile``) and a compiler entry point
    (``torch.compile``, which makes TorchInductor generate C++/Triton and
    invoke a real compiler) without the policy having to pick one. Like
    *safe_load_receivers* it fails closed on a receiver the walker cannot
    resolve.

    *library_roots* names the receivers that are LIBRARIES rather than the
    caller's own objects, which lets two rules be stated structurally instead
    of by enumeration:

    * their PRIVATE attributes are refused -- ``collections._sys`` *is* the
      real ``sys`` module, and no list of forbidden names will ever contain
      every private alias in every library;
    * assigning to their attributes is refused -- ``torch.zeros = mine``
      rebinds the module for every other node in the process.

    Both are opt-in because ``self._cache`` and ``self.x = 1`` are ordinary
    code; only a *library* receiver makes them suspicious.

    *denial_hint* is appended to every message raised here, so one caller can
    point users at the escape hatch its policy implies without every rule
    growing a special case.
    """
    allowed = frozenset(allowed_modules) if allowed_modules else frozenset()
    unlocked = granted_modules(capabilities)
    allowlist = frozenset(import_allowlist) if import_allowlist is not None else None
    denied_names = _DANGEROUS_NAMES | frozenset(extra_denied_names or ())
    denied_attrs = frozenset(denied_attributes or ())
    libraries = frozenset(library_roots or ())
    load_receivers = (
        frozenset(safe_load_receivers) if safe_load_receivers is not None else None
    )
    scoped_attrs = {
        attr: frozenset(allowed_roots)
        for attr, allowed_roots in (receiver_scoped_attrs or {}).items()
    }

    def fail(message: str, node: ast.AST | None = None) -> PluginValidationError:
        return PluginValidationError(
            f"{message}{denial_hint}", lineno=getattr(node, "lineno", None)
        )

    def check_import(
        module: str | None,
        node: ast.AST,
        *,
        level: int = 0,
        names: Iterable[str] = (),
        from_import: bool = False,
    ) -> None:
        """Apply whichever import rule this call selected.

        *names* are the leaves an ``ImportFrom`` binds. They matter because
        importing a denied attribute BY NAME opens exactly the door the
        attribute rule closes, without ever writing an attribute access:
        ``from torch.utils.cpp_extension import load_inline`` then calls a
        bare ``load_inline(...)``. The same goes for a denied component
        anywhere in the dotted path.

        *from_import* distinguishes ``from X import y`` from ``import X``,
        which matters for exactly one rule -- the ``os.path`` exception below
        -- because the two forms bind different objects.
        """
        names = list(names)
        # ``from json import __builtins__ as b`` then ``b['eval']``. Every
        # OTHER spelling of a forbidden dunder is refused -- as a bare name, as
        # an attribute, as a subscript target, through a literal getattr -- but
        # an import BINDS it without writing any of those node types, so
        # nothing looked at it. Every module has a ``__builtins__``, so the
        # receiver did not even have to be interesting. Pre-existing, and it
        # applies to BOTH gate shapes (a script's ``json`` is on its allowlist),
        # which is why it runs before either verdict.
        for leaf in names:
            if leaf in _FORBIDDEN_DUNDERS:
                raise fail(
                    f"Importing the name {leaf!r} is not allowed in {filename}: "
                    f"importing a dunder binds it without writing an attribute "
                    f"access, which is the one spelling the other rules cannot "
                    f"see",
                    node,
                )
        # Allowlist verdict first, so a module that is simply off the list
        # reads as "not allowed" rather than as whatever the denied-attribute
        # rule happens to say about one component of its name. The root check
        # only clears the ROOT, so the component scan below still has to run:
        # ``import torch.hub`` has an allowlisted root and a closed leaf.
        if allowlist is not None:
            if level or not module:
                raise fail(f"Relative imports are not allowed in {filename}", node)
            # The FULL dotted name, not just the root: ``torch.utils`` and
            # ``numpy.f2py`` have an allowlisted root and each one held a
            # working escape, so *import_allowlist* is a list of module PATHS
            # and every component of the path has to be on it.
            if module not in allowlist:
                raise fail(f"Importing '{module}' is not allowed in {filename}", node)
        if denied_attrs:
            for part in (module or "").split("."):
                if part in denied_attrs:
                    raise fail(
                        f"Importing '{module}' is not allowed in {filename}: "
                        f"'{part}' is closed under this policy",
                        node,
                    )
            for leaf in names:
                if leaf in denied_attrs:
                    raise fail(
                        f"Importing '{leaf}' is not allowed in {filename}",
                        node,
                    )
        if allowlist is not None:
            return
        # ``from builtins import eval`` binds a bare ``eval`` that no Call
        # rule below can tell from an ordinary local. The name rules refuse
        # ``eval(...)`` and ``builtins.eval(...)``; this is the third
        # spelling, and it is the only one that needs the import to see it.
        if module == "builtins":
            for leaf in names:
                if leaf in denied_names:
                    raise fail(
                        f"Importing '{leaf}' from 'builtins' is not allowed in "
                        f"{filename}: binding it to a name does not make it a "
                        f"different function",
                        node,
                    )
        root = (module or "").split(".")[0]
        if not root or root not in _DANGEROUS_MODULES or root in allowed:
            return
        if _is_tier0_path_import(module, names, from_import):
            return
        if root in unlocked:
            return
        raise fail(_capability_denial(module or root, root, filename), node)

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

    roots = _import_roots(tree)

    def check_library_mutation(targets: Iterable[ast.expr], node: ast.AST) -> None:
        """``torch.zeros = mine`` rebinds the module for the whole process.

        Module objects are shared with the host, so poisoning one is not a
        script-local act -- every other node in the run sees it. The runtime
        proxy refuses the assignment outright; this is the same verdict early
        enough to be a red line in the editor.
        """
        if not libraries:
            return
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            base = _receiver_root(target)
            if base is not None and roots.get(base, base) in libraries:
                raise fail(
                    f"Assigning to '{base}.{target.attr}' is not allowed in "
                    f"{filename}: library modules are shared with the host "
                    f"process, so rebinding one changes it for every node",
                    node,
                )

    for node in ast.walk(tree):
        # ── Import / ImportFrom ──────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_import(alias.name, node)
        elif isinstance(node, ast.ImportFrom):
            check_import(
                node.module,
                node,
                level=node.level or 0,
                names=[alias.name for alias in node.names],
                from_import=True,
            )

        # ── Assignment onto a library module ─────────────────────────
        elif isinstance(node, ast.Assign):
            check_library_mutation(node.targets, node)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            check_library_mutation([node.target], node)
        elif isinstance(node, ast.Delete):
            check_library_mutation(node.targets, node)

        # ── Bare names ───────────────────────────────────────────────
        #
        # The escape this closes: ``__loader__`` is an ordinary module global
        # (CPython's BuiltinImporter), so ``__loader__.load_module('nt')``
        # hands back the real ``os`` module without an import statement, and
        # ``b = __builtins__`` launders the subscript guard below in one
        # assignment. Neither is an Attribute or a Subscript at the point
        # that matters, so nothing but a Name check sees them.
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_DUNDERS:
                raise fail(
                    f"Use of the name {node.id!r} is not allowed in {filename}",
                    node,
                )

        # ── Attribute access (Foo.__class__, x.__bases__, etc.) ──────
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_DUNDERS:
                raise fail(
                    f"Access to attribute {node.attr!r} is not allowed in {filename}",
                    node,
                )
            if node.attr in _FRAME_INTROSPECTION_ATTRS:
                raise fail(
                    f"Access to frame attribute {node.attr!r} is not allowed "
                    f"in {filename}: reading another frame's state reaches the "
                    f"host process's own globals",
                    node,
                )
            if node.attr in denied_attrs:
                raise fail(
                    f"Access to '.{node.attr}' is not allowed in {filename}",
                    node,
                )
            # ``builtins.eval`` is the builtin ``eval``, and binding it to a
            # name first (``e = builtins.eval``) means no Call-keyed rule ever
            # sees it. Refused as an ATTRIBUTE for the same reason ``.load``
            # is, and scoped to the one receiver that makes it true.
            if node.attr in denied_names:
                base = _receiver_root(node)
                if base is not None and roots.get(base, base) in _BUILTINS_RECEIVERS:
                    raise fail(
                        f"Reading '.{node.attr}' from the 'builtins' module is "
                        f"not allowed in {filename}: binding it to a name does "
                        f"not make it a different function",
                        node,
                    )
            if scoped_attrs:
                message = _receiver_scope_violation(
                    node.attr, _receiver_root(node), roots, scoped_attrs
                )
                if message is not None:
                    raise fail(f"{message} in {filename}", node)
            # A library's PRIVATE names are how its own imports leak out:
            # ``collections._sys`` is the real ``sys``, ``statistics.random``
            # then ``._os`` is the real ``os``. Refused only for the receivers
            # the caller named, because ``self._cache`` is ordinary code.
            if libraries and node.attr.startswith("_"):
                base = _receiver_root(node)
                if base is not None and roots.get(base, base) in libraries:
                    raise fail(
                        f"Access to the private attribute '.{node.attr}' of "
                        f"'{base}' is not allowed in {filename}: a library's "
                        f"private names are where its own imports live",
                        node,
                    )
            # ``f = torch.load`` then ``f(path)``: the receiver rule below
            # fires on the CALL, and one assignment means there is no call
            # with a ``.load`` leaf to fire on. In allowlist mode the same
            # verdict is applied to the ATTRIBUTE, so the name cannot be
            # carried out of reach of the rule.
            if load_receivers is not None and node.attr in ("load", "loads"):
                base = _receiver_root(node)
                if base is None or roots.get(base, base) not in load_receivers:
                    shown = f"'{base}'" if base else "this expression"
                    raise fail(
                        f"Reading '.{node.attr}' from {shown} is not allowed in "
                        f"{filename}; under this policy only "
                        f"{', '.join(sorted(load_receivers))} may be loaded "
                        f"from, and binding the function to a name does not "
                        f"change what it does",
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
            if name in denied_names and _calls_the_builtin(node.func, roots):
                if name in ("getattr", "setattr", "delattr"):
                    # Tighter version of the original exception: literal
                    # 2nd-arg string is still allowed, but only after we
                    # verify it isn't being used to retrieve a forbidden
                    # dunder or applied to a forbidden first-arg name.
                    _check_getattr_arg_safety(
                        node,
                        denial_hint,
                        denied_attrs,
                        roots=roots,
                        safe_load_receivers=load_receivers,
                        denied_names=denied_names,
                        scoped_attrs=scoped_attrs,
                    )
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
                if node.func.attr in ("load", "loads"):
                    _enforce_safe_load(
                        node, filename, denial_hint, roots, load_receivers
                    )
                elif node.func.attr in _DANGEROUS_ATTR_LEAVES:
                    raise fail(
                        f"Call to '.{node.func.attr}(...)' is not allowed in {filename}",
                        node,
                    )


def _enforce_safe_load(
    call: ast.Call,
    filename: str,
    denial_hint: str = "",
    roots: dict[str, str] | None = None,
    safe_load_receivers: frozenset[str] | None = None,
) -> None:
    """Gate ``X.load(...)`` / ``X.loads(...)`` on what X actually is.

    Catches the pickle-RCE pattern behind ``torch.load(...)`` /
    ``numpy.load(allow_pickle=True)``. The receiver is resolved through the
    file's import aliases and down the whole attribute chain, so
    ``torch.hub.load`` (which downloads and EXECUTES a remote ``hubconf.py``)
    and ``t.load`` after ``import torch as t`` are caught alongside the
    spelled-out form — both of which walked straight through the old
    ``isinstance(func.value, ast.Name)`` check.

    It is equally a fix in the other direction: the leaf name alone used to
    condemn ``json.loads(...)``, a Tier-0 module's headline function, and told
    the user to go write a custom node to parse JSON.

    With *safe_load_receivers* the verdict inverts: anything that does not
    resolve to a named-safe receiver is refused, which is the only form that
    survives a receiver the walker cannot resolve (``(lambda: torch)()``,
    ``things[0]``). See :func:`validate_python_source` for the tradeoff.

    In that form the ``weights_only=True`` escape hatch does NOT apply. The
    runtime module proxy that backs this policy hands out attributes, not
    calls: it cannot see a keyword argument, so ``torch.load`` reachable *at
    all* means ``f = torch.load; f(p)`` reachable, kwargs and all. A tier that
    inverts the rule gets no file reading either -- ``open`` is denied there
    too -- so the two agree instead of the gate promising what the runtime
    refuses.
    """
    leaf = call.func.attr if isinstance(call.func, ast.Attribute) else "load"
    if safe_load_receivers is not None:
        _enforce_load_allowlist(call, leaf, filename, denial_hint, roots,
                                safe_load_receivers)
        return
    # numpy.load with allow_pickle=True → reject
    for kw in call.keywords:
        if kw.arg == "allow_pickle" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            raise PluginValidationError(
                f"Call to '.{leaf}(allow_pickle=True)' is not allowed in {filename}; "
                f"it can execute arbitrary code from the source file{denial_hint}",
                lineno=call.lineno,
            )
    # torch.load → require explicit weights_only=True
    for kw in call.keywords:
        if kw.arg == "weights_only":
            if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return  # explicit safe call → OK
            raise PluginValidationError(
                f"Call to '.{leaf}(weights_only={ast.unparse(kw.value)})' is not allowed "
                f"in {filename}; only weights_only=True is permitted{denial_hint}",
                lineno=call.lineno,
            )
    # No weights_only / allow_pickle kwarg supplied. Resolve the receiver.
    receiver = _receiver_root(call.func)
    resolved = (roots or {}).get(receiver, receiver) if receiver is not None else None

    # Blocklist form (installed plugins and custom nodes). ``json.load`` /
    # ``yaml.safe_load`` are fine, ``torch.*.load`` / ``np.load`` /
    # ``pickle.loads`` are not. An unresolvable receiver (``things[0].loads()``)
    # is left alone -- the user chose to install this file, and condemning
    # every method called ``loads`` was a false positive with no matching
    # security value.
    if receiver is None:
        return
    if resolved in _UNPICKLING_RECEIVERS:
        shown = receiver if receiver == resolved else f"{receiver} ({resolved})"
        raise PluginValidationError(
            f"Bare {shown}.{leaf}(...) is not allowed in {filename}; "
            f"pass weights_only=True explicitly to make the intent "
            f"obvious{denial_hint}",
            lineno=call.lineno,
        )


def _enforce_load_allowlist(
    call: ast.Call,
    leaf: str,
    filename: str,
    denial_hint: str,
    roots: dict[str, str] | None,
    safe_load_receivers: frozenset[str],
) -> None:
    """Allowlist form of the pickle rule: only named receivers may be loaded.

    A receiver the walker cannot resolve -- the whole point of
    ``(lambda: torch)().load(x)`` and ``things[0].load(x)`` -- fails closed
    instead of being waved through.
    """
    receiver = _receiver_root(call.func)
    resolved = (roots or {}).get(receiver, receiver) if receiver is not None else None
    if resolved in safe_load_receivers:
        return
    if receiver is None:
        shown = "this expression"
    elif receiver == resolved:
        shown = repr(receiver)
    else:
        shown = f"{receiver!r} ({resolved})"
    raise PluginValidationError(
        f"Calling '.{leaf}(...)' on {shown} is not allowed in {filename}; "
        f"under this policy only "
        f"{', '.join(sorted(safe_load_receivers))} may be loaded from, "
        f"because every other '.{leaf}' within reach executes code from "
        f"the file it reads{denial_hint}",
        lineno=call.lineno,
    )
