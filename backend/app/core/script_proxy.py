"""Restricted module proxies -- the RUNTIME boundary for in-canvas scripts.

Why this exists
---------------
The Tier-0 AST gate (:mod:`app.core.script_policy`) is keyed on NAMES: it
refuses ``torch.os`` because the attribute is spelled ``os``, and
``numpy.savetxt`` because the leaf is spelled ``savetxt``. Three consecutive
adversarial reviews walked through it, each time through a name nobody had
listed::

    collections._sys                      # the real sys, under an alias
    collections._sys.modules['os']        # every imported module, by subscript
    statistics.random._os.getcwd()        # os, two private hops away
    json.codecs.builtins.eval             # builtins, through an unlisted module
    f = osmod.system; f(cmd)              # bound local: no Call-keyed rule fires
    f = torch.load;   f(path)             # ditto for the pickle rule

A name is not a boundary. Every one of those escapes worked because the
execution namespace handed the script the host's REAL module objects, and a
real module hands over whatever it holds.

What this module does instead
-----------------------------
The namespace hands out :class:`RestrictedModule` proxies. A proxy answers an
attribute access by looking at the RESOLVED OBJECT:

* a module whose top-level package is on :data:`TIER0_MODULES` -> a nested
  proxy;
* a module that is NOT -> refused, whatever the attribute was called. This is
  the rule that kills ``collections._sys``, ``statistics.random``,
  ``json.codecs``, ``torch.cuda.tunable.mp`` and every future library
  reshuffle in one line, because it asks *what did I just get* rather than
  *what was it called*;
* anything else -> the plain value. A returned ``torch.Tensor`` is a tensor,
  not a proxy, so tensor work runs at full speed.

On top of the resolved-object rule the proxy refuses, by name: private and
dunder attributes (``_sys``, ``__loader__``, ``__globals__``), the frame
introspection attributes, the Tier-0 denied-attribute set (the doors inside
numpy and torch that lead back out), ``.load``/``.loads`` on anything but
``json``, subscripting (so ``proxy.modules['os']`` cannot work even if some
proxy did expose a mapping), calling, and mutation (so a script can no longer
rebind ``torch.zeros`` for the whole host process).

What it does NOT do
-------------------
This is still a guardrail, not a sandbox. It bounds the LIBRARY SURFACE a
script can navigate; it does not bound what the reachable functions do
(``numpy`` can still write files if a door is left open), it does not contain
CPU or memory, and it does not intercept attribute access on the PLAIN values
a library returns -- a module reached as an attribute of some non-module
object a library hands back is checked by the AST gate's name rules only. The
gate stays in place for exactly that reason, and for reflection on plain
objects (dunders, frames, ``getattr`` with a computed name), which no proxy
can see.
"""

from __future__ import annotations

import builtins
import io
import mmap
import os
import pathlib
import types
from typing import Any

from .plugin_validator import dangerous_modules, frame_introspection_attrs
from .script_policy import (
    ESCAPE_HATCH_HINT,
    SCRIPT_PROXY_DENIED_ATTRS,
    TIER0_MODULE_PATHS,
    TIER0_MODULES,
    TIER0_RECEIVER_SCOPED_ATTRS,
    TIER0_SAFE_LOAD_RECEIVERS,
    ScriptPolicyError,
)

#: The exact module paths a proxy may hand back. Anything else -- reached
#: under ANY attribute name, at ANY depth -- is refused.
#:
#: Keyed on the full dotted name rather than the root since review round 5:
#: ``torch.utils``, ``torch.package``, ``torch.fx``, ``torch.serialization``,
#: ``numpy.f2py``, ``numpy.testing`` and ``numpy.lib.format`` all have an
#: allowlisted root and each one held a working escape.
_ALLOWED_MODULE_PATHS: frozenset[str] = frozenset(TIER0_MODULE_PATHS)

#: Attribute names refused whatever they resolve to. The union of the Tier-0
#: denied doors, the blocked-module names, and the RCE leaves, so the runtime
#: and the gate cannot drift apart on what "closed" means.
_DENIED_ATTRS: frozenset[str] = frozenset(SCRIPT_PROXY_DENIED_ATTRS)

_FRAME_ATTRS: frozenset[str] = frame_introspection_attrs()

#: Roots whose ``.load`` / ``.loads`` is a parser rather than an unpickler.
_SAFE_LOAD_ROOTS: frozenset[str] = frozenset(TIER0_SAFE_LOAD_RECEIVERS)

_LOAD_NAMES = frozenset({"load", "loads"})

#: ``{attribute name: the roots it may be reached on}`` -- the runtime half of
#: :data:`app.core.script_policy.TIER0_RECEIVER_SCOPED_ATTRS`, so ``re.compile``
#: works and ``torch.compile`` (TorchInductor: generates C++/Triton and invokes
#: a compiler) is refused at both layers rather than only at the gate.
_SCOPED_ATTRS: dict[str, frozenset[str]] = {
    attr: frozenset(roots) for attr, roots in TIER0_RECEIVER_SCOPED_ATTRS.items()
}

#: Capability-bearing TYPES a proxy will not hand over, as the class itself or
#: as an instance of it. Checked with ``issubclass`` / ``isinstance``, so a
#: subclass or an alias under any name is covered -- which is the point:
#: ``pathlib.Path`` was reachable as ``numpy.f2py.crackfortran.Path``,
#: ``torch.fx.graph_module.Path`` and ``torch.package.package_exporter.Path``,
#: and an *instance* of it as ``numpy.testing.NUMPY_ROOT``. A ``Path`` is
#: arbitrary file read AND write (``read_text``, ``write_text``, ``unlink``,
#: ``mkdir``) -- the escape a review proved by writing into the backend's own
#: source tree, which a later import would then execute.
#:
#: The proxy's module rule could not see any of it: a class is not a module.
#: This is the same question asked of values -- *what is this thing* -- rather
#: than of names.
#:
#: Only types whose DEFINING module is not already blocked need listing here;
#: everything defined by :func:`dangerous_modules` (``subprocess.Popen``,
#: ``socket.socket``, ``threading.Thread``, ``tempfile.TemporaryDirectory``,
#: ``ctypes.CDLL``, ...) is caught by the rule below without being named.
_CAPABILITY_TYPES: tuple[type, ...] = (
    pathlib.PurePath,   # __module__ is 'pathlib', but subclasses need not be
    io.IOBase,          # __module__ is '_io' for the concrete file types
    mmap.mmap,          # a file mapped into memory
) + ((os.DirEntry,) if isinstance(getattr(os, "DirEntry", None), type) else ())

#: The C-extension modules that IMPLEMENT the blocked ones. A blocked module
#: is usually a thin Python wrapper, and the thing it re-exports declares the
#: implementation as its ``__module__``: ``os.getcwd.__module__`` is ``nt``
#: (``posix`` on Linux), ``io.FileIO``'s is ``_io``, and CI found
#: ``numpy.random.bit_generator.RLock`` resolving to ``_thread`` on one numpy
#: version and ``threading`` on another. Blocking only the wrapper would make
#: the rule depend on which of the two a library happened to import from.
_BLOCKED_IMPLEMENTATION_ROOTS: frozenset[str] = frozenset({
    "nt", "posix", "_io", "_thread", "_socket", "_ssl", "_ctypes",
    "_pickle", "_winapi", "msvcrt", "_posixsubprocess", "_multiprocessing",
    "_imp", "_frozen_importlib", "_frozen_importlib_external",
})

#: Top-level modules whose classes and functions a script may not hold, even
#: when an allowlisted library hands one over under a harmless name. Derived
#: from the shared blocklist rather than re-listed.
_BLOCKED_DEFINING_ROOTS: frozenset[str] = (
    dangerous_modules() | _BLOCKED_IMPLEMENTATION_ROOTS
)

#: Values that answer for their OWN defining module rather than through their
#: type. Builtin functions are in here because ``os.getcwd`` and ``os.system``
#: are ``builtin_function_or_method`` instances -- asking their type gives
#: ``builtins`` and lets the real thing through, while asking them gives
#: ``nt`` / ``posix``.
_SELF_DESCRIBING: tuple[type, ...] = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
)

#: The builtins Tier 0 removed from the namespace, held as OBJECTS.
#:
#: The namespace allowlist means a script cannot *name* ``open`` or ``eval``.
#: It says nothing about a library handing one over under some other name,
#: and identity is the only check that does not care what that name is:
#: ``numpy.anything is builtins.open`` is the same function whatever it is
#: called. Built from the policy's own denied-call list so the two cannot
#: drift.
_BANNED_BUILTINS: frozenset[Any] = frozenset(
    obj
    for obj in (
        getattr(builtins, name, None)
        for name in (
            "open", "eval", "exec", "compile", "__import__", "input",
            "breakpoint", "globals", "locals", "vars", "dir", "exit", "quit",
            "help",
            # NOT ``memoryview``: the node's builtins allowlist deliberately
            # exposes it, so refusing it here would make the two layers
            # disagree about the same name -- and a memoryview needs a buffer
            # someone already handed you, which is not a capability of its own.
        )
    )
    if callable(obj)
)

_MISS = object()

#: One proxy per module object, process-wide. Proxies are immutable views, so
#: sharing them across executions is safe and keeps the resolution cache warm:
#: after the first script touches ``torch.nn.functional``, every later one
#: gets it back from a dict.
_PROXIES: dict[types.ModuleType, "RestrictedModule"] = {}


def _refuse(message: str) -> ScriptPolicyError:
    return ScriptPolicyError(f"{message}{ESCAPE_HATCH_HINT}")


def module_proxy(module: types.ModuleType) -> "RestrictedModule":
    """Wrap *module*, or refuse it because of what it IS.

    The verdict is taken from the module's own ``__name__`` -- its canonical
    identity -- not from the attribute the script happened to use to get
    here. ``torch.cuda.tunable.mp`` is the stdlib ``multiprocessing``; asking
    the object rather than the alias is what makes that visible.
    """
    name = getattr(module, "__name__", "") or ""
    parts = name.split(".")
    if not name or name not in _ALLOWED_MODULE_PATHS:
        raise _refuse(
            f"Reaching the '{name or '?'}' module is not allowed: it is not on "
            "the Tier-0 list, and how a script got hold of it does not change "
            "that"
        )
    for part in parts[1:]:
        if part in _DENIED_ATTRS:
            raise _refuse(
                f"Reaching the '{name}' module is not allowed: '{part}' is "
                "closed under this policy"
            )
    proxy = _PROXIES.get(module)
    if proxy is None:
        proxy = RestrictedModule(module)
        _PROXIES[module] = proxy
    return proxy


def _resolve(target: types.ModuleType, label: str, name: str) -> Any:
    """Apply the policy to ``target.name`` and return what a script may see."""
    if name.startswith("__") and name.endswith("__"):
        # Dunders are interpreter machinery, and the interpreter itself probes
        # them: ``isinstance(x, T)`` consults ``x.__class__`` when the fast
        # type check misses, and a hard policy error there would turn an
        # ordinary type test into a crash. "There is no such attribute" is
        # both accurate -- the proxy really does not provide it -- and the
        # answer CPython's own machinery knows how to handle, while still
        # being a refusal. The AST gate refuses dunder access outright, with
        # a message, while the script is being typed.
        raise AttributeError(f"{label!r} proxy exposes no attribute {name!r}")
    if name.startswith("_"):
        raise _refuse(
            f"'{label}.{name}' is not available: a script may only use a "
            "library's public API, and private names are how a library's own "
            "imports (the real 'sys', 'os', 'random', ...) leak out of it"
        )
    if name in _FRAME_ATTRS:
        raise _refuse(
            f"'{label}.{name}' is not available: reading another frame's "
            "state reaches the host process's own globals"
        )
    if name in _DENIED_ATTRS:
        raise _refuse(f"'{label}.{name}' is not available under this policy")
    if name in _LOAD_NAMES and label.split(".")[0] not in _SAFE_LOAD_ROOTS:
        raise _refuse(
            f"'{label}.{name}' is not available: under this policy only "
            f"{', '.join(sorted(_SAFE_LOAD_ROOTS))} may be loaded from, "
            f"because every other '.{name}' within reach executes code from "
            "the file it reads"
        )
    scope = _SCOPED_ATTRS.get(name)
    if scope is not None and label.split(".")[0] not in scope:
        raise _refuse(
            f"'{label}.{name}' is not available: under this policy "
            f"'.{name}' is available only on {', '.join(sorted(scope))}"
        )
    value = getattr(target, name)  # AttributeError propagates unchanged
    if isinstance(value, types.ModuleType):
        return module_proxy(value)
    _check_capability(value, label, name)
    return value


def _defining_root(value: Any) -> str:
    """Top-level module that DEFINES *value*, as it declares itself.

    A class or a function answers for itself; anything else answers through
    its type. ``pathlib.Path`` and an instance of it both come back as
    ``pathlib`` whatever attribute they were reached through, which is what
    makes the rule survive aliasing.

    ``__module__`` is not always a string: on some C-defined classes it is a
    slot descriptor, and CI found one (``'member_descriptor' object has no
    attribute 'split'``) on a numpy version this machine did not have. A
    policy check must never be the thing that crashes a legitimate script, so
    anything that is not a ``str`` answers "unknown" and the value is allowed
    through to the rules that do not depend on it.
    """
    if isinstance(value, _SELF_DESCRIBING):
        owner: Any = value
    else:
        owner = type(value)
    module = getattr(owner, "__module__", "")
    if not isinstance(module, str):
        return ""
    return module.split(".")[0]


def _check_capability(value: Any, label: str, name: str) -> None:
    """Refuse a value that IS a capability, however it was named.

    The module rule asks "is this a module I may hand over"; a class is not a
    module, so ``numpy.f2py.crackfortran.Path`` sailed past it and gave a
    script arbitrary file read and write. This asks the same question of the
    value: is it one of the known capability types, or is it defined by a
    module this policy already refuses?

    Type-based on purpose. A name-based version would have had to know about
    ``crackfortran.Path``, ``graph_module.Path``, ``package_exporter.Path``
    AND ``testing.NUMPY_ROOT`` -- four spellings of one class, which is
    exactly the failure mode the proxy replaced for modules.

    The type list is finite and this is a guardrail, not a sandbox: a
    capability whose type is defined by an allowlisted library, or a bare C
    function like ``os.getcwd`` bound under some harmless name, is not covered
    here. The docs say so.
    """
    if isinstance(value, type):
        try:
            if issubclass(value, _CAPABILITY_TYPES):
                raise _refuse(
                    f"'{label}.{name}' is not available: it is a "
                    f"{value.__name__!r} class, which reads and writes files. "
                    "A script reaches the filesystem through a custom node, "
                    "not through a library that happens to re-export the class"
                )
        except TypeError:  # pragma: no cover - exotic metaclass
            pass
    elif isinstance(value, _CAPABILITY_TYPES):
        raise _refuse(
            f"'{label}.{name}' is not available: it is a "
            f"{type(value).__name__} object, which reads and writes files"
        )

    if isinstance(value, _SELF_DESCRIBING) and not isinstance(value, type):
        # Identity, not name: the namespace allowlist stops a script writing
        # ``open``, and this stops a library handing the same object over as
        # ``numpy.something``.
        if value in _BANNED_BUILTINS:
            raise _refuse(
                f"'{label}.{name}' is not available: it is the builtin "
                f"{getattr(value, '__name__', '?')!r}, which this policy "
                "removes from the namespace; reaching it through a library "
                "does not make it a different function"
            )

    root = _defining_root(value)
    if root in _BLOCKED_DEFINING_ROOTS:
        raise _refuse(
            f"'{label}.{name}' is not available: it is defined by the "
            f"'{root}' module, which is not on the Tier-0 list, and being "
            "re-exported by a library that is does not change what it does"
        )


#: Proxy state, held OFF the proxy object: ``id(proxy) -> (proxy, module,
#: label, memo)``. The first slot is the proxy itself, so the entry keeps it
#: alive and its ``id`` can never be recycled onto a different object.
#:
#: This indirection is not decoration. The first cut of this class kept the
#: wrapped module in ``__slots__``, and a slot is a descriptor on the CLASS::
#:
#:     d = type(collections)._target      # ``type`` is an allowed builtin
#:     m = d.__get__(collections)         # the REAL collections module
#:     m._sys.modules['os'].getcwd()      # ...and straight back out
#:
#: which was live -- a proxy that leaked what it wrapped. With no instance
#: state there is no descriptor to fetch, no ``__dict__`` to read, and
#: ``object.__getattribute__(proxy, ...)`` has nothing to hand over either.
_STATE: dict[int, tuple[Any, types.ModuleType, str, dict[str, Any]]] = {}


class RestrictedModule:
    """A policy-checked view onto one module, holding nothing of its own.

    Every attribute access goes through :meth:`__getattribute__` -- not
    ``__getattr__``, which is only consulted after a normal lookup fails and
    would therefore be skipped for anything the object really had.
    """

    __slots__ = ()

    def __init__(self, module: types.ModuleType) -> None:
        # The memo caches already-APPROVED attributes. The verdict for a given
        # (module, name) is deterministic, so this is a pure speed-up: after
        # warm-up a proxy attribute costs two dict lookups, which is what
        # keeps tensor work at full speed.
        _STATE[id(self)] = (self, module, getattr(module, "__name__", "?"), {})

    def __getattribute__(self, name: str) -> Any:
        # NOTE: ``memo`` caches APPROVED values by name, for the life of the
        # process. That is safe -- the verdict for a given (module, name) is
        # deterministic, and a script cannot write to a proxy (``__setattr__``
        # refuses) so it cannot poison an entry. It does mean a probe that
        # monkey-patches a module attribute AFTER something has read it sees
        # the old value; that falsified a synthetic probe run during review,
        # and is worth knowing before writing one.
        _, target, label, memo = _STATE[id(self)]
        cached = memo.get(name, _MISS)
        if cached is not _MISS:
            return cached
        value = _resolve(target, label, name)
        memo[name] = value
        return value

    # ── everything below is refused, and says why ────────────────────────

    def __setattr__(self, name: str, value: Any) -> None:
        raise _refuse(
            f"Assigning to '{_STATE[id(self)][2]}.{name}' is not allowed: "
            "library modules are shared with the host process, so rebinding "
            "one changes it for every node in this run"
        )

    def __delattr__(self, name: str) -> None:
        raise _refuse(
            f"Deleting '{_STATE[id(self)][2]}.{name}' is not allowed: library "
            "modules are shared with the host process"
        )

    def __getitem__(self, key: Any) -> Any:
        raise _refuse(
            f"'{_STATE[id(self)][2]}' cannot be subscripted: a mapping of "
            "modules (sys.modules being the one that matters) is a way around "
            "the attribute rules, so it is closed whatever the module exposes"
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        raise _refuse(f"'{_STATE[id(self)][2]}' cannot be subscripted")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise _refuse(f"'{_STATE[id(self)][2]}' is a module, not a callable")

    def __iter__(self) -> Any:
        raise _refuse(
            f"'{_STATE[id(self)][2]}' is a module and cannot be iterated"
        )

    def __repr__(self) -> str:
        return f"<restricted module {_STATE[id(self)][2]!r}>"


def unwrap(proxy: RestrictedModule) -> types.ModuleType:
    """The module behind *proxy*. For tests and host-side code only.

    Deliberately a module-level function rather than anything reachable from
    a proxy: a script has no way to name this module, and giving the proxy an
    ``unwrap`` attribute would undo the whole design.
    """
    return _STATE[id(proxy)][1]


def tier0_module_namespace() -> dict[str, "RestrictedModule"]:
    """The pre-bound Tier-0 modules, each as a proxy.

    A module that fails to import is left out rather than raising, so the node
    stays usable for pure-stdlib statistics on an installation without torch;
    an explicit ``import torch`` there still raises the real ImportError.
    """
    import importlib

    namespace: dict[str, RestrictedModule] = {}
    for name in TIER0_MODULES:
        try:
            namespace[name] = module_proxy(importlib.import_module(name))
        except ImportError:  # pragma: no cover - torch/numpy are hard deps
            continue
    return namespace
