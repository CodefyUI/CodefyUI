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

import types
from typing import Any

from .plugin_validator import frame_introspection_attrs
from .script_policy import (
    ESCAPE_HATCH_HINT,
    SCRIPT_PROXY_DENIED_ATTRS,
    TIER0_MODULES,
    TIER0_SAFE_LOAD_RECEIVERS,
    ScriptPolicyError,
)

#: Top-level packages a proxy may hand back. Anything else -- reached under
#: ANY attribute name, at ANY depth -- is refused.
_ALLOWED_ROOTS: frozenset[str] = frozenset(TIER0_MODULES)

#: Attribute names refused whatever they resolve to. The union of the Tier-0
#: denied doors, the blocked-module names, and the RCE leaves, so the runtime
#: and the gate cannot drift apart on what "closed" means.
_DENIED_ATTRS: frozenset[str] = frozenset(SCRIPT_PROXY_DENIED_ATTRS)

_FRAME_ATTRS: frozenset[str] = frame_introspection_attrs()

#: Roots whose ``.load`` / ``.loads`` is a parser rather than an unpickler.
_SAFE_LOAD_ROOTS: frozenset[str] = frozenset(TIER0_SAFE_LOAD_RECEIVERS)

_LOAD_NAMES = frozenset({"load", "loads"})

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
    if not name or parts[0] not in _ALLOWED_ROOTS:
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
    value = getattr(target, name)  # AttributeError propagates unchanged
    if isinstance(value, types.ModuleType):
        return module_proxy(value)
    return value


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
