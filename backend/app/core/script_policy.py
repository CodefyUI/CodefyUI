"""Policy for user-authored Python that CodefyUI runs in its own process.

There are three ways user Python reaches this process, and they differ only
in how much the author is trusted:

* **In-canvas scripts** (the ``PythonScript`` node, core#131) -- typed into
  the browser, never reviewed, executed on every run. Tier 0: a small
  *allowlist* of importable modules and nothing else.
* **Custom nodes** -- files the user dropped into their own project.
* **Plugin packs** -- installed deliberately, optionally with
  ``--trust-author`` widening the gate.

The module lists live HERE, in one place, rather than in the node that
happens to need them first: core#133 generalises the plugin-pack gate into
the same tiered policy and imports :data:`TIER0_MODULES` from this module.
Changing the tier-0 list therefore changes both consumers at once, which is
the point.

Two layers, and only one of them is the boundary
------------------------------------------------
1. **The AST gate** (this module's :func:`validate_script_source`) runs before
   the code is compiled and again on every keystroke in the editor. It is the
   fast first line and the source of immediate feedback. Its rules are keyed
   on NAMES, and a name is not a boundary: three consecutive adversarial
   reviews walked through it through a name nobody had listed
   (``__loader__``, ``torch.os``, ``collections._sys``).
2. **The runtime module proxy** (:mod:`app.core.script_proxy`) is the actual
   boundary. The namespace no longer contains the host's real module objects;
   it contains proxies that judge the RESOLVED OBJECT -- *is what I just got a
   module, and is it on the list?* -- so underscore aliases, ``sys.modules``
   subscripts, values bound to a local, and next year's library reshuffle are
   all covered structurally instead of by enumeration.

The gate's rules are kept where the proxy makes them redundant, because two
independent locks on one door is the point, and because a rejection while
typing is worth more to a user than a failure mid-run.

Honest framing -- the same one :mod:`app.core.plugin_validator` gives, and
worth repeating because this is the surface where a *stranger's* code runs:
**this is a guardrail, not a sandbox.**

What the proxy bounds is the LIBRARY SURFACE a script can navigate. It does
not, and cannot, bound what the reachable functions DO: numpy and torch are
large enough to contain file IO on their own (``numpy.savetxt`` writes,
``numpy.fromfile`` reads, ``torch.hub.load`` downloads and executes a remote
``hubconf.py``, ``torch.utils.cpp_extension.load_inline`` compiles and runs
C++). :data:`TIER0_DENIED_ATTRS` closes the doors we know about, and that
list is still a best-effort blocklist over two enormous APIs.

Nothing here contains CPU or memory either: nodes run on the interpreter's
DEFAULT thread pool, so an in-canvas ``while True:`` starves every node
execution in the process, not merely its own thread, until the server is
restarted.

Treat "who can reach this editor" as the real boundary.
"""

from __future__ import annotations

import ast

from .plugin_validator import (
    PluginValidationError,
    dangerous_attr_leaves,
    dangerous_modules,
    validate_python_source,
)


class ScriptPolicyError(RuntimeError):
    """Raised at RUN TIME when a script reaches for something Tier 0 refuses.

    Lives here rather than in :mod:`app.core.script_proxy` so the proxy can
    import the policy constants without the policy importing the proxy back.
    A ``RuntimeError`` subclass on purpose: the engine reports
    ``str(exception)`` for anything deriving from ``Exception``, so a script
    that trips this gets the policy message and a line number in the
    Execution Log, exactly like any other failure.
    """

#: Modules an in-canvas script may import, and which are pre-bound in its
#: namespace under these exact names. Deliberately short: numeric work, the
#: two tensor libraries, and the stdlib pieces statistics code reaches for.
#: Anything doing file, network, process or import machinery is absent by
#: construction rather than by blocklist.
TIER0_MODULES: tuple[str, ...] = (
    "collections",
    "itertools",
    "json",
    "math",
    "numpy",
    "re",
    "statistics",
    "torch",
)

#: Builtin names denied on top of :mod:`app.core.plugin_validator`'s own set
#: (``exec``/``eval``/``compile``/``globals``/...). These are the ones a
#: *script* has no business calling: file access, a prompt that would park a
#: worker thread forever, and the interpreter-exit helpers.
#:
#: This is the AST half. The runtime half -- see
#: ``python_script_node._ALLOWED_BUILTINS`` -- is an explicit ALLOWLIST rather
#: than the mirror image of this tuple, because a blocklist over
#: ``vars(builtins)`` left ``__loader__`` sitting in the namespace, and
#: ``__loader__.load_module('nt')`` is the real ``os`` module.
TIER0_DENIED_CALLS: tuple[str, ...] = (
    "open",
    "input",
    "exit",
    "quit",
    "help",
    "license",
    "credits",
)

#: Attribute names a tier-0 script may not reach at all, whatever the
#: receiver. These are the doors INSIDE the allowlisted libraries that lead
#: back out to the filesystem, the network, a compiler or another process.
#:
#: Best-effort and openly so: it is a blocklist over numpy and torch, two
#: APIs far too large to enumerate. Everything here is a real, checked
#: escape; nothing here implies the list is complete -- which is why it is no
#: longer the boundary, only one of the locks. :mod:`app.core.script_proxy`
#: enforces the same names at run time, on top of its resolved-object rule.
TIER0_DENIED_ATTRS: tuple[str, ...] = (
    # torch: fetch-and-execute, compile-and-execute, IPC, distributed IO
    "hub",                        # torch.hub.load runs a remote hubconf.py
    "cpp_extension",              # compiles and loads C++ / CUDA
    "load_inline",
    "load_library",               # torch.ops.load_library -> dlopen
    "load_state_dict_from_url",
    "download_url_to_file",
    "from_file",                  # torch.from_file maps a file into a tensor
    "distributed",
    "multiprocessing",
    "onnx",                       # onnx.export writes files
    "tensorboard",                # SummaryWriter writes files
    # numpy: file IO in both directions, plus the ctypes bridge
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "tofile",
    "memmap",
    "ctypeslib",
    "DataSource",
    # numpy.testing's temp-file helpers: they create real directories and
    # files on disk and hand back a plain ``str``, so neither the capability
    # type rule nor the blocked-defining-module rule in
    # :mod:`app.core.script_proxy` can see them -- they are numpy's own
    # functions returning numpy's own strings. Listed here openly as what
    # they are: a NAME-based patch on the residual those rules leave, closing
    # the one instance found in the tier-0 surface rather than the class.
    "tempdir",
    "temppath",
)

#: Attribute names that survive the gateway rule below despite naming a
#: blocked module. ``torch.signal`` is torch's OWN DSP namespace
#: (``torch.signal.windows.hann``), not the stdlib's process-signal module,
#: and refusing it would cost a real statistics API for nothing: every route
#: to the stdlib ``signal`` runs through a component this policy already
#: closes. ``code`` is exempt as a word, not as a module -- it is far too
#: ordinary an attribute name (this node's own parameter is called ``code``)
#: and the ``code`` module is not reachable from any tier-0 module at all.
TIER0_MODULE_ATTR_EXEMPTIONS: frozenset[str] = frozenset({"signal", "code"})

#: Blocked module names, refused as ATTRIBUTES as well as imports.
#:
#: The import allowlist is worth nothing on its own, because an allowlisted
#: module will hand you a blocked one if you ask it by name. Every one of
#: these was live::
#:
#:     torch.os.getcwd()                    # the real os
#:     torch.sys.executable
#:     torch.serialization.pickle           # the real pickle
#:     json.codecs.sys
#:     numpy.f2py.subprocess.run([...])
#:
#: A scan of the tier-0 namespace two levels deep finds 14 blocked modules
#: reachable this way -- ``os`` down 41 distinct paths, ``sys`` down 57 --
#: so enumerating the paths is hopeless and the NAME has to be the rule.
#:
#: Derived from :func:`app.core.plugin_validator.dangerous_modules` rather
#: than re-listed, so a module added to that blocklist closes its attribute
#: route in the same commit. Receiver-independent, like the rest of
#: :data:`TIER0_DENIED_ATTRS`: a receiver the walker cannot resolve is
#: exactly what ``(lambda: torch)().os`` is for.
#: Conventional aliases torch binds blocked modules under, which a rule
#: keyed on the module's own NAME cannot see. Both of these are
#: ``multiprocessing``: ``torch.cuda.tunable.mp`` was, after the name rule
#: landed, the last remaining root from which a depth-4 scan could still
#: reach a blocked module -- and everything the scan found beyond it
#: (``.connection``, ``.queues``, ``.reduction``, ``.spawn``, ...) lives
#: INSIDE multiprocessing, so cutting the root cuts the tree without
#: blocking a pile of ordinary English words.
#:
#: Two entries is not a category. A library is free to bind ``os`` under any
#: name it likes and this rule will not see it; that is the structural limit
#: of a name-keyed blocklist, and the reason the docs promise a raised cost
#: rather than an unreachable filesystem.
TIER0_GATEWAY_MODULE_ALIASES: tuple[str, ...] = ("mp", "python_multiprocessing")

TIER0_GATEWAY_MODULE_ATTRS: tuple[str, ...] = tuple(
    sorted(
        (dangerous_modules() - TIER0_MODULE_ATTR_EXEMPTIONS)
        | set(TIER0_GATEWAY_MODULE_ALIASES)
    )
)

#: The only receivers a tier-0 script may call ``.load()`` / ``.loads()`` on.
#:
#: An ALLOWLIST rather than the usual "is this receiver a known unpickler?"
#: question, because that question can only be asked of a receiver the walker
#: can resolve, and there are four one-line ways to make it unresolvable::
#:
#:     b = torch; b.load(x)          getattr(torch, 'load')(x)
#:     (lambda: torch)().load(x)     (torch,)[0].load(x)
#:
#: all of which reached the real ``torch.load`` -- i.e. pickle, i.e. arbitrary
#: code from a file the script names. Inverting the rule makes an unresolvable
#: receiver fail closed. The cost is that a script's own ``obj.load()`` helper
#: is refused as well; that is the same cost :data:`TIER0_DENIED_ATTRS`
#: already imposes on a script's own ``obj.save()``, and the message says
#: which name it objected to.
#:
#: ``json`` is here because parsing JSON is the one legitimate ``.loads`` a
#: tier-0 script has: of the eight allowed modules only ``json``, ``numpy``
#: and ``torch`` define one at all, and the other two are the pickle doors.
TIER0_SAFE_LOAD_RECEIVERS: tuple[str, ...] = ("json",)

#: The RCE leaves (``system``, ``popen``, ``spawnl``, ``execfile``, ...),
#: refused by tier 0 as ATTRIBUTES rather than only as calls.
#:
#: The shared walker keys those on the CALL, which one assignment evades::
#:
#:     f = osmod.system     # no Call node with a dangerous leaf
#:     f('cmd /c ...')      # the call's leaf is 'f'
#:
#: That was live. Making it an attribute rule costs a tier-0 script the
#: ability to have its own ``.system`` attribute, which is a price only tier 0
#: pays -- an installed plugin with a ``self.system`` field is ordinary code
#: and the shared walker still allows it.
TIER0_RCE_ATTR_LEAVES: tuple[str, ...] = tuple(sorted(dangerous_attr_leaves()))

#: Every attribute name tier 0 refuses, in one place, so the AST gate and the
#: runtime proxy cannot drift on what "closed" means.
SCRIPT_PROXY_DENIED_ATTRS: tuple[str, ...] = tuple(
    sorted(
        set(TIER0_DENIED_ATTRS)
        | set(TIER0_GATEWAY_MODULE_ATTRS)
        | set(TIER0_RCE_ATTR_LEAVES)
    )
)

#: Filename the script is compiled under. Shows up in tracebacks and in the
#: error a failing node reports, so it must read as "your code", not as a
#: temp path the user has never seen.
SCRIPT_FILENAME = "<PythonScript>"

#: Ceiling on script length. The validation endpoint runs on every keystroke,
#: and ``ast.parse`` is linear but not free; a script this long belongs in a
#: custom node anyway. Generous enough that no honest script meets it.
MAX_SCRIPT_CHARS = 100_000

#: Name the script must define. The node calls it; the editor warns when it
#: is missing rather than waiting for the run to fail.
ENTRY_POINT = "run"

#: Appended to every policy rejection. A refusal that does not say where the
#: capability *does* live just reads as an arbitrary wall. Opens with its own
#: full stop because the messages it is appended to end mid-sentence.
#:
#: Deliberately says "which libraries", not "what they can do": promising
#: that files and the network are unreachable would be a promise this gate
#: cannot keep, printed at the exact moment a user is deciding how much to
#: trust it.
ESCAPE_HATCH_HINT = (
    ". In-canvas scripts run under the Tier-0 policy, which limits them to "
    "these libraries: "
    + ", ".join(TIER0_MODULES)
    + ". If you need a library outside that list -- or you would rather not "
    "rely on a guardrail at all -- write a custom node (docs: Advanced > "
    "Custom Nodes) or a plugin pack (docs: Advanced > Plugins): those are "
    "files you author and install deliberately, so they run with your own "
    "trust rather than the canvas's."
)


def validate_script_source(code: str, filename: str = SCRIPT_FILENAME) -> None:
    """Raise ``PluginValidationError`` unless *code* satisfies Tier 0.

    A thin, named front door onto the shared AST gate so callers (the node,
    the ``/api/nodes/script/validate`` endpoint, core#133) cannot drift on
    which knobs tier 0 sets.

    Passing here does NOT mean a script can do what it is asking for: this is
    the fast, name-keyed first line, and the boundary is the runtime proxy in
    :mod:`app.core.script_proxy`. Code the gate accepts can still be refused
    mid-run, with the same message shape and a line number.
    """
    if len(code) > MAX_SCRIPT_CHARS:
        raise PluginValidationError(
            f"Script is {len(code)} characters; the limit is "
            f"{MAX_SCRIPT_CHARS}. Move code this long into a custom node."
        )
    validate_python_source(
        code,
        filename,
        import_allowlist=TIER0_MODULES,
        extra_denied_names=TIER0_DENIED_CALLS,
        denied_attributes=SCRIPT_PROXY_DENIED_ATTRS,
        safe_load_receivers=TIER0_SAFE_LOAD_RECEIVERS,
        library_roots=TIER0_MODULES,
        denial_hint=ESCAPE_HATCH_HINT,
    )
    _reject_async_entry_point(code, filename)


def entry_point_kind(code: str) -> str | None:
    """``"sync"``, ``"async"`` or ``None`` for the module-level ``run``.

    A failure to parse answers ``None`` rather than raising: this feeds an
    editor hint and must never be the thing that breaks.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if getattr(node, "name", None) != ENTRY_POINT:
            continue
        if isinstance(node, ast.AsyncFunctionDef):
            return "async"
        if isinstance(node, ast.FunctionDef):
            return "sync"
    return None


def defines_entry_point(code: str) -> bool:
    """True when *code* defines a usable module-level ``def run(...)``."""
    return entry_point_kind(code) == "sync"


def _reject_async_entry_point(code: str, filename: str) -> None:
    """Refuse ``async def run`` at the gate.

    The node calls ``run`` on a worker thread with no event loop, so an async
    one returns a coroutine object that nothing awaits: the downstream port
    receives ``<coroutine object run>``, Python warns on stderr, and the
    editor -- which only ever asked whether the policy was satisfied -- says
    it is fine. Better to refuse it while it is being typed.
    """
    if entry_point_kind(code) != "async":
        return
    line = next(
        (
            node.lineno
            for node in ast.parse(code).body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == ENTRY_POINT
        ),
        None,
    )
    raise PluginValidationError(
        f"'async def {ENTRY_POINT}' is not supported in {filename}: the node "
        "calls it on a worker thread with no event loop, so nothing would "
        f"await it. Define a plain 'def {ENTRY_POINT}(inputs, params)'.",
        lineno=line,
    )
