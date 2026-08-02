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

Honest framing -- the same one :mod:`app.core.plugin_validator` gives, and
worth repeating because this is the surface where a *stranger's* code runs:
**this is a guardrail, not a sandbox.**

What the tier-0 gate actually does is limit WHICH LIBRARIES a script can
reach. It does not, and cannot, limit what those libraries can DO. numpy and
torch are large enough to contain file IO on their own (``numpy.savetxt``
writes, ``numpy.fromfile`` reads, ``torch.hub.load`` downloads and executes a
remote ``hubconf.py``, ``torch.utils.cpp_extension.load_inline`` compiles and
runs C++). :data:`TIER0_DENIED_ATTRS` closes the doors we know about, but
that list is a best-effort blocklist over two enormous APIs and should be
read as raising the cost of an escape, never as a guarantee.

Nothing here contains CPU or memory either: nodes run on the interpreter's
DEFAULT thread pool, so an in-canvas ``while True:`` starves every node
execution in the process, not merely its own thread, until the server is
restarted. Module objects are shared with the host too -- a script can
rebind ``torch.zeros`` for everything else running in this process.

Treat "who can reach this editor" as the real boundary.
"""

from __future__ import annotations

import ast

from .plugin_validator import PluginValidationError, validate_python_source

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
#: escape; nothing here implies the list is complete.
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
        denied_attributes=TIER0_DENIED_ATTRS,
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
