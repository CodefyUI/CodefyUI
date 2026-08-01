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
**this is a guardrail, not a sandbox.** It blocks the easy escapes (imports
off the allowlist, ``exec``/``eval``, dunder walking, pickle loads) before
the code is compiled, and it hands the code a namespace with those names
removed. A determined attacker who can already type into your canvas can
probably still get out. Nothing here contains CPU or memory either: an
in-canvas ``while True:`` occupies a worker thread until the server is
restarted. Treat "who can reach this editor" as the real boundary.
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
TIER0_DENIED_CALLS: tuple[str, ...] = (
    "open",
    "input",
    "exit",
    "quit",
    "help",
    "license",
    "credits",
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
ESCAPE_HATCH_HINT = (
    ". In-canvas scripts run under the Tier-0 policy, which allows only: "
    + ", ".join(TIER0_MODULES)
    + ". For file, network or process access, write a custom node "
    "(docs: Advanced > Custom Nodes) or a plugin pack (docs: Advanced > "
    "Plugins) -- those are files you author and install deliberately, so "
    "they run with your own trust rather than the canvas's."
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
        denial_hint=ESCAPE_HATCH_HINT,
    )


def defines_entry_point(code: str) -> bool:
    """True when *code* defines a module-level ``def run(...)``.

    Asked only after :func:`validate_script_source` passed, so the parse is
    known to succeed; a failure here still answers False rather than raising,
    because this feeds a hint and must never be the thing that breaks.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == ENTRY_POINT
        for node in tree.body
    )
