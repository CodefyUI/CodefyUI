"""The gate: nothing gets imported that nobody has read.

A plugin is code the user chose to install, from a stranger, that the server
then imports at full trust into its own process. The AST validator in
``app.core.plugin_validator`` decides what a source file may say; this module
decides WHICH FILES it is asked about, and that is the half that has failed
open twice.

Both failures were the same mistake -- reasoning about what a directory or a
filename conventionally holds instead of asking what Python's import system
can reach:

* core#182: the walk covered ``nodes/`` while ``install_plugin_finder``
  hands the WHOLE plugin directory to a namespace package's ``__path__``, so
  ``from ..tests import payload`` pulled in unscanned code at full trust.
* core#220: the walk globbed ``*.py`` while the loader accepts every suffix
  in ``importlib.machinery.all_suffixes()``, so a pack shipping
  ``nodes/helper.pyc`` and no ``helper.py`` was never scanned at all. Not the
  gate failing open on a rule -- the gate never running, which is worse.

So the rule here is a single sentence: every file under the plugin root that
an ``import`` statement could resolve is either SCANNED as source or REFUSED
by name. There is no third outcome, and "walked past it" is the bug both
issues were.

Moved out of ``scripts/plugins.py`` unchanged. It is the same gate on the
same files whether an install came from the CLI or from a route, and two
copies of it would be two answers to "was this code looked at?".
"""

from __future__ import annotations

import importlib.machinery
from collections.abc import Iterable
from pathlib import Path

from app.core.plugin_validator import (
    PluginValidationError,
    dangerous_modules,
    validate_python_source,
)
from app.core.script_policy import TIER0_DENIED_ATTRS
from app.core.security_tiers import CAPABILITIES

__all__ = [
    "LOADER_SUFFIXES",
    "SCANNABLE_SUFFIXES",
    "PluginValidationError",
    "loader_suffix",
    "validate_nodes_dir",
    "validate_plugin_dir",
]


def _denied_attributes_for(allowed_modules: list[str]) -> frozenset[str]:
    """core#179's ``denied_attributes`` set, lifted at Tier 2.

    A method on a value a Tier-0 import hands back (``numpy.zeros(3).dump(
    path)``) is an arbitrary file write with zero capability declared: the
    blocklist gate is keyed on Import nodes, so it never sees what an
    allowed library's return value can do. Already closed for in-canvas
    scripts via ``TIER0_DENIED_ATTRS`` passed as ``denied_attributes``; this
    is the same constant, not a re-derived smaller list, so the two surfaces
    cannot drift on what "closed" means. Deliberately NOT
    ``SCRIPT_PROXY_DENIED_ATTRS``, which also folds in the module-gateway
    attrs (``.hub``'s sibling problem, not this one) and the RCE leaves as
    attributes -- both closed for scripts for reasons specific to an
    allowlisted, unreviewed surface that do not hold for a file the user
    chose to install.

    Lifted entirely at Tier 2 (``allowed_modules`` non-empty, which only
    happens once ``--trust-author`` has already been accepted -- see
    ``_install_github``, which refuses to call either caller of this
    function with a non-empty ``allowed`` otherwise). Closing these at Tier
    0/1 is right: a plugin that declared nothing, or only a capability, gets
    no new file-write / remote-fetch-and-execute surface for free, and
    before core#179 a plugin could not even define a method named ``save``
    without failing installation outright. Refusing them at Tier 2 is
    incoherent, and the dunder/RCE-leaf precedent (never lifted by any
    tier -- see the walker's own denied-attrs handling) does not transfer:
    those rules refuse REFLECTION, which core#133's own docs say no
    capability ever buys. ``.dump`` / ``.hub`` / ``.save`` are not
    reflection -- they are file writes and remote code fetches, and
    ``--trust-author`` has already granted an equivalent or greater version
    of both by a shorter route (bare ``subprocess`` reaches further than
    ``numpy.save`` ever could), so refusing the narrower path while granting
    the wider one protects nothing.
    """
    return frozenset() if allowed_modules else TIER0_DENIED_ATTRS


# ── core#220: the walk covers what the LOADER covers, by extension ─────────
#
# ``validate_plugin_dir`` used to enumerate ``*.py``. The loader imports more
# than that: ``plugin_loader.install_plugin_finder`` sets the synthetic
# package's ``__path__`` to the plugin directory, which puts the stock
# ``FileFinder`` loaders in charge, and those accept every suffix in
# ``importlib.machinery.all_suffixes()``. Verified on a real interpreter, not
# reasoned about -- a plugin whose ``nodes/`` held only ``helper.pyc``,
# ``w.pyw`` and ``native.cp314-win_amd64.pyd``::
#
#     loader suffixes:        ['.py', '.pyw', '.pyc', '.cp314-win_amd64.pyd', '.pyd']
#     files a *.py glob sees: ['__init__.py']            (i.e. none of them)
#     pkgutil reports:        ['helper', 'native', 'w']  (i.e. all of them)
#     validate_plugin_dir():  ACCEPTED, no exception
#
# ...and the ``.pyc`` then imported and ran its ``os.system`` payload. That is
# not the gate failing open on a rule; it is the gate never running, which is
# strictly worse and is why this is keyed on the interpreter's own answer
# rather than on a hardcoded list -- the same "ask, don't assume" move
# ``_compute_os_path_module_leaves`` makes for ``os.path``.
#
# ``.pyw`` and the extension suffixes are unioned in explicitly ON TOP of
# ``all_suffixes()`` because that function answers for the CURRENT
# interpreter, and a tarball is not installed on the machine that built it:
# ``.pyw`` is a source suffix only on Windows, and ``.so`` / ``.pyd`` /
# ``.dylib`` each exist on exactly one platform. Scanning the union means the
# verdict on a given tarball does not depend on which OS ran the installer.
_EXTRA_SOURCE_SUFFIXES = frozenset({".pyw"})
_CROSS_PLATFORM_BINARY_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dylib",
})

#: Suffixes that are Python SOURCE and can therefore be handed to the AST
#: gate. ``.pyw`` is ordinary Python text -- only the launcher treats it
#: differently -- so it is scanned, not refused.
SCANNABLE_SUFFIXES: frozenset[str] = (
    frozenset(importlib.machinery.SOURCE_SUFFIXES) | _EXTRA_SOURCE_SUFFIXES
)

#: Every suffix an ``import`` statement can resolve to a file, anywhere this
#: tarball might be installed. Asserted against ``all_suffixes()`` by a
#: standing test, so a future CPython that grows a loader suffix fails loudly
#: instead of silently widening the hole this closed.
LOADER_SUFFIXES: frozenset[str] = (
    frozenset(importlib.machinery.all_suffixes())
    | SCANNABLE_SUFFIXES
    | _CROSS_PLATFORM_BINARY_SUFFIXES
)


def loader_suffix(path: Path) -> str | None:
    """The loader suffix *path* would be imported under, or ``None``.

    Longest match wins, because the extension suffixes nest:
    ``native.cp314-win_amd64.pyd`` ends with both ``.cp314-win_amd64.pyd``
    and ``.pyd``, and the longer one is the right answer twice over -- it is
    what the refusal message should name, and stripping only ``.pyd`` would
    leave a stem of ``native.cp314-win_amd64``, which the identifier test in
    :func:`_names_an_importable_module` would then wave through.

    A name that is ONLY a suffix (a file literally called ``.py``) answers
    ``None``: the stem would be empty, so there is no module name for an
    ``import`` statement to spell.
    """
    name = path.name
    best: str | None = None
    for suffix in LOADER_SUFFIXES:
        if len(name) > len(suffix) and name.endswith(suffix):
            if best is None or len(suffix) > len(best):
                best = suffix
    return best


def _names_an_importable_module(path: Path, suffix: str) -> bool:
    """Whether an ``import`` statement could name the module *path* holds.

    The same structural question ``_VALIDATION_SKIP_DIRS`` asks about
    ``.git``, applied to a FILE: strip the loader suffix and ask whether what
    is left is a valid Python identifier. If it is not, no import statement
    -- absolute or relative -- can name it, so nothing can reach it.

    This exists for exactly one shape, and getting it wrong in either
    direction is a real failure, so it was verified rather than reasoned
    about. CPython writes its own bytecode cache as
    ``__pycache__/real.cpython-314.pyc``; an attacker writes
    ``__pycache__/payload.pyc``. Both are ``.pyc`` files in the same
    directory, and they are not the same thing::

        __pycache__/payload.pyc            stem 'payload'          identifier
        __pycache__/real.cpython-314.pyc   stem 'real.cpython-314' NOT

        pkgutil.iter_modules(__pycache__)          -> ['payload']
        import <pkg>.nodes.__pycache__.payload     -> OK, ran the bytecode
        import <pkg>.nodes.__pycache__.real        -> ModuleNotFoundError

    So the refusal has to fire on the first and must NOT fire on the second:
    a plugin directory that any interpreter has ever imported from has a
    ``__pycache__`` full of the second kind, including this repo's own
    built-in packs, and refusing to install over an ordinary compilation
    artifact would be a false positive with no matching security value.

    Applied only to the suffixes that get REFUSED. Source files are scanned
    whatever they are called: scanning is never wrong, and the previous
    ``*.py`` glob scanned an unimportably-named ``my-node.py`` too.
    """
    return path.name[: -len(suffix)].isidentifier()


# ── #413: a refusal the plugin's author can act on ─────────────────────────
#
# The validator answers for one source text and knows nothing of the plugin
# around it, so three things are said here instead. Where the file is: its
# path in the plugin and the line, where the validator knew only a base name
# ("conftest.py"). Why a file outside nodes/ is read at all. And whether any
# [security] grant could let the file through: the validator ends an import
# refusal by telling the author to declare the grant that lifts THAT line,
# which is a dead end when another line is refused at every tier. The
# conftest `cdui plugin new` used to write was refused for `import sys`, told
# to ask for it with --trust-author, and then refused for `setattr`, which
# nothing lifts.

#: What the validator calls a file it refuses. The gate names the file at the
#: start of the refusal, so the validator's sentence reads "is not allowed in
#: this file" rather than naming it a second time.
_REFUSED_FILE = "this file"

#: How ``plugin_validator._capability_denial`` starts the sentence telling
#: the author to declare the grant that lifts an import refusal: Tier 2, then
#: Tier 1. The gate cuts that sentence wherever the grant is not advice it
#: stands behind, and ``test_plugin_gate_refusal.py`` pins both openings to
#: the validator's text.
_GRANT_ADVICE_OPENINGS = (
    "A plugin that genuinely needs it has to be installed with --trust-author",
    "Declare it in the plugin manifest as [security] capabilities",
)

_WHY_THE_SCAN_READS_IT = (
    "The scan reads every Python file in the plugin, tests/ included, because "
    "a node can import any file in the plugin, not only the ones in nodes/."
)
_CHANGE_IT_INSTEAD = (
    "If only tests or tooling use this file, change it rather than declaring "
    "a [security] grant, which would cover the whole plugin, its nodes included."
)
#: The docs section with a test setup that needs no grant. The base is the
#: one the editor links to (``DOCS_BASE`` in ``frontend/src/utils/docsUrl.ts``).
#: Last on its line, with no full stop after it, so a copied URL is whole.
_TESTS_THAT_PASS = (
    "Tests that need no grant: "
    "https://docs.codefyui.com/advanced/plugins#tests-are-scanned-too"
)


def _sentence(text: str) -> str:
    """*text* ending as a sentence. The Plugin Center shows a refusal's lines
    as one paragraph, so each one has to end like one."""
    return text if text.endswith((".", "?", "!")) else text + "."


def _without_grant_advice(message: str) -> str:
    """*message* without the sentence telling the author to declare a
    ``[security]`` grant.

    Only that sentence goes. The capability the refusal names stays, and so
    does what follows the sentence, such as the note that
    ``from os.path import join`` needs no capability at all.
    """
    kept = [
        sentence
        for sentence in message.split(". ")
        if not sentence.startswith(_GRANT_ADVICE_OPENINGS)
    ]
    return _sentence(". ".join(kept))


def _refused_under_every_grant(content: bytes) -> PluginValidationError | None:
    """What still refuses *content* under the most a manifest can ask for.

    That is every capability, plus every blocklisted module under
    ``allowed_modules``, which also lifts the Tier-0 attribute names (see
    :func:`_denied_attributes_for`). A refusal means no grant lets the file
    through, and names the line. ``None`` means some set of grants does, or
    that this pass could not tell.

    Only advice rides on this answer, never the verdict. The pass reads past
    the line the scan refused, so it can fail where the scan did not -- a
    later expression nested too deep for the validator's ``ast.unparse``
    raises ``RecursionError`` -- and then the refusal keeps the validator's
    own words instead of turning into a traceback.
    """
    everything = sorted(dangerous_modules())
    try:
        validate_python_source(
            content,
            _REFUSED_FILE,
            allowed_modules=everything,
            capabilities=list(CAPABILITIES),
            denied_attributes=_denied_attributes_for(everything),
        )
    except PluginValidationError as exc:
        return exc.with_traceback(None)
    except Exception:
        return None
    return None


def _refusal(
    rel_parts: tuple[str, ...], content: bytes, exc: PluginValidationError
) -> PluginValidationError:
    """The validator's refusal of one file, rewritten for the plugin's author.

    The validator's instruction to declare a grant is kept only for a file in
    ``nodes/`` that some set of grants lets through. When no grant does, it
    is a dead end. Outside ``nodes/`` it would widen what the whole plugin
    may do for the sake of a file CodefyUI never runs on its own, usually a
    test; #413 asked for no ``--trust-author`` advice there, and a capability
    is the same trade. Such a refusal ends with the docs section that shows a
    test setup needing no grant.
    """
    where = "/".join(rel_parts)
    if exc.lineno:
        where += f", line {exc.lineno}"
    in_nodes = rel_parts[0] == "nodes"
    every_tier = _refused_under_every_grant(content)
    message = str(exc)
    if every_tier is not None or not in_nodes:
        message = _without_grant_advice(message)
    lines = [f"{where}: {message}"]
    if not in_nodes:
        lines.append(_WHY_THE_SCAN_READS_IT)
    if every_tier is None:
        if not in_nodes:
            lines.append(_CHANGE_IT_INSTEAD)
    elif every_tier.lineno == exc.lineno and str(every_tier) == str(exc):
        lines.append(
            "No [security] setting in cdui.plugin.toml lifts this refusal, so "
            "the code has to change."
        )
    else:
        at = f"line {every_tier.lineno}" if every_tier.lineno else "another line"
        lines.append(
            f"No [security] setting in cdui.plugin.toml lets this file through, "
            f"because {at} is refused whatever the manifest declares: "
            + _sentence(str(every_tier))
        )
    if not in_nodes:
        lines.append(_TESTS_THAT_PASS)
    return PluginValidationError("\n".join(lines), lineno=exc.lineno)


def _validate_importable_tree(
    root: Path,
    allowed_modules: list[str],
    capabilities: Iterable[str],
    *,
    skip_dirs: frozenset[str] = frozenset(),
    prefix: tuple[str, ...] = (),
) -> None:
    """AST-scan every importable source file under *root*; refuse the rest.

    "The rest" is the whole point. A ``.pyc`` cannot be AST-scanned without
    decompiling it and a ``.pyd`` / ``.so`` cannot be scanned even in
    principle, so the only two honest answers are "refuse" and "import
    unexamined code at full trust". This picks the first, by name, with a
    message that says which file and why -- never a silent skip, which is
    exactly what the ``*.py`` glob was doing.

    *prefix* is where *root* sits in the plugin (``("nodes",)`` for
    :func:`validate_nodes_dir`), so every refusal names the file by its path
    in the plugin; a source file's refusal also names the line (#413).
    """
    if not root.exists():
        return
    root_resolved = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.resolve().relative_to(root_resolved).parts
        if skip_dirs and any(part in skip_dirs for part in rel_parts):
            continue
        suffix = loader_suffix(path)
        if suffix is None:
            continue
        rel_parts = prefix + rel_parts
        rel = "/".join(rel_parts)
        if suffix not in SCANNABLE_SUFFIXES:
            if not _names_an_importable_module(path, suffix):
                continue
            raise PluginValidationError(
                f"'{rel}' cannot be installed: Python's import system loads "
                f"'{suffix}' files, but they are compiled rather than source, "
                f"so the security scan cannot look inside one. Every file a "
                f"plugin can have imported has to be readable as source -- "
                f"ship the '.py' this was built from instead"
            )
        content = path.read_bytes()
        if not content.strip():
            continue
        refused = None
        try:
            validate_python_source(
                content,
                _REFUSED_FILE,
                allowed_modules=allowed_modules,
                capabilities=list(capabilities),
                denied_attributes=_denied_attributes_for(allowed_modules),
            )
        except PluginValidationError as exc:
            # Only its message and line are needed. Its traceback holds the
            # frames that hold the parsed tree, and _refusal parses the file
            # again, so the tree goes first -- and the rewrite runs outside
            # this block, where no exception is being handled.
            refused = exc.with_traceback(None)
        if refused is not None:
            raise _refusal(rel_parts, content, refused) from refused


def validate_nodes_dir(
    nodes_dir: Path,
    allowed_modules: list[str],
    capabilities: Iterable[str] = (),
) -> None:
    _validate_importable_tree(
        nodes_dir, allowed_modules, capabilities, prefix=("nodes",)
    )


# Directories within an extracted plugin tarball that are *not* reachable
# through Python's import system, whatever the loader's ``__path__`` covers --
# safe to skip the AST gate because there is no route from an ``import``
# statement to a file in here. Everything else -- ``examples/``, ``tests/``,
# ``docs/``, ``assets/``, ``__pycache__``, any other top-level helper -- gets
# scanned, because ``plugin_loader.install_plugin_finder`` registers the
# WHOLE plugin directory as a PEP-420 namespace package's ``__path__`` (not
# only ``nodes/``), so ``from .. import _helpers`` -- or ``from ..tests
# import payload`` -- from inside a scanned ``nodes/foo.py`` would otherwise
# pull in unscanned code at full trust, automatically, at server boot or
# reload (core#182).
#
# ``__pycache__`` is deliberately NOT on this set, despite an earlier version
# of this comment claiming it was safe to skip because "a real __pycache__
# never holds a *.py this glob would match." That is true of the directory
# CPython writes and irrelevant here: the attacker supplies the tarball, so
# `__pycache__/payload.py` exists because they put it there, and
# `"__pycache__".isidentifier()` is `True` -- PEP-420 namespace resolution
# imports it exactly like any other directory name. Verified directly, not
# merely argued: with a plugin installed through the real loader, both
# `importlib.import_module("cdui_plugins.<id>.__pycache__.payload")` and a
# relative `from ..__pycache__ import payload` inside `nodes/` resolve to the
# planted file, and `validate_plugin_dir` (before this fix) accepted it
# silently. Reasoning about what a directory name conventionally holds is not
# the same claim as reasoning about what Python's import system can reach,
# and only the second one is what this set is for.
#
# ``.git`` is the one name that passes that test for real: `.git` is not a
# valid identifier (`".git".isidentifier()` is `False`, and `.` cannot appear
# inside one), so no `import` statement -- absolute or relative -- can ever
# name a package component spelled that way. That is a structural guarantee
# independent of what is inside the directory, which is the property this
# set exists to require before trusting a name to be un-scanned.
#
# Narrowing the LOADER's ``__path__`` instead, so a skipped directory were
# unreachable rather than merely unscanned, was considered and rejected:
# PEP-420 namespace packages have no native "every subdirectory except these"
# carve-out, so that route needs a custom import finder/loader -- new import
# machinery, not an extension of either existing one -- for a difference this
# scan already erases by scanning first.
_VALIDATION_SKIP_DIRS = frozenset({".git"})


def validate_plugin_dir(
    plugin_root: Path,
    allowed_modules: list[str],
    capabilities: Iterable[str] = (),
) -> None:
    """Walk the entire plugin directory and validate every importable file.

    The original ``validate_nodes_dir`` only checked ``nodes/`` which left a
    bypass via top-level helpers. This visits every file in the tree that
    Python's import system could load, except the ones under
    :data:`_VALIDATION_SKIP_DIRS` -- ``.git`` alone, the one name provably
    unreachable through Python's import system (not a valid identifier, so
    no ``import`` can ever name it). Nothing else is skipped: ``examples/``,
    ``tests/``, ``docs/``, ``assets/`` and ``__pycache__`` all get scanned
    too (core#182), because the plugin loader registers the WHOLE directory
    as a namespace package's ``__path__``, so a scanned ``nodes/foo.py`` can
    ``from ..tests import payload`` -- or ``from ..__pycache__ import
    payload`` -- into any of them.

    "Every file the import system could load" is by EXTENSION as well as by
    directory since core#220: this used to glob ``*.py`` while the loader
    also accepts ``.pyw``, ``.pyc`` and ``.pyd`` / ``.so``, so a plugin
    shipping ``nodes/helper.pyc`` and no ``helper.py`` was never scanned at
    all -- the gate did not fail open, it never ran. Source suffixes are
    scanned; compiled ones are refused by name (see
    :func:`_validate_importable_tree`), because a plugin has no legitimate
    reason to ship a bytecode-only or binary module through this path and
    neither can be AST-scanned.

    *capabilities* are the ones the user confirmed at install time. They are
    passed to every file rather than per-file, because a capability is a
    property of the INSTALL, not of a source file: a plugin granted
    ``network`` may reach it from wherever it likes inside its own tree.

    *allowed_modules* also lifts the ``denied_attributes`` closed by
    core#179 -- see :func:`_denied_attributes_for` -- because a non-empty
    list here only happens once ``--trust-author`` has already been
    accepted, and refusing ``arr.dump()`` to a plugin trusted with
    ``subprocess`` protects nothing.

    A refusal names the file by its path in the plugin and the line, and
    tells the author to declare a ``[security]`` grant only where some set of
    grants would let the file through (see :func:`_refusal`).
    """
    _validate_importable_tree(
        plugin_root,
        allowed_modules,
        capabilities,
        skip_dirs=_VALIDATION_SKIP_DIRS,
    )
