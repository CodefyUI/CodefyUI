"""core#133 -- the three-tier import policy, module by module and tier by tier.

The gate an installed plugin passes is a BLOCKLIST: the user chose the file,
so anything not known-dangerous is fine. What core#133 changes is the shape of
the refusal. Before, a blocked root had exactly one escape hatch --
``--trust-author`` plus ``[security].allowed_modules``, which grants
everything -- so "I POST a metric" and "I load a C library" cost the same.
Now a blocked root belongs to at most one capability, a manifest may ask for
that capability by name, and the user is shown the request before anything is
written to disk.

The matrix below is the specification. Read a row as: this module, at this
tier, is accepted or refused, and the refusal says which tier would unlock it.
"""

from __future__ import annotations

import pytest

from app.core import security_tiers as tiers
from app.core.plugin_validator import (
    PluginValidationError,
    dangerous_modules,
    validate_python_source,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _tier0(code: str) -> None:
    """Validate as an installed plugin that declared nothing at all."""
    validate_python_source(code, "plugin.py")


def _tier1(code: str, *capabilities: str) -> None:
    """Validate as a plugin whose manifest declared *capabilities*."""
    validate_python_source(code, "plugin.py", capabilities=list(capabilities))


def _tier2(code: str, *allowed: str) -> None:
    """Validate as a plugin installed with ``--trust-author``."""
    validate_python_source(code, "plugin.py", allowed_modules=list(allowed))


def _refusal(code: str, **kwargs) -> str:
    with pytest.raises(PluginValidationError) as excinfo:
        validate_python_source(code, "plugin.py", **kwargs)
    return str(excinfo.value)


# ── the constants themselves ───────────────────────────────────────────────

def test_every_capability_module_is_actually_on_the_blocklist():
    """A capability that unlocks nothing is decoration.

    ``CAPABILITY_MODULES`` is a partition of the blocklist, not a wish list:
    if a root is not blocked, the plugin gate never asks about it and naming
    it here would suggest a gate that does not exist.
    """
    blocked = dangerous_modules()
    for capability, roots in tiers.CAPABILITY_MODULES.items():
        for root in roots:
            assert root in blocked, (
                f"capability {capability!r} claims to unlock {root!r}, "
                "but that root is not on the blocklist so nothing was locked"
            )


def test_no_root_is_unlocked_by_two_capabilities():
    """Otherwise the refusal message cannot name one tier to point at."""
    seen: dict[str, str] = {}
    for capability, roots in tiers.CAPABILITY_MODULES.items():
        for root in roots:
            assert root not in seen, f"{root!r} is in both {seen[root]} and {capability}"
            seen[root] = capability


def test_the_pure_compute_tier_is_never_on_the_blocklist():
    """Tier 0 is a promise: ``import math`` never becomes a capability.

    The plugin gate is a blocklist, so these roots are allowed by
    construction. This asserts that a future hardening pass cannot take one
    away without failing a test that says, in words, that it was a promise.
    """
    overlap = set(tiers.TIER0_PURE_COMPUTE_MODULES) & dangerous_modules()
    assert overlap == set(), f"Tier-0 roots quietly became blocked: {sorted(overlap)}"


def test_no_capability_unlocks_a_dedicated_code_execution_module():
    """The line the three capability names are drawn on.

    Deliberately narrower than "capabilities cannot spawn a process", which is
    what this test used to be called and did not check: it compared module
    NAMES, so it certified an invariant it never tested. ``process-env`` grants
    ``os``, and ``os`` spawns processes -- see the test below, which pins that
    rather than wishing it away.

    What does hold: no capability hands over a module whose PURPOSE is running
    code or reaching the interpreter hosting the plugin.
    """
    never_granted = {
        "subprocess", "sys", "importlib", "ctypes", "pickle", "marshal",
        "dill", "shelve", "runpy", "code", "codeop", "compileall", "signal",
        "atexit", "webbrowser", "threading", "asyncio", "multiprocessing",
    }
    for root in never_granted:
        assert tiers.capability_for_module(root) is None, (
            f"{root!r} was mapped to a capability; it runs code or reaches "
            "the interpreter, so it belongs to --trust-author"
        )


def test_process_env_really_does_grant_process_spawning():
    """Assert what is TRUE, not what the capability's name suggests.

    Every one of these passes the gate under ``process-env`` alone. They are
    pinned here so that nobody -- reader, reviewer, or a future edit to the
    summary string -- can come away believing this capability is read-only.
    The summary and both doc locales say so in words; this says so in code.
    """
    for line in (
        "os.execv('/bin/sh', ['sh'])",
        "os.startfile('x.exe')",
        "os.spawnve(os.P_NOWAIT, 'x', ['x'], {})",
        "os.remove('/some/file')",
        "os.rmdir('/some/dir')",
        "os.write(os.open('/f', os.O_WRONLY), b'x')",
        "os.environ['API_KEY'] = 'stolen'",
    ):
        _tier1(f"import os\n\ndef go():\n    {line}\n", "process-env")


def test_the_process_env_summary_admits_what_the_os_module_is():
    """The prompt is the consent. If it undersells the grant, the consent is
    not informed -- so the wording is a test, not a docstring."""
    summary = tiers.CAPABILITY_SUMMARY["process-env"]
    assert "os module" in summary
    assert "start" in summary          # ... other programs
    assert "delete" in summary or "rename" in summary
    assert "change" in summary         # environment writes, not just reads


def test_the_filesystem_summary_does_not_claim_to_gate_writes():
    """``open(p, 'w')`` is a builtin and passes at Tier 0 with nothing
    declared, so a summary promising "the plugin can write files only with
    this" would be false the moment it was read."""
    assert "open()" in tiers.CAPABILITY_SUMMARY["filesystem"]
    _tier0("def go(p):\n    with open(p, 'w') as fh:\n        fh.write('x')\n")


def test_the_network_summary_admits_it_implies_a_file_write():
    """``urllib.request.urlretrieve(url, dest)`` is one call."""
    assert "disk" in tiers.CAPABILITY_SUMMARY["network"]
    _tier1(
        "import urllib.request\n\ndef go(u, d):\n"
        "    return urllib.request.urlretrieve(u, d)\n",
        "network",
    )


def test_every_capability_has_a_summary_a_human_can_answer_yes_to():
    for capability in tiers.CAPABILITIES:
        summary = tiers.CAPABILITY_SUMMARY[capability]
        assert summary and summary == summary.strip()
        assert len(summary) > 30, "a y/N prompt needs a real sentence"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ()),
        ([], ()),
        (["network"], ("network",)),
        (["Network", " FileSystem "], ("filesystem", "network")),
        (["network", "network"], ("network",)),
        ("network", ("network",)),          # the typo everyone makes once
        (["network", 3, None], ("network",)),
        (42, ()),
    ],
)
def test_normalize_capabilities(raw, expected):
    assert tiers.normalize_capabilities(raw) == expected


def test_unknown_capabilities_are_reported_not_swallowed():
    assert tiers.unknown_capabilities(["network", "gpu"]) == ("gpu",)
    assert tiers.unknown_capabilities(["network"]) == ()


def test_granted_modules_unions_the_groups():
    granted = tiers.granted_modules(["network", "process-env"])
    assert "requests" in granted and "os" in granted
    assert "pathlib" not in granted


# ── the tier matrix ────────────────────────────────────────────────────────
#
# module -> (capability that unlocks it, or None for "trusted only")

_MATRIX: list[tuple[str, str | None]] = [
    ("requests", "network"),
    ("urllib.request", "network"),
    ("http.client", "network"),
    ("socket", "network"),
    ("pathlib", "filesystem"),
    ("tempfile", "filesystem"),
    ("shutil", "filesystem"),
    ("zipfile", "filesystem"),
    ("glob", "filesystem"),
    ("sqlite3", "filesystem"),
    ("codecs", "filesystem"),
    ("os", "process-env"),
    ("subprocess", None),
    ("ctypes", None),
    ("importlib", None),
    ("sys", None),
    ("pickle", None),
    ("threading", None),
]


@pytest.mark.parametrize(("module", "capability"), _MATRIX)
def test_tier0_refuses_every_blocked_module(module, capability):
    message = _refusal(f"import {module}\n")
    assert f"Importing '{module}' is not allowed" in message


@pytest.mark.parametrize(("module", "capability"), _MATRIX)
def test_tier1_accepts_exactly_the_declared_group(module, capability):
    if capability is None:
        message = _refusal(
            f"import {module}\n", capabilities=list(tiers.CAPABILITIES)
        )
        assert "--trust-author" in message
        return
    _tier1(f"import {module}\n", capability)


@pytest.mark.parametrize(("module", "capability"), _MATRIX)
def test_tier1_refuses_a_module_from_a_group_that_was_not_declared(module, capability):
    others = [c for c in tiers.CAPABILITIES if c != capability]
    message = _refusal(f"import {module}\n", capabilities=others)
    assert f"Importing '{module}' is not allowed" in message


@pytest.mark.parametrize(("module", "capability"), _MATRIX)
def test_tier2_accepts_everything_it_was_told_to(module, capability):
    _tier2(f"import {module}\n", module.split(".")[0])


@pytest.mark.parametrize(
    "module", ["math", "statistics", "collections", "itertools", "functools",
               "json", "re", "dataclasses", "typing", "enum", "decimal",
               "random", "numpy", "torch", "pandas"]
)
def test_tier0_accepts_the_pure_compute_list_with_zero_declarations(module):
    _tier0(f"import {module}\n")
    _tier0(f"from {module} import *\n")


# ── the message is the feature ─────────────────────────────────────────────

def test_an_undeclared_capability_module_names_the_capability_and_the_key():
    """The acceptance criterion, verbatim: the refusal has to be actionable.

    Before core#133 this said "Importing 'requests' is not allowed" and
    stopped, leaving the author to find ``--trust-author`` in the docs and
    conclude the project did not want their plugin.
    """
    message = _refusal("import requests\n")
    assert "requires capability 'network'" in message
    assert "[security]" in message
    assert 'capabilities = ["network"]' in message


def test_a_trusted_only_module_says_so_instead_of_naming_a_capability():
    message = _refusal("import subprocess\n")
    assert "capability" in message
    assert "--trust-author" in message
    assert "allowed_modules" in message
    assert "requires capability" not in message


def test_the_capability_refusal_quotes_what_the_user_would_be_agreeing_to():
    message = _refusal("import socket\n")
    assert tiers.CAPABILITY_SUMMARY["network"] in message


def test_asking_for_os_points_at_the_path_only_form():
    """``import os`` is usually ``os.path.join``, and that needs no capability."""
    message = _refusal("import os\n")
    assert "requires capability 'process-env'" in message
    assert "from os.path import" in message


def test_the_refusal_carries_a_line_number_for_the_editor():
    with pytest.raises(PluginValidationError) as excinfo:
        validate_python_source("x = 1\nimport requests\n", "plugin.py")
    assert excinfo.value.lineno == 2


# ── the os.path exception ──────────────────────────────────────────────────

def test_tier0_allows_the_path_helpers_by_their_binding_form():
    _tier0("from os.path import join, basename, splitext\n")
    _tier0("from os.path import join as j\n")
    _tier0("from os.path import dirname, sep, normpath\n")
    # NOTE: this test used to list ``exists``, ``isfile`` and ``abspath`` too.
    # They are refused now -- they stat() the disk or resolve against the
    # working directory -- and the failure of this assertion is what the round-2
    # fix looks like from the outside. See
    # ``test_the_path_exception_binds_only_pure_string_helpers``.


def test_tier0_still_refuses_the_forms_that_bind_the_whole_os_module():
    for line in ("import os", "import os.path", "import os.path as p"):
        message = _refusal(f"{line}\n")
        assert "process-env" in message, line


def test_tier0_refuses_pulling_anything_but_path_out_of_os():
    message = _refusal("from os import path, environ\n")
    assert "process-env" in message


# ── the escape that review found in the first cut of the exception ─────────
#
# ``from os import path`` was in the exception, and leaves were screened by
# NAME against the blocklist. Both were wrong, and it was verified end to end
# with capabilities=[]: ``os.path`` IS ``ntpath`` / ``posixpath``, which
# re-export ``os``, ``sys``, ``stat`` and ``genericpath`` as ordinary
# attributes, so ``path.os.remove(p)`` deleted a real file and
# ``path.sys.modules['subprocess'].run([...])`` ran a real command. The merge
# base refused both forms, so it was a regression of the gate's verdict, in
# the exact commit that attaches a consent prompt and a docs promise to it.

@pytest.mark.parametrize(
    "line",
    [
        "from os import path",                    # binds ntpath/posixpath
        "from os import path as p",
        "from os.path import os",                 # the real os, re-exported
        "from os.path import sys",                # the real sys
        "from os.path import genericpath",        # .os is the real os too
        "from os.path import stat",
        "from os.path import join, genericpath",  # one bad leaf poisons it
        "from os.path import *",                  # __all__ holds expanduser
        "from os.path.genericpath import os",
        "import ntpath",                          # the pre-existing route
        "import posixpath",
        "import genericpath",
        "from ntpath import os",
        "import ntpath as p",
    ],
)
def test_no_spelling_of_os_path_hands_over_a_module(line):
    with pytest.raises(PluginValidationError):
        _tier0(f"{line}\n")


# ── round 2: os.path's surface is not string manipulation ─────────────────
#
# The module screen above closed the escape that had been DEMONSTRATED. It
# could not see the property behind it: ``os.path`` is a real module and most
# of what it exports touches the environment or the disk. None of these is a
# module, so no module screen would ever have caught them -- and
# ``expandvars`` reads the exact thing ``process-env``'s consent line promises
# to gate.

@pytest.mark.parametrize(
    ("leaf", "why"),
    [
        ("expandvars", "reads os.environ -- returned a real API key"),
        ("expanduser", "returns the user's home directory"),
        ("exists", "stat() on any path"),
        ("lexists", "lstat() on any path"),
        ("isfile", "stat() on any path"),
        ("isdir", "stat() on any path"),
        ("islink", "lstat() on any path"),
        ("ismount", "stat() on any path"),
        ("getsize", "stat() -- a true byte size for any file"),
        ("getmtime", "stat()"),
        ("getatime", "stat()"),
        ("getctime", "stat()"),
        ("samefile", "stat() on two paths"),
        ("realpath", "resolves against the working directory"),
        ("abspath", "resolves against the working directory"),
        ("relpath", "resolves against the working directory"),
    ],
)
def test_the_path_exception_binds_only_pure_string_helpers(leaf, why):
    message = _refusal(f"from os.path import {leaf}\n")
    assert "process-env" in message, why
    # ... and it stays refused when smuggled in beside a legitimate one.
    with pytest.raises(PluginValidationError):
        _tier0(f"from os.path import join, {leaf}\n")


def test_every_allowlisted_path_helper_is_real_and_is_not_a_module():
    """The allowlist is a list of NAMES, so it gets a structural guard.

    Checks the two things a name cannot tell you: that it exists on this
    platform's ``os.path`` at all (the tuple is shared by Windows and POSIX
    CI), and that it is not a module.
    """
    import os.path
    import types

    for name in tiers.TIER0_PATH_HELPERS:
        assert hasattr(os.path, name), f"{name!r} is not on this platform's os.path"
        assert not isinstance(getattr(os.path, name), types.ModuleType)


# ── the purity guard ───────────────────────────────────────────────────────
#
# Three screens, and the reason there are three is this file's own recurring
# lesson applied to its own test. The first version inspected the RETURNED
# VALUE, which tests the symptom rather than the property: the whole predicate
# family (``exists``, ``isfile``, ``isdir``, ``islink``, ``ismount``,
# ``lexists``) sails past it by returning a bool while still calling
# ``stat()`` on any path you name.
#
# Screen 2 was specified as an audit hook. Measured on this interpreter it
# catches ZERO of them, because **``os.stat`` raises no audit event** -- the
# CPython audit table covers ``open``, ``os.listdir``, ``os.scandir``,
# ``os.chdir`` and friends, but not ``stat`` or ``getcwd``. It is kept
# anyway: it costs four lines and it is the only screen that sees a helper
# reaching the disk through something other than the ``os`` names screen 3
# replaces.
#
# Screen 3 is the one that does the work: replace the ``os`` functions the
# path modules actually call -- plus ``os.environ`` and, on Windows, the
# private ``os.path._get*`` helpers -- and fail if the call touches any of
# them. That asks the property directly: does this helper consult the
# filesystem, the environment, or the working directory?

_OS_INTERCEPTS = (
    "stat", "lstat", "fstat", "readlink", "getcwd", "getcwdb",
    "listdir", "scandir", "open", "statvfs",
)

#: Windows routes ``abspath`` / ``ismount`` / ``relpath`` through these
#: instead of through ``os``. They are why ``abspath`` read as pure to a
#: source audit that only looked for ``os.``.
_OS_PATH_INTERCEPTS = (
    "_getfullpathname", "_getfinalpathname", "_getfinalpathname_nonstrict",
    "_getvolumepathname", "_nt_readlink", "_abspath_fallback",
)

#: Call shapes for the handful of helpers that do not take one path.
_PATH_HELPER_ARGS: dict[str, tuple] = {
    "join": ("a", "b"),
    "commonpath": (["a", "b"],),
    "commonprefix": (["a", "b"],),
    "samefile": ("a", "b"),
    "sameopenfile": (1, 2),
    "samestat": (None, None),
    "relpath": ("a", "b"),
}

_AUDIT_STATE: dict[str, object] = {"installed": False, "armed": False, "events": []}


def _install_audit_hook() -> None:
    """Add the process-wide hook once. It cannot be removed, so it is cheap
    and inert unless ``armed``."""
    if _AUDIT_STATE["installed"]:
        return
    import sys as _sys

    def _hook(event: str, _args) -> None:
        if _AUDIT_STATE["armed"] and (event.startswith("os.") or event == "open"):
            _AUDIT_STATE["events"].append(event)  # type: ignore[union-attr]

    _sys.addaudithook(_hook)
    _AUDIT_STATE["installed"] = True


def _probe_path_helper(name: str) -> set[str]:
    """Call ``os.path.<name>`` under all three screens; return what it touched.

    An empty set means: touched no intercepted ``os`` function, raised no
    audited event, and returned neither the probe environment variable's value
    nor the working directory.
    """
    import os
    import os.path

    _install_audit_hook()
    touched: list[str] = []

    def _tracker(label: str):
        def _call(*_a, **_k):
            touched.append(label)
            raise OSError("intercepted by the purity guard")
        return _call

    class _EnvProxy:
        def __contains__(self, _key):
            touched.append("os.environ")
            return False

        def __getitem__(self, key):
            touched.append("os.environ")
            raise KeyError(key)

        def get(self, _key, default=None):
            touched.append("os.environ")
            return default

    value = getattr(os.path, name)
    if not callable(value):
        return set()

    saved_os = {k: getattr(os, k) for k in _OS_INTERCEPTS if hasattr(os, k)}
    saved_path = {
        k: getattr(os.path, k) for k in _OS_PATH_INTERCEPTS if hasattr(os.path, k)
    }
    real_environ = os.environ
    marker = "SHOULD-NEVER-APPEAR"
    os.environ["CDUI_PATH_PURITY_PROBE"] = marker
    cwd = os.getcwd()
    # The ``~`` has to LEAD: ``expanduser`` expands only that position, and a
    # probe path with ``~`` in the middle let it slip the guard entirely.
    args = _PATH_HELPER_ARGS.get(name, ("~/%CDUI_PATH_PURITY_PROBE%/x",))

    _AUDIT_STATE["events"] = []
    try:
        for key in saved_os:
            setattr(os, key, _tracker(f"os.{key}"))
        for key in saved_path:
            setattr(os.path, key, _tracker(f"os.path.{key}"))
        os.environ = _EnvProxy()  # type: ignore[assignment]
        _AUDIT_STATE["armed"] = True
        try:
            result = str(value(*args))
        except Exception:
            result = ""
    finally:
        _AUDIT_STATE["armed"] = False
        os.environ = real_environ  # type: ignore[assignment]
        for key, original in saved_os.items():
            setattr(os, key, original)
        for key, original in saved_path.items():
            setattr(os.path, key, original)
        real_environ.pop("CDUI_PATH_PURITY_PROBE", None)

    touched.extend(_AUDIT_STATE["events"])  # type: ignore[arg-type]
    if marker in result:
        touched.append("returned os.environ")
    if cwd and cwd in result:
        touched.append("returned the cwd")
    return set(touched)


def test_no_allowlisted_path_helper_touches_the_environment_the_disk_or_the_cwd():
    """Every entry on the allowlist, verified by CALLING it under all three
    screens -- on whichever platform is running the suite, so ``ntpath`` here
    and ``posixpath`` on CI."""
    import os.path

    for name in tiers.TIER0_PATH_HELPERS:
        value = getattr(os.path, name)
        if not callable(value):
            assert isinstance(value, (str, bool, type(None))), (
                f"{name} is a {type(value).__name__}, which is not an inert constant"
            )
            continue
        touched = _probe_path_helper(name)
        assert touched == set(), f"{name} touched {sorted(touched)}"


def test_the_purity_guard_catches_every_impure_name_on_os_path():
    """The guard's own sensitivity, measured rather than asserted.

    A guard that inspects nothing passes trivially -- which is exactly how the
    value-only version shipped while catching 3 of the 25 names it would have
    to catch. So: take every public name on ``os.path`` that is NOT on the
    allowlist, and require that adding it would be caught by SOME screen in
    this file. Each name lands in exactly one bucket, and a name that lands in
    none fails the test.
    """
    import os.path
    import types

    #: Genuinely inert: not callable, so there is nothing to call.
    inert = {"devnull", "supports_unicode_filenames", "ALLOW_MISSING"}
    #: Pure: compares two ``stat_result`` objects and calls nothing. A plugin
    #: cannot obtain one at Tier 0 anyway -- the same shape as ``json.dump``
    #: needing a file object it cannot get.
    pure_but_unusable = {"samestat"}

    by_module_screen: list[str] = []
    by_interception: list[str] = []
    accounted_inert: list[str] = []
    unaccounted: list[str] = []

    for name in sorted(dir(os.path)):
        if name.startswith("_") or name in set(tiers.TIER0_PATH_HELPERS):
            continue
        value = getattr(os.path, name)
        if isinstance(value, types.ModuleType):
            by_module_screen.append(name)          # the module screen catches these
        elif name in inert or name in pure_but_unusable:
            accounted_inert.append(name)
        elif _probe_path_helper(name):
            by_interception.append(name)
        else:
            unaccounted.append(name)

    assert unaccounted == [], (
        f"these os.path names are neither allowlisted nor caught by any screen: "
        f"{unaccounted} -- either they are pure and belong on TIER0_PATH_HELPERS, "
        f"or the guard has a hole"
    )
    # Non-vacuity: the buckets must actually contain the names we know about.
    assert {"os", "sys", "genericpath"} <= set(by_module_screen)
    assert {
        "expandvars", "expanduser", "exists", "isfile", "isdir", "islink",
        "lexists", "getsize", "getmtime", "abspath", "realpath", "relpath",
    } <= set(by_interception), f"interception only caught {by_interception}"


def test_the_helpers_a_plugin_actually_wants_still_work():
    _tier0(
        "from os.path import join, basename, dirname, splitext, sep\n\n"
        "def under(root, name):\n"
        "    stem, _ext = splitext(basename(name))\n"
        "    return join(root, dirname(name), stem) + sep\n"
    )
    _tier0("from os.path import normpath, normcase, isabs, splitdrive, split\n")
    _tier0("from os.path import commonpath, commonprefix, curdir, pardir\n")


def test_the_module_leaf_screen_asks_what_a_leaf_IS_not_what_it_is_called():
    """``genericpath`` cleared a blocklist-NAME screen and handed back a module
    whose ``.os`` is the real thing. The screen is computed from the live
    ``os.path`` instead, so a future CPython re-export is covered with no edit.
    """
    import os.path
    import types

    from app.core.plugin_validator import _OS_PATH_MODULE_LEAVES  # noqa: SLF001

    actual = {
        name
        for name in dir(os.path)
        if isinstance(getattr(os.path, name, None), types.ModuleType)
    }
    assert _OS_PATH_MODULE_LEAVES == actual
    assert {"os", "sys", "genericpath"} <= actual


def test_the_os_path_aliases_are_reachable_only_with_process_env():
    """They ARE ``os.path``, and granting ``os`` already gives you them."""
    for root in ("ntpath", "posixpath", "genericpath"):
        assert root in dangerous_modules()
        assert tiers.capability_for_module(root) == "process-env"
        _tier1(f"import {root}\n", "process-env")


def test_a_dunder_cannot_be_imported_under_a_name():
    """``from json import __builtins__ as b`` then ``b['eval']``: every other
    spelling of a forbidden dunder is refused, but an import BINDS it without
    writing a Name, Attribute, Subscript or getattr node, so nothing looked at
    it -- and every module has a ``__builtins__``. Pre-existing on both gate
    shapes."""
    from app.core.script_policy import validate_script_source

    for code in (
        "from json import __builtins__ as b\n",
        "from json import __builtins__\n",
        "import json\nfrom json import __loader__\n",
    ):
        with pytest.raises(PluginValidationError):
            _tier0(code)
    with pytest.raises(PluginValidationError):
        validate_script_source("from json import __builtins__ as b\n")


def test_process_env_still_unlocks_the_whole_os_module():
    _tier1("import os\nprint(os.environ.get('WANDB_API_KEY'))\n", "process-env")


def test_the_rce_leaves_stay_refused_even_with_process_env():
    """A capability grants a resource. ``os.system`` is not a resource."""
    with pytest.raises(PluginValidationError):
        _tier1("import os\nos.system('whoami')\n", "process-env")
    with pytest.raises(PluginValidationError):
        _tier1("import os\nos.popen('whoami')\n", "process-env")


# ── acceptance criteria ────────────────────────────────────────────────────

_WANDB_SHAPED_PLUGIN = """\
import json

import requests

from app.core.node_base import BaseNode


class MetricLogger(BaseNode):
    NODE_NAME = "MetricLogger"
    CATEGORY = "Logging"
    DESCRIPTION = "POST a scalar metric to a run tracker."

    def execute(self, inputs, params, context=None):
        payload = json.dumps({"loss": float(inputs["loss"])})
        response = requests.post(params["url"], data=payload, timeout=5)
        return {"status": response.status_code}
"""


def test_a_wandb_shaped_plugin_validates_with_the_network_capability():
    """Acceptance 1 -- and note what it does NOT need: ``--trust-author``."""
    _tier1(_WANDB_SHAPED_PLUGIN, "network")


def test_the_same_plugin_is_refused_with_no_declaration():
    """Acceptance 3."""
    message = _refusal(_WANDB_SHAPED_PLUGIN)
    assert "requires capability 'network'" in message


def test_torch_load_still_needs_weights_only_in_every_tier():
    """Rule 2 of the spec: the pickle gate is not a capability."""
    for kwargs in ({}, {"capabilities": list(tiers.CAPABILITIES)},
                   {"allowed_modules": ["os", "subprocess"]}):
        with pytest.raises(PluginValidationError):
            validate_python_source(
                "import torch\n\ndef go(p):\n    return torch.load(p)\n",
                "plugin.py",
                **kwargs,
            )
    validate_python_source(
        "import torch\n\ndef go(p):\n    return torch.load(p, weights_only=True)\n",
        "plugin.py",
    )


def test_dunder_access_is_still_refused_in_every_tier():
    """Rule 2 again. A capability never buys reflection."""
    for kwargs in ({}, {"capabilities": list(tiers.CAPABILITIES)},
                   {"allowed_modules": ["os"]}):
        with pytest.raises(PluginValidationError):
            validate_python_source(
                "def go(x):\n    return x.__class__.__bases__\n", "plugin.py", **kwargs
            )
        with pytest.raises(PluginValidationError):
            validate_python_source("def go():\n    return eval('1')\n",
                                   "plugin.py", **kwargs)


def test_absent_capabilities_mean_exactly_what_they_meant_before():
    """Acceptance 4 -- the migration.

    A lockfile written before core#133 has no ``capabilities`` key, which
    reads as the empty tuple, which is the pre-#133 behaviour of every path
    that does not pass the argument at all.
    """
    for absent in (None, (), []):
        message = _refusal("import requests\n", capabilities=absent)
        assert "requires capability 'network'" in message
    # ... and the trusted path is untouched by any of this.
    _tier2("import requests\n", "requests")


# ── false positives the tiered rewrite was supposed to fix ─────────────────

def test_re_compile_is_not_the_builtin_compile():
    """core#178. ``_DANGEROUS_NAMES`` matched a call's LEAF whatever the
    receiver was, so ``re.compile`` -- the headline function of an advertised
    Tier-0 module -- was refused as if it were ``compile()``."""
    _tier0("import re\n\nPATTERN = re.compile(r'\\d+')\n")


def test_torch_compile_is_not_the_builtin_compile_either():
    """core#178, the other half."""
    _tier0("import torch\n\ndef go(m):\n    return torch.compile(m)\n")


def test_a_module_can_be_put_in_eval_mode():
    """core#174. ``model.eval()`` is how every PyTorch plugin switches off
    dropout, and it was refused as if it were ``eval()``."""
    _tier0("def go(model):\n    model.eval()\n    return model\n")
    _tier0("import torch\n\ndef go(m):\n    return torch.nn.Sequential(m).eval()\n")


@pytest.mark.parametrize(
    "snippet",
    [
        "obj.vars()",
        "self.dir()",
        "registry.globals()",
        "cfg.locals()",
    ],
)
def test_ordinary_methods_that_share_a_name_with_a_builtin_are_allowed(snippet):
    _tier0(f"def go(obj, self, registry, cfg):\n    return {snippet}\n")


def test_the_builtins_module_is_still_the_builtins_module():
    """The relaxation above is receiver-aware, not blanket: reaching the real
    ``eval`` through the module that defines it is the whole shape it must not
    open."""
    for code in (
        "import builtins\n\ndef go(s):\n    return builtins.eval(s)\n",
        "import builtins as b\n\ndef go(s):\n    return b.exec(s)\n",
        "import builtins\n\ndef go(s):\n    return builtins.compile(s, '<s>', 'eval')\n",
        "from builtins import eval as ev\n",
    ):
        with pytest.raises(PluginValidationError):
            _tier0(code)


def test_a_bare_dangerous_builtin_is_refused_exactly_as_before():
    for code in ("eval('1')", "exec('x=1')", "compile('1', '<s>', 'eval')",
                 "__import__('os')", "globals()", "vars()"):
        with pytest.raises(PluginValidationError):
            _tier0(f"def go():\n    return {code}\n")


# ── the in-canvas script tier must not have moved ──────────────────────────

def test_none_of_this_widens_the_script_node_allowlist():
    """core#131's boundary, restated as a test in core#133's own suite.

    Tier 0 for an INSTALLED file and Tier 0 for code typed into the browser
    are different surfaces with the same name. The script node's list is an
    allowlist of dotted paths audited across six adversarial rounds, and
    nothing in core#133 may add to it.
    """
    from app.core.script_policy import TIER0_MODULE_PATHS, validate_script_source

    assert set(TIER0_MODULE_PATHS) == {
        "collections", "collections.abc", "itertools", "json", "math",
        "numpy", "numpy.linalg", "numpy.random", "re", "statistics",
        "torch", "torch.nn", "torch.nn.functional", "torch.signal",
        "torch.signal.windows",
    }
    for refused in ("os", "os.path", "pathlib", "pandas", "typing",
                    "functools", "requests", "dataclasses"):
        with pytest.raises(PluginValidationError):
            validate_script_source(f"import {refused}\n")


def test_the_script_tier_gets_the_compile_fix_too():
    """``re.compile`` was refused for in-canvas scripts as well, and ``re`` is
    on the four-module list the editor advertises."""
    from app.core.script_policy import validate_script_source

    validate_script_source(
        "import re\n\n\ndef run(inputs, params):\n"
        "    return len(re.compile(r'\\d+').findall(params['text']))\n"
    )


def test_the_script_tier_refuses_the_builtins_attribute_by_name():
    """The receiver-aware relaxation leans on ``builtins`` being unreachable
    from a tier-0 script; assert the gate says so rather than assuming it."""
    from app.core.script_policy import SCRIPT_PROXY_DENIED_ATTRS

    assert "builtins" in SCRIPT_PROXY_DENIED_ATTRS


def test_torch_compile_stays_refused_for_the_script_tier():
    """``torch.compile`` hands the graph to TorchInductor, which generates
    C++/Triton and shells out to a compiler -- the same door round 5 of
    core#131 closed by refusing ``torch.utils.cpp_extension``. Relaxing the
    call rule to fix ``re.compile`` must not open it by accident.
    """
    from app.core.script_policy import validate_script_source

    for code in (
        "import torch\n\n\ndef run(i, p):\n    return torch.compile(p['m'])\n",
        "import torch as t\n\n\ndef run(i, p):\n    return t.compile(p['m'])\n",
        "import torch\n\n\ndef run(i, p):\n    f = torch.compile\n    return f(p['m'])\n",
        "import torch\n\n\ndef run(i, p):\n"
        "    return getattr(torch, 'compile')(p['m'])\n",
        "import torch\n\n\ndef run(i, p):\n"
        "    return (lambda: torch)().compile(p['m'])\n",
    ):
        with pytest.raises(PluginValidationError, match="compile"):
            validate_script_source(code)


def test_the_scoped_rule_has_a_runtime_lock_too():
    """A gate-only rule is a rule a graph that never met the editor evades."""
    import torch

    from app.core.script_proxy import module_proxy
    from app.core.script_policy import ScriptPolicyError

    proxy = module_proxy(torch)
    with pytest.raises(ScriptPolicyError, match="compile"):
        proxy.compile


def test_the_module_compile_method_is_a_documented_tier0_false_positive():
    """PyTorch >= 2.2 put ``compile`` on ``nn.Module`` itself, so ``m.compile()``
    is refused for in-canvas scripts along with ``torch.compile(m)`` -- it is
    the same TorchInductor path by another spelling. A real false positive for
    anyone who wanted only the speed-up, so it is stated in both doc locales
    rather than left to be discovered, and pinned here so the docs stay true.
    """
    import torch

    from app.core.script_policy import validate_script_source

    assert hasattr(torch.nn.Module, "compile"), "the premise of this test moved"
    with pytest.raises(PluginValidationError, match=r"\.compile"):
        validate_script_source(
            "import torch\n\n\ndef run(inputs, params):\n"
            "    m = torch.nn.Linear(2, 2)\n    m.compile()\n    return 1\n"
        )


def test_an_installed_plugin_may_still_call_torch_compile():
    """The scoping is a tier-0 argument, not a global rule: core#178 asked for
    ``torch.compile`` to work in plugins, and a plugin is a file the user
    chose to install."""
    _tier0("import torch\n\n\ndef go(m):\n    return torch.compile(m)\n")


def test_the_builtins_module_cannot_be_laundered_through_an_attribute():
    """``e = builtins.eval`` then ``e(x)`` writes no Call with an ``eval``
    leaf, so the call rule never saw it -- the same one-assignment evasion
    core#131 found for ``torch.load`` and ``os.system``."""
    for code in (
        "import builtins\n\ndef go(s):\n    e = builtins.eval\n    return e(s)\n",
        "import builtins as b\n\ndef go(s):\n    f = b.exec\n    return f(s)\n",
        "import builtins\n\ndef go(s):\n    return getattr(builtins, 'eval')(s)\n",
        "import builtins\n\ndef go():\n    c = builtins\n    return c.compile\n",
    ):
        with pytest.raises(PluginValidationError):
            _tier0(code)


def test_no_relaxed_call_name_reaches_a_real_builtin():
    """The empirical basis for the receiver-aware call rule, as a standing check.

    Relaxing ``x.eval()`` / ``x.compile()`` is safe only while nothing on the
    tier-0 surface hands back the actual builtin under one of those names. That
    is a statement about today's numpy and torch, not a property of the rule --
    which is exactly the kind of claim core#131 kept having to retract -- so it
    is measured here rather than asserted in a comment.

    Today: 12 attributes match by name (``re.compile`` and eleven TorchScript
    C++ methods on ``torch._C`` classes -- ``ScriptObject.getattr``,
    ``Node.input``, ...), and **none of them is the builtin**. A torch release
    that re-exported ``builtins.eval`` under one of these names would fail
    here instead of waiting for a reviewer.
    """
    import builtins

    from app.core.plugin_validator import _DANGEROUS_NAMES  # noqa: SLF001
    from app.core.script_policy import (
        SCRIPT_PROXY_DENIED_ATTRS,
        TIER0_DENIED_CALLS,
        ScriptPolicyError,
    )
    from app.core.script_proxy import RestrictedModule, tier0_module_namespace, unwrap

    watched = set(_DANGEROUS_NAMES) | set(TIER0_DENIED_CALLS)
    denied = set(SCRIPT_PROXY_DENIED_ATTRS)
    banned = [
        obj
        for obj in (getattr(builtins, name, None) for name in watched)
        if obj is not None
    ]

    seen: set[str] = set()
    hits: list[str] = []
    leaked: list[str] = []
    members = 0

    def check(value, path: str) -> None:
        nonlocal leaked
        hits.append(path)
        if any(value is candidate for candidate in banned):
            leaked.append(path)

    def walk(proxy, label: str, depth: int) -> None:
        nonlocal members
        if label in seen or depth > 3:
            return
        seen.add(label)
        for name in sorted(dir(unwrap(proxy))):
            if name.startswith("_"):
                continue
            try:
                value = getattr(proxy, name)
            except (ScriptPolicyError, AttributeError, Exception):
                continue
            if isinstance(value, RestrictedModule):
                walk(value, f"{label}.{name}", depth + 1)
                continue
            if name in watched:
                check(value, f"{label}.{name}")
            if isinstance(value, type) and depth <= 2:
                for attr in dir(value):
                    if attr.startswith("_") or attr in denied:
                        continue
                    members += 1
                    if attr not in watched:
                        continue
                    try:
                        check(getattr(value, attr), f"{label}.{name}.{attr}")
                    except Exception:  # pragma: no cover - exotic descriptor
                        continue

    for root, proxy in sorted(tier0_module_namespace().items()):
        walk(proxy, root, 1)

    assert leaked == [], f"a real builtin is reachable as: {leaked}"
    # Non-vacuity: a sweep that inspects nothing passes trivially, which is how
    # a surface goes unswept for five rounds.
    assert len(seen) >= 8, f"only walked {sorted(seen)}"
    assert members > 3_000, f"only inspected {members} class members"
    assert "re.compile" in hits, "the sweep never reached the name it exists for"


def test_the_path_exception_cannot_be_used_to_pull_os_back_out():
    """``ntpath`` / ``posixpath`` import ``os``, ``sys`` and ``stat``, so
    ``from os.path import os`` would have bound the real ``os`` module through
    the very module the Tier-0 exception waves through."""
    for line in ("from os.path import os", "from os.path import join, os",
                 "from os.path import sys", "from os import *"):
        with pytest.raises(PluginValidationError):
            _tier0(f"{line}\n")
    # ... while the genuine path helpers keep working.
    _tier0("from os.path import join, basename, dirname, splitext, sep\n")
