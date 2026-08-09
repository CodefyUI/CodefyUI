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
    # core#183 -- the raw C modules `os` is built on (CPython's own `os.py`
    # does `from nt import *` / `from posix import *`, then `del`s the name).
    # Reaching them by their own name is reaching `os` itself: same grant,
    # same tier.
    ("nt", "process-env"),
    ("posix", "process-env"),
    # core#177 CI round 2 -- surfaced by the enumeration test meeting a real
    # Linux interpreter (nineteen/twenty unclassified names, platform- and
    # version-dependent). Two of them earned an outright decision rather than
    # deferral to ACCEPTED_UNGATED_MODULES, each independently verified, not
    # assumed:
    #   `readline.add_history(s)` + `.write_history_file(path)` writes
    #   attacker-directed content to an attacker-chosen path -- verified
    #   directly (wrote a marked payload, read it back from the target file).
    #   Same shape as CAPABILITY_MODULES["filesystem"]'s other members.
    ("readline", "filesystem"),
    # `spwd.getspnam()` reads the shadow password-hash database. Verified
    # directly (not assumed from "you have to be root", which the module's
    # own docstring claims and which turned out to be false in practice): a
    # plain `open("/etc/shadow")` correctly raised PermissionError for an
    # unprivileged, non-`shadow`-group test account, but `spwd.getspnam()`
    # from the SAME account succeeded -- NSS-mediated access can bypass the
    # file's own permission bits. No existing capability fits "read the
    # shadow database"; Tier 2 only, same bucket as `pickle` / `ctypes`.
    ("spwd", None),
    ("subprocess", None),
    ("ctypes", None),
    ("importlib", None),
    ("sys", None),
    ("pickle", None),
    ("threading", None),
]

# core#215 / core#216 -- the same matrix, for the names core#183 stopped
# short of. Two shapes, and they are kept in one list because the tier
# question is identical for both:
#
#   * the raw C module a blocklisted one is BUILT ON. `os` had `nt`/`posix`;
#     `socket` has `_socket`, `pickle` has `_pickle`, and so on down the
#     stdlib. Reaching the private module by name is reaching the public one,
#     so it takes the public one's tier -- which is what makes `_socket`
#     "network" and `_thread` trusted-only, rather than one blanket answer.
#   * a whole surface nobody had enumerated in either direction (`winreg`,
#     the sub-interpreter family, the POSIX account databases, the tty
#     modules). Tier 2 for all of them: no existing capability's consent
#     sentence describes any of these, and `spwd` above is the precedent --
#     "no capability fits" means trusted-only, not a new capability invented
#     in passing.
#
# Every entry's verification is recorded in `plugin_validator`'s own comment
# next to the name, or in `CAPABILITY_MODULES` for the ones a capability
# covers. Platform is irrelevant here: the gate is a static AST walk, so
# `import msvcrt` is refused on Linux and `import pwd` on Windows, which is
# the correct answer for a tarball that can be installed anywhere.
_MATRIX += [
    # -- raw C implementations, capability-covered ----------------------
    ("_socket", "network"),
    ("_ssl", "network"),
    # `ssl` itself was never on the blocklist at all, and it is not merely a
    # cipher library: see the dedicated test below.
    ("ssl", "network"),
    ("_sqlite3", "filesystem"),
    # -- raw C implementations, trusted-only ----------------------------
    ("_thread", None),
    ("_ctypes", None),
    ("_pickle", None),
    ("_signal", None),
    ("_asyncio", None),
    ("_imp", None),
    ("_frozen_importlib", None),
    ("_frozen_importlib_external", None),
    ("_multiprocessing", None),
    ("_posixsubprocess", None),
    ("_posixshmem", None),
    ("_winapi", None),
    ("_overlapped", None),
    ("msvcrt", None),
    # -- surfaces nobody had considered, trusted-only -------------------
    ("winreg", None),
    ("_interpreters", None),
    ("_xxsubinterpreters", None),
    ("_interpchannels", None),
    ("_xxinterpchannels", None),
    ("_interpqueues", None),
    ("pwd", None),
    ("grp", None),
    ("syslog", None),
    ("termios", None),
    ("_curses", None),
    ("_curses_panel", None),
    ("ossaudiodev", None),
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


# ── core#183: `nt` / `posix` are `os`, not a different surface ─────────────

def test_import_nt_or_posix_reaches_the_real_environ_and_remove_at_tier0():
    """The exact escape core#183 reported, reproduced directly rather than
    only through the matrix above.

    ``nt`` / ``posix`` are the raw C modules CPython's own ``Lib/os.py``
    builds ``os`` from (``from nt import *`` on Windows, ``from posix
    import *`` elsewhere) -- every ``os.remove``, ``os.environ`` and
    ``os.system`` originates there. Neither name was ever on the blocklist
    in either direction: not on ``_DANGEROUS_MODULES``, not in
    ``CAPABILITY_MODULES["process-env"]``, not in
    ``TIER0_PURE_COMPUTE_MODULES``. Before the fix, ``import nt`` at Tier 0
    (nothing declared) was accepted outright, and only ``nt.system`` /
    ``nt.popen`` tripped anything -- the receiver-independent RCE-leaf list,
    which has no idea what module it is looking at. A real, writable
    ``nt.environ`` and a real ``nt.remove(path)`` passed clean.
    """
    for module in ("nt", "posix"):
        _refusal(f"import {module}\n")
        _refusal(f"import {module}\n{module}.environ\n")
        _refusal(f"import {module}\n{module}.remove('C:/tmp/x')\n")
        # already caught before the fix -- confirms the leaf rule still
        # fires independently of the module-level one added for this issue.
        _refusal(f"import {module}\n{module}.system('x')\n")


def test_nt_and_posix_are_gated_by_process_env_not_a_new_capability():
    """Same door as ``os``, not a new one: no capability vocabulary change,
    just closing the two names that reached the same module by a side door."""
    _tier1("import nt\nnt.environ\n", "process-env")
    _tier1("import posix\nposix.environ\n", "process-env")
    message = _refusal("import nt\n", capabilities=["network", "filesystem"])
    assert "process-env" in message


def test_the_platform_native_os_backing_module_is_always_on_the_blocklist():
    """core#177 -- a standing check for the SHAPE of core#183, not just its
    two names, so a future regression here fails a test instead of waiting
    for someone to run the validator by hand again.

    core#183 happened because ``nt`` / ``posix`` were never enumerated
    anywhere, in either direction -- nobody had to remove them from a list;
    they simply were never added. A test that re-asserts ``"nt" in
    dangerous_modules()`` guards against the first kind of regression
    (someone deletes the line) but not the second (the same shape recurring
    under a name nobody typed into a test file), and it only ever checks the
    name, never the platform that made it true.

    So this asks the INTERPRETER instead, the same "ask, don't hardcode" move
    :func:`app.core.plugin_validator._compute_os_path_module_leaves` makes for
    ``os.path``: CPython's own ``Lib/os.py`` sets ``os.name`` to exactly the
    raw module it built its implementation from (``'nt'`` on Windows,
    ``'posix'`` everywhere else CI runs) and then discards the module
    reference (``from nt import *`` ... ``del nt``) -- so ``os.name`` is the
    interpreter's own answer to "which raw module must never be missing from
    the blocklist", not an assumption this test bakes in. Run on CI's Linux
    box this exercises ``posix``; on a Windows dev machine, ``nt``; on any
    future platform Python grows, whatever ``os.name`` says there. A
    regression that drops one of the two names from the blocklist while
    leaving the other still fails this test on whichever platform runs it.
    """
    import os
    import sys

    native = os.name
    assert native in sys.builtin_module_names, (
        f"expected the interpreter to report {native!r} as a builtin module -- "
        f"if this fails, os.name no longer names a real builtin module and the "
        f"assumption this test relies on needs revisiting, not the assertion"
    )
    assert native in dangerous_modules(), (
        f"'{native}' is the raw module os.py builds os.remove / os.system / "
        f"os.environ from (Lib/os.py: `from {native} import *`); it must stay "
        f"on the blocklist or importing it directly reaches everything "
        f"'process-env' gates with zero capability declared (core#183)"
    )


# ── core#215: the siblings core#183 stopped short of ───────────────────────

def test_the_runtime_implementation_blocklist_is_ported_to_the_install_time_gate():
    """core#215's literal complaint, as a standing check.

    ``script_proxy._BLOCKED_IMPLEMENTATION_ROOTS`` is the RUNTIME boundary
    for in-canvas scripts, and it has known since long before core#215 that
    ``_thread``, ``_socket``, ``_ctypes``, ``_pickle``, ``_winapi``,
    ``msvcrt``, ``_posixsubprocess``, ``_multiprocessing``, ``_imp`` and the
    two ``_frozen_importlib`` modules are dangerous -- it refuses any value
    whose defining module is one of them. That recognition was never carried
    across to the INSTALL-time gate, so the same codebase held both of these
    at once::

        collections._ctypes    # refused: a script may not hold that
        import _ctypes         # accepted: a plugin may, zero declaration

    The two boundaries protect the same process from the same code, so a
    name one of them calls dangerous cannot be a name the other has never
    heard of. ``_io`` is the single exception and it is not a gap: it is a
    documented ACCEPT in ``ACCEPTED_UNGATED_MODULES``, because ``open()`` is
    deliberately ungated for installed files, so the assertion asks for
    "classified", not "blocked".
    """
    from app.core import script_proxy

    blocked = dangerous_modules()
    accepted = set(tiers.ACCEPTED_UNGATED_MODULES)
    unclassified = sorted(
        name for name in script_proxy._BLOCKED_IMPLEMENTATION_ROOTS
        if name not in blocked and name not in accepted
    )
    assert unclassified == [], (
        "the in-canvas RUNTIME boundary refuses values defined by these "
        f"modules, but the install-time gate has never heard of them: "
        f"{unclassified}. A plugin can therefore `import` what a script "
        "cannot even be handed. Put each on _DANGEROUS_MODULES, or record "
        "an explicit accept in ACCEPTED_UNGATED_MODULES the way _io is"
    )


def test_the_underscore_implementation_of_a_blocked_module_is_classified_too():
    """The SHAPE of core#215, not its list of names -- so the next one is
    caught mechanically instead of by somebody auditing by hand again.

    CPython's own convention is that a stdlib module ``x`` is a thin Python
    wrapper over a private C module ``_x``: ``socket``/``_socket``,
    ``pickle``/``_pickle``, ``signal``/``_signal``, ``ssl``/``_ssl``,
    ``sqlite3``/``_sqlite3``, ``bz2``/``_bz2``, ``asyncio``/``_asyncio``.
    Every one of those pairs was a hole. So: for each name on the blocklist,
    if this interpreter actually HAS a module called ``_<name>``, that module
    must be classified too -- blocked, or accepted with a reason.

    Deliberately keyed on the interpreter (``importlib.util.find_spec``)
    rather than a hardcoded list, the same "ask, don't assume" move
    ``_compute_os_path_module_leaves`` makes: a future CPython that grows a
    ``_webbrowser`` accelerator fails this test the day it ships.

    What this canNOT see, said plainly so nobody mistakes it for total
    coverage: the pairs whose names do not rhyme. ``_thread`` is behind
    ``threading``, ``_posixsubprocess`` behind ``subprocess``,
    ``_overlapped`` behind ``asyncio``. Those were found by importing every
    blocklisted module in a fresh interpreter and diffing ``sys.modules``,
    which is a sweep to re-run when auditing, not an assertion cheap enough
    to run per-test. This covers the mechanical majority; the sweep covers
    the rest.
    """
    import importlib.util

    blocked = dangerous_modules()
    accepted = set(tiers.ACCEPTED_UNGATED_MODULES)
    unclassified = []
    for name in sorted(blocked):
        private = "_" + name
        if private in blocked or private in accepted:
            continue
        try:
            exists = importlib.util.find_spec(private) is not None
        except (ImportError, ValueError):
            exists = False
        if exists:
            unclassified.append((private, name))
    assert unclassified == [], (
        "these private C modules sit directly behind a module this gate "
        f"already blocks, and are themselves unclassified: {unclassified}. "
        "Reaching the private one by name reaches the public one's surface "
        "(core#183 for nt/posix, core#215 for the rest) -- decide about each: "
        "blocklist it (plus a CAPABILITY_MODULES group if its wrapper has "
        "one), or record why it grants nothing in ACCEPTED_UNGATED_MODULES"
    )


@pytest.mark.parametrize(
    ("module", "reaches"),
    [
        # Each line is a primitive VERIFIED to work through the raw module
        # alone, with no import of its public wrapper anywhere. The verdicts
        # are recorded next to the names in `plugin_validator`; these pin
        # that the gate refuses the shape an attacker would actually write.
        ("_thread", "_thread.start_new_thread(payload, ())"),
        ("_socket", "_socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)"),
        ("_ctypes", "_ctypes.LoadLibrary('kernel32.dll', 0)"),
        ("_pickle", "_pickle.loads(blob)"),
        ("_signal", "_signal.raise_signal(15)"),
        ("_imp", "_imp.create_dynamic(spec)"),
        ("_frozen_importlib", "_frozen_importlib.__import__('subprocess')"),
        ("_winapi", "_winapi.CreateProcess(None, cmd, None, None, 0, 0, None, None, None)"),
        ("_posixsubprocess", "_posixsubprocess.fork_exec(argv, argv, 1, (), None, None)"),
        ("_posixshmem", "_posixshmem.shm_open('/x', 66, 384)"),
        ("winreg", "winreg.SetValueEx(key, 'Run', 0, 1, payload)"),
        ("_interpreters", "_interpreters.exec(iid, src)"),
        ("_xxsubinterpreters", "_xxsubinterpreters.run_string(iid, src)"),
        ("pwd", "pwd.getpwall()"),
        ("grp", "grp.getgrall()"),
        ("syslog", "syslog.openlog(ident='sshd')"),
        ("termios", "termios.tcsetattr(fd, 0, attrs)"),
        ("_curses", "_curses.initscr()"),
    ],
)
def test_the_verified_escape_is_refused_at_tier0(module, reaches):
    """One row per primitive that was actually made to run during the
    core#215 / core#216 triage, refused at Tier 0 in the spelling an
    attacker would use rather than only as a bare import.

    The import is what the gate sees, so the import is what has to be
    refused -- but writing the CALL out here is the point: it records what
    the module was verified to REACH, in code, next to the assertion that it
    is closed. ``_interpreters.exec(...)`` is the one worth reading twice:
    ``exec`` is in ``_DANGEROUS_NAMES``, and the call rule still does not
    fire on it, because ``_calls_the_builtin`` asks whether the receiver
    resolves to ``builtins`` and this receiver is a module. Only the import
    rule closes that door.
    """
    message = _refusal(f"import {module}\n\ndef go():\n    {reaches}\n")
    assert f"Importing '{module}' is not allowed" in message


def test_the_subinterpreter_exec_primitive_is_blocked_under_both_of_its_names():
    """The rename is the trap, and gating only the name in the issue would
    have shipped a no-op.

    core#216 reports ``_xxsubinterpreters``. CPython RENAMED that module to
    ``_interpreters`` in 3.13, so on 3.13+ -- which includes the interpreter
    this repo's own developers run -- ``_xxsubinterpreters`` does not exist
    and gating it protects nothing, while the live primitive sits under a
    name still absent from every list. The reverse holds on 3.10-3.12, which
    is what CI runs. Both names, or the gate is version-dependent.

    Whichever name this interpreter actually has must also be REAL: the
    assertion below refuses to pass vacuously on a Python that has neither.
    """
    import importlib.util

    for name in ("_xxsubinterpreters", "_interpreters"):
        assert name in dangerous_modules(), (
            f"{name!r} executes arbitrary source in a sub-interpreter "
            f"(create() + exec(id, src), verified to spawn a process from "
            f"inside one); it must be blocked under BOTH spellings because "
            f"CPython renamed it in 3.13"
        )
        message = _refusal(f"import {name}\n")
        assert "--trust-author" in message, (
            "no capability may ever unlock an exec() primitive"
        )

    present = [
        name for name in ("_xxsubinterpreters", "_interpreters")
        if importlib.util.find_spec(name) is not None
    ]
    assert present, (
        "this interpreter has neither name, so the test above proved nothing "
        "about a reachable module -- if CPython renamed it again, add the new "
        "spelling to the blocklist and to this test"
    )


def test_ssl_alone_reaches_the_network_and_now_says_so():
    """``ssl`` was never on the blocklist in either direction, and it is not
    just a cipher library.

    ``ssl.get_server_certificate((host, port))`` opens the connection
    ITSELF -- its own source calls ``create_connection`` -- so ``import
    ssl`` was outbound network egress at Tier 0 with nothing declared,
    verified by fetching 1411 bytes of live certificate from a real host.
    That makes an attacker-chosen host:port reachable, which is a beacon
    whatever comes back.

    core#215 flagged ``_ssl`` and noted in passing that its wrapper "is not
    on the blocklist at all today". Gating the raw engine while leaving the
    wrapper open would have been a gate protecting nothing, so both are on
    ``network`` -- the capability whose consent line already promises
    exactly this ("send and receive data from any host").
    """
    assert "ssl" in dangerous_modules()
    assert tiers.capability_for_module("ssl") == "network"
    assert tiers.capability_for_module("_ssl") == "network"
    _tier1("import ssl\n\ndef go(h):\n    return ssl.get_server_certificate((h, 443))\n",
           "network")
    message = _refusal("import ssl\n", capabilities=["filesystem", "process-env"])
    assert "requires capability 'network'" in message


def test_the_capability_summaries_still_cover_what_was_added_to_their_groups():
    """A group grew; the sentence a user answers y/N to has to still be true
    of everything in it.

    ``network`` gained ``_socket`` / ``_ssl`` / ``ssl`` and ``filesystem``
    gained ``_sqlite3``. Both are covered by the existing wording -- the raw
    modules are the same surface as the wrappers already named -- and this
    pins that the wording was checked rather than assumed, the same way
    ``test_the_process_env_summary_admits_what_the_os_module_is`` pins the
    ``os`` framing.
    """
    network = tiers.CAPABILITY_SUMMARY["network"]
    assert "any host" in network
    assert "ssl" in network, "ssl joined the group; the consent line should say so"
    filesystem = tiers.CAPABILITY_SUMMARY["filesystem"]
    assert "sqlite3" in filesystem


def test_the_compression_accelerators_were_verified_rather_than_matched():
    """The one place this wave deliberately did NOT follow the wrapper's tier.

    ``bz2`` and ``lzma`` are gated by ``filesystem``, and core#215 recorded
    ``_bz2`` / ``_lzma`` as "a mechanical match to an already-blocked
    wrapper's own capability". Checking rather than matching says otherwise:
    ``dir(_bz2)`` is exactly ``{BZ2Compressor, BZ2Decompressor}`` and
    ``_lzma`` adds only filter constants -- neither takes a path, and the
    file access the wrappers are gated for lives in ``BZ2File`` /
    ``LZMAFile``, ordinary Python calling the already-unrestricted
    ``open()``. So they are accepted, in the same category as ``zlib``,
    which was already accepted on exactly this reasoning.

    This test exists because "same shape as something already blocked" is
    the reasoning that would have blocked them, and it would have been
    wrong. If a future CPython gives ``_bz2`` a path-taking entry point,
    this fails and the decision gets re-made.
    """
    import importlib.util

    for name in ("_bz2", "_lzma"):
        assert name in tiers.ACCEPTED_UNGATED_MODULES
        assert name not in dangerous_modules()
        if importlib.util.find_spec(name) is None:  # pragma: no cover
            continue
        module = importlib.import_module(name)
        public = {n for n in dir(module) if not n.startswith("_")}
        takers = {n for n in public if "file" in n.lower() or "open" in n.lower()}
        assert takers == set(), (
            f"{name} grew a file-facing entry point ({sorted(takers)}); the "
            f"accept in ACCEPTED_UNGATED_MODULES rests on it having none, so "
            f"re-decide rather than editing this assertion"
        )
    _tier0("import _bz2\nimport _lzma\n")


def test_nothing_is_left_deferred_in_the_accepted_set():
    """``ACCEPTED_UNGATED_MODULES`` used to hold two kinds of entry: real
    accepts, and known-dangerous names parked there with an issue number
    while somebody decided. The parked ones were indistinguishable from the
    accepts except by reading the reason string, which is how twenty-nine
    live gaps sat in a green suite.

    core#215 and core#216 emptied that second kind. This pins it: an entry
    here is a VERDICT, so no reason string may still be pointing at an issue
    as the place where the decision will be made.
    """
    parked = sorted(
        name for name, reason in tiers.ACCEPTED_UNGATED_MODULES.items()
        if "core#215" in reason or "core#216" in reason
    )
    assert parked == [], (
        f"{parked} are recorded as accepted but their reason still defers to "
        "an issue. Either the module is accepted -- say why, without the "
        "issue number doing the work -- or it belongs on the blocklist"
    )


def test_every_builtin_module_is_classified_as_blocked_or_accepted():
    """core#177's actual ask, not the narrower one above: a CLASSIFICATION,
    not one name.

    core#183 happened because ``nt`` / ``posix`` were never enumerated
    anywhere, in either direction -- nobody removed them from a list, they
    were simply never added, and the suite stayed green through it because
    nothing ever asked "is EVERY name accounted for". That question only has
    teeth on a BLOCKLIST, where the default is grant: an unclassified name
    is reachable with zero declaration until someone notices by hand, which
    is exactly what happened. It does not have the same teeth on an
    ALLOWLIST, where the default is deny -- see
    ``test_the_purity_guard_can_see_impurity_in_both_path_implementations``'s
    own docstring for why a "does every name fall into a pre-approved
    bucket" assertion over ``TIER0_PATH_HELPERS`` was rejected as pointless
    there. ``dangerous_modules()`` is the other shape, which is why this
    test exists and that one deliberately doesn't.

    Every name CPython actually compiled into this interpreter
    (``sys.builtin_module_names``) must be either on ``dangerous_modules()``
    or in ``tiers.ACCEPTED_UNGATED_MODULES``, with a reason recorded there.
    Landing in the accepted set is not a safety verdict by itself -- most of
    its entries are known-dangerous and simply not yet triaged (core#215) --
    it is a commitment that the gap is TRACKED, in source and in CI output,
    rather than silently passing. A future CPython that compiles in a new
    extension module nobody has classified yet fails this test by name,
    which is the mechanical catch #183 needed and the narrower test above,
    on its own, does not provide: CI (``backend-test.yml``) runs
    ``ubuntu-latest`` only, so ``os.name`` there is always ``'posix'`` --
    deleting ``"nt"`` from the blocklist alone stays green in CI and would
    only be caught by someone running the suite on Windows. This test
    catches that same deletion on ANY platform, because it asks about every
    compiled-in name at once, not the one platform happens to select.
    """
    import sys

    blocked = dangerous_modules()
    accepted = set(tiers.ACCEPTED_UNGATED_MODULES)
    unclassified = sorted(
        name for name in sys.builtin_module_names
        if name not in blocked and name not in accepted
    )
    assert unclassified == [], (
        "these builtin modules are neither on the blocklist nor recorded in "
        f"ACCEPTED_UNGATED_MODULES: {unclassified}. Classify each: add it to "
        "_DANGEROUS_MODULES (and, if it should be capability-unlockable, a "
        "CAPABILITY_MODULES group), or -- only if it is genuinely inert, the "
        "way _io is next to open() -- add it to ACCEPTED_UNGATED_MODULES "
        "with a one-line reason"
    )


def test_the_enumeration_is_not_vacuous():
    """Non-vacuity for the classification test above: prove it actually
    reacts to a name losing its classification, rather than trivially
    passing because the blocklist happens to be complete right now.

    Uses THIS platform's own native os-backing module (``os.name`` -- 'nt'
    on Windows, 'posix' everywhere CI runs) as the probe, the same technique
    ``test_the_platform_native_os_backing_module_is_always_on_the_blocklist``
    uses: it is guaranteed to be a real entry in ``sys.builtin_module_names``
    on whatever platform runs this, unlike hardcoding ``"nt"`` specifically,
    which is not even present in ``sys.builtin_module_names`` outside
    Windows -- so a probe that hardcoded it would prove nothing when run in
    this repo's Linux-only CI. Evaluates the classification logic against
    ``dangerous_modules() - {native}`` directly (no monkeypatching, no
    subprocess, no second interpreter needed): a pure function of local data
    that runs identically wherever the suite runs.
    """
    import os
    import sys

    native = os.name
    accepted = set(tiers.ACCEPTED_UNGATED_MODULES)
    assert native not in accepted, (
        f"test assumption broken: {native!r} is in ACCEPTED_UNGATED_MODULES, "
        f"so removing it from the blocklist would not surface as unclassified "
        f"and this probe would not prove anything"
    )
    blocked_without_native = dangerous_modules() - {native}
    unclassified = [
        name for name in sys.builtin_module_names
        if name not in blocked_without_native and name not in accepted
    ]
    assert native in unclassified, (
        f"removing {native!r} from the blocklist should have surfaced it as "
        f"unclassified; it did not, so the enumeration above would not "
        f"catch this platform's own os-backing module being dropped"
    )


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
# Screen 2 is an audit hook. It catches ZERO of today's impure ``os.path``
# names, because **``os.stat`` raises no audit event** -- the CPython audit
# table covers ``open``, ``os.listdir``, ``os.scandir``, ``os.chdir`` and
# friends, but not ``stat``, ``lstat`` or ``getcwd``. It is not decoration
# though, and this was measured rather than assumed: it is the ONLY screen
# that sees a helper reaching the disk without going through the ``os`` names
# screen 3 replaces. A hypothetical ``os.path`` helper built on ``io.open``
# or on ``pathlib.Path.read_text`` is invisible to screen 3 and shows up here
# as an ``open`` event.
#
# Screen 3 is the one that does the work: replace the ``os`` functions the
# path modules actually call -- plus ``os.environ`` and, on Windows, the
# private ``os.path._get*`` helpers -- and fail if the call touches any of
# them. That asks the property directly: does this helper consult the
# filesystem, the environment, or the working directory?
#
# Screen 4 (core#258) exists because screens 1-3 are all *Python-level*
# interception, and CPython 3.12 put four of these helpers out of Python's
# reach on Windows. ``Lib/ntpath.py`` opens with::
#
#     try:
#         from nt import _path_isdir as isdir
#         from nt import _path_isfile as isfile
#         from nt import _path_islink as islink
#         from nt import _path_exists as exists
#     except ImportError:
#
# so from 3.12 (3.14 for ``lexists``) those names are C builtins that call
# the Win32 API directly. They never touch ``os.stat``, and ``stat`` raises no
# audit event, so all three earlier screens go blind at once and report a
# ``stat()``-on-any-path helper as pure. ``nt`` does not exist on Linux, so the
# ``except ImportError`` always fires there and no amount of Python versions in
# an ubuntu-only CI matrix can surface it -- which is why this arrived together
# with the Windows job in ``backend-test.yml``.
#
# Screen 4 asks the one question no implementation detail can hide from: hold
# the input string fixed and vary the state of the filesystem underneath it. A
# string transform answers identically in every world; anything that consults
# the disk cannot. It is a pure ADDITION -- it can only ever find more, never
# excuse a helper the other screens caught.

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

#: Screen 4's worlds. Every basename is the SAME LENGTH and lowercase on
#: purpose: a pure helper's answer may mention the path it was handed, so the
#: comparison canonicalises that substring away before asking whether two
#: worlds disagreed. Equal-length, dot-free, case-stable names keep that
#: substitution exact under ``normcase``, ``splitext`` and ``splitdrive``.
_WORLD_NAMES = {"absent": "absent", "file": "wfilea", "dir": "wdirbb",
                "symlink": "wlinkc"}
_WORLD_TOKEN = "<world>"

#: ``{"root": Path, "paths": {world: str}, "symlink": bool}``, built once.
_WORLDS: dict[str, object] = {}


def _path_worlds() -> dict[str, object]:
    """Build (once) a directory holding one path per filesystem state.

    The four worlds differ ONLY in what exists at the probe path: nothing, a
    file, a directory, a symbolic link. Creating a symlink needs a privilege on
    Windows (Developer Mode, or an elevated process), so whether that world
    exists is reported rather than assumed -- see the precondition in
    ``test_the_purity_guard_can_see_impurity_in_both_path_implementations``
    for why ``islink`` specifically cannot be judged without it.
    """
    if _WORLDS:
        return _WORLDS
    import atexit
    import os
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="cdui-path-worlds-")
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    paths = {w: os.path.join(root, n) for w, n in _WORLD_NAMES.items()}
    target = os.path.join(root, "targetf")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("cdui")
    with open(paths["file"], "w", encoding="utf-8") as fh:
        fh.write("cdui")
    os.mkdir(paths["dir"])
    has_symlink = True
    try:
        os.symlink(target, paths["symlink"])
    except (OSError, NotImplementedError, AttributeError):
        # Unprivileged Windows without Developer Mode. Report it; do not
        # quietly drop the world, because one helper is only visible in it.
        has_symlink = False
        paths.pop("symlink")
    _WORLDS.update({"root": root, "paths": paths, "symlink": has_symlink,
                    "target": target})
    return _WORLDS


def _is_os_primitive(value: object) -> bool:
    """Is *value* a C builtin defined by the platform's raw OS module?

    ``nt`` / ``posix`` are what ``os`` itself is built on, so a callable that
    answers with one of them executes below Python entirely -- which is exactly
    the condition screens 1-3 cannot see through. Used only to decide whether a
    precondition is REQUIRED, never to decide whether a helper is impure:
    ``nt`` exports pure string transforms too (``_path_normpath`` is
    ``os.path.normpath`` on Windows from 3.12).
    """
    return getattr(value, "__module__", None) in ("nt", "posix")


def _world_args(name: str, path: str, other: str) -> "tuple | None":
    """Call shape for screen 4, with *path* in the position under test.

    ``None`` means "not path-shaped" -- ``sameopenfile`` takes file
    descriptors and ``samestat`` takes stat results, so varying a path
    underneath them measures nothing.
    """
    if name in ("sameopenfile", "samestat"):
        return None
    if name in ("commonpath", "commonprefix"):
        return ([path, other],)
    if name in ("samefile", "relpath"):
        return (path, other)
    if name == "join":
        return (path, "b")
    return (path,)


def _screen4_filesystem_differential(name: str, module) -> bool:
    """True when ``module.name``'s answer depends on the state of the disk.

    Runs OUTSIDE the interception window of screens 1-3 on purpose: this screen
    needs the real ``os.stat`` in place, because the whole point is to let the
    helper reach a real filesystem and then watch whether the filesystem's
    state shows up in the answer.
    """
    value = getattr(module, name, None)
    if not callable(value):
        return False
    worlds = _path_worlds()
    paths: dict = worlds["paths"]  # type: ignore[assignment]
    other: str = worlds["target"]  # type: ignore[assignment]
    seen: set[str] = set()
    for world, path in paths.items():
        args = _world_args(name, path, other)
        if args is None:
            return False
        try:
            answer = repr(value(*args))
        except Exception as exc:            # noqa: BLE001 - the class IS the answer
            answer = f"!{type(exc).__name__}"
        # Canonicalise the one substring a pure helper is entitled to echo,
        # so "it mentioned the path I gave it" is not mistaken for "it looked".
        seen.add(answer.replace(_WORLD_NAMES[world], _WORLD_TOKEN))
    return len(seen) > 1


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


def _probe_path_helper(name: str, module=None) -> set[str]:
    """Call ``<module>.<name>`` under all three screens; return what it touched.

    An empty set means: touched no intercepted ``os`` function, raised no
    audited event, and returned neither the probe environment variable's value
    nor the working directory.

    *module* defaults to the live ``os.path`` -- the one the gate's verdict
    actually has to be right about -- but the caller may pass ``ntpath`` or
    ``posixpath`` explicitly. Both are importable on both platforms, and
    checking both is what turns "correct on the machine that ran it" into
    "correct on either CI runner". That is not theoretical: the first version
    of this probe passed on Windows and failed on Linux, because
    ``posixpath.expandvars`` returns early when the path holds no ``$``.
    """
    import os
    import os.path

    if module is None:
        module = os.path
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

    value = getattr(module, name, None)
    if not callable(value):
        return set()

    saved_os = {k: getattr(os, k) for k in _OS_INTERCEPTS if hasattr(os, k)}
    saved_path = {
        k: getattr(module, k) for k in _OS_PATH_INTERCEPTS if hasattr(module, k)
    }
    real_environ = os.environ
    marker = "SHOULD-NEVER-APPEAR"
    os.environ["CDUI_PATH_PURITY_PROBE"] = marker
    cwd = os.getcwd()
    # Every shape has to be present at once, and each one was learned by the
    # guard missing something: the ``~`` must LEAD (``expanduser`` expands only
    # that position), ``%VAR%`` is what ``ntpath.expandvars`` reads, and
    # ``$VAR`` is what ``posixpath.expandvars`` reads -- the latter returns
    # early when the path holds no ``$``, which is how a probe that passed on
    # Windows let ``expandvars`` slip on Linux.
    args = _PATH_HELPER_ARGS.get(
        name, ("~/%CDUI_PATH_PURITY_PROBE%/$CDUI_PATH_PURITY_PROBE/x",)
    )

    _AUDIT_STATE["events"] = []
    try:
        for key in saved_os:
            setattr(os, key, _tracker(f"os.{key}"))
        for key in saved_path:
            setattr(module, key, _tracker(f"{module.__name__}.{key}"))
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
            setattr(module, key, original)
        real_environ.pop("CDUI_PATH_PURITY_PROBE", None)

    touched.extend(_AUDIT_STATE["events"])  # type: ignore[arg-type]
    if marker in result:
        touched.append("returned os.environ")
    if cwd and cwd in result:
        touched.append("returned the cwd")
    # Screen 4, deliberately after the `finally` above: it needs the REAL os
    # functions back before it can let the helper reach a real filesystem.
    if _screen4_filesystem_differential(name, module):
        touched.append("answer depends on the filesystem")
    return set(touched)


def test_no_allowlisted_path_helper_touches_the_environment_the_disk_or_the_cwd():
    """Every entry on the allowlist, verified by CALLING it under all three
    screens, in BOTH path implementations.

    The live ``os.path`` is the one the gate's verdict has to be right about
    today, but ``ntpath`` and ``posixpath`` are both importable everywhere and
    a plugin manifest is portable: a helper that is a pure string transform on
    one platform and reads the environment on the other would be a hole for
    every user on the other platform. Both are checked here so that is a test
    failure rather than a support ticket.
    """
    import ntpath
    import os.path
    import posixpath

    for module in (os.path, ntpath, posixpath):
        for name in tiers.TIER0_PATH_HELPERS:
            where = f"{module.__name__}.{name}"
            assert hasattr(module, name), f"{where} does not exist"
            value = getattr(module, name)
            if not callable(value):
                assert isinstance(value, (str, bool, type(None))), (
                    f"{where} is a {type(value).__name__}, not an inert constant"
                )
                continue
            touched = _probe_path_helper(name, module)
            assert touched == set(), f"{where} touched {sorted(touched)}"


def test_the_purity_guard_can_see_impurity_in_both_path_implementations():
    """The guard's own sensitivity, measured rather than asserted.

    A guard that inspects nothing passes trivially -- which is how the
    value-only version shipped while catching 3 of the 25 names it would have
    had to catch. So this drives the probe over every public name in BOTH
    ``ntpath`` and ``posixpath`` and requires that the ones known to touch the
    environment or the disk come back caught.

    **What this deliberately does NOT assert** is that every name falls into a
    pre-approved bucket. An earlier version did, and it was wrong twice over.
    It cost a CI round per CPython release -- 3.12 alone added ``splitroot``,
    ``isjunction`` and ``isdevdrive`` -- and it was enforcing something that is
    not a security property: ``TIER0_PATH_HELPERS`` is an ALLOWLIST, so a name
    nobody has heard of is refused by the gate whether it is pure or not. A new
    *impure* name is a non-event for the same reason, and the day somebody
    tries to allowlist one,
    ``test_no_allowlisted_path_helper_touches_the_environment_the_disk_or_the_cwd``
    is what refuses it.

    So the load-bearing claim is the floor below: these names, which really do
    read the environment or hit the disk, are visible to the guard on both
    implementations. If a future edit blunts a screen -- drops ``os.stat`` from
    the interception list, say -- this is the test that notices.

    core#258: it also noticed something no edit of ours caused. On Windows from
    CPython 3.12, ``exists`` / ``isdir`` / ``isfile`` / ``islink`` (and
    ``lexists`` from 3.14) are ``nt`` C builtins rather than ``genericpath``
    Python functions, and every screen this test had was Python-level
    interception. It failed here honestly -- the probe really had gone blind on
    the platform most of this project's users are on -- and stayed invisible to
    CI only because CI was ubuntu-only, where ``from nt import ...`` always
    raises ``ImportError``. Screen 4 restores the sensitivity; the Windows job
    in ``backend-test.yml`` is what keeps it honest.
    """
    import ntpath
    import os.path
    import posixpath
    import types

    #: Names whose impurity the guard MUST be able to see, per implementation.
    #: ``ntpath`` and ``posixpath`` reach the disk by different routes --
    #: ``_getfullpathname`` versus ``os.getcwd`` -- which is the whole reason
    #: both are driven here.
    must_be_caught = {
        "expandvars", "expanduser", "exists", "lexists", "isfile", "isdir",
        "islink", "ismount", "getsize", "getmtime", "getatime", "getctime",
        "samefile", "realpath", "abspath", "relpath",
    }

    # ``islink`` is the one name whose impurity NO other world can expose: it
    # answers False for a missing path, a file and a directory alike, so only a
    # real symbolic link separates it from a string transform. Where its
    # implementation is a raw OS primitive, screens 1-3 cannot see it either --
    # so if we also cannot build that world, the probe would under-report and
    # say nothing about it. Fail loudly instead of passing quietly.
    if not _path_worlds()["symlink"]:
        for impl in (os.path, ntpath, posixpath):
            assert not _is_os_primitive(getattr(impl, "islink", None)), (
                f"{impl.__name__}.islink is a C primitive from "
                f"{getattr(impl.islink, '__module__', '?')!r}, and this "
                "machine cannot create a symbolic link -- the probe has no "
                "way to tell it apart from a pure string helper. Enable "
                "Developer Mode (or run elevated) so the symlink world exists."
            )

    allowlisted = set(tiers.TIER0_PATH_HELPERS)
    for module in (os.path, ntpath, posixpath):
        where = module.__name__
        caught: set[str] = set()
        modules_seen: set[str] = set()
        for name in sorted(dir(module)):
            if name.startswith("_") or name in allowlisted:
                continue
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                modules_seen.add(name)      # the module screen owns these
            elif _probe_path_helper(name, module):
                caught.add(name)

        missed = {n for n in must_be_caught if hasattr(module, n)} - caught
        assert missed == set(), (
            f"[{where}] the guard cannot see that these touch the environment "
            f"or the disk: {sorted(missed)}. Either a screen was blunted, or "
            f"the probe no longer reaches them."
        )
        # Non-vacuity: a probe that inspects nothing would report nothing.
        assert len(caught) >= 12, f"[{where}] only caught {sorted(caught)}"
        assert {"os", "sys", "genericpath"} <= modules_seen, where


# ── core#258: does WHICH module implements a helper change the verdict? ────
#
# The security question the issue raises, stated precisely: both gates have a
# rule keyed on a module NAME, and ``nt`` is on the blocklist. From CPython
# 3.12 on Windows, ``os.path.exists`` IS an ``nt`` function. So the answer to
# "which module does this callable belong to" differs between the platform CI
# runs and the platform students run. Does a verdict differ with it?
#
# No -- and the two tests below are why, one per gate, rather than an argument
# in a PR body that nothing re-checks.


def test_every_os_path_implementation_module_is_blocked_so_a_cpython_move_is_a_no_op():
    """The structural reason the ``nt`` migration cannot change a verdict.

    ``script_proxy`` is the gate that reasons about objects: it takes a
    callable's own ``__module__`` (``_defining_root``) and refuses anything
    whose root is on ``_BLOCKED_DEFINING_ROOTS``. A CPython release that
    re-homes ``os.path.exists`` from ``genericpath`` to ``nt`` changes that
    root -- so the verdict is stable only if the set of roots an ``os.path``
    helper can possibly answer with is CLOSED under the blocklist.

    It is: ``ntpath``, ``posixpath``, ``genericpath``, ``nt`` and ``posix`` are
    all blocked, so the move is from one refused root to another refused root.
    Asserted over the live modules, so a future CPython that invents a sixth
    implementation module fails here instead of silently opening a door.
    """
    import ntpath
    import os.path
    import posixpath

    from app.core.script_proxy import _BLOCKED_DEFINING_ROOTS  # noqa: SLF001

    assert {"nt", "posix", "ntpath", "posixpath", "genericpath"} <= (
        _BLOCKED_DEFINING_ROOTS
    )

    for module in (os.path, ntpath, posixpath):
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            value = getattr(module, name)
            if not callable(value) or isinstance(value, type):
                continue
            root = (getattr(value, "__module__", "") or "").split(".")[0]
            assert root in _BLOCKED_DEFINING_ROOTS, (
                f"{module.__name__}.{name} is defined by {root!r}, which the "
                "runtime gate does NOT refuse -- a library re-exporting it "
                "under a harmless name would hand a script a path helper the "
                "policy has never classified."
            )


def test_the_install_time_verdict_does_not_depend_on_which_module_implements_it():
    """The other gate, which never asks the question at all -- on purpose.

    ``validate_python_source`` is a static AST walk: it screens the LEAF NAMES
    of ``from os.path import ...`` against ``TIER0_PATH_HELPERS``, and never
    imports, resolves or introspects the callable. So its verdict is the same
    string comparison on every platform and every version, whatever ``nt``
    holds today. Pinned with one accepted name and one refused name that
    CPython has actually moved, plus the observed attribution, so the test
    documents the platform split it is proving irrelevant.
    """
    import ntpath

    # ``normpath`` moved to ``nt`` on Windows 3.12+; ``exists`` did too.
    # Neither is a module and neither answer depends on that.
    assert ntpath.normpath.__module__ in ("nt", "ntpath")
    assert ntpath.exists.__module__ in ("nt", "genericpath")

    _tier0("from os.path import normpath\n")               # allowlisted: accepted
    assert "process-env" in _refusal("from os.path import exists\n")


def test_the_purity_probe_can_see_through_a_c_fast_path():
    """Screen 4's own sensitivity -- the guard for the guard's newest screen.

    Screens 1-3 are Python-level interception and are blind to a C builtin by
    construction. This asserts the property screen 4 was added for, using the
    live ``os.path``: a helper that consults the disk is caught even when the
    interception screens see nothing, and a helper that is a pure string
    transform is NOT caught even when it is the same kind of C builtin
    (``os.path.normpath`` is ``nt._path_normpath`` on Windows 3.12+, and it
    stays clean -- which is what stops screen 4 from being a rule that just
    says "implemented in C means dangerous").
    """
    import os.path

    assert _screen4_filesystem_differential("exists", os.path)
    assert _screen4_filesystem_differential("isdir", os.path)
    assert _screen4_filesystem_differential("isfile", os.path)
    for pure in ("normpath", "basename", "dirname", "join", "splitext",
                 "normcase", "isabs", "split", "splitdrive"):
        assert not _screen4_filesystem_differential(pure, os.path), pure


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
