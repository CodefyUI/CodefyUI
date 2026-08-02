"""The three-tier policy: what user Python may import, and on whose say-so.

CodefyUI runs Python that CodefyUI did not write, from three sources that
differ only in how much the author is trusted: code typed into the canvas
(:mod:`app.core.script_policy`), a ``.py`` file uploaded as a custom node, and
an installed plugin pack. Before core#133 there were two answers -- "blocked"
and "blocked unless the user typed ``--trust-author``" -- and the gap between
them was wide enough that the most natural community contributions could not
be written politely. A logger that POSTs a metric needed the same blanket
"trust me with everything" flag as a plugin that wanted ``ctypes``.

This module is the shared vocabulary for a middle answer. It holds DATA and
two-line helpers only, and imports nothing from the modules that use it, so
both :mod:`app.core.plugin_validator` (which owns the AST walker) and
:mod:`app.core.script_policy` (which owns the in-canvas policy) can import it
without a cycle -- and so can ``scripts/plugins.py``, the CLI that has to
print a capability request to a human before anything is installed.

The three tiers
---------------
**Tier 0 -- default allow.** No declaration, no flag. Pure-compute stdlib and
the numeric ecosystem: :data:`TIER0_PURE_COMPUTE_MODULES`. A plugin doing
arithmetic, statistics, tensors or JSON never touches this file's other
constants at all, which is the point: the common case should cost nothing.

**Tier 1 -- declared capabilities.** The plugin manifest asks, in the open::

    [security]
    capabilities = ["network"]

Each capability unlocks one module GROUP (:data:`CAPABILITY_MODULES`), and
``cdui plugin install`` prints the request and waits for a yes before a single
file is written. What is granted is recorded in the lockfile, so "which of my
plugins can reach the network" is a question with an answer.

**Tier 2 -- trusted.** ``--trust-author`` plus ``[security].allowed_modules``,
unchanged from before core#133. Anything at all, loudly, because at that point
the user has said they read the code.

What a capability is, and what it is not
----------------------------------------
A capability is a DECLARATION, not a sandbox. Three limits, stated here
rather than left to be discovered:

1. **The gate is per-file and name-keyed.** It sees the plugin's own
   ``import`` statements. It does not see what the imported library does, and
   it does not scan that library's source.
2. **The groups cover the blocklisted roots, not the category.**
   :data:`CAPABILITY_MODULES` is a partition of
   :func:`app.core.plugin_validator.dangerous_modules`, which is a list of
   stdlib roots plus ``requests``. A plugin that imports ``httpx`` reaches the
   network with no declaration at all, because ``httpx`` was never on the
   blocklist. Adding every HTTP client on PyPI is not a thing a list can do.
3. **A capability is not a permission boundary around an ACTION, only around
   an IMPORT.** Two consequences that surprise people, so they are written
   here and in the docs rather than left to be found:

   * ``filesystem`` does not gate writing files. Plain ``open(p, "w")`` is a
     builtin, needs no import, and passes at Tier 0 with nothing declared.
     What the capability gates is the file LIBRARIES. Gating ``open`` was
     considered and rejected: the mode is often computed
     (``open(p, "w" if x else "r")``), so the check would be evaded by one
     variable while breaking honest plugins -- a false positive with no
     matching security value.
   * ``network`` implies a file write, because
     ``urllib.request.urlretrieve(url, dest)`` is one call.

4. **Nothing here loosens Tier 0 for in-canvas scripts.** The script node's
   surface is an ALLOWLIST of dotted module paths
   (``script_policy.TIER0_MODULE_PATHS``), audited path by path across six
   adversarial rounds in core#131. This module's Tier-0 list is the policy
   STATEMENT -- "these roots are never denied to an installed file" -- and the
   script node deliberately exposes a subset of it. See
   ``test_tiered_policy.py`` for the invariants that keep the two honest.

So the promise is the same one the rest of this package makes: a guardrail
that raises the cost of a drive-by, and a declaration a user can read before
they consent. Not a boundary.
"""

from __future__ import annotations

from typing import Any, Iterable

#: Tier identifiers, used in messages and in the docs rather than in control
#: flow -- the walker branches on "is this capability granted", not on a tier
#: number. Kept as constants so the three names have one spelling.
TIER_DEFAULT = "default"
TIER_CAPABILITY = "capability"
TIER_TRUSTED = "trusted"

TIERS: tuple[str, ...] = (TIER_DEFAULT, TIER_CAPABILITY, TIER_TRUSTED)

#: Tier 0: roots that are pure computation and are therefore never denied to
#: an installed file, whatever else this policy grows.
#:
#: This is a STATEMENT, and it is enforced as one: a test asserts that no
#: entry here is ever added to
#: :func:`app.core.plugin_validator.dangerous_modules`, so a future hardening
#: pass cannot accidentally make ``import math`` a capability. That is the
#: whole job of the constant -- the plugin gate is a blocklist, so these roots
#: are already allowed by construction, and the list exists to keep them that
#: way.
#:
#: ``pandas`` is here for installed files only. It is emphatically NOT on the
#: in-canvas script node's allowlist: ``pandas.read_pickle`` executes the file
#: it reads, ``read_html`` fetches a URL, and admitting an API that size to an
#: unreviewed-code surface needs its own audit round, not a line in a tuple.
TIER0_PURE_COMPUTE_MODULES: tuple[str, ...] = (
    "collections",
    "dataclasses",
    "decimal",
    "enum",
    "functools",
    "itertools",
    "json",
    "math",
    "numpy",
    "pandas",
    "random",
    "re",
    "statistics",
    "torch",
    "typing",
)

#: The capabilities a manifest may declare, in the order they are printed.
CAPABILITIES: tuple[str, ...] = ("network", "filesystem", "process-env")

#: Capability -> the blocklisted module roots it unlocks.
#:
#: Every value here is on :func:`app.core.plugin_validator.dangerous_modules`;
#: a root that is NOT on that blocklist needs no capability, because the
#: plugin gate is a blocklist and never asked about it. A test asserts the
#: containment, so an entry added here without a matching blocklist entry
#: fails rather than quietly doing nothing.
#:
#: What is deliberately absent is as much of the design as what is present.
#: ``subprocess``, ``sys``, ``importlib``, ``ctypes``, ``pickle``, ``marshal``,
#: ``dill``, ``shelve``, ``runpy``, ``code``, ``codeop``, ``compileall``,
#: ``signal``, ``atexit``, ``webbrowser``, ``threading``, ``asyncio`` and
#: ``multiprocessing`` map to NO capability and stay Tier 2: no capability
#: unlocks a module whose PURPOSE is running code or reaching the interpreter.
#:
#: That is the real invariant, and it is narrower than "capabilities cannot
#: spawn a process", which is what this comment used to imply. ``process-env``
#: grants ``os``, and ``os`` spawns processes. The distinction worth keeping is
#: that no capability hands over a module built for executing code; the one
#: that hands over a general-purpose module says so in its summary.
CAPABILITY_MODULES: dict[str, tuple[str, ...]] = {
    "network": ("http", "requests", "socket", "urllib"),
    "filesystem": (
        "bz2",
        "codecs",
        "fileinput",
        "glob",
        "gzip",
        "lzma",
        "pathlib",
        "shutil",
        "sqlite3",
        "tarfile",
        "tempfile",
        "zipfile",
    ),
    # Named for the reason people ask for it -- reading ``os.environ`` -- but
    # what it GRANTS is the ``os`` module, and this comment exists because the
    # first version of it claimed otherwise. Under this capability alone,
    # ``os.execv``, ``os.startfile``, ``os.spawnve``, ``os.fork``,
    # ``os.remove``, ``os.open`` + ``os.write`` and ``os.environ[k] = v`` are
    # all reachable, and ``os.system`` / ``os.popen`` are refused only as
    # CALLS (``f = os.system; f(cmd)`` is one assignment away). The honest
    # framing is the one the summary and the docs now use: this is the os
    # module, process spawning included. Narrowing it to a genuine read-only
    # slice would mean a blocklist over ~300 functions, and a promise resting
    # on an incomplete blocklist is the failure mode core#131 spent six rounds
    # unlearning.
    "process-env": ("os", "ntpath", "posixpath", "genericpath"),
}

#: One line per capability, shown in the install prompt and quoted verbatim in
#: the validator's refusal. English; the CLI pairs each with a zh-TW string
#: through its own ``t(zh, en)`` helper, the same as every other CLI string.
#:
#: Written as what the user is agreeing to, not as what the plugin asked for:
#: somebody is about to answer y/N on the strength of this sentence.
CAPABILITY_SUMMARY: dict[str, str] = {
    "network": (
        "reach the network -- send and receive data from any host, and write "
        "what it downloads to disk (requests, urllib, http, socket)"
    ),
    "filesystem": (
        "use the file libraries -- pathlib, tempfile, shutil, zip/tar/gzip, "
        "sqlite3, glob. Note that plain open() needs no declaration at all"
    ),
    "process-env": (
        "use the whole os module -- read AND change this process's "
        "environment (any API keys in it included), start other programs, "
        "and delete or rename files"
    ),
}

#: The read-only-intent slice of ``os`` that Tier 0 keeps.
#:
#: core#133's brief asked for ``os.path`` at Tier 0, and that is worth having:
#: ``os.path.join`` / ``basename`` / ``splitext`` are string manipulation, and
#: telling an author to declare ``process-env`` to join two path components is
#: the kind of refusal that teaches people the policy is noise.
#:
#: It is granted through the IMPORT FORM rather than the module name, because
#: the two spellings differ in what they bind:
#:
#: * ``from os.path import join`` binds ``join`` -- a function;
#: * ``from os import path`` binds ``ntpath`` / ``posixpath`` -- the path
#:   module;
#: * ``import os.path`` binds ``os``. So does ``import os``. Both hand over
#:   the whole module, ``os.environ`` and ``os.remove`` included, and both
#:   stay behind ``process-env``.
#:
#: This grants nothing new in absolute terms -- ``import posixpath`` and
#: ``import ntpath`` were never on the blocklist and reach the same module
#: today -- so the exception is a usability fix rather than a widening. The
#: refusal for ``import os.path`` names the working form.
TIER0_PATH_MODULE: str = "os.path"
TIER0_PATH_ROOT: str = "os"

#: The ONLY names ``from os.path import ...`` may bind at Tier 0.
#:
#: An ALLOWLIST, after two rounds of getting this wrong in the same way. The
#: first cut allowed ``from os import path`` and screened leaves against the
#: blocklist by NAME; review showed that handed over the real ``os`` and
#: ``sys``. The second screened leaves by whether they ARE modules, which
#: closed that -- and review then pointed out the screen was still scoped to
#: the escape that had been demonstrated rather than to the property behind
#: it. ``os.path`` is a real module, and its surface is emphatically not
#: string manipulation::
#:
#:     expandvars("%WANDB_API_KEY%")   # returned a real secret, capabilities=[]
#:     expanduser("~")                 # the user's home directory
#:     getsize(p) / exists(p) / isfile(p)   # a real stat() on any path
#:
#: All of those are functions, not modules, so a module screen could never
#: see them -- and ``expandvars`` reads the very thing ``process-env``'s
#: consent line promises to gate.
#:
#: So the rule is inverted: these names, and nothing else. Every entry was
#: verified against the LIVE ``os.path`` rather than assumed -- called with a
#: path that does not exist, containing an unexpanded ``%VAR%``, ``$VAR`` and
#: ``~``, and checked to return neither the variable's value nor the working
#: directory.
#:
#: Deliberately absent, and each for a checked reason:
#:
#: * ``expandvars`` / ``expanduser`` -- read ``os.environ``;
#: * ``exists`` / ``lexists`` / ``isfile`` / ``isdir`` / ``islink`` /
#:   ``ismount`` / ``getsize`` / ``getmtime`` / ``getatime`` / ``getctime`` /
#:   ``samefile`` -- call ``stat()`` on a real path;
#: * ``abspath`` / ``realpath`` / ``relpath`` -- resolve against the working
#:   directory, so they disclose where CodefyUI is installed. ``abspath`` is
#:   worth naming twice: a source audit for ``os.`` usage said it was pure,
#:   because on Windows it reaches ``nt._getfullpathname`` through a name the
#:   audit was not looking for. That is why these were verified by CALLING
#:   them, not by reading them.
#:
#: A plugin that needs any of those is asking for the filesystem or the
#: environment, which is what the capabilities are for.
TIER0_PATH_HELPERS: tuple[str, ...] = (
    # pure string transforms
    "basename",
    "commonpath",
    "commonprefix",
    "dirname",
    "isabs",
    "join",
    "normcase",
    "normpath",
    "split",
    "splitdrive",
    "splitext",
    # module-level string constants
    "altsep",
    "curdir",
    "defpath",
    "extsep",
    "pardir",
    "pathsep",
    "sep",
)


def capability_for_module(root: str) -> str | None:
    """Which capability unlocks *root*, or ``None`` if none does.

    ``None`` means two different things and the caller has to say which:
    either the root is not blocked at all (Tier 0 already allows it), or it is
    blocked and no capability will ever unlock it (Tier 2 only). The walker
    checks the blocklist first, so by the time it asks this question the
    second reading is the only one left.
    """
    for capability, roots in CAPABILITY_MODULES.items():
        if root in roots:
            return capability
    return None


def modules_for_capability(capability: str) -> tuple[str, ...]:
    """The module roots *capability* unlocks; empty for an unknown name."""
    return CAPABILITY_MODULES.get(capability, ())


def normalize_capabilities(raw: Any) -> tuple[str, ...]:
    """Clean a manifest's ``capabilities`` value into a sorted, unique tuple.

    Deliberately forgiving about SHAPE and strict about CONTENT: a manifest is
    written by hand, so ``["Network", " filesystem "]`` should mean what it
    obviously means, while a non-string entry is dropped rather than crashing
    an install. Unknown names survive normalisation on purpose --
    :func:`unknown_capabilities` is what reports them, and it can only do that
    if they are still here.

    A bare string is accepted as a one-element list, because
    ``capabilities = "network"`` is the typo everyone makes at least once and
    silently reading it as ``{"n","e","t",...}`` would be worse than either
    accepting it or refusing it.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: Iterable[Any] = [raw]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = raw
    else:
        return ()
    cleaned = {
        item.strip().lower()
        for item in items
        if isinstance(item, str) and item.strip()
    }
    return tuple(sorted(cleaned))


def unknown_capabilities(raw: Any) -> tuple[str, ...]:
    """The declared names this build does not recognise.

    An unknown capability is an error at install time rather than a silent
    no-op: a manifest asking for ``"gpu"`` is either a typo or was written
    against a newer CodefyUI, and both deserve to be said out loud instead of
    being granted nothing and failing later at the import.
    """
    return tuple(
        name for name in normalize_capabilities(raw) if name not in CAPABILITIES
    )


def granted_modules(capabilities: Iterable[str] | None) -> frozenset[str]:
    """Every module root the given capabilities unlock, as one set.

    *capabilities* goes to :func:`normalize_capabilities` untouched. Wrapping
    it in ``list()`` first would have re-created the exact bug that function
    exists to prevent: ``list("network")`` is ``['n','e','t',...]``, which is
    ``frozenset("os")`` wearing a different hat.
    """
    roots: set[str] = set()
    for capability in normalize_capabilities(capabilities):
        roots.update(modules_for_capability(capability))
    return frozenset(roots)


def describe_capability(capability: str) -> str:
    """One-line English summary, or a neutral fallback for an unknown name."""
    return CAPABILITY_SUMMARY.get(
        capability, f"an unrecognised capability ({capability!r})"
    )
