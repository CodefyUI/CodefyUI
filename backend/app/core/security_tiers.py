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
        # core#177 CI round 2 -- readline.add_history(s) +
        # .write_history_file(path) is an attacker-directed write of
        # attacker-directed content, verified directly. See
        # plugin_validator._DANGEROUS_MODULES for the full writeup.
        "readline",
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
    #
    # core#183 -- ``nt`` / ``posix`` join the group for the same reason
    # ``ntpath`` / ``posixpath`` already had: not a different surface, only a
    # different name for the same one. ``os`` is built directly on top of
    # whichever of the two the platform provides (CPython's own ``os.py``
    # does ``from nt import *`` / ``from posix import *``), so importing the
    # raw module by name reaches ``.environ`` / ``.remove`` / ``.system`` --
    # everything this capability already grants through ``os`` -- and until
    # this line did so with no declaration at all.
    "process-env": ("os", "ntpath", "posixpath", "genericpath", "nt", "posix"),
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

#: Names in ``sys.builtin_module_names`` that are deliberately NOT on
#: :func:`app.core.plugin_validator.dangerous_modules`, each with a reason.
#:
#: core#183 happened because ``nt`` / ``posix`` were never enumerated
#: anywhere, in either direction: not blocked, not capability-mapped, not
#: declared safe. Nobody removed them from a list -- they were simply never
#: added, and nothing ever asked "is EVERY compiled-in module accounted
#: for" to notice. core#177's standing check asks exactly that: every name
#: in ``sys.builtin_module_names`` must be either on the blocklist or listed
#: HERE. An entry in this dict is not automatically a safety verdict -- most
#: of the entries below ARE closed, verified decisions, but the ones tagged
#: with an issue number are known-dangerous and simply not yet triaged.
#: Either way, being here is a commitment that the module is ACCOUNTED FOR,
#: in source and in CI output, rather than silently passing a green suite
#: the way ``nt``/``posix`` did.
#:
#: **This set is a UNION across platforms and CPython versions, not a
#: snapshot of one interpreter.** ``sys.builtin_module_names`` differs by
#: both: Windows compiles in ``nt`` / ``winreg`` / ``msvcrt`` / ``_winapi``
#: and never ``posix`` / ``pwd`` / ``grp`` / ``spwd`` / ``fcntl`` / ...; the
#: reverse holds on Linux; and even within Linux, 3.10/3.11 lack
#: ``_xxinterpchannels`` (3.12+ only) while 3.12 merges ``_sha256`` /
#: ``_sha512`` into a single ``_sha2``. core#177's CI round-2 failure was
#: exactly this: a dict built and tested on one interpreter (Windows) went
#: green there and red on all three of CI's Linux runners, 19-20 names at
#: once. Adding an entry here is a commitment about that MODULE, not about
#: any one build -- expect this dict to keep growing as it meets
#: interpreters it hasn't met yet, and do not assume an entry present here
#: implies the module exists on whatever platform is running the suite.
#:
#: That asymmetry is safe by construction, not merely hoped: the enumeration
#: test below (:func:`test_every_builtin_module_is_classified_as_blocked_or_accepted`
#: in ``test_tiered_policy.py``) walks ``sys.builtin_module_names`` -- the
#: CURRENT interpreter's real list -- and asks only whether each of THOSE
#: names is classified. It never asks the reverse (whether every entry in
#: this dict exists on the current interpreter), so a Windows-only entry
#: sitting unused while the suite runs on Linux is inert, not an error --
#: verified directly, not merely reasoned about, by re-reading the
#: assertion's comprehension before extending this dict for CI round 2.
#:
#: Four shapes below, not three:
#:
#: 1. **Closed: the wrapper is already unrestricted, so the accelerator
#:    grants nothing new.** ``_io`` (behind plain ``open()``, which
#:    ``filesystem``'s own docstring explains was deliberately left
#:    ungated) and ``_sre`` (behind ``re``, already in
#:    :data:`TIER0_PURE_COMPUTE_MODULES` with zero declaration).
#: 2. **Closed: pure computation, verified by import and inspection, not
#:    assumed.** Every remaining stdlib C accelerator below: data
#:    structures, numeric/hashing/encoding transforms, introspection that
#:    grants no NEW capability beyond what is already open. Each was
#:    imported and its public surface read before being listed here --
#:    see core#177's own review for the specific check that caught
#:    ``__pycache__`` being asserted safe on the strength of an argument
#:    that was never actually tried.
#: 3. **Deferred: raw C-implementation module sitting directly behind an
#:    already-blocklisted or capability-gated wrapper.**
#:    :data:`app.core.script_proxy._BLOCKED_IMPLEMENTATION_ROOTS` (the
#:    runtime boundary for in-canvas scripts) already recognises every one
#:    of them as dangerous; that recognition was simply never ported to the
#:    install-time blocklist (core#215). Not resolved here even where the
#:    wrapper's own capability is a one-line mechanical match (``_bz2``
#:    behind ``bz2``, already ``filesystem``): CI round 2 kept the
#:    you-decide-the-check / maintainer-decides-the-blocklist split rather
#:    than treating "structurally the same shape as something already
#:    deferred" as licence to resolve five more of them unilaterally.
#: 4. **Deferred: a whole surface nobody had considered, not a hidden
#:    implementation behind something already blocked.** ``winreg`` (full
#:    Windows registry read/write/delete), ``_xxsubinterpreters``
#:    (``run_string(id, script, shared)`` executes an arbitrary string in a
#:    sub-interpreter -- an ``exec()``-equivalent primitive under a name
#:    ``_DANGEROUS_NAMES`` never looks for), ``pwd`` / ``grp`` (POSIX user /
#:    group database reads) and more added over two CI rounds -- tracked
#:    separately (core#216).
#:
#: Two names broke shape 3/4's "defer, don't decide" pattern in CI round 2,
#: each independently verified rather than assumed dangerous or assumed
#: safe: ``readline`` (an attacker-directed file WRITE, proven) and ``spwd``
#: (a shadow-password-database READ that bypasses its own file's permission
#: bits via NSS, proven) are both on the blocklist outright -- see
#: :data:`app.core.plugin_validator._DANGEROUS_MODULES` for the writeup on
#: each.
ACCEPTED_UNGATED_MODULES: dict[str, str] = {
    # ── shape 1: the wrapper is already unrestricted or already Tier 0 ──
    "_io": (
        "io / open() are unrestricted for installed files by design; "
        "gating the C accelerator under it grants nothing io doesn't "
        "already"
    ),
    "_sre": (
        "the regex engine behind re, already in TIER0_PURE_COMPUTE_MODULES "
        "with zero declaration; gating the accelerator grants nothing re "
        "doesn't already"
    ),
    # ── shape 2: pure computation / data structures, no I/O of any kind ──
    "_abc": "ABC machinery (isinstance/issubclass bookkeeping); no I/O",
    "_ast": "AST node types for the ast module; a data structure, not a compiler",
    "_bisect": "binary search over an in-memory sequence; no I/O",
    "_collections": "deque/OrderedDict/etc. C implementations; no I/O",
    "_contextvars": "context-local variable storage; no I/O",
    "_functools": "reduce/partial/lru_cache C implementations; no I/O",
    "_heapq": "heap-queue algorithm over an in-memory list; no I/O",
    "_operator": "operator.add/itemgetter/etc. C implementations; no I/O",
    "_random": "Mersenne Twister PRNG; reads no external entropy source directly",
    "_stat": "S_IS*/S_IF* bit-interpretation constants; does not itself call stat()",
    "_statistics": "mean/variance/etc. over in-memory data; no I/O",
    "_string": "str.format() helper C implementation; no I/O",
    "_struct": "binary pack/unpack over in-memory bytes; no I/O",
    "_symtable": "compiler symbol-table introspection (the `symtable` module); "
                 "reads a code object already in hand, does not execute one",
    "_tokenize": "C tokenizer backend for the `tokenize` module; text in, tokens out",
    "_typing": "typing module's runtime support (Generic, TypeVar, ...); no I/O",
    "_warnings": "warnings.warn() plumbing; writes to stderr the same as print()",
    "_weakref": "weak reference machinery; no I/O",
    "array": "fixed-type in-memory array; no I/O",
    "audioop": "raw audio sample transforms over in-memory bytes; no I/O "
               "(deprecated upstream, kept ungated for the same reason)",
    "binascii": "hex/base64/etc. transforms over in-memory bytes; no I/O",
    "builtins": "the builtins module itself; its dangerous NAMES (eval, exec, "
                "open, ...) are refused elsewhere by this walker whatever "
                "module they are reached through",
    "cmath": "complex-number math; no I/O",
    "errno": "errno constant definitions; no I/O",
    "gc": "garbage-collector controls (enable/disable/collect); can affect "
          "process performance, not file/network/process/environment access",
    "itertools": "already in TIER0_PURE_COMPUTE_MODULES with zero declaration",
    "math": "already in TIER0_PURE_COMPUTE_MODULES with zero declaration",
    "time": "clock reads and sleep; not a secret, same category as datetime.now()",
    "xxsubtype": "CPython's own C-extension example/test module; no I/O",
    "zlib": "compress/decompress over in-memory bytes; no I/O",
    # hashing: pure, deterministic transforms of in-memory bytes
    "_blake2": "BLAKE2 hash; pure transform of in-memory bytes",
    "_md5": "MD5 hash; pure transform of in-memory bytes",
    "_sha1": "SHA-1 hash; pure transform of in-memory bytes",
    "_sha256": "SHA-256 hash; pure transform of in-memory bytes",
    "_sha3": "SHA-3 hash; pure transform of in-memory bytes",
    "_sha512": "SHA-512 hash; pure transform of in-memory bytes",
    "_sha2": "SHA-2 family hash (some CPython versions merge _sha256/_sha512 "
             "into this one name); pure transform of in-memory bytes",
    # XML parsing: pure transform of an already-held string/bytes, and
    # verified (not assumed) that the default parser does not resolve
    # external entities -- `ET.fromstring()` against a DOCTYPE with a
    # `SYSTEM "file://..."` entity raised `ParseError: undefined entity`
    # rather than disclosing the file, checked directly against a real
    # interpreter rather than relied on from memory.
    "_elementtree": "C accelerator for xml.etree.ElementTree; default parser "
                     "verified NOT to resolve external entities (no XXE)",
    "pyexpat": "the same expat parser xml.etree.ElementTree is built on; "
               "same verified no-XXE behaviour under the default parser",
    # text/data encoding: pure transforms, not file or network access
    "_codecs": "codec registry lookup; the transform itself, not a file read",
    "_codecs_cn": "CJK codec tables; pure transform",
    "_codecs_hk": "CJK codec tables; pure transform",
    "_codecs_iso2022": "CJK codec tables; pure transform",
    "_codecs_jp": "CJK codec tables; pure transform",
    "_codecs_kr": "CJK codec tables; pure transform",
    "_codecs_tw": "CJK codec tables; pure transform",
    "_multibytecodec": "CJK multibyte codec engine; pure transform",
    "_csv": "CSV parse/format over an already-held string or file object; "
            "does not itself open a path",
    "_json": "JSON parse/format over an already-held string; no I/O",
    "_datetime": "C-accelerated datetime; reads the system clock only, not "
                 "itself blocklisted, not a secret-bearing environment read",
    "_locale": "locale NAME lookup/formatting; not process environment or a secret",
    # introspection / debugging: grant no capability beyond what is already open
    "_lsprof": "cProfile's C backend; timing/call-count introspection, no "
               "file/network/process access of its own",
    "_opcode": "bytecode opcode metadata (stack_effect etc.) used by dis/"
               "compiler internals; does not assemble or execute a code object",
    "_tracemalloc": "memory-allocation tracing for debugging; introspection only",
    "faulthandler": "crash-traceback dumping; needs an already-open file object "
                     "to write anywhere, which requires open() (already "
                     "unrestricted) -- grants no new capability",
    "mmap": "memory-maps an existing fd (already reachable via unrestricted "
            "open()) or an anonymous in-memory buffer; grants no new "
            "capability beyond what open() already does",
    "fcntl": "POSIX: operates on an already-open fd (locking, flags); does "
             "not itself open a path -- same reasoning as mmap",
    "select": "POSIX/cross-platform: waits on already-open fds/sockets for "
              "readiness; does not itself create a new fd or socket",
    "unicodedata": "Unicode character database lookups (category, "
                   "normalization); pure data lookup, no I/O",
    "_types": "types module runtime support (GenericAlias etc.); no I/O",
    "_sysconfig": "build-time configuration variables (compiler flags, "
                  "install paths); comparable disclosure to abspath() "
                  "revealing where CodefyUI is installed, already accepted "
                  "as low-severity in the os.path exception's own docs",
    "_suggestions": "typo-suggestion machinery behind error messages "
                    "('did you mean ...'); pure introspection, no I/O",
    # core#177 CI round 2 (Linux-only names, surfaced only once the
    # enumeration test met a real Linux interpreter):
    "_decimal": "the C accelerator behind decimal, already in "
                "TIER0_PURE_COMPUTE_MODULES with zero declaration; grants "
                "nothing decimal doesn't already",
    "_hashlib": "OpenSSL-backed hash dispatcher (hashlib.new(...)); pure "
                "transform of in-memory bytes, same category as the "
                "individual _md5/_sha1/etc. accelerators above",
    "_queue": "thread-safe FIFO held entirely in memory; no I/O of its own, "
              "and needs _thread (already tracked, core#215) to do anything "
              "concurrent -- inert on its own",
    "_uuid": "UUID generation; uuid1() reads the network interface's MAC "
             "address for uniqueness, a minor hardware-identifier read, not "
             "a file/network/process/environment capability",
    "_zoneinfo": "reads the IANA timezone database (e.g. "
                 "/usr/share/zoneinfo); a public, non-secret, system-standard "
                 "dataset, not user data -- same category as _locale above",
    "_testinternalcapi": "CPython's own internal-test-only C API, not a "
                         "stable or supported surface for third-party code; "
                         "test scaffolding, not shipped functionality",
    "_xxtestfuzz": "CPython's own fuzz-testing scaffold module; test "
                   "infrastructure, not shipped functionality",
    # ── shape 3: real capability, verified, deferred to the maintainer ──
    "pwd": "POSIX: reads the password database (usernames, UIDs, home "
           "directories, login shells) for every account on the machine "
           "-- core#216",
    "grp": "POSIX: reads the group database (group names, GIDs, membership) "
           "-- core#216",
    "syslog": "POSIX: writes to the system log, outside normal file I/O and "
              "not covered by any existing capability -- core#216",
    "_signal": "raw process-control primitives (raise_signal, set_wakeup_fd, "
               "signal handler registration) behind signal, itself already "
               "Tier-2-only -- core#215",
    "_thread": "raw os-thread spawn behind threading (Tier-2-only today) -- core#215",
    "_socket": 'raw socket behind socket (gated by "network" today) -- core#215',
    "_ssl": "raw TLS behind ssl (not on the blocklist at all today) -- core#215",
    "_ctypes": "raw native-call bridge behind ctypes (Tier-2-only today) -- core#215",
    "_pickle": "raw deserialize-executes-code behind pickle (Tier-2-only today) -- core#215",
    "_winapi": (
        "Windows: raw CreateProcess and friends, ungated by name today -- core#215"
    ),
    "msvcrt": "Windows C runtime bindings, console/file-locking primitives -- core#215",
    "_posixsubprocess": (
        "the fork_exec primitive behind subprocess (Tier-2-only today) -- core#215"
    ),
    "_multiprocessing": (
        "raw multiprocessing C helpers behind multiprocessing "
        "(Tier-2-only today) -- core#215"
    ),
    "_imp": "raw import machinery, lower-level than importlib (Tier-2-only today) -- core#215",
    "_frozen_importlib": "CPython's own bootstrap import machinery -- core#215",
    "_frozen_importlib_external": "same as _frozen_importlib -- core#215",
    # core#177 CI round 2 -- five more of shape 3, each a mechanical match to
    # an already-blocked wrapper's own capability, deliberately NOT resolved
    # here (see the header: "structurally the same shape" was not treated as
    # licence to decide five more of these unilaterally).
    "_asyncio": "raw C accelerator behind asyncio, itself already "
                "Tier-2-only (no capability) -- core#215",
    "_bz2": 'raw C accelerator behind bz2, itself already capability-gated '
            '("filesystem") -- core#215',
    "_lzma": 'raw C accelerator behind lzma, itself already capability-gated '
             '("filesystem") -- core#215',
    "_posixshmem": (
        "raw POSIX shared-memory primitive (shm_open/shm_unlink) behind "
        "multiprocessing.shared_memory; multiprocessing is already "
        "Tier-2-only. Verified directly reachable WITHOUT ever importing "
        "multiprocessing at all -- created a real /dev/shm object through "
        "_posixshmem alone -- core#215"
    ),
    "_sqlite3": (
        'raw C accelerator behind sqlite3, itself already capability-gated '
        '("filesystem", and that module\'s own blocklist comment already '
        "names extension-loading as part of the reason). Worth the "
        "maintainer's attention regardless: verified directly that "
        "Connection.enable_load_extension(True) + .load_extension(path) "
        "loads an arbitrary shared library into the process -- a "
        "ctypes-equivalent native-code primitive, which may mean "
        "\"filesystem\" already under-describes what sqlite3/_sqlite3 grant, "
        "not only that _sqlite3 needs the same tag its wrapper has -- "
        "core#215"
    ),
    "winreg": "Windows: full registry read/write/delete -- persistence and "
              "data-exfiltration primitive, no existing capability covers "
              "it -- core#216",
    "_xxsubinterpreters": (
        "run_string(id, script, shared) executes an arbitrary string in a "
        "sub-interpreter -- an exec()-equivalent primitive under a name "
        "_DANGEROUS_NAMES never looks for -- core#216"
    ),
    # core#177 CI round 2 -- six more of shape 4 (standalone surfaces, no
    # already-blocked wrapper to match):
    "_curses": "terminal UI manipulation (raw mode, screen control) on a "
               "REAL controlling tty if one is attached (e.g. `cdui start` "
               "run in a terminal); not a file/network/process/environment "
               "capability -- a display/input-takeover primitive scoped to "
               "whatever tty the process happens to have -- core#216",
    "_curses_panel": "curses' window-stacking extension; adds no capability "
                     "beyond _curses itself -- core#216",
    "ossaudiodev": "opens OSS audio device files (e.g. /dev/dsp) if present; "
                   "a real, if obsolete and typically absent on a modern "
                   "Linux/CI host, audio-capture primitive -- core#216",
    "resource": "reads/sets the CURRENT process's own resource limits "
                "(open files, CPU time, memory, ...); verified an "
                "unprivileged process cannot raise a hard limit past its "
                "existing ceiling (ValueError: not allowed to raise maximum "
                "limit), so this is self-DoS-shaped, not privilege "
                "escalation -- and CPU/memory DoS against the host process "
                "is already explicitly out of this gate's stated scope (see "
                "script_policy's own \"nothing here contains CPU or memory "
                "either\") -- core#216",
    "termios": "terminal I/O mode control (raw/cooked, echo) on a REAL "
               "controlling tty if one is attached; same scope and severity "
               "class as _curses above, tracked together -- core#216",
    "_xxinterpchannels": "3.12+ only: channel-based communication between "
                         "sub-interpreters, the companion module to "
                         "_xxsubinterpreters above; does not itself execute "
                         "code but is part of the same PEP 554 family -- "
                         "core#216",
}


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
