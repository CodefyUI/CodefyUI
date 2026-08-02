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


def test_capabilities_never_grant_the_ability_to_run_code():
    """The line the three capability names are drawn on.

    A capability grants a RESOURCE. Executing other code, reaching into the
    interpreter, or starting a process is not a resource, and is Tier 2 only
    however sympathetic the use case.
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
    _tier0("from os import path\n")
    _tier0("from os.path import join as j\n")


def test_tier0_still_refuses_the_forms_that_bind_the_whole_os_module():
    for line in ("import os", "import os.path", "import os.path as p"):
        message = _refusal(f"{line}\n")
        assert "process-env" in message, line


def test_tier0_refuses_pulling_anything_but_path_out_of_os():
    message = _refusal("from os import path, environ\n")
    assert "process-env" in message


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
