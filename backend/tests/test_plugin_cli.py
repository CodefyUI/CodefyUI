"""Unit tests for scripts/plugins.py — source parsing, manifest validation, lockfile ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import pytest

import plugins as plugin_cli
from app.core import plugin_loader
from app.core.plugin_validator import PluginValidationError


# ── parse_source ───────────────────────────────────────────────────────────

def test_parse_source_catalog_case_insensitive():
    # 'foundations' / 'deep' / 'rl' live in plugins/registry.json
    assert plugin_cli.parse_source("foundations") == ("catalog", "foundations", "", "")
    assert plugin_cli.parse_source("Foundations") == ("catalog", "foundations", "", "")
    assert plugin_cli.parse_source("RL") == ("catalog", "rl", "", "")


def test_parse_source_github_short_no_ref():
    assert plugin_cli.parse_source("alice/extras") == ("github", "alice", "extras", "")


def test_parse_source_github_short_with_ref():
    assert plugin_cli.parse_source("alice/extras@v1.2.3") == (
        "github", "alice", "extras", "v1.2.3",
    )


def test_parse_source_github_full_url():
    assert plugin_cli.parse_source("https://github.com/alice/extras") == (
        "github", "alice", "extras", "",
    )
    assert plugin_cli.parse_source("https://github.com/alice/extras.git") == (
        "github", "alice", "extras", "",
    )
    assert plugin_cli.parse_source("http://www.github.com/alice/extras") == (
        "github", "alice", "extras", "",
    )


def test_parse_source_rejects_garbage():
    with pytest.raises(ValueError):
        plugin_cli.parse_source("not a valid source spec")


# ── validate_manifest ──────────────────────────────────────────────────────

def _good_manifest(plugin_id: str = "test-pack") -> dict:
    return {"plugin": {"id": plugin_id, "name": "Test", "version": "0.0.1", "schema_version": 1}}


def test_validate_manifest_accepts_well_formed():
    plugin_cli.validate_manifest(_good_manifest())


def test_validate_manifest_rejects_missing_plugin_table():
    with pytest.raises(ValueError, match="\\[plugin\\]"):
        plugin_cli.validate_manifest({"content": {"nodes_dir": "nodes"}})


def test_validate_manifest_rejects_unknown_schema():
    bad = _good_manifest()
    bad["plugin"]["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        plugin_cli.validate_manifest(bad)


def test_validate_manifest_rejects_uppercase_id():
    bad = _good_manifest("CapsId")
    with pytest.raises(ValueError, match="Invalid plugin id"):
        plugin_cli.validate_manifest(bad)


def test_validate_manifest_rejects_trailing_dash_id():
    bad = _good_manifest("trailing-")
    with pytest.raises(ValueError, match="Invalid plugin id"):
        plugin_cli.validate_manifest(bad)


# ── validate_nodes_dir uses the AST validator on every .py ────────────────

def test_validate_nodes_dir_passes_clean_code(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (nodes / "ok.py").write_text(
        "from app.core.node_base import BaseNode\n"
        "class X(BaseNode):\n"
        "    NODE_NAME = 'X'\n"
        "    CATEGORY = 'Test'\n"
        "    DESCRIPTION = ''\n",
        encoding="utf-8",
    )
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_nodes_dir_rejects_dangerous_import(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "bad.py").write_text("import os\nos.system('whoami')\n", encoding="utf-8")
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_nodes_dir_honours_allowed_modules(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "ok_with_pathlib.py").write_text(
        "from pathlib import Path\np = Path('/tmp')\n", encoding="utf-8"
    )
    # pathlib is in the default blocklist; allow_modules opens it back up.
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=["pathlib"])


def test_validate_nodes_dir_allows_getattr_with_literal(tmp_path):
    """`getattr(obj, "literal")` is the common idiom for optional attrs —
    refining the AST gate so plugins that just want `getattr(context,
    "verbose", False)` aren't false-positived."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "verbose_check.py").write_text(
        "def f(context):\n"
        "    return getattr(context, 'verbose', False)\n",
        encoding="utf-8",
    )
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_nodes_dir_rejects_dynamic_getattr(tmp_path):
    """Dynamic attribute names — the actual sandbox-bypass shape — stay blocked."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "bad.py").write_text(
        "def f(name):\n"
        "    return getattr(__builtins__, name)\n",
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


# ── core#179: a library VALUE's own method escapes import-time gating ──────

def test_validate_nodes_dir_rejects_numpy_dump_to_an_arbitrary_path(tmp_path):
    """``numpy.zeros(3).dump(path)`` pickles straight to any path the file
    names, with substantially attacker-chosen content
    (``numpy.frombuffer(payload, dtype=numpy.uint8).dump(p)`` embeds the
    payload verbatim after a fixed pickle prefix) -- arbitrary file WRITE,
    zero capability declared.

    ``numpy`` is Tier 0 (``TIER0_PURE_COMPUTE_MODULES``), so ``import numpy``
    needs no declaration at all, and the import-time capability gate only
    ever looks at module names on ``Import`` nodes -- it never inspects what
    a Tier-0-allowed library hands back. ``.dump`` is already closed for
    in-canvas scripts via ``script_policy.TIER0_DENIED_ATTRS`` passed as
    ``denied_attributes``; this proves the same mechanism is wired into the
    installed-plugin surface, not just re-implemented for one leaf.
    """
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "bad.py").write_text(
        "import numpy\n"
        "def pwn(path):\n"
        "    numpy.zeros(3).dump(path)\n",
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])


def test_validate_plugin_dir_rejects_numpy_dump_to_an_arbitrary_path(tmp_path):
    """Same escape, the other entry point: ``validate_plugin_dir`` walks the
    whole extracted tarball (not only ``nodes/``) and has to close the same
    door -- it is a separate call to ``validate_python_source`` and does not
    inherit whatever ``validate_nodes_dir`` was told."""
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "bad.py").write_text(
        "import numpy\n"
        "def pwn(path):\n"
        "    numpy.zeros(3).dump(path)\n",
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_plugin_dir(root, [])


@pytest.mark.parametrize(
    "label,source",
    [
        ("own method: self.save(path)",
         "class X:\n    def pwn(self, path):\n        self.save(path)\n"),
        ("own method: self.dump(x)",
         "class X:\n    def pwn(self, x):\n        self.dump(x)\n"),
        ("keras-ish: model.save(p)",
         "def pwn(model, p):\n    model.save(p)\n"),
        ("torch.distributed.init_process_group",
         "import torch\ndef pwn():\n    torch.distributed.init_process_group('gloo')\n"),
        ("torch.hub.download_url_to_file",
         "import torch\ndef pwn():\n    torch.hub.download_url_to_file('x', 'y')\n"),
        ("torch.onnx.export",
         "import torch\ndef pwn(model, x, p):\n    torch.onnx.export(model, x, p)\n"),
    ],
)
def test_denied_attributes_is_refused_at_tier0_but_liftable_at_tier2(
    tmp_path, label, source
):
    """core#179 follow-up: closing these at Tier 0/1 is right (a plugin that
    declared nothing, or only a capability, gets no new file-write / remote-
    fetch-and-execute surface for free) -- refusing them at Tier 2 is
    incoherent. ``--trust-author`` already hands over ``subprocess``,
    ``ctypes`` and ``importlib``; a plugin trusted with those but refused
    ``arr.dump()`` or its OWN ``self.save()`` method makes no sense, and
    before this fix there was no way to write a plugin with a method named
    ``save`` at all -- the only fix available to the author was patching
    CodefyUI itself.

    The dunder/RCE precedent does not transfer here: those rules refuse
    REFLECTION, which no capability ever buys (core#133's own docs are
    explicit about this). ``.dump`` / ``.hub`` / ``.save`` are not
    reflection -- they are file writes and remote code fetches,
    ``--trust-author`` has already granted an equivalent or greater version
    of that by a shorter route (``subprocess`` can write files and fetch
    code far more directly than ``numpy.save`` can), so refusing the
    narrower path while granting the wider one protects nothing.
    """
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "n.py").write_text(source, encoding="utf-8")

    # Tier 0/1: nothing declared beyond (at most) a capability -- refused.
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, allowed_modules=[])

    # Tier 2: --trust-author (modelled here as a non-empty allowed_modules,
    # exactly the condition validate_nodes_dir/validate_plugin_dir's callers
    # gate on -- see _install_github, which never calls either function with
    # a non-empty `allowed` unless args.trust_author was already true)
    # accepts it.
    plugin_cli.validate_nodes_dir(nodes, allowed_modules=["torch"])


def test_denied_attributes_lift_also_applies_to_validate_plugin_dir(tmp_path):
    """Same Tier-2 lift, the other entry point -- a separate call to
    ``validate_python_source`` that must not drift from what
    ``validate_nodes_dir`` does."""
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "n.py").write_text(
        "class X:\n    def pwn(self, path):\n        self.save(path)\n",
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_plugin_dir(root, [])
    plugin_cli.validate_plugin_dir(root, ["torch"])


# ── core#182: a skipped directory is still importable at runtime ───────────

@pytest.mark.parametrize(
    "skipped_dir", ["tests", "examples", "assets", "docs", "__pycache__"]
)
def test_validate_plugin_dir_scans_directories_the_loader_can_still_import(
    tmp_path, skipped_dir
):
    """``plugin_loader.install_plugin_finder`` registers the WHOLE plugin
    root as a PEP-420 namespace package's ``__path__`` (not just ``nodes/``),
    so ``cdui_plugins.<id>.tests``, ``.examples``, ``.docs``, ``.assets`` --
    and ``.__pycache__`` -- are all importable -- a scanned ``nodes/foo.py``
    doing ``from ..tests import payload`` executes whatever is in ``tests/``
    at full trust, automatically, at server boot or plugin reload.

    ``_VALIDATION_SKIP_DIRS`` used to carve exactly these five names back out
    of the scan that ``validate_plugin_dir``'s own docstring says was widened
    to whole-tree specifically because the loader exposes the whole
    directory -- reopening the same hole the widening had just closed. The
    scan's view of "what is in-tree" has to match the loader's, so nothing
    admits code the AST gate never looked at.

    ``__pycache__`` belongs in this list, not the "genuinely non-importable"
    one below: it reads as safe only if you reason about what a REAL
    ``__pycache__`` holds (``.pyc`` files, which never match this walk's own
    ``*.py`` glob). That reasoning is about the wrong actor -- the directory
    the interpreter writes is irrelevant here, because the attacker controls
    every byte of the tarball and can name any directory ``__pycache__`` on
    purpose. ``__pycache__`` IS a valid Python identifier
    (``"__pycache__".isidentifier()`` is ``True``), so PEP-420 namespace
    resolution imports it exactly like any other directory name. Verified
    directly against the real loader (not merely argued): with a plugin
    installed via ``install_plugin_finder``, both
    ``importlib.import_module("cdui_plugins.<id>.__pycache__.payload")`` and
    a relative ``from ..__pycache__ import payload`` inside ``nodes/``
    resolve to the file placed there.
    """
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    (root / skipped_dir).mkdir(parents=True)
    (root / skipped_dir / "payload.py").write_text(
        "import os\nos.system('whoami')\n", encoding="utf-8"
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_plugin_dir(root, [])


def test_pycache_is_a_valid_python_identifier_unlike_git():
    """The property the split between the two tests here rests on, checked
    structurally rather than assumed: ``__pycache__`` is an ordinary
    identifier (any ``import`` statement can name it) and ``.git`` is not
    (the leading ``.`` makes it a syntax error as an import-path component),
    which is the actual reason one of them has to be scanned and the other
    provably cannot be reached through Python's import system at all."""
    assert "__pycache__".isidentifier()
    assert not ".git".isidentifier()


def test_validate_plugin_dir_still_skips_the_one_genuinely_unreachable_dir(tmp_path):
    """``.git`` is the only name left in ``_VALIDATION_SKIP_DIRS``, and it
    stays there because it is PROVABLY unreachable through Python's import
    system, not merely assumed to be: ``.git`` is not a valid identifier (see
    ``test_pycache_is_a_valid_python_identifier_unlike_git``), so no
    ``import`` statement -- absolute or relative -- can ever name a package
    component spelled that way. ``__pycache__`` no longer gets this
    exemption; see the parametrized test above for why "looks like an
    implementation-detail directory" was never the same claim as "cannot be
    imported"."""
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "payload.py").write_text(
        "import os\nos.system('whoami')\n", encoding="utf-8"
    )
    plugin_cli.validate_plugin_dir(root, [])  # does not raise


# ── core#220: the scan globbed *.py, the loader imports more ──────────────
#
# core#182 established the property "what the scan considers in-tree matches
# what the loader can import" and enforced it by DIRECTORY. The same property
# was broken by file EXTENSION one layer down, and this is the worse half:
# core#182's gap meant unscanned code in a directory the walk skipped, while
# this one meant the walk never produced a single file to scan. The gate did
# not fail open on a rule -- it never ran.


def _plugin_with(tmp_path, filename: str, payload: bytes) -> Path:
    """A minimal plugin tree whose ``nodes/`` holds one extra file."""
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    (root / "nodes" / filename).write_bytes(payload)
    return root


@pytest.mark.parametrize(
    ("filename", "suffix"),
    [
        ("helper.pyc", ".pyc"),
        ("helper.pyo", ".pyo"),
        ("native.pyd", ".pyd"),
        ("native.so", ".so"),
        ("native.dylib", ".dylib"),
    ],
)
def test_validate_plugin_dir_refuses_compiled_modules_it_cannot_scan(
    tmp_path, filename, suffix
):
    """A plugin shipping a compiled module is refused BY NAME, loudly.

    Before this, ``validate_plugin_dir`` globbed ``*.py`` while
    ``plugin_loader.install_plugin_finder`` handed the whole plugin
    directory to the stock ``FileFinder`` loaders, which accept every suffix
    in ``importlib.machinery.all_suffixes()``. A pack whose ``nodes/`` held
    ``helper.pyc`` and no ``helper.py`` was imported by
    ``NodeRegistry.discover`` at server boot -- at full trust, with no
    capability declared and without ``--trust-author`` -- having never been
    opened by the gate.

    Refusal rather than scanning is the honest answer for both kinds:
    ``.pyc`` would need decompiling, and a ``.pyd`` / ``.so`` cannot be
    AST-scanned even in principle. The alternative is importing code nobody
    looked at, which is the thing this whole module exists to prevent.

    The binary suffixes of OTHER platforms are refused too. A tarball is not
    installed on the machine that built it, so a verdict that depended on
    which OS ran the installer would be a verdict an attacker picks.
    """
    root = _plugin_with(tmp_path, filename, b"\x00\x01\x02 not source")
    with pytest.raises(PluginValidationError) as excinfo:
        plugin_cli.validate_plugin_dir(root, [])
    message = str(excinfo.value)
    assert filename in message, "the refusal must name the file"
    assert suffix in message, "the refusal must name why it could not be scanned"


def test_the_refused_bytecode_really_is_importable(tmp_path):
    """Non-vacuity for the test above: prove the file it refuses is a file
    the import system would genuinely have loaded, rather than a shape that
    only looks dangerous.

    Compiles a real payload to bytecode, deletes the source, and checks that
    ``pkgutil`` -- the same enumeration ``NodeRegistry.discover`` walks with
    -- reports it as an importable module while a ``*.py`` glob sees
    nothing. Without this, the refusal could be guarding a phantom.
    """
    import pkgutil
    import py_compile

    root = tmp_path / "pack"
    nodes = root / "nodes"
    nodes.mkdir(parents=True)
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    src = nodes / "payload_src.py"
    src.write_text("import os\nos.system('whoami')\n", encoding="utf-8")
    py_compile.compile(str(src), cfile=str(nodes / "helper.pyc"), doraise=True)
    src.unlink()

    scanned_by_the_old_glob = sorted(p.name for p in root.rglob("*.py"))
    importable = sorted(m.name for m in pkgutil.iter_modules([str(nodes)]))
    assert "helper" in importable, (
        "the premise of core#220: pkgutil (what NodeRegistry.discover walks) "
        "reports the bytecode-only module as importable"
    )
    assert scanned_by_the_old_glob == ["__init__.py"], (
        "the other half of the premise: a *.py glob never sees it"
    )
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_plugin_dir(root, [])


def test_validate_plugin_dir_scans_pyw_as_the_source_it_is(tmp_path):
    """``.pyw`` is a loader suffix and is ordinary Python text, so it is
    SCANNED rather than refused -- refusing it would be a false positive,
    and skipping it was the bug.

    Scanned on every platform, not only where ``SOURCE_SUFFIXES`` lists it.
    ``.pyw`` is Windows-only as a loader suffix, so a scan running on Linux
    would otherwise wave through a file that a Windows install of the same
    tarball imports.
    """
    root = _plugin_with(tmp_path, "w.pyw", b"import os\nos.system('whoami')\n")
    with pytest.raises(PluginValidationError) as excinfo:
        plugin_cli.validate_plugin_dir(root, [])
    assert "Importing 'os' is not allowed" in str(excinfo.value), (
        "a .pyw should reach the ordinary AST rules, not the refuse-by-suffix "
        "path -- it is source"
    )

    clean = _plugin_with(tmp_path / "ok", "w.pyw", b"VALUE = 1\n")
    plugin_cli.validate_plugin_dir(clean, [])   # does not raise


def test_the_scan_covers_every_suffix_the_import_system_accepts(tmp_path):
    """The standing check core#220 asks for: a future CPython that adds a
    loader suffix fails here instead of silently widening the hole.

    Asked of the interpreter (``importlib.machinery.all_suffixes()``) rather
    than compared against a copy of today's answer, and asserted on
    BEHAVIOUR rather than on set membership -- every suffix gets a real file
    written under a real plugin root, and the walk has to do one of exactly
    two things with it: scan it as source, or refuse it by name. A third
    outcome, "walked straight past it", is the state this closed, and a
    membership check would not have distinguished the third from the second.
    """
    import importlib.machinery

    accepts = set(importlib.machinery.all_suffixes())
    assert accepts <= plugin_cli.LOADER_SUFFIXES, (
        "this interpreter's import system accepts suffixes the plugin scan "
        f"does not consider: {sorted(accepts - plugin_cli.LOADER_SUFFIXES)}"
    )
    assert set(importlib.machinery.SOURCE_SUFFIXES) <= plugin_cli.SCANNABLE_SUFFIXES

    for index, suffix in enumerate(sorted(accepts | plugin_cli.LOADER_SUFFIXES)):
        root = _plugin_with(
            tmp_path / f"probe{index}",
            "probe" + suffix,
            b"import os\nos.system('whoami')\n",
        )
        with pytest.raises(PluginValidationError) as excinfo:
            plugin_cli.validate_plugin_dir(root, [])
        message = str(excinfo.value)
        if suffix in plugin_cli.SCANNABLE_SUFFIXES:
            assert "Importing 'os' is not allowed" in message, (
                f"{suffix!r} is source, so it must reach the AST rules"
            )
        else:
            assert suffix in message and "probe" in message, (
                f"{suffix!r} cannot be scanned, so the refusal must name the "
                f"file and say why -- silence is the core#220 bug"
            )


def test_longest_suffix_wins_so_a_versioned_extension_is_still_refused(tmp_path):
    """``native.cp314-win_amd64.pyd`` ends with two loader suffixes, and
    picking the short one would defeat the refusal.

    Strip only ``.pyd`` and the stem is ``native.cp314-win_amd64``, which is
    not an identifier -- so the "can an import statement name this?" screen
    would skip it, and a real compiled extension (the exact filename
    ``setuptools`` produces) would sail through unexamined. Strip the full
    ``.cp314-win_amd64.pyd`` and the stem is ``native``, which it can.
    """
    import importlib.machinery

    versioned = importlib.machinery.EXTENSION_SUFFIXES[0]
    root = _plugin_with(tmp_path, "native" + versioned, b"\x00binary")
    assert plugin_cli.loader_suffix(
        root / "nodes" / ("native" + versioned)
    ) == versioned, "longest match must win"
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_plugin_dir(root, [])


def test_a_cpython_written_bytecode_cache_is_not_mistaken_for_a_payload(tmp_path):
    """The false positive the refusal must NOT produce.

    Any plugin directory an interpreter has ever imported from carries a
    ``__pycache__`` full of ``<name>.cpython-NNN.pyc`` -- this repo's own
    built-in packs included, which is how this surfaced. Those are not
    reachable by any import statement: strip ``.pyc`` and the stem is
    ``real.cpython-314``, which is not a valid identifier, so no import can
    name it. An attacker-placed ``payload.pyc`` in the SAME directory has
    the stem ``payload``, which is.

    Verified rather than argued, because the distinction carries the whole
    rule: with a plugin exposed through the real loader,
    ``import <pkg>.nodes.__pycache__.payload`` ran the planted bytecode
    while ``import <pkg>.nodes.__pycache__.real`` raised
    ``ModuleNotFoundError``, and ``pkgutil.iter_modules`` listed only the
    first. So the refusal fires on the one that is reachable and stays quiet
    on the one that is not -- the same structural test ``.git`` gets, applied
    to a filename instead of a directory name.
    """
    import py_compile

    root = tmp_path / "pack"
    nodes = root / "nodes"
    cache = nodes / "__pycache__"
    cache.mkdir(parents=True)
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (nodes / "real.py").write_text("VALUE = 1\n", encoding="utf-8")
    py_compile.compile(str(nodes / "real.py"), doraise=True)

    artifacts = sorted(p.name for p in cache.iterdir())
    assert artifacts, "py_compile should have written a cache file"
    for name in artifacts:
        assert not name[: -len(".pyc")].isidentifier(), (
            f"{name} is CPython's own cache artifact; the premise of this "
            "test is that its stem is unspellable as a module name"
        )
    plugin_cli.validate_plugin_dir(root, [])   # does not raise

    # ...and the same directory with an attacker-named file in it does raise.
    py_compile.compile(
        str(nodes / "real.py"), cfile=str(cache / "payload.pyc"), doraise=True
    )
    with pytest.raises(PluginValidationError) as excinfo:
        plugin_cli.validate_plugin_dir(root, [])
    assert "payload.pyc" in str(excinfo.value)


def test_validate_nodes_dir_refuses_the_same_compiled_module(tmp_path):
    """The other entry point. ``validate_nodes_dir`` is a separate call into
    the walker and used to carry its own ``*.py`` glob, so a fix applied to
    one of the two would have left the narrower one open."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "helper.pyc").write_bytes(b"\x00\x01 not source")
    with pytest.raises(PluginValidationError):
        plugin_cli.validate_nodes_dir(nodes, [])


def test_the_git_directory_exemption_survives_the_suffix_rule(tmp_path):
    """``.git`` holds loose objects and packfiles with arbitrary names, and
    it is skipped for a structural reason that the suffix rule must not
    quietly undo: ``.git`` is not a valid identifier, so no import can name
    a component spelled that way (core#182). A ``.pyc`` planted in there is
    still unreachable."""
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "payload.pyc").write_bytes(b"\x00\x01 not source")
    plugin_cli.validate_plugin_dir(root, [])   # does not raise


# ── load_catalog ────────────────────────────────────────────────────────────

def test_load_catalog_returns_three_direction_packs():
    catalog = plugin_cli.load_catalog()
    plugins = catalog.get("plugins", {})
    assert set(plugins.keys()) >= {"foundations", "deep", "rl"}
    for pid in ("foundations", "deep", "rl"):
        assert plugins[pid].get("kind") == "builtin"
        assert plugins[pid].get("path") == f"plugins/{pid}"


# ── _install_deps spec construction ───────────────────────────────────────

def test_install_deps_builds_correct_pip_specs(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, check=False):
        captured.append(cmd)
        class _R:
            returncode = 0
        return _R()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = plugin_cli._install_deps({
        "foo": ">=1.0",         # version constraint passes through
        "bar": "==2.3.4",       # explicit equality
        "baz": "1.0.0",         # bare version → coerced to ==1.0.0
        "qux": "",              # no constraint
    })
    assert rc == 0
    assert captured, "uv pip install should have been invoked"
    cmd = captured[0]
    assert cmd[:3] == ["uv", "pip", "install"]
    specs = cmd[3:]
    assert "foo>=1.0" in specs
    assert "bar==2.3.4" in specs
    assert "baz==1.0.0" in specs
    assert "qux" in specs


# ── _manifest_has_frontend ────────────────────────────────────────────────

def test_manifest_has_frontend_detection():
    assert plugin_cli._manifest_has_frontend({"frontend": {"entry": "frontend/index.js"}}) is True
    assert plugin_cli._manifest_has_frontend({}) is False
    assert plugin_cli._manifest_has_frontend({"frontend": {}}) is False
    assert plugin_cli._manifest_has_frontend({"frontend": {"entry": ""}}) is False


# ── link / unlink / reload (local dev loop) ────────────────────────────────

@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect the lockfile to a temp dir and stub the server hot-reload so
    CLI tests never touch real user data or a running server.

    ``plugins_user_root`` is patched in BOTH modules on purpose:
    ``scripts/plugins.py`` imported the name, so it holds its own reference,
    and patching only ``plugin_loader``'s would move the lockfile while
    leaving any install path writing files into the real user data dir.
    """
    target = tmp_path / "plugins"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return target


def _write_plugin_dir(root: Path, plugin_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cdui.plugin.toml").write_text(
        dedent(f"""\
            [plugin]
            id = "{plugin_id}"
            name = "Local {plugin_id}"
            version = "0.1.0"
            schema_version = 1
            """),
        encoding="utf-8",
    )
    nodes = root / "nodes"
    nodes.mkdir(exist_ok=True)
    (nodes / "__init__.py").write_text("", encoding="utf-8")


def test_cmd_link_writes_local_lockfile_entry(isolated_lockfile, tmp_path):
    work = tmp_path / "work" / "my-dev-plugin"
    _write_plugin_dir(work, "my-dev-plugin")

    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))
    assert rc == 0

    entry = plugin_loader.load_lockfile()["plugins"]["my-dev-plugin"]
    assert entry["source_kind"] == "local"
    assert Path(entry["path"]) == work.resolve()
    assert entry["enabled"] is True


def test_cmd_link_rejects_dir_without_manifest(isolated_lockfile, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(empty), force=False))
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}


def test_cmd_link_rejects_catalog_id_collision(isolated_lockfile, tmp_path):
    work = tmp_path / "shadow"
    _write_plugin_dir(work, "foundations")  # a real built-in catalog id
    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))
    assert rc == 1
    assert "foundations" not in plugin_loader.load_lockfile()["plugins"]


def test_cmd_link_existing_id_requires_force(isolated_lockfile, tmp_path):
    work = tmp_path / "dup"
    _write_plugin_dir(work, "dup-plugin")
    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False)) == 0
    # Re-linking without --force is rejected; with --force it succeeds.
    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False)) == 1
    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=True)) == 0


def test_cmd_unlink_removes_entry_without_deleting_files(isolated_lockfile, tmp_path):
    work = tmp_path / "work" / "linked"
    _write_plugin_dir(work, "linked")
    plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))

    rc = plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="linked"))
    assert rc == 0
    assert "linked" not in plugin_loader.load_lockfile()["plugins"]
    # The author's working tree is untouched.
    assert (work / "cdui.plugin.toml").exists()


def test_cmd_unlink_refuses_non_local_entry(isolated_lockfile):
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["deep"] = {
        "source_kind": "builtin", "source": "deep", "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)

    rc = plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="deep"))
    assert rc == 1
    assert "deep" in plugin_loader.load_lockfile()["plugins"]  # refused to drop it


def test_cmd_unlink_missing_plugin_errors(isolated_lockfile):
    assert plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="nope")) == 1


def test_cmd_reload_no_server_returns_zero(isolated_lockfile):
    # _backend_reload stubbed to False (no server) — reload is a graceful no-op.
    assert plugin_cli.cmd_reload(argparse.Namespace()) == 0


def test_link_unlink_reload_parser_wired():
    parser = plugin_cli.build_parser()
    a = parser.parse_args(["link", "/some/path"])
    assert a._func is plugin_cli.cmd_link and a.path == "/some/path" and a.force is False
    a = parser.parse_args(["link", "/p", "--force"])
    assert a.force is True
    a = parser.parse_args(["unlink", "foo"])
    assert a._func is plugin_cli.cmd_unlink and a.plugin_id == "foo"
    a = parser.parse_args(["reload"])
    assert a._func is plugin_cli.cmd_reload


# ── dev (watch mode) ───────────────────────────────────────────────────────

def test_scan_plugin_files_covers_relevant_dirs(tmp_path):
    root = tmp_path / "p"
    _write_plugin_dir(root, "p")  # manifest + nodes/__init__.py
    (root / "frontend").mkdir()
    (root / "frontend" / "index.js").write_text("export default function(){}", encoding="utf-8")
    (root / "README.md").write_text("ignored", encoding="utf-8")
    (root / "nodes" / "__pycache__").mkdir()
    (root / "nodes" / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")

    keys = {Path(k).name for k in plugin_cli._scan_plugin_files(root)}
    assert {"cdui.plugin.toml", "__init__.py", "index.js"} <= keys
    assert "README.md" not in keys   # not a reload-relevant dir
    assert "x.pyc" not in keys       # __pycache__ ignored


def test_scan_plugin_files_detects_new_file(tmp_path):
    root = tmp_path / "p"
    _write_plugin_dir(root, "p")
    before = plugin_cli._scan_plugin_files(root)
    (root / "nodes" / "extra.py").write_text("# new node", encoding="utf-8")
    after = plugin_cli._scan_plugin_files(root)
    assert after != before
    assert any(Path(k).name == "extra.py" for k in after)


def test_cmd_dev_once_links_and_returns_zero(isolated_lockfile, tmp_path):
    root = tmp_path / "devplug"
    _write_plugin_dir(root, "devplug")
    rc = plugin_cli.cmd_dev(argparse.Namespace(path=str(root), once=True, interval=1.0))
    assert rc == 0
    entry = plugin_loader.load_lockfile()["plugins"]["devplug"]
    assert entry["source_kind"] == "local"
    assert Path(entry["path"]) == root.resolve()


def test_cmd_dev_once_bad_dir_returns_one(isolated_lockfile, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = plugin_cli.cmd_dev(argparse.Namespace(path=str(empty), once=True, interval=1.0))
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}


def test_dev_parser_wired():
    parser = plugin_cli.build_parser()
    a = parser.parse_args(["dev", "/p"])
    assert a._func is plugin_cli.cmd_dev and a.path == "/p" and a.once is False
    a = parser.parse_args(["dev", "/p", "--once", "--interval", "0.5"])
    assert a.once is True and a.interval == 0.5
