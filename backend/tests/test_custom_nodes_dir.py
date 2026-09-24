"""A custom nodes directory outside the install loads from where it is (#519).

``NodeRegistry.discover`` walked the directory it was handed but imported each
module it found BY NAME, as ``app.custom_nodes.<file>`` -- and Python finds
that name through the package's own ``__path__``, which is
``backend/app/custom_nodes`` whatever ``CODEFYUI_CUSTOM_NODES_DIR`` says. So a
directory anywhere else was listed by the Custom Node Manager and never
loaded: each import failed with "No module named ...", and a file sharing a
name with one in the repo loaded the repo's file instead.

The registry tests discover into a fresh ``NodeRegistry``. The route tests
go through the upload and reload routes, which rebuild the process-wide
registries, so their fixture puts both back afterwards.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

import pytest

import app.custom_nodes
from app.config import settings
from app.core import plugin_loader
from app.core.node_registry import NodeRegistry, registry
from app.core.preset_registry import preset_registry

#: The directory the repo's own custom nodes live in.
_REPO_DIR = Path(app.custom_nodes.__file__).resolve().parent


def _node_source(node_name: str, *, class_name: str = "ProbeNode",
                 relative: bool = False) -> str:
    """A complete custom node file defining *node_name*.

    ``relative=True`` imports the base class the way the shipped example
    (``backend/app/custom_nodes/example_custom_node.py``) does, which only
    resolves when the module sits one package below ``app``.
    """
    source = "..core.node_base" if relative else "app.core.node_base"
    return (
        f"from {source} import BaseNode, DataType, PortDefinition\n"
        "\n"
        "\n"
        f"class {class_name}(BaseNode):\n"
        f"    NODE_NAME = {node_name!r}\n"
        "    CATEGORY = 'Custom'\n"
        "    DESCRIPTION = 'probe'\n"
        "\n"
        "    @classmethod\n"
        "    def define_inputs(cls):\n"
        "        return []\n"
        "\n"
        "    @classmethod\n"
        "    def define_outputs(cls):\n"
        "        return [PortDefinition(name='value', data_type=DataType.ANY)]\n"
        "\n"
        "    def execute(self, inputs, params):\n"
        "        return {'value': 1}\n"
    )


def _loaded_from(cls: type) -> Path:
    """The file the module defining *cls* was really read from."""
    return Path(sys.modules[cls.__module__].__file__).resolve()


class _RaisingInit(importlib.abc.Loader):
    """The loader of a package whose ``__init__.py`` raises when it runs."""

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError("broken __init__.py")


class _BrokenRepoPackage(importlib.abc.MetaPathFinder):
    """Finds ``app.custom_nodes`` where it really is, with that loader."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "app.custom_nodes":
            return None
        spec = importlib.machinery.ModuleSpec(fullname, _RaisingInit(),
                                              is_package=True)
        spec.submodule_search_locations = [str(_REPO_DIR)]
        return spec


@pytest.fixture
def broken_repo_package(monkeypatch):
    """``backend/app/custom_nodes/__init__.py`` raising, as a bad edit or a
    bad merge leaves it. The package and its modules are forgotten for the
    test, so the next import runs the finder above; ``monkeypatch`` puts the
    real ones back."""
    for name in [name for name in sys.modules
                 if name == "app.custom_nodes"
                 or name.startswith("app.custom_nodes.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BrokenRepoPackage(), *sys.meta_path])


# ── the registry ─────────────────────────────────────────────────────────


def test_a_node_in_a_directory_outside_the_install_is_registered(tmp_path):
    (tmp_path / "ext_probe.py").write_text(_node_source("ExtProbe"),
                                           encoding="utf-8")

    found = NodeRegistry()
    assert found.discover(tmp_path, "app.custom_nodes") == 1

    node = found.get("ExtProbe")
    assert node is not None
    assert _loaded_from(node) == (tmp_path / "ext_probe.py").resolve()
    # Not a name under the repo's package, where it could meet a file of the
    # same name in backend/app/custom_nodes.
    assert not node.__module__.startswith("app.custom_nodes.")


def test_a_relative_import_resolves_as_it_does_in_the_repo_directory(tmp_path):
    """The shipped example imports ``from ..core.node_base``; a copy of it
    moved into the configured directory has to keep working."""
    (tmp_path / "ext_relative.py").write_text(
        _node_source("ExtRelative", relative=True), encoding="utf-8")

    found = NodeRegistry()
    found.discover(tmp_path, "app.custom_nodes")

    assert found.get("ExtRelative") is not None


@pytest.mark.parametrize("force_reload", [False, True],
                         ids=["startup", "reload"])
def test_a_same_named_file_in_the_repo_package_does_not_shadow_it(
        tmp_path, force_reload):
    """``example_custom_node.py`` exists in both places. The configured
    directory's copy is the one that loads, at startup and on a reload."""
    # Imported first, as it is in any server that has run a discovery of the
    # default directory: the cached module is what used to win.
    import app.custom_nodes.example_custom_node  # noqa: F401

    (tmp_path / "example_custom_node.py").write_text(
        _node_source("ExtShadow"), encoding="utf-8")

    found = NodeRegistry()
    found.discover(tmp_path, "app.custom_nodes", force_reload=force_reload)

    assert found.get("ExtShadow") is not None
    assert found.get("AddScalar") is None


def test_a_reload_picks_up_an_edit(tmp_path):
    node_file = tmp_path / "ext_edit.py"
    node_file.write_text(_node_source("ExtBefore", class_name="EditNode"),
                         encoding="utf-8")
    before = NodeRegistry()
    before.discover(tmp_path, "app.custom_nodes")
    assert before.get("ExtBefore") is not None

    # A different LENGTH as well as different text: a same-size rewrite in
    # the same second can be served from the stale bytecode cache, which is
    # Python's rule and not what this test is about.
    node_file.write_text(_node_source("ExtAfterTheEdit", class_name="EditNode"),
                         encoding="utf-8")
    after = NodeRegistry()
    after.discover(tmp_path, "app.custom_nodes", force_reload=True)

    assert after.get("ExtAfterTheEdit") is not None
    assert after.get("ExtBefore") is None


def test_another_directory_is_read_afresh(tmp_path):
    """The setting moved (a test, or a restart with a new value): a module
    already imported from the old directory must not answer for the new one,
    even without a forced reload."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "same.py").write_text(_node_source("ExtOld"), encoding="utf-8")
    (new / "same.py").write_text(_node_source("ExtNew"), encoding="utf-8")

    NodeRegistry().discover(old, "app.custom_nodes")
    found = NodeRegistry()
    found.discover(new, "app.custom_nodes")

    assert found.get("ExtNew") is not None
    assert found.get("ExtOld") is None
    assert _loaded_from(found.get("ExtNew")) == (new / "same.py").resolve()


def test_another_spelling_of_the_same_directory_keeps_its_modules(tmp_path):
    """Only a DIFFERENT directory drops what was imported; ``sub/..`` is this
    one, so its classes stay the objects already registered."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "ext_same.py").write_text(_node_source("ExtSame"),
                                          encoding="utf-8")

    first = NodeRegistry()
    first.discover(tmp_path / "sub" / "..", "app.custom_nodes")
    second = NodeRegistry()
    second.discover(tmp_path, "app.custom_nodes")

    assert first.get("ExtSame") is not None
    assert second.get("ExtSame") is first.get("ExtSame")


def test_files_in_the_directory_import_each_other_relatively(tmp_path):
    """What the docs tell node authors: ``from .helpers import x`` reads the
    configured directory."""
    (tmp_path / "helpers.py").write_text("SUFFIX = 'Helped'\n",
                                         encoding="utf-8")
    source = _node_source("ExtPlaceholder").replace(
        "NODE_NAME = 'ExtPlaceholder'", "NODE_NAME = 'Ext' + SUFFIX")
    (tmp_path / "ext_helped.py").write_text(
        "from .helpers import SUFFIX\n" + source, encoding="utf-8")

    found = NodeRegistry()
    found.discover(tmp_path, "app.custom_nodes")

    assert found.get("ExtHelped") is not None


def test_a_broken_repo_package_does_not_stop_its_own_discovery(
        broken_repo_package):
    """The server's startup and every reload call this. A broken
    ``__init__.py`` fails each module under it, logged, and the discovery
    goes on to the plugins -- it must not raise."""
    found = NodeRegistry()

    assert found.discover(_REPO_DIR, "app.custom_nodes") == 0


def test_a_broken_repo_package_does_not_stop_the_configured_directory(
        broken_repo_package, tmp_path):
    (tmp_path / "ext_probe.py").write_text(_node_source("ExtProbe"),
                                           encoding="utf-8")
    found = NodeRegistry()

    assert found.discover(tmp_path, "app.custom_nodes") == 1
    assert found.get("ExtProbe") is not None


def test_an_install_without_the_repo_package_still_loads_the_directory(
        tmp_path, monkeypatch):
    """An install that deleted ``backend/app/custom_nodes`` and points the
    setting elsewhere: the discovery must not fail on the missing package."""
    monkeypatch.setitem(sys.modules, "app.custom_nodes", None)  # unimportable
    (tmp_path / "ext_alone.py").write_text(_node_source("ExtAlone"),
                                           encoding="utf-8")

    found = NodeRegistry()

    assert found.discover(tmp_path, "app.custom_nodes") == 1
    assert found.get("ExtAlone") is not None


def test_the_repo_directory_keeps_its_own_package():
    """The default configuration is unchanged: the repo's nodes are still
    ``app.custom_nodes.<file>``, the very classes a direct import returns."""
    from app.custom_nodes.example_custom_node import AddScalarNode

    found = NodeRegistry()
    found.discover(_REPO_DIR, "app.custom_nodes")

    assert found.get("AddScalar") is AddScalarNode


def test_the_repo_directory_is_recognised_by_another_spelling():
    """A relative spelling of the same directory is still the repo's
    package, so its classes stay the ones already imported."""
    from app.custom_nodes.example_custom_node import AddScalarNode

    try:
        spelled = Path(os.path.relpath(_REPO_DIR))
    except ValueError:  # the working directory is on another drive
        pytest.skip("no relative spelling of the repo directory from here")

    found = NodeRegistry()
    found.discover(spelled, "app.custom_nodes")

    assert found.get("AddScalar") is AddScalarNode


# ── through the routes ───────────────────────────────────────────────────


@pytest.fixture
def configured_dir(tmp_path, monkeypatch):
    """``CUSTOM_NODES_DIR`` pointed outside the install for one test.

    The upload and reload routes clear and rebuild the process-wide node and
    preset registries from whatever the settings say at that moment, and
    ``monkeypatch`` only puts the SETTING back. So both registries are
    restored as they were: a node registered from this directory (or the
    repo's ``AddScalar``, dropped while the setting pointed here) would
    otherwise carry over into every later test.
    """
    directory = tmp_path / "my_nodes"
    directory.mkdir()
    monkeypatch.setattr(settings, "CUSTOM_NODES_DIR", directory)
    # The rediscovery would also re-import every pack in this machine's real
    # lockfile; CI has none, and neither does this test.
    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        plugin_loader.empty_lockfile)
    nodes = dict(registry._nodes)
    presets = dict(preset_registry._presets)
    try:
        yield directory
    finally:
        registry._nodes.clear()
        registry._nodes.update(nodes)
        preset_registry._presets.clear()
        preset_registry._presets.update(presets)


def _upload(filename: str, source: str) -> dict:
    return {"files": {"file": (filename, source.encode("utf-8"),
                               "text/x-python")}}


async def test_an_uploaded_node_reaches_the_palette(test_client, configured_dir):
    """What the issue reported: the manager listed the node and the palette
    never had it. Both now say the same thing."""
    response = await test_client.post(
        "/api/custom-nodes/upload",
        **_upload("ext_probe.py", _node_source("ExtProbe")))
    assert response.status_code == 200, response.text

    listed = (await test_client.get("/api/custom-nodes")).json()
    assert listed == [
        {"filename": "ext_probe.py", "enabled": True, "nodes": ["ExtProbe"]}]

    palette = {entry["node_name"]: entry
               for entry in (await test_client.get("/api/nodes")).json()}
    assert "ExtProbe" in palette
    assert palette["ExtProbe"]["provider"] == "custom"


async def test_a_reload_does_not_load_the_repo_copy_of_a_same_named_file(
        test_client, configured_dir):
    (configured_dir / "example_custom_node.py").write_text(
        _node_source("ExtShadow"), encoding="utf-8")

    response = await test_client.post("/api/nodes/reload")
    assert response.status_code == 200, response.text
    assert response.json()["custom"] == 1

    assert registry.get("ExtShadow") is not None
    assert registry.get("AddScalar") is None


async def test_a_reload_with_a_broken_repo_package_still_answers(
        test_client, configured_dir, broken_repo_package):
    """The reload clears the registry first; a raise after that left only the
    built-ins behind, and answered 500."""
    (configured_dir / "ext_probe.py").write_text(_node_source("ExtProbe"),
                                                 encoding="utf-8")

    response = await test_client.post("/api/nodes/reload")

    assert response.status_code == 200, response.text
    assert response.json()["custom"] == 1
    assert registry.get("ExtProbe") is not None


async def test_uploading_a_new_version_replaces_the_old_node(
        test_client, configured_dir):
    for node_name in ("ExtV1", "ExtVersionTwo"):
        response = await test_client.post(
            "/api/custom-nodes/upload",
            **_upload("ext_version.py",
                      _node_source(node_name, class_name="VersionNode")))
        assert response.status_code == 200, response.text

    assert registry.get("ExtVersionTwo") is not None
    assert registry.get("ExtV1") is None
