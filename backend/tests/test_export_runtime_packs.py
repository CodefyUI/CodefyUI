"""An exported script loads the plugin packs bundled with CodefyUI by itself.

The script lists the node types it runs in ``_REQUIRED_NODE_TYPES`` and used
to find plugin nodes only through the plugin lockfile in the user data dir. A
grader runs ``python <file>.py`` with an empty one, so a graph that used
``edu:FFNLayer`` stopped with "Unknown node type" while the pack sat in this
installation's ``plugins/`` directory. ``initialize_export_runtime`` loads
the bundled packs the script names, writes nothing, and names the pack when
one is not there or is turned off.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from app.core import plugin_loader
from app.core import runtime as runtime_module
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry

#: Printed by the graph's one Print node, so a run that got that far shows.
PRINT_MARKER = "the-graph-ran"

#: Torch alone takes seconds to import, and a cold Windows runner is slow.
RUN_TIMEOUT_S = 120

TINY_NODE = """\
    from app.core.node_base import BaseNode


    class Tiny(BaseNode):
        NODE_NAME = "Tiny"
        CATEGORY = "Test"
        DESCRIPTION = "Takes nothing, gives nothing"

        @classmethod
        def define_inputs(cls):
            return []

        @classmethod
        def define_outputs(cls):
            return []

        def execute(self, inputs, params, **_):
            return {}
    """

OTHER_NODE = TINY_NODE.replace('"Tiny"', '"Other"').replace("class Tiny", "class Other")

#: A pack whose only module cannot be imported.
BROKEN_NODE = """\
    import a_module_this_environment_does_not_have  # noqa: F401
    """


def _write_pack(root: Path, pack_id: str, module_source: str) -> None:
    """A pack under *root*: a manifest and a ``nodes`` package of one module."""
    nodes = root / pack_id / "nodes"
    nodes.mkdir(parents=True)
    (root / pack_id / plugin_loader.MANIFEST_FILENAME).write_text(
        dedent(f"""\
            [plugin]
            id = "{pack_id}"
            name = "{pack_id}"
            version = "0.0.1"
            schema_version = 1
            """),
        encoding="utf-8",
    )
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (nodes / "pack_node.py").write_text(dedent(module_source), encoding="utf-8")


def _write_lockfile(user_data_dir: Path, plugins: dict) -> Path:
    """The plugin lockfile in *user_data_dir*, listing *plugins*."""
    path = user_data_dir / "plugins" / "installed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "plugins": plugins}), encoding="utf-8")
    return path


def _forget_pack(pack_id: str) -> None:
    """Drop what discovery imported for a pack outside the built-in root."""
    plugin_loader.purge_plugin_modules(pack_id)
    namespace = sys.modules.get(plugin_loader.NAMESPACE_PACKAGE)
    if namespace is not None:
        vars(namespace).pop(pack_id.replace("-", "_"), None)


@pytest.fixture
def user_data_dir(tmp_path, monkeypatch):
    """An empty user data dir for this test, not yet created.

    ``initialize_runtime`` clears the process-wide node and preset registries
    and rebuilds them from that dir, which drops the packs the session
    loaded. Both registries are restored exactly as they were afterwards, so
    the next test does not depend on what this one discovered.
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "user-data"))
    nodes = dict(registry._nodes)
    presets = dict(preset_registry._presets)
    try:
        yield tmp_path / "user-data"
    finally:
        registry._nodes.clear()
        registry._nodes.update(nodes)
        preset_registry._presets.clear()
        preset_registry._presets.update(presets)


@pytest.fixture
def builtin_root(tmp_path, monkeypatch, user_data_dir):
    """A stand-in for ``<repo>/plugins``, empty until a test writes packs."""
    root = tmp_path / "builtin"
    root.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root", lambda: root)
    try:
        yield root
    finally:
        namespace = sys.modules.get(plugin_loader.NAMESPACE_PACKAGE)
        for pack in root.iterdir():
            plugin_loader.purge_plugin_modules(pack.name)
            if namespace is not None:
                # The attribute carries the importable spelling of the id.
                vars(namespace).pop(pack.name.replace("-", "_"), None)


# -- In process, against a stand-in built-in root -----------------------------


@pytest.mark.parametrize(
    "lockfile",
    [
        None,
        {"schema": 1, "plugins": {
            "tinypack": {"source_kind": "builtin", "enabled": False}}},
        {"schema": 1, "plugins": {}, "removed": {
            "tinypack": {"removed_at": "2026-10-05T00:00:00+00:00",
                         "source_kind": "builtin"}}},
    ],
    ids=["no lockfile", "disabled", "uninstalled"],
)
def test_a_bundled_pack_the_script_names_loads_whatever_the_lockfile_says(
    builtin_root, user_data_dir, lockfile,
):
    _write_pack(builtin_root, "tinypack", TINY_NODE)
    lockfile_path = user_data_dir / "plugins" / "installed.json"
    if lockfile is not None:
        lockfile_path.parent.mkdir(parents=True)
        lockfile_path.write_text(json.dumps(lockfile), encoding="utf-8")
    before = lockfile_path.read_bytes() if lockfile is not None else None

    problems = runtime_module.initialize_export_runtime(["Start", "tinypack:Tiny"])

    assert problems == []
    assert registry.get("tinypack:Tiny") is not None
    # The lockfile governs the editor and is left exactly as it was.
    if lockfile is None:
        assert not user_data_dir.exists(), "the user data dir was written"
    else:
        assert lockfile_path.read_bytes() == before


def test_a_bundled_pack_whose_id_has_a_hyphen_loads_under_that_id(builtin_root):
    # Imported as cdui_plugins.tiny_pack, since a module path cannot hold a
    # hyphen; registered, like the type in the graph, as the manifest's id.
    _write_pack(builtin_root, "tiny-pack", TINY_NODE)

    assert runtime_module.initialize_export_runtime(["tiny-pack:Tiny"]) == []

    assert registry.get("tiny-pack:Tiny") is not None


def test_a_script_whose_types_all_resolve_loads_no_pack(builtin_root):
    _write_pack(builtin_root, "tinypack", TINY_NODE)

    assert runtime_module.initialize_export_runtime(["Start", "Print"]) == []

    assert registry.get("tinypack:Tiny") is None
    assert "cdui_plugins.tinypack" not in sys.modules


def test_only_the_packs_the_script_names_load(builtin_root):
    _write_pack(builtin_root, "tinypack", TINY_NODE)
    _write_pack(builtin_root, "otherpack", OTHER_NODE)

    assert runtime_module.initialize_export_runtime(["tinypack:Tiny"]) == []

    assert registry.get("otherpack:Other") is None
    assert "cdui_plugins.otherpack" not in sys.modules


def test_a_pack_this_installation_lacks_is_named_with_its_install_command(
    builtin_root,
):
    problems = runtime_module.initialize_export_runtime(["ghostpack:Ghost"])

    assert problems == [
        "Unknown node type: ghostpack:Ghost -- it comes from the plugin pack "
        "'ghostpack', which this CodefyUI installation does not have. "
        "Install it with: cdui plugin install ghostpack"
    ]


@pytest.mark.parametrize("linked", [False, True], ids=["downloaded", "linked"])
def test_a_disabled_pack_installed_here_is_named_with_its_enable_command(
    builtin_root, user_data_dir, tmp_path, linked,
):
    """Installed and turned off: the user needs ``enable``, not ``install``.
    The pack is not loaded -- unlike a bundled pack, a third-party one may be
    off because it is broken or untrusted -- and the lockfile is untouched."""
    plugins = user_data_dir / "plugins"
    if linked:
        # `cdui plugin link`: loaded in place from the author's checkout.
        _write_pack(tmp_path / "checkout", "offpack", TINY_NODE)
        entry = {"source_kind": "local",
                 "path": str(tmp_path / "checkout" / "offpack"), "enabled": False}
    else:
        _write_pack(plugins, "offpack", TINY_NODE)
        entry = {"source_kind": "github", "enabled": False}
    lockfile_path = plugins / "installed.json"
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    lockfile_path.write_text(
        json.dumps({"schema": 1, "plugins": {"offpack": entry}}), encoding="utf-8")
    before = lockfile_path.read_bytes()

    try:
        problems = runtime_module.initialize_export_runtime(["offpack:Tiny"])
        loaded = "cdui_plugins.offpack" in sys.modules
    finally:
        # Only a regression loads it, and then it must not outlive this test.
        plugin_loader.purge_plugin_modules("offpack")
        namespace = sys.modules.get(plugin_loader.NAMESPACE_PACKAGE)
        if namespace is not None:
            vars(namespace).pop("offpack", None)

    assert problems == [
        "Unknown node type: offpack:Tiny -- it comes from the plugin pack "
        "'offpack', which is installed here but disabled. Enable it with: "
        "cdui plugin enable offpack"
    ]
    assert not loaded, "the disabled pack was loaded"
    assert lockfile_path.read_bytes() == before


def test_an_installed_enabled_pack_that_loads_no_node_is_not_sent_to_install(
    builtin_root, user_data_dir,
):
    """Installed, turned on, files on disk, and every module fails to import.
    ``cdui plugin install`` answers "already installed" for such a pack, so
    the line says what happened instead of sending the user there."""
    _write_pack(user_data_dir / "plugins", "brokenpack", BROKEN_NODE)
    _write_lockfile(user_data_dir, {"brokenpack": {
        "source_kind": "github_url", "source": "someone/brokenpack",
        "enabled": True}})

    try:
        problems = runtime_module.initialize_export_runtime(["brokenpack:Thing"])
    finally:
        _forget_pack("brokenpack")

    assert problems == [
        "Unknown node type: brokenpack:Thing -- it comes from the plugin pack "
        "'brokenpack', which is installed and enabled here but none of its "
        "nodes could be loaded (see the log above)"
    ]


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_a_listed_pack_whose_files_are_gone_is_named_with_its_reinstall_command(
    builtin_root, user_data_dir, enabled,
):
    """The lockfile still lists it, so a plain install is refused as "already
    installed": the command reinstalls with --force, from where it came."""
    _write_lockfile(user_data_dir, {"gonepack": {
        "source_kind": "github_url", "source": "someone/gonepack@v1",
        "enabled": enabled}})

    problems = runtime_module.initialize_export_runtime(["gonepack:Thing"])

    assert problems == [
        "Unknown node type: gonepack:Thing -- it comes from the plugin pack "
        "'gonepack', which is listed as installed here but its files are "
        "missing. Reinstall it with: "
        "cdui plugin install someone/gonepack@v1 --force"
    ]


@pytest.mark.parametrize(
    ("source", "recorded"),
    [
        ("someone/gonepack; rm -rf ~", " (recorded as 'someone/gonepack; rm -rf ~')"),
        ("someone/gone pack", " (recorded as 'someone/gone pack')"),
        ("-x/gonepack", " (recorded as '-x/gonepack')"),
        ("someone/x" + chr(27) + "[2J", " (recorded as 'someone/x\\x1b[2J')"),
        (42, ""),
        (None, ""),
    ],
    ids=["semicolon", "space", "option-like", "escape sequence", "not a string",
         "no source"],
)
def test_a_source_that_is_not_owner_repo_is_quoted_and_kept_out_of_the_command(
    builtin_root, user_data_dir, source, recorded,
):
    """A hand-edited ``source`` never goes into a command someone may paste
    into a shell, and neither does the pack id, which ``cdui plugin install``
    resolves only as a catalog name. The source is quoted as data, with any
    control character escaped, and the command shows a placeholder."""
    entry = {"source_kind": "github_url", "enabled": True}
    if source is not None:
        entry["source"] = source
    _write_lockfile(user_data_dir, {"gonepack": entry})

    problems = runtime_module.initialize_export_runtime(["gonepack:Thing"])

    assert problems == [
        "Unknown node type: gonepack:Thing -- it comes from the plugin pack "
        "'gonepack', which is listed as installed here but its files are "
        f"missing. Reinstall it from the repository it came from{recorded} "
        "with: cdui plugin install <owner/repo> --force"
    ]


@pytest.mark.parametrize(
    "folder",
    ["devpack", None, "dev" + chr(27) + "[2Jpack"],
    ids=["path recorded", "no path", "escape sequence in the path"],
)
def test_a_linked_pack_whose_folder_is_gone_is_named_with_its_link_command(
    builtin_root, user_data_dir, tmp_path, folder,
):
    """`cdui plugin link` refuses an id the lockfile already has without
    --force, so the command carries it. The recorded path is shown with any
    control character escaped and every other character as it is."""
    entry = {"source_kind": "local", "enabled": True}
    where = ""
    if folder is not None:
        moved = tmp_path / "checkout" / folder  # never created
        entry.update(source=str(moved), path=str(moved))
        shown = str(moved).replace(chr(27), "\\x1b")
        where = f" ({shown})"
    _write_lockfile(user_data_dir, {"devpack": entry})

    problems = runtime_module.initialize_export_runtime(["devpack:Thing"])

    assert problems == [
        "Unknown node type: devpack:Thing -- it comes from the plugin pack "
        f"'devpack', whose linked folder{where} is missing. Link it again "
        "with: cdui plugin link <folder> --force"
    ]
    assert chr(27) not in problems[0]


def test_a_linked_folder_that_lost_its_manifest_is_not_called_missing(
    builtin_root, user_data_dir, tmp_path,
):
    """The folder is there and only its manifest is gone: putting that file
    back fixes the link, and linking the right folder is the other way."""
    folder = tmp_path / "checkout" / "devpack"
    (folder / "nodes").mkdir(parents=True)
    _write_lockfile(user_data_dir, {"devpack": {
        "source_kind": "local", "source": str(folder), "path": str(folder),
        "enabled": True}})

    problems = runtime_module.initialize_export_runtime(["devpack:Thing"])

    assert problems == [
        "Unknown node type: devpack:Thing -- it comes from the plugin pack "
        f"'devpack', linked from {folder}, which has no cdui.plugin.toml. Put "
        "that file back, or link the folder that has it with: "
        "cdui plugin link <folder> --force"
    ]


def test_a_bundled_pack_the_lockfile_turns_off_is_never_sent_to_enable(
    builtin_root, user_data_dir,
):
    """A bundled pack loads whatever the lockfile says, so enabling it would
    change nothing: when it loads no node, its line says that instead."""
    _write_pack(builtin_root, "brokenpack", BROKEN_NODE)
    lockfile_path = user_data_dir / "plugins" / "installed.json"
    lockfile_path.parent.mkdir(parents=True)
    lockfile_path.write_text(json.dumps({"schema": 1, "plugins": {
        "brokenpack": {"source_kind": "builtin", "enabled": False}}}),
        encoding="utf-8")

    problems = runtime_module.initialize_export_runtime(["brokenpack:Thing"])

    assert problems == [
        "Unknown node type: brokenpack:Thing -- the plugin pack 'brokenpack' "
        "ships with this CodefyUI but none of its nodes could be loaded"
    ]


def test_a_node_the_loaded_pack_lacks_is_named_with_the_likely_causes(builtin_root):
    _write_pack(builtin_root, "tinypack", TINY_NODE)

    problems = runtime_module.initialize_export_runtime(["tinypack:Nope"])

    assert problems == [
        "Unknown node type: tinypack:Nope -- the plugin pack 'tinypack' here "
        "has no node 'Nope' (a different CodefyUI version, or one of its "
        "modules failed to load; see the log above)"
    ]
    # The pack itself did load: it is the node that is missing.
    assert registry.get("tinypack:Tiny") is not None


def test_a_bundled_pack_that_loads_no_node_says_so(builtin_root):
    _write_pack(builtin_root, "brokenpack", BROKEN_NODE)

    problems = runtime_module.initialize_export_runtime(["brokenpack:Thing"])

    assert problems == [
        "Unknown node type: brokenpack:Thing -- the plugin pack 'brokenpack' "
        "ships with this CodefyUI but none of its nodes could be loaded"
    ]


@pytest.mark.parametrize(
    "node_type",
    ["NotARealNode", "../x:Y", "Not_A_Pack:Y"],
    ids=["bare name", "path in the pack id", "not a pack id"],
)
def test_a_type_that_names_no_valid_pack_stays_a_plain_unknown_type(
    builtin_root, node_type,
):
    # A real pack at <root>/../x: a pack id is only ever a directory name, so
    # one that is not a valid id must never be joined onto the root.
    _write_pack(builtin_root.parent, "x", TINY_NODE.replace('"Tiny"', '"Y"'))

    problems = runtime_module.initialize_export_runtime([node_type])

    assert problems == [f"Unknown node type: {node_type}"]


def test_problems_come_one_per_type_in_sorted_order(builtin_root):
    _write_pack(builtin_root, "tinypack", TINY_NODE)

    problems = runtime_module.initialize_export_runtime(
        ["tinypack:Nope", "Start", "NotARealNode", "ghostpack:Ghost", "NotARealNode"])

    assert [problem.split(" -- ")[0] for problem in problems] == [
        "Unknown node type: NotARealNode",
        "Unknown node type: ghostpack:Ghost",
        "Unknown node type: tinypack:Nope",
    ]


# -- The exported file --------------------------------------------------------


def _small_graph() -> tuple[list[dict], list[dict]]:
    """Start -> TensorCreate -> Print, built-in nodes only."""
    def node(node_id: str, node_type: str, **params) -> dict:
        return {"id": node_id, "type": node_type,
                "position": {"x": 0, "y": 0}, "data": {"params": params}}

    nodes = [
        node("start", "Start"),
        node("zeros", "TensorCreate", shape="2", fill="zeros", value=0.0),
        node("show", "Print", label=PRINT_MARKER),
    ]
    edges = [
        {"id": "t", "source": "start", "target": "zeros",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d", "source": "zeros", "target": "show",
         "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
    ]
    return nodes, edges


def _also_requiring(script: str, *extra: str) -> str:
    """*script* with *extra* added to its ``_REQUIRED_NODE_TYPES`` line."""
    prefix = "_REQUIRED_NODE_TYPES = "
    [line] = [line for line in script.splitlines() if line.startswith(prefix)]
    required = ast.literal_eval(line[len(prefix):])
    return script.replace(line, prefix + repr([*required, *extra]))


def _run_like_a_grader(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """``python <file>.py`` in a folder of its own, with no argument, no
    CODEFYUI_* variable and an empty user data dir: no plugin lockfile."""
    folder = tmp_path / "grader"
    folder.mkdir()
    (folder / "exported_graph.py").write_text(script, encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("CODEFYUI_")
    }
    env["CODEFYUI_USER_DATA_DIR"] = str(tmp_path / "user-data")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-I", "exported_graph.py"],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT_S,
        check=False,
    )


def test_an_exported_script_naming_a_pack_it_cannot_find_exits_2_and_names_it(
    tmp_path: Path,
):
    from app.core.codegen import generate_python

    script = generate_python(*_small_graph(), name="missing pack")
    script = _also_requiring(script, "ghostpack:Ghost", "edu:NoSuchNode")

    run = _run_like_a_grader(script, tmp_path)

    assert run.returncode == 2, run.stderr
    assert "Exported graph validation failed: " in run.stderr
    assert "Unknown node type: ghostpack:Ghost" in run.stderr
    assert "cdui plugin install ghostpack" in run.stderr
    # edu ships with CodefyUI, so the script loaded it; the node is what is
    # missing.
    assert "Unknown node type: edu:NoSuchNode" in run.stderr
    assert "the plugin pack 'edu' here has no node 'NoSuchNode'" in run.stderr
    # With no GraphOutput, Print writes to stdout; either stream would show it.
    assert PRINT_MARKER not in run.stdout + run.stderr, "nodes ran before setup failed"
    assert not (tmp_path / "user-data").exists(), "the user data dir was written"


def test_a_script_run_by_a_codefyui_without_the_new_call_checks_as_before(
    user_data_dir, monkeypatch,
):
    """A file exported now, run by a CodefyUI older than this change."""
    from app.core.codegen import generate_python

    script = generate_python(*_small_graph(), name="older backend")
    module: dict = {"__name__": "exported_graph_on_an_older_backend"}
    exec(compile(script, "<exported graph>", "exec"), module)  # noqa: S102
    monkeypatch.delattr(runtime_module, "initialize_export_runtime")

    rt = module["_load_runtime"](None)

    assert rt.initialize_export_runtime(["Start"]) == []
    assert rt.initialize_export_runtime(["Start", "NotARealNode"]) == [
        "Unknown node type: NotARealNode"
    ]
