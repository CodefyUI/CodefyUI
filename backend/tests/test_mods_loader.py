"""Phase 4C step 2: the one gated loader for a project's mods/.

Every test that must NOT run project code gives the code a side effect (a
line appended to a marker file outside the project) and checks the marker
is still absent afterwards.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import mods, projects
from app.core.auth import TOKEN_HEADER, session_token
from app.core.node_registry import provider_for, registry
from app.core.plugins.errors import ManifestError
from app.core.plugins.manifest import validate_manifest
from app.core.preset_registry import preset_registry
from app.core.project import scaffold_project
from app.main import app

from .test_projects_startup import isolated  # noqa: F401  (fixture)

BASE_URL = f"http://127.0.0.1:{settings.PORT}"


def _write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _node(name: str, cls: str = "N", extra: str = "") -> str:
    return (
        "from app.core.node_base import BaseNode\n"
        f"{extra}\n\n"
        f"class {cls}(BaseNode):\n"
        f"    NODE_NAME = {name!r}\n"
        "    CATEGORY = 'Project'\n"
        "    DESCRIPTION = 'from mods'\n"
        "    @classmethod\n"
        "    def define_inputs(cls):\n"
        "        return []\n"
        "    @classmethod\n"
        "    def define_outputs(cls):\n"
        "        return []\n"
        "    def execute(self, inputs, params):\n"
        "        return {}\n"
    )


def _marking(marker: Path) -> str:
    """A module body that leaves a trace when it runs."""
    return f"open({str(marker)!r}, 'a').write('ran\\n')\n"


def _project(root: Path, files: dict[str, str | bytes]) -> Path:
    scaffold_project(root)
    for rel, data in files.items():
        _write(root / "mods" / rel, data)
    return root.resolve()


def _symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")


@pytest.fixture(autouse=True)
def _unloaded(monkeypatch):
    monkeypatch.setattr(mods, "_ran", set())
    mods.unload_project_mods()
    yield
    mods.unload_project_mods()


def _project_keys() -> list[str]:
    return sorted(k for k in registry.nodes if k.startswith("project:"))


# ── Loading only after trust ─────────────────────────────────────────────

def test_an_untrusted_clone_runs_nothing(isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    p = _project(isolated / "p", {"nodes/loss.py": _node("FocalLoss", extra=_marking(marker))})
    status = mods.load_project_mods(p)
    assert status["state"] == "untrusted"
    assert status["loaded"] is False
    assert _project_keys() == []
    assert not marker.exists()
    assert not any(m.startswith("codefyui_project_") for m in sys.modules if "." in m)


def test_trusted_nodes_register_qualified_and_never_bare(isolated):  # noqa: F811
    p = _project(isolated / "p", {"nodes/loss.py": _node("FocalLoss")})
    tree = mods.walk_mods(p)
    mods.record_trust(p, tree, full=False)
    status = mods.load_project_mods(p)
    assert status["state"] == "trusted"
    assert status["nodes"] == ["project:FocalLoss"]
    cls = registry.get("project:FocalLoss")
    assert cls is not None
    assert cls.__module__ == f"{mods.package_name(p)}.loss"
    assert provider_for("project:FocalLoss", cls) == "project"
    assert registry.get("FocalLoss") is None   # the bare name never reaches it
    assert status["restart_recommended"] is False   # only its own code ran


def test_a_project_conv2d_leaves_the_builtin_alone(isolated):  # noqa: F811
    builtin = registry.get("Conv2d")
    assert builtin is not None
    p = _project(isolated / "p", {"nodes/conv.py": _node("Conv2d", cls="MyConv")})
    mods.record_trust(p, mods.walk_mods(p), full=False)
    mods.load_project_mods(p)
    assert registry.get("project:Conv2d").__name__ == "MyConv"
    assert registry.get("Conv2d") is builtin
    mods.unload_project_mods()
    assert registry.get("Conv2d") is builtin
    assert registry.get("project:Conv2d") is None


def test_helpers_import_relatively_and_only_trusted_files(isolated):  # noqa: F811
    p = _project(isolated / "p", {
        "nodes/helper.py": "K = 3\n",
        "nodes/loss.py": _node("UsesHelper", extra="from .helper import K\nVALUE = K"),
        "nodes/bad.py": _node("ImportsMissing", cls="B", extra="from . import absent"),
    })
    mods.record_trust(p, mods.walk_mods(p), full=False)
    status = mods.load_project_mods(p)
    assert "project:UsesHelper" in status["nodes"]
    assert sys.modules[f"{mods.package_name(p)}.loss"].VALUE == 3
    assert "project:ImportsMissing" not in status["nodes"]
    assert any(s["path"] == "nodes/bad.py" for s in status["skipped"])


def test_only_classes_defined_in_the_module_register(isolated):  # noqa: F811
    p = _project(isolated / "p", {
        "nodes/loss.py": _node("Own", extra="from app.nodes.cnn.conv2d_node import Conv2dNode"),
    })
    mods.record_trust(p, mods.walk_mods(p), full=False)
    assert mods.load_project_mods(p)["nodes"] == ["project:Own"]


def test_files_with_findings_load_only_when_trusted_completely(isolated):  # noqa: F811
    p = _project(isolated / "p", {
        "nodes/clean.py": _node("Clean"),
        "nodes/risky.py": _node("Risky", cls="R", extra="import os"),
    })
    scan = mods.scan_mods(p)
    assert next(f for f in scan["files"] if f["path"] == "nodes/risky.py")["finding"]
    mods.record_trust(p, mods.walk_mods(p), full=False)
    status = mods.load_project_mods(p)
    assert status["nodes"] == ["project:Clean"]
    assert any(s["path"] == "nodes/risky.py" for s in status["skipped"])
    mods.record_trust(p, mods.walk_mods(p), full=True)
    assert sorted(mods.load_project_mods(p)["nodes"]) == ["project:Clean", "project:Risky"]


def test_duplicate_node_names_load_neither(isolated):  # noqa: F811
    p = _project(isolated / "p", {"nodes/a.py": _node("Same"), "nodes/b.py": _node("Same")})
    mods.record_trust(p, mods.walk_mods(p), full=False)
    status = mods.load_project_mods(p)
    assert status["nodes"] == []
    assert sorted(s["path"] for s in status["skipped"]) == ["nodes/a.py", "nodes/b.py"]


# ── The required refusals ────────────────────────────────────────────────

def test_a_pyc_beside_a_clean_file_blocks_everything(isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    import importlib.util
    tag = sys.implementation.cache_tag
    p = _project(isolated / "p", {"nodes/x.py": _node("X")})
    # Bytecode whose source hash nobody checks, that would leave a mark.
    code = compile(_marking(marker) + _node("X"), "x.py", "exec")
    pyc = importlib.util.MAGIC_NUMBER + b"\x01\x00\x00\x00" + b"\0" * 8
    import marshal
    _write(p / "mods" / "nodes" / "__pycache__" / f"x.{tag}.pyc", pyc + marshal.dumps(code))
    assert mods.mods_trust_state(p)["state"] == "blocked"
    mods.record_trust(p, mods.walk_mods(p), full=True)   # even a record does not help
    status = mods.load_project_mods(p)
    assert status["state"] == "blocked"
    assert _project_keys() == []
    assert not marker.exists()


def test_a_package_under_nodes_never_runs(isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    p = _project(isolated / "p", {
        "nodes/clean.py": _node("Clean"),
        "nodes/pkg/__init__.py": _marking(marker),
        "nodes/pkg/inner.py": _marking(marker),
    })
    tree = mods.walk_mods(p)
    assert any("package" in reason for _, reason in tree.refused)
    mods.record_trust(p, tree, full=True)
    assert mods.load_project_mods(p)["state"] == "blocked"
    assert _project_keys() == []
    assert not marker.exists()


def test_a_symlink_out_of_the_project_is_not_read(isolated, tmp_path_factory):  # noqa: F811
    outside_dir = tmp_path_factory.mktemp("outside")
    marker = outside_dir / "ran"
    outside = _write(outside_dir / "evil.py", _marking(marker) + _node("Evil"))
    p = _project(isolated / "p", {"nodes/clean.py": _node("Clean")})
    _symlink(p / "mods" / "nodes" / "x.py", outside)
    tree = mods.walk_mods(p)
    assert "nodes/x.py" not in tree.files
    assert any(path == "nodes/x.py" for path, _ in tree.refused)
    mods.record_trust(p, tree, full=True)
    assert mods.load_project_mods(p)["state"] == "blocked"
    assert not marker.exists()


def test_a_file_changed_after_it_was_trusted_does_not_run(isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    p = _project(isolated / "p", {"nodes/x.py": _node("X")})
    mods.record_trust(p, mods.walk_mods(p), full=True)
    _write(p / "mods" / "nodes" / "x.py", _marking(marker) + _node("X"))
    status = mods.load_project_mods(p)
    assert status["state"] == "changed"
    assert not marker.exists()


def test_the_finder_checks_the_hash_at_exec(isolated, tmp_path_factory):  # noqa: F811
    """The window between the load decision and exec: the finder reads the
    bytes once and compares them with the hash the user agreed to."""
    marker = tmp_path_factory.mktemp("out") / "ran"
    p = _project(isolated / "p", {"nodes/x.py": _marking(marker)})
    finder = mods._ModsFinder("codefyui_project_t", p / "mods" / "nodes", {"x": "0" * 64})
    spec = finder.find_spec("codefyui_project_t.x")
    import importlib.util
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError, match="changed after it was trusted"):
        finder.exec_module(module)
    assert not marker.exists()
    assert finder.find_spec("codefyui_project_t.other") is None
    assert finder.find_spec("codefyui_project_t.x.y") is None


def test_the_plugin_id_project_is_reserved():
    manifest = {"plugin": {"schema_version": 1, "id": "project", "name": "x",
                           "version": "0.1.0"}}
    with pytest.raises(ManifestError, match="reserved"):
        validate_manifest(manifest)


# ── Presets ──────────────────────────────────────────────────────────────

def _preset(name: str, node_type: str) -> str:
    return json.dumps({"preset_name": name, "category": "Project", "description": "",
                       "nodes": [{"id": "a", "type": node_type, "params": {}}],
                       "edges": [], "exposed_inputs": [], "exposed_outputs": [],
                       "exposed_params": []})


def test_presets_load_after_nodes_and_never_replace_builtins(isolated):  # noqa: F811
    builtin = preset_registry.get("Simple CNN Classifier")
    assert builtin is not None
    p = _project(isolated / "p", {
        "nodes/loss.py": _node("FocalLoss"),
        "presets/mine.json": _preset("Uses Focal", "project:FocalLoss"),
        "presets/clash.json": _preset("Simple CNN Classifier", "Conv2d"),
    })
    mods.record_trust(p, mods.walk_mods(p), full=False)
    status = mods.load_project_mods(p)
    assert status["presets"] == ["Uses Focal"]
    assert any(s["path"] == "presets/clash.json" for s in status["skipped"])
    assert preset_registry.get("Simple CNN Classifier") is builtin
    mods.unload_project_mods()
    assert preset_registry.get("Uses Focal") is None
    assert preset_registry.get("Simple CNN Classifier") is builtin


def test_a_preset_using_unloaded_project_nodes_is_skipped(isolated):  # noqa: F811
    p = _project(isolated / "p", {
        "nodes/loss.py": _node("FocalLoss"),
        "presets/mine.json": _preset("Uses Focal", "project:FocalLoss"),
    })
    status = mods.load_project_mods(p)   # not trusted
    assert status["presets"] == []
    assert preset_registry.get("Uses Focal") is None


# ── Unloading ────────────────────────────────────────────────────────────

def test_switching_away_unloads_and_recommends_a_restart(isolated):  # noqa: F811
    p = _project(isolated / "p", {"nodes/loss.py": _node("FocalLoss")})
    mods.record_trust(p, mods.walk_mods(p), full=False)
    mods.load_project_mods(p)
    package = mods.package_name(p)
    assert package in sys.modules
    status = mods.load_project_mods(None)
    assert _project_keys() == []
    assert not any(m == package or m.startswith(package + ".") for m in sys.modules)
    assert not any(isinstance(f, mods._ModsFinder) for f in sys.meta_path)
    assert status["restart_recommended"] is True


# ── Untrusted, through every door ────────────────────────────────────────

@pytest.fixture
async def http(isolated):  # noqa: F811
    saved = getattr(app.state, "run_service", None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as client:
        yield client
    app.state.run_service = saved


async def test_untrusted_mods_never_run_through_any_door(http, isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    p = _project(isolated / "ext" / "clone", {
        "nodes/loss.py": _node("FocalLoss", extra=_marking(marker))})
    other = _project(isolated / "ext" / "other", {})
    # Open through the API.
    resp = await http.post("/api/projects/open", json={"path": str(p)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mods"]["needs_consent"] is True
    # Reload Nodes.
    assert (await http.post("/api/nodes/reload")).status_code == 200
    # Switch away and back.
    assert (await http.post("/api/projects/open", json={"path": str(other)})).status_code == 200
    assert (await http.post("/api/projects/open", json={"path": str(p)})).status_code == 200
    # A restart runs the same gate as startup does.
    mods.unload_project_mods()
    mods.load_project_mods(settings.PROJECT_DIR)
    assert not marker.exists()
    assert _project_keys() == []


async def test_consent_with_a_stale_digest_is_refused(http, isolated):  # noqa: F811
    p = _project(isolated / "ext" / "clone", {"nodes/loss.py": _node("FocalLoss")})
    assert (await http.post("/api/projects/open", json={"path": str(p)})).status_code == 200
    seen = (await http.get("/api/projects/mods")).json()
    assert [f["nodes"] for f in seen["files"]] == [["FocalLoss"]]
    _write(p / "mods" / "nodes" / "loss.py", _node("FocalLoss") + "# changed\n")
    resp = await http.post("/api/projects/trust-mods", json={"sha256": seen["digest"]})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "mods_changed"
    assert _project_keys() == []
    fresh = (await http.get("/api/projects/mods")).json()
    resp = await http.post("/api/projects/trust-mods", json={"sha256": fresh["digest"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mods"]["nodes"] == ["project:FocalLoss"]
    assert registry.get("project:FocalLoss") is not None
    assert projects._trust_entries()  # recorded in the user's data


# ── Wording: run / validate / publish ────────────────────────────────────

def test_validate_explains_a_project_type_that_is_not_loaded():
    from app.core.graph_engine import validate_graph

    nodes = [{"id": "s", "type": "Start", "data": {"params": {}}},
             {"id": "f", "type": "project:FocalLoss", "data": {"params": {}}},
             {"id": "u", "type": "NoSuchThing", "data": {"params": {}}}]
    errors = validate_graph(nodes, [])
    focal = next(e for e in errors if "project:FocalLoss" in e)
    assert "mods/ folder, which is not loaded" in focal
    assert "cdui project mods --trust" in focal
    assert focal.params.get("reason") == "mods_not_loaded"
    other = next(e for e in errors if "NoSuchThing" in e)
    assert "mods/" not in other and "reason" not in other.params


async def test_a_declined_project_node_survives_save_then_trust(http, isolated):  # noqa: F811
    """The required round trip on the server side: decline, open the
    graph, edit something else, save (as a folder graph), trust -- the
    project node comes back with its type, label, params and wires."""
    p = _project(isolated / "ext" / "clone", {"nodes/loss.py": _node("FocalLoss")})
    # A scaffolded project stores new graphs as folders ([storage] dir).
    assert (await http.post("/api/projects/open", json={"path": str(p)})).status_code == 200
    assert registry.get("project:FocalLoss") is None     # declined: not loaded
    graph = {
        "name": "uses", "description": "",
        "nodes": [
            {"id": "s", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "f", "type": "project:FocalLoss", "position": {"x": 200, "y": 0},
             "data": {"label": "My focal", "params": {"gamma": 3.5, "alpha": 0.25}}},
            {"id": "p", "type": "Print", "position": {"x": 400, "y": 0},
             "data": {"params": {"label": "edited"}}},
        ],
        "edges": [{"id": "e1", "source": "f", "sourceHandle": "loss",
                   "target": "p", "targetHandle": "value"}],
        "presets": [], "segmentGroups": [],
    }
    resp = await http.post("/api/graph/save", json={**graph, "file": "uses"})
    assert resp.status_code == 200, resp.text
    assert (p / "graphs" / "uses").is_dir()
    index = json.loads((p / "graphs" / "uses" / "index.json").read_text(encoding="utf-8"))
    assert index["requires"]["mods"] == {"nodes": ["project:FocalLoss"]}

    seen = (await http.get("/api/projects/mods")).json()
    assert (await http.post("/api/projects/trust-mods",
                            json={"sha256": seen["digest"]})).status_code == 200
    loaded = (await http.get("/api/graph/load/uses")).json()
    focal = next(n for n in loaded["nodes"] if n["id"] == "f")
    assert focal["type"] == "project:FocalLoss"
    assert focal["data"]["label"] == "My focal"
    assert focal["data"]["params"] == {"gamma": 3.5, "alpha": 0.25}
    assert [(e["source"], e.get("sourceHandle"), e["target"], e.get("targetHandle"))
            for e in loaded["edges"]] == [("f", "loss", "p", "value")]
