"""``/api/skills`` and ``/api/agent/context``: discovery, shadowing, reading
files safely, enabling, deleting, and the session token on every write.

Each test gets its own user, built-in, project and plugin skill roots in
``tmp_path`` (the real built-in skills are read by one test only, the one
that checks them), so nothing here depends on what is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes_agent_context
from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.skills import registry
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
REPO_SKILLS = Path(__file__).resolve().parents[2] / "skills"


def make_skill(root: Path, folder: str, *, name: str | None = None,
               description: str | None = "Does a thing. Use when testing.",
               body: str = "# Steps\n", files: dict[str, str | bytes] | None = None) -> Path:
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    front = [f"name: {name or folder}"]
    if description is not None:
        front.append(f"description: {description}")
    (path / "SKILL.md").write_text("---\n" + "\n".join(front) + "\n---\n" + body,
                                   encoding="utf-8")
    for rel, content in (files or {}).items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return path


class Roots:
    def __init__(self, tmp: Path):
        self.user_data = tmp / "user"
        self.user = self.user_data / "skills"
        self.builtin = tmp / "builtin"
        self.project = tmp / "project"
        self.plugin = tmp / "plugin-demo"
        for d in (self.user, self.builtin, self.project / "skills", self.plugin / "skills"):
            d.mkdir(parents=True)


@pytest.fixture
def roots(tmp_path, monkeypatch) -> Roots:
    r = Roots(tmp_path)
    monkeypatch.setattr(registry, "_user_data_root", lambda: r.user_data)
    monkeypatch.setattr(settings, "SKILLS_BUILTIN_DIR", r.builtin)
    monkeypatch.setattr(settings, "SKILLS_EXTRA_DIRS", "")
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(registry.plugin_loader, "iter_plugin_dirs", lambda *a, **k: [])
    return r


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as c:
        yield c


@pytest.fixture
async def anon():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as c:
        yield c


def _by_id(listing: list[dict]) -> dict[str, dict]:
    return {s["id"]: s for s in listing}


# ── the built-in skills ────────────────────────────────────────────────────

async def test_the_three_builtin_skills_are_listed_valid_with_no_issues(client, monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_user_data_root", lambda: tmp_path)
    monkeypatch.setattr(settings, "SKILLS_BUILTIN_DIR", REPO_SKILLS)
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(settings, "SKILLS_EXTRA_DIRS", "")
    resp = await client.get("/api/skills")
    assert resp.status_code == 200
    builtin = {s["name"]: s for s in resp.json() if s["source"] == "builtin"}
    assert set(builtin) == {"build-graph", "train-and-evaluate", "run-experiments"}
    for skill in builtin.values():
        assert skill["valid"] is True, skill
        assert skill["issues"] == [], skill
        assert skill["has_scripts"] is False
        assert skill["shadowed_by"] is None


def test_builtin_skill_bodies_stay_under_500_lines():
    for skill_md in REPO_SKILLS.glob("*/SKILL.md"):
        assert len(skill_md.read_text(encoding="utf-8").splitlines()) <= 500, skill_md


# ── listing ────────────────────────────────────────────────────────────────

async def test_a_broken_skill_is_listed_invalid_and_costs_nothing_else(client, roots):
    make_skill(roots.builtin, "good")
    make_skill(roots.user, "no-description", description=None)
    (roots.user / "garbage").mkdir()
    (roots.user / "garbage" / "SKILL.md").write_bytes(b"\xff\xfe not utf8")
    (roots.user / "not-a-skill").mkdir()  # no SKILL.md: not listed at all
    resp = await client.get("/api/skills")
    assert resp.status_code == 200
    skills = _by_id(resp.json())
    assert set(skills) == {"builtin/good", "user/no-description", "user/garbage"}
    assert skills["builtin/good"]["valid"] is True
    bad = skills["user/no-description"]
    assert bad["valid"] is False
    assert [i["code"] for i in bad["issues"]] == ["description_missing"]
    assert skills["user/garbage"]["valid"] is False


async def test_list_entry_shape(client, roots):
    make_skill(roots.builtin, "demo", files={
        "references/guide.md": "x", "scripts/run.py": "print(1)", "assets/t.txt": "t", "notes.txt": "n"})
    entry = (await client.get("/api/skills")).json()[0]
    assert set(entry) == {"id", "name", "description", "source", "license", "compatibility",
                          "metadata", "allowed_tools", "enabled", "valid", "issues",
                          "shadowed_by", "file_count", "has_scripts"}
    assert entry["file_count"] == 4
    assert entry["has_scripts"] is True


async def test_a_name_mismatch_loads_under_the_folder_name_with_a_warning(client, roots):
    make_skill(roots.user, "folder-name", name="declared-name")
    entry = _by_id((await client.get("/api/skills")).json())["user/folder-name"]
    assert entry["valid"] is True
    assert entry["name"] == "folder-name"
    assert entry["issues"][0]["level"] == "warning"


async def test_sources_and_priority(client, roots, monkeypatch):
    monkeypatch.setattr(settings, "PROJECT_DIR", roots.project)
    monkeypatch.setattr(registry.plugin_loader, "iter_plugin_dirs",
                        lambda *a, **k: [("demo", roots.plugin)])
    extra = roots.user_data.parent / "extra"
    monkeypatch.setattr(settings, "SKILLS_EXTRA_DIRS", str(extra))
    for root in (roots.project / "skills", roots.user, roots.plugin / "skills",
                 roots.builtin, extra):
        make_skill(root, "same")
    make_skill(extra, "only-extra")
    skills = (await client.get("/api/skills")).json()
    ids = [s["id"] for s in skills]
    assert ids == ["project/same", "user/same", "plugin:demo/same", "builtin/same",
                   "extra/only-extra", "extra/same"]
    by = _by_id(skills)
    assert by["project/same"]["shadowed_by"] is None
    for loser in ("user/same", "plugin:demo/same", "builtin/same", "extra/same"):
        assert by[loser]["shadowed_by"] == "project/same"
    assert by["extra/only-extra"]["shadowed_by"] is None


async def test_extra_dirs_are_not_read_by_default(client, roots):
    assert settings.SKILLS_EXTRA_DIRS == ""
    assert [r for r in registry.source_roots() if r[0] == "extra"] == []


async def test_project_skill_shadows_builtin_and_context_returns_only_project(client, roots, monkeypatch):
    monkeypatch.setattr(settings, "PROJECT_DIR", roots.project)
    monkeypatch.setattr(settings, "GRAPHS_DIR", roots.project / "graphs")
    make_skill(roots.builtin, "build-graph", description="builtin copy")
    make_skill(roots.project / "skills", "build-graph", description="project copy")
    by = _by_id((await client.get("/api/skills")).json())
    assert by["builtin/build-graph"]["shadowed_by"] == "project/build-graph"
    assert by["project/build-graph"]["shadowed_by"] is None
    context = (await client.get("/api/agent/context")).json()
    assert context["skills"] == [{"id": "project/build-graph", "name": "build-graph",
                                  "description": "project copy", "source": "project"}]


async def test_an_invalid_or_disabled_skill_does_not_shadow(client, roots):
    make_skill(roots.user, "x", description=None)
    make_skill(roots.builtin, "x")
    by = _by_id((await client.get("/api/skills")).json())
    assert by["builtin/x"]["shadowed_by"] is None
    assert by["user/x"]["shadowed_by"] == "builtin/x"

    make_skill(roots.user, "y")
    make_skill(roots.builtin, "y")
    resp = await client.put("/api/skills/enabled", json={"id": "user/y", "enabled": False})
    assert resp.status_code == 200 and resp.json()["enabled"] is False
    by = _by_id((await client.get("/api/skills")).json())
    assert by["builtin/y"]["shadowed_by"] is None
    assert by["user/y"]["shadowed_by"] == "builtin/y"


# ── one skill ──────────────────────────────────────────────────────────────

async def test_get_one_skill_has_body_and_files(client, roots):
    make_skill(roots.builtin, "demo", body="# Hello\nworld\n", files={
        "references/guide.md": "g", "scripts/x.sh": "echo", "assets/a.png": b"\x89PNG", "README": "r"})
    resp = await client.get("/api/skills/builtin/demo")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["body"] == "# Hello\nworld\n"
    kinds = {f["path"]: f["kind"] for f in detail["files"]}
    assert kinds == {"references/guide.md": "reference", "scripts/x.sh": "script",
                     "assets/a.png": "asset", "README": "other"}
    assert all(isinstance(f["size"], int) for f in detail["files"])


async def test_plugin_ids_with_a_colon_resolve(client, roots, monkeypatch):
    monkeypatch.setattr(registry.plugin_loader, "iter_plugin_dirs",
                        lambda *a, **k: [("deep", roots.plugin)])
    make_skill(roots.plugin / "skills", "attention-lab")
    resp = await client.get("/api/skills/plugin:deep/attention-lab")
    assert resp.status_code == 200
    assert resp.json()["source"] == "plugin:deep"


async def test_unknown_skill_is_404(client, roots):
    assert (await client.get("/api/skills/builtin/nope")).status_code == 404
    assert (await client.get("/api/skills/file", params={"id": "builtin/nope", "path": "a"})).status_code == 404


# ── reading a file ─────────────────────────────────────────────────────────

async def test_read_a_reference_file(client, roots):
    make_skill(roots.builtin, "demo", files={"references/guide.md": "# Guide\n"})
    resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": "references/guide.md"})
    assert resp.status_code == 200
    assert resp.json() == {"path": "references/guide.md", "content": "# Guide\n", "truncated": False}


@pytest.mark.parametrize("path", [
    "../other/SKILL.md", "references/../../other/SKILL.md", "..", "/etc/passwd",
    "\\windows\\system32", "C:/Windows/win.ini", "references\\..\\..\\other\\SKILL.md",
])
async def test_traversal_and_absolute_paths_are_400(client, roots, path):
    make_skill(roots.builtin, "demo")
    make_skill(roots.builtin, "other")
    resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": path})
    assert resp.status_code == 400, path


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
async def test_a_symlink_leaving_the_skill_is_400(client, roots, tmp_path):
    skill = make_skill(roots.builtin, "demo")
    secret = tmp_path / "secret.txt"
    secret.write_text("token", encoding="utf-8")
    (skill / "references").mkdir()
    (skill / "references" / "leak.md").symlink_to(secret)
    (skill / "updir").symlink_to(tmp_path, target_is_directory=True)
    for path in ("references/leak.md", "updir/secret.txt"):
        resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": path})
        assert resp.status_code == 400, path
    # And the listing does not offer what cannot be read.
    files = (await client.get("/api/skills/builtin/demo")).json()["files"]
    assert "references/leak.md" not in {f["path"] for f in files}


async def test_a_binary_file_is_415(client, roots):
    make_skill(roots.builtin, "demo", files={"assets/logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"})
    resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": "assets/logo.png"})
    assert resp.status_code == 415


async def test_a_large_file_is_cut_at_256_kb(client, roots):
    big = ("é" * 200_000)  # 400 000 bytes of two-byte characters
    make_skill(roots.builtin, "demo", files={"references/big.md": big})
    resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": "references/big.md"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["truncated"] is True
    assert len(body["content"].encode("utf-8")) <= 256 * 1024
    assert set(body["content"]) == {"é"}


async def test_a_missing_file_is_404(client, roots):
    make_skill(roots.builtin, "demo")
    resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": "references/none.md"})
    assert resp.status_code == 404


# ── enable / delete ────────────────────────────────────────────────────────

async def test_disable_persists_in_state_json(client, roots):
    make_skill(roots.builtin, "demo")
    resp = await client.put("/api/skills/enabled", json={"id": "builtin/demo", "enabled": False})
    assert resp.status_code == 200
    assert '"builtin/demo"' in (roots.user / "state.json").read_text(encoding="utf-8")
    assert _by_id((await client.get("/api/skills")).json())["builtin/demo"]["enabled"] is False
    assert (await client.get("/api/agent/context")).json()["skills"] == []
    await client.put("/api/skills/enabled", json={"id": "builtin/demo", "enabled": True})
    assert _by_id((await client.get("/api/skills")).json())["builtin/demo"]["enabled"] is True


async def test_enable_unknown_is_404(client, roots):
    resp = await client.put("/api/skills/enabled", json={"id": "user/nope", "enabled": False})
    assert resp.status_code == 404


async def test_delete_only_user_skills(client, roots):
    user = make_skill(roots.user, "mine")
    make_skill(roots.builtin, "shipped")
    assert (await client.delete("/api/skills/builtin/shipped")).status_code == 400
    assert (roots.builtin / "shipped").is_dir()
    resp = await client.delete("/api/skills/user/mine")
    assert resp.status_code == 200
    assert not user.exists()
    assert (await client.delete("/api/skills/user/mine")).status_code == 404


# ── the session token ──────────────────────────────────────────────────────

async def test_writes_without_the_session_token_are_refused(anon, roots):
    make_skill(roots.user, "mine")
    responses = [
        await anon.put("/api/skills/enabled", json={"id": "user/mine", "enabled": False}),
        await anon.post("/api/skills/install", json={"source": str(roots.user / "mine")}),
        await anon.delete("/api/skills/user/mine"),
    ]
    assert [r.status_code in (401, 403) for r in responses] == [True, True, True]
    assert (roots.user / "mine").is_dir()
    assert not (roots.user / "state.json").exists()
    # Reads stay open, like the rest of the API.
    assert (await anon.get("/api/skills")).status_code == 200


# ── agent context ──────────────────────────────────────────────────────────

async def test_context_outside_project_mode(client, roots):
    make_skill(roots.builtin, "demo")
    make_skill(roots.builtin, "broken", description=None)
    context = (await client.get("/api/agent/context")).json()
    assert context["project"] is None
    assert context["instructions"] is None
    assert [s["id"] for s in context["skills"]] == ["builtin/demo"]


async def test_context_in_project_mode(client, roots, monkeypatch):
    project = roots.project
    (project / "graphs").mkdir()
    (project / "graphs" / "train.graph.json").write_text("{}", encoding="utf-8")
    (project / "graphs" / "serve.graph.json").write_text("{}", encoding="utf-8")
    (project / "codefyui.project.toml").write_text('[project]\nname = "my-service"\n', encoding="utf-8")
    (project / "AGENTS.md").write_text("Always seed.\n", encoding="utf-8")
    monkeypatch.setattr(settings, "PROJECT_DIR", project)
    monkeypatch.setattr(settings, "GRAPHS_DIR", project / "graphs")
    context = (await client.get("/api/agent/context")).json()
    assert context["project"] == {"dir": str(project), "name": "my-service", "graphs": ["serve", "train"]}
    assert context["instructions"] == {"path": "AGENTS.md", "content": "Always seed.\n", "truncated": False}


async def test_context_instructions_are_cut_at_32_kb(client, roots, monkeypatch):
    (roots.project / "AGENTS.md").write_text("a" * (40 * 1024), encoding="utf-8")
    monkeypatch.setattr(settings, "PROJECT_DIR", roots.project)
    monkeypatch.setattr(settings, "GRAPHS_DIR", roots.project / "graphs")
    instructions = (await client.get("/api/agent/context")).json()["instructions"]
    assert instructions["truncated"] is True
    assert len(instructions["content"]) == routes_agent_context.MAX_INSTRUCTIONS_BYTES
    # A project with no manifest is named after its folder.
    project = (await client.get("/api/agent/context")).json()["project"]
    assert project["name"] == roots.project.name


# ── the built-in skills name real nodes ───────────────────────────────────

_DOTTED = __import__("re").compile(r"`([A-Z][A-Za-z0-9]+)\.([a-z_][a-z0-9_]*)`")
_BARE = __import__("re").compile(r"`([A-Z][A-Za-z0-9]+)`")
#: Backticked capitalised words in the skills that are not node types.
_NOT_NODES = {"ANY", "TENSOR", "MODEL", "DATASET", "DATALOADER", "OPTIMIZER", "LOSS_FN",
              "SCALAR", "STRING", "IMAGE", "LIST", "TRANSFORM", "TRIGGER", "Shift",
              "CrossEntropyLoss", "NLLLoss", "AdamW", "Compose"}


def _skill_texts():
    for path in sorted(REPO_SKILLS.rglob("*.md")):
        yield path, path.read_text(encoding="utf-8")


def test_every_node_port_and_param_a_builtin_skill_names_exists(registry_with_nodes):
    missing = []
    for path, text in _skill_texts():
        for node_type, member in _DOTTED.findall(text):
            cls = registry_with_nodes.get(node_type)
            if cls is None:
                missing.append(f"{path.name}: {node_type} (node)")
                continue
            names = ({p.name for p in cls.define_inputs()}
                     | {p.name for p in cls.define_outputs()}
                     | {p.name for p in cls.define_params()})
            if member not in names:
                missing.append(f"{path.name}: {node_type}.{member}")
        for word in _BARE.findall(text):
            if word in _NOT_NODES or "_" in word:
                continue
            if registry_with_nodes.get(word) is None:
                missing.append(f"{path.name}: {word} (node)")
    assert missing == []
