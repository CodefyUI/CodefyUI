"""Phase 1 review, round 1: the security items (S1-S7, N1, N2) and the
multi-skill install (F2). Each test here fails on the code before its fix.

Links are made with ``os.symlink``; the module is skipped on Windows, where
that needs a privilege CI does not have.
"""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.plugins import git_source
from app.core.skills import install as skill_install
from app.core.skills import registry
from app.core.skills.spec import (
    DESCRIPTION_MAX, MAX_ALLOWED_TOOLS, MAX_FRONTMATTER_BYTES, MAX_METADATA_KEYS,
    MAX_METADATA_VALUE, parse_skill,
)
from app.main import app

pytestmark = pytest.mark.skipif(os.name == "nt", reason="symlinks need a privilege on Windows")

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
GIT = shutil.which("git")


def skill_md(name: str, description: str = "Fixture skill. Use when testing.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n"


def make_skill(parent: Path, name: str, **files: str) -> Path:
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(skill_md(name), encoding="utf-8")
    for rel, text in files.items():
        target = folder / rel.replace("__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return folder


class Env:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.user_data = tmp / "userdata"
        self.user = self.user_data / "skills"
        self.builtin = tmp / "builtin"
        self.project = tmp / "project"
        self.plugin = tmp / "plugin"
        self.outside = tmp / "outside"
        for d in (self.user, self.builtin, self.project / "skills",
                  self.plugin / "skills", self.outside):
            d.mkdir(parents=True)
        self.secret = self.outside / "secret.txt"
        self.secret.write_text("TOP-SECRET-KEY", encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    e = Env(tmp_path)
    monkeypatch.setattr(registry, "_user_data_root", lambda: e.user_data)
    monkeypatch.setattr(settings, "SKILLS_BUILTIN_DIR", e.builtin)
    monkeypatch.setattr(settings, "SKILLS_EXTRA_DIRS", "")
    monkeypatch.setattr(settings, "PROJECT_DIR", e.project)
    monkeypatch.setattr(settings, "GRAPHS_DIR", e.project / "graphs")
    monkeypatch.setattr(registry.plugin_loader, "iter_plugin_dirs",
                        lambda *a, **k: [("demo", e.plugin)])
    return e


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as c:
        yield c


def by_id(listing):
    return {s["id"]: s for s in listing}


# ── S1: AGENTS.md ──────────────────────────────────────────────────────────

async def test_s1_a_linked_agents_md_is_not_read(client, env):
    (env.project / "AGENTS.md").symlink_to(env.secret)
    context = (await client.get("/api/agent/context")).json()
    assert context["instructions"] is None
    assert "TOP-SECRET" not in str(context)


async def test_s1_a_real_agents_md_is_still_read(client, env):
    (env.project / "AGENTS.md").write_text("seed runs", encoding="utf-8")
    context = (await client.get("/api/agent/context")).json()
    assert context["instructions"]["content"] == "seed runs"


# ── S2: SKILL.md that is a link ────────────────────────────────────────────

@pytest.mark.parametrize("source", ["project", "user", "builtin"])
async def test_s2_a_linked_skill_md_is_invalid_and_unread(client, env, source):
    leak = env.outside / "notes.md"
    leak.write_text(skill_md("evil", "LEAKED-DESCRIPTION") + "LEAKED-BODY\n", encoding="utf-8")
    root = {"project": env.project / "skills", "user": env.user, "builtin": env.builtin}[source]
    folder = root / "evil"
    folder.mkdir()
    (folder / "SKILL.md").symlink_to(leak)
    entry = by_id((await client.get("/api/skills")).json())[f"{source}/evil"]
    assert entry["valid"] is False
    assert [i["code"] for i in entry["issues"]] == ["unsafe_link"]
    detail = (await client.get(f"/api/skills/{source}/evil")).json()
    assert "LEAKED" not in str(detail)
    context = (await client.get("/api/agent/context")).json()
    assert f"{source}/evil" not in [s["id"] for s in context["skills"]]


# ── S3: a skill folder that is a link ──────────────────────────────────────

def _outside_skill(env: Env) -> Path:
    return make_skill(env.outside, "outside-skill", **{"data.txt": "OUTSIDE-DATA"})


@pytest.mark.parametrize("source", ["project", "plugin:demo"])
async def test_s3_a_linked_skill_folder_from_a_repo_source_is_refused(client, env, source):
    target = _outside_skill(env)
    root = env.project / "skills" if source == "project" else env.plugin / "skills"
    (root / "outside-skill").symlink_to(target, target_is_directory=True)
    skill_id = f"{source}/outside-skill"
    entry = by_id((await client.get("/api/skills")).json())[skill_id]
    assert entry["valid"] is False and entry["issues"][0]["code"] == "unsafe_link"
    assert entry["file_count"] == 0
    detail = (await client.get(f"/api/skills/{skill_id}")).json()
    assert detail["body"] == "" and detail["files"] == []
    resp = await client.get("/api/skills/file", params={"id": skill_id, "path": "data.txt"})
    assert resp.status_code == 400
    assert "OUTSIDE" not in resp.text


async def test_s3_a_project_skills_root_linked_out_of_the_project_is_refused(client, env):
    shutil.rmtree(env.project / "skills")
    holder = env.outside / "skills"
    make_skill(holder, "elsewhere", **{"data.txt": "OUTSIDE-DATA"})
    (env.project / "skills").symlink_to(holder, target_is_directory=True)
    entry = by_id((await client.get("/api/skills")).json())["project/elsewhere"]
    assert entry["valid"] is False and entry["issues"][0]["code"] == "unsafe_link"


@pytest.mark.parametrize("source", ["user", "builtin"])
async def test_s3_a_linked_skill_folder_on_the_users_own_machine_is_followed(client, env, source):
    target = _outside_skill(env)
    root = env.user if source == "user" else env.builtin
    (root / "outside-skill").symlink_to(target, target_is_directory=True)
    skill_id = f"{source}/outside-skill"
    entry = by_id((await client.get("/api/skills")).json())[skill_id]
    assert entry["valid"] is True
    resp = await client.get("/api/skills/file", params={"id": skill_id, "path": "data.txt"})
    assert resp.status_code == 200 and resp.json()["content"] == "OUTSIDE-DATA"


async def test_s3_links_inside_a_project_skill_still_refused(client, env):
    folder = make_skill(env.project / "skills", "inner")
    (folder / "references").mkdir()
    (folder / "references" / "key.txt").symlink_to(env.secret)
    (folder / "assets").symlink_to(env.outside, target_is_directory=True)
    for path in ("references/key.txt", "assets/secret.txt"):
        resp = await client.get("/api/skills/file", params={"id": "project/inner", "path": path})
        assert resp.status_code == 400, path


# ── S4: YAML alias amplification and size caps ─────────────────────────────

def test_s4_aliases_are_refused():
    text = ("---\nname: amp\ndescription: d\nb: &s \"x x x\"\n"
            "allowed-tools: [" + ",".join("*s" for _ in range(1000)) + "]\n---\n")
    skill = parse_skill(text, "amp")
    assert not skill.valid
    assert [i.code for i in skill.issues] == ["bad_yaml"]
    assert skill.allowed_tools == []


def test_s4_the_review_poc_is_now_cheap():
    blob = "x " * 10_000
    text = (f"---\nname: amp\ndescription: d\nb: &s \"{blob}\"\nmetadata:\n"
            + "".join(f"  k{i}: *s\n" for i in range(200))
            + "allowed-tools: [" + ",".join("*s" for _ in range(200)) + "]\n---\nbody\n")
    skill = parse_skill(text, "amp")
    assert not skill.valid
    assert skill.metadata == {} and skill.allowed_tools == []


def test_s4_a_frontmatter_over_16_kb_is_invalid():
    text = f"---\nname: big\ndescription: d\nnotes: \"{'x' * MAX_FRONTMATTER_BYTES}\"\n---\n"
    skill = parse_skill(text, "big")
    assert not skill.valid
    assert [i.code for i in skill.issues] == ["frontmatter_too_large"]


def test_s4_collections_are_capped_and_description_cut():
    tools = " ".join(f"T{i}" for i in range(MAX_ALLOWED_TOOLS + 10))
    meta = "".join(f"  k{i}: v\n" for i in range(MAX_METADATA_KEYS + 5))
    text = (f"---\nname: caps\ndescription: {'d' * (DESCRIPTION_MAX + 50)}\n"
            f"allowed-tools: {tools}\nmetadata:\n{meta}  zlong: {'y' * (MAX_METADATA_VALUE + 9)}\n---\n")
    skill = parse_skill(text, "caps")
    assert skill.valid
    assert len(skill.allowed_tools) == MAX_ALLOWED_TOOLS
    assert len(skill.metadata) == MAX_METADATA_KEYS
    assert len(skill.description) == DESCRIPTION_MAX
    codes = {i.code for i in skill.issues}
    assert {"allowed_tools_truncated", "metadata_truncated", "description_too_long"} <= codes

    meta_long = parse_skill(f"---\nname: m\ndescription: d\nmetadata:\n  k: {'y' * 5000}\n---\n", "m")
    assert len(meta_long.metadata["k"]) == MAX_METADATA_VALUE


async def test_s4_a_poisoned_skill_can_still_be_listed_and_deleted(client, env):
    folder = env.user / "amp"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: amp\ndescription: d\nb: &s x\nallowed-tools: [*s, *s]\n---\n", encoding="utf-8")
    entry = by_id((await client.get("/api/skills")).json())["user/amp"]
    assert entry["valid"] is False
    assert (await client.delete("/api/skills/user/amp")).status_code == 200
    assert not folder.exists()


# ── S5: install and delete only from this machine ──────────────────────────

async def test_s5_install_and_delete_refused_off_loopback(client, env, monkeypatch, tmp_path):
    make_skill(env.user, "mine")
    src = make_skill(tmp_path / "src", "theirs")
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_SKILL_INSTALL", False)
    resp = await client.post("/api/skills/install", json={"source": str(src)})
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "remote_install_refused"
    resp = await client.delete("/api/skills/user/mine")
    assert resp.status_code == 403
    assert (env.user / "mine").is_dir() and not (env.user / "theirs").exists()
    # Enable/disable acts on what is already here: token only, as for plugins.
    resp = await client.put("/api/skills/enabled", json={"id": "user/mine", "enabled": False})
    assert resp.status_code == 200
    # The explicit opt-in opens it.
    monkeypatch.setattr(settings, "ALLOW_REMOTE_SKILL_INSTALL", True)
    resp = await client.post("/api/skills/install", json={"source": str(src)})
    assert resp.status_code == 200


@pytest.mark.parametrize("source", ["\\\\host\\share\\skill", "//host/share/skill",
                                    "file://host/share/skill", "FILE://evil.example/x"])
async def test_s5_network_shares_are_refused_without_being_looked_up(client, env, monkeypatch, source):
    def boom(_spec):
        raise AssertionError("a network path was looked up")
    monkeypatch.setattr(skill_install, "path_from_spec", boom)
    resp = await client.post("/api/skills/install", json={"source": source})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "not_supported"


# ── S6: deleting a linked user skill ───────────────────────────────────────

async def test_s6_deleting_a_linked_skill_removes_only_the_link(client, env):
    target = make_skill(env.outside, "kept", **{"references__r.md": "r"})
    os.chmod(target, 0o755)
    (env.user / "kept").symlink_to(target, target_is_directory=True)
    resp = await client.delete("/api/skills/user/kept")
    assert resp.status_code == 200
    assert not (env.user / "kept").is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert (target / "SKILL.md").is_file() and (target / "references" / "r.md").is_file()


def test_s6_remove_tree_never_chmods_through_a_link(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    os.chmod(target, 0o755)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        git_source._remove_tree(link)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


async def test_s6_a_failed_delete_is_not_reported_as_success(client, env, monkeypatch):
    make_skill(env.user, "stuck")
    monkeypatch.setattr(git_source, "_remove_tree", lambda path: None)
    resp = await client.delete("/api/skills/user/stuck")
    assert resp.status_code == 500
    assert resp.json()["detail"]["code"] == "delete_failed"


# ── S7 ─────────────────────────────────────────────────────────────────────

def _zip(path: Path, entries: dict[str, str | None]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in entries.items():
            if text is None:
                zf.writestr(zipfile.ZipInfo(name), "")  # a name ending in / is a folder
            else:
                zf.writestr(name, text)
    return path


async def install(client, source) -> tuple[int, dict]:
    resp = await client.post("/api/skills/install", json={"source": str(source)})
    return resp.status_code, resp.json()


async def test_s7_folders_count_toward_the_archive_cap(client, env, monkeypatch, tmp_path):
    monkeypatch.setattr(skill_install, "MAX_SOURCE_ENTRIES", 20)
    entries: dict[str, str | None] = {"SKILL.md": skill_md("dirs")}
    entries |= {f"d{i}/": None for i in range(30)}
    status, body = await install(client, _zip(tmp_path / "dirs.zip", entries))
    assert (status, body["detail"]["code"]) == (400, "too_many_files")
    assert sorted(p.name for p in env.user.iterdir()) == []


async def test_s7_folders_count_toward_the_per_skill_cap(client, env, monkeypatch, tmp_path):
    monkeypatch.setattr(skill_install, "MAX_FILES", 10)
    folder = make_skill(tmp_path / "src", "nested")
    for i in range(15):
        (folder / f"d{i}").mkdir()
    status, body = await install(client, folder)
    assert (status, body["detail"]["code"]) == (400, "too_many_files")


async def test_s7_a_tar_is_walked_one_member_at_a_time(client, env, monkeypatch, tmp_path):
    archive = tmp_path / "many.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        for i in range(200):
            info = tarfile.TarInfo(f"f{i}.txt")
            tf.addfile(info, io.BytesIO(b""))
    monkeypatch.setattr(skill_install, "MAX_SOURCE_ENTRIES", 5)
    calls = {"n": 0}
    original = tarfile.TarFile.next

    def counting(self):
        calls["n"] += 1
        return original(self)
    monkeypatch.setattr(tarfile.TarFile, "next", counting)
    status, body = await install(client, archive)
    assert (status, body["detail"]["code"]) == (400, "too_many_files")
    assert calls["n"] <= 8


@pytest.mark.parametrize("path", [".git/config", "references/.secret", ".hidden/x.md",
                                  "node_modules/pkg/index.js"])
async def test_s7_hidden_and_skipped_paths_are_not_served(client, env, path):
    make_skill(env.builtin, "demo", **{path.replace("/", "__"): "PRIVATE"})
    resp = await client.get("/api/skills/file", params={"id": "builtin/demo", "path": path})
    assert resp.status_code == 400, path
    files = (await client.get("/api/skills/builtin/demo")).json()["files"]
    assert path not in [f["path"] for f in files]


def test_s7_git_runs_with_lfs_filters_off():
    args = git_source._config_args()
    pairs = {args[i + 1] for i in range(0, len(args) - 1) if args[i] == "-c"}
    assert {"filter.lfs.smudge=", "filter.lfs.process=", "filter.lfs.required=false"} <= pairs


async def test_s7_a_second_install_while_one_runs_is_busy(client, env, tmp_path):
    src = make_skill(tmp_path / "src", "later")
    assert skill_install._install_lock.acquire(blocking=False)
    try:
        status, body = await install(client, src)
    finally:
        skill_install._install_lock.release()
    assert (status, body["detail"]["code"]) == (409, "busy")
    status, _ = await install(client, src)
    assert status == 200


# ── N1 / N2 ────────────────────────────────────────────────────────────────

async def test_n1_a_folder_without_a_skill_is_refused_before_any_copy(client, env, monkeypatch, tmp_path):
    (tmp_path / "notaskill" / "sub").mkdir(parents=True)
    (tmp_path / "notaskill" / "sub" / "file.txt").write_text("x", encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("copied before a skill was found")
    monkeypatch.setattr(skill_install, "_copy_folder", boom)
    status, body = await install(client, tmp_path / "notaskill")
    assert (status, body["detail"]["code"]) == (400, "no_skill")


async def test_n2_a_tar_with_an_absolute_member_is_refused(client, env, tmp_path):
    archive = tmp_path / "abs.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, data in (("SKILL.md", skill_md("absy").encode()), ("/tmp/x.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    status, body = await install(client, archive)
    assert (status, body["detail"]["code"]) == (400, "unsafe_member")
    assert list(env.user.iterdir()) == []


# ── F2: repositories of several skills ─────────────────────────────────────

def _repo(tmp: Path, layout: str) -> Path:
    repo = tmp / f"repo-{layout.replace('/', '-')}"
    (repo / "README.md").parent.mkdir(parents=True)
    (repo / "README.md").write_text("readme", encoding="utf-8")
    make_skill(repo / layout, "alpha")
    make_skill(repo / layout, "beta")
    return repo


@pytest.mark.parametrize("layout", ["skills", ".claude/skills", ".agents/skills"])
async def test_f2_every_skill_in_a_collection_is_installed(client, env, tmp_path, layout):
    status, body = await install(client, _repo(tmp_path, layout))
    assert status == 200, body
    assert sorted(s["name"] for s in body["installed"]) == ["alpha", "beta"]
    assert body["skipped"] == []


async def test_f2_an_archive_of_a_repository_folder_is_searched_inside(client, env, tmp_path):
    archive = _zip(tmp_path / "repo-main.zip", {
        "repo-main/README.md": "r",
        "repo-main/skills/alpha/SKILL.md": skill_md("alpha"),
        "repo-main/skills/beta/SKILL.md": skill_md("beta"),
    })
    status, body = await install(client, archive)
    assert status == 200, body
    assert sorted(s["name"] for s in body["installed"]) == ["alpha", "beta"]


async def test_f2_existing_and_invalid_skills_are_skipped_the_rest_installed(client, env, tmp_path):
    repo = _repo(tmp_path, "skills")
    broken = repo / "skills" / "gamma"
    broken.mkdir()
    (broken / "SKILL.md").write_text("---\nname: gamma\n---\n", encoding="utf-8")
    make_skill(env.user, "alpha")
    status, body = await install(client, repo)
    assert status == 200, body
    assert [s["name"] for s in body["installed"]] == ["beta"]
    assert {(s["name"], s["code"]) for s in body["skipped"]} == {
        ("alpha", "exists"), ("gamma", "invalid_skill")}
    status, body = await install(client, repo)
    assert status == 400 and body["detail"]["code"] == "nothing_installed"
    assert sorted(p.name for p in env.user.iterdir() if not p.name.startswith(".")) == ["alpha", "beta"]


async def test_f2_all_skipped_as_existing_is_409(client, env, tmp_path):
    repo = _repo(tmp_path, "skills")
    assert (await install(client, repo))[0] == 200
    status, body = await install(client, repo)
    assert status == 409 and body["detail"]["code"] == "exists"
    assert {s["code"] for s in body["detail"]["skipped"]} == {"exists"}


async def test_f2_a_collection_holding_a_link_out_is_not_followed(client, env, tmp_path):
    repo = _repo(tmp_path, "skills")
    (repo / "skills" / "linked").symlink_to(_outside_skill(env), target_is_directory=True)
    status, body = await install(client, repo)
    assert status == 200
    assert sorted(s["name"] for s in body["installed"]) == ["alpha", "beta"]


def _git(*args: str, cwd: Path) -> None:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "T",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run([GIT, *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.fixture
def collection_remote(tmp_path, monkeypatch) -> Path:
    if GIT is None:
        pytest.skip("git is not on PATH")
    work = _repo(tmp_path, "skills")
    _git("init", "-q", "-b", "main", cwd=work)
    _git("add", "-A", cwd=work)
    _git("commit", "-q", "-m", "init", cwd=work)
    bare = tmp_path / "remote.git"
    _git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    _git("push", "-q", str(bare), "HEAD:main", cwd=work)
    monkeypatch.setattr(git_source, "_remote_for", lambda url: bare.as_uri())
    monkeypatch.setattr(git_source, "ALLOWED_PROTOCOLS", ("https", "file"))
    return bare


async def test_f2_a_git_collection_installs_every_skill(client, env, collection_remote):
    status, body = await install(client, "https://github.com/someone/kohaku-frontend-skill")
    assert status == 200, body
    assert sorted(s["name"] for s in body["installed"]) == ["alpha", "beta"]


async def test_f2_a_git_subpath_installs_only_that_folder(client, env, collection_remote):
    status, body = await install(client, "https://git.example.test/a/b@main#skills/beta")
    assert status == 200, body
    assert [s["name"] for s in body["installed"]] == ["beta"]


@pytest.mark.parametrize("sub", ["../..", "/etc", "skills/../..", ".git", "nope"])
async def test_f2_a_bad_git_subpath_is_refused(client, env, collection_remote, sub):
    status, body = await install(client, f"https://git.example.test/a/b#{sub}")
    assert status == 400
    assert body["detail"]["code"] == "bad_subpath"
    assert [p for p in env.user.iterdir() if not p.name.startswith(".")] == []


async def test_f2_a_huge_clone_is_refused_by_the_source_cap(client, env, collection_remote, monkeypatch):
    monkeypatch.setattr(skill_install, "MAX_SOURCE_ENTRIES", 3)
    status, body = await install(client, "https://git.example.test/a/b")
    assert (status, body["detail"]["code"]) == (400, "too_many_files")
