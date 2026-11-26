"""``POST /api/skills/install``: a folder, an archive or an HTTPS git URL
into the user source, under caps, with nothing run and nothing left behind
when the source is refused.

The git repository is real: a bare repository in ``tmp_path`` reached
through the seam :mod:`app.core.plugins.git_source` has for tests
(``_remote_for`` and ``ALLOWED_PROTOCOLS``), the same one the plugin git
tests use. Skipped when ``git`` is not on PATH; CI has it.
"""

from __future__ import annotations

import io
import os
import shutil
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
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
GIT = shutil.which("git")

SKILL_MD = "---\nname: {name}\ndescription: A skill for tests. Use when testing.\n---\n# Steps\n"


@pytest.fixture
def user_root(tmp_path, monkeypatch) -> Path:
    data = tmp_path / "userdata"
    monkeypatch.setattr(registry, "_user_data_root", lambda: data)
    monkeypatch.setattr(settings, "SKILLS_BUILTIN_DIR", tmp_path / "no-builtin")
    monkeypatch.setattr(settings, "SKILLS_EXTRA_DIRS", "")
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(registry.plugin_loader, "iter_plugin_dirs", lambda *a, **k: [])
    return data / "skills"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as c:
        yield c


def skill_folder(parent: Path, name: str = "demo-skill", *, folder: str | None = None,
                 extra: dict[str, str] | None = None) -> Path:
    path = parent / (folder or name)
    (path / "references").mkdir(parents=True)
    (path / "SKILL.md").write_text(SKILL_MD.format(name=name), encoding="utf-8")
    (path / "references" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    for rel, text in (extra or {}).items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8")
    return path


def zip_of(entries: dict[str, str], path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return path


def tar_of(entries: dict[str, str], path: Path) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, text in entries.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def leftovers(user_root: Path) -> list[str]:
    return sorted(p.name for p in user_root.iterdir()) if user_root.exists() else []


async def install(client, source) -> tuple[int, dict]:
    resp = await client.post("/api/skills/install", json={"source": str(source)})
    return resp.status_code, resp.json()


# ── the three kinds of source ─────────────────────────────────────────────

async def test_install_from_a_folder(client, user_root, tmp_path):
    source = skill_folder(tmp_path / "src")
    status, body = await install(client, source)
    assert status == 200, body
    assert body["id"] == "user/demo-skill"
    assert body["valid"] is True and body["file_count"] == 1
    assert (user_root / "demo-skill" / "references" / "guide.md").is_file()
    assert source.is_dir()  # a copy, not a move
    listing = (await client.get("/api/skills")).json()
    assert [s["id"] for s in listing] == ["user/demo-skill"]


async def test_install_from_a_zip_of_the_folder_or_its_contents(client, user_root, tmp_path):
    folder_zip = zip_of({"demo-skill/SKILL.md": SKILL_MD.format(name="demo-skill"),
                         "demo-skill/references/r.md": "r",
                         "__MACOSX/demo-skill/._SKILL.md": "junk"}, tmp_path / "a.zip")
    status, body = await install(client, folder_zip)
    assert status == 200, body
    contents_zip = zip_of({"SKILL.md": SKILL_MD.format(name="second")}, tmp_path / "b.zip")
    status, body = await install(client, contents_zip)
    assert status == 200, body
    assert sorted(p.name for p in user_root.iterdir()) == ["demo-skill", "second"]


async def test_install_from_a_tarball(client, user_root, tmp_path):
    archive = tar_of({"tar-skill/SKILL.md": SKILL_MD.format(name="tar-skill")}, tmp_path / "s.tar.gz")
    status, body = await install(client, archive)
    assert status == 200, body
    assert (user_root / "tar-skill" / "SKILL.md").is_file()


def _git(*args: str, cwd: Path) -> None:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "T",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run([GIT, *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.fixture
def bare_repo(tmp_path, monkeypatch) -> Path:
    if GIT is None:
        pytest.skip("git is not on PATH")
    work = skill_folder(tmp_path / "work", "git-skill", folder="repo",
                        extra={"scripts/setup.sh": "touch $HOME/PWNED\n"})
    _git("init", "-q", "-b", "main", cwd=work)
    _git("add", "-A", cwd=work)
    _git("commit", "-q", "-m", "init", cwd=work)
    bare = tmp_path / "remote.git"
    _git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    _git("push", "-q", str(bare), "HEAD:main", cwd=work)
    monkeypatch.setattr(git_source, "_remote_for", lambda url: bare.as_uri())
    monkeypatch.setattr(git_source, "ALLOWED_PROTOCOLS", ("https", "file"))
    return bare


async def test_install_from_a_git_url(client, user_root, bare_repo):
    status, body = await install(client, "https://git.example.test/alice/git-skill.git")
    assert status == 200, body
    installed = user_root / "git-skill"
    assert (installed / "SKILL.md").is_file()
    assert not (installed / ".git").exists()
    # The script is copied as text and never run.
    assert (installed / "scripts" / "setup.sh").is_file()
    assert body["has_scripts"] is True


async def test_install_from_github_com_goes_through_git_too(client, user_root, bare_repo):
    status, body = await install(client, "https://github.com/alice/git-skill@main")
    assert status == 200, body


async def test_a_failing_clone_is_refused_and_redacted(client, user_root, monkeypatch):
    def fail(*_a, **_k):
        raise git_source.GitError(
            "git could not clone https://user:ghp_abcdefghijklmnopqrstuvwxyz0123456789@git.example.test/a/b.",
            reason="not_found",
            hint="fatal: https://user:ghp_abcdefghijklmnopqrstuvwxyz0123456789@git.example.test/a/b not found")
    monkeypatch.setattr(git_source, "clone", fail)
    status, body = await install(client, "https://git.example.test/a/b")
    assert status == 400
    text = str(body)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in text
    assert body["detail"]["code"] == "git_not_found"
    assert leftovers(user_root) == []


# ── refusals ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("member", ["../evil.txt", "demo/../../evil.txt", "/abs/evil.txt",
                                    "C:/evil.txt", "..\\evil.txt"])
async def test_zip_slip_is_refused(client, user_root, tmp_path, member):
    archive = zip_of({"SKILL.md": SKILL_MD.format(name="slip"), member: "pwned"}, tmp_path / "slip.zip")
    status, body = await install(client, archive)
    assert status == 400, body
    assert body["detail"]["code"] == "unsafe_member"
    assert not list(tmp_path.rglob("evil.txt"))
    assert leftovers(user_root) == []


async def test_a_tar_with_traversal_or_links_is_refused(client, user_root, tmp_path):
    for name, maker in (("t1.tar.gz", "traversal"), ("t2.tar.gz", "symlink")):
        archive = tmp_path / name
        with tarfile.open(archive, "w:gz") as tf:
            data = SKILL_MD.format(name="t").encode()
            info = tarfile.TarInfo("SKILL.md")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            if maker == "traversal":
                bad = tarfile.TarInfo("../evil.txt")
                bad.size = 1
                tf.addfile(bad, io.BytesIO(b"x"))
            else:
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                tf.addfile(link)
        status, body = await install(client, archive)
        assert status == 400, (name, body)
        assert body["detail"]["code"] == "unsafe_member"
    assert not list(tmp_path.rglob("evil.txt"))
    assert leftovers(user_root) == []


@pytest.mark.parametrize("entries, code", [
    ({"README.md": "no skill here"}, "no_skill"),
    ({"a/SKILL.md": "x", "b/SKILL.md": "y"}, "no_skill"),
    ({"SKILL.md": "# no frontmatter\n"}, "invalid_skill"),
    ({"SKILL.md": "---\nname: x\n---\n"}, "invalid_skill"),
    ({"SKILL.md": "---\nname: Bad Name\ndescription: d\n---\n"}, "invalid_name"),
    ({"SKILL.md": "---\ndescription: d\n---\n"}, "invalid_name"),
])
async def test_a_source_without_a_valid_skill_is_refused_and_leaves_nothing(
        client, user_root, tmp_path, entries, code):
    status, body = await install(client, zip_of(entries, tmp_path / "x.zip"))
    assert status == 400, body
    assert body["detail"]["code"] == code
    assert leftovers(user_root) == []


async def test_a_name_already_installed_is_409_and_not_overwritten(client, user_root, tmp_path):
    status, _ = await install(client, skill_folder(tmp_path / "one"))
    assert status == 200
    marker = user_root / "demo-skill" / "references" / "guide.md"
    marker.write_text("mine", encoding="utf-8")
    status, body = await install(client, skill_folder(tmp_path / "two"))
    assert status == 409
    assert body["detail"]["code"] == "exists"
    assert marker.read_text(encoding="utf-8") == "mine"
    assert leftovers(user_root) == ["demo-skill"]


async def test_caps_on_files_and_size(client, user_root, tmp_path, monkeypatch):
    monkeypatch.setattr(skill_install, "MAX_FILES", 3)
    many = {"SKILL.md": SKILL_MD.format(name="many")} | {f"references/{i}.md": "x" for i in range(5)}
    status, body = await install(client, zip_of(many, tmp_path / "many.zip"))
    assert (status, body["detail"]["code"]) == (400, "too_many_files")
    folder = skill_folder(tmp_path / "f", "many", extra={f"references/{i}.md": "x" for i in range(5)})
    status, body = await install(client, folder)
    assert (status, body["detail"]["code"]) == (400, "too_many_files")

    monkeypatch.setattr(skill_install, "MAX_FILES", 500)
    monkeypatch.setattr(skill_install, "MAX_TOTAL_BYTES", 1000)
    big = {"SKILL.md": SKILL_MD.format(name="big"), "assets/blob.txt": "x" * 5000}
    status, body = await install(client, zip_of(big, tmp_path / "big.zip"))
    assert (status, body["detail"]["code"]) == (400, "too_large")
    status, body = await install(client, tar_of(big, tmp_path / "big.tgz"))
    assert (status, body["detail"]["code"]) == (400, "too_large")
    assert leftovers(user_root) == []


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
async def test_a_folder_with_a_symlink_is_refused(client, user_root, tmp_path):
    folder = skill_folder(tmp_path / "src")
    (folder / "references" / "leak.md").symlink_to("/etc/hosts")
    status, body = await install(client, folder)
    assert (status, body["detail"]["code"]) == (400, "unsafe_member")
    assert leftovers(user_root) == []


@pytest.mark.parametrize("source, code", [
    ("", "no_source"),
    ("/definitely/not/here", "not_found"),
    ("http://git.example.test/a/b", "not_supported"),
    ("git@github.com:a/b.git", "not_supported"),
])
async def test_unusable_sources(client, user_root, source, code):
    status, body = await install(client, source)
    assert status == 400
    assert body["detail"]["code"] == code


async def test_a_plain_file_is_not_an_archive(client, user_root, tmp_path):
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    status, body = await install(client, tmp_path / "notes.txt")
    assert (status, body["detail"]["code"]) == (400, "not_supported")
