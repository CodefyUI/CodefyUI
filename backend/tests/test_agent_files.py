"""Project files for an in-editor agent (round 6, H5): ``GET``/``PUT
/api/agent/files`` and ``GET /api/agent/data-files``.

The design and its review live in the program repository
(``reports/round6-files-design.md``, ``reviews/round6-files-design.md``);
each block below is one section of the test list there.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import agent_files, mods, projects
from app.core.auth import TOKEN_HEADER, session_token
from app.core.fs_paths import canonical_path
from app.main import PROJECT_HEADER, app

from .test_projects_startup import isolated  # noqa: F401  (fixture)

BASE_URL = f"http://127.0.0.1:{settings.PORT}"

NODE = (
    "from app.core.node_base import BaseNode\n\n\n"
    "class Focal(BaseNode):\n"
    "    NODE_NAME = 'AgentFocal'\n"
    "    CATEGORY = 'Loss'\n"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _symlink(link: Path, target: Path, directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")


@pytest.fixture
def project(isolated, monkeypatch):  # noqa: F811
    root = isolated / "proj"
    root.mkdir()
    (root / "README.md").write_bytes(b"# hi\n")
    root = canonical_path(root)
    monkeypatch.setattr(settings, "PROJECT_DIR", root)
    return root


@pytest.fixture
async def client(project):
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as c:
        yield c


async def put(client, path, content, expected=None, **headers):
    return await client.put("/api/agent/files", headers=headers,
                            json={"path": path, "content": content, "expected_sha256": expected})


def code(resp) -> str:
    return resp.json()["detail"]["code"]


# ── path syntax ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "..", "a/../b", "/etc/passwd", "C:/x", "C:x", "\\\\srv\\s", "a\\b", "a//b", "./a",
    "a/./b", "a:b.md", "con.md", "notes/nul.txt", "a\x00b", "a" * 1025, "", "docs/",
])
def test_bad_path_syntax_is_refused(raw):
    with pytest.raises(agent_files.AgentFileError) as caught:
        agent_files.parse_path(raw)
    assert caught.value.code == "invalid_path"


async def test_bad_syntax_over_http(client):
    resp = await client.get("/api/agent/files", params={"path": "../x"})
    assert resp.status_code == 400 and code(resp) == "invalid_path"
    resp = await put(client, "docs/../../x.md", "x")
    assert resp.status_code == 400 and code(resp) == "invalid_path"


# ── containment ──────────────────────────────────────────────────────────

async def test_a_linked_folder_or_file_is_refused_both_ways(client, project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.md").write_bytes(b"secret\n")
    _symlink(project / "notes", outside, directory=True)
    _symlink(project / "docs" / "link.md", outside / "a.md")
    for path in ("notes/a.md", "docs/link.md"):
        resp = await client.get("/api/agent/files", params={"path": path})
        assert resp.status_code == 403 and code(resp) == "path_not_readable", path
        resp = await put(client, path, "x", sha(b"secret\n"))
        assert resp.status_code == 403 and code(resp) == "path_not_writable", path
    resp = await put(client, "notes/new.md", "x")
    assert resp.status_code == 403
    assert (outside / "a.md").read_bytes() == b"secret\n"
    assert not (outside / "new.md").exists()


async def test_a_project_reached_through_a_link_works(isolated, monkeypatch):  # noqa: F811
    real = isolated / "real"
    real.mkdir()
    link = isolated / "via"
    _symlink(link, real, directory=True)
    monkeypatch.setattr(settings, "PROJECT_DIR", link)
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as c:
        resp = await put(c, "notes/a.md", "hello")
        assert resp.status_code == 200
    assert (real / "notes" / "a.md").read_bytes() == b"hello"


def test_a_windows_reparse_point_counts_as_a_link():
    class Info:
        st_mode = 0o040755
        st_file_attributes = 0x400
    assert agent_files._is_link(Info())  # type: ignore[arg-type]


# ── case ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("parts", [["MODS", "nodes", "x.py"], ["mods", "Nodes", "x.py"],
                                   ["Docs", "a.md"], ["assets", "DATA", "a.csv"]])
def test_a_category_folder_must_be_spelled_exactly(parts):
    with pytest.raises(agent_files.AgentFileError) as caught:
        agent_files.write_rule(parts)
    assert caught.value.code == "path_not_writable"
    assert "exactly" in caught.value.message


async def test_on_a_case_insensitive_disk_the_existing_spelling_wins(client, project):
    (project / "mods" / "nodes").mkdir(parents=True)
    probe = project / "MODS"
    if not probe.exists():
        pytest.skip("case-sensitive filesystem")
    resp = await put(client, "MODS/nodes/x.py", NODE)
    assert resp.status_code == 200
    assert resp.json()["path"] == "mods/nodes/x.py"
    assert resp.json()["mods_consent_required"] is True


# ── read ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    ".git/config", ".git/hooks/pre-commit", "sub/.git/x", ".env", "config/.env.local", ".ENV",
    ".ssh/id_rsa", ".aws/credentials", ".netrc", ".npmrc", ".pypirc", "server.pem", "a/tls.KEY",
    "cert.p12", "c.pfx", "app.keystore", "id_rsa.pub", "id_ed25519", "credentials.json",
    "Secrets.yaml", "hf_token.json", "notes/.hidden/x.md",
])
def test_secret_shaped_paths_are_never_read(path):
    assert agent_files.read_refusal(path.split("/")) is not None


@pytest.mark.parametrize("path", [
    ".gitignore", ".gitattributes", ".env.example", ".github/workflows/ci.yml", "README.md",
    "graphs/a.graph.json", "codefyui.toml", "mods/nodes/x.py", "notes/tokenizer.md",
])
def test_ordinary_paths_read(path):
    assert agent_files.read_refusal(path.split("/")) is None


async def test_a_secret_is_refused_over_http_with_a_reason(client, project):
    (project / "server.pem").write_bytes(b"-----BEGIN\n")
    (project / ".env").write_bytes(b"KEY=1\n")
    resp = await client.get("/api/agent/files", params={"path": "server.pem"})
    assert resp.status_code == 403 and code(resp) == "path_not_readable"
    assert "credential" in resp.json()["detail"]["message"]
    resp = await client.get("/api/agent/files", params={"path": ".env"})
    assert resp.status_code == 403


async def test_the_listing_hides_git_and_dotenv_shows_secret_names_and_omits_links(
        client, project, tmp_path):
    (project / ".git").mkdir()
    (project / ".env").write_bytes(b"K=1\n")
    (project / ".env.example").write_bytes(b"K=\n")
    (project / "id_rsa").write_bytes(b"x")
    (project / "docs").mkdir()
    _symlink(project / "elsewhere", tmp_path, directory=True)
    resp = await client.get("/api/agent/files")
    assert resp.status_code == 200
    body = resp.json()
    names = [e["path"] for e in body["entries"]]
    assert names == sorted(names)
    assert ".git" not in names and ".env" not in names
    assert {".env.example", "id_rsa", "docs", "README.md"} <= set(names)
    assert "elsewhere" not in names and body["omitted"] == 1
    assert body["path"] == ""
    assert {"path": "docs", "kind": "dir", "size": None} in body["entries"]
    assert {"path": "README.md", "kind": "file", "size": 5} in body["entries"]


async def test_a_listing_stops_at_500(client, project):
    many = project / "data"
    many.mkdir()
    for i in range(505):
        (many / f"f{i:03}.txt").write_bytes(b"")
    body = (await client.get("/api/agent/files", params={"path": "data"})).json()
    assert len(body["entries"]) == 500 and body["truncated"] is True
    assert body["entries"][0]["path"] == "data/f000.txt"


async def test_a_long_file_is_cut_on_a_character_with_the_whole_hash(client, project):
    data = ("中" * 30000).encode("utf-8")  # 90 000 bytes; 64 KB is not a multiple of 3
    (project / "notes").mkdir()
    (project / "notes" / "long.md").write_bytes(data)
    body = (await client.get("/api/agent/files", params={"path": "notes/long.md"})).json()
    assert body["truncated"] is True and body["binary"] is False
    assert body["sha256"] == sha(data) and body["size"] == len(data)
    assert set(body["content"]) == {"中"}
    assert len(body["content"].encode("utf-8")) <= agent_files.READ_CONTENT_BYTES


async def test_a_short_file_reads_whole(client, project):
    body = (await client.get("/api/agent/files", params={"path": "README.md"})).json()
    assert body == {"path": "README.md", "content": "# hi\n", "sha256": sha(b"# hi\n"),
                    "truncated": False, "size": 5, "binary": False}


async def test_a_binary_file_gives_metadata_only(client, project):
    (project / "w.bin").write_bytes(b"\x80\x00\x01")
    (project / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    for path in ("w.bin", "latin.txt"):
        body = (await client.get("/api/agent/files", params={"path": path})).json()
        assert body["binary"] is True and body["content"] is None and body["sha256"]


async def test_a_huge_file_is_not_hashed(client, project, monkeypatch):
    monkeypatch.setattr(agent_files, "HASH_LIMIT_BYTES", 10)
    (project / "big.txt").write_bytes(b"x" * 11)
    body = (await client.get("/api/agent/files", params={"path": "big.txt"})).json()
    assert body["sha256"] is None and body["content"] is None and body["size"] == 11


async def test_nothing_there_is_404(client):
    resp = await client.get("/api/agent/files", params={"path": "notes/none.md"})
    assert resp.status_code == 404 and code(resp) == "path_not_found"


# ── write ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,consent", [
    ("mods/nodes/focal.py", True), ("mods/presets/aug.json", True), ("docs/a/b/c.md", False),
    ("notes/x.yaml", False), ("data/t.csv", False), ("data/sub/rows.JSONL", False),
    ("assets/data/x.tsv", False), ("NOTES.md", False), ("docs/table.CSV", False),
])
async def test_every_allowlist_row_writes(client, project, path, consent):
    content = "{}" if path.endswith(".json") else NODE if path.endswith(".py") else "a\n"
    resp = await put(client, path, content)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"path": path, "sha256": sha(content.encode()), "created": True,
                    "mods_consent_required": consent, "agent_instructions": False}
    assert (project / path).read_bytes() == content.encode()


@pytest.mark.parametrize("path", [
    "mods/nodes/sub/x.py", "mods/nodes/__init__.py", "mods/x.py", "mods/nodes/x.PY",
    "graphs/a.graph.json", "codefyui.toml", ".gitignore", "skills/x/SKILL.md",
    "sub/README.md", "assets/models/x.txt", "data/model.pt", "data/x.pkl", "notes/run.py",
    "docs/a.ipynb", "data/noext", "notes/README", "data/samples/iris.csv",
    "data/Samples/new.csv", "assets/data/samples/x.csv", "layout/a.json", "README.txt",
])
async def test_everything_else_is_not_writable(client, project, path):
    resp = await put(client, path, "x\n")
    assert resp.status_code == 403 and code(resp) == "path_not_writable", path
    assert "Only these can be written" in resp.json()["detail"]["message"]
    assert not (project / path).exists()


async def test_data_samples_refusal_says_why(client):
    resp = await put(client, "data/samples/iris.csv", "a\n")
    assert "bundled samples" in resp.json()["detail"]["message"]


async def test_agents_md_is_flagged(client, project):
    for name in ("AGENTS.md", "agents.md"):
        expected = None
        if (project / name).exists():
            expected = sha((project / name).read_bytes())
        resp = await put(client, name, "Be careful.\n", expected)
        assert resp.status_code == 200
        assert resp.json()["agent_instructions"] is True
    resp = await put(client, "NOTES.md", "x")
    assert resp.json()["agent_instructions"] is False


async def test_crlf_is_stored_as_lf_with_the_stored_hash(client, project):
    resp = await put(client, "notes/a.md", "one\r\ntwo\rthree\n")
    assert (project / "notes" / "a.md").read_bytes() == b"one\ntwo\nthree\n"
    assert resp.json()["sha256"] == sha(b"one\ntwo\nthree\n")


async def test_content_checks(client):
    resp = await put(client, "notes/a.md", "x" * (agent_files.WRITE_LIMIT_BYTES + 1))
    assert resp.status_code == 413 and code(resp) == "file_too_large"
    resp = await put(client, "notes/a.md", "a\x00b")
    assert resp.status_code == 400 and code(resp) == "invalid_content"
    resp = await put(client, "mods/presets/p.json", "{not json")
    assert resp.status_code == 400 and code(resp) == "invalid_content"


async def test_compare_and_swap(client, project):
    first = await put(client, "notes/a.md", "one\n")
    assert first.json()["created"] is True
    again = await put(client, "notes/a.md", "two\n")
    assert again.status_code == 409 and code(again) == "file_changed"
    assert again.json()["detail"]["current_sha256"] == sha(b"one\n")
    stale = await put(client, "notes/a.md", "two\n", sha(b"zero\n"))
    assert stale.status_code == 409
    ok = await put(client, "notes/a.md", "two\n", sha(b"one\n"))
    assert ok.status_code == 200 and ok.json()["created"] is False
    gone = await put(client, "notes/missing.md", "x", sha(b"one\n"))
    assert gone.status_code == 409 and gone.json()["detail"]["current_sha256"] is None
    bad = await put(client, "notes/a.md", "x", "XYZ")
    assert bad.status_code == 400


@pytest.mark.skipif(not hasattr(os, "link"), reason="needs hard links")
async def test_a_hard_linked_target_leaves_the_other_link_alone(client, project, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"keep\n")
    (project / "notes").mkdir()
    try:
        os.link(outside, project / "notes" / "h.md")
    except OSError:
        pytest.skip("hard links unavailable here")
    resp = await put(client, "notes/h.md", "new\n", sha(b"keep\n"))
    assert resp.status_code == 200
    assert outside.read_bytes() == b"keep\n"
    assert (project / "notes" / "h.md").read_bytes() == b"new\n"


async def test_a_failed_replace_leaves_no_temporary_file(client, project, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "rename", boom)
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        await put(client, "notes/a.md", "x")
    assert [p.name for p in (project / "notes").iterdir()] == []


# ── transport ────────────────────────────────────────────────────────────

async def test_no_token_is_403(project):
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as anon:
        assert (await anon.get("/api/agent/files")).status_code == 403
        assert (await anon.get("/api/agent/data-files")).status_code == 403
        assert (await put(anon, "notes/a.md", "x")).status_code == 403
        # Unchanged: the context stays open (review Q3).
        assert (await anon.get("/api/agent/context")).status_code == 200


def test_every_agent_route_but_context_requires_the_token():
    from fastapi.routing import APIRoute

    from app.api import routes_agent_files
    from app.core.api_keys import require_session_token

    paths = set()
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api/agent/"):
            paths.add(route.path)
    assert paths >= {"/api/agent/files", "/api/agent/data-files", "/api/agent/context"}
    deps = [d.dependency for d in routes_agent_files.router.dependencies]
    assert require_session_token in deps


async def test_a_window_on_another_project_gets_409(client):
    stale = {PROJECT_HEADER: "0" * 16}
    assert code(await client.get("/api/agent/files", headers=stale)) == "project_changed"
    assert code(await client.get("/api/agent/data-files", headers=stale)) == "project_changed"
    assert code(await put(client, "notes/a.md", "x", **stale)) == "project_changed"
    good = {PROJECT_HEADER: projects.project_id(settings.PROJECT_DIR)}
    assert (await client.get("/api/agent/files", headers=good)).status_code == 200


async def test_no_project_is_409(isolated, monkeypatch):  # noqa: F811
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as c:
        assert code(await c.get("/api/agent/files")) == "no_project"
        assert code(await put(c, "notes/a.md", "x")) == "no_project"
        body = (await c.get("/api/agent/data-files")).json()
        assert body["project"] == [] and body["bundled"]


# ── mods trust ───────────────────────────────────────────────────────────

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
                        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_an_agent_node_in_an_owned_unmoved_project_asks_before_it_loads(client, project):
    (project / "mods" / "nodes").mkdir(parents=True)
    (project / "mods" / "nodes" / "mine.py").write_bytes(NODE.replace("AgentFocal", "Mine").encode())
    _git(project, "init", "-q")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "one")
    projects.mark_owned(project)
    assert mods.mods_trust_state(project)["state"] == "trusted"
    # The owner's own edit with HEAD unmoved still loads without asking.
    (project / "mods" / "nodes" / "mine.py").write_bytes(
        NODE.replace("AgentFocal", "Mine").encode() + b"# edit\n")
    assert mods.mods_trust_state(project)["state"] == "trusted"

    resp = await put(client, "mods/nodes/agent.py", NODE)
    assert resp.json()["mods_consent_required"] is True
    assert mods.mods_trust_state(project)["state"] in ("changed", "untrusted")
    try:
        status = mods.load_project_mods(project)
        assert "project:AgentFocal" not in status["nodes"]
        digest = mods.mods_digest(mods.walk_mods(project))
        consent = await client.post("/api/projects/trust-mods", json={"sha256": digest})
        assert consent.status_code == 200
        assert mods.agent_writes_pending(project) == {}
        assert "project:AgentFocal" in mods.current_status(project)["nodes"]
    finally:
        mods.unload_project_mods()


async def test_an_owned_project_without_git_asks_too(client, project):
    projects.mark_owned(project)
    await put(client, "mods/nodes/agent.py", NODE)
    assert mods.mods_trust_state(project)["state"] in ("changed", "untrusted")


async def test_an_agent_preset_alone_asks(client, project):
    projects.mark_owned(project)
    resp = await put(client, "mods/presets/aug.json", "{}")
    assert resp.json()["mods_consent_required"] is True
    assert mods.mods_trust_state(project)["state"] in ("changed", "untrusted")
    mods.record_trust(project, mods.walk_mods(project), full=False)
    assert mods.mods_trust_state(project)["state"] == "none"


async def test_the_marker_is_written_before_the_file(client, project, monkeypatch):
    def boom(*a, **k):
        raise OSError("crash")
    # Only the file's own rename fails; the trust record (os.replace) is
    # written as usual.
    if agent_files._HAS_DIR_FD:
        monkeypatch.setattr(os, "rename", boom)
    else:
        real_replace = os.replace

        def replace(src, dst, *a, **k):
            if str(dst).endswith("agent.py"):
                raise OSError("crash")
            return real_replace(src, dst, *a, **k)
        monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError):
        await put(client, "mods/nodes/agent.py", NODE)
    assert "nodes/agent.py" in mods.agent_writes_pending(project)
    assert not (project / "mods" / "nodes" / "agent.py").exists()


async def test_a_write_imports_nothing(client, project):
    from app.core.node_registry import registry

    before_modules = set(sys.modules)
    before_nodes = set(registry.list_names()) if hasattr(registry, "list_names") else None
    await put(client, "mods/nodes/agent.py", NODE + "raise SystemExit('ran')\n")
    assert set(sys.modules) == before_modules
    if before_nodes is not None:
        assert set(registry.list_names()) == before_nodes


# ── data-files ───────────────────────────────────────────────────────────

async def test_data_files(client, project, monkeypatch, tmp_path):
    from app.core.data_paths import bundled_sample_names
    from app.nodes.data.csv_reader_node import CSVReaderNode

    (project / "data").mkdir()
    (project / "data" / "t.csv").write_bytes(b"a, b,label\n1,2,x\n3,4,y\n")
    (project / "assets" / "data").mkdir(parents=True)
    (project / "assets" / "data" / "n.txt").write_bytes(b"hi\n")
    (project / "data" / "server.pem").write_bytes(b"x")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "up.csv").write_bytes(b"x\n1\n")
    monkeypatch.setattr(settings, "DATA_FILES_DIR", uploads)
    body = (await client.get("/api/agent/data-files")).json()
    assert [e["path"] for e in body["bundled"]] == bundled_sample_names()
    for entry in body["bundled"]:
        assert CSVReaderNode._resolve_path(entry["path"]).exists()
    iris = next(e for e in body["bundled"] if e["path"] == "data/samples/iris.csv")
    assert iris["kind"] == "csv" and "species" in iris["columns"] and iris["rows"] == 150
    rag = next(e for e in body["bundled"] if e["path"] == "data/samples/rag")
    assert rag["kind"] == "dir"
    assert body["project"] == [
        {"path": "data/t.csv", "size": 23, "kind": "csv", "columns": ["a", "b", "label"], "rows": 2},
        {"path": "assets/data/n.txt", "size": 3, "kind": "text"},
    ]
    assert body["uploads"][0]["path"] == "up.csv"
    assert body["truncated"] == {"bundled": False, "project": False, "uploads": False}


async def test_a_large_csv_gives_its_header_only_and_groups_cap(client, project, monkeypatch):
    monkeypatch.setattr(agent_files, "CSV_COUNT_LIMIT", 10)
    monkeypatch.setattr(agent_files, "DATA_FILES_LIMIT", 2)
    (project / "data").mkdir()
    (project / "data" / "big.csv").write_bytes(b"x,y\n" + b"1,2\n" * 10)
    (project / "data" / "c.txt").write_bytes(b"")
    (project / "data" / "d.txt").write_bytes(b"")
    body = (await client.get("/api/agent/data-files")).json()
    assert body["project"][0] == {"path": "data/big.csv", "size": 44, "kind": "csv",
                                  "columns": ["x", "y"]}
    assert len(body["project"]) == 2 and body["truncated"]["project"] is True
    assert body["truncated"]["bundled"] is True


# ── context ──────────────────────────────────────────────────────────────

async def test_the_context_reports_git_format_and_mods(client, project):
    body = (await client.get("/api/agent/context")).json()["project"]
    assert body["git"] is False
    assert body["graph_format"] == "file"
    assert body["mods"] == {"state": "none", "nodes": []}
