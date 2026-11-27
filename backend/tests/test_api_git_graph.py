"""Phase 4B: GET /api/git/graph -- one saved graph's logic at one ref.

v1 (a single file) and v2 (a folder) answer the same document, read from the
object database without a checkout.
"""

from __future__ import annotations

import json
import shutil

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.graph_storage import GraphStore
from app.main import app
from tests.test_api_git import _git_service  # noqa: F401
from tests.test_git_service import isolated_git, make_repo  # noqa: F401

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="the host has no git")

BASE_URL = f"http://127.0.0.1:{settings.PORT}"


@pytest.fixture
def repo(make_repo, monkeypatch, app_state):  # noqa: F811
    r = make_repo()
    monkeypatch.setattr(settings, "PROJECT_DIR", r.root)
    app_state("git_service", _git_service())
    return r


@pytest.fixture
async def http():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as c:
        yield c


def _graph(value="MNIST", extra=False):
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {"id": "ds", "type": "Dataset", "position": {"x": 5, "y": 5},
         "data": {"params": {"name": value, "code": "x = 1\n" * 100}}},
    ]
    if extra:
        nodes.append({"id": "new", "type": "Print", "position": {"x": 9, "y": 9},
                      "data": {"params": {}}})
    return {"name": "g", "description": "", "nodes": nodes,
            "edges": [{"id": "e", "source": "start", "target": "ds",
                       "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"}]}


def _store(root, fmt):
    (root / "codefyui.project.toml").write_text(
        f'[project]\nname = "p"\n[storage]\ngraph_format = "{fmt}"\n')
    return GraphStore.for_project(root)


@pytest.mark.parametrize("fmt", ["dir", "file"])
async def test_each_ref_answers_its_own_version(repo, http, fmt):
    store = _store(repo.root, fmt)
    store.write("g", _graph("A"))
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "a")
    store.write("g", _graph("B"))
    repo.git("add", "-A")
    store.write("g", _graph("C", extra=True))

    def value(body):
        return next(n for n in body["logic"]["nodes"] if n["id"] == "ds")["data"]["params"]["name"]

    answers = {}
    for ref in ("HEAD", "index", "worktree"):
        r = await http.get("/api/git/graph", params={"name": "g", "ref": ref})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["storage"] == fmt and body["repairs"] == []
        answers[ref] = value(body)
    assert answers == {"HEAD": "A", "index": "B", "worktree": "C"}
    sha = repo.git("rev-parse", "HEAD").strip()
    r = await http.get("/api/git/graph", params={"name": "g", "ref": sha})
    assert value(r.json()) == "A"
    # The code param comes back inline for a folder, as the file graph has it.
    assert r.json()["logic"]["nodes"][1]["data"]["params"]["code"] == "x = 1\n" * 100


async def test_v1_and_v2_answer_the_same_document(repo, http):
    _store(repo.root, "file").write("one", {**_graph(), "name": "same"})
    _store(repo.root, "dir").write("two", {**_graph(), "name": "same"})
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "both")
    one = (await http.get("/api/git/graph", params={"name": "one", "ref": "HEAD"})).json()
    two = (await http.get("/api/git/graph", params={"name": "two", "ref": "HEAD"})).json()
    a, b = dict(one["logic"]), dict(two["logic"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


async def test_missing_and_bad_names(repo, http):
    r = await http.get("/api/git/graph", params={"name": "nope", "ref": "HEAD"})
    assert r.status_code == 404
    r = await http.get("/api/git/graph", params={"name": "../x", "ref": "HEAD"})
    assert r.status_code == 400
    r = await http.get("/api/git/graph", params={"name": "g", "ref": "zzz"})
    assert r.status_code == 400
