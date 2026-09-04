"""The Source Control tab's REST surface: ``/api/git``.

Five promises are tested here rather than in the service tests, because
each of them only exists once there is a REQUEST in front of the service:

* **the reads are open, the writes are not.** ``GET /api/git/status`` is
  what the tab polls to draw its panel, so it needs no session token; every
  route that changes the repository needs one, and gets it from the global
  ``auth_guard`` rather than from a dependency of its own -- which is why
  the walk below hits every mutating route anonymously instead of trusting
  that six handlers each remembered.
* **the response keys are a contract.** The SPA is typed from
  ``core/git/models.py``, so a renamed key is a broken panel; the shape
  tests are key-set equality rather than "contains".
* **a failure is a code, not a sentence.** The user may not read English,
  so every refusal travels as ``detail.code`` from a closed vocabulary --
  and the tests assert the code, not the prose, for exactly that reason.
* **nothing in that envelope is a credential.** git's stderr can carry one
  (a remote URL with a token in it), and the redaction that stops it is in
  the route layer, which is the only place that can be tested for it.
* **"no project open" is a screen and not an error.** It is the state most
  users see first, and answering it with a 409 would make the tab draw an
  error toast on a machine where nothing is wrong.

The repository is real in every test here: ``tmp_path``, a ``git init``,
and the service pointed at it through ``settings.PROJECT_DIR`` exactly as
the lifespan points it -- which the test client does not run, so the
service is installed by hand (the ``pack_service`` precedent in
``test_api_packs.py``). Nothing touches a network, and the developer's own
git config is kept out of it by the fixture imported below.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.api import routes_git
from app.config import settings
from app.core.auth import TOKEN_HEADER
from app.core.git.errors import GitError
from app.core.git.service import GitService
from app.main import _AUTH_EXEMPT_PREFIXES, _prefix_exempt, app

# ``Repo`` is the two-line repository helper; ``isolated_git`` and
# ``make_repo`` are fixtures, autouse and factory respectively, and are used
# by NAME below rather than by reference -- which is what pytest wants and
# what ruff cannot see.
from tests.test_git_service import (  # noqa: F401
    Repo,
    isolated_git,
    make_repo,
)

#: Every test here drives a real repository, so a machine without git has
#: nothing to say about these routes. The service tests carry the same mark,
#: for the same reason.
pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="the host has no git")

BASE_URL = f"http://127.0.0.1:{settings.PORT}"

#: A token that could not be mistaken for anything else, so a failing
#: redaction assertion says which string escaped.
TOKEN = "ghp_16C7e42F292c6912E7710c838347Ae178B4a"

#: Something worth hiding, for the tests about not serving it.
SECRET = "OPENAI_API_KEY=sk-DO-NOT-LEAK"

# Every key of every model the tab is typed against. Additive is a
# decision: a new field should break these once, on purpose.
STATUS_KEYS = {"repo", "status"}
REPO_KEYS = {"state", "project_dir", "git_version", "nested_toplevel"}
GIT_STATUS_KEYS = {"branch", "detached", "head", "unborn", "upstream", "ahead",
                   "behind", "upstream_gone", "staged", "unstaged",
                   "untracked", "conflicted", "stash_count",
                   "merge_in_progress", "rebase_in_progress"}
FILE_KEYS = {"path", "orig_path", "kind", "xy", "score"}
COMMIT_KEYS = {"sha", "short", "parents", "author_name", "author_email",
               "authored_at", "refs", "subject", "body"}
LOG_KEYS = {"commits", "has_more", "unborn"}
DIFF_KEYS = {"patch", "binary", "truncated", "old_ref", "new_ref", "old_text",
             "new_text", "old_missing", "new_missing"}
FILE_AT_REF_KEYS = {"text", "binary", "size", "truncated"}
IDENTITY_KEYS = {"name", "email", "name_scope", "email_scope"}
MUTATION_KEYS = {"status", "changed_paths", "head", "detail"}
BRANCHES_KEYS = {"current", "detached", "local", "remote"}
BRANCH_KEYS = {"name", "sha", "current", "upstream", "ahead", "behind",
               "gone", "subject", "committed_at"}
REMOTE_BRANCH_KEYS = {"name", "remote", "sha", "subject", "committed_at"}
REMOTE_KEYS = {"name", "fetch_url", "push_url"}

#: The failure envelope, which is a contract of its own: the frontend
#: switches on ``code`` and shows the rest to whoever is debugging.
ERROR_KEYS = {"code", "message", "hint", "stderr"}


def _git_service() -> GitService:
    """A service that finds the project the way the one in ``main.py`` does.

    It reads the directory through a closure over ``settings``, so the
    monkeypatched setting is what points it at a repository -- and it is
    built by hand because the lifespan that would have built it does not run
    under an ASGI transport.

    Installing it is the CALLER's job, through ``monkeypatch.setattr``:
    putting it on ``app.state`` in here would mean the state was already
    changed by the time monkeypatch read the value it is meant to put back,
    and the fixture's service would outlive its test.
    """
    return GitService(project_dir=lambda: settings.PROJECT_DIR)


@pytest.fixture
def project(make_repo, monkeypatch) -> Repo:  # noqa: F811
    """A real repository, open as the project, with a service in front of it.

    One commit and the scaffold, from the factory the service tests use, so
    the fixtures cannot drift apart.
    """
    repo = make_repo()
    monkeypatch.setattr(settings, "PROJECT_DIR", repo.root)
    monkeypatch.setattr(app.state, "git_service", _git_service(),
                        raising=False)
    return repo


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    """A project directory that is NOT a repository yet -- what init is for."""
    root = tmp_path / "fresh"
    root.mkdir()
    monkeypatch.setattr(settings, "PROJECT_DIR", root)
    monkeypatch.setattr(app.state, "git_service", _git_service(),
                        raising=False)
    return root


@pytest.fixture
async def anon_client():
    """No session token: what a browser that never bootstrapped would send."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
    ) as http:
        yield http


def _link_or_skip(target, link) -> None:
    """Make a symbolic link, or skip on a machine that will not.

    Windows needs Developer Mode or an elevated process; Linux CI always
    can, so the tests that use this run there whatever this box says.
    """
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this OS would not create a symbolic link: {exc}")


async def _detail(response) -> dict:
    """The failure envelope of a response, checked for shape as it is read."""
    detail = response.json()["detail"]
    assert set(detail) >= ERROR_KEYS, detail
    return detail


# --- the shape of the answers -----------------------------------------------


async def test_status_carries_exactly_the_documented_keys(test_client, project):
    """The tab is typed from these three models; a rename breaks it."""
    project.write("new.txt", "new\n")

    response = await test_client.get("/api/git/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == STATUS_KEYS
    assert set(body["repo"]) == REPO_KEYS
    assert body["repo"]["state"] == "ready"
    assert body["repo"]["project_dir"] == str(project.root)
    assert set(body["status"]) == GIT_STATUS_KEYS
    assert body["status"]["branch"] == "main"
    assert set(body["status"]["untracked"][0]) == FILE_KEYS


async def test_log_carries_exactly_the_documented_keys(test_client, project):
    response = await test_client.get("/api/git/log")

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == LOG_KEYS
    assert set(body["commits"][0]) == COMMIT_KEYS
    assert body["commits"][0]["subject"] == "first"
    assert body["has_more"] is False
    assert body["unborn"] is False


async def test_diff_carries_exactly_the_documented_keys(test_client, project):
    project.write("a.txt", "one\nedited\n")

    response = await test_client.get(
        "/api/git/diff", params={"path": "a.txt", "scope": "worktree"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == DIFF_KEYS
    assert "+edited" in body["patch"]
    # Not asked for, so both sides stay absent rather than shipping two
    # copies of every file on every request.
    assert body["old_text"] is None and body["new_text"] is None


async def test_a_diff_with_blobs_fills_both_sides(test_client, project):
    project.write("a.txt", "one\nedited\n")

    response = await test_client.get(
        "/api/git/diff",
        params={"path": "a.txt", "scope": "worktree", "blobs": 1})

    body = response.json()
    assert body["old_text"] == "one\ntwo\n"
    assert body["new_text"] == "one\nedited\n"


async def test_the_files_of_one_commit_are_git_file_shaped(test_client,
                                                           project):
    response = await test_client.get(
        f"/api/git/commits/{project.head()}/files")

    assert response.status_code == 200, response.text
    files = response.json()
    assert set(files[0]) == FILE_KEYS
    # The first commit is the scaffold and the fixture's own file, so this
    # is also the check that a ROOT commit is diffed against the empty tree
    # rather than skipped for having no parent.
    assert {row["path"] for row in files} == {".gitattributes", ".gitignore",
                                              "a.txt"}
    assert {row["kind"] for row in files} == {"added"}


async def test_a_file_at_a_ref_is_file_at_ref_shaped(test_client, project):
    response = await test_client.get(
        "/api/git/file", params={"path": "a.txt", "ref": "HEAD"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == FILE_AT_REF_KEYS
    assert body["text"] == "one\ntwo\n"
    assert body["binary"] is False


@pytest.mark.parametrize("ref", ["HEAD", "index", "worktree"])
async def test_a_file_reads_at_every_named_ref(test_client, project, ref):
    """Three readers underneath -- the object database twice and the
    filesystem once -- behind one route and one shape."""
    response = await test_client.get("/api/git/file",
                                     params={"path": "a.txt", "ref": ref})

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "one\ntwo\n"


async def test_a_file_reads_at_a_commit_id(test_client, project):
    """The fourth form of ``ref``: not a name, a sha."""
    response = await test_client.get(
        "/api/git/file", params={"path": "a.txt", "ref": project.head()})

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "one\ntwo\n"


async def test_a_ref_that_is_neither_a_name_nor_a_sha_is_400(test_client,
                                                             project):
    """``HEAD~3`` and ``--output=x`` are the same refusal: the grammar is
    closed, and a ref never reaches an argument list unchecked."""
    response = await test_client.get("/api/git/file",
                                     params={"path": "a.txt", "ref": "banana"})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_ref"


async def test_a_write_answers_with_the_status_it_left_behind(test_client,
                                                              project):
    """Every mutation carries a fresh status, so the tab never draws a
    panel one operation out of date."""
    project.write("new.txt", "new\n")

    response = await test_client.post("/api/git/stage",
                                      json={"paths": ["new.txt"]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == MUTATION_KEYS
    assert set(body["status"]) == GIT_STATUS_KEYS
    assert body["changed_paths"] == ["new.txt"]
    assert [entry["path"] for entry in body["status"]["staged"]] == ["new.txt"]
    assert body["status"]["untracked"] == []
    assert body["head"] == project.head()


async def test_unstage_puts_it_back(test_client, project):
    """The wiring of the other write that takes a selection."""
    project.write("new.txt", "new\n")
    await test_client.post("/api/git/stage", json={"paths": ["new.txt"]})

    response = await test_client.post("/api/git/unstage",
                                      json={"paths": ["new.txt"]})

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["status"]["staged"] == []
    assert [entry["path"] for entry in body["status"]["untracked"]] == [
        "new.txt"]


async def test_branches_carry_exactly_the_documented_keys(test_client,
                                                          project):
    """Both lists in one answer: the section draws them together."""
    project.git("branch", "feat")
    project.git("update-ref", "refs/remotes/origin/main", project.head())

    response = await test_client.get("/api/git/branches")

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == BRANCHES_KEYS
    assert body["current"] == "main" and body["detached"] is False
    assert [entry["name"] for entry in body["local"]] == ["feat", "main"]
    assert set(body["local"][0]) == BRANCH_KEYS
    assert set(body["remote"][0]) == REMOTE_BRANCH_KEYS
    assert body["remote"][0] == {
        "name": "main", "remote": "origin",
        "sha": body["local"][0]["sha"], "subject": "first",
        "committed_at": body["local"][0]["committed_at"]}


async def test_remotes_carry_exactly_the_documented_keys(test_client, project):
    project.git("remote", "add", "origin", "https://example.com/owner/repo.git")

    response = await test_client.get("/api/git/remotes")

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body[0]) == REMOTE_KEYS
    assert body[0]["name"] == "origin"
    assert body[0]["fetch_url"] == "https://example.com/owner/repo.git"


async def test_the_remotes_read_never_serves_a_credential(anon_client, project):
    """This GET needs no token -- like every read in the app, and on a
    server that is deliberately servable to a LAN. A remote URL is one of
    the two strings here that can carry a live one, so the SECRET half is
    masked before it is serialised; the username, host and path stay,
    because a row nobody can place is a row nobody can act on."""
    project.git("remote", "add", "origin",
                f"https://alice:{TOKEN}@github.com/owner/repo.git")
    project.git("remote", "add", "upstream",
                "ssh://git@github.com/other/repo.git")

    response = await anon_client.get("/api/git/remotes")

    assert response.status_code == 200, response.text
    assert TOKEN not in response.text
    listed = {entry["name"]: entry["fetch_url"] for entry in response.json()}
    assert listed["origin"] == "https://alice:***@github.com/owner/repo.git"
    # The commonest remote shape there is, through the same masking path.
    assert listed["upstream"] == "ssh://git@github.com/other/repo.git"


async def test_a_branch_write_answers_with_a_mutation_result(test_client,
                                                             project):
    """Every write in this router answers with the same four keys, so the
    tab has one code path for all of them."""
    response = await test_client.post("/api/git/branches",
                                      json={"name": "feat"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == MUTATION_KEYS
    assert body["status"]["branch"] == "feat"
    assert body["detail"] == {"branch": "feat", "checkout": True,
                              "start_point": None}


async def test_a_branch_name_with_a_slash_survives_the_url(test_client,
                                                           project):
    """``{name:path}`` is the only converter that matches ``feat/scm`` --
    and the ``%2F`` a client sends instead, which the server has already
    turned back into a slash by the time the router sees it."""
    await test_client.post("/api/git/branches",
                           json={"name": "feat/scm", "checkout": False})

    renamed = await test_client.put("/api/git/branches/feat/scm",
                                    json={"new_name": "feat/source-control"})
    assert renamed.status_code == 200, renamed.text

    deleted = await test_client.delete(
        "/api/git/branches/feat%2Fsource-control")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["detail"] == {"branch": "feat/source-control",
                                        "forced": False}


async def test_checkout_reports_the_files_it_replaced(test_client, project):
    project.git("switch", "-q", "-c", "feat")
    project.commit("on feat", {"a.txt": "feat\n"})
    project.git("switch", "-q", "main")

    response = await test_client.post("/api/git/checkout",
                                      json={"target": "feat",
                                            "kind": "local"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"]["branch"] == "feat"
    assert "a.txt" in body["changed_paths"]


async def test_a_checkout_without_a_kind_is_422(test_client, project):
    """``kind`` has no default: "the client forgot to say" and "the user
    picked local" are different requests, and the wrong one of those
    creates a branch nobody asked for."""
    response = await test_client.post("/api/git/checkout",
                                      json={"target": "feat"})

    assert response.status_code == 422, response.text


async def test_deleting_the_branch_you_are_on_is_400(test_client, project):
    response = await test_client.delete("/api/git/branches/main")

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_value"


async def test_an_unmerged_branch_needs_the_force_flag(test_client, project):
    project.git("switch", "-q", "-c", "feat")
    project.commit("only on feat", {"b.txt": "two\n"})
    project.git("switch", "-q", "main")

    refused = await test_client.delete("/api/git/branches/feat")
    assert refused.status_code == 409
    assert (await _detail(refused))["code"] == "branch_not_merged"

    forced = await test_client.delete("/api/git/branches/feat?force=true")
    assert forced.status_code == 200, forced.text
    assert forced.json()["detail"] == {"branch": "feat", "forced": True}


async def test_the_remote_routes_add_change_and_forget(test_client, project):
    added = await test_client.post(
        "/api/git/remotes",
        json={"name": "origin", "url": "https://example.com/a.git"})
    assert added.status_code == 200, added.text
    assert set(added.json()) == MUTATION_KEYS

    duplicate = await test_client.post(
        "/api/git/remotes",
        json={"name": "origin", "url": "https://example.com/b.git"})
    assert duplicate.status_code == 409
    assert (await _detail(duplicate))["code"] == "remote_exists"

    changed = await test_client.put(
        "/api/git/remotes/origin", json={"url": "https://example.com/b.git"})
    assert changed.status_code == 200, changed.text

    listed = await test_client.get("/api/git/remotes")
    assert listed.json()[0]["fetch_url"] == "https://example.com/b.git"

    removed = await test_client.delete("/api/git/remotes/origin")
    assert removed.status_code == 200, removed.text
    assert (await test_client.get("/api/git/remotes")).json() == []


async def test_a_remote_url_the_server_will_not_hand_to_git_is_400(
        test_client, project):
    response = await test_client.post(
        "/api/git/remotes",
        json={"name": "origin", "url": "ssh://-oProxyCommand=x/repo.git"})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_url"


async def test_a_branch_body_with_an_unknown_key_is_422(test_client, project):
    """``extra="forbid"``: a key nobody defined is a client bug, not an
    instruction to ignore."""
    response = await test_client.post("/api/git/branches",
                                      json={"name": "feat", "checkut": True})

    assert response.status_code == 422, response.text


async def test_the_identity_reads_and_writes_with_its_scope(test_client,
                                                            project):
    """The scope is half the answer: "this repository" or "this machine"."""
    before = await test_client.get("/api/git/config")
    assert before.status_code == 200, before.text
    assert set(before.json()) == IDENTITY_KEYS

    response = await test_client.put(
        "/api/git/config",
        json={"name": "Grace Hopper", "email": "grace@example.com"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == IDENTITY_KEYS
    assert body["name"] == "Grace Hopper"
    assert body["email"] == "grace@example.com"
    # Written --local, and read back as such: the tab never writes the
    # machine's global config from a web request.
    assert body["name_scope"] == "local"
    assert body["email_scope"] == "local"


async def test_an_identity_write_may_set_one_half(test_client, project):
    """A name without an email means what it says, and the answer is the
    identity as it now READS -- half of it still unset."""
    response = await test_client.put("/api/git/config", json={"name": "Ada"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "Ada"
    assert body["name_scope"] == "local"
    assert body["email"] is None
    assert body["email_scope"] is None


# --- "no project open" is a screen, not an error -----------------------------


async def test_status_without_a_project_is_a_200_with_a_state(test_client,
                                                              monkeypatch):
    """The first screen most people see. A 409 here would draw an error
    toast on a machine where nothing at all is wrong."""
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(app.state, "git_service", _git_service(),
                        raising=False)

    response = await test_client.get("/api/git/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["repo"]["state"] == "no_project"
    assert body["repo"]["project_dir"] is None
    assert body["status"] is None


async def test_a_write_without_a_project_is_a_409_with_a_code(test_client,
                                                              monkeypatch):
    """Every OTHER route has to fail, and with a code the tab translates."""
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(app.state, "git_service", _git_service(),
                        raising=False)

    response = await test_client.post("/api/git/stage", json={"all": True})

    assert response.status_code == 409
    detail = await _detail(response)
    assert detail["code"] == "no_project"
    assert detail["hint"]


# --- auth --------------------------------------------------------------------


async def test_every_mutating_route_needs_the_session_token(anon_client,
                                                            project):
    """Walked from the router, so a future write that forgets fails here.

    None of these routes declares an auth dependency of its own -- they are
    covered by ``auth_guard`` for being POSTs and PUTs under ``/api/`` --
    which is exactly why this asks the app rather than reading the code.
    """
    mutating = [
        (method, route.path)
        for route in routes_git.router.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods)
        if method not in {"GET", "HEAD", "OPTIONS"}
    ]
    assert len(mutating) >= 13, "router walk is broken"

    for method, path in mutating:
        response = await anon_client.request(method, path, json={})
        assert response.status_code == 403, f"{method} {path}"
        assert TOKEN_HEADER in response.json()["detail"]


async def test_the_reads_are_open_gets(anon_client, project):
    """The tab polls the status before it has a token, like every other
    read in the app."""
    response = await anon_client.get("/api/git/status")

    assert response.status_code == 200, response.text
    assert response.json()["repo"]["state"] == "ready"


async def test_the_git_router_is_not_auth_exempt():
    """An exemption would silently drop authentication from every write:
    the exempt prefixes owe a route-level dependency instead, and these
    routes have none. ``test_auth_drift.py`` pins the pair itself."""
    assert "/api/git" not in _AUTH_EXEMPT_PREFIXES
    assert not _prefix_exempt("/api/git")
    assert not _prefix_exempt("/api/git/commit")


async def test_there_is_no_loopback_gate(test_client, project, monkeypatch):
    """Deliberate, and the opposite of the Package Center: this runs the
    user's own git in the directory they opened, and access control for a
    server somebody deliberately serves to a LAN is IT's job (#247)."""
    monkeypatch.setattr(settings, "HOST", "192.168.1.20")
    project.write("new.txt", "new\n")

    response = await test_client.post("/api/git/stage", json={"all": True})

    assert response.status_code == 200, response.text


# --- the failures, as codes --------------------------------------------------


async def test_a_path_that_leaves_the_project_is_400_invalid_path(test_client,
                                                                  project):
    response = await test_client.post("/api/git/discard",
                                      json={"paths": ["../outside.txt"]})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_path"


async def test_an_unknown_commit_is_404_not_found(test_client, project):
    """Well-shaped and not there -- which is a different answer from a sha
    that is not a sha at all."""
    response = await test_client.get(f"/api/git/commits/{'0' * 40}/files")

    assert response.status_code == 404
    assert (await _detail(response))["code"] == "not_found"


async def test_a_sha_that_is_not_a_sha_is_400_invalid_ref(test_client,
                                                          project):
    response = await test_client.get("/api/git/commits/HEAD~3/files")

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_ref"


async def test_an_empty_commit_message_is_400_invalid_value(test_client,
                                                            project):
    """Refused as a code, not as pydantic's English: the message is a plain
    ``str`` on the wire for exactly this reason."""
    project.write("a.txt", "one\nedited\n")
    await test_client.post("/api/git/stage", json={"all": True})

    response = await test_client.post("/api/git/commit", json={"message": "  "})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_value"


async def test_a_commit_with_nothing_staged_is_409_nothing_to_commit(
        test_client, project):
    response = await test_client.post("/api/git/commit",
                                      json={"message": "nothing"})

    assert response.status_code == 409
    assert (await _detail(response))["code"] == "nothing_to_commit"


@pytest.mark.parametrize("path", [".env", ".env.local", "config/.env"])
async def test_a_dotenv_diff_is_403_ignored(test_client, project, path):
    """At ANY ref and whatever is on disk: a ``.env`` committed once is in
    every later tree, and this is an open GET."""
    response = await test_client.get(
        "/api/git/diff", params={"path": path, "scope": "worktree"})

    assert response.status_code == 403
    detail = await _detail(response)
    assert detail["code"] == "ignored"
    assert detail["hint"]


@pytest.mark.parametrize("ref", ["HEAD", "index", "worktree"])
async def test_a_dotenv_file_read_is_403_at_every_ref(test_client, project,
                                                      ref):
    """The secret is COMMITTED first, so every ref really holds one.

    That is the case the guard exists for: ``.gitignore`` does not un-commit
    a file, so a ``.env`` that was committed once is in HEAD, in the index
    and on disk -- and this is an open GET.
    """
    project.write(".gitignore", "# nothing is ignored here\n")
    project.commit("commit the secret", {".env": f"{SECRET}\n"})

    response = await test_client.get(
        "/api/git/file", params={"path": ".env", "ref": ref})

    assert response.status_code == 403
    assert (await _detail(response))["code"] == "ignored"
    assert SECRET not in response.text


async def test_a_second_write_while_one_runs_is_409_busy(test_client, project):
    """The lock is the whole reason the service exists, and the refusal
    names the operation holding it so the tab can say which."""
    service = app.state.git_service
    async with service.lock:
        service.current_op = "commit"
        response = await test_client.post("/api/git/stage", json={"all": True})
    service.current_op = None

    assert response.status_code == 409
    detail = await _detail(response)
    assert detail["code"] == "busy"
    assert detail["op"] == "commit"


async def test_a_ref_write_takes_the_same_lock_as_every_other(test_client,
                                                              project):
    """The branch and remote writes are ordinary mutations: one lock, one
    at a time, and a refusal that names the operation holding it."""
    service = app.state.git_service
    async with service.lock:
        service.current_op = "checkout"
        response = await test_client.post("/api/git/branches",
                                          json={"name": "feat"})
    service.current_op = None

    assert response.status_code == 409
    detail = await _detail(response)
    assert detail["code"] == "busy"
    assert detail["op"] == "checkout"


async def test_a_branch_read_does_not_queue_behind_a_write(test_client,
                                                           project):
    """The Branches section refreshes on a poll; it must not stall behind a
    checkout that is replacing half the working tree."""
    service = app.state.git_service
    async with service.lock:
        service.current_op = "checkout"
        response = await asyncio.wait_for(
            test_client.get("/api/git/branches"), timeout=5)
    service.current_op = None

    assert response.status_code == 200, response.text
    assert response.json()["current"] == "main"


async def test_a_read_does_not_queue_behind_a_running_write(test_client,
                                                            project):
    """A status poll must not stall while a commit runs the user's hooks.

    The timeout is the assertion, really: a read that DID take the mutation
    lock would block forever here, and a test that hangs is one somebody
    kills rather than reads.
    """
    service = app.state.git_service
    async with service.lock:
        service.current_op = "commit"
        response = await asyncio.wait_for(test_client.get("/api/git/status"),
                                          timeout=5)
    service.current_op = None

    assert response.status_code == 200, response.text
    assert response.json()["repo"]["state"] == "ready"


async def test_a_server_without_the_service_answers_503(test_client,
                                                        monkeypatch):
    """The lifespan builds it; a server whose startup failed has no
    repository to talk about either."""
    monkeypatch.delattr(app.state, "git_service", raising=False)

    read = await test_client.get("/api/git/status")
    write = await test_client.post("/api/git/init", json={})

    assert read.status_code == 503
    assert write.status_code == 503
    assert (await _detail(read))["code"] == "git_service_unavailable"
    assert (await _detail(write))["code"] == "git_service_unavailable"


# --- what the query string may say -------------------------------------------


@pytest.mark.parametrize("limit", [0, 101, -1])
async def test_a_page_size_out_of_range_is_422(test_client, project, limit):
    """Enforced by the signature, so it costs no code here and no process."""
    response = await test_client.get("/api/git/log", params={"limit": limit})

    assert response.status_code == 422


async def test_the_page_size_bounds_themselves_are_allowed(test_client,
                                                           project):
    for limit in (1, 100):
        response = await test_client.get("/api/git/log",
                                         params={"limit": limit})
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("skip", [
    pytest.param(-1, id="negative"),
    pytest.param(2**31, id="past-what-git-parses"),
    pytest.param(10**20, id="absurd"),
])
async def test_a_skip_outside_the_range_is_422(test_client, project, skip):
    """A page past the end of a history is empty; a ``skip`` past what git
    parses is a FAILURE.

    ``--skip=`` is a signed 32-bit integer to git, and one more than that is
    "fatal: '2147483648': not an integer", exit 128 (measured on 2.53) --
    which arrived as a 500 from a GET that needs no token. The bound is in
    the signature, so it costs no process and no code here.
    """
    response = await test_client.get("/api/git/log", params={"skip": skip})

    assert response.status_code == 422


async def test_the_largest_skip_git_accepts_is_still_a_page(test_client,
                                                            project):
    """The bound is git's own, so its edge has to be a 200 and not a 500."""
    response = await test_client.get("/api/git/log",
                                     params={"skip": 2**31 - 1})

    assert response.status_code == 200, response.text
    assert response.json()["commits"] == []


async def test_a_commit_diff_without_a_commit_is_400_invalid_value(test_client,
                                                                   project):
    response = await test_client.get(
        "/api/git/diff", params={"path": "a.txt", "scope": "commit"})

    assert response.status_code == 400
    detail = await _detail(response)
    assert detail["code"] == "invalid_value"
    assert detail["hint"]


async def test_an_empty_sha_reads_as_no_sha(test_client, project):
    """The tab builds this query from one form whose fields are always
    present, so ``sha=`` is what "no commit" looks like on the wire."""
    response = await test_client.get(
        "/api/git/diff",
        params={"path": "a.txt", "scope": "commit", "sha": ""})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_value"


async def test_a_worktree_diff_carrying_a_sha_is_400(test_client, project):
    """Silently ignoring it would show the user a diff of something else."""
    response = await test_client.get(
        "/api/git/diff",
        params={"path": "a.txt", "scope": "worktree",
                "sha": project.head()})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_value"


async def test_an_unknown_diff_scope_is_422(test_client, project):
    response = await test_client.get(
        "/api/git/diff", params={"path": "a.txt", "scope": "everything"})

    assert response.status_code == 422


@pytest.mark.parametrize("body", [
    pytest.param({}, id="neither"),
    pytest.param({"paths": []}, id="an-empty-selection"),
    pytest.param({"paths": ["a.txt"], "all": True}, id="both"),
    pytest.param({"all": True, "force": True}, id="a-key-nobody-defined"),
])
async def test_a_selection_that_is_not_one_of_the_two_forms_is_422(
        test_client, project, body):
    """``add -A --`` with an empty pathspec stages the WHOLE tree, so
    "nothing was selected" must never be able to become "all" further
    down."""
    response = await test_client.post("/api/git/stage", json=body)

    assert response.status_code == 422


async def test_an_identity_write_with_neither_half_is_400(test_client,
                                                          project):
    response = await test_client.put("/api/git/config", json={})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_value"


async def test_an_email_without_an_at_sign_is_400(test_client, project):
    response = await test_client.put("/api/git/config",
                                     json={"email": "grace"})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_value"


# --- nothing in the envelope is a credential ---------------------------------


async def test_the_envelope_never_carries_a_credential(test_client, project,
                                                       monkeypatch):
    """The one failure most likely to hold a working token.

    ``auth_required`` is raised with git's own stderr attached, and a URL
    somebody pasted into ``git remote add`` can carry the token inside it.
    The service is made to raise exactly that, because the alternative is a
    test that needs a real remote and a real credential.
    """
    async def refuse(**_kwargs):
        raise GitError(
            "auth_required", 409,
            f"unable to access 'https://octocat:{TOKEN}@github.com/o/r/'",
            hint=f"the remote is https://octocat:{TOKEN}@github.com/o/r/",
            stderr=f"fatal: unable to access 'https://x-access-token:{TOKEN}"
                   "@github.com/o/r/': The requested URL returned error: 403")

    monkeypatch.setattr(app.state.git_service, "log", refuse)

    response = await test_client.get("/api/git/log")

    assert response.status_code == 409
    detail = await _detail(response)
    assert detail["code"] == "auth_required"
    # The whole body, not only the field the leak was expected in.
    assert TOKEN not in response.text
    # And what is KEPT: which remote failed, and what git said about it.
    assert "github.com/o/r/" in detail["message"]
    assert "The requested URL returned error: 403" in detail["stderr"]
    assert "github.com/o/r/" in detail["hint"]


# --- a write may not travel through a link -----------------------------------


async def test_a_discard_through_a_link_is_refused_end_to_end(test_client,
                                                              project):
    """Measured before the guard existed: this deleted the real file.

    The link points INSIDE the project, so the containment check that would
    have caught one pointing out of it passes, and ``clean -f`` down it
    removed ``secrets/keys.txt``. Through the route, because that is the
    path a browser can actually take.
    """
    project.write("secrets/keys.txt", f"{SECRET}\n")
    _link_or_skip(project.root / "secrets", project.root / "notes")

    response = await test_client.post("/api/git/discard",
                                      json={"paths": ["notes/keys.txt"]})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_path"
    assert (project.root / "secrets" / "keys.txt").read_text(
        encoding="utf-8") == f"{SECRET}\n"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows'")
async def test_a_discard_through_a_junction_is_refused_end_to_end(test_client,
                                                                  project):
    """A junction is not a symbolic link and ``is_symlink`` cannot see it."""
    import subprocess

    project.write("secrets/keys.txt", f"{SECRET}\n")
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(project.root / "notes"),
         str(project.root / "secrets")], capture_output=True)
    if made.returncode != 0:
        pytest.skip("this box would not create a junction")

    response = await test_client.post("/api/git/discard",
                                      json={"paths": ["notes/keys.txt"]})

    assert response.status_code == 400
    assert (await _detail(response))["code"] == "invalid_path"
    assert (project.root / "secrets" / "keys.txt").exists()


# --- the whole thing, once ---------------------------------------------------


async def test_init_needs_no_body_at_all(test_client, empty_project):
    """There is nothing to choose, so there is nothing to send -- and a
    client that sends an empty object anyway is not corrected for it."""
    response = await test_client.post("/api/git/init")

    assert response.status_code == 200, response.text
    assert (empty_project / ".git").is_dir()


async def test_init_stage_commit_log_and_diff_over_http(test_client,
                                                        empty_project):
    """One pass through the tab's whole first session, over the wire.

    Every step is a route, and every assertion is on what a browser would
    have received: the panel after the init, the sha the commit answers
    with, the history that then has one row, and the patch of that commit
    holding the content that was written to disk.
    """
    root = empty_project

    created = await test_client.post("/api/git/init", json={})
    assert created.status_code == 200, created.text
    assert created.json()["status"]["unborn"] is True
    assert set(created.json()["detail"]["scaffold"]) == {".gitignore",
                                                         ".gitattributes"}
    assert (root / ".git").is_dir()

    (root / "hello.txt").write_bytes(b"one\ntwo\n")

    staged = await test_client.post("/api/git/stage",
                                    json={"paths": ["hello.txt"]})
    assert staged.status_code == 200, staged.text
    assert [entry["path"] for entry in staged.json()["status"]["staged"]] == [
        "hello.txt"]

    committed = await test_client.post("/api/git/commit",
                                       json={"message": "first commit"})
    assert committed.status_code == 200, committed.text
    sha = committed.json()["detail"]["sha"]
    assert committed.json()["detail"]["short"] == sha[:7]
    assert committed.json()["head"] == sha
    assert committed.json()["status"]["staged"] == []

    history = await test_client.get("/api/git/log")
    assert history.status_code == 200, history.text
    assert [row["sha"] for row in history.json()["commits"]] == [sha]
    assert history.json()["commits"][0]["subject"] == "first commit"

    # Only what was staged: the scaffold this init wrote is still untracked,
    # which is what makes this a commit of one file and not of three.
    files = await test_client.get(f"/api/git/commits/{sha}/files")
    assert [row["path"] for row in files.json()] == ["hello.txt"]

    patch = await test_client.get(
        "/api/git/diff",
        params={"path": "hello.txt", "scope": "commit", "sha": sha})
    assert patch.status_code == 200, patch.text
    assert "+one" in patch.json()["patch"]
    assert patch.json()["new_ref"] == sha

    final = await test_client.get("/api/git/status")
    assert final.json()["repo"]["state"] == "ready"
    assert final.json()["status"]["head"] == sha
    assert final.json()["status"]["unborn"] is False
