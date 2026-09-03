"""The Source Control service, against real repositories in ``tmp_path``.

Nothing here fakes git. Every test makes a repository, runs the real thing
against it and looks at what happened on disk -- because every bug this
layer can have is a bug about what git actually does, and a mock would
happily agree with a wrong idea of it. Three of these tests exist because
git 2.53 did not do what the plan assumed (``-m --first-parent`` does not
restrict ``diff-tree`` to the first parent; ``check-ref-format`` accepts
``refs/heads/-x``; ``git commit`` says "nothing added to commit" on stdout),
and a fake would have hidden all three.

The developer's own git is kept out of it, in both directions:
``GIT_CONFIG_GLOBAL`` points at a file in ``tmp_path``,
``GIT_CONFIG_NOSYSTEM`` turns off ``/etc/gitconfig``, and the identity comes
from ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*``. So no test can read the
machine's user.name, no test can write it, and the same test passes on a
laptop whose git is configured for a different person. Nothing here touches
a network or ``settings.PROJECT_DIR``: the service takes its project
directory as an injected callable.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from app.core.git import diff as diff_ops
from app.core.git import log as log_ops
from app.core.git import repo as repo_ops
from app.core.git import runner
from app.core.git.errors import GitBusy, GitError
from app.core.git.service import (
    GitService,
    changed_paths,
    check_ref_format,
    commit_changes,
    discard_paths,
    resolve_repo,
    stage_paths,
    unstage_paths,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="the host has no git")

#: A fixed author date, so a log assertion is about the log and not about
#: what time the test ran.
_WHEN = "1700000000 +0000"


@pytest.fixture(autouse=True)
def isolated_git(tmp_path, monkeypatch):
    """The machine's own git config is neither read nor written.

    ``GIT_CONFIG_GLOBAL`` is a file inside ``tmp_path`` (named
    ``.gitconfig``, because the scope the identity read reports is derived
    from the filename), ``GIT_CONFIG_NOSYSTEM`` removes the system file, and
    the identity comes from the environment -- so a commit works here on a
    machine where git has never been configured, and the identity tests can
    unset it deliberately.

    ``git_env`` passes these through because it starts from the process
    environment; the test below pins that, since everything else here
    depends on it.
    """
    config = tmp_path / "githome" / ".gitconfig"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[init]\n\tdefaultBranch = main\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test Author")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "author@example.com")
    monkeypatch.setenv("GIT_AUTHOR_DATE", _WHEN)
    monkeypatch.setenv("GIT_COMMITTER_DATE", _WHEN)
    # The runner's caches (where git is, what version, is core.sshCommand
    # set) are process-wide and would otherwise carry another test file's
    # fakes into these real calls.
    runner._reset_for_tests()
    yield config
    runner._reset_for_tests()


class Repo:
    """A real repository, with the two-line helpers every test needs."""

    def __init__(self, root: Path):
        self.root = root

    # -- driving git directly, for the ARRANGE half of a test ------------

    def git(self, *args: str) -> str:
        """Run git in this repository. Fails the test if git does.

        Deliberately NOT the service: a test that built its fixture with the
        code under test would pass when both halves are wrong the same way.
        """
        proc = subprocess.run(["git", "-C", str(self.root), *args],
                              capture_output=True)
        assert proc.returncode == 0, (
            f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace')}")
        return proc.stdout.decode("utf-8", errors="replace")

    def write(self, rel: str, text: str = "one\n") -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return path

    def write_bytes(self, rel: str, data: bytes) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def commit(self, message: str = "a commit",
               files: dict[str, str] | None = None) -> str:
        """Write *files*, stage everything, commit; returns the new sha."""
        for rel, text in (files or {}).items():
            self.write(rel, text)
        self.git("add", "-A", "--", ".")
        self.git("commit", "-q", "-m", message)
        return self.head()

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    @property
    def service(self) -> GitService:
        """A service pointed at this repository -- no settings involved."""
        return GitService(project_dir=lambda: self.root)


@pytest.fixture
def make_repo(tmp_path):
    """Factory: an initialised repository, optionally with a first commit."""

    def _make(name: str = "project", *, first_commit: bool = True) -> Repo:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        repo = Repo(root)
        repo.git("init", "-q")
        repo_ops.ensure_scaffold(root)
        if first_commit:
            repo.commit("first", {"a.txt": "one\ntwo\n"})
        return repo

    return _make


@pytest.fixture
def repo(make_repo) -> Repo:
    """A repository with the project scaffold and one commit."""
    return make_repo()


def _error(exc: pytest.ExceptionInfo[GitError]) -> GitError:
    return exc.value


# --- the environment the rest of this file rests on -------------------------


def test_the_isolated_config_reaches_git(repo, isolated_git):
    """``git_env`` keeps ``GIT_CONFIG_GLOBAL`` and the identity variables.

    It starts from the process environment, which is what makes the whole
    fixture above work; if that ever changed, every test here would quietly
    start reading the developer's own ``.gitconfig``.
    """
    env = runner.git_env()

    assert env["GIT_CONFIG_GLOBAL"] == str(isolated_git)
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_AUTHOR_NAME"] == "Test Author"


# --- which repository -------------------------------------------------------


async def test_no_project_directory_is_a_state_not_an_error():
    """"Nothing is open" is a screen the tab draws, not a failure."""
    info = await GitService(project_dir=lambda: None).repo_info()

    assert info.state == "no_project"
    assert info.project_dir is None


async def test_a_missing_git_is_reported_with_the_project_still_named(
        repo, monkeypatch):
    from app.core.git import service as service_module
    monkeypatch.setattr(service_module, "git_executable", lambda: None)

    info = await repo.service.repo_info()

    assert info.state == "git_missing"
    assert info.project_dir == str(repo.root)


async def test_a_git_older_than_restore_is_refused(repo, monkeypatch):
    """``restore`` and ``switch`` arrived in 2.23; unstage and discard are
    spelled with them, so an older git is a version number and not a tab."""
    from app.core.git import service as service_module
    monkeypatch.setattr(service_module, "git_version", lambda: (2, 22, 0))

    info = await repo.service.repo_info()

    assert info.state == "git_too_old"
    assert info.git_version == "2.22.0"


async def test_a_git_whose_version_cannot_be_read_is_not_trusted(
        repo, monkeypatch):
    """Something on PATH called git that will not say what it is."""
    from app.core.git import service as service_module
    monkeypatch.setattr(service_module, "git_version", lambda: None)

    info = await repo.service.repo_info()

    assert info.state == "git_too_old"
    assert info.git_version is None


async def test_a_directory_that_is_not_a_repository(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    info = await GitService(project_dir=lambda: plain).repo_info()

    assert info.state == "not_repo"
    assert info.nested_toplevel is None


async def test_a_project_inside_another_repository_is_never_operated_on(
        make_repo, tmp_path):
    """The one failure that would look like it worked.

    A project directory inside the CodefyUI checkout (or a home directory
    somebody ran ``git init`` in) resolves to THAT repository's top level,
    and every stage, discard and commit would be applied there.
    """
    outer = make_repo("outer")
    inner = outer.root / "nested" / "project"
    inner.mkdir(parents=True)

    info = await GitService(project_dir=lambda: inner).repo_info()

    assert info.state == "not_repo"
    assert Path(info.nested_toplevel).resolve() == outer.root.resolve()


async def test_a_ready_repository_reports_its_git(repo):
    info = await repo.service.repo_info()

    assert info.state == "ready"
    assert info.project_dir == str(repo.root)
    assert info.git_version and info.git_version[0].isdigit()


async def test_status_is_none_until_there_is_a_repository(tmp_path):
    """``GET /api/git/status`` answers 200 with a state, always."""
    plain = tmp_path / "plain"
    plain.mkdir()

    response = await GitService(project_dir=lambda: plain).status()

    assert response.repo.state == "not_repo"
    assert response.status is None


async def test_status_reads_every_group(repo):
    repo.write("a.txt", "one\nedited\n")
    repo.write("new.txt", "new\n")
    repo.git("add", "-A", "--", "new.txt")

    response = await repo.service.status()

    assert response.repo.state == "ready"
    status = response.status
    assert status.branch == "main"
    assert [entry.path for entry in status.staged] == ["new.txt"]
    assert [entry.path for entry in status.unstaged] == ["a.txt"]
    assert status.untracked == []
    assert not status.merge_in_progress and not status.rebase_in_progress


async def test_a_half_finished_merge_is_visible_in_the_status(repo):
    """Porcelain v2 does not mention it; ``rev-parse --git-path`` does.

    The tab disables half its buttons on this flag, so it has to be true
    exactly when ``.git/MERGE_HEAD`` exists.
    """
    repo.git("checkout", "-q", "-b", "side")
    repo.commit("side", {"a.txt": "side\n"})
    repo.git("checkout", "-q", "main")
    repo.commit("main", {"a.txt": "main\n"})
    conflicted = subprocess.run(
        ["git", "-C", str(repo.root), "merge", "--no-ff", "-m", "merge",
         "side"], capture_output=True)
    assert conflicted.returncode != 0, "the fixture was meant to conflict"

    status = (await repo.service.status()).status

    assert status.merge_in_progress is True
    assert [entry.path for entry in status.conflicted] == ["a.txt"]


async def test_an_operation_on_a_non_repository_fails_with_its_state(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(GitError) as excinfo:
        await GitService(project_dir=lambda: plain).stage(all_paths=True)

    assert _error(excinfo).code == "not_repo"
    assert _error(excinfo).status == 409


async def test_an_operation_without_a_project_says_so():
    with pytest.raises(GitError) as excinfo:
        await GitService(project_dir=lambda: None).stage(all_paths=True)

    assert _error(excinfo).code == "no_project"


@pytest.mark.parametrize("attribute,value,code,status", [
    pytest.param("git_executable", lambda: None, "git_missing", 503,
                 id="no-git"),
    pytest.param("git_version", lambda: (2, 22, 0), "git_too_old", 409,
                 id="git-too-old"),
])
async def test_an_operation_on_an_unusable_git_fails_with_its_status(
        repo, monkeypatch, attribute, value, code, status):
    """The states that are a screen for ``GET /status`` are an ERROR for
    every other route, and each carries its own HTTP status: a missing git
    is a 503 (the server is not equipped), a git too old is a 409."""
    from app.core.git import service as service_module
    monkeypatch.setattr(service_module, attribute, value)

    with pytest.raises(GitError) as excinfo:
        await repo.service.stage(all_paths=True)

    assert _error(excinfo).code == code
    assert _error(excinfo).status == status
    assert _error(excinfo).hint


# --- init and the scaffold --------------------------------------------------


async def test_init_makes_a_repository_and_hides_the_secrets(tmp_path):
    """The button on the empty-state screen. Both files, and a real repo."""
    root = tmp_path / "fresh"
    root.mkdir()
    service = GitService(project_dir=lambda: root)

    result = await service.init()

    assert (root / ".git").exists()
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").split()
    assert "*.json text eol=lf" in (root / ".gitattributes").read_text(
        encoding="utf-8")
    # A mutation always answers with a real status -- here, of a repository
    # that did not exist a moment ago.
    assert result.status.unborn is True
    assert set(result.detail["scaffold"]) == {".gitignore", ".gitattributes"}
    assert (await service.repo_info()).state == "ready"


async def test_init_appends_env_to_an_existing_gitignore(tmp_path):
    """Never rewrite the user's file; add the one line that matters.

    The append starts with a newline so a file whose last line has no
    terminator does not gain ``*.pyco.env``.
    """
    root = tmp_path / "fresh"
    root.mkdir()
    (root / ".gitignore").write_bytes(b"*.pyc")

    await GitService(project_dir=lambda: root).init()

    assert (root / ".gitignore").read_bytes() == b"*.pyc\n.env\n"


async def test_init_leaves_a_gitignore_that_already_hides_env_alone(tmp_path):
    root = tmp_path / "fresh"
    root.mkdir()
    (root / ".gitignore").write_bytes(b"# mine\n.env\n*.log\n")

    await GitService(project_dir=lambda: root).init()

    assert (root / ".gitignore").read_bytes() == b"# mine\n.env\n*.log\n"


async def test_init_never_rewrites_an_existing_gitattributes(tmp_path):
    """Its rules decide how git stores every file; ours are a default, not
    a correction."""
    root = tmp_path / "fresh"
    root.mkdir()
    (root / ".gitattributes").write_bytes(b"* -text\n")

    await GitService(project_dir=lambda: root).init()

    assert (root / ".gitattributes").read_bytes() == b"* -text\n"


async def test_init_works_inside_another_repository(make_repo):
    """The nested case is exactly what init is FOR."""
    outer = make_repo("outer")
    inner = outer.root / "project"
    inner.mkdir()
    service = GitService(project_dir=lambda: inner)

    await service.init()

    info = await service.repo_info()
    assert info.state == "ready"
    assert info.nested_toplevel is None


async def test_init_on_a_missing_directory_is_a_404(tmp_path):
    service = GitService(project_dir=lambda: tmp_path / "not-there")

    with pytest.raises(GitError) as excinfo:
        await service.init()

    assert _error(excinfo).code in ("not_found", "not_repo")


# --- stage / unstage --------------------------------------------------------


async def test_stage_names_the_files_it_moved(repo):
    repo.write("a.txt", "one\nedited\n")
    repo.write("new.txt", "new\n")

    result = await repo.service.stage(["a.txt"])

    assert [entry.path for entry in result.status.staged] == ["a.txt"]
    assert result.changed_paths == ["a.txt"]
    assert [entry.path for entry in result.status.untracked] == ["new.txt"]


async def test_stage_all_takes_the_whole_tree(repo):
    repo.write("a.txt", "one\nedited\n")
    repo.write("dir/new.txt", "new\n")

    result = await repo.service.stage(all_paths=True)

    assert sorted(entry.path for entry in result.status.staged) == [
        "a.txt", "dir/new.txt"]
    assert result.status.unstaged == []


async def test_stage_includes_a_deletion(repo):
    """"Stage this file" has to mean staging its removal too; plain ``add``
    would skip it and the commit would silently keep the file."""
    (repo.root / "a.txt").unlink()

    result = await repo.service.stage(["a.txt"])

    assert [entry.kind for entry in result.status.staged] == ["deleted"]


async def test_an_empty_selection_is_never_the_whole_tree(repo):
    """``git add -A --`` with no pathspec stages EVERYTHING. A UI bug that
    sent an empty selection must not be promoted into that."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.stage([])

    assert _error(excinfo).code == "invalid_path"


async def test_unstage_puts_a_file_back_in_the_unstaged_group(repo):
    repo.write("a.txt", "one\nedited\n")
    repo.git("add", "-A", "--", "a.txt")

    result = await repo.service.unstage(["a.txt"])

    assert result.status.staged == []
    assert [entry.path for entry in result.status.unstaged] == ["a.txt"]
    assert repo.read("a.txt") == "one\nedited\n"


async def test_unstage_of_an_mm_file_keeps_the_worktree_edit(repo):
    """``MM``: staged AND modified again. Unstaging is about the index, and
    the newer edit on disk is not ours to throw away."""
    repo.write("a.txt", "staged\n")
    repo.git("add", "-A", "--", "a.txt")
    repo.write("a.txt", "worktree\n")

    result = await repo.service.unstage(["a.txt"])

    assert result.status.staged == []
    assert [entry.xy for entry in result.status.unstaged] == [".M"]
    assert repo.read("a.txt") == "worktree\n"


async def test_unstage_all_empties_the_index(repo):
    repo.write("a.txt", "one\nedited\n")
    repo.write("new.txt", "new\n")
    repo.git("add", "-A", "--", ".")

    result = await repo.service.unstage(all_paths=True)

    assert result.status.staged == []
    assert [entry.path for entry in result.status.unstaged] == ["a.txt"]
    assert [entry.path for entry in result.status.untracked] == ["new.txt"]


async def test_unstage_on_an_unborn_branch_uses_rm_cached(make_repo):
    """There is no HEAD to restore from yet, so ``restore --staged`` and
    ``reset`` both fail; ``rm --cached`` is the spelling that works."""
    repo = make_repo(first_commit=False)
    repo.write("x.txt", "x\n")
    repo.git("add", "-A", "--", ".")

    result = await repo.service.unstage(all_paths=True)

    assert result.status.unborn is True
    assert result.status.staged == []
    assert sorted(entry.path for entry in result.status.untracked) == [
        ".gitattributes", ".gitignore", "x.txt"]
    assert (repo.root / "x.txt").exists()


async def test_unstage_one_path_on_an_unborn_branch(make_repo):
    repo = make_repo(first_commit=False)
    repo.write("x.txt", "x\n")
    repo.write("y.txt", "y\n")
    repo.git("add", "-A", "--", ".")

    result = await repo.service.unstage(["x.txt"])

    assert "y.txt" in [entry.path for entry in result.status.staged]
    assert "x.txt" in [entry.path for entry in result.status.untracked]


# --- discard ----------------------------------------------------------------


async def test_discard_restores_a_tracked_file(repo):
    repo.write("a.txt", "ruined\n")

    result = await repo.service.discard(["a.txt"])

    assert repo.read("a.txt") == "one\ntwo\n"
    assert result.status.unstaged == []
    assert result.changed_paths == ["a.txt"]


async def test_discard_deletes_an_untracked_file(repo):
    """The only thing "discard" can mean for a file with no other copy."""
    repo.write("scratch.txt", "temporary\n")

    result = await repo.service.discard(["scratch.txt"])

    assert not (repo.root / "scratch.txt").exists()
    assert result.status.untracked == []


async def test_discard_of_an_mm_file_keeps_what_was_staged(repo):
    """VS Code's rule, and the recoverable one: the index is untouched, so
    the version the user deliberately staged survives."""
    repo.write("a.txt", "staged\n")
    repo.git("add", "-A", "--", "a.txt")
    repo.write("a.txt", "worktree\n")

    result = await repo.service.discard(["a.txt"])

    assert repo.read("a.txt") == "staged\n"
    assert [entry.xy for entry in result.status.staged] == ["M."]


async def test_discard_of_a_path_with_nothing_to_discard_is_a_400(repo):
    """Either it is already clean or the status the request was built from
    is stale; both are answered by asking again, not by deleting something."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.discard(["a.txt"])

    assert _error(excinfo).code == "path_not_in_status"
    assert _error(excinfo).status == 400


async def test_discard_all_never_deletes_an_ignored_env(repo):
    """``clean -fd``, never ``-x``. The ignored files are the user's API
    keys, their virtualenv and their model weights."""
    repo.write(".env", "OPENAI_API_KEY=sk-secret\n")
    repo.write("a.txt", "ruined\n")
    repo.write("scratch/new.txt", "temporary\n")

    result = await repo.service.discard(all_paths=True)

    assert (repo.root / ".env").read_text(encoding="utf-8") == (
        "OPENAI_API_KEY=sk-secret\n")
    assert repo.read("a.txt") == "one\ntwo\n"
    assert not (repo.root / "scratch").exists()
    assert result.status.unstaged == [] and result.status.untracked == []


async def test_discard_all_before_the_first_commit_still_cleans(tmp_path):
    """The state every new repository starts in.

    ``restore --worktree -- .`` fails on an empty index -- "pathspec '.' did
    not match any file(s) known to git", which reads as a 404 -- and the
    clean after it never ran, so "discard everything" answered with an error
    and left the files where they were.
    """
    root = tmp_path / "fresh"
    root.mkdir()
    service = GitService(project_dir=lambda: root)
    await service.init()
    (root / "scratch.txt").write_text("temporary\n", encoding="utf-8")

    result = await service.discard(all_paths=True)

    assert not (root / "scratch.txt").exists()
    assert result.status.untracked == []
    assert result.status.unstaged == []
    assert result.status.unborn is True


async def test_discard_all_before_the_first_commit_restores_a_staged_edit(
        make_repo):
    """...and the restore still happens when there IS something in the index:
    an unborn branch can have a file staged and then edited again."""
    repo = make_repo(first_commit=False)
    repo.write("x.txt", "staged\n")
    repo.git("add", "-A", "--", "x.txt")
    repo.write("x.txt", "edited after staging\n")

    result = await repo.service.discard(all_paths=True)

    assert repo.read("x.txt") == "staged\n"
    assert result.status.unstaged == []


async def test_discard_refuses_a_submodule(repo):
    """``restore --worktree`` on a gitlink succeeds and does nothing at all
    (measured on git 2.53), so the tab would report a discard that did not
    happen. The changes are in the other repository."""
    sub = repo.root / "sub"
    sub.mkdir()
    inner = Repo(sub)
    inner.git("init", "-q")
    inner.commit("sub first", {"s.txt": "one\n"})
    repo.git("update-index", "--add", "--cacheinfo",
             f"160000,{inner.head()},sub")
    repo.git("commit", "-q", "-m", "add the submodule")
    inner.commit("sub second", {"s.txt": "two\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.discard(["sub"])

    assert _error(excinfo).code == "invalid_path"
    assert "submodule" in (_error(excinfo).hint or "")


async def test_changed_paths_covers_every_file_a_discard_touched(repo):
    repo.write("a.txt", "ruined\n")
    repo.write("scratch.txt", "temporary\n")

    result = await repo.service.discard(["a.txt", "scratch.txt"])

    assert result.changed_paths == ["a.txt", "scratch.txt"]
    assert result.detail["restored"] == 1
    assert result.detail["removed"] == 1


# --- commit -----------------------------------------------------------------


async def test_commit_reports_the_sha_it_made(repo):
    repo.write("a.txt", "committed\n")
    repo.git("add", "-A", "--", "a.txt")

    result = await repo.service.commit("second commit")

    assert result.detail["sha"] == repo.head()
    assert result.detail["short"] == repo.head()[:7]
    assert result.head == repo.head()
    assert result.status.staged == []
    assert result.changed_paths == ["a.txt"]


async def test_commit_all_stages_first(repo):
    repo.write("a.txt", "edited\n")
    repo.write("new.txt", "new\n")

    result = await repo.service.commit("everything", all_paths=True)

    assert result.status.staged == [] and result.status.untracked == []
    committed = log_ops.commit_files(repo.root, result.detail["sha"])
    assert sorted(entry.path for entry in committed) == ["a.txt", "new.txt"]


async def test_a_commit_message_travels_on_stdin(repo):
    """It is the one value here meant to contain newlines, and an argument
    that starts with ``-`` is an option."""
    repo.write("a.txt", "edited\n")

    await repo.service.commit("-not-an-option\n\nbody line\n", all_paths=True)

    assert repo.git("log", "-1", "--format=%s").strip() == "-not-an-option"
    assert repo.git("log", "-1", "--format=%b").strip() == "body line"


async def test_amend_replaces_the_last_commit(repo):
    first = repo.head()
    repo.write("a.txt", "amended\n")

    result = await repo.service.commit("a better message", all_paths=True,
                                       amend=True)

    assert result.detail["sha"] != first
    assert repo.git("log", "-1", "--format=%s").strip() == "a better message"
    assert repo.git("rev-list", "--count", "HEAD").strip() == "1"


async def test_amend_without_a_commit_to_amend_is_a_404(make_repo):
    """git says "You have nothing to amend", which no classification rule
    should have to know: the status already says the branch is unborn."""
    repo = make_repo(first_commit=False)
    repo.write("x.txt", "x\n")
    repo.git("add", "-A", "--", ".")

    with pytest.raises(GitError) as excinfo:
        await repo.service.commit("nothing yet", amend=True)

    assert _error(excinfo).code == "not_found"
    assert _error(excinfo).status == 404


async def test_a_commit_with_nothing_staged_is_not_a_500(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.commit("nothing to say")

    assert _error(excinfo).code == "nothing_to_commit"
    assert _error(excinfo).status == 409


async def test_a_commit_of_untracked_files_only_is_not_a_500(repo):
    """git says this one on STDOUT ("nothing added to commit but untracked
    files present"), which is why both streams are classified."""
    repo.write("new.txt", "new\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.commit("only new files")

    assert _error(excinfo).code == "nothing_to_commit"


async def test_a_commit_without_an_identity_says_which_problem_it_is(
        repo, isolated_git, monkeypatch):
    """``user.useConfigOnly`` stops git guessing an identity from the
    hostname, which is the only way this failure is the same on every
    machine -- a host with a real domain would otherwise commit happily as
    ``user@host``."""
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    isolated_git.write_text("[user]\n\tuseConfigOnly = true\n",
                            encoding="utf-8")
    repo.write("a.txt", "edited\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.commit("who am i", all_paths=True)

    assert _error(excinfo).code == "identity_missing"


# --- log --------------------------------------------------------------------


async def test_the_log_reads_every_field(repo):
    repo.git("tag", "v1")

    page = await repo.service.log(limit=5)

    assert page.unborn is False and page.has_more is False
    commit = page.commits[0]
    assert commit.sha == repo.head()
    assert commit.short == repo.head()[:7]
    assert commit.parents == []
    assert commit.author_name == "Test Author"
    assert commit.author_email == "author@example.com"
    assert commit.authored_at == 1700000000
    assert set(commit.refs) == {"HEAD -> main", "tag: v1"}
    assert commit.subject == "first"
    assert commit.body == ""


async def test_a_body_with_blank_lines_survives_the_format(repo):
    """The message is the LAST field precisely so that anything in it lands
    in the body, newlines included."""
    repo.write("a.txt", "edited\n")
    repo.git("add", "-A", "--", ".")
    repo.git("commit", "-q", "-m", "subject", "-m", "first para\n\nsecond")

    page = await repo.service.log(limit=1)

    assert page.commits[0].subject == "subject"
    assert page.commits[0].body == "first para\n\nsecond"


async def test_the_log_pages(repo):
    for index in range(4):
        repo.commit(f"commit {index}", {"a.txt": f"{index}\n"})

    first = await repo.service.log(limit=2)
    last = await repo.service.log(skip=4, limit=2)

    assert [commit.subject for commit in first.commits] == ["commit 3",
                                                            "commit 2"]
    assert first.has_more is True
    assert [commit.subject for commit in last.commits] == ["first"]
    assert last.has_more is False


async def test_the_log_of_an_unborn_branch_is_empty_not_broken(make_repo):
    page = await make_repo(first_commit=False).service.log(limit=5)

    assert page.unborn is True
    assert page.commits == [] and page.has_more is False


async def test_a_limit_outside_the_range_is_refused(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.log(limit=1000)

    assert _error(excinfo).code == "invalid_value"


# --- the files in one commit ------------------------------------------------


async def test_commit_files_of_a_root_commit_is_every_file_in_it(repo):
    files = await repo.service.commit_files(repo.head())

    assert sorted(entry.path for entry in files) == [
        ".gitattributes", ".gitignore", "a.txt"]
    assert {entry.kind for entry in files} == {"added"}
    assert {entry.xy for entry in files} == {"A."}


async def test_commit_files_of_a_merge_is_the_first_parent_diff(repo):
    """The measured bug this is here for: ``diff-tree -m --first-parent``
    prints one diff PER PARENT (git 2.53), so a merge appeared to have
    changed every file that changed on either side of it."""
    repo.git("checkout", "-q", "-b", "side")
    repo.commit("on the side", {"side.txt": "side\n"})
    repo.git("checkout", "-q", "main")
    repo.commit("on main", {"main.txt": "main\n"})
    repo.git("merge", "--no-ff", "-m", "merge side", "side")

    files = await repo.service.commit_files(repo.head())

    assert [entry.path for entry in files] == ["side.txt"]


async def test_commit_files_reports_a_rename_with_where_it_came_from(repo):
    repo.git("mv", "a.txt", "b.txt")
    repo.git("commit", "-q", "-m", "renamed")

    files = await repo.service.commit_files(repo.head())

    assert [(entry.path, entry.orig_path, entry.kind) for entry in files] == [
        ("b.txt", "a.txt", "renamed")]
    assert files[0].score == 100


async def test_commit_files_of_an_unknown_commit_is_a_404(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.commit_files("0" * 40)

    assert _error(excinfo).code == "not_found"
    assert _error(excinfo).status == 404


async def test_a_commit_id_that_is_not_one_never_reaches_git(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.commit_files("--output=/tmp/x")

    assert _error(excinfo).code == "invalid_ref"


# --- diff -------------------------------------------------------------------


async def test_the_worktree_diff_is_the_index_against_disk(repo):
    repo.write("a.txt", "one\nchanged\n")

    response = await repo.service.diff("a.txt", "worktree")

    assert "-two" in response.patch and "+changed" in response.patch
    assert (response.old_ref, response.new_ref) == ("index", "worktree")
    assert response.binary is False and response.truncated is False


async def test_the_index_diff_is_head_against_the_index(repo):
    repo.write("a.txt", "one\nstaged\n")
    repo.git("add", "-A", "--", "a.txt")

    response = await repo.service.diff("a.txt", "index")

    assert "+staged" in response.patch
    assert (response.old_ref, response.new_ref) == ("HEAD", "index")


async def test_an_untracked_file_is_diffed_against_nothing(repo):
    """``git diff`` says nothing at all about an untracked file, and an
    empty patch for a file the user can see is the worst possible answer."""
    repo.write("new.txt", "brand new\n")

    response = await repo.service.diff("new.txt", "worktree")

    assert "+brand new" in response.patch
    assert response.old_missing is True


async def test_the_commit_diff_names_both_sides(repo):
    second = repo.commit("second", {"a.txt": "one\nsecond\n"})

    response = await repo.service.diff("a.txt", "commit", sha=second)

    assert "+second" in response.patch
    assert response.old_ref == f"{second}^"
    assert response.new_ref == second


async def test_the_diff_of_a_root_commit_has_no_old_side(repo):
    response = await repo.service.diff("a.txt", "commit", sha=repo.head())

    assert "+one" in response.patch
    assert response.old_ref is None
    assert response.old_missing is True


async def test_a_binary_file_is_reported_rather_than_decoded(repo):
    header = b"\x89PNG\r\n\x1a\n"
    repo.write_bytes("logo.png", header + bytes(range(256)))
    repo.git("add", "-A", "--", ".")
    repo.git("commit", "-q", "-m", "a png")
    repo.write_bytes("logo.png", header + bytes(range(256)) * 2)

    response = await repo.service.diff("logo.png", "worktree")

    assert response.binary is True
    assert "Binary files" in response.patch


async def test_a_text_file_that_mentions_binary_files_is_not_binary(repo):
    """The marker is anchored to the start of a line; inside a hunk it
    arrives with a ``+`` in front of it."""
    repo.write("notes.md", "one\nBinary files a and b differ\n")

    response = await repo.service.diff("notes.md", "worktree")

    assert "+Binary files a and b differ" in response.patch
    assert response.binary is False


async def test_a_huge_patch_is_cut_at_the_cap(repo):
    """A megabyte of diff is a generated file or a vendored tree, and the
    tab says so instead of sending it."""
    repo.write("big.txt", "".join(f"line {index:07d}\n" for index in range(90_000)))

    response = await repo.service.diff("big.txt", "worktree")

    assert response.truncated is True
    assert len(response.patch.encode("utf-8")) <= diff_ops.MAX_PATCH_BYTES


async def test_a_diff_can_carry_both_whole_sides(repo):
    repo.write("a.txt", "one\nchanged\n")

    response = await repo.service.diff("a.txt", "worktree", blobs=True)

    assert response.old_text == "one\ntwo\n"
    assert response.new_text == "one\nchanged\n"
    assert response.old_missing is False and response.new_missing is False


async def test_the_added_side_of_a_diff_is_missing_not_empty(repo):
    """"This file was added here" is an answer, not a failure."""
    second = repo.commit("added", {"new.txt": "new\n"})

    response = await repo.service.diff("new.txt", "commit", sha=second,
                                       blobs=True)

    assert response.old_missing is True and response.old_text is None
    assert response.new_text == "new\n"


async def test_an_index_diff_before_the_first_commit_shows_the_file_as_added(
        make_repo):
    """The first thing a user does in a fresh repository: stage a file and
    click it. There is no HEAD to compare against, so git compares the index
    with the empty tree -- and the old side is missing rather than empty."""
    repo = make_repo(first_commit=False)
    repo.write("a.txt", "one\n")
    repo.git("add", "-A", "--", "a.txt")

    response = await repo.service.diff("a.txt", "index", blobs=True)

    assert "+one" in response.patch
    assert response.old_missing is True and response.old_text is None
    assert response.new_text == "one\n"


# --- a path is one file, never a pattern and never a directory --------------

#: What must never come back through a read route, whatever was asked for.
SECRET = "OPENAI_API_KEY=sk-DO-NOT-LEAK"


@pytest.fixture
def secretive(make_repo) -> Repo:
    """A repository with a committed ``.env`` at the root and one in a folder.

    Committed deliberately -- ``.gitignore`` is emptied first -- because that
    is the case the guard exists for: a secret that was committed once is in
    every later tree, and no later ignore rule takes it back out.
    """
    repo = make_repo()
    repo.write(".gitignore", "# nothing is ignored here\n")
    repo.commit("commit the secrets", {
        ".env": f"{SECRET}\n",
        "sub/.env": f"{SECRET}\n",
        "sub/notes.txt": "public\n"})
    # A worktree edit, so a worktree diff would have something to show.
    repo.write(".env", f"{SECRET}-NEWER\n")
    repo.write("sub/.env", f"{SECRET}-NEWER\n")
    return repo


@pytest.mark.parametrize("scope", ["worktree", "index", "commit"])
@pytest.mark.parametrize("path", [
    pytest.param("sub", id="a-directory"),
    pytest.param("sub/", id="a-directory-with-a-slash"),
    pytest.param("*", id="everything"),
    pytest.param("*env", id="glob-suffix"),
    pytest.param(".en?", id="glob-single-character"),
    pytest.param("[.]env", id="glob-character-class"),
    pytest.param(".", id="the-root"),
])
async def test_a_pathspec_that_is_not_one_file_never_returns_a_secret(
        secretive, path, scope):
    """The guard checks a NAME; git used to be free to expand it.

    Every string here is a pathspec that matches ``.env`` or a directory
    containing one, while passing a check on its own final segment --
    ``[.]env`` is not a dotenv filename, it is a pattern that matches one.
    Two things now stop it: ``--literal-pathspecs`` in the runner's fixed
    prefix (a path is a name), and the requirement that a diff names exactly
    one file in the scope it asks about.

    Refused, or a 200 that does not contain the secret. Nothing else.
    """
    sha = secretive.head() if scope == "commit" else None

    try:
        response = await secretive.service.diff(path, scope, sha=sha,
                                                blobs=True)
    except GitError as exc:
        assert exc.code in ("invalid_path", "not_found", "ignored")
        return

    assert SECRET not in response.patch
    assert SECRET not in (response.old_text or "")
    assert SECRET not in (response.new_text or "")


@pytest.mark.parametrize("scope", ["worktree", "index", "commit"])
async def test_a_directory_diff_says_to_ask_for_one_file(secretive, scope):
    """The specific answer for the specific mistake, in every scope."""
    sha = secretive.head() if scope == "commit" else None

    with pytest.raises(GitError) as excinfo:
        await secretive.service.diff("sub", scope, sha=sha)

    assert _error(excinfo).code == "invalid_path"
    assert "one file" in (_error(excinfo).hint or "")


async def test_an_ordinary_file_in_a_folder_still_diffs(secretive):
    """The positive control: the single-file rule must not refuse real work."""
    secretive.write("sub/notes.txt", "public\nedited\n")

    response = await secretive.service.diff("sub/notes.txt", "worktree")

    assert "+edited" in response.patch


@pytest.mark.parametrize("path", [".env/x", "sub/.env/y", ".env.local/x"])
async def test_a_path_inside_a_dotenv_directory_is_refused(repo, path):
    """One file of secrets per environment is a directory called ``.env``,
    so the guard reads EVERY segment and not only the last."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref(path, "worktree")

    assert _error(excinfo).code == "ignored"
    assert _error(excinfo).status == 403


async def test_a_glob_never_stages_more_than_the_file_it_names(repo):
    """``a[1].txt`` is a character class matching ``a1.txt``.

    The same property as ``git add -- '*'`` meaning the whole tree, in the
    only spelling that can be tested on every platform: Windows cannot hold
    a file named ``*``, but ``[`` and ``]`` are ordinary characters there.
    """
    repo.write("a1.txt", "matched by the pattern\n")
    repo.write("a[1].txt", "the file that was asked for\n")

    result = await repo.service.stage(["a[1].txt"])

    assert [entry.path for entry in result.status.staged] == ["a[1].txt"]
    assert [entry.path for entry in result.status.untracked] == ["a1.txt"]


async def test_staging_a_star_is_not_staging_everything(repo):
    """``git add -A -- '*'`` without ``--literal-pathspecs`` is "stage the
    whole tree" wearing a pathspec's clothes."""
    repo.write("new.txt", "new\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stage(["*"])

    assert _error(excinfo).code == "not_found"
    assert (await repo.service.status()).status.staged == []


# --- one name, two trees: a file on one side and a folder on the other ------


def _swap_to_a_folder(repo: Repo) -> str:
    """Commit a FILE called ``sub``, then replace it with a folder of secrets.

    The shape that defeats a one-sided check: at the second commit ``sub`` is
    a tree, at the first it is a blob, and a diff of the pathspec ``sub``
    prints everything that arrived under it.
    """
    repo.commit("sub is a file", {"sub": "just a file\n"})
    (repo.root / "sub").unlink()
    repo.commit("sub is a folder", {"sub/.env": f"{SECRET}\n",
                                    "sub/keep.txt": "keep\n"})
    return repo.head()


def _swap_to_a_file(repo: Repo) -> str:
    """The reverse: a folder of secrets, then a plain file with its name."""
    repo.commit("sub is a folder", {"sub/.env": f"{SECRET}\n",
                                    "sub/keep.txt": "keep\n"})
    shutil.rmtree(repo.root / "sub")
    repo.commit("sub is a file", {"sub": "just a file\n"})
    return repo.head()


@pytest.mark.parametrize("blobs", [False, True])
@pytest.mark.parametrize("build", [_swap_to_a_folder, _swap_to_a_file],
                         ids=["file-became-a-folder", "folder-became-a-file"])
async def test_a_commit_that_swaps_a_file_for_a_folder_is_refused(
        repo, build, blobs):
    """A blob on one side and a tree on the other is still a tree.

    Answering "one side is a blob, fine" printed the removal (or the
    arrival) of every file under the folder -- ``sub/.env`` included -- from
    a route that needs no token. With ``blobs=True`` it was worse: a 500,
    because ``cat-file blob <tree>`` is not a phrase the missing-object list
    knows.
    """
    sha = build(repo)

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("sub", "commit", sha=sha, blobs=blobs)

    assert _error(excinfo).code == "invalid_path"
    assert _error(excinfo).status == 400
    assert "one file" in (_error(excinfo).hint or "")


async def test_a_file_staged_over_a_folder_is_refused(repo):
    """A path can be a file in the INDEX and a folder in HEAD.

    ``diff --cached -- sub`` then prints the removal of everything that used
    to be under it. The status short-circuit used to accept this: ``sub`` is
    a perfectly ordinary staged file, in the index, by that name.
    """
    repo.commit("sub is a folder", {"sub/.env": f"{SECRET}\n",
                                    "sub/keep.txt": "keep\n"})
    shutil.rmtree(repo.root / "sub")
    repo.write("sub", "just a file\n")
    repo.git("add", "-A", "--", ".")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("sub", "index")

    assert _error(excinfo).code == "invalid_path"
    assert _error(excinfo).status == 400


async def test_a_folder_staged_over_a_file_is_refused(repo):
    """The mirror image, which the index cannot report as a tree at all:
    ``:0:sub`` is not an object when the index holds ``sub/...`` entries, so
    the only way to see it is to ask which names the pathspec matches."""
    repo.commit("sub is a file", {"sub": "just a file\n"})
    (repo.root / "sub").unlink()
    repo.write("sub/.env", f"{SECRET}\n")
    repo.write("sub/keep.txt", "keep\n")
    repo.git("add", "-A", "--", ".")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("sub", "index")

    assert _error(excinfo).code == "invalid_path"
    assert _error(excinfo).status == 400


@pytest.mark.parametrize("scope", ["worktree", "index", "commit"])
async def test_no_swap_shape_returns_the_secret_in_any_scope(repo, scope):
    """The catch-all: whatever the answer is, it does not contain the key."""
    sha = _swap_to_a_folder(repo)

    try:
        response = await repo.service.diff(
            "sub", scope, sha=sha if scope == "commit" else None, blobs=True)
    except GitError as exc:
        assert exc.code in ("invalid_path", "not_found", "ignored")
        return

    assert SECRET not in response.patch
    assert SECRET not in (response.old_text or "")
    assert SECRET not in (response.new_text or "")


async def test_a_deleted_file_still_diffs(repo):
    """The case the tree check must NOT catch: absent on one side, a blob on
    the other, which is every deletion in the history."""
    repo.commit("add it", {"gone.txt": "here\n"})
    (repo.root / "gone.txt").unlink()
    sha = repo.commit("delete it")

    response = await repo.service.diff("gone.txt", "commit", sha=sha,
                                       blobs=True)

    assert "-here" in response.patch
    assert response.new_missing is True
    assert response.old_text == "here\n"


async def test_reading_a_folder_at_a_ref_is_a_400_not_a_500(repo):
    """``cat-file blob HEAD:sub`` answers "bad file" -- the object is there
    and is not a file, which is the caller's mistake and not a server one."""
    repo.commit("a folder", {"sub/keep.txt": "keep\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("sub", "HEAD")

    assert _error(excinfo).code == "invalid_path"
    assert _error(excinfo).status == 400


# --- a link is a path to somewhere else -------------------------------------


def _link(target: Path, link: Path) -> None:
    """Make a symlink, or skip the test on a machine that will not.

    Windows needs Developer Mode or an elevated process to create one; Linux
    CI always can, so the tests below run there whatever this box says.
    """
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this OS would not create a symbolic link: {exc}")


async def test_a_worktree_read_never_follows_a_symbolic_link(repo):
    """The link is the filesystem's own redirection, and every check above
    it passes: ``notes.txt`` is tracked, is not ignored, and is a perfectly
    ordinary name -- while ``.gitignore`` hides what it points AT."""
    repo.write(".env", f"{SECRET}\n")
    _link(repo.root / ".env", repo.root / "notes.txt")
    repo.commit("track the link")

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("notes.txt", "worktree")

    assert _error(excinfo).code == "invalid_path"
    assert "symbolic link" in (_error(excinfo).hint or "")


async def test_a_worktree_read_through_a_linked_folder_is_refused(repo):
    """A link ANYWHERE in the path counts, not only as its last segment.

    The link points INSIDE the repository on purpose. A link that pointed
    out of it would be refused by ``validate_rel_path``'s containment check
    -- which is a different rule, and would leave this one untested.
    """
    repo.write("sub/keys.txt", f"{SECRET}\n")
    _link(repo.root / "sub", repo.root / "linked")
    repo.commit("a link to a folder inside the project")

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("linked/keys.txt", "worktree")

    assert _error(excinfo).code == "invalid_path"
    assert "symbolic link" in (_error(excinfo).hint or "")


@pytest.mark.parametrize("blobs", [False, True])
async def test_a_worktree_diff_through_a_linked_folder_is_refused(repo, blobs):
    """The diff reads the file on disk too -- both branches of it.

    An untracked path becomes ``diff --no-index -- /dev/null <path>``, which
    prints the WHOLE file; a tracked one is read to be diffed. Guarding only
    ``file_at_ref`` left this route serving what the link pointed at.
    """
    repo.write("sub/keys.txt", f"{SECRET}\n")
    _link(repo.root / "sub", repo.root / "linked")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("linked/keys.txt", "worktree", blobs=blobs)

    assert _error(excinfo).code == "invalid_path"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows'")
@pytest.mark.parametrize("blobs", [False, True])
async def test_a_worktree_diff_through_a_junction_is_refused(repo, blobs):
    """A junction is a redirection that is NOT a symbolic link.

    ``Path.is_symlink`` answers False for one and git follows it like any
    directory, so the per-component walk cannot see it -- the
    resolved-against-lexical comparison is what does. Measured on this box:
    ``mklink /J`` over a folder of secrets made ``diff`` serve them.
    """
    repo.write("secrets/keys.txt", f"{SECRET}\n")
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(repo.root / "notes"),
         str(repo.root / "secrets")], capture_output=True)
    if made.returncode != 0:
        pytest.skip("this box would not create a junction")
    assert not (repo.root / "notes").is_symlink(), (
        "a junction that is_symlink CAN see needs no second check")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("notes/keys.txt", "worktree", blobs=blobs)

    assert _error(excinfo).code == "invalid_path"


async def test_a_diff_that_would_read_a_link_is_refused_too(repo):
    """``blobs=True`` reads the worktree side through the same function."""
    repo.write(".env", f"{SECRET}\n")
    _link(repo.root / ".env", repo.root / "notes.txt")
    repo.commit("track the link")
    repo.write("other.txt", "so there is something to diff\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("notes.txt", "worktree", blobs=True)

    assert _error(excinfo).code == "invalid_path"


async def test_a_link_at_a_ref_is_its_target_path_not_its_target(repo):
    """Reading a link from the OBJECT database leaks nothing and stays
    allowed: git stores a symlink as a blob holding the path it points at."""
    repo.write(".env", f"{SECRET}\n")
    _link(repo.root / ".env", repo.root / "notes.txt")
    repo.commit("track the link")

    content = await repo.service.file_at_ref("notes.txt", "HEAD")

    assert SECRET not in content.text
    assert content.text.endswith(".env")


async def test_an_ordinary_file_beside_a_link_still_reads(repo):
    """The positive control for the link rule."""
    repo.write("plain.txt", "ordinary\n")

    assert (await repo.service.file_at_ref("plain.txt",
                                           "worktree")).text == "ordinary\n"


# --- a file in a conflict is still a file ------------------------------------


def _conflicted(repo: Repo) -> Repo:
    """Leave *repo* mid-merge, with ``a.txt`` unmerged in the index."""
    repo.commit("base", {"a.txt": "base\n"})
    repo.git("checkout", "-q", "-b", "side")
    repo.commit("side", {"a.txt": "side\n"})
    repo.git("checkout", "-q", "main")
    repo.commit("main", {"a.txt": "main\n"})
    failed = subprocess.run(
        ["git", "-C", str(repo.root), "merge", "--no-ff", "-m", "merge",
         "side"], capture_output=True)
    assert failed.returncode != 0, "the fixture was meant to conflict"
    return repo


async def test_a_conflicted_file_still_diffs_in_the_worktree(repo):
    """The regression this round is about.

    An unmerged path has no stage-0 entry and is listed once PER STAGE, so a
    check that refused "more than one index entry for this path" called the
    file a directory -- and a conflicted file is the one a user most wants
    to look at.
    """
    _conflicted(repo)

    response = await repo.service.diff("a.txt", "worktree")

    assert "<<<<<<<" in response.patch
    assert response.binary is False


async def test_a_conflicted_file_still_diffs_in_the_index(repo):
    """git's own answer for this one is a single line, and it is the truth:
    the index has three versions and no staged one."""
    _conflicted(repo)

    response = await repo.service.diff("a.txt", "index")

    assert "Unmerged path a.txt" in response.patch


async def test_a_conflicted_file_read_from_the_index_is_a_409(repo):
    """"The index version" is a question with no answer during a conflict --
    there are three of them -- so it is a state to report, not a missing
    object (which is what the 500 before this said)."""
    _conflicted(repo)

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("a.txt", "index")

    assert _error(excinfo).code == "conflict"
    assert _error(excinfo).status == 409


async def test_a_conflicted_diff_with_blobs_says_why_it_cannot(repo):
    """The documented choice: ``blobs=True`` on a conflicted file is the
    same 409, not a patch with two empty sides. A side-by-side view has no
    two sides here, and the tab can ask again without the blobs."""
    _conflicted(repo)

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("a.txt", "worktree", blobs=True)

    assert _error(excinfo).code == "conflict"


async def test_a_conflict_does_not_disturb_the_log_or_a_commits_files(repo):
    """The read paths that never look at the index at all."""
    _conflicted(repo)

    page = await repo.service.log(limit=5)
    files = await repo.service.commit_files(page.commits[0].sha)

    assert page.commits[0].subject == "main"
    assert [entry.path for entry in files] == ["a.txt"]


@pytest.mark.parametrize("scope", ["worktree", "index"])
async def test_a_directory_and_file_conflict_is_still_refused(repo, scope):
    """The shape the loosened rule must NOT let through.

    One branch makes ``sub`` a file, the other a folder of secrets; the
    merge conflicts. The set comparison is what refuses it -- the entries
    the pathspec matches are ``{sub/keys.txt}``, which is not ``{sub}`` --
    and in the worktree scope the directory on disk answers first.

    (git 2.53 does not leave both names in the index: it moves the file side
    aside as ``sub~HEAD``. So the shape that reaches this check is "entries
    UNDER the path and none at it", which is the one that leaks if allowed.)
    """
    repo.git("checkout", "-q", "-b", "folder-side")
    repo.commit("sub is a folder", {"sub/keys.txt": f"{SECRET}\n"})
    repo.git("checkout", "-q", "main")
    repo.commit("sub is a file", {"sub": "plain\n"})
    failed = subprocess.run(
        ["git", "-C", str(repo.root), "merge", "--no-ff", "-m", "merge",
         "folder-side"], capture_output=True)
    assert failed.returncode != 0, "the fixture was meant to conflict"

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("sub", scope, blobs=True)

    assert _error(excinfo).code == "invalid_path"


# --- a submodule is another repository --------------------------------------


def _with_gitlink(repo: Repo) -> Repo:
    """Register ``sub`` as a gitlink whose commit this repository HAS.

    A real submodule's commit lives in the other repository's object store,
    which this one cannot read; using a local commit keeps the test about
    the refusal rather than about a missing object.
    """
    sha = repo.head()
    repo.git("update-index", "--add", "--cacheinfo", f"160000,{sha},sub")
    return repo


@pytest.mark.parametrize("scope", ["worktree", "index"])
async def test_a_submodule_is_refused_by_name(repo, scope):
    """Deliberately, and with the same sentence ``discard`` uses: what is at
    that path is another repository, and every answer this tab could give
    would be about the wrong one. It used to be refused by accident, as "a
    directory in the index"."""
    _with_gitlink(repo)

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("sub", scope)

    assert _error(excinfo).code == "invalid_path"
    assert "submodule" in (_error(excinfo).hint or "")


async def test_a_submodule_is_refused_in_a_commit_too(repo):
    _with_gitlink(repo)
    repo.git("commit", "-q", "-m", "add the gitlink")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("sub", "commit", sha=repo.head())

    assert _error(excinfo).code == "invalid_path"
    assert "submodule" in (_error(excinfo).hint or "")


async def test_an_unknown_diff_scope_is_refused(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("a.txt", "everything")

    assert _error(excinfo).code == "invalid_value"


async def test_a_commit_diff_without_a_commit_is_refused(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.diff("a.txt", "commit")

    assert _error(excinfo).code == "invalid_value"


# --- one file at one ref ----------------------------------------------------


async def test_a_file_can_be_read_at_head_index_worktree_and_a_sha(repo):
    first = repo.head()
    repo.write("a.txt", "staged\n")
    repo.git("add", "-A", "--", "a.txt")
    repo.write("a.txt", "on disk\n")

    service = repo.service
    assert (await service.file_at_ref("a.txt", "HEAD")).text == "one\ntwo\n"
    assert (await service.file_at_ref("a.txt", "index")).text == "staged\n"
    assert (await service.file_at_ref("a.txt", "worktree")).text == "on disk\n"
    assert (await service.file_at_ref("a.txt", first)).text == "one\ntwo\n"


async def test_a_file_that_is_not_at_that_ref_is_a_404(repo):
    repo.write("new.txt", "new\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("new.txt", "HEAD")

    assert _error(excinfo).code == "not_found"
    assert _error(excinfo).status == 404


async def test_a_binary_file_comes_back_as_a_flag_not_as_mojibake(repo):
    data = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    repo.write_bytes("logo.png", data)
    repo.git("add", "-A", "--", ".")
    repo.git("commit", "-q", "-m", "a png")

    content = await repo.service.file_at_ref("logo.png", "HEAD")

    assert content.binary is True
    assert content.text == ""
    assert content.size == len(data)


async def test_a_file_past_the_cap_is_cut_and_says_how_big_it_was(
        repo, monkeypatch):
    """``size`` is what git HAD, before the cut -- so a truncated response
    can still say what it truncated. The cap is patched down rather than fed
    two megabytes, which tests the same three lines in a tenth of a second.
    """
    monkeypatch.setattr(diff_ops, "MAX_FILE_BYTES", 8)
    repo.commit("a long file", {"long.txt": "0123456789abcdef\n"})

    content = await repo.service.file_at_ref("long.txt", "HEAD")

    assert content.truncated is True
    assert content.text == "01234567"
    assert content.size == 17


async def test_a_worktree_read_of_an_untracked_file_is_allowed(repo):
    """It is part of the project the moment git lists it as untracked."""
    repo.write("new.txt", "new\n")

    assert (await repo.service.file_at_ref("new.txt", "worktree")).text == "new\n"


async def test_a_worktree_read_of_an_ignored_file_is_refused(repo):
    """``.gitignore`` is where a project says which files are not part of
    it; a read endpoint that serves them anyway makes that meaningless."""
    repo.write(".gitignore", ".env\nsecrets/\n")
    repo.write("secrets/token.txt", "hunter2\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("secrets/token.txt", "worktree")

    assert _error(excinfo).code == "ignored"
    assert _error(excinfo).status == 403


async def test_a_tracked_file_that_matches_an_ignore_rule_is_still_served(repo):
    """The common case that the ignore check must not break: a file that was
    committed before the rule existed. ``check-ignore`` consults the index
    unless it is told not to, so it answers "not ignored" for a tracked path
    (measured on git 2.53) -- which is why the tracked check does not have to
    run first."""
    repo.write("app.log", "kept\n")
    repo.git("add", "-f", "--", "app.log")
    repo.commit("track a log file", {".gitignore": "*.log\n"})

    content = await repo.service.file_at_ref("app.log", "worktree")

    assert content.text == "kept\n"


async def test_a_worktree_read_of_a_file_git_has_never_heard_of_is_a_404(repo):
    """Not tracked, not in the status: there is nothing to serve."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("nowhere.txt", "worktree")

    assert _error(excinfo).code == "not_found"


@pytest.mark.parametrize("ref", ["HEAD", "index", "worktree"])
async def test_a_dotenv_file_is_refused_at_every_ref(repo, ref):
    """A secret that was committed once is in every later tree, and
    ``.gitignore`` does not un-commit it. So the guard is on the READ."""
    repo.write(".gitignore", "# nothing is ignored here\n")
    repo.commit("commit the secret", {".env": "OPENAI_API_KEY=sk-secret\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref(".env", ref)

    assert _error(excinfo).code == "ignored"
    assert _error(excinfo).status == 403


async def test_a_dotenv_file_is_refused_by_the_diff_too(repo):
    repo.write(".gitignore", "# nothing is ignored here\n")
    repo.commit("commit the secret", {".env": "OPENAI_API_KEY=sk-secret\n"})
    repo.write(".env", "OPENAI_API_KEY=sk-newer\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.diff(".env", "worktree")

    assert _error(excinfo).code == "ignored"


async def test_a_dotenv_example_is_not_a_secret(repo):
    """It exists to be read: it is the committed template of the keys a
    project needs."""
    repo.commit("template", {".env.example": "OPENAI_API_KEY=\n"})

    content = await repo.service.file_at_ref(".env.example", "HEAD")

    assert content.text == "OPENAI_API_KEY=\n"


async def test_a_path_outside_the_project_never_reaches_git(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.file_at_ref("../../etc/passwd", "worktree")

    assert _error(excinfo).code == "invalid_path"


# --- identity ---------------------------------------------------------------


async def test_an_unset_identity_is_reported_as_unset(repo):
    """The isolated config has no ``[user]`` section, and the environment
    identity these tests commit with is not a CONFIGURED one -- which is
    exactly the state the tab has to offer to fix."""
    identity = await repo.service.identity()

    assert identity.name is None and identity.name_scope is None


async def test_a_global_identity_is_reported_as_global(repo, isolated_git):
    isolated_git.write_text(
        "[user]\n\tname = Global Person\n\temail = person@example.com\n",
        encoding="utf-8")

    identity = await repo.service.identity()

    assert identity.name == "Global Person"
    assert identity.name_scope == "global"
    assert identity.email_scope == "global"


async def test_writing_an_identity_writes_it_locally(repo, isolated_git):
    isolated_git.write_text("[user]\n\tname = Global Person\n",
                            encoding="utf-8")

    result = await repo.service.set_identity(name="Local Person",
                                             email="local@example.com")

    assert result.detail == {"name": "Local Person",
                            "email": "local@example.com"}
    identity = await repo.service.identity()
    assert identity.name == "Local Person"
    assert identity.name_scope == "local"
    # The user's own config is untouched: this only ever writes --local.
    assert "Global Person" in isolated_git.read_text(encoding="utf-8")
    assert "Local Person" not in isolated_git.read_text(encoding="utf-8")


async def test_an_identity_that_would_forge_a_header_is_refused(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.set_identity(name="Someone\nAuthor: nobody")

    assert _error(excinfo).code == "invalid_value"


# --- the mutation lock ------------------------------------------------------


async def test_a_second_mutation_while_one_runs_is_refused(repo):
    """git serialises writes with the index lock and answers the loser with
    a sentence about ``.git/index.lock``. This is the same refusal, in a
    shape the tab can show: nothing was attempted, and the operation that
    holds the lock is named.

    The parked operation waits on a ``threading.Event`` and not an
    ``asyncio`` one because ``mutate`` runs it on a worker thread -- which
    is the point of the test: the loop is free the whole time.
    """
    started = threading.Event()
    release = threading.Event()

    def parked(root):
        started.set()
        assert release.wait(10), "the parked mutation was never released"
        return {}

    service = repo.service
    running = asyncio.create_task(
        service.mutate("parked", parked, worktree=False))
    await asyncio.to_thread(started.wait, 10)

    try:
        with pytest.raises(GitBusy) as excinfo:
            await service.stage(all_paths=True)
        assert excinfo.value.op == "parked"
        assert excinfo.value.code == "busy"
        assert excinfo.value.status == 409
    finally:
        release.set()
        await running

    # The lock is free again, and the service is not stuck saying "busy".
    assert service.current_op is None
    await service.stage(all_paths=True)


async def test_a_write_that_cannot_be_read_back_is_a_failed_request(
        repo, monkeypatch):
    """``MutationResult.status`` is required, and this is what makes that
    true: the tab is never handed a result with a hole where the status
    should be, so it never has to decide what to draw for one."""
    from app.core.git import service as service_module

    def _broken(root):
        raise GitError("git_failed", 500, "the status could not be read")

    monkeypatch.setattr(service_module.repo, "read_status", _broken)

    with pytest.raises(GitError) as excinfo:
        await repo.service.mutate("noop", lambda root: {}, worktree=False)

    assert _error(excinfo).code == "git_failed"


async def test_a_read_never_waits_for_the_lock(repo):
    """A status poll while a commit runs the user's hooks must not queue."""
    started = threading.Event()
    release = threading.Event()

    def parked(root):
        started.set()
        assert release.wait(10)
        return {}

    service = repo.service
    running = asyncio.create_task(
        service.mutate("parked", parked, worktree=False))
    await asyncio.to_thread(started.wait, 10)

    try:
        response = await asyncio.wait_for(service.status(), 10)
        assert response.repo.state == "ready"
    finally:
        release.set()
        await running


# --- the plain functions, called the way a route never would ----------------


def test_the_operations_work_without_a_service(repo):
    """Every op is a function of a root: no loop, no lock, no service."""
    repo.write("a.txt", "edited\n")
    repo.write("new.txt", "new\n")

    assert stage_paths(repo.root, ["a.txt"]) == {"paths": ["a.txt"]}
    assert unstage_paths(repo.root, ["a.txt"]) == {"paths": ["a.txt"]}
    assert discard_paths(repo.root, ["new.txt"])["removed"] == 1
    detail = commit_changes(repo.root, "direct", all_paths=True)
    assert detail["sha"] == repo.head()


def test_changed_paths_reports_what_a_commit_swallowed(repo):
    """When HEAD moves, the status difference is not the whole story: the
    files that went INTO the commit are gone from every group."""
    before = repo_ops.read_status(repo.root)
    repo.commit("second", {"a.txt": "changed\n"})
    after = repo_ops.read_status(repo.root)

    assert changed_paths(repo.root, before, after) == ["a.txt"]


def test_a_branch_name_git_would_refuse_is_refused(repo):
    with pytest.raises(GitError) as excinfo:
        check_ref_format(repo.root, "bad..name")

    assert _error(excinfo).code == "invalid_ref"


def test_a_branch_name_that_is_an_option_is_refused_before_git_sees_it(repo):
    """git ACCEPTS ``refs/heads/-x`` (measured, exit 0). On a command line
    ``-x`` is an option, so the regex is the defence, not git."""
    accepted = subprocess.run(
        ["git", "-C", str(repo.root), "check-ref-format", "refs/heads/-x"],
        capture_output=True)
    assert accepted.returncode == 0, "git changed its mind about a leading -"

    with pytest.raises(GitError) as excinfo:
        check_ref_format(repo.root, "-x")

    assert _error(excinfo).code == "invalid_ref"


def test_a_branch_name_git_accepts_comes_back_unchanged(repo):
    assert check_ref_format(repo.root, "feat/source-control") == (
        "feat/source-control")


def test_resolve_repo_is_a_plain_function_of_a_directory(repo, tmp_path):
    """The service injects it; nothing about it needs a service."""
    assert resolve_repo(repo.root).state == "ready"
    assert resolve_repo(None).state == "no_project"
    assert resolve_repo(tmp_path / "gone").state == "not_repo"


def test_the_status_read_is_the_same_one_every_module_uses(repo):
    """One reader, so a diff, a discard and a poll cannot disagree about
    what is untracked."""
    repo.write("new.txt", "new\n")

    status = repo_ops.read_status(repo.root)

    assert [entry.path for entry in status.untracked] == ["new.txt"]
    assert status.branch == "main"
    assert status.head == repo.head()
