"""The shapes the Source Control tab sends and receives.

One module for the whole vocabulary, because these types ARE the contract
between the backend and the tab: the parser fills them, the service returns
them, the routes serialise them, and the frontend's TypeScript is written
from this file. A field added in a route handler instead of here is a field
the frontend never learns about.

Three decisions are worth stating once, here, rather than re-deciding per
model:

* **Responses default to the empty answer; requests do not.** Every field of
  a response that can be absent has the default that means "git did not
  say" -- None, ``False``, ``0``, an empty list -- so a producer never has to
  spell out an absence, and an unfilled ``GitStatus()`` claims nothing about
  the repository. What a response exists to CARRY stays required (a patch, a
  file's text and size, a commit's sha), so it cannot be left out by
  accident. Requests are the opposite: they carry ``extra="forbid"``, so a
  body with a key nobody defined is a 422 rather than a silently ignored
  instruction -- the same reasoning ``routes_packs.InstallRequest`` gives,
  where the schema rather than the handler is what makes "the client cannot
  smuggle in an argument" a guarantee.
* **``kind`` and ``xy`` are both kept, and they are not redundant.** ``kind``
  is the summary the UI draws an icon from; ``xy`` is git's own two letters,
  which say strictly more. ``DU`` and ``UU`` are both ``kind="conflict"``
  and they want different buttons ("keep ours" means "keep it deleted" in
  one of them), and ``MM`` is one file that is in the staged list AND the
  unstaged list. Throwing the letters away here would mean re-running git to
  get them back.
* **``FileKind`` is closed.** The frontend switches on it; a kind it has
  never heard of renders as nothing at all. So the parser maps anything
  unexpected onto a member of this set rather than inventing one -- see
  ``status.kind_from_letter``.

Requests validate SHAPE only. Whether a path escapes the repository, whether
a message is too long, whether an email looks like an email -- that is
``paths.py``'s job, and it belongs there because those failures have to come
back as a :class:`~app.core.git.errors.GitError` with a code the frontend
translates, not as pydantic's English prose in a 422.

G3 adds the refs: ``BranchInfo`` / ``RemoteBranchInfo`` / ``BranchesResponse``
/ ``RemoteInfo`` and the requests that create, switch, rename and delete
them. ``StashInfo`` and the stash and network requests are the rest of G3.

No subprocess and no runner import: the parser and the tests can build a
status without a git on PATH.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: What happened to one file, as the tab draws it.
#:
#: ``untracked`` and ``conflict`` are not git status letters -- they are the
#: RECORD types ``?`` and ``u`` -- but they belong in the same set because
#: the UI asks one question of a file ("what icon, what actions") and wants
#: one answer.
FileKind = Literal[
    "modified",
    "added",
    "deleted",
    "renamed",
    "copied",
    "typechange",
    "untracked",
    "conflict",
]

#: Where git read a config value from. ``None`` means the value is unset.
ConfigScope = Literal["local", "global", "system"]

#: How far ``repo_info()`` got. Resolved in this order, first hit wins:
#: no project directory is open, git is not installed, git is too old for
#: ``restore``/``switch``, the project directory is not the top level of a
#: repository, everything works.
RepoState = Literal[
    "no_project",
    "git_missing",
    "git_too_old",
    "not_repo",
    "ready",
]


class GitFile(BaseModel):
    """One file in one of the four status groups.

    The same path can appear twice in one status -- ``MM`` is staged AND
    unstaged -- and those are two entries with the same ``xy`` and different
    ``kind``. That is deliberate: it is what VS Code shows, and it is the
    only way to offer "unstage" and "discard" on the same file at once.
    """

    #: Repository-relative, POSIX separators, never quoted (git runs with
    #: ``core.quotepath=false`` and ``-z``, so a CJK name arrives as itself).
    path: str
    #: Where a rename or copy came from; ``None`` for every other kind.
    orig_path: str | None = None
    kind: FileKind
    #: git's two status letters, exactly as porcelain v2 printed them
    #: (``MM``, ``.M``, ``A.``, ``UU``, ``DU``). Untracked entries have no
    #: letters in porcelain v2 and get the ``??`` of porcelain v1, so that
    #: every entry the frontend sees has two characters here.
    xy: str
    #: Similarity percentage of a rename or copy (git's ``R100`` / ``C75``).
    score: int | None = None


class GitStatus(BaseModel):
    """Everything one ``git status --porcelain=v2 --branch`` call said.

    Constructed with no arguments, every field means "git did not mention
    it": no branch, no commit, nothing changed. That is the honest default
    rather than a pretend one -- a status object nobody filled in claims
    nothing about the repository.
    """

    #: The current branch, or ``None`` when HEAD is detached.
    branch: str | None = None
    detached: bool = False
    #: The commit HEAD points at; ``None`` on an unborn branch.
    head: str | None = None
    #: No commit yet (git prints ``# branch.oid (initial)``).
    unborn: bool = False
    #: The upstream branch as git names it (``origin/main``), if configured.
    upstream: str | None = None
    #: Commits this branch has that the upstream does not, and the reverse.
    #: Both are ``None`` when there is no upstream -- or when there is one
    #: and it no longer exists, which is what ``upstream_gone`` is for.
    ahead: int | None = None
    behind: int | None = None
    #: An upstream is configured but its ref is gone (the remote branch was
    #: deleted and pruned). git then prints ``# branch.upstream`` with no
    #: ``# branch.ab``, so "configured but uncounted" is the whole signal
    #: porcelain v2 gives.
    upstream_gone: bool = False
    staged: list[GitFile] = Field(default_factory=list)
    unstaged: list[GitFile] = Field(default_factory=list)
    untracked: list[GitFile] = Field(default_factory=list)
    conflicted: list[GitFile] = Field(default_factory=list)
    #: Entries on the stash stack (``# stash N``; absent means none).
    stash_count: int = 0
    #: Set by the SERVICE, not the parser: porcelain v2 says nothing about
    #: them, and the answer is whether ``MERGE_HEAD`` / ``rebase-merge`` /
    #: ``rebase-apply`` exist under the git directory.
    merge_in_progress: bool = False
    rebase_in_progress: bool = False


class RepoInfo(BaseModel):
    """Whether the tab can talk to a repository at all, and why not.

    Always answered, even when the answer is "no": ``GET /api/git/status``
    is a 200 with ``status=None`` rather than an error, because "there is no
    project open" is a screen the tab draws, not a failure it reports.
    """

    state: RepoState
    #: The open project directory, as an absolute path string.
    project_dir: str | None = None
    #: ``git --version``'s answer, when there is a git to ask.
    git_version: str | None = None
    #: Set when the project directory sits INSIDE some other repository (the
    #: CodefyUI checkout, or a home directory somebody ran ``git init`` in).
    #: The tab must never operate on that repository, so the path is
    #: reported and the state stays ``not_repo``.
    nested_toplevel: str | None = None


class StatusResponse(BaseModel):
    """``GET /api/git/status``: the repository, and the status if there is one."""

    repo: RepoInfo
    status: GitStatus | None = None


class CommitInfo(BaseModel):
    """One row of the history."""

    sha: str
    short: str
    #: Parent shas; empty for the root commit, two or more for a merge.
    parents: list[str] = Field(default_factory=list)
    author_name: str
    author_email: str
    #: Author date as a unix timestamp (git's ``%at``). A number, not a
    #: string: the browser formats it in the user's own locale and timezone,
    #: which a server-side format would get wrong for everybody else.
    authored_at: int
    #: Ref names pointing here (``HEAD -> main``, ``origin/main``, tags).
    refs: list[str] = Field(default_factory=list)
    subject: str
    body: str = ""


class LogResponse(BaseModel):
    """``GET /api/git/log``: one page of history.

    ``has_more`` comes from asking git for one commit more than the page
    size and dropping it, which is cheaper and steadier than counting the
    whole history to paginate against.
    """

    commits: list[CommitInfo] = Field(default_factory=list)
    has_more: bool = False
    #: No commits exist yet, so an empty page is the complete answer.
    unborn: bool = False


class DiffResponse(BaseModel):
    """``GET /api/git/diff``: a patch, and optionally both sides in full.

    The patch is what the tab renders by default. ``old_text`` / ``new_text``
    are filled only when the caller asks for the blobs (a side-by-side view
    needs the whole file, not the hunks), and stay ``None`` otherwise rather
    than shipping two copies of every file on every request.
    """

    patch: str
    #: git said "Binary files ... differ": there is no text to show.
    binary: bool = False
    #: The patch hit the size cap and was cut.
    truncated: bool = False
    #: What was compared, in the tab's own words (``HEAD``, ``index``,
    #: ``worktree``, a sha) -- for the header above the diff.
    old_ref: str | None = None
    new_ref: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    #: The file does not exist on that side (added, or deleted).
    old_missing: bool = False
    new_missing: bool = False


class FileAtRef(BaseModel):
    """``GET /api/git/file``: one file's content at one ref.

    ``truncated`` does not promise a prefix. A file read from DISK is cut
    at the cap and ``text`` holds what came before the cut; a file read out
    of the object database is not read at all once its size is known to be
    over it, and ``text`` is then empty beside a real ``size``. The tab
    draws the same "too big to show" panel for both, and the alternative
    -- streaming a megabyte nobody will look at, or buffering a 200 MB blob
    to throw most of it away -- costs the server more than the answer is
    worth.
    """

    #: Empty for a binary file, for an empty one, and for a file past the
    #: cap that was never read -- which is why ``binary`` and ``truncated``
    #: are fields of their own rather than something to infer.
    text: str
    #: A NUL in the first bytes: the tab shows a placeholder, not mojibake.
    binary: bool = False
    #: Size in bytes of what git had, BEFORE any truncation. Required, so a
    #: binary file still reports how big it is.
    size: int
    truncated: bool = False


class Identity(BaseModel):
    """Who commits, and where that was configured.

    The scope is shown because it is the difference between "this changes
    one repository" and "this changes every repository on the machine", and
    the tab only ever WRITES ``local``.
    """

    name: str | None = None
    email: str | None = None
    name_scope: ConfigScope | None = None
    email_scope: ConfigScope | None = None


class BranchInfo(BaseModel):
    """One local branch, as the Branches section draws a row of it.

    ``ahead`` / ``behind`` follow :class:`GitStatus`'s reading exactly, so
    the tab has one rule and not two: both are ``None`` when the branch has
    no upstream AND when the upstream is ``gone``, and a branch that is in
    step with a live upstream is ``0`` / ``0`` rather than null. git prints
    only the non-zero half (``[ahead 1]``), so the other one is filled in
    here -- an absent number would otherwise mean "no upstream" and
    "nothing to push" at the same time.
    """

    #: Without ``refs/heads/``: ``main``, ``feat/source-control``.
    name: str
    #: The tip, abbreviated the way git abbreviates it here.
    sha: str
    #: HEAD is on this branch.
    current: bool = False
    #: The upstream as git names it (``origin/main``), if configured.
    upstream: str | None = None
    ahead: int | None = None
    behind: int | None = None
    #: An upstream is configured and its ref no longer exists.
    gone: bool = False
    #: The tip commit's subject and commit date, for the row's second line.
    #: Both carry git's "did not say" default rather than being required:
    #: they are decoration, and a branch is worth listing without them.
    subject: str = ""
    committed_at: int = 0


class RemoteBranchInfo(BaseModel):
    """One remote-tracking branch: ``origin/main`` split into its two parts.

    Split, rather than shipped as one string, because the tab needs both
    halves separately -- the remote groups the list, the name is what a
    Switch creates locally -- and splitting a name that may itself contain
    slashes (``origin/feat/x``) is a rule better applied once here than in
    every component that renders one.
    """

    #: The branch on the remote, without the remote's name: ``main``.
    name: str
    #: The remote it lives on: ``origin``.
    remote: str
    sha: str
    subject: str = ""
    committed_at: int = 0


class BranchesResponse(BaseModel):
    """``GET /api/git/branches``: every branch, local and remote-tracking.

    ``current`` is the branch HEAD is on -- which exists on an UNBORN
    branch, where ``local`` is empty and there is still a name to show --
    and is ``None`` exactly when ``detached`` is true.
    """

    current: str | None = None
    detached: bool = False
    local: list[BranchInfo] = Field(default_factory=list)
    remote: list[RemoteBranchInfo] = Field(default_factory=list)


class RemoteInfo(BaseModel):
    """One configured remote, and the two URLs it may have.

    Both URLs are strings and never null: a remote whose push URL is not
    configured separately reports the fetch URL in both, which is what git
    itself prints and what actually happens on a push.

    **Both URLs are for DISPLAY ONLY.** They arrive REDACTED -- the
    userinfo of ``https://alice:ghp_xxx@github.com/owner/repo.git`` is
    masked to ``***`` before it leaves ``refs.list_remotes``, because this
    model is served by an open, unauthenticated GET. So the string here is
    not the string git holds, and a UI must never round-trip it: a "Change
    URL" prompt pre-filled from this field and submitted unchanged would
    write ``https://***@github.com/...`` into the config and destroy the
    credential the user had. The prompt starts empty, or from what the user
    types; nothing reads a URL back out of this API.
    """

    name: str
    fetch_url: str = ""
    push_url: str = ""


class MutationResult(BaseModel):
    """What one write left behind.

    Every mutation answers with a fresh ``status``, so the tab never has to
    ask twice and can never draw a stale panel after a stage or a commit.
    Required and NOT nullable, because that is the contract: the frontend is
    typed from this file, and a nullable field here would ship it a branch
    it has to handle and the service is required never to produce. A write
    that succeeds and then cannot be read back is a FAILED request -- an
    error with a code, which the tab already knows what to do with -- not a
    result with a hole in it.
    """

    status: GitStatus
    #: Paths whose state this operation changed, for the tab to highlight or
    #: to reload in an open editor.
    changed_paths: list[str] = Field(default_factory=list)
    #: HEAD after the operation, when there is one.
    head: str | None = None
    #: Operation-specific extras (which remote was pushed to, how many files
    #: a discard touched). Deliberately open: it is display sugar, and a
    #: closed model per operation would be a dozen models for one line of UI.
    detail: dict[str, Any] = Field(default_factory=dict)


class PathsRequest(BaseModel):
    """The body of stage / unstage / discard: some paths, or everything.

    Exactly one of the two forms, and that is enforced here rather than in
    three handlers. ``git add -A --`` with an empty pathspec stages the WHOLE
    tree, so "nothing was selected" arriving as ``paths=[]`` must never be
    able to turn into "all" further down; and a body carrying both is a
    client bug that would otherwise be resolved by whichever branch the
    handler happened to test first.
    """

    model_config = ConfigDict(extra="forbid")

    paths: list[str] | None = None
    #: Shadows the builtin, and stays: it is the wire name the frontend
    #: sends, and renaming it would put a mapping in every handler.
    all: bool = False

    @model_validator(mode="after")
    def _exactly_one_form(self) -> PathsRequest:
        """Refuse both, neither, and the empty list."""
        named = self.paths is not None
        if named and self.all:
            raise ValueError(
                "send either paths or all=true, not both")
        if not named and not self.all:
            raise ValueError(
                "send either paths (a non-empty list) or all=true")
        if named and not self.paths:
            raise ValueError(
                "paths must name at least one file; use all=true for "
                "the whole tree")
        return self


class CommitRequest(BaseModel):
    """The body of commit.

    ``message`` is a plain ``str`` on purpose: its length and content are
    checked by ``paths.validate_commit_message``, so an over-long message
    comes back as a 400 with the code ``invalid_value`` like every other
    refused value, instead of a 422 full of pydantic's English.
    """

    model_config = ConfigDict(extra="forbid")

    message: str
    #: Stage everything first (VS Code's "Commit All").
    all: bool = False
    #: Replace the previous commit instead of adding one. The UI hides this
    #: when the commit has already been pushed.
    amend: bool = False


class IdentityRequest(BaseModel):
    """The body of the identity write: either field, or both.

    Both may be omitted -- that is a request that changes nothing, which is
    a no-op rather than an error; the values themselves are checked by
    ``paths.validate_identity``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    email: str | None = None


class BranchCreateRequest(BaseModel):
    """The body of ``POST /branches``: a new branch, and whether to go to it.

    ``checkout`` defaults to true because that is what the button says --
    "Create Branch" in every editor that has one leaves you ON it -- and it
    is also the field that decides whether this write can move files, which
    is why the service reads it rather than assuming either answer.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    checkout: bool = True
    #: A branch name or a commit id to start from; ``None`` means HEAD.
    start_point: str | None = None


class CheckoutRequest(BaseModel):
    """The body of ``POST /checkout``: which branch, and which kind it is.

    ``kind`` is REQUIRED and not defaulted, because the two are different
    commands: a local switch moves HEAD to a branch that exists, and a
    remote one creates a new local branch tracking ``origin/<name>``. A
    default would make "the client forgot to say" indistinguishable from
    "the user picked local", and the wrong one of those creates a branch
    nobody asked for.
    """

    model_config = ConfigDict(extra="forbid")

    #: ``main`` for a local branch, ``origin/main`` for a remote one.
    target: str
    kind: Literal["local", "remote"]


class BranchRenameRequest(BaseModel):
    """The body of ``PUT /branches/{name}``: what to call it instead."""

    model_config = ConfigDict(extra="forbid")

    new_name: str


class RemoteCreateRequest(BaseModel):
    """The body of ``POST /remotes``: a name and where it points."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str


class RemoteUrlRequest(BaseModel):
    """The body of ``PUT /remotes/{name}``: where it points instead.

    Only the URL: renaming a remote is not offered, because a rename
    rewrites every ``branch.<name>.remote`` in the config and the tab has
    no way to show what that changed.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
