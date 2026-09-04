"""REST surface for the Source Control tab.

    GET    /api/git/status                 the repository, and its status
    GET    /api/git/log                    one page of history
    GET    /api/git/commits/{sha}/files    what one commit changed
    GET    /api/git/diff                   the patch for one file
    GET    /api/git/file                   one file's content at one ref
    GET    /api/git/config                 who commits here, and from where
    GET    /api/git/branches               every branch, local and remote
    GET    /api/git/remotes                every remote, and its two URLs
    POST   /api/git/init                   make the project a repository
    POST   /api/git/stage /unstage /discard
    POST   /api/git/commit
    PUT    /api/git/config                 write user.name / user.email
    POST   /api/git/branches               make a branch
    POST   /api/git/checkout               go to one
    PUT    /api/git/branches/{name}        rename one
    DELETE /api/git/branches/{name}        delete one (?force=1 unmerged)
    POST   /api/git/remotes                add a remote
    PUT    /api/git/remotes/{name}         point it somewhere else
    DELETE /api/git/remotes/{name}         forget it

Every route is three lines long, and that is the design rather than an
accident: the service on ``app.state`` owns the repository, the lock, the
worker threads and every decision about what git may be asked. What is left
here is the four things that only exist once there is a REQUEST in front of
one.

* **Auth is the house rule and nothing else.** ``auth_guard`` in ``main.py``
  requires the session token for every mutating ``/api/`` call, so the six
  writes here are covered by being POSTs and PUTs; the reads are open GETs
  like every other read in the app. This router is deliberately NOT in
  ``_AUTH_EXEMPT_PREFIXES`` -- an exemption would silently drop
  authentication from every write, because none of these routes declares a
  route-level dependency -- and there is no loopback gate either. That was
  decided: access control for a server somebody deliberately serves to a
  LAN is the deploying organisation's job (issue #247), and unlike a pack
  install this runs the user's own git against the directory they opened,
  with the credentials they already have.
* **One failure shape, redacted once.** Every ``GitError`` becomes an
  ``HTTPException`` with a dict ``detail`` carrying ``code`` (the closed
  vocabulary the frontend translates), ``message``, ``hint`` and ``stderr``
  -- through :func:`_http_error` and nowhere else, because that function is
  also where :func:`~app.core.git.errors.redact` runs. git's stderr can hold
  a credential (``https://user:ghp_xxx@github.com/...`` is a URL somebody
  really does paste into ``git remote add``), and a route that built its own
  envelope would be the one that leaked it. Nothing here logs ``.stderr`` at
  all: the tail travels in the response, redacted, and there is no second
  copy of it to get wrong.
* **A missing service is a 503, not a traceback.** The lifespan builds the
  service; a test client does not run the lifespan, and a server whose
  startup failed has no repository to talk about either.
* **What the query string may say is a closed set.** ``limit`` is 1..100,
  ``skip`` is 0..2**31-1 (git's own parser stops there, and one past it is
  a failure rather than an empty page) and ``scope`` is one of three words
  -- all three enforced by the signature, so an out-of-range page is a 422
  before any code here runs. Everything with a meaning -- a path, a ref, a
  sha, a message -- is validated by ``core/git/paths.py`` instead, which
  answers with a ``code`` the frontend can translate rather than with
  pydantic's English.
* **A branch name in a path is ``{name:path}``.** Half the branches anybody
  has contain a slash (``feat/source-control``), and a plain ``{name}``
  stops at one -- so ``PUT /branches/feat/x`` would be a 404 from the
  router, and so would the ``%2F`` a client sends instead, because the ASGI
  server has already turned it back into a slash by the time the router
  sees it. The converter is the only spelling that matches both.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Literal, TypeVar

from fastapi import APIRouter, HTTPException, Query, Request

from ..core.git.errors import GitBusy, GitError, redact
from ..core.git.log import DEFAULT_LOG_LIMIT, MAX_LOG_LIMIT
from ..core.git.models import (
    BranchCreateRequest,
    BranchesResponse,
    BranchRenameRequest,
    CheckoutRequest,
    CommitRequest,
    DiffResponse,
    FileAtRef,
    GitFile,
    Identity,
    IdentityRequest,
    LogResponse,
    MutationResult,
    PathsRequest,
    RemoteCreateRequest,
    RemoteInfo,
    RemoteUrlRequest,
    StatusResponse,
)
from ..core.git.service import GitService

router = APIRouter(prefix="/api/git", tags=["git"])

_T = TypeVar("_T")

#: The three things a diff can be about, as the wire spells them. Declared
#: as a ``Literal`` so a fourth word is a 422 from the signature rather than
#: a string this module has to check.
DiffScope = Literal["worktree", "index", "commit"]

#: The largest ``skip`` git will take. ``--skip=`` is parsed as a signed
#: 32-bit integer, and one past it is not "an empty page" but a failure:
#: ``fatal: '2147483648': not an integer``, exit 128 (measured on git 2.53),
#: which reached the browser as a 500 from a GET that needs no token.
#: Bounded in the signature, so it is a 422 before a process starts.
MAX_LOG_SKIP = 2**31 - 1

#: What a request gets when the lifespan never built a service. Deliberately
#: NOT a member of ``errors.CODES``: that table is the vocabulary of things
#: GIT can fail at, and this is the server failing to have started -- the
#: same distinction that keeps ``no_project`` in it and this out.
SERVICE_UNAVAILABLE_CODE = "git_service_unavailable"


def _http_error(exc: GitError) -> HTTPException:
    """The ONE place a git failure becomes a response body.

    Four keys, always the same four, because the frontend is typed against
    them: ``code`` is what it switches on and translates, ``message`` is the
    English fallback for a log or a developer, ``hint`` is the one fact the
    code cannot carry (which file, which branch), and ``stderr`` is git's
    own tail -- kept, because a classification can be wrong and a wrong
    answer with no evidence under it cannot be argued with.

    ``message`` and ``stderr`` are REDACTED on the way out, and ``hint``
    with them: a credential can only reach a client through one of those
    three strings, and putting the call in one place is what makes "no token
    is ever serialised" a property of the code rather than of every handler
    remembering. Nothing is logged here -- the redacted tail is in the
    response, and a second copy in a log file is a second thing to get
    wrong.

    ``busy`` additionally carries ``op``, the operation that holds the lock,
    so the tab can say "wait for the commit" rather than "wait". It is the
    only code with a fifth key, and an extra key is what a client that does
    not know about it ignores.
    """
    detail: dict[str, object] = {
        "code": exc.code,
        "message": redact(exc.message),
        "hint": redact(exc.hint) if exc.hint else exc.hint,
        "stderr": redact(exc.stderr) if exc.stderr else exc.stderr,
    }
    if isinstance(exc, GitBusy):
        detail["op"] = exc.op
    return HTTPException(status_code=exc.status, detail=detail)


def _service(request: Request) -> GitService:
    """The service the lifespan built, or 503.

    Optional on ``app.state`` for the same reason every other store is: the
    lifespan does not run under httpx's ASGITransport, so a test reaches
    these routes with nothing there unless it installs one.
    """
    service = getattr(request.app.state, "git_service", None)
    if service is None:
        raise _http_error(GitError(
            SERVICE_UNAVAILABLE_CODE, 503,
            "source control is not available on this server",
            hint="the server started without it; restart the server"))
    return service


async def _answer(call: Awaitable[_T]) -> _T:
    """Await one service call, turning its failure into the envelope.

    Takes the coroutine rather than wrapping the route, so the mapping is a
    single expression at every call site and FastAPI still sees the real
    signature of every handler (a decorator that hid one would take the
    query-parameter validation above with it).
    """
    try:
        return await call
    except GitError as exc:
        raise _http_error(exc) from None


# --- reads ------------------------------------------------------------------


@router.get("/status")
async def read_status(request: Request) -> StatusResponse:
    """The repository, and its status when there is one to read.

    ALWAYS a 200 for a repository that is not there: no project open, no
    git installed, a directory that is not a repository and one that sits
    inside somebody else's are states the tab draws a screen for, not
    failures it reports. ``status`` is null beside them.
    """
    return await _answer(_service(request).status())


@router.get("/log")
async def read_log(
    request: Request,
    skip: int = Query(0, ge=0, le=MAX_LOG_SKIP),
    limit: int = Query(DEFAULT_LOG_LIMIT, ge=1, le=MAX_LOG_LIMIT),
) -> LogResponse:
    """One page of history, newest first.

    ``has_more`` comes from asking git for one commit more than the page
    and dropping it, so "is there another page" costs nothing extra and an
    unborn branch answers with an empty page rather than an error.

    Both numbers are bounded here: a page past the end of the history is an
    empty page, but a ``skip`` past what git's own parser accepts is a
    failure with nothing to say to the user.
    """
    return await _answer(_service(request).log(skip=skip, limit=limit))


@router.get("/commits/{sha}/files")
async def read_commit_files(request: Request, sha: str) -> list[GitFile]:
    """The files one commit changed, against its first parent."""
    return await _answer(_service(request).commit_files(sha))


@router.get("/diff")
async def read_diff(
    request: Request,
    path: str,
    scope: DiffScope,
    sha: str | None = None,
    blobs: bool = False,
) -> DiffResponse:
    """The patch for one file, in one of the three scopes.

    ``blobs=1`` additionally fills both sides in full, for a side-by-side
    view. It costs two more reads of the repository, so it is off unless
    asked for.
    """
    service = _service(request)
    commit = _commit_for_scope(scope, sha)
    return await _answer(service.diff(path, scope, sha=commit, blobs=blobs))


def _commit_for_scope(scope: str, sha: str | None) -> str | None:
    """The commit a diff is about, when its scope has one.

    An EMPTY ``sha`` counts as an absent one. The tab builds this query
    string from one form whose fields are always present, so ``sha=`` is
    what "no commit" looks like on the wire, and reading it as a commit id
    would turn every worktree diff into a 400.

    Both directions are refused, because they are both a client bug and the
    silent readings are worse than a 400: a commit diff without a commit
    would be answered by the layer below with the same code but no hint,
    and a worktree diff carrying a sha would quietly ignore it and show the
    user a diff of something else.
    """
    commit = sha or None
    if scope == "commit" and commit is None:
        raise _http_error(GitError(
            "invalid_value", 400,
            "a commit diff needs the commit it is about",
            hint="pass sha=<commit> with scope=commit"))
    if scope != "commit" and commit is not None:
        raise _http_error(GitError(
            "invalid_value", 400,
            f"a {scope} diff is not about one commit",
            hint="drop sha, or ask for scope=commit"))
    return commit


@router.get("/file")
async def read_file(request: Request, path: str, ref: str) -> FileAtRef:
    """One file's content at ``HEAD``, ``index``, ``worktree`` or a sha.

    A dotenv file is refused at every ref (403 ``ignored``), and so is an
    ignored file read from the worktree: ``.gitignore`` is where a project
    says which files are not part of it, and an open GET that served them
    anyway would make that promise worthless.
    """
    return await _answer(_service(request).file_at_ref(path, ref))


@router.get("/config")
async def read_config(request: Request) -> Identity:
    """``user.name`` / ``user.email``, and which config file each came from.

    The scope is half the answer: it is the difference between "this
    repository" and "every repository on this machine", and the tab shows it
    before it offers to commit.
    """
    return await _answer(_service(request).identity())


@router.get("/branches")
async def read_branches(request: Request) -> BranchesResponse:
    """Every branch, local and remote-tracking, and what HEAD is on.

    One read, both lists: the Branches section draws them together, and a
    remote branch's only job in that panel is to be the thing a Switch turns
    into a local one.
    """
    return await _answer(_service(request).branches())


@router.get("/remotes")
async def read_remotes(request: Request) -> list[RemoteInfo]:
    """Every configured remote, with its fetch and push URLs."""
    return await _answer(_service(request).remotes())


# --- writes (the session token, via auth_guard) -----------------------------


@router.post("/init")
async def init_repository(request: Request) -> MutationResult:
    """Make the project directory a repository, with the shared scaffold.

    No body, and one is ignored if sent: there is nothing to choose. The
    scaffold gives a new repository the same ``.gitignore`` and
    ``.gitattributes`` ``cdui project init`` writes -- the first line of
    which is ``.env``, because a repository that does not ignore it will
    sooner or later have the user's API keys in its history, where no later
    ``.gitignore`` can reach them.
    """
    return await _answer(_service(request).init())


@router.post("/stage")
async def stage(request: Request, payload: PathsRequest) -> MutationResult:
    """Stage the named paths, or the whole tree with ``all``."""
    service = _service(request)
    return await _answer(service.stage(payload.paths, all_paths=payload.all))


@router.post("/unstage")
async def unstage(request: Request, payload: PathsRequest) -> MutationResult:
    """Take the named paths, or everything, back out of the index."""
    service = _service(request)
    return await _answer(service.unstage(payload.paths, all_paths=payload.all))


@router.post("/discard")
async def discard(request: Request, payload: PathsRequest) -> MutationResult:
    """Throw away working-tree changes. The one write that destroys.

    A tracked file is restored from the index and an untracked one is
    deleted, which is the only thing "discard" can mean for a file that has
    no other copy. Ignored files are never touched.
    """
    service = _service(request)
    return await _answer(service.discard(payload.paths, all_paths=payload.all))


@router.post("/commit")
async def commit(request: Request, payload: CommitRequest) -> MutationResult:
    """Commit the index, or the whole tree with ``all``; ``amend`` replaces.

    ``detail.sha`` and ``detail.short`` name the commit this made, so the
    tab can link to it without reading the log again.
    """
    service = _service(request)
    return await _answer(service.commit(payload.message, all_paths=payload.all,
                                        amend=payload.amend))


@router.put("/config")
async def write_config(request: Request, payload: IdentityRequest) -> Identity:
    """Write ``user.name`` / ``user.email`` into THIS repository.

    Always ``--local``: writing the machine's global config from a web
    request would change every repository on it, including ones this app has
    never been pointed at.

    Answers with the identity as it now READS -- which is not the same thing
    as what was written. A name written locally can still be read back with
    a different scope beside it (the email may remain global), and that pair
    is what the tab shows before it lets a commit be made.
    """
    service = _service(request)
    if payload.name is None and payload.email is None:
        raise _http_error(GitError(
            "invalid_value", 400, "send a name, an email, or both",
            hint="an identity write with neither would change nothing"))
    await _answer(service.set_identity(payload.name, payload.email))
    return await _answer(service.identity())


# --- the refs (the static paths first, then the named ones) -----------------


@router.post("/branches")
async def create_branch(request: Request,
                        payload: BranchCreateRequest) -> MutationResult:
    """Make a branch, and go to it unless ``checkout`` says otherwise.

    ``detail.branch`` names it, so the tab can say which branch it is on
    without reading the status back out of the result.
    """
    service = _service(request)
    return await _answer(service.create_branch(
        payload.name, checkout=payload.checkout,
        start_point=payload.start_point))


@router.post("/checkout")
async def checkout(request: Request,
                   payload: CheckoutRequest) -> MutationResult:
    """Go to a local branch, or to a new one tracking a remote's.

    Its own route rather than a ``PUT`` on something: a checkout is not a
    change TO a branch, it is a change to the working tree, and
    ``changed_paths`` on the result is the list an open editor reloads from.
    """
    service = _service(request)
    return await _answer(service.checkout(payload.target, kind=payload.kind))


@router.put("/branches/{name:path}")
async def rename_branch(request: Request, name: str,
                        payload: BranchRenameRequest) -> MutationResult:
    """Give a branch another name."""
    service = _service(request)
    return await _answer(service.rename_branch(name, payload.new_name))


@router.delete("/branches/{name:path}")
async def delete_branch(request: Request, name: str,
                        force: bool = False) -> MutationResult:
    """Delete a branch; ``force=1`` deletes one that is not merged.

    Two requests on purpose. The first is ``-d``, which git refuses with
    ``branch_not_merged`` when the branch holds work no other branch has;
    the tab then says so and asks again, and the second carries ``force``.
    Deleting unmerged work is the only thing in this router that cannot be
    undone from the tab, so the consent for it is a separate click.
    """
    service = _service(request)
    return await _answer(service.delete_branch(name, force=force))


@router.post("/remotes")
async def add_remote(request: Request,
                     payload: RemoteCreateRequest) -> MutationResult:
    """Point a name at a repository somewhere else."""
    service = _service(request)
    return await _answer(service.add_remote(payload.name, payload.url))


@router.put("/remotes/{name}")
async def set_remote_url(request: Request, name: str,
                         payload: RemoteUrlRequest) -> MutationResult:
    """Point an existing remote somewhere else.

    A plain ``{name}``, unlike the branch routes: a remote name is a closed
    grammar with no slash in it (``paths.REMOTE_NAME_RE``), so there is
    nothing for the ``path`` converter to rescue.
    """
    service = _service(request)
    return await _answer(service.set_remote_url(name, payload.url))


@router.delete("/remotes/{name}")
async def remove_remote(request: Request, name: str) -> MutationResult:
    """Forget a remote. Nothing is fetched, pushed or deleted anywhere else."""
    service = _service(request)
    return await _answer(service.remove_remote(name))
