"""What the user typed, turned into something the installer can act on.

One field accepts four shapes -- ``deep``, ``alice/extras``,
``alice/extras@v1.2.3``, ``https://github.com/alice/extras`` -- because a
person who has a plugin in mind should not have to know which of the two
install paths it goes down. Sorting that out is this module's whole job, and
it is deliberately the only place that knows the answer: the CLI and the
routes both get the same verdict on the same string.

A path on this machine is a fifth shape (``/x``, ``./x``, ``~/x``,
``file:///x``): a folder or an archive the inspection copies and reviews
like any other source. A relative path has to start with ``./`` because
``plugins/x`` is already ``owner/repo``.

An HTTPS URL on any other host is a sixth (``https://host/owner/repo[.git]
[@ref]``): GitLab, Gitea, Codeberg, Bitbucket, a school's own server. It is
cloned with ``git``. github.com keeps going down the GitHub API road, and
nothing else that looks like a git remote (``ssh://``, ``git@host:x``,
``http://``, a URL with a user name in it) is a source.

The catalog wins over the GitHub patterns, and it is checked first for a
reason: ``deep`` is a legal repository name, so a bare word that IS a catalog
id has to resolve to the pack that ships here rather than to whatever
``deep`` happens to be on GitHub.

A refusal is structured (see :mod:`.errors`), never prose. There are two, and
the difference matters to the person reading it: a bare word can only ever
have been meant as a catalog name, so the answer is "your catalog does not
have that one, here is what it does have"; anything else needs the general
"here are the shapes this field takes". Which sentence says that, and in
which language, belongs to the caller.
"""

from __future__ import annotations

import re
from typing import Any, Literal, NamedTuple

from . import catalog as catalog_module
from .errors import UnknownCatalogName, UnparseableSource
from .local import path_from_spec
from .manifest import REF_SEGMENT, REPO_SEGMENT

# Accepts owner/repo or owner/repo@ref, in the same characters the catalog
# accepts in an entry's ``repo`` -- one shape rule, spelled once in
# :data:`~app.core.plugins.manifest.REPO_SEGMENT`, because a source this build
# takes from the catalog and refuses when typed is indefensible either way.
# The ref half is :data:`~app.core.plugins.manifest.REF_SEGMENT` for the same
# reason: the URL form used to end in ``(?:@(.+))?``, so the same ref was a
# legal source through a URL and an illegal one through the short form.
_GITHUB_SHORT = re.compile(
    rf"^({REPO_SEGMENT}+)/({REPO_SEGMENT}+?)(?:@({REF_SEGMENT}+))?$"
)
_GITHUB_URL = re.compile(
    rf"^https?://(?:www\.)?github\.com/({REPO_SEGMENT}+)/({REPO_SEGMENT}+?)"
    rf"(?:\.git)?/?(?:@({REF_SEGMENT}+))?$"
)
# One word, no slash and no scheme: the only thing it can have been meant as is
# a catalog name. See UnknownCatalogName for why that deserves its own error.
_BARE_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*$")

# A path on this machine: absolute (``/x``, ``C:\x``, ``C:/x``, ``\\server``),
# explicitly relative (``./x``, ``../x``, ``.``, ``..``), home-relative
# (``~``, ``~/x``) or a ``file://`` URL. ``plugins/x`` is NOT one: it is the
# ``owner/repo`` shape, and a relative path has to say so with ``./``.
_LOCAL_PATH = re.compile(
    r"^(?:/|\\\\|[A-Za-z]:[\\/]|\.{1,2}(?:$|[\\/])|~|file://)",
    re.IGNORECASE,
)

# A repository on any HTTPS git host: a DNS host name (no user name, so no
# ``@`` before the path), an optional port, at least two path segments in
# the characters GitHub allows, and an optional ``@ref``. Group 1 is the URL
# without the ref, group 2 the host, group 3 the ref.
_GIT_HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_GIT_URL = re.compile(
    rf"^((?i:https)://({_GIT_HOST_LABEL}(?:\.{_GIT_HOST_LABEL})*)(?::\d{{1,5}})?"
    rf"(?:/{REPO_SEGMENT}+){{2,}})/?(?:@({REF_SEGMENT}+))?$"
)
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})


class ParsedSource(NamedTuple):
    """Where a plugin is to come from.

    ``name_or_owner`` carries the plugin id for a catalog source, the GitHub
    owner for a repository one, and the absolute path for a local one -- one
    field because they are mutually exclusive and the ``kind`` says which is
    meant. ``repo`` and ``ref`` are empty strings for a catalog or a local
    source, and ``ref`` is empty for a repository whose default branch is
    wanted.
    """

    kind: Literal["catalog", "github", "local", "git"]
    name_or_owner: str
    repo: str
    ref: str


def is_local_path(spec: str) -> bool:
    """Does *spec* have the shape of a path on this machine? Reads nothing."""
    return bool(_LOCAL_PATH.match(spec))


def parse_git_url(spec: str) -> tuple[str, str] | None:
    """``(url, ref)`` when *spec* is an HTTPS repository on a host other than
    GitHub, else ``None``. ``ref`` is ``""`` for the default branch.

    Refused here rather than at the clone: a path segment of ``.`` or
    ``..`` (the URL would name a different repository than it reads as), a
    ref that walks up or starts with ``-`` (it is handed to ``git`` as an
    argument), and github.com, which has its own road.
    """
    m = _GIT_URL.match(spec)
    if m is None or m.group(2).lower() in _GITHUB_HOSTS:
        return None
    url, ref = m.group(1), m.group(3) or ""
    segments = url.split("/")[3:]
    if any(segment in (".", "..") for segment in segments):
        return None
    if ref and (_walks_up(ref) or ref.startswith("-")):
        return None
    return url, ref


def parse_github_url(url: str) -> tuple[str, str] | None:
    """``(owner, repo)`` when *url* is a GitHub repository URL, else ``None``.

    The same pattern :func:`parse_source` reaches for, exported because a
    second reader asks a narrower question -- a lockfile wrote a ``url`` down,
    and something wants to know which repository it names -- with no catalog
    lookup, no short form and no refusal to catch.

    The HOST is half of the answer, which is why this exists at all rather
    than a caller taking the last two path segments:
    ``https://evil.example.com/CodefyUI/CodefyUI-Plugin-Self-Learning`` ends
    in the owner and repository of a plugin CodefyUI vouches for, and is not
    that plugin. Nothing here is lower-cased -- GitHub names are not
    case-sensitive, but this reports what was recorded and leaves the
    comparison to the caller.
    """
    m = _GITHUB_URL.match(url)
    return (m.group(1), m.group(2)) if m else None


def _walks_up(ref: str) -> bool:
    """Does *ref* contain ``..`` as a PATH SEGMENT?

    A ref is interpolated into URL paths --
    ``api.github.com/repos/<o>/<r>/commits/<ref>`` and, at the resolved sha,
    ``raw.githubusercontent.com/...`` -- so a ``..`` segment climbs out of
    the endpoint the caller meant to ask about. Segment-wise rather than a
    substring test, because ``v1..2`` is a legal tag name and only ``..``
    standing alone between slashes is the traversal.
    """
    return ".." in ref.split("/")


def parse_source(
    spec: str,
    *,
    catalog: dict[str, Any] | None = None,
) -> ParsedSource:
    """Resolve *spec* to a :class:`ParsedSource`, or raise a
    :class:`~.errors.SourceError`.

    Catalog lookup is case-insensitive; the id in the result is lower-cased,
    because that is the directory name the pack lives under.

    *catalog* overrides what is read from disk. It exists for
    ``scripts/plugins.py``, whose tests fake the catalog by patching the CLI's
    own ``load_catalog``: without it this would reach past that patch and
    answer from the real ``registry.json``.

    A ref that walks up (see :func:`_walks_up`) is unparseable rather than
    passed on, because there is nothing a caller could usefully do with it
    except refuse it later, further from the string that was typed.
    """
    data = catalog_module.load_catalog() if catalog is None else catalog
    plugins = data.get("plugins", {})

    if spec.lower() in plugins:
        return ParsedSource("catalog", spec.lower(), "", "")

    if is_local_path(spec):
        # Shape only: whether anything is there is the inspection's question,
        # asked where the answer can say what WAS found.
        return ParsedSource("local", path_from_spec(spec), "", "")

    m = _GITHUB_URL.match(spec) or _GITHUB_SHORT.match(spec)
    if m:
        ref = m.group(3) or ""
        if _walks_up(ref):
            raise UnparseableSource(spec)
        return ParsedSource("github", m.group(1), m.group(2), ref)

    git = parse_git_url(spec)
    if git is not None:
        return ParsedSource("git", git[0], "", git[1])

    if _BARE_NAME.match(spec):
        raise UnknownCatalogName(spec, sorted(plugins), catalog_module.catalog_path())

    raise UnparseableSource(spec)
