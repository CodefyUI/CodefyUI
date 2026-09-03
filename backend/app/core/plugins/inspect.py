"""Everything true about a plugin you have not installed yet, in one answer.

"Do you want to install this?" is a question nobody can answer from a name.
What it takes is the manifest -- what the plugin is, what it asks for, what
it would run in your browser -- plus what THIS install already knows about
it, which is the half that turns "install" into "update" and a re-grant into
a new decision. Both halves used to be assembled twice: once by
``cdui plugin info``, printing as it went, and once, differently, by the
install path.

So the assembling happens here and produces a value. :class:`Inspection` is
flat and fully populated -- absent things are ``()``, ``{}`` or ``""``, and
``None`` only where it means "this kind of source does not have one" -- so a
template can render it without asking what shape a field is this time, and a
route can serialise it without a second pass.

Two rules the shape encodes:

* ``installed`` is the LOCKFILE's record, never the manifest's. A plugin
  that rewrote its manifest after install would otherwise be shown as if
  what it now asks for is what the user agreed to.
* ``capabilities_added`` and ``allowed_modules_added`` are the whole reason
  an update is worth inspecting: capability creep between the version
  somebody consented to and the one about to replace it is the supply-chain
  shape a plugin manager can actually catch.

The lockfile arrives as an argument rather than being read here, because the
caller already has one -- and because a test that wants to describe an
update should be able to say so in a dict instead of writing a file.

Nothing here prints, prompts or installs. :data:`FRONTEND_WARNING` and
:data:`ALLOWED_MODULES_WARNING` are English, one sentence each, and a caller
that speaks another language is expected to key off ``has_frontend`` and
``allowed_modules`` instead of translating them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

from app.core import plugin_loader

from . import catalog as catalog_module
from . import github, listing, sources
from .errors import (
    NotInstalled,
    NotUpdatable,
    PluginInstallError,
    ReservedPluginId,
)
from .manifest import (
    manifest_allowed_modules,
    manifest_capabilities,
    manifest_has_frontend,
    manifest_python_deps,
    read_manifest,
    validate_manifest,
)
from .sources import _GITHUB_SHORT, _GITHUB_URL

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport -- same API.

#: Said whenever a plugin ships browser code. Its own warning because the AST
#: gate has nothing to say about JavaScript: frontend code runs inside the
#: editor, in the user's session, with everything the editor can reach.
FRONTEND_WARNING = (
    "This plugin ships JavaScript that runs in the editor with full access."
)

#: Said whenever a plugin declares ``allowed_modules``. Those imports are the
#: gate being switched off by name, which is a decision about the AUTHOR
#: rather than about the code.
ALLOWED_MODULES_WARNING = (
    "This plugin asks to import: {modules}. "
    "Installing requires trusting the author."
)

#: What to do instead, per ``source_kind`` that has no update to look for.
#: English and one sentence each, like the warnings above: a caller speaking
#: another language branches on :attr:`~.errors.NotUpdatable.source_kind`.
NO_UPDATE_HINTS = {
    "builtin": (
        "A pack that ships in this release updates with CodefyUI itself: "
        "run cdui update."
    ),
    "local": (
        "A linked directory is whatever is on its author's disk right now; "
        "there is nothing to fetch."
    ),
}


@dataclass(frozen=True)
class Inspection:
    """One source, read and compared against what this install already has."""

    kind: Literal["builtin", "github"]
    mode: Literal["install", "update"]
    plugin_id: str
    catalog_id: str | None
    official: bool
    source: str
    #: The repository this was read from, split, and ``None`` for a kind
    #: that has none. Kept apart from ``source``/``url`` because an install
    #: needs the pair back -- ``download_tarball`` takes an owner and a repo,
    #: not a URL -- and re-parsing a formatted string to recover what the
    #: caller already had is how the two spellings drift apart.
    owner: str | None
    repo: str | None
    url: str | None
    ref: str | None
    sha: str | None
    name: str
    version: str
    description: str
    homepage: str
    manifest: dict[str, Any]
    capabilities: tuple[str, ...]
    allowed_modules: tuple[str, ...]
    python_deps: dict[str, str]
    has_frontend: bool
    chapters: tuple[str, ...]
    lessons: tuple[str, ...]
    consent_required: bool
    installed: dict[str, Any] | None
    up_to_date: bool
    capabilities_added: tuple[str, ...]
    allowed_modules_added: tuple[str, ...]
    warnings: tuple[str, ...]


def _text(value: Any) -> str:
    """*value* when it is a string, else the empty string."""
    return value if isinstance(value, str) else ""


def _strings(value: Any) -> tuple[str, ...]:
    """The string members of *value* when it is a list, else empty."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _lockfile_entry(lockfile: dict[str, Any], plugin_id: str) -> dict[str, Any] | None:
    """This install's record for *plugin_id*, or ``None``."""
    plugins = lockfile.get("plugins")
    if not isinstance(plugins, dict):
        return None
    entry = plugins.get(plugin_id)
    return entry if isinstance(entry, dict) else None


def _installed_record(entry: dict[str, Any]) -> dict[str, Any]:
    """The lockfile fields a reader has to see before deciding on an update.

    ``version`` comes out of the recorded ``manifest`` table rather than off
    the disk: it is the version the user consented to, which is the only one
    worth comparing the incoming manifest against.
    """
    recorded = entry.get("manifest")
    recorded = recorded if isinstance(recorded, dict) else {}
    return {
        "sha": entry.get("sha"),
        "version": _text(recorded.get("version")),
        "capabilities": _strings(entry.get("capabilities")),
        "trusted_modules": _strings(entry.get("trusted_modules")),
        "enabled": plugin_loader.is_enabled(entry),
        "source_kind": _text(entry.get("source_kind")),
    }


def _inspection(
    *,
    kind: Literal["builtin", "github"],
    plugin_id: str,
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    source: str,
    owner: str | None = None,
    repo: str | None = None,
    url: str | None = None,
    ref: str | None = None,
    sha: str | None = None,
    catalog_id: str | None = None,
    official: bool = False,
    consent_required: bool,
) -> Inspection:
    """Assemble one :class:`Inspection` from a manifest and a lockfile.

    The one place the two halves meet, so that "what is new since the
    install" is computed identically for a catalog pack and a repository.
    """
    plugin_meta = manifest.get("plugin")
    plugin_meta = plugin_meta if isinstance(plugin_meta, dict) else {}
    lessons_meta = manifest.get("lessons")
    lessons_meta = lessons_meta if isinstance(lessons_meta, dict) else {}

    capabilities = manifest_capabilities(manifest)
    allowed_modules = tuple(manifest_allowed_modules(manifest))
    has_frontend = manifest_has_frontend(manifest)

    entry = _lockfile_entry(lockfile, plugin_id)
    installed = _installed_record(entry) if entry is not None else None
    if installed is None:
        capabilities_added: tuple[str, ...] = ()
        allowed_modules_added: tuple[str, ...] = ()
        up_to_date = False
    else:
        known_caps = set(installed["capabilities"])
        known_modules = set(installed["trusted_modules"])
        capabilities_added = tuple(c for c in capabilities if c not in known_caps)
        allowed_modules_added = tuple(
            m for m in allowed_modules if m not in known_modules
        )
        # A built-in pack has no sha on either side, and two ``None`` is the
        # honest "the copy on disk is the copy you installed".
        up_to_date = (installed["sha"] or None) == (sha or None)

    warnings: list[str] = []
    if has_frontend:
        warnings.append(FRONTEND_WARNING)
    if allowed_modules:
        warnings.append(
            ALLOWED_MODULES_WARNING.format(modules=", ".join(allowed_modules))
        )

    return Inspection(
        kind=kind,
        mode="update" if installed is not None else "install",
        plugin_id=plugin_id,
        catalog_id=catalog_id,
        official=official,
        source=source,
        owner=owner,
        repo=repo,
        url=url,
        ref=ref,
        sha=sha,
        name=_text(plugin_meta.get("name")) or plugin_id,
        version=_text(plugin_meta.get("version")),
        description=_text(plugin_meta.get("description")),
        homepage=_text(plugin_meta.get("homepage")),
        manifest=manifest,
        capabilities=capabilities,
        allowed_modules=allowed_modules,
        python_deps=manifest_python_deps(manifest),
        has_frontend=has_frontend,
        chapters=_strings(lessons_meta.get("chapters")),
        lessons=_strings(lessons_meta.get("lessons")),
        consent_required=consent_required,
        installed=installed,
        up_to_date=up_to_date,
        capabilities_added=capabilities_added,
        allowed_modules_added=allowed_modules_added,
        warnings=tuple(warnings),
    )


def inspect_builtin(plugin_id: str, *, lockfile: dict[str, Any]) -> Inspection:
    """Describe a pack that ships in this release, read from disk.

    ``consent_required`` is False and ``official`` is True, and both are
    statements about provenance rather than about the code: a built-in pack
    arrived with the release, through a pull request in this repository, so
    there is no third party here for the user to be asked about. The
    capabilities it declares are still reported -- ``cdui plugin list`` has
    to answer "which of my plugins reaches the network" for every pack,
    wherever it came from.

    Raises ``FileNotFoundError`` when the directory has no manifest and
    :class:`~.errors.ManifestError` when it has one this build will not
    install.
    """
    manifest = read_manifest(plugin_loader.plugins_builtin_root() / plugin_id)
    validate_manifest(manifest)
    return _inspection(
        kind="builtin",
        plugin_id=plugin_id,
        manifest=manifest,
        lockfile=lockfile,
        source=plugin_id,
        # A pack that ships in this release has no repository to name: its
        # files are already on disk, and an install neither resolves nor
        # downloads anything.
        owner=None,
        repo=None,
        catalog_id=plugin_id,
        official=True,
        consent_required=False,
    )


def inspect_github(
    owner: str,
    repo: str,
    ref: str,
    *,
    lockfile: dict[str, Any],
    pinned_sha: str | None = None,
    catalog_id: str | None = None,
    official: bool = False,
) -> Inspection:
    """Describe a repository at one commit, without downloading it.

    The ref is resolved to a full SHA first and the manifest is read AT that
    SHA, so what the user is shown and what an install would fetch are the
    same commit -- a branch that moves between the two is exactly the
    substitution this ordering removes. *pinned_sha* skips the resolve for a
    restore, where re-resolving a tag that has since moved would defeat the
    pin.

    Refuses an id this build reserves, and the decision is
    :func:`~.catalog.reserved_id_holder`'s rather than this function's: a
    route name, a pack that ships here, or a ``github`` catalog row belonging
    to a DIFFERENT repository. That last clause is why the repository is
    passed in at all -- a catalog ``github`` id is not reserved outright,
    because refusing it would make the official pack the one thing nobody can
    install, but a fork may not take its place.

    Raises :class:`~.errors.GitHubError`, ``tomllib.TOMLDecodeError``,
    :class:`~.errors.ManifestError` or :class:`~.errors.ReservedPluginId`
    (a :class:`~.errors.PluginInstallError`, so every caller that already
    catches the base keeps catching this).
    """
    sha = pinned_sha or github.resolve_sha(owner, repo, ref)
    manifest = tomllib.loads(github.fetch_manifest_text(owner, repo, sha))
    validate_manifest(manifest)
    plugin_id = manifest["plugin"]["id"]

    taken_by = catalog_module.reserved_id_holder(plugin_id, owner=owner, repo=repo)
    if taken_by is not None:
        # The message and the hint are what the CLI prints, word for word;
        # the attributes are what a panel draws a control from. Both, rather
        # than either -- a reader of the terminal needs the sentence, and a
        # client that had to recover the id from it would be parsing English.
        raise ReservedPluginId(
            f"Plugin id {plugin_id!r} is reserved by this build.",
            plugin_id=plugin_id,
            taken_by=taken_by,
            hint=(
                f"{owner}/{repo} declares an id that names {taken_by}; it "
                f"cannot be installed under that id."
            ),
        )

    return _inspection(
        kind="github",
        plugin_id=plugin_id,
        manifest=manifest,
        lockfile=lockfile,
        source=f"{owner}/{repo}" + (f"@{ref}" if ref else ""),
        owner=owner,
        repo=repo,
        url=f"https://github.com/{owner}/{repo}",
        ref=ref,
        sha=sha,
        catalog_id=catalog_id,
        official=official,
        consent_required=True,
    )


def inspect_source(spec: str, *, lockfile: dict[str, Any]) -> Inspection:
    """Describe whatever the user typed: a catalog name, ``owner/repo``, a URL.

    The dispatch is :func:`~.sources.parse_source`'s, so the Plugin Center
    and ``cdui plugin install`` agree about what a string means. A catalog
    entry carries its id and its ``official`` flag into the answer; free text
    cannot, and gets ``catalog_id=None`` with ``official=False``, because
    "official" is a claim only the catalog is entitled to make.
    """
    parsed = sources.parse_source(spec)
    if parsed.kind != "catalog":
        return inspect_github(
            parsed.name_or_owner, parsed.repo, parsed.ref, lockfile=lockfile
        )

    entry = catalog_module.catalog_entry(parsed.name_or_owner)
    if entry is None:
        # parse_source matched the raw registry and validate_catalog dropped
        # the row. The pack is named but not installable, and saying which of
        # those it is beats either half on its own.
        raise PluginInstallError(
            f"The catalog entry for {parsed.name_or_owner!r} is malformed.",
            hint=f"Its row in {catalog_module.catalog_path()} was not readable.",
        )
    if entry.kind == "builtin":
        return inspect_builtin(entry.id, lockfile=lockfile)
    owner, _, repo = (entry.repo or "").partition("/")
    return inspect_github(
        owner,
        repo,
        entry.ref,
        lockfile=lockfile,
        catalog_id=entry.id,
        official=entry.official,
    )


def updatable_entry(plugin_id: str, *, lockfile: dict[str, Any]) -> dict[str, Any]:
    """This install's record for a plugin that CAN be updated. Reads nothing.

    Only ``github_url`` plugins have an update to look for: a built-in pack
    updates with CodefyUI itself, and a linked development directory is
    whatever is on the developer's disk right now.

    Public because the question is asked at two different moments.
    :func:`inspect_installed` asks it on the worker thread that is about to
    read GitHub; ``PluginService.update`` asks it on the event loop, BEFORE
    it claims the inspection slot -- "you do not have that plugin" and "a
    built-in pack updates with CodefyUI" are true without a network, and
    neither should have to queue behind somebody else's read to be said. One
    function, so which plugins have an update at all is decided once.

    :raises NotInstalled: no lockfile entry under that id.
    :raises NotUpdatable: it is installed, from something that cannot be
        re-fetched. ``hint`` says what to do instead.
    """
    entry = _lockfile_entry(lockfile, plugin_id)
    if entry is None:
        raise NotInstalled(
            f"Plugin {plugin_id!r} is not installed.", plugin_id=plugin_id
        )
    source_kind = _text(entry.get("source_kind"))
    if source_kind != "github_url":
        raise NotUpdatable(
            f"Plugin {plugin_id!r} does not update from a repository.",
            plugin_id=plugin_id,
            source_kind=source_kind,
            hint=NO_UPDATE_HINTS.get(
                source_kind, f"Its source_kind is {source_kind!r}."
            ),
        )
    return entry


def inspect_installed(plugin_id: str, *, lockfile: dict[str, Any]) -> Inspection:
    """Describe the update available for an installed repository plugin.

    The source is taken from the lockfile rather than re-typed, which is what
    makes ``cdui plugin update`` and the Plugin Center's update button follow
    the same repository and the same ref the user originally agreed to.
    :func:`updatable_entry` decides whether there is one to follow.

    Provenance is carried forward rather than dropped. Re-inspecting is the
    same plugin seen a second time, so an update dialog must be able to say
    "this is the catalog's own pack" exactly where the install did: the
    recorded ``catalog_id`` is passed straight through, and ``official`` is
    :func:`~.listing.is_official` -- the same rule the Plugin Center's badge
    uses, so a row cannot be official in the panel and unofficial in the
    dialog that updates it. Without this, every update of an official plugin
    read as third-party free text.
    """
    entry = updatable_entry(plugin_id, lockfile=lockfile)
    url = _text(entry.get("url"))
    match = _GITHUB_URL.match(url) or _GITHUB_SHORT.match(_text(entry.get("source")))
    if match is None:
        # Installed from a repository whose name is no longer in the record:
        # a hand-edited lockfile, or one written by a build that spelled the
        # url differently. Still "nothing to fetch", so it is the same
        # refusal -- with the two fields that were looked at, because this
        # one is a bug report rather than a decision.
        raise NotUpdatable(
            f"Cannot tell which repository {plugin_id!r} came from.",
            plugin_id=plugin_id,
            source_kind="github_url",
            hint=f"url={entry.get('url')!r} source={entry.get('source')!r}",
        )
    recorded_catalog_id = _text(entry.get("catalog_id")) or None
    return inspect_github(
        match.group(1),
        match.group(2),
        _text(entry.get("ref")),
        lockfile=lockfile,
        catalog_id=recorded_catalog_id,
        official=listing.is_official(catalog_module.catalog_entry(plugin_id), entry),
    )
