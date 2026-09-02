"""What the user typed, turned into something the installer can act on.

One field accepts four shapes -- ``deep``, ``alice/extras``,
``alice/extras@v1.2.3``, ``https://github.com/alice/extras`` -- because a
person who has a plugin in mind should not have to know which of the two
install paths it goes down. Sorting that out is this module's whole job, and
it is deliberately the only place that knows the answer: the CLI and the
routes both get the same verdict on the same string.

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
from pathlib import Path
from typing import Any, Literal, NamedTuple

from . import catalog as catalog_module
from .errors import UnknownCatalogName, UnparseableSource

# Accepts owner/repo or owner/repo@ref; owner/repo names are GitHub-permissible.
_GITHUB_SHORT = re.compile(r"^([\w.-]+)/([\w.-]+?)(?:@([\w./-]+))?$")
_GITHUB_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?(?:@(.+))?$"
)
# One word, no slash and no scheme: the only thing it can have been meant as is
# a catalog name. See UnknownCatalogName for why that deserves its own error.
_BARE_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*$")


class ParsedSource(NamedTuple):
    """Where a plugin is to come from.

    ``name_or_owner`` carries the plugin id for a catalog source and the
    GitHub owner for a repository one -- one field because the two are
    mutually exclusive and the ``kind`` says which is meant. ``repo`` and
    ``ref`` are empty strings for a catalog source, and ``ref`` is empty for a
    repository whose default branch is wanted.
    """

    kind: Literal["catalog", "github"]
    name_or_owner: str
    repo: str
    ref: str


def parse_source(
    spec: str,
    *,
    catalog: dict[str, Any] | None = None,
    catalog_path: Path | None = None,
) -> ParsedSource:
    """Resolve *spec* to a :class:`ParsedSource`, or raise a
    :class:`~.errors.SourceError`.

    Catalog lookup is case-insensitive; the id in the result is lower-cased,
    because that is the directory name the pack lives under.

    *catalog* and *catalog_path* override what is read from disk. They exist
    for ``scripts/plugins.py``, whose tests fake the catalog by patching the
    CLI's own ``load_catalog``: without them this would reach past that patch
    and answer from the real ``registry.json``.
    """
    data = catalog_module.load_catalog() if catalog is None else catalog
    plugins = data.get("plugins", {})

    if spec.lower() in plugins:
        return ParsedSource("catalog", spec.lower(), "", "")

    m = _GITHUB_URL.match(spec)
    if m:
        return ParsedSource("github", m.group(1), m.group(2), m.group(3) or "")

    m = _GITHUB_SHORT.match(spec)
    if m:
        return ParsedSource("github", m.group(1), m.group(2), m.group(3) or "")

    if _BARE_NAME.match(spec):
        raise UnknownCatalogName(
            spec,
            sorted(plugins),
            catalog_module.catalog_path() if catalog_path is None else catalog_path,
        )

    raise UnparseableSource(spec)
