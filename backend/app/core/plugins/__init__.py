"""Plugin logic the server and the CLI both need, with neither one inside it.

Everything about a plugin used to live in ``scripts/plugins.py``: what a
manifest may say, what ``owner/repo@ref`` means, which packs the catalog
ships, and which files the AST gate will refuse. None of it was reachable
from the FastAPI app -- ``scripts/`` is only on ``sys.path`` when the CLI or
the test suite puts it there -- so a Plugin Center in the browser would have
had to grow a second copy of every one of those rules, and a second copy of a
SECURITY rule is the kind that drifts quietly.

So the rules moved here and ``cdui plugin`` became a front end over them. The
split is by ANSWER, not by caller: this package decides what is true (is this
manifest valid, what does this spec name, may this file be imported), and the
caller decides how to say it. That is why nothing in here formats bilingual
prose or prints -- the CLI's zh/en pairs stay in the CLI, the routes'
envelopes stay in the routes, and both read the same structured answer.

The submodules, in dependency order: :mod:`errors` (stdlib only),
:mod:`deps`, :mod:`consent`, :mod:`manifest`, :mod:`catalog`, :mod:`sources`,
:mod:`github`, :mod:`gate`, and :mod:`inspect`, which is the one that reads
a source through several of the others and answers "what would installing
this do?" in a single value. :mod:`flows` answers the question after that --
one install, step by step, run by the CLI and by the server's job through
the same code, so a terminal and the Plugin Center cannot disagree about
what installing a plugin does. Three more answer about an install that already
happened: :mod:`lifecycle` (enable, disable, uninstall -- the writes),
:mod:`listing` (one row per plugin, installed or not, for the Plugin Center)
and :mod:`reload` (the one re-discovery call every caller shares).

Only the error classes are re-exported here, so ``from app.core.plugins
import ManifestError`` stays a cheap import that pulls in neither the AST
validator nor the filesystem.
"""

from __future__ import annotations

from .errors import (
    AlreadyInstalled,
    ConsentRequired,
    GitHubError,
    InspectBusy,
    InspectionExpired,
    ManifestError,
    NotInstalled,
    NotUpdatable,
    PluginBusy,
    PluginCancelled,
    PluginInstallError,
    PluginNeedsRestart,
    ReservedPluginId,
    SourceError,
    TrustAuthorRequired,
    UnknownCatalogName,
    UnparseableSource,
)

__all__ = [
    "AlreadyInstalled",
    "ConsentRequired",
    "GitHubError",
    "InspectBusy",
    "InspectionExpired",
    "ManifestError",
    "NotInstalled",
    "NotUpdatable",
    "PluginBusy",
    "PluginCancelled",
    "PluginInstallError",
    "PluginNeedsRestart",
    "ReservedPluginId",
    "SourceError",
    "TrustAuthorRequired",
    "UnknownCatalogName",
    "UnparseableSource",
]
