"""Source Control: run the host's own git against the open project directory.

The tab in the editor is a front end for the git the user already has, with
the credentials they already have -- this app stores no tokens and knows no
remote. What it adds is a boundary: one process spawner (``runner``), one
error vocabulary (``errors``), one set of closed grammars for everything the
browser is allowed to name (``paths``), one set of wire shapes (``models``),
one parser for the format all of it turns on (``status``), and above them the
service and routes that decide what a request may ask for.

Only the two exception types are re-exported here. ``errors`` is stdlib-only,
so catching a git failure -- which any caller may want to do -- never drags
``subprocess`` or the runner's caches in with it, the same promise
``app.core.packs`` makes for pack installs.
"""

from __future__ import annotations

from .errors import GitBusy, GitError

__all__ = ["GitBusy", "GitError"]
