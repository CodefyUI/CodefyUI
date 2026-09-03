"""One ``.gitignore`` and one ``.gitattributes``, whoever writes them.

A project directory can become a git repository two ways now -- ``cdui
project init`` at the command line, and the Source Control tab's
"Initialize Repository" -- and the interesting failure is not that one of
them writes the wrong text: it is that they DRIFT, so a project scaffolded
by the CLI and a project scaffolded by the tab ignore different files, and
the difference is only ever noticed by whoever commits their API keys.

So these tests pin the shared constants (identity, not content: a copy that
happens to match today is the drift these tests exist to prevent) and the
one line in each file that would cost something if it went missing.

``project`` here is ``scripts/project.py`` -- conftest puts ``scripts/`` on
``sys.path``.
"""

from __future__ import annotations

import argparse

import project

from app.core.project import PROJECT_GITATTRIBUTES, PROJECT_GITIGNORE


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace(dir=None, adopt=None, force=False)
    for key, value in kw.items():
        setattr(ns, key, value)
    return ns


def test_the_cli_writes_the_shared_scaffold():
    """The SAME objects, not two copies that agree.

    ``cdui project init`` is where these strings used to live; the tab's
    ``init`` needs the same ones, and a second copy would be a second
    ``.gitignore`` waiting to lose a line.
    """
    assert project.PROJECT_GITIGNORE is PROJECT_GITIGNORE
    assert project.PROJECT_GITATTRIBUTES is PROJECT_GITATTRIBUTES


def test_the_scaffolded_gitignore_still_hides_the_secrets(tmp_path):
    """``.env`` first, and the model formats after it.

    ``.env`` is the whole reason the tab refuses to serve that file at any
    ref: a secret that is never committed cannot be read out of history.
    """
    target = tmp_path / "svc"

    assert project.cmd_init(_args(dir=str(target))) == 0

    ignored = (target / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    assert "*.safetensors" in ignored
    # An interrupted atomic write leaves one of these behind; a project that
    # tracked them would commit half a graph.
    assert "*.tmp-*" in ignored


def test_the_scaffolded_gitattributes_pins_json_line_endings(tmp_path):
    """``*.json text eol=lf`` -- the line this change added.

    The graph and layout files ARE the project's source and the server
    writes them with ``\\n``. On a Windows checkout with
    ``core.autocrlf=true``, without this line every save rewrites every line
    of every graph, and a one-node change is reviewed as a whole-file diff.
    """
    target = tmp_path / "svc"

    assert project.cmd_init(_args(dir=str(target))) == 0

    attributes = (target / ".gitattributes").read_text(
        encoding="utf-8").splitlines()
    assert "*.json text eol=lf" in attributes
    assert "layout/*.layout.json linguist-generated=true" in attributes
