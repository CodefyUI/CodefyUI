"""Which files under the data root a node may write to (#224), and which
name a route may take from a client (#483).

Every node that turns a graph parameter into a filesystem write shares one
rule, and it lives here so there is exactly one copy of it. Before #224
there were three: ``core.checkpoints.resolve_checkpoint_path``,
``nodes/io/model_saver_node.py`` and ``nodes/io/image_writer_node.py`` each
open-coded "resolve, then require ``MODELS_DIR.parent``", which meant three
places to fix and three places to forget. The routes that take a file name
from a client had the same problem with a narrower rule, "this name, inside
this one directory", spelled out eight times across ``app/api``; that rule
is :func:`resolve_under` (#483). It is not the node rule: several of those
directories (custom nodes, presets, examples) are outside the data root.
Everything below is about the node rule.

The rule
--------
1. A relative path is taken as relative to the caller's *base* directory
   (``MODELS_DIR`` for checkpoints and saved models, ``<data>/output`` for
   written images) -- the caller passes it, because the defaults differ.
2. The resolved result must stay under the **data root**,
   ``MODELS_DIR.parent``. Unchanged: this is what has always kept a graph
   parameter from writing to an arbitrary place on the machine.
3. The resolved result must not be one of :func:`protected_paths` --
   CodefyUI's own storage, which is under the data root but is engine state
   rather than anything a node produces.

Rule 3 is what #224 adds. Rules 1-2 alone let a graph parameter name the
run database: with the default ``cdui start`` and no ``--project``,
``PROJECT_DIR`` is ``None``, the project-mode derivation in ``config.py``
never runs, and ``MODELS_DIR`` falls back to ``backend/data/models`` --
whose parent is the directory holding ``codefyui.db``. So
``path="../codefyui.db"`` on a ``CheckpointSaver`` or a ``ModelSaver``
resolved to the live database and a training run wrote a ``.pt`` payload
straight over it. No mislabelled artifact row and no plugin required --
just a path typed into a node parameter.

The issue this comes from argued the installed layout was narrower than
dev because ``MODELS_DIR`` is ``<project>/assets/models``. That is true in
project mode and only in project mode, which is not the default.

``ImageWriter`` is here for consistency, not because it was reachable the
same way: it forces the file extension to match its ``format`` parameter,
so ``../codefyui.db`` was rewritten to ``codefyui.png`` and written BESIDE
the database rather than over it. What it could do was overwrite any file
under the data root ending in an image extension, which is the same
over-broad containment rule and reason enough to share one definition of
it. Both writers that rewrite an extension re-validate afterwards, so the
path written is the path checked -- ``DB_PATH`` is env-overridable, and a
database named ``store.safetensors`` would otherwise be reachable through
exactly that rewrite.

What is protected, and what deliberately is not
-----------------------------------------------
:func:`protected_paths` names the SQLite database and its three possible
sidecars (``-wal``, ``-shm``, ``-journal`` -- ``core.db`` runs in WAL mode,
where truncating the ``-wal`` corrupts the database just as effectively as
truncating the main file). All four are derived from ``settings.DB_PATH``
at call time rather than hardcoded, so an installation that moves the
database keeps the protection.

``GRAPHS_DIR`` was considered and rejected. In project mode it already sits
outside the data root, so protecting it would change nothing there; in
default mode it is ``<data>/graphs``, which a node writing an image into it
would indeed clobber. But it is a DIRECTORY whose position relative to the
data root is configuration-dependent -- several tests already point
``GRAPHS_DIR`` at the same directory they use as the data root, and a
protected entry that can be configured to equal the root would deny every
legitimate write. ``DB_PATH`` is a file and can never degenerate that way.
Widening to directories wants an explicit containment invariant first.

Case folding, and why there is none here
----------------------------------------
``../CODEFYUI.DB-WAL`` names the same file as ``../codefyui.db-wal`` on
Windows and a different one on POSIX, and this module does nothing about
that -- because ``pathlib`` already does. Comparison and ``is_relative_to``
run on the platform flavour, and ``PureWindowsPath`` compares
case-insensitively. ``Path.resolve()`` is NOT enough on its own: it
canonicalises the case of components that exist on disk but falls back to
lexical normalisation for ones that do not, and the ``-wal`` sidecar is
absent whenever the database is closed. An earlier revision of this module
folded case explicitly through ``os.path.normcase``; mutation-testing the
guard showed removing that layer changed no outcome, because the comparison
underneath was already doing it. ``test_a_case_differing_spelling_of_the_
database_is_refused`` pins the behaviour rather than the mechanism, so this
stays correct if the mechanism ever needs to come back.
"""

from __future__ import annotations

from pathlib import Path

from ..config import settings

#: Suffixes SQLite appends to the database filename for its journals. WAL
#: mode (``core.db`` sets ``PRAGMA journal_mode=WAL``) uses the first two;
#: ``-journal`` is the rollback-journal name any non-WAL fallback would use.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def data_root() -> Path:
    """The directory a node's writes must stay inside.

    Read from ``settings`` on every call, never cached: tests monkeypatch
    ``MODELS_DIR``, and project mode repoints it during settings validation.
    """
    return settings.MODELS_DIR.parent.resolve()


def protected_paths() -> tuple[Path, ...]:
    """Files under the data root that no node may write to or delete.

    CodefyUI's own storage. See the module docstring for why this is the
    database and its sidecars specifically, and why it holds in both
    directions rather than only on the delete side.
    """
    db = settings.DB_PATH.resolve()
    return (db, *(db.with_name(db.name + s) for s in _DB_SIDECAR_SUFFIXES))


def is_protected(path: Path) -> bool:
    """True when *path* is part of CodefyUI's own storage.

    *path* must already be absolute and resolved. ``is_relative_to`` rather
    than ``==`` so the check keeps working if anything directory-shaped is
    ever added to :func:`protected_paths`; for a plain file the two agree.
    """
    return any(path.is_relative_to(reserved) for reserved in protected_paths())


def resolve_data_path(path: str | Path, *, base: Path) -> Path:
    """Absolute, validated destination for a node's file write.

    *base* is what a relative *path* is taken as relative to. Raises
    ``ValueError`` with a message meant for the user's node error panel:
    one for leaving the data root, a different one for naming CodefyUI's
    own storage, because those are different mistakes with different fixes.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base / resolved
    resolved = resolved.resolve()

    if not resolved.is_relative_to(data_root()):
        raise ValueError("Output path must be within the project data directory")
    if is_protected(resolved):
        raise ValueError(
            "Output path is part of CodefyUI's own storage (the run "
            f"database) and cannot be written by a node: {resolved}"
        )
    return resolved


def resolve_under(directory: Path, name: str, *,
                  direct_child: bool = False) -> Path | None:
    """*name* resolved inside *directory*, or None when it cannot be one.

    The routes' rule for a file name a client supplied (#483). Each route
    turns None into its own answer -- a 400 with its own detail, or a 404 --
    so the rule is written once and every answer stays what it was.
    Resolve-then-compare rather than a check on the string: ``..`` has
    several spellings over the wire, and a symlink is not one of them at all.
    The comparison is ``Path.is_relative_to``, never ``str.startswith``,
    which would put ``/repo/examples-evil`` inside ``/repo/examples``.

    None covers every way *name* can fail to be such a path:

    - It is not a string. A JSON body can hand a route anything, and
      ``directory / 5`` raises ``TypeError``.
    - It holds a NUL (``%00`` in a URL, ``\\u0000`` in JSON, the raw byte in a
      hand-built multipart filename), which no filesystem stores. Refused
      before ``resolve`` because ``resolve`` does not refuse it everywhere:
      it raises ``ValueError`` on POSIX and on Windows up to Python 3.12,
      but on Windows from 3.13 ``ntpath.realpath`` hands the path back
      unchanged (gh-106242) and the NUL fails only at the write.
    - ``resolve`` refuses it: ``OSError`` for a path the operating system
      will not look up, ``ValueError`` for one it cannot encode. Either one
      escaping turns the refusal into a 500 with a traceback in the log, for
      anyone who can reach the port -- several of these routes are open
      GETs.
    - It resolves outside *directory*.

    *direct_child* also refuses a nested path and *directory* itself:
    ``target.parent == base`` is the containment check and both refusals in
    one comparison. Presets need it, because the registry globs one flat
    directory. Media and model downloads must not have it, because they
    serve nested names by design. Without it an empty name resolves to
    *directory* itself, and the route's own "is it a file" check answers
    for that.
    """
    if not isinstance(name, str) or "\x00" in name:
        return None
    try:
        base = directory.resolve()
        target = (base / name).resolve()
    except (OSError, ValueError):
        return None
    if direct_child:
        return target if target.parent == base else None
    return target if target.is_relative_to(base) else None
