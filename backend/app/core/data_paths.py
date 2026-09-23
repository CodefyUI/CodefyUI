"""Which files under the data root a node may write to (#224), which name a
route may take from a client (#483), and which names a file can be stored
under at all (#520).

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
What a stored file may be CALLED is a third rule, :func:`check_file_name`
(#520), explained where it is defined. Everything below is about the node
rule.

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

import os
from pathlib import Path, PurePosixPath
from typing import Any

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


# ── which names a file can be stored under (#520) ───────────────────────

#: What makes a name a PATH rather than a filename: both separators and the
#: colon, on every platform. ``WindowsPath`` honours the backslash and
#: ``PosixPath`` does not, so a rule that asked the host which one to care
#: about would be a rule CI (Linux) cannot check on behalf of the users
#: (Windows). The colon does two jobs of its own on Windows: after one
#: letter it is a drive, which turned the upload ``a:b.csv`` into ``b.csv``;
#: after a longer prefix it names an NTFS alternate data stream, so
#: ``run 12:30.csv`` was written into a stream of a zero-byte file
#: ``run 12`` that no list shows.
_SEPARATOR_CHARACTERS = "/\\:"

#: The rest of what Windows refuses anywhere in a file name. There the write
#: itself fails (every upload route answered 500); on Linux and macOS it
#: succeeds, and the file then cannot be cloned, copied or synced onto
#: Windows.
_RESERVED_CHARACTERS = '<>"|?*'

#: The digits Windows reads in a COM or LPT device name: the superscripts too.
_DEVICE_DIGITS = ("123456789\N{SUPERSCRIPT ONE}\N{SUPERSCRIPT TWO}"
                  "\N{SUPERSCRIPT THREE}")

#: Names Windows resolves to a DEVICE rather than a file, whatever extension
#: follows them: ``com1.json`` opens the serial port, and before Windows 11
#: ``nul.csv`` swallowed whatever was written to it. The set is the one
#: ``ntpath.isreserved`` (Python 3.13) holds, which covers every Windows
#: version: the two console names, and COM/LPT with superscript digits.
#: Lowercase, because the name is lowercased before it is looked up here.
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{digit}" for digit in _DEVICE_DIGITS}
    | {f"lpt{digit}" for digit in _DEVICE_DIGITS}
)

#: The longest name a file can be stored under. ext4 and the other Linux file
#: systems count UTF-8 bytes, NTFS counts UTF-16 code units, and a name within
#: 255 bytes is within 255 units, so the stored-name rule measures bytes and
#: holds on every server. Past it the write failed and every upload route
#: answered 500; a name of 90 Chinese characters (270 bytes) stores on NTFS
#: and not on ext4.
_NAME_LIMIT = 255

#: Whether a lookup refuses the characters Windows cannot store, beyond the
#: control characters every server refuses. On a Windows server such a name
#: addresses something other than the file it spells: ``a:b.csv`` a drive,
#: ``run 12:30.csv`` a stream of ``run 12``, ``x\b.csv`` a file inside ``x``.
#: On Linux and macOS it is an ordinary name a stored file can have --
#: uploaded before #520, written by a node such as ModelSaver, placed there by
#: hand -- and :func:`resolve_under` keeps the lookup inside its directory
#: anyway. A module constant so tests can take either branch on any host.
_LOOKUP_REFUSES_WINDOWS_CHARACTERS = os.name == "nt"

_RENAME = "Rename the file and try again."


class UnstorableName(ValueError):
    """Why no file can be stored under a name on every OS CodefyUI runs on.

    One refusal, read two ways. ``str()`` is an English sentence naming what
    is wrong: the upload, download, delete and toggle routes answer it as
    their ``detail``, and the editor shows it as written. ``code`` and
    ``fields`` are the same refusal for a client that writes its own
    sentence in the user's language: ``routes_presets`` answers
    ``{"detail": {"code": ..., **fields}}`` (#476) and the toolbar translates
    it.
    """

    def __init__(self, code: str, message: str, **fields: Any) -> None:
        super().__init__(message)
        self.code = code
        self.fields = fields


def _character_refusal(character: str) -> UnstorableName | None:
    """The refusal of one character of a name, or None when it may stay."""
    codepoint = ord(character)
    if codepoint < 32 or codepoint == 127:
        # Invisible, so the codepoint is the only way to say which one.
        return UnstorableName(
            "name_control_character",
            f"File names cannot contain the control character "
            f"U+{codepoint:04X}. {_RENAME}",
            codepoint=codepoint,
        )
    if character in _SEPARATOR_CHARACTERS:
        code = "name_separator"
    elif character in _RESERVED_CHARACTERS:
        code = "name_reserved_character"
    else:
        return None
    return UnstorableName(
        code,
        f"File names cannot contain '{character}', which Windows does not "
        f"allow. {_RENAME}",
        character=character,
    )


def _utf8_size(name: str) -> int:
    """*name*'s length in UTF-8 bytes. A lone surrogate, which a JSON string
    can carry, is counted rather than raising."""
    return len(name.encode("utf-8", "surrogatepass"))


def _too_long(size: int) -> UnstorableName:
    """The refusal of a name *size* bytes long."""
    return UnstorableName(
        "name_too_long",
        f"File names can be at most {_NAME_LIMIT} bytes long, and this one is "
        f"{size} (a Chinese character takes 3 bytes). Shorten it and try "
        f"again.",
        limit=_NAME_LIMIT,
    )


def _check_characters(name: str) -> None:
    """Raise the refusal of the first character of *name* that no file name
    may hold."""
    for character in name:
        refusal = _character_refusal(character)
        if refusal is not None:
            raise refusal


def check_file_name(name: str, *, stored_as: str | None = None) -> None:
    """Raise :class:`UnstorableName` unless a file can be stored under *name*
    on every OS (#520).

    The rule for a name a file is about to be CREATED under: an upload, a
    preset. It is the rule #476 wrote for presets, moved here so that the
    upload routes, which had none, share it. It is the same on every OS on
    purpose: a name Linux accepts and Windows does not makes a project that
    cannot move between them, and in project mode uploaded images land in
    the project directory, which Windows machines clone too.

    *stored_as* is the file name really written, when the caller derives one
    from *name*: a preset called ``My Block`` is stored as ``my_block.json``.
    The characters are checked in *name*, so a refusal points at one the
    user typed; the length and the device name in the file written, which is
    what the file system holds.

    Refused, in this order, so that the problem reported is the first one:

    - an empty or blank name;
    - a separator or colon, a control character, or a character Windows
      reserves;
    - a name over 255 bytes of UTF-8;
    - a name made only of dots: ``.`` and ``..`` are directories, and
      Windows drops trailing dots, which leaves ``...`` no name at all;
    - a Windows device name before the first dot, spaces before the dot
      included (``con``, ``com1.csv``, ``nul .txt``).

    A trailing dot or space is not refused, although Windows drops it: every
    upload route's extension check refuses such a name already, and a
    preset's file always ends in ``.json``.
    """
    stored = name if stored_as is None else stored_as
    if not name.strip():
        raise UnstorableName("name_empty", "The file name is empty.")
    _check_characters(name)
    size = _utf8_size(stored)
    if size > _NAME_LIMIT:
        raise _too_long(size)
    if set(name.strip()) == {"."}:
        raise UnstorableName(
            "name_dot_segment",
            f"A file name cannot be made only of dots. {_RENAME}")
    # Windows resolves a device name from the part before the FIRST dot, and
    # drops the spaces in front of that dot.
    device = stored.partition(".")[0].rstrip(" ").lower()
    if device in _WINDOWS_DEVICE_NAMES:
        raise UnstorableName(
            "name_reserved_device",
            f"'{device}' is a name Windows keeps for a device, so no file "
            f"can be called that. {_RENAME}",
            reserved=device,
        )


def check_lookup_name(name: object) -> None:
    """Raise :class:`UnstorableName` for a name no file on THIS server can
    have, before a route looks one up by it (#520).

    Looser than :func:`check_file_name` on purpose: a lookup must keep
    reaching every file that is there, whatever put it there.

    - ``/`` separates the parts of a nested name: a model saved into
      ``runs/exp1/`` is downloaded by that path.
    - A control character is refused on every server.
    - The rest of what Windows cannot store is refused on a Windows server
      only (:data:`_LOOKUP_REFUSES_WINDOWS_CHARACTERS`).

    No length is refused. File systems measure a name differently -- ext4 in
    UTF-8 bytes, NTFS, HFS+ and APFS in characters or UTF-16 units -- so any
    one limit here would refuse a file some volume really holds, and a name
    too long for the volume in use is answered by :func:`lookup_exists`.
    An empty, dots-only or device name is not refused either: the routes'
    own checks and :func:`resolve_under` answer those, and ``....`` is an
    ordinary directory name on Linux. A *name* that is not a string is left
    alone: :func:`resolve_under` refuses it, as the "Invalid filename" it
    always was (#483).
    """
    if not isinstance(name, str):
        return
    for character in name:
        if character == "/":
            continue
        refusal = _character_refusal(character)
        if refusal is not None and (
                _LOOKUP_REFUSES_WINDOWS_CHARACTERS
                or refusal.code == "name_control_character"):
            raise refusal


def lookup_exists(path: Path) -> bool:
    """Whether the file a lookup names is there; False when the OS will not
    say (#520).

    ``Path.exists`` raises, instead of answering False, for an error it does
    not expect. On Python 3.10-3.12 that includes ENAMETOOLONG, a name part
    longer than the volume holds, so such a lookup answered 500. No file can
    be there under that name, so "not found" is the true answer, and it is
    the answer whatever the volume counts in.
    """
    try:
        return path.exists()
    except OSError:
        return False


def upload_file_name(filename: str) -> str:
    """The name an uploaded file is stored under, or :class:`UnstorableName`.

    The part after the last ``/``: a browser never sends one, and a name
    that arrives with one (``../../evil.csv``) is a path whose last part is
    the name. ``PurePosixPath`` rather than ``Path`` on purpose: a
    ``WindowsPath`` also cut at a backslash and after a drive letter, so on
    a Windows server ``a:b.csv`` and ``x\\b.csv`` were stored as ``b.csv``,
    over whatever had that name. Here those characters stay in the name and
    :func:`check_file_name` refuses them, the same on every OS.
    """
    name = PurePosixPath(filename).name
    check_file_name(name)
    return name
