"""One spelling per directory, and "is it inside" that a spelling cannot fool.

macOS and Windows filesystems are case-insensitive by default, and
``Path.resolve()`` keeps whatever case it was given: ``/Users/me/PROJ`` and
``/Users/me/proj`` resolve to two different strings for one folder. Phase 4A
keys a project by its path (the registry, ``trust.json``, the id a window
sends, the root a run was made in) and refuses some locations by path, so
a second spelling would be a second project and a way around a refusal.

:func:`canonical_path` gives the spelling the directory listing has, and
:func:`is_within` compares by identity (``os.path.samefile``) as well as by
string, for what canonical spelling still cannot see (Unicode normalisation,
Windows 8.3 short names).

No imports from the rest of the app: ``config`` uses this too.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def canonical_path(path: Path | str) -> Path:
    """*path* expanded, resolved, and spelled the way the disk spells it.

    Each existing component is replaced by the entry its parent directory
    lists for it, when the given spelling is not listed itself; the first
    component that does not exist, and everything after it, is kept as
    given (``create`` asks about a folder it has yet to make). On a
    case-sensitive filesystem a name that is not listed does not exist, so
    nothing changes there.
    """
    resolved = Path(path).expanduser().resolve()
    parts = resolved.parts
    if not parts:
        return resolved
    out = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        candidate = out / part
        try:
            exists = candidate.exists()
            names = os.listdir(out) if exists else []
        except OSError:
            return out.joinpath(*parts[index:])
        if not exists:
            return out.joinpath(*parts[index:])
        if part not in names:
            folded = _fold(part)
            matches = [name for name in names if _fold(name) == folded]
            if len(matches) == 1:
                part = matches[0]
        out = out / part
    return out


def _existing_ancestors(path: Path) -> list[Path]:
    """*path* (when it exists) and each existing parent, nearest first."""
    chain = [path, *path.parents]
    return [p for p in chain if os.path.lexists(p)]


def is_within(path: Path, root: Path) -> bool:
    """True when *path* is *root* or inside it.

    By string on the canonical spellings, and then by identity: if *path*
    or any of its existing ancestors is the same directory as *root*, it is
    inside, whatever either is called.
    """
    path, root = canonical_path(path), canonical_path(root)
    try:
        path.relative_to(root)
        return True
    except ValueError:
        pass
    try:
        root_stat = os.stat(root)
    except OSError:
        return False
    for ancestor in _existing_ancestors(path):
        try:
            st = os.stat(ancestor)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (root_stat.st_dev, root_stat.st_ino):
            return True
    return False


def same_dir(a: Path, b: Path) -> bool:
    """True when *a* and *b* name one directory (both must exist)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return canonical_path(a) == canonical_path(b)
