"""Optional pack support (Package Center). See catalog.py for the allowlist.

A pack is a curated bundle of pip packages and model files that a stock
CodefyUI install deliberately does NOT ship: the base install stays small
enough to hand to a classroom, and the four hundred megabytes a sentence
embedder needs arrive only when a lesson asks for them.

This package stays free of imports from ``app.api`` so node code and the
routes can both depend on it.

The functions below are the whole surface a NODE needs: read a requirement
written by a node author, ask whether the pack (or one particular model in
it) is there, refuse to run without it, and find what it downloaded. They
import ``state`` lazily so that ``from app.core.packs import PackMissingError``
-- which every pack-aware node does at import time -- stays a stdlib-only
import that touches neither the filesystem nor the import machinery.
"""

from __future__ import annotations

from pathlib import Path

# Re-exported so a caller catching an install failure never has to know which
# module inside the package raised it. ``errors`` is stdlib-only, so this
# keeps the promise above: importing this package still touches neither the
# filesystem nor the import machinery.
from .errors import (
    PackCancelled,
    PackInstallError,
    PackInsufficientDisk,
    PackNeedsRestart,
)

__all__ = [
    "PackCancelled",
    "PackInstallError",
    "PackInsufficientDisk",
    "PackMissingError",
    "PackNeedsRestart",
    "asset_path",
    "model_dir",
    "pack_available",
    "parse_requirement",
    "require_pack",
]


class PackMissingError(RuntimeError):
    """A node needs an optional pack that is not installed.

    The message always ends with ``(pack=<id>)`` so the frontend can extract
    the id; ``pack_id`` carries it for Python callers.
    """

    def __init__(self, pack_id: str, message: str):
        self.pack_id = pack_id
        super().__init__(f"{message} (pack={pack_id})")


def parse_requirement(value: str) -> tuple[str, str | None]:
    """Split a pack requirement into ``(pack_id, item_id | None)``.

    The convention a node author writes in ``REQUIRES_PACK`` or in an
    ``option_packs`` value: ``"rag"`` for "this pack", or
    ``"sentence-embeddings:all-MiniLM-L6-v2"`` when a SELECT option needs one
    particular model rather than any of the pack's four.

    Purely syntax -- it does not check that either id exists. A malformed
    value raises ``ValueError`` rather than returning an empty id, because
    gating on the pack whose id is the empty string fails open: no pack
    matches, so the check would quietly never pass and the node would report
    a missing pack nobody can install.
    """
    pack_id, separator, item_id = value.partition(":")
    pack_id, item_id = pack_id.strip(), item_id.strip()

    if not pack_id or ":" in item_id or (separator and not item_id):
        raise ValueError(
            f"malformed pack requirement {value!r}; "
            f"expected '<pack_id>' or '<pack_id>:<item_id>'")
    return pack_id, (item_id or None)


def pack_available(pack_id: str, item_id: str | None = None) -> bool:
    """Can a node run against this pack right now?

    Without *item_id*: the pack's packages are importable and it has
    something to run with -- either it downloads nothing, or at least ONE of
    its items is there. The four sentence-embedding models are alternatives,
    so a learner who fetched one of them is not short of anything.

    With *item_id*: the packages are importable AND that specific item is
    downloaded, which is what a SELECT option naming one model needs.

    An unknown pack or item is False rather than an error: a graph saved
    against a plugin that is no longer here should report a missing pack, not
    crash the run with a KeyError.
    """
    from . import state

    probed = state.probe_all().get(pack_id)
    if probed is None:
        return False
    if item_id is None:
        return probed.usable
    return probed.pip_ready and any(
        item.item_id == item_id and item.present for item in probed.items)


def require_pack(pack_id: str, item_id: str | None = None) -> None:
    """Refuse to run without this pack, in words the editor can act on.

    A graph run NEVER downloads. Four hundred megabytes arriving mid-run, on
    a classroom connection, with no progress bar and no way to cancel, is not
    a thing a "Run" button may do -- so the failure names the one place that
    can install it, and the run stops there.
    """
    if pack_available(pack_id, item_id):
        return

    from .catalog import find_pack

    pack = find_pack(pack_id)
    title = pack.title if pack is not None else pack_id
    if item_id is None:
        message = (f"{title} is not installed. Open Package Center "
                   "(toolbar > Settings > Optional packs) to install it; "
                   "graph runs never download")
    else:
        message = (f"Model '{item_id}' from the {title} pack is not "
                   "downloaded. Open Package Center "
                   "(toolbar > Settings > Optional packs) to download it; "
                   "graph runs never download")
    raise PackMissingError(pack_id, message)


def model_dir(repo_id: str) -> Path | None:
    """Where the catalog's snapshot of *repo_id* was downloaded, or None.

    Read fresh rather than from ``probe_all``'s cache: a node asking this is
    about to open the directory, and it must not be handed a path that was
    true one poll ago. ``ItemState.snapshot_dir`` is None unless the item is
    present, so a stale or half-finished download comes back as None here.
    """
    from . import state
    from .catalog import iter_packs

    for pack in iter_packs():
        for item in pack.items:
            if item.kind == "hf" and item.repo_id == repo_id:
                return state.item_state(pack, item).snapshot_dir
    return None


def asset_path(pack_id: str, filename: str) -> Path | None:
    """The downloaded file *filename* from *pack_id*, or None if it is not
    there. Unknown pack or unknown file is None, for the same reason
    ``pack_available`` is False."""
    from . import state
    from .catalog import find_pack

    pack = find_pack(pack_id)
    if pack is None:
        return None
    for item in pack.items:
        if item.kind == "asset" and item.filename == filename:
            return state.item_state(pack, item).snapshot_dir
    return None
