"""Optional pack support (Package Center). See catalog.py for the allowlist.

A pack is a curated bundle of pip packages and model files that a stock
CodefyUI install deliberately does NOT ship: the base install stays small
enough to hand to a classroom, and the four hundred megabytes a sentence
embedder needs arrive only when a lesson asks for them.

This package stays free of imports from ``app.api`` so node code and the
routes can both depend on it.

The four functions below are the whole surface a NODE needs: ask whether a
pack is there, refuse to run without it, and find what it downloaded. They
import ``state`` lazily so that ``from app.core.packs import PackMissingError``
-- which every pack-aware node does at import time -- stays a stdlib-only
import that touches neither the filesystem nor the import machinery.
"""

from __future__ import annotations

from pathlib import Path


class PackMissingError(RuntimeError):
    """A node needs an optional pack that is not installed.

    The message always ends with ``(pack=<id>)`` so the frontend can extract
    the id; ``pack_id`` carries it for Python callers.
    """

    def __init__(self, pack_id: str, message: str):
        self.pack_id = pack_id
        super().__init__(f"{message} (pack={pack_id})")


def pack_available(pack_id: str) -> bool:
    """Is this pack installed and usable right now?

    An unknown id is False rather than an error: a graph saved against a
    plugin that is no longer here should report a missing pack, not crash the
    run with a KeyError.
    """
    from . import state

    probed = state.probe_all().get(pack_id)
    return probed is not None and probed.installed


def require_pack(pack_id: str) -> None:
    """Refuse to run without *pack_id*, in words the editor can act on.

    A graph run NEVER downloads. Four hundred megabytes arriving mid-run, on
    a classroom connection, with no progress bar and no way to cancel, is not
    a thing a "Run" button may do -- so the failure names the one place that
    can install it, and the run stops there.
    """
    if pack_available(pack_id):
        return

    from .catalog import find_pack

    pack = find_pack(pack_id)
    title = pack.title if pack is not None else pack_id
    raise PackMissingError(
        pack_id,
        f"{title} is not installed. Open Package Center "
        "(toolbar > Settings > Optional packs) to install it; "
        "graph runs never download")


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
