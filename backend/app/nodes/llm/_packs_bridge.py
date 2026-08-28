"""Thin, patchable seam between LLM nodes and ``app.core.packs``.

One module, for two reasons.

A node must stay IMPORTABLE in an install where the packs package is not
there. The registry imports every node module at startup to build the
palette; a node that raised ``ModuleNotFoundError`` at import time would not
just fail itself, it would take the whole palette down. So every import of
``app.core.packs`` below happens inside a function, and an install without
it reads as "nothing is available" rather than as a crash.

A node TEST must be able to say "pretend the pack is installed" with one
patch. Nodes call these names and nothing else, so a test patches
``app.nodes.llm._packs_bridge.pack_available`` (see the
``fake_sentence_transformers`` fixture in ``tests/conftest.py``) instead of
reaching into the probe cache, the sentinel files and the catalog.

The bridge adds no policy: it forwards arguments unchanged and returns what
the real function returned. ``requirement`` is the one exception, and it
touches no packs code at all -- it is string formatting that happens to be
the inverse of ``packs.parse_requirement``.
"""

from __future__ import annotations

from pathlib import Path


class PacksUnavailableError(RuntimeError):
    """``app.core.packs`` itself could not be imported.

    Deliberately shaped like ``PackMissingError``: same ``RuntimeError``
    base, same ``pack_id`` attribute, and a message that ends with
    ``(pack=<id>)`` so the editor can still read the id back and offer the
    Package Center. A learner hitting this sees a broken install, not a
    missing download -- but they see it as a message, not a traceback.
    """

    def __init__(self, pack_id: str, message: str):
        self.pack_id = pack_id
        super().__init__(f"{message} (pack={pack_id})")


def pack_available(pack_id: str, item_id: str | None = None) -> bool:
    """Can a node run against this pack right now? False if packs is absent."""
    try:
        from ...core.packs import pack_available as _impl
    except ImportError:
        return False
    return bool(_impl(pack_id, item_id))


def require_pack(pack_id: str, item_id: str | None = None) -> None:
    """Refuse to run without this pack.

    Delegates, so the message a learner reads -- which pack, which model,
    where to install it -- is written in exactly one place. The fallback is
    only for the install that has no packs package at all.
    """
    try:
        from ...core.packs import require_pack as _impl
    except ImportError as exc:
        # ``from exc`` deliberately: the frontend shows the message, and the
        # server log keeps the import that actually failed -- which is the
        # only clue to why an install has no packs package.
        raise PacksUnavailableError(
            pack_id,
            "Optional pack support is missing from this install, so the "
            f"'{pack_id}' pack cannot be checked or installed. Reinstall "
            "CodefyUI to restore it") from exc
    _impl(pack_id, item_id)


def model_dir(repo_id: str) -> Path | None:
    """Where the downloaded snapshot of *repo_id* is, or None."""
    try:
        from ...core.packs import model_dir as _impl
    except ImportError:
        return None
    return _impl(repo_id)


def asset_path(pack_id: str, filename: str) -> Path | None:
    """The downloaded file *filename* from *pack_id*, or None."""
    try:
        from ...core.packs import asset_path as _impl
    except ImportError:
        return None
    return _impl(pack_id, filename)


def record_derived(pack_id: str, item_id: str, path: Path) -> None:
    """Say that *path* was made out of *item_id*'s download and goes with it.

    The only WRITE on this bridge. A node that converts a download -- today
    just ``_glove`` turning the GloVe gzip into an npz -- has to leave a
    record, or ``remove_item`` deletes the 69 MB download, reports the space
    as freed, and leaves 83 MB behind that nothing will ever find again.

    A missing packs package is a no-op rather than an error, unlike the read
    helpers' ``None``: the caller has already written the file and its work
    is done, and an install with no packs package has no sentinel to record
    against and no Package Center to uninstall from either. Failing here
    would turn a bookkeeping gap into a failed graph run.
    """
    try:
        from ...core.packs import record_derived as _impl
    except ImportError:
        return
    _impl(pack_id, item_id, path)


def requirement(pack_id: str, item_id: str | None = None) -> str:
    """The ``"pack"`` / ``"pack:item"`` string a node writes in
    ``REQUIRES_PACK`` or in an ``option_packs`` value.

    Pure formatting -- no import, so it answers the same in an install with
    no packs package. It exists so the colon convention is written down in
    one place instead of being retyped into every node's ``option_packs``
    dict, where ``packs.parse_requirement`` would reject the typo only once
    a learner picked that option.
    """
    return f"{pack_id}:{item_id}" if item_id else pack_id
