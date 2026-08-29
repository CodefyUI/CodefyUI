"""Which optional packs are usable right now.

Answered from the outside, cheaply. Nothing here imports a pack's packages:
``find_spec`` says whether ``sentence_transformers`` is importable without
paying the seconds and the gigabyte that importing it costs, and the Package
Center polls this on a timer.

A downloaded model is judged by its SENTINEL, never by a directory listing.
An interrupted 470 MB snapshot leaves a directory that looks exactly like a
finished one, so "the directory is there" is not evidence. The sentinel is a
small JSON file written LAST, once the bytes have landed::

    {"schema": 1, "pack_id": ..., "item_id": ..., "kind": "hf" | "asset",
     "repo_id" | "url": ..., "revision": ...,
     "snapshot_dir" (hf) | "path" (asset): ..., "bytes": ...,
     "sha256" (asset): ..., "at": "<iso timestamp>"}

Later tasks write those; this module reads them, and re-checks each one
against the catalog. A pack that moves to a new revision must not report
yesterday's download as today's model, however complete it is on disk.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .catalog import ModelItem, Pack, get_pack, iter_packs
from .paths import sentinel_path

#: The one pack whose "installed" is not "are its files there". Torch is
#: always importable; this pack asks whether the ACCELERATED build is the one
#: in place, which is a property of the wheel already installed rather than of
#: anything the Package Center downloaded.
_GPU_TORCH_PACK_ID = "gpu-torch"

#: Both ids are interpolated straight into a sentinel FILENAME. Today they
#: only ever come from the catalog; this is the guard for the day one arrives
#: from a request body instead.
#:
#: Matched with ``fullmatch``, and it has to stay that way: ``$`` also matches
#: BEFORE a trailing newline, so ``re.match(r"^...$", "..\n")`` succeeds --
#: which would slip a traversal id past both this pattern and the explicit
#: ``{".", ".."}`` check below, since ``"..\n" != ".."``.
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ItemState:
    """Whether one downloadable item of one pack is on disk."""

    item_id: str
    present: bool
    #: Where the sentinel would live -- set whether or not it exists, so a
    #: caller can report the path it looked at.
    sentinel: Path
    #: What the sentinel points at, and only when the item is ``present``.
    #: For an ``hf`` item this is the snapshot DIRECTORY; for an ``asset``
    #: item it is the downloaded FILE.
    snapshot_dir: Path | None


@dataclass(frozen=True)
class PackState:
    """Whether one pack is usable, and if not, what is missing."""

    pack_id: str
    #: Every module in ``probe_modules`` is importable. Vacuously true for a
    #: pack that is pure data.
    pip_ready: bool
    items: tuple[ItemState, ...]
    #: Packages AND every download are there. This is the UI's "Installed"
    #: pill, and the honest answer to "is there anything left to fetch".
    installed: bool
    #: Packages are there AND there is something to run with: either the pack
    #: downloads nothing, or at least one of its items is present.
    #:
    #: The two differ because a pack's items can be ALTERNATIVES. The four
    #: sentence-embedding models are four choices of embedder, not four parts
    #: of one -- a learner who fetched the 90 MB English one and skipped the
    #: three multilingual ones has a working node, and must not be told to
    #: download 1 GB more before they may run it.
    usable: bool
    #: Dependency packs that are not themselves ready, so installing this one
    #: would not make it usable.
    blocked_by: tuple[str, ...]


def _module_available(name: str) -> bool:
    """Is *name* importable, without importing it?

    ``find_spec`` still imports PARENT packages to ask them, so a
    half-installed dependency can raise anything at all from its
    ``__init__``. Every one of those means the same thing here -- the module
    is not usable -- and none of them may take a status poll down.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def pip_ready(pack: Pack) -> bool:
    """Are all of *pack*'s Python packages importable?

    True for a pack with nothing to probe: ``word-vectors`` ships data and no
    packages, and must not read as permanently unready.
    """
    return all(_module_available(name) for name in pack.probe_modules)


def read_sentinel(path: Path) -> dict | None:
    """The sentinel at *path*, or None if it is missing, unreadable or not a
    JSON object. A corrupt sentinel is treated as no sentinel: the honest
    answer is "we cannot show this was downloaded".

    ``ValueError`` on the READ as well as on the parse: a file of arbitrary
    bytes raises ``UnicodeDecodeError`` -- a ValueError, not an OSError -- and
    letting that escape would take ``probe_all`` down with it, so one corrupt
    byte in one sentinel would break every pack query in the process.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _validate_id(value: str, label: str) -> str:
    if (not isinstance(value, str)
            or not _SAFE_ID.fullmatch(value)
            or value in {".", ".."}):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def write_sentinel(pack_id: str, item_id: str, payload: dict) -> Path:
    """Record that one item finished downloading. Returns the sentinel path.

    Staged through a sibling temp file and ``os.replace``d into place, so a
    crash mid-write leaves either the previous sentinel or none -- never a
    truncated one that a reader would have to guess about. *payload* is
    written verbatim; its shape is the schema in this module's docstring.
    """
    _validate_id(pack_id, "pack_id")
    _validate_id(item_id, "item_id")

    path = sentinel_path(pack_id, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        staged.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return path


def remove_sentinel(pack_id: str, item_id: str) -> bool:
    """Forget one item's download. True if there was a sentinel to remove."""
    _validate_id(pack_id, "pack_id")
    _validate_id(item_id, "item_id")

    try:
        sentinel_path(pack_id, item_id).unlink()
    except FileNotFoundError:
        return False
    return True


def item_state(pack: Pack, item: ModelItem) -> ItemState:
    """Is *item* downloaded? Sentinel first, then the bytes it vouches for.

    Both halves have to agree. A sentinel with no files behind it is the
    residue of a cache someone cleaned out by hand; files with no sentinel are
    an interrupted download that would fail on first use.
    """
    sentinel = sentinel_path(pack.pack_id, item.item_id)
    absent = ItemState(item_id=item.item_id, present=False,
                       sentinel=sentinel, snapshot_dir=None)

    data = read_sentinel(sentinel)
    if data is None:
        return absent

    if item.kind == "hf":
        if data.get("repo_id") != item.repo_id or data.get("revision") != item.revision:
            return absent
        recorded = data.get("snapshot_dir")
    else:
        recorded = data.get("path")

    if not isinstance(recorded, str) or not recorded:
        return absent
    target = Path(recorded)
    if not target.exists():
        return absent

    return ItemState(item_id=item.item_id, present=True,
                     sentinel=sentinel, snapshot_dir=target)


def torch_variant() -> str | None:
    """Which PyTorch build is installed: ``"cu128"``, ``"cpu"``, ``"rocm6.2"``
    -- or None when it cannot be told.

    Read off the local version tag, which is where the wheel records what it
    was built against (``2.11.0+cu128``). An UNTAGGED version is PyPI's
    default wheel, and what that means depends on the platform: the CPU build
    on Windows and Linux, but the MPS build on macOS -- where acceleration
    ships in the default wheel and leaves no tag to read. Guessing "cpu"
    there would tell a Mac user their GPU pack is missing when it is not, so
    the answer is None: unknown.
    """
    try:
        import torch
    except Exception:
        return None

    version = str(getattr(torch, "__version__", ""))
    if "+" in version:
        return version.split("+", 1)[1].strip() or None
    return "cpu" if sys.platform in {"win32", "linux"} else None


def pack_state(pack: Pack) -> PackState:
    """The full state of one pack, read fresh from the interpreter and disk."""
    ready = pip_ready(pack)
    items = tuple(item_state(pack, item) for item in pack.items)

    if pack.pack_id == _GPU_TORCH_PACK_ID:
        # This pack downloads nothing, so the general rules below would call
        # it both installed and usable on any machine at all. Its readiness
        # lives in the installed WHEEL instead, and both answers come from
        # there -- otherwise ``require_pack("gpu-torch")`` would wave through
        # a CPU-only box, which is the one question this pack exists to ask.
        variant = torch_variant()
        installed = usable = variant is not None and variant != "cpu"
    else:
        installed = ready and all(item.present for item in items)
        usable = ready and (not items or any(item.present for item in items))

    blocked_by = tuple(dep for dep in pack.depends_on
                       if not pip_ready(get_pack(dep)))
    return PackState(pack_id=pack.pack_id, pip_ready=ready, items=items,
                     installed=installed, usable=usable, blocked_by=blocked_by)


#: Filled by the first ``probe_all()`` and dropped by ``invalidate()``. The
#: probe is cheap but not free -- a handful of ``find_spec`` calls and a
#: handful of stats -- and the Package Center polls it while a download runs.
_cache: dict[str, PackState] | None = None

#: Bumped by every ``invalidate()``. It exists so a probe that was already
#: running when an install finished cannot store its now-stale answer: the
#: generation it started in no longer matches, so it returns its result and
#: caches nothing.
_generation: int = 0


def probe_all() -> dict[str, PackState]:
    """Every pack's state, keyed by pack id. Cached until ``invalidate()``.

    The returned dict is a copy; ``PackState`` is frozen, so a caller can
    hold it, and edit it, without the next caller inheriting the result.
    """
    global _cache

    cached = _cache
    if cached is not None:
        return dict(cached)

    started_in = _generation
    fresh = {pack.pack_id: pack_state(pack) for pack in iter_packs()}
    if started_in == _generation:
        _cache = fresh
    return dict(fresh)


def invalidate() -> None:
    """Drop the cached probe. Call after any install, download or uninstall.

    Also flushes the import system's cached directory listings: a package
    installed into ``site-packages`` a second ago is on disk, but ``find_spec``
    would keep answering from the listing it made before the install and
    report it missing until the server restarts.
    """
    global _cache, _generation

    # Bump BEFORE clearing, never the other way round. A probe that finishes
    # in the window between these two statements re-checks the generation it
    # started in: bumped first, it sees the change and declines to cache;
    # cleared first, it passes that check and installs an answer computed
    # before the install finished -- which then outlives the very invalidate
    # that was meant to drop it. The window is two adjacent statements wide,
    # so it cannot be entered from a test without contriving a seam here.
    _generation += 1
    _cache = None
    importlib.invalidate_caches()
