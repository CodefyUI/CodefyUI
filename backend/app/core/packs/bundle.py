"""Offline pack bundles: export on a connected machine, install on one without.

A classroom's lab machines often have no internet. A bundle carries one
pack from a machine where it is installed to one where it is not::

    <dir>/manifest.json                 written LAST, so a partial export has none
    <dir>/items/<item_id>/...           an hf item's snapshot files
    <dir>/items/<item_id>/<filename>    an asset item's one file
    <dir>/wheels/*.whl                  the pack's packages, for one platform

The CATALOG stays the allowlist. A manifest names a pack id and item ids,
and every one of them has to be in ``catalog.CATALOG`` with the same repo,
revision, URL and filename, or the bundle is refused before a byte is
copied. Where files land is derived from the catalog item (never from a path
in the manifest), and the pip specs installed are the catalog's own, so a
bundle can supply bytes for what the catalog names and nothing else.

The sha256 of every file is recorded at export and checked at install. For
an asset item with a catalog digest that is a real check against the
catalog; for a Hugging Face snapshot and for the wheels it detects a copy
gone wrong, and the bundle is otherwise trusted the way the machine that
made it trusted its downloads.

Wheels are platform-specific: ``pip download`` fetches them for the
interpreter running the export, and the manifest records that platform so an
install on another one is refused with a reason instead of failing in uv.
Only the packages the pack adds on top of the base install are bundled; the
base install (torch included) is expected to be on the target already.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from packaging.requirements import InvalidRequirement, Requirement

from . import runner, state
from .catalog import ModelItem, Pack, get_item
from .constraints import _canonical, _is_editable, _machine_chosen
from .errors import (
    BundleInvalid,
    PackCancelled,
    PackInstallError,
    PackInsufficientDisk,
)

MANIFEST_NAME = "manifest.json"
BUNDLE_FORMAT = "codefyui-pack-bundle"
BUNDLE_SCHEMA = 1
ITEMS_DIR = "items"
WHEELS_DIR = "wheels"

#: A manifest is a few kilobytes. Anything this large is not one.
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
#: One path component of a file inside a snapshot, or a snapshot commit.
_SAFE_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._ +=@-]*")
_WHEEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.whl")

_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class BundleFile:
    """One file of a bundled item, relative to that item's directory."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BundleItem:
    """One item as the manifest describes it, already checked against the
    catalog."""

    item: ModelItem
    #: The snapshot directory's name (the commit) for an hf item; None for
    #: an asset.
    commit: str | None
    files: tuple[BundleFile, ...]

    @property
    def size(self) -> int:
        return sum(entry.size for entry in self.files)


@dataclass(frozen=True)
class Bundle:
    """A bundle directory whose manifest matched the catalog."""

    root: Path
    pack: Pack
    items: dict[str, BundleItem]
    wheels: tuple[BundleFile, ...]
    platform: dict
    catalog_version: str

    def item_dir(self, item: ModelItem) -> Path:
        return self.root / ITEMS_DIR / item.item_id

    @property
    def wheels_dir(self) -> Path:
        return self.root / WHEELS_DIR


@dataclass(frozen=True)
class ExportOutcome:
    """What one export wrote."""

    pack_id: str
    dest: Path
    items: tuple[str, ...]
    wheels: tuple[str, ...]
    #: Packages the pack needs that could not be bundled (a local build such
    #: as ``torch+cu128``, or one installed from a URL).
    skipped: tuple[str, ...]


def catalog_version() -> str:
    """The CodefyUI release whose catalog wrote or reads a bundle."""
    from ..version import get_version

    return get_version()


def current_platform() -> dict:
    """What the wheels in a bundle made here are built for."""
    return {
        "sys_platform": sys.platform,
        "machine": platform.machine().lower(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_hashed(source: Path, target: Path, *,
                on_bytes: Callable[[int], None] | None = None,
                cancel_check: Callable[[], bool] | None = None) -> str:
    """Copy *source* to *target* and return the sha256 of what was copied.

    Symlinks are followed: a Hugging Face snapshot is links into ``blobs/``,
    and the bundle has to hold the bytes.
    """
    digest = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, open(target, "wb") as dst:
        for chunk in iter(lambda: src.read(_CHUNK), b""):
            if cancel_check is not None and cancel_check():
                raise PackCancelled(f"copy of {source.name} cancelled")
            dst.write(chunk)
            digest.update(chunk)
            if on_bytes is not None:
                on_bytes(len(chunk))
    return digest.hexdigest()


class _Meter:
    """Byte progress for one item, as the ``progress`` events downloads send.

    One event per copied megabyte at most, which is the chunk size.
    """

    def __init__(self, emit: Callable[[dict], None], item_id: str,
                 total: int) -> None:
        self._emit = emit
        self._item_id = item_id
        self.total = total
        self.done = 0

    def add(self, n: int) -> None:
        self.done += n
        percent = (round(min(100.0, 100.0 * self.done / self.total), 1)
                   if self.total else None)
        self._emit({"type": "progress", "item": self._item_id,
                    "bytes_done": self.done,
                    "bytes_total": self.total or None, "percent": percent})


def _relative_file(value: object) -> str:
    """A manifest file path, if it is a plain relative POSIX path."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise BundleInvalid(f"unsafe file path in manifest: {value!r}")
    parts = PurePosixPath(value).parts
    if (PurePosixPath(value).is_absolute() or not parts
            or any(not _SAFE_PART.fullmatch(part) for part in parts)):
        raise BundleInvalid(f"unsafe file path in manifest: {value!r}")
    return value


def inside(root: Path, relative: str) -> Path:
    """``root / relative``, refused if it resolves outside *root*.

    The second line of defence after :func:`_relative_file`: a symlink in
    the bundle directory could still point elsewhere.
    """
    candidate = root / relative
    try:
        resolved = candidate.resolve()
        base = root.resolve()
    except OSError as exc:
        raise BundleInvalid(f"cannot resolve {candidate}: {exc}") from exc
    if not resolved.is_relative_to(base):
        raise BundleInvalid(f"{relative} points outside the bundle")
    return candidate


def _file_entry(value: object) -> BundleFile:
    if not isinstance(value, dict):
        raise BundleInvalid("a manifest file entry is not an object")
    size = value.get("size")
    digest = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise BundleInvalid(f"bad size in manifest: {size!r}")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise BundleInvalid(f"bad sha256 in manifest: {digest!r}")
    return BundleFile(path=_relative_file(value.get("path")), size=size,
                      sha256=digest)


def _bundle_item(pack: Pack, entry: object) -> BundleItem:
    """One manifest item, checked field by field against the catalog."""
    if not isinstance(entry, dict):
        raise BundleInvalid("a manifest item is not an object")
    item_id = entry.get("item_id")
    try:
        item = get_item(pack, item_id) if isinstance(item_id, str) else None
    except KeyError:
        item = None
    if item is None:
        raise BundleInvalid(
            f"the bundle names item {item_id!r}, which pack "
            f"{pack.pack_id!r} does not have in this catalog")
    if entry.get("kind") != item.kind:
        raise BundleInvalid(
            f"item {item.item_id!r} is kind {item.kind!r} in the catalog and "
            f"{entry.get('kind')!r} in the bundle")

    files_raw = entry.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise BundleInvalid(f"item {item.item_id!r} lists no files")
    files = tuple(_file_entry(value) for value in files_raw)
    if len({entry_.path for entry_ in files}) != len(files):
        raise BundleInvalid(f"item {item.item_id!r} lists a file twice")

    if item.kind == "hf":
        if (entry.get("repo_id") != item.repo_id
                or entry.get("revision") != item.revision):
            raise BundleInvalid(
                f"item {item.item_id!r} in the bundle is "
                f"{entry.get('repo_id')}@{entry.get('revision')}; this "
                f"catalog wants {item.repo_id}@{item.revision}. Export the "
                f"bundle again with this CodefyUI release")
        commit = entry.get("commit")
        if not isinstance(commit, str) or not _SAFE_PART.fullmatch(commit):
            raise BundleInvalid(
                f"item {item.item_id!r} has no usable snapshot commit")
        return BundleItem(item=item, commit=commit, files=files)

    if entry.get("url") != item.url or entry.get("filename") != item.filename:
        raise BundleInvalid(
            f"item {item.item_id!r} in the bundle is a different file from "
            f"the one this catalog names")
    if len(files) != 1 or files[0].path != item.filename:
        raise BundleInvalid(
            f"asset item {item.item_id!r} must be exactly {item.filename}")
    if item.sha256 is not None and files[0].sha256 != item.sha256:
        raise BundleInvalid(
            f"item {item.item_id!r}: the bundle's sha256 is not the one the "
            f"catalog records")
    return BundleItem(item=item, commit=None, files=files)


def _read_manifest(root: Path) -> dict:
    path = root / MANIFEST_NAME
    if not root.is_dir():
        raise BundleInvalid(f"{root} is not a directory")
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise BundleInvalid(f"{path} is too large to be a manifest")
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleInvalid(
            f"{root} has no {MANIFEST_NAME}; it is not a pack bundle, or its "
            f"export did not finish") from exc
    except (OSError, ValueError) as exc:
        raise BundleInvalid(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleInvalid(f"{path} is not a JSON object")
    if data.get("format") != BUNDLE_FORMAT:
        raise BundleInvalid(f"{path} is not a CodefyUI pack bundle manifest")
    if data.get("schema") != BUNDLE_SCHEMA:
        raise BundleInvalid(
            f"{path} has schema {data.get('schema')!r}; this release reads "
            f"schema {BUNDLE_SCHEMA}")
    return data


def load_bundle(root: Path | str, pack: Pack) -> Bundle:
    """Read the bundle at *root* and check it against *pack*'s catalog entry.

    Cheap: it reads the manifest and nothing else, so the service can call it
    before it accepts a job. The bytes are checked while they are copied.

    :raises BundleInvalid: for every reason the bundle cannot be installed.
    """
    root = Path(root)
    data = _read_manifest(root)

    if data.get("pack_id") != pack.pack_id:
        raise BundleInvalid(
            f"the bundle is for pack {data.get('pack_id')!r}, not "
            f"{pack.pack_id!r}")
    if pack.install_mode == "restart":
        raise BundleInvalid(
            f"pack {pack.pack_id!r} installs in restart mode and cannot be "
            f"installed from a bundle")
    if data.get("pip") != list(pack.pip):
        raise BundleInvalid(
            f"the bundle was made for the packages {data.get('pip')!r}; this "
            f"catalog installs {list(pack.pip)!r}. Export the bundle again "
            f"with this CodefyUI release")

    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise BundleInvalid("the manifest has no item list")
    items: dict[str, BundleItem] = {}
    for entry in raw_items:
        bundled = _bundle_item(pack, entry)
        if bundled.item.item_id in items:
            raise BundleInvalid(
                f"item {bundled.item.item_id!r} is in the manifest twice")
        items[bundled.item.item_id] = bundled

    raw_wheels = data.get("wheels", [])
    if not isinstance(raw_wheels, list):
        raise BundleInvalid("the manifest's wheel list is not a list")
    wheels = []
    for value in raw_wheels:
        entry = _file_entry(value)
        if not _WHEEL_NAME.fullmatch(entry.path):
            raise BundleInvalid(f"{entry.path!r} is not a wheel file name")
        wheels.append(entry)

    target = data.get("platform")
    return Bundle(root=root, pack=pack, items=items, wheels=tuple(wheels),
                  platform=target if isinstance(target, dict) else {},
                  catalog_version=str(data.get("catalog_version") or "unknown"))


def select_items(bundle: Bundle, item_ids: Sequence[str] | None
                 ) -> list[ModelItem]:
    """Which items an install from *bundle* is for.

    ``None`` means every item the bundle carries that is not already on
    disk, in catalog order. A named item has to be in the pack
    (``ValueError``) and in the bundle (:class:`BundleInvalid`).
    """
    pack = bundle.pack
    if item_ids is None:
        return [item for item in pack.items
                if item.item_id in bundle.items
                and not state.item_state(pack, item).present]
    items = []
    for item_id in item_ids:
        try:
            item = get_item(pack, item_id)
        except KeyError as exc:
            raise ValueError(
                f"pack {pack.pack_id!r} has no item {item_id!r}") from exc
        if item_id not in bundle.items:
            raise BundleInvalid(f"the bundle does not contain {item_id!r}")
        items.append(item)
    return items


def check_wheels(bundle: Bundle) -> Path:
    """The wheel directory, once its platform and every digest check out.

    :raises BundleInvalid: no wheels, or wheels for another platform.
    :raises PackInstallError: a wheel whose bytes are not the manifest's.
    """
    if not bundle.wheels:
        raise BundleInvalid(
            f"the bundle has no wheels, and {bundle.pack.title}'s packages "
            f"are not installed here. Export it again without --no-wheels")
    here = current_platform()
    if bundle.platform != here:
        raise BundleInvalid(
            f"the bundle's wheels are for {_platform_text(bundle.platform)}; "
            f"this machine is {_platform_text(here)}. Export the bundle on a "
            f"machine like this one")
    for wheel in bundle.wheels:
        path = inside(bundle.wheels_dir, wheel.path)
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise PackInstallError(
                f"wheel {wheel.path} is missing from the bundle",
                hint=str(exc)) from exc
        if actual != wheel.sha256:
            raise PackInstallError(
                f"wheel {wheel.path} does not match the manifest; the bundle "
                f"is damaged", hint=f"expected {wheel.sha256}, got {actual}")
    return bundle.wheels_dir


def _platform_text(value: dict) -> str:
    return (f"{value.get('sys_platform', '?')}/{value.get('machine', '?')} "
            f"Python {value.get('python', '?')}")


# ── export ───────────────────────────────────────────────────────────────


def _installed_index() -> tuple[dict[str, object], set[str]]:
    """``({canonical: dist}, editable names)`` for this interpreter."""
    index: dict[str, object] = {}
    editable: set[str] = set()
    for dist in importlib.metadata.distributions():
        try:
            metadata = dist.metadata
            name = _canonical(str((metadata["Name"] if metadata else None) or ""))
            is_editable = _is_editable(dist)
        except Exception:
            continue
        if not name:
            continue
        if is_editable:
            editable.add(name)
        index.setdefault(name, dist)
    return index, editable


def _wanted(requirement: Requirement, extras: Iterable[str]) -> bool:
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({"extra": extra})
               for extra in (tuple(extras) or ("",)))


def _closure(roots: Iterable[Requirement], index: dict[str, object]
             ) -> set[str]:
    """Every installed distribution *roots* pull in, by canonical name."""
    seen: dict[str, set[str]] = {}
    stack = [(root, frozenset()) for root in roots]
    while stack:
        requirement, parent_extras = stack.pop()
        if not _wanted(requirement, parent_extras):
            continue
        name = _canonical(requirement.name)
        dist = index.get(name)
        if dist is None:
            continue
        extras = set(requirement.extras)
        known = seen.get(name)
        if known is not None and extras <= known:
            continue
        seen[name] = (known or set()) | extras
        for raw in getattr(dist, "requires", None) or ():
            try:
                stack.append((Requirement(raw), frozenset(seen[name])))
            except InvalidRequirement:
                continue
    return set(seen)


def wheel_requirements(pack: Pack) -> tuple[list[str], list[str]]:
    """``(pins, skipped)``: what to ``pip download`` for *pack*.

    The packages *pack*'s specs pull in, minus what the editable project
    (the base install) already pulls in, each pinned to the version
    installed here. A package picked for this machine's hardware (the
    PyTorch family, a local build tag) or installed from a URL exists on no
    index under that version, so it is listed in ``skipped`` instead.
    """
    index, editable = _installed_index()
    roots = []
    for spec in pack.pip:
        try:
            roots.append(Requirement(spec))
        except InvalidRequirement:
            continue
    base_roots = []
    for name in editable:
        for raw in getattr(index.get(name), "requires", None) or ():
            try:
                base_roots.append(Requirement(raw))
            except InvalidRequirement:
                continue

    needed = _closure(roots, index) - _closure(base_roots, index) - editable
    pins: list[str] = []
    skipped: list[str] = []
    for name in sorted(needed):
        dist = index[name]
        version = str(getattr(dist, "version", "") or "")
        try:
            from_url = bool(dist.read_text("direct_url.json"))
        except Exception:
            from_url = False
        if not version or from_url or _machine_chosen(name, version):
            skipped.append(f"{name}=={version}" if version else name)
        else:
            pins.append(f"{name}=={version}")
    return pins, skipped


def pip_download_argv(pins: Sequence[str], dest: Path) -> list[str]:
    """The command that fetches *pins* as wheels for this interpreter.

    This interpreter's own pip when it has one; otherwise pip run by
    ``uv tool run`` on this interpreter, so the wheels match its Python
    version and platform either way. ``--no-deps``: *pins* is already the
    whole set (see :func:`wheel_requirements`).
    """
    tail = ["download", "--no-deps", "--only-binary=:all:",
            "--dest", str(dest), *pins]
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", *tail]
    return [runner.find_uv() or "uv", "tool", "run", "--python",
            sys.executable, "pip", *tail]


def _export_sources(pack: Pack, item: ModelItem
                    ) -> tuple[list[tuple[Path, str]], str | None]:
    """``([(source file, path in the bundle)], commit)`` of one installed
    item. The asset's bundle path is the catalog filename."""
    probe = state.item_state(pack, item)
    if not probe.present or probe.snapshot_dir is None:
        raise ValueError(
            f"{item.item_id} is not installed here; run "
            f"`cdui packs install {pack.pack_id} --items {item.item_id}` "
            f"first")
    if item.kind == "asset":
        return [(probe.snapshot_dir, item.filename)], None
    root = probe.snapshot_dir
    files = sorted(
        (path, path.relative_to(root).as_posix())
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("."))
    if not files:
        raise ValueError(f"{item.item_id}'s snapshot at {root} is empty")
    return files, root.name


def export_pack(
    pack: Pack,
    item_ids: Sequence[str] | None,
    dest: Path | str,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
    download_wheels: bool = True,
) -> ExportOutcome:
    """Write a bundle of *pack* into *dest*.

    *item_ids* ``None`` means every item installed here. A failed or
    cancelled export removes what it wrote. The manifest is written last, so
    an export killed outright leaves a directory :func:`load_bundle` refuses.

    :raises ValueError: a restart-mode pack, an item that is not installed,
        packages that are not installed, or a *dest* that is not empty.
    :raises PackInsufficientDisk: *dest* cannot hold the items.
    :raises PackInstallError: ``pip download`` failed.
    """
    dest = Path(dest)
    if pack.install_mode == "restart":
        raise ValueError(
            f"pack {pack.pack_id!r} installs in restart mode; it has nothing "
            f"to bundle. Use `cdui install --gpu <variant>` on each machine")
    if pack.pip and not state.pip_ready(pack):
        raise ValueError(
            f"{pack.title}'s packages are not installed here; run "
            f"`cdui packs install {pack.pack_id}` first")

    if item_ids is None:
        items = [item for item in pack.items
                 if state.item_state(pack, item).present]
    else:
        items = []
        for item_id in item_ids:
            try:
                items.append(get_item(pack, item_id))
            except KeyError as exc:
                raise ValueError(
                    f"pack {pack.pack_id!r} has no item {item_id!r}") from exc
    if not items and not pack.pip:
        raise ValueError(
            f"nothing of {pack.title} is installed here to export")
    sources = {item.item_id: _export_sources(pack, item) for item in items}

    created = not dest.exists()
    if not created and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"{dest} is not an empty directory")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        return _write_bundle(pack, items, sources, dest, emit=emit,
                             cancel_check=cancel_check,
                             download_wheels=download_wheels)
    except BaseException:
        # Everything in *dest* was written by this call (it was new or
        # empty), so a failed export is removed and a retry can use the
        # same directory.
        for child in dest.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        if created:
            try:
                dest.rmdir()
            except OSError:
                pass
        raise


def _write_bundle(pack: Pack, items: list[ModelItem], sources: dict,
                  dest: Path, *, emit, cancel_check,
                  download_wheels: bool) -> ExportOutcome:
    """The copying half of :func:`export_pack`, into an empty *dest*."""
    needed = sum(source.stat().st_size
                 for files, _ in sources.values() for source, _ in files)
    free = shutil.disk_usage(dest).free
    if free < needed:
        raise PackInsufficientDisk(
            f"not enough free space at {dest}: {needed // 1_000_000} MB "
            f"needed, {free // 1_000_000} MB free", needed=needed, free=free)

    manifest_items = []
    for item in items:
        files, commit = sources[item.item_id]
        step = f"export:{item.item_id}"
        emit({"type": "step_started", "step": step,
              "label": f"Copying {item.item_id} into the bundle"})
        meter = _Meter(emit, item.item_id,
                       sum(source.stat().st_size for source, _ in files))

        entries = []
        for source, rel in files:
            digest = copy_hashed(source, dest / ITEMS_DIR / item.item_id / rel,
                                 on_bytes=meter.add, cancel_check=cancel_check)
            entries.append({"path": rel, "size": source.stat().st_size,
                            "sha256": digest})
        entry: dict = {"item_id": item.item_id, "kind": item.kind,
                       "files": entries}
        if item.kind == "hf":
            entry.update(repo_id=item.repo_id, revision=item.revision,
                         commit=commit)
        else:
            if item.sha256 is not None and entries[0]["sha256"] != item.sha256:
                raise PackInstallError(
                    f"{item.item_id} on this machine does not match the "
                    f"catalog's sha256; reinstall it before exporting")
            entry.update(url=item.url, filename=item.filename)
        manifest_items.append(entry)
        emit({"type": "step_done", "step": step})

    wheels: list[dict] = []
    skipped: list[str] = []
    if pack.pip and download_wheels:
        wheels, skipped = _download_wheels(pack, dest / WHEELS_DIR,
                                           emit=emit, cancel_check=cancel_check)

    manifest = {
        "format": BUNDLE_FORMAT,
        "schema": BUNDLE_SCHEMA,
        "pack_id": pack.pack_id,
        "catalog_version": catalog_version(),
        "pip": list(pack.pip),
        "platform": current_platform(),
        "items": manifest_items,
        "wheels": wheels,
        "skipped_packages": skipped,
        "created_at": _now_iso(),
    }
    staged = dest / f".{MANIFEST_NAME}.tmp-{os.getpid()}"
    try:
        staged.write_text(json.dumps(manifest, indent=2), encoding="utf-8",
                          newline="\n")
        os.replace(staged, dest / MANIFEST_NAME)
    finally:
        staged.unlink(missing_ok=True)

    return ExportOutcome(pack_id=pack.pack_id, dest=dest,
                         items=tuple(item.item_id for item in items),
                         wheels=tuple(entry["path"] for entry in wheels),
                         skipped=tuple(skipped))


def _download_wheels(pack: Pack, wheel_dir: Path, *, emit, cancel_check
                     ) -> tuple[list[dict], list[str]]:
    pins, skipped = wheel_requirements(pack)
    emit({"type": "step_started", "step": "wheels",
          "label": f"Downloading wheels for {', '.join(pack.pip)}"})
    for name in skipped:
        emit({"type": "log", "line": (
            f"{name} is not bundled: it is a build chosen for this machine or "
            f"installed from a URL, and the target needs it installed already")})
    wheel_dir.mkdir(parents=True, exist_ok=True)
    if pins:
        tail: list[str] = []
        returncode = runner.run_argv(
            pip_download_argv(pins, wheel_dir), emit=emit,
            cancel_check=cancel_check, cwd=wheel_dir, tail=tail)
        if returncode != 0:
            raise PackInstallError(
                f"downloading the wheels for {pack.title} failed (exit "
                f"{returncode})", hint="\n".join(tail).strip())
    wheels = [{"path": path.name, "size": path.stat().st_size,
               "sha256": sha256_file(path)}
              for path in sorted(wheel_dir.glob("*.whl"))]
    emit({"type": "step_done", "step": "wheels"})
    return wheels, skipped
