"""One install, start to finish. The server job and the CLI both call this.

Deliberately SYNCHRONOUS and free of any transport. The async job in the API
layer and ``cdui packs`` differ only in where the events go -- a WebSocket in
one case, a terminal in the other -- so the order of an install, what counts
as a failure, and what a failure is called are written down exactly once,
here, and both get the same answers.

The order is the design:

1. resolve which items were asked for. An unknown id is a caller mistake and
   is raised BEFORE anything is touched.
2. the disk precheck. Finding out at 90% of a 470 MB download that the disk
   was always too small wastes the download and leaves a half-written cache.
3. pip, under the constraints file, and only when the packages are not
   already importable. Packages before models: a failed pip run should cost
   seconds rather than a download that is then unusable.
4. the models, one step each, cancellable between and during.
5. the import probe, in a CHILD interpreter -- a package installed a second
   ago is not importable in THIS one, and asking here would report a
   perfectly good install as broken.

And ``state.invalidate()`` at the end whatever happened. A failed or
cancelled install still changed what is on disk; a probe cache that outlives
it makes the Package Center lie in both directions.

One failure is not like the others. uv cannot replace a package this process
has already imported -- that is the whole point of the constraints file -- so
a resolver conflict is not "the install failed", it is "not while the server
is running", and :class:`PackNeedsRestart` carries the command to type
instead.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import download, runner, state
from .catalog import ModelItem, Pack, get_item
from .constraints import write_constraints_file
from .errors import PackCancelled, PackInstallError, PackNeedsRestart
from .paths import asset_dir, hf_cache_dir, sentinel_dir, sentinel_path

log = logging.getLogger(__name__)

#: Where uv runs. The project is installed editable from here, so a relative
#: path in a spec or a constraint resolves the way it does for every other
#: install of this repository.
BACKEND_DIR = Path(__file__).resolve().parents[3]

#: How long the import probe may take. ``import torch`` on a cold Windows
#: filesystem is genuinely tens of seconds; two minutes is "something is
#: wrong" rather than "this machine is slow".
PROBE_TIMEOUT_S = 120

#: The pack and item whose download needs a conversion pass afterwards.
_GLOVE = ("word-vectors", "glove-50d")

#: The module that pass lives in. It ships with the backend, so its absence
#: now means a trimmed or partial build -- which is survivable -- while its
#: presence-but-broken is not. See ``_converter_absent``.
_GLOVE_MODULE = "app.nodes.llm._glove"

#: What huggingface_hub names a repo's directory. Checked before deleting
#: one, so a folder name that came from anywhere else cannot be removed.
_HF_FOLDER_PREFIX = "models--"


@dataclass(frozen=True)
class InstallOutcome:
    """What one install actually did, for the caller to report."""

    pack_id: str
    pip_installed: bool
    items_done: tuple[str, ...]


def _resolve_items(pack: Pack, item_ids: Sequence[str] | None
                   ) -> list[ModelItem]:
    """Which items this install is for.

    ``None`` means "the whole pack", minus what is already downloaded: a
    learner adding a second embedding model must not re-fetch the first.
    An id the pack does not have raises ``ValueError`` -- a caller mistake,
    and one worth catching before the disk check rather than three minutes
    into a download.
    """
    if item_ids is None:
        return [item for item in pack.items
                if not state.item_state(pack, item).present]

    items: list[ModelItem] = []
    for item_id in item_ids:
        try:
            items.append(get_item(pack, item_id))
        except KeyError as exc:
            raise ValueError(
                f"pack {pack.pack_id!r} has no item {item_id!r}") from exc
    return items


def _tail_text(lines: Sequence[str]) -> str:
    return "\n".join(lines).strip()


#: An argument every shell passes through untouched: letters, digits, and
#: the punctuation that is never syntax in bash, cmd or PowerShell. The
#: characters deliberately NOT in it are the ones that bite -- whitespace,
#: and the ``<`` / ``>`` of a PEP 508 version spec.
_BARE_ARGUMENT = re.compile(r"[A-Za-z0-9._/\\:+-]+")


def _shell_quote(part: str) -> str:
    """One argument of a command line a human is going to paste.

    ``shlex.quote`` is POSIX in its choice of quote character, and one of
    the parts here is usually a Windows path, where ``'C:\\...'`` is not a
    quoted string at all. Double quotes are what bash, cmd and PowerShell
    all read the same way, for a path and for a version spec alike.

    Quoting only on whitespace was not enough, and the only pack that can
    reach this path proves it: ``sentence-transformers>=3.0,<6`` has no
    space in it, and a bare ``>`` is REDIRECTION in all three shells --
    bash writes a file named ``=3.0,``, PowerShell refuses to parse, cmd
    fails on ``6``. So the test is inverted: a part is left bare only when
    every character in it is one no shell reads as syntax.
    """
    return part if _BARE_ARGUMENT.fullmatch(part) else f'"{part}"'


def _restart_command(pack: Pack) -> str:
    """The command that finishes this install with the server stopped.

    Spelled out in full rather than pointing at a ``cdui packs`` flag: no
    flag installs a pack into a stopped server, so naming one would send the
    user to an exit code instead of to a working install. The constraints
    file is deliberately absent -- it pins what THIS process has already
    imported, which is the very thing that has to be replaced.
    """
    argv = ["uv", "pip", "install", "--python", sys.executable, *pack.pip]
    return " ".join(_shell_quote(part) for part in argv)


def _run_pip_step(pack: Pack, *, emit, cancel_check) -> None:
    """Install *pack*'s packages, or explain why it could not be done here."""
    emit({"type": "step_started", "step": "pip",
          "label": f"Installing packages: {', '.join(pack.pip)}"})

    tail: list[str] = []
    with tempfile.TemporaryDirectory(prefix="codefyui-packs-") as workdir:
        # The constraints file describes THIS interpreter at THIS moment, so
        # it is written per job and thrown away with the job -- caching it
        # would pin an install to a machine state that has since changed.
        constraints_path = write_constraints_file(Path(workdir))
        returncode = runner.run_pip(
            pack.pip,
            constraints_path=constraints_path,
            emit=emit,
            cancel_check=cancel_check,
            cwd=BACKEND_DIR,
            tail=tail,
        )

    if returncode != 0:
        detail = _tail_text(tail)
        if runner.looks_like_resolver_conflict(tail):
            command = _restart_command(pack)
            raise PackNeedsRestart(
                f"{pack.title} cannot be installed while the server is "
                f"running: it would have to replace a package already in use",
                command=command,
                hint=f"stop the server, then run:\n{command}\n\n{detail}")
        raise PackInstallError(
            f"installing {pack.title} failed (uv exited {returncode})",
            hint=detail)

    emit({"type": "step_done", "step": "pip"})


def _download_step(pack: Pack, item: ModelItem, *, emit, cancel_check) -> Path:
    """Fetch one item; returns where its bytes landed.

    The path is the step's real output, not a detail: the GloVe convert step
    that follows needs the FILE that was just downloaded, and asking the
    sentinel for it again would be a second answer to a question this call
    already answered.
    """
    step = f"download:{item.item_id}"
    emit({"type": "step_started", "step": step,
          "label": f"Downloading {item.item_id}"})
    if item.kind == "hf":
        landed = download.download_hf_item(pack, item, emit=emit,
                                           cancel_check=cancel_check)
    else:
        landed = download.download_asset_item(pack, item, emit=emit,
                                              cancel_check=cancel_check)
    emit({"type": "step_done", "step": step})
    return landed


def _converter_absent(exc: ImportError) -> bool:
    """Did the import fail because the converter is not in this build?

    "The converter is not in this build" and "the converter is broken" are
    both ``ImportError``. The first is survivable and may be shrugged off; the
    second is a GloVe pack that can never convert, and shrugging THAT off
    would report the install as finished.

    The exception cannot tell the two apart, which is why it is not asked:
    a module that IS here but does not export ``ensure_npz`` raises an
    ``ImportError`` whose ``name`` is the converter's own module -- exactly
    what a missing module raises -- so trusting ``exc.name`` calls a broken
    converter absent and finishes the install. PRESENCE is asked instead.
    Already imported means present; importable means present; anything else
    is a build that does not ship it yet. *exc* is kept in the signature so
    the question reads as one about this import at the call site.
    """
    if _GLOVE_MODULE in sys.modules:
        return False
    try:
        return importlib.util.find_spec(_GLOVE_MODULE) is None
    except (ImportError, ValueError, AttributeError):
        # A parent package that raises, or a module with no spec. Either way
        # nothing importable is there.
        return True


def _convert_glove_step(gz_path: Path, *, emit) -> None:
    """Turn the downloaded GloVe gzip into the ``.npz`` the node loads.

    ``ensure_npz(gz_path, progress=...)`` converts *gz_path* into
    ``glove-50d.npz`` beside it, once, and returns that path. Unpacking 400k
    word vectors is slow enough to need its own bar, so its ``progress``
    callback is forwarded as ordinary ``progress`` events for ``glove-50d``
    -- the UI already knows how to draw those, and to a learner this is the
    same wait as the download that came before it.

    The converter ships with the backend, so a build without it is a trimmed
    or partial one rather than a version that predates it. The download is
    still worth having there -- so an absent converter is a log line and a
    finished step, not a failed install that throws away 69 MB somebody just
    waited for.
    """
    item_id = _GLOVE[1]
    emit({"type": "step_started", "step": f"convert:{item_id}",
          "label": "Preparing GloVe vectors"})
    try:
        from ...nodes.llm._glove import ensure_npz
    except ImportError as exc:
        if not _converter_absent(exc):
            raise
        emit({"type": "log", "line": (
            "GloVe conversion is not available in this build; the downloaded "
            "table is kept and will be converted on first use")})
    else:
        def _forward(payload: dict) -> None:
            # Every progress frame carries the same three keys whoever sent
            # it, so a consumer never has to ask which producer this one came
            # from. ``bytes_done`` defaults to 0 rather than None because the
            # event contract types it as an int -- "no bytes yet" is zero,
            # where an unknown total and an uncomputable percent really are
            # nothing. Type and item are stamped LAST: whatever the converter
            # reports, what leaves here is a frame for THIS item.
            frame = dict(payload)
            frame.setdefault("bytes_done", 0)
            for key in ("bytes_total", "percent"):
                frame.setdefault(key, None)
            frame["type"] = "progress"
            frame["item"] = item_id
            emit(frame)

        npz_path = ensure_npz(gz_path, progress=_forward)
        record_derived(_GLOVE[0], item_id, npz_path)
        emit({"type": "log", "line": f"GloVe vectors ready at {npz_path}"})
    emit({"type": "step_done", "step": f"convert:{item_id}"})


def record_derived(pack_id: str, item_id: str, *paths: Path) -> None:
    """Note on the item's sentinel that the download now owns *paths* too.

    The npz the conversion writes is 83 MB that the CATALOG does not describe
    -- it has one filename per item -- so ``remove_item`` would delete the
    69 MB download, report the bytes as freed, and leave the larger file
    behind forever. The sentinel is the file that already says this item was
    downloaded, so it is where "and this came with it" belongs.

    Extended, not replaced: everything the download recorded stays, and
    ``derived`` is overwritten rather than appended to, so converting twice
    does not list the same file twice. No sentinel means no download record
    to extend -- which cannot happen after a real download, since
    ``download_asset_item`` writes one before this step runs -- and inventing
    a partial one would be a worse answer than leaving it alone.

    Public, and re-exported as ``packs.record_derived``, because the install
    is no longer the only converter. ``WordVector`` reaches
    ``_glove.load_glove_50d`` on a table the install never converted -- an
    older install, or one whose convert step was skipped -- and a node that
    writes 83 MB without recording it leaves exactly the orphan this
    function exists to prevent. It takes *pack_id* for the same reason:
    a second caller outside this module must not have to know that the one
    pack with a derived file today is the one hardcoded here.
    """
    sentinel = state.read_sentinel(sentinel_path(pack_id, item_id))
    if sentinel is None:
        log.warning("no sentinel for pack %s item %s; %s will not be removed "
                    "with it", pack_id, item_id,
                    ", ".join(str(path) for path in paths))
        return
    sentinel["derived"] = [str(path) for path in paths]
    state.write_sentinel(pack_id, item_id, sentinel)


def verify_imports(pack: Pack, *, emit) -> None:
    """Check that *pack*'s packages import, in a child interpreter.

    In THIS process the answer would be wrong: ``site-packages`` changed a
    second ago, and an import system that has already cached its directory
    listings would report a perfectly good install as missing. A child pays
    a fresh interpreter's startup and gives the true answer.

    A pack with nothing to probe (``word-vectors`` ships data and no
    packages) has no verify step at all rather than an empty one.

    Started the same way the install itself is: no inherited ``PYTHONPATH``
    (a dev shell's would put this repo inside the probe and it would pass
    for the wrong reason), no console window on Windows, and no stdin to
    block on if something in the import chain decides to ask a question.
    """
    if not pack.probe_modules:
        return

    emit({"type": "step_started", "step": "verify",
          "label": f"Verifying {', '.join(pack.probe_modules)}"})
    argv = [sys.executable, "-c", "import " + ", ".join(pack.probe_modules)]
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=PROBE_TIMEOUT_S, check=False,
                                stdin=subprocess.DEVNULL,
                                env=runner.pip_env(),
                                creationflags=runner.creation_flags())
    except subprocess.TimeoutExpired as exc:
        raise PackInstallError(
            f"{pack.title} installed but importing it timed out",
            hint=" ".join(argv)) from exc

    if result.returncode != 0:
        raise PackInstallError(
            f"{pack.title} installed but its packages cannot be imported",
            hint=_tail_text((result.stderr or result.stdout or "").splitlines()))
    emit({"type": "step_done", "step": "verify"})


def install_pack_live(
    pack: Pack,
    item_ids: Sequence[str] | None,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
) -> InstallOutcome:
    """Install *pack* (and the given items), reporting every step to *emit*.

    Synchronous and blocking: the caller owns the thread. Raises
    :class:`PackCancelled` when *cancel_check* goes true, and one of the
    :mod:`.errors` types when something fails -- see the module docstring for
    which failure means what.

    A restart-mode pack is a ``ValueError``, not an install. That kind of pack
    replaces something the calling process has already imported, so there is
    no version of "live" that works -- and the failure mode without this guard
    is the quiet one: ``gpu-torch`` has no pip specs, no probe modules and no
    items, so every step below is a no-op and the install reports SUCCESS
    having changed nothing at all. A caller that reaches here with one has a
    bug, not a runtime problem, which is why it is a ValueError rather than a
    :class:`PackInstallError`.
    """
    if pack.install_mode == "restart":
        raise ValueError(
            f"pack {pack.pack_id!r} installs in restart mode; "
            f"install_pack_live cannot install it")

    items = _resolve_items(pack, item_ids)
    pip_installed = False
    done: list[str] = []

    try:
        download.check_disk(items)

        if pack.pip and not state.pip_ready(pack):
            _run_pip_step(pack, emit=emit, cancel_check=cancel_check)
            pip_installed = True

        landed: dict[str, Path] = {}
        for item in items:
            if cancel_check():
                raise PackCancelled(f"install of {pack.pack_id} cancelled")
            landed[item.item_id] = _download_step(
                pack, item, emit=emit, cancel_check=cancel_check)
            done.append(item.item_id)

        glove_pack_id, glove_item_id = _GLOVE
        if pack.pack_id == glove_pack_id and glove_item_id in done:
            _convert_glove_step(landed[glove_item_id], emit=emit)

        verify_imports(pack, emit=emit)
    finally:
        # Whatever happened -- finished, failed, cancelled -- the disk is not
        # what it was, and the next status poll has to see that.
        state.invalidate()

    return InstallOutcome(pack_id=pack.pack_id, pip_installed=pip_installed,
                          items_done=tuple(done))


def _resolved(path: Path) -> Path | None:
    """*path* resolved, or None when it cannot be. A path we cannot even
    name is not one we may delete."""
    try:
        return path.resolve()
    except OSError:
        return None


def _hf_repo_folder_name(repo_id: str) -> str:
    """huggingface_hub's own name for *repo_id*'s directory.

    Asked of the library rather than derived, because the library is what
    created the directory. The documented fallback (``models--org--name``) is
    only for an install too old to expose the helper.
    """
    try:
        from huggingface_hub.file_download import repo_folder_name
    except ImportError:
        pass
    else:
        try:
            return repo_folder_name(repo_id=repo_id, repo_type="model")
        except Exception:
            pass
    return _HF_FOLDER_PREFIX + repo_id.replace("/", "--")


def _hf_removal_target(item: ModelItem) -> Path | None:
    """The directory holding every byte of *item*, or None if it is not there.

    The whole REPO folder, not the snapshot. huggingface_hub stores the bytes
    once in ``blobs/`` and links (or, on a filesystem without links -- which
    on Windows is most of them -- copies) them into ``snapshots/<revision>/``.
    Deleting a snapshot therefore frees little or nothing while telling the
    user their 470 MB is back.

    Derived from the CATALOG, never from the sentinel: a sentinel is a file
    on disk, and a corrupt or hand-edited one must not be able to steer a
    recursive delete. The guards below are the second half of that -- the
    target has to be a direct child of the pack cache and to be named the way
    huggingface_hub names one.
    """
    if not item.repo_id:
        return None
    root = _resolved(hf_cache_dir())
    folder = _resolved(hf_cache_dir() / _hf_repo_folder_name(item.repo_id))
    if root is None or folder is None:
        return None
    if folder.parent != root or not folder.name.startswith(_HF_FOLDER_PREFIX):
        return None
    return folder if folder.is_dir() else None


def _asset_removal_target(item: ModelItem, recorded: str | None) -> Path | None:
    """The one file *item* is allowed to delete, or None.

    Exactly ``asset_dir() / item.filename`` and nothing else. The asset
    directory IS the cache root, so "somewhere under the root" would be
    satisfied by the root itself, by the directory the sentinels live in, and
    by every other pack's downloads -- one "remove this model" could take the
    whole cache with it.

    The filename has to be ONE plain component. ``""`` joins to the cache
    root, ``".."`` to its parent, and ``"../x"`` to a file outside the cache
    altogether -- three spellings of "delete something that is not this
    model", and the last one leaves the cache entirely. ``".."`` is named
    explicitly because pathlib does not resolve it away: ``Path("..").name``
    is ``".."``, so it would pass a same-name check.

    The four checks below OVERLAP -- today each one refuses every path the
    others do, so no test can tell any single one of them apart. That is on
    purpose and they are not to be tidied into one: they state four separate
    facts (what a filename may look like, that the result is not the root,
    that it sits directly in the asset directory, that it is not a sentinel),
    and the day ``paths.py`` moves a directory or a symlink appears in the
    cache, whichever one still holds is the one that stops an ``rmtree`` over
    somebody's whole cache.
    """
    if (not item.filename
            or item.filename in {".", ".."}
            or Path(item.filename).name != item.filename):
        return None
    root = _resolved(asset_dir())
    canonical = _resolved(asset_dir() / item.filename)
    sentinels = _resolved(sentinel_dir())
    if root is None or canonical is None or canonical == root:
        return None
    if canonical.parent != root:
        return None
    if sentinels is not None and canonical.is_relative_to(sentinels):
        return None
    if recorded is not None and _resolved(Path(recorded)) != canonical:
        return None
    return canonical if canonical.is_file() else None


def _derived_removal_targets(sentinel: dict | None) -> list[Path]:
    """The extra files this item's sentinel says came with the download.

    Written by ``record_derived`` -- today just the converted GloVe npz.
    Checked exactly as strictly as the asset itself, and for exactly the same
    reason: this list is a string in a FILE on disk, so a corrupt or
    hand-edited sentinel must not be able to point a delete at the cache
    root, at the directory the sentinels live in, or at anything that is not
    a direct child of the asset directory.

    Anything that fails a check is skipped rather than raised on. A removal
    that refused to run because one line of one sentinel was odd would leave
    the user with no way to free the space at all.
    """
    if sentinel is None:
        return []
    recorded = sentinel.get("derived")
    if not isinstance(recorded, list):
        return []

    root = _resolved(asset_dir())
    sentinels = _resolved(sentinel_dir())
    targets: list[Path] = []
    for value in recorded:
        if not isinstance(value, str) or not value:
            continue
        candidate = _resolved(Path(value))
        if root is None or candidate is None or candidate == root:
            continue
        if candidate.parent != root:
            continue
        if sentinels is not None and candidate.is_relative_to(sentinels):
            continue
        if candidate.is_file():
            targets.append(candidate)
    return targets


def remove_item(pack: Pack, item_id: str) -> bool:
    """Delete one downloaded item and forget it. True if the BYTES went.

    For a Hugging Face item that is the repo folder, which is where the bytes
    are; for an asset it is the one file in the asset directory, plus
    whatever the install DERIVED from it -- the converted GloVe table is
    bigger than the download it came from, and the catalog does not name it.
    Every target comes from the catalog or from the item's own sentinel and
    is checked before anything is deleted -- see the three helpers above for
    what each refuses and why.

    The sentinel is removed either way, so an item whose files somebody
    cleaned out by hand stops reporting itself as downloaded. The RETURN
    value is only about the item's OWN bytes: a derived file that could not
    be deleted is logged, not reported here, because "the download is gone"
    is the question the caller asked. On Windows a file another process holds
    open cannot be deleted and ``rmtree(ignore_errors=True)`` says nothing
    about it, so "we removed the record" must not be reported as "we freed
    the space" either.
    """
    item = get_item(pack, item_id)
    sentinel = state.read_sentinel(sentinel_path(pack.pack_id, item_id))
    recorded = None
    if sentinel is not None and item.kind == "asset":
        value = sentinel.get("path")
        recorded = value if isinstance(value, str) and value else None

    if item.kind == "hf":
        target = _hf_removal_target(item)
    else:
        target = _asset_removal_target(item, recorded)
        # Before the download itself, because the sentinel that names these
        # goes away with it: a derived file left behind after its record has
        # been deleted is one nothing will ever find again.
        for extra in _derived_removal_targets(sentinel):
            try:
                extra.unlink()
            except OSError:
                log.warning("could not remove %s, which pack %s item %s "
                            "derived; something is holding it open",
                            extra, pack.pack_id, item_id)

    removed = False
    if target is not None:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.unlink()
            except OSError:
                pass
        removed = not target.exists()
        if not removed:
            log.warning(
                "could not remove %s for pack %s item %s; something is "
                "holding it open", target, pack.pack_id, item_id)

    state.remove_sentinel(pack.pack_id, item_id)
    state.invalidate()
    return removed
