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
from .paths import asset_dir, hf_cache_dir, sentinel_path

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
            raise PackNeedsRestart(
                f"{pack.title} cannot be installed while the server is "
                f"running: it would have to replace a package already in use",
                command=f"cdui packs install {pack.pack_id} --restart",
                hint=detail)
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


def _convert_glove_step(gz_path: Path, *, emit) -> None:
    """Turn the downloaded GloVe gzip into the ``.npz`` the node loads.

    ``ensure_npz(gz_path, progress=...)`` converts *gz_path* into
    ``glove-50d.npz`` beside it, once, and returns that path. Unpacking 400k
    word vectors is slow enough to need its own bar, so its ``progress``
    callback is forwarded as ordinary ``progress`` events for ``glove-50d``
    -- the UI already knows how to draw those, and to a learner this is the
    same wait as the download that came before it.

    The converter arrives in a later PR. Until then the download is still
    worth having -- so a missing converter is a log line and a finished step,
    not a failed install that throws away 66 MB somebody just waited for.
    """
    item_id = _GLOVE[1]
    emit({"type": "step_started", "step": f"convert:{item_id}",
          "label": "Preparing GloVe vectors"})
    try:
        from ...nodes.llm._glove import ensure_npz
    except ImportError:
        emit({"type": "log", "line": (
            "GloVe conversion is not available in this build; the downloaded "
            "table is kept and will be converted on first use")})
    else:
        def _forward(payload: dict) -> None:
            # Stamped LAST, so whatever the converter reports, what leaves
            # here is a progress frame for THIS item and nothing else.
            emit({**payload, "type": "progress", "item": item_id})

        npz_path = ensure_npz(gz_path, progress=_forward)
        emit({"type": "log", "line": f"GloVe vectors ready at {npz_path}"})
    emit({"type": "step_done", "step": f"convert:{item_id}"})


def verify_imports(pack: Pack, *, emit) -> None:
    """Check that *pack*'s packages import, in a child interpreter.

    In THIS process the answer would be wrong: ``site-packages`` changed a
    second ago, and an import system that has already cached its directory
    listings would report a perfectly good install as missing. A child pays
    a fresh interpreter's startup and gives the true answer.

    A pack with nothing to probe (``word-vectors`` ships data and no
    packages) has no verify step at all rather than an empty one.
    """
    if not pack.probe_modules:
        return

    emit({"type": "step_started", "step": "verify",
          "label": f"Verifying {', '.join(pack.probe_modules)}"})
    argv = [sys.executable, "-c", "import " + ", ".join(pack.probe_modules)]
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=PROBE_TIMEOUT_S, check=False)
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
    """
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


def _inside_the_cache(target: Path) -> bool:
    """Is *target* somewhere this application is allowed to delete?

    A sentinel is a file on disk. One naming ``C:/Windows`` -- corrupted,
    hand-edited, or restored from a backup taken on another machine -- would
    otherwise turn "remove this model" into "delete that directory".

    A path that cannot even be resolved answers False: "we could not prove
    this is ours" and "delete it" must never be the same answer.
    """
    try:
        resolved = target.resolve()
        roots = [hf_cache_dir().resolve(), asset_dir().resolve()]
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def remove_item(pack: Pack, item_id: str) -> bool:
    """Delete one downloaded item and forget it. True if anything went.

    Deletes what the SENTINEL names, not what the catalog would name today:
    a pack that has since moved to a new revision still has to be able to
    free the bytes of the old one.
    """
    item = get_item(pack, item_id)
    sentinel = state.read_sentinel(sentinel_path(pack.pack_id, item_id))
    removed = False

    recorded = None
    if sentinel is not None:
        recorded = (sentinel.get("snapshot_dir") if item.kind == "hf"
                    else sentinel.get("path"))

    if isinstance(recorded, str) and recorded:
        target = Path(recorded)
        if _inside_the_cache(target):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed = not target.exists()
            elif target.exists():
                target.unlink()
                removed = True

    removed = state.remove_sentinel(pack.pack_id, item_id) or removed
    state.invalidate()
    return removed
