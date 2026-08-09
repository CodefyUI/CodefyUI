"""What a node may write, and what retention may delete (#224).

Two directions, one root cause. ``resolve_checkpoint_path`` was a WRITE
rule -- "do not let a node write outside the project data directory" -- and
it ended up doing DELETE duty as well, guarding ``RunStore.prune``'s
unattended unlink of every pruned ``checkpoint``-kind artifact.

Both were reachable, and neither needed a hostile plugin:

* **Write.** With the default ``cdui start`` and no ``--project``,
  ``PROJECT_DIR`` is ``None``, so ``config.py``'s project derivation never
  runs and ``MODELS_DIR`` stays ``backend/data/models`` -- one level below
  ``codefyui.db``. ``path="../codefyui.db"`` typed into a ``CheckpointSaver``
  (or ``ModelSaver``, or ``ImageWriter``) resolved to the live database and
  passed the guard, because the database is inside the data directory. The
  issue's own mitigation ("the installed layout is narrower, ``MODELS_DIR``
  is ``assets/models``") describes project mode only.
* **Delete.** ``kind`` is a free-text column and
  ``ExecutionContext.log_artifact`` -- reachable from the plugin API --
  writes both the kind and the path, so a row claiming ``kind="checkpoint"``
  with a path pointing at anything else under the data root had that file
  unlinked by a background sweep, with no user action and no confirmation.

The fix is two rules, not one widened rule, because the questions differ.
The write rule stays permissive (the data directory is a node's to write
into) minus CodefyUI's own storage. The delete rule is
prove-we-wrote-it: a generated filename under a directory this server owns.
Tests below are grouped by direction, then by "the ordinary thing still
works", which is the constraint that rules out simply tightening both.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from app.config import Settings, settings
from app.core import data_paths
from app.core.checkpoints import (
    INTERRUPT_DIRNAME,
    PERIODIC_DIRNAME,
    interrupt_checkpoint_path,
    owned_checkpoint_path,
    periodic_checkpoint_path,
    resolve_checkpoint_path,
    write_checkpoint,
)
from app.core.db import Database
from app.core.run_store import RunProvenance, RunStore
from app.nodes.io.checkpoint_node import CheckpointLoaderNode, CheckpointSaverNode
from app.nodes.io.image_writer_node import ImageWriterNode
from app.nodes.io.model_saver_node import ModelSaverNode

#: Contents given to the stand-in database file. Any assertion that the
#: database survived compares against these bytes rather than merely
#: checking ``exists()`` -- a write that truncated it to zero bytes would
#: still leave the path there.
DB_BYTES = b"SQLite format 3\x00-- the real database, not a checkpoint"

GRAPH = {
    "name": "demo",
    "nodes": [{"id": "a", "type": "Start", "position": {"x": 0, "y": 0},
               "data": {"params": {}}}],
    "edges": [],
}
OPTIONS = {"device": "cpu", "seed": 7, "record_outputs": True,
           "lane": "queued"}


def _model_and_opt():
    model = nn.Linear(4, 2)
    return model, torch.optim.SGD(model.parameters(), lr=0.01)


# ── layout fixtures ───────────────────────────────────────────────────────
#
# Both mirror a real install rather than inventing a shape. conftest.py
# already redirects settings.DB_PATH to a temp directory OUTSIDE the data
# root for isolation, which would make every test here vacuously pass, so
# each fixture places the database where that mode actually puts it.


@pytest.fixture
def default_layout(tmp_path, monkeypatch):
    """The default (no ``--project``) layout: the database is one level
    above MODELS_DIR, inside the data root."""
    data = tmp_path / "data"
    (data / "models").mkdir(parents=True)
    db = data / "codefyui.db"
    db.write_bytes(DB_BYTES)
    monkeypatch.setattr(settings, "MODELS_DIR", data / "models")
    monkeypatch.setattr(settings, "IMAGES_DIR", data / "images")
    monkeypatch.setattr(settings, "DB_PATH", db)
    return SimpleNamespace(data=data, models=data / "models", db=db)


@pytest.fixture
def project_layout(tmp_path, monkeypatch):
    """Project mode, with the roots taken from the REAL ``Settings``
    derivation rather than hand-built, so this keeps testing the shipped
    layout if that derivation changes. DB_PATH is install-global (never
    derived from PROJECT_DIR -- see ``_derive_project_roots``) and so sits
    outside the data root entirely."""
    proj = tmp_path / "proj"
    proj.mkdir()
    derived = Settings(PROJECT_DIR=proj)
    assert derived.MODELS_DIR == proj / "assets" / "models"

    install = tmp_path / "install"
    install.mkdir()
    db = install / "codefyui.db"
    db.write_bytes(DB_BYTES)

    derived.MODELS_DIR.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", derived.MODELS_DIR)
    monkeypatch.setattr(settings, "IMAGES_DIR", derived.IMAGES_DIR)
    monkeypatch.setattr(settings, "GRAPHS_DIR", derived.GRAPHS_DIR)
    monkeypatch.setattr(settings, "DB_PATH", db)
    return SimpleNamespace(proj=proj, models=derived.MODELS_DIR, db=db)


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "runstore.db")
    database.connect()
    try:
        yield RunStore(database)
    finally:
        database.close()


async def _make_run(store: RunStore):
    return await store.create_run(graph_snapshot=GRAPH, options=OPTIONS,
                                  provenance=RunProvenance())


# ── 1. the write direction ────────────────────────────────────────────────


def test_the_database_is_inside_the_data_root_on_a_default_install(
    default_layout,
):
    """The premise, pinned. Everything below is only interesting because
    the containment rule -- the whole of the old guard -- says yes here."""
    resolved = (settings.MODELS_DIR / "../codefyui.db").resolve()
    assert resolved == default_layout.db.resolve()
    assert resolved.is_relative_to(data_paths.data_root()), (
        "if the database ever stops being inside the data root on a default "
        "install, this whole file is testing a shape that no longer exists"
    )


def test_a_checkpoint_path_cannot_name_the_run_database(default_layout):
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        resolve_checkpoint_path("../codefyui.db")


def test_the_checkpoint_saver_node_cannot_overwrite_the_database(
    default_layout,
):
    """End to end through the node, which is how a user reaches this: a
    path typed into a graph parameter, no plugin and no mislabelled row."""
    model, opt = _model_and_opt()
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt},
            {"path": "../codefyui.db", "epoch": 1},
        )
    assert default_layout.db.read_bytes() == DB_BYTES


def test_the_model_saver_node_cannot_overwrite_the_database(default_layout):
    model, _ = _model_and_opt()
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        ModelSaverNode().execute({"model": model},
                                 {"path": "../codefyui.db"})
    assert default_layout.db.read_bytes() == DB_BYTES


def test_the_image_writer_node_cannot_overwrite_the_database(default_layout):
    """ImageWriter resolves a relative path one level deeper than the
    others (``<data>/output``), so the traversal that reaches the database
    is shorter -- the guard has to be about the RESOLVED path, not the
    number of ``..`` segments."""
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        ImageWriterNode().execute({"image": torch.zeros(3, 4, 4)},
                                  {"path": "../codefyui.db"})
    assert default_layout.db.read_bytes() == DB_BYTES


def test_an_absolute_path_to_the_database_is_refused_too(default_layout):
    """The traversal is incidental; naming the file outright is the same
    mistake and gets the same answer."""
    model, opt = _model_and_opt()
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt},
            {"path": str(default_layout.db)},
        )
    assert default_layout.db.read_bytes() == DB_BYTES


@pytest.mark.parametrize("sidecar", ["-wal", "-shm", "-journal"])
def test_the_databases_sqlite_sidecars_are_refused(default_layout, sidecar):
    """``core.db`` runs in WAL mode. Truncating ``codefyui.db-wal`` loses
    every committed transaction still sitting in it, so protecting only the
    main file would leave the corruption reachable by a one-character
    change to the path."""
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        resolve_checkpoint_path(f"../codefyui.db{sidecar}")


@pytest.mark.skipif(os.name != "nt",
                    reason="only Windows treats these as the same file")
@pytest.mark.parametrize("spelling", ["../CODEFYUI.DB", "../CODEFYUI.DB-WAL"])
def test_a_case_differing_spelling_of_the_database_is_refused(
    default_layout, spelling,
):
    """``Path.resolve()`` canonicalises the case of components that EXIST on
    disk but falls back to lexical normalisation for ones that do not, so an
    exact string comparison is not sound on a case-insensitive filesystem.

    Both spellings are here because only the second one actually needs the
    case folding: ``codefyui.db`` exists in the fixture, so ``resolve()``
    hands back the on-disk casing by itself. The ``-wal`` sidecar does not
    exist -- it is created when the database is opened and removed on a
    clean close -- so nothing canonicalises it, and a case-sensitive
    comparison lets it straight through to a write that corrupts the
    database the moment SQLite reopens it.

    On POSIX these ARE different files and refusing them would be wrong,
    hence the skip rather than a platform branch inside the rule."""
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        resolve_checkpoint_path(spelling)


def test_a_path_outside_the_data_root_still_reports_the_old_reason(
    default_layout, tmp_path,
):
    """Two ways to fail, two messages: leaving the data directory is a
    different mistake from naming CodefyUI's own storage, and the user
    fixes them differently."""
    with pytest.raises(ValueError, match="within the project data directory"):
        resolve_checkpoint_path(tmp_path / "elsewhere" / "x.pt")


def test_the_checkpoint_loader_surfaces_which_rule_it_failed(default_layout):
    """The node used to flatten every ValueError to the containment
    message, which would now be a lie for the protected case."""
    model, opt = _model_and_opt()
    with pytest.raises(ValueError, match="CodefyUI's own storage"):
        CheckpointLoaderNode().execute(
            {"model": model, "optimizer": opt},
            {"path": "../codefyui.db", "device": "cpu"},
        )


# ── 2. the delete direction ───────────────────────────────────────────────


async def test_prune_does_not_delete_a_mislabelled_row_pointing_at_the_database(
    store, default_layout,
):
    """The issue's own scenario, with the worst available target. The row
    is inside the data root, so the write-scoped guard said yes; the
    ownership guard says no."""
    run = await _make_run(store)
    await store.add_artifact(run.id, "checkpoint", str(default_layout.db))
    await store.mark_finished(run.id, "succeeded")

    assert await store.prune(keep_last=0) == 1
    assert default_layout.db.read_bytes() == DB_BYTES, (
        "an unattended prune deleted the run database because a row "
        "claimed the file was a checkpoint"
    )


async def test_prune_does_not_delete_a_checkpoint_named_file_it_did_not_write(
    store, default_layout,
):
    """Not every ``.pt`` under MODELS_DIR is ours -- a user's own saved
    model sits right there, and a ``CheckpointSaver`` they wired up writes
    exactly here by default."""
    users_own = default_layout.models / "my_best_model.pt"
    users_own.write_bytes(b"hours of training")

    run = await _make_run(store)
    await store.add_artifact(run.id, "checkpoint", str(users_own))
    await store.mark_finished(run.id, "succeeded")

    assert await store.prune(keep_last=0) == 1
    assert users_own.exists(), "retention deleted a file it never wrote"


async def test_prune_does_not_delete_a_foreign_name_inside_an_owned_directory(
    store, default_layout,
):
    """Owning the directory is not enough on its own: the filename has to
    be one the generators produce. Something else dropping a file into
    ``periodic/`` does not make that file ours."""
    owned_dir = default_layout.models / PERIODIC_DIRNAME
    owned_dir.mkdir(parents=True)
    intruder = owned_dir / "notes.pt"
    intruder.write_bytes(b"not a generated name")

    run = await _make_run(store)
    await store.add_artifact(run.id, "checkpoint", str(intruder))
    await store.mark_finished(run.id, "succeeded")

    assert await store.prune(keep_last=0) == 1
    assert intruder.exists()


async def test_prune_does_not_follow_a_symlink_out_of_an_owned_directory(
    store, default_layout,
):
    """A correctly-named file in the right directory that is really a link
    to the database. ``resolve()`` runs before the containment check, so
    the link resolves to the database, lands outside the owned directory
    and is refused -- and the unlink would have removed the TARGET, not the
    link, had it gone ahead."""
    owned_dir = default_layout.models / PERIODIC_DIRNAME
    owned_dir.mkdir(parents=True)
    link = owned_dir / "r1-n1-e1.pt"
    try:
        link.symlink_to(default_layout.db)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable (Windows needs the privilege)")

    run = await _make_run(store)
    await store.add_artifact(run.id, "checkpoint", str(link))
    await store.mark_finished(run.id, "succeeded")

    assert await store.prune(keep_last=0) == 1
    assert default_layout.db.read_bytes() == DB_BYTES


async def test_prune_still_deletes_a_periodic_checkpoint_this_server_wrote(
    store, default_layout,
):
    """The whole point of the sweep, exercised through the real generator
    rather than a hand-written path: retention that stopped removing its
    own files would leak a model-sized orphan per checkpoint event."""
    target = periodic_checkpoint_path("run-abc", "loop1", epoch=4)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"a periodic checkpoint")

    run = await _make_run(store)
    await store.add_artifact(run.id, "checkpoint", str(target))
    await store.mark_finished(run.id, "succeeded")

    assert await store.prune(keep_last=0) == 1
    assert not target.exists(), "retention leaked its own checkpoint file"


def test_generated_checkpoint_names_are_recognised_as_owned(default_layout):
    """The anti-drift assertion. ``owned_checkpoint_path`` matches a
    filename pattern that the two generators have to keep producing; if
    either one's naming changes without the pattern following, retention
    silently stops cleaning up and nothing else notices."""
    generated = [
        interrupt_checkpoint_path("run-abc", "loop1", epoch=0, batch=0),
        interrupt_checkpoint_path("a" * 80, "b" * 80, epoch=12, batch=340),
        interrupt_checkpoint_path("!!!", "@@@", epoch=-1, batch=-1),
        periodic_checkpoint_path("run-abc", "loop1", epoch=4),
        periodic_checkpoint_path("preset1__inner", "n/o:d*e", epoch=999),
        periodic_checkpoint_path("!!!", "@@@", epoch=-1),
    ]
    for path in generated:
        assert owned_checkpoint_path(path) == path.resolve(), (
            f"{path.name!r} is written by this module but retention would "
            "refuse to clean it up"
        )
    assert {p.parent.name for p in generated} == {INTERRUPT_DIRNAME,
                                                 PERIODIC_DIRNAME}


# ── 3. the ordinary paths, which must not have moved ──────────────────────


def test_an_ordinary_checkpoint_write_still_works(default_layout):
    model, opt = _model_and_opt()
    res = CheckpointSaverNode().execute(
        {"model": model, "optimizer": opt},
        {"path": "training.pt", "epoch": 3},
    )
    assert (default_layout.models / "training.pt").exists()
    assert res["model"] is model

    loaded = CheckpointLoaderNode().execute(
        {"model": nn.Linear(4, 2), "optimizer": torch.optim.SGD(
            nn.Linear(4, 2).parameters(), lr=0.01)},
        {"path": "training.pt", "device": "cpu"},
    )
    assert loaded["epoch"] == 3


def test_an_ordinary_model_save_still_works(default_layout):
    model, _ = _model_and_opt()
    res = ModelSaverNode().execute({"model": model},
                                   {"path": "weights.pt"})
    assert (default_layout.models / "weights.pt").exists()
    assert res["path"].endswith("weights.pt")


def test_an_ordinary_image_write_still_works(default_layout):
    res = ImageWriterNode().execute({"image": torch.zeros(3, 8, 8)},
                                    {"path": "picture.png", "format": "PNG"})
    written = default_layout.data / "output" / "picture.png"
    assert written.exists()
    assert res["path"] == str(written)


def test_a_write_elsewhere_under_the_data_root_still_works(default_layout):
    """The write rule is still permissive by design -- a sibling directory
    of MODELS_DIR that nothing in particular owns is an ordinary place for
    a graph to put a file, and narrowing to an allow-list of known
    sub-directories would have broken that."""
    model, _ = _model_and_opt()
    ModelSaverNode().execute({"model": model},
                             {"path": "../scratch/run7/weights.pt"})
    assert (default_layout.data / "scratch" / "run7" / "weights.pt").exists()


def test_ordinary_writes_still_work_in_project_mode(project_layout):
    """The layout the issue's mitigation argument described. The database
    is install-global here, so it is outside the data root and the
    protected set never fires -- what this pins is that adding it changed
    nothing for the mode that was already safe."""
    model, opt = _model_and_opt()
    CheckpointSaverNode().execute(
        {"model": model, "optimizer": opt},
        {"path": "training.pt", "epoch": 1},
    )
    ModelSaverNode().execute({"model": model}, {"path": "weights.pt"})
    ImageWriterNode().execute({"image": torch.zeros(3, 4, 4)},
                              {"path": "shot.png"})

    assert (project_layout.models / "training.pt").exists()
    assert (project_layout.models / "weights.pt").exists()
    assert (project_layout.proj / "assets" / "output" / "shot.png").exists()
    assert project_layout.db.read_bytes() == DB_BYTES


def test_the_engines_own_checkpoint_writers_still_work(default_layout):
    """``write_checkpoint(resolve=False)`` is how the interrupt and
    periodic paths write, and their targets have to survive both rules:
    accepted on the way in, recognised as ours on the way out."""
    model, opt = _model_and_opt()
    for target in (interrupt_checkpoint_path("r", "n", epoch=1, batch=2),
                   periodic_checkpoint_path("r", "n", epoch=1)):
        written = write_checkpoint(target, model, opt, epoch=1, resolve=False)
        assert written.exists()
        assert resolve_checkpoint_path(written) == written.resolve()
        assert owned_checkpoint_path(written) == written.resolve()
