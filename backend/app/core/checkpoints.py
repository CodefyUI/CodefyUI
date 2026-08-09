"""Where a training checkpoint goes, and what is in it.

Extracted from ``nodes/io/checkpoint_node.py`` in #122 so that the file
format and the path rules have ONE definition. Two callers now write
checkpoints: ``CheckpointSaver`` (a user wired it into the graph) and the
interrupt path of every long-running training node (nobody wired anything;
the run was stopped and its progress must not evaporate). Those two must
agree byte for byte, because the second is resumed by the same
``CheckpointLoader`` as the first.

``core`` owns this rather than the node package because ``core`` never
imports ``nodes`` -- see ``loop_control.save_interrupt_checkpoint``, which
is core-side and needs exactly these primitives.

Checkpoint payload format
-------------------------
A checkpoint is a plain dict written with ``torch.save`` and read back with
``torch.load(..., weights_only=True)``. Every key is optional on read; the
loader must degrade gracefully when one is missing so that checkpoints
written by an older CodefyUI keep working.

===========================  ============================================
key                          contents
===========================  ============================================
``epoch``                    int, the epoch the checkpoint was taken at
``model_state_dict``         ``model.state_dict()``
``optimizer_state_dict``     ``optimizer.state_dict()``
``initial_lrs``              list[float], one per param group (#149)
``losses``                   optional tensor of per-epoch training losses
``scheduler_state_dict``     optional ``lr_scheduler.state_dict()`` (#118)
``scheduler_class``          optional class name guarding the restore (#118)
``scaler_state_dict``        optional ``GradScaler.state_dict()`` (#135)
===========================  ============================================

``losses`` covers only the epochs the WRITING CALL ran, matching
``TrainingLoopNode``'s own "arrays are per call" rule -- the interrupt and
periodic paths both pass their node's own ``epoch_losses`` as it stands at
the moment they fire, never the whole history back to epoch 0. A chained
resume (checkpoint -> resume -> checkpoint again, the realistic
long-running-job shape) therefore has each LEG's checkpoint carry only
that leg's own losses; ``epoch`` (always absolute) is what stays accurate
across the whole chain, not the ``losses`` tensor. Pre-existing for the
interrupt path; #203 makes it fire routinely rather than only on a stop.

**Honest base_lrs (#149).** ``LRScheduler.__init__`` stamps
``group.setdefault("initial_lr", group["lr"])`` onto each optimizer param
group at CONSTRUCTION time, and ``Optimizer.load_state_dict`` happens to
carry that key through a round trip -- verified against the installed torch
2.11.0+cu128: ``update_group`` replaces each LIVE param group with the
SAVED group's dict wholesale (only ``params``/``param_names`` are patched in
from the live side), so a saved dict that already has ``initial_lr`` keeps
it, accidentally, with no code anywhere naming the guarantee.

**This is a defensive invariant, not a fix for a reachable bug.**
``initial_lrs`` is derived from ``g.get("initial_lr", g["lr"]) for g in
optimizer.param_groups`` -- the SAME ``optimizer.param_groups`` that
``optimizer.state_dict()`` ALSO reads, in the same dict literal, with
nothing mutating them in between. So for any checkpoint this module can
actually produce, ``initial_lrs`` and whatever
``optimizer_state_dict["param_groups"][i].get("initial_lr", lr)`` would
independently reconstruct are mathematically the same value -- proven,
not merely tested: when the live group carries ``initial_lr``, both read
it; when it does not, both fall back to the SAME current ``lr``. There is
no code path, resume chain or ``update_group`` replacement that makes them
disagree, because "the live optimizer at save time" is the one and only
source both draw from. A checkpoint whose optimizer never had a scheduler
attached has no OTHER, richer truth for either field to recover -- nothing
in this process ever knew a different number.

What this key buys instead: CodefyUI's own behaviour no longer *depends*
on ``optimizer.state_dict()``/``load_state_dict()`` choosing to carry
``initial_lr`` through -- it is read from ``optimizer.param_groups``
directly and restored explicitly by ``CheckpointLoaderNode``, before
anything downstream can construct a scheduler. If a future torch release
(or a plugin's custom ``Optimizer`` subclass) ever stops serialising extra
param-group keys, or a scheduler ever stops using ``initial_lr`` as its
name for this, CodefyUI's own contract keeps holding -- what breaks is
visible as a difference in THIS key, not as a silent, no-diff LR
corruption discovered by comparing loss curves. See
``test_initial_lr_is_captured_from_the_live_optimizer_not_its_state_dict``
in ``test_training_resume.py`` for what actually exercises this: it
simulates exactly that kind of divergence (a ``state_dict()`` that drops
the key while the live object keeps it) because no REAL torch behaviour
today produces one.

``scaler_state_dict`` is present only for an fp16 run, and holds five plain
scalars (``scale``, ``growth_factor``, ``backoff_factor``,
``growth_interval``, ``_growth_tracker``) -- no tensors, so it costs nothing
and satisfies the ``weights_only=True`` rule below by construction. Losing
it is survivable (a fresh scaler re-finds its loss scale within a few
hundred steps) but not free: those steps are taken at the wrong scale, and
the ones that overflow are skipped outright.

**The format is append-only.** New keys are added with ``.get()`` on the read
side and never made mandatory, so a newer loader reads an older checkpoint and
an older loader ignores what it does not know. Everything stored must survive
``weights_only=True`` unpickling, i.e. tensors and plain Python containers --
no arbitrary objects.

Note what the interrupt path does NOT add: no new key, no marker inside the
file. An interrupt checkpoint is an ordinary checkpoint that happens to have
been written by the engine, so ``CheckpointLoader`` needs no special case
and a user can resume from it exactly as from a hand-placed one. Where it
came from is recorded on the ``exec_run_artifacts`` row instead, which is
the layer that owns run history. The periodic path (#203, below) makes the
identical choice for the identical reason.

No row, no file (#122, #203)
----------------------------
That last sentence is a constraint, not a description. The artifact ROW is
the only index of an interrupt or periodic checkpoint: nothing scans
``MODELS_DIR/interrupted/`` or ``MODELS_DIR/periodic/`` directly, and the
filename is readable but not enumerable by any consumer. So a run with
nowhere to put the row must not write the file either -- it would be an
orphan that grows a model-sized hole in the user's disk on every timeout,
forever.

The runs that have nowhere: the REST contract runner
(``routes_graph_run.execute_contract_run``, which passes no ``on_signal``
and cancels its context on timeout), the exported Python script, and any
bare ``execute_graph``. ``loop_control.save_interrupt_checkpoint`` and
``save_periodic_checkpoint`` both ask ``context.can_record_artifacts()``
and decline. The alternative -- write the file and log the path -- was
rejected: a path in a log line that the user has to find and clean up by
hand is not a feature, and the interactive and queued lanes (which DO
record rows) are where interruption and long training both actually
happen.

Retention (#156) is the other half of the invariant: the row is only a
reliable index for as long as SOMETHING deletes the file when the row goes.
``RunStore.prune`` does, for ``checkpoint``-kind artifacts, via
:func:`unlink_checkpoint` -- see that function and ``RunStore.prune``'s own
docstring for why the delete has to happen inside the same transaction as
the row read. Retention only ever reaches a run once it is terminal AND has
aged out of the keep-last-N window, so it does not, and structurally cannot,
bound how many periodic checkpoints a single STILL-RUNNING job accumulates
-- see ``TrainingLoopNode``'s ``checkpoint_every`` docs for that trade-off.

The sweep deletes only what it can prove it wrote (#224). The delete guard
is :func:`owned_checkpoint_path`, NOT the write guard: "may a node write
here" is a permissive question and "did we write this" is not, and letting
one answer both made an unattended, irreversible sweep as broad as an
ordinary write. Which is what makes the paragraph above true in the first
place -- ``interrupted/`` and ``periodic/`` are not merely where these files
happen to go, they are the definition of what retention is allowed to
remove.

Durability
----------
``write_checkpoint`` stages to a temporary file in the destination
directory and ``os.replace``s it into place, which is atomic on the same
volume on both POSIX and Windows. A crash or a full disk part-way through a
multi-hundred-megabyte ``torch.save`` therefore leaves the PREVIOUS
checkpoint intact rather than a truncated file that fails to unpickle --
the failure mode that matters, because a checkpoint is only ever read after
something went wrong.

Deliberately NOT fsync'd. Surviving a power cut would need an fsync of the
file and of its directory, which on a large checkpoint costs seconds of
real time on every epoch a graph saves. The guarantee bought here is
"atomic with respect to this process and any reader", which covers the
crash, the exception, the cancellation and the out-of-space; surviving the
machine losing power mid-save is not worth that price for a file that can
be regenerated by re-running.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import settings
from .data_paths import resolve_data_path

logger = logging.getLogger(__name__)

#: Sub-directory of ``MODELS_DIR`` for checkpoints nobody asked for by name.
INTERRUPT_DIRNAME = "interrupted"

#: Sub-directory of ``MODELS_DIR`` for checkpoints a HEALTHY run writes on
#: its own schedule (``TrainingLoop.checkpoint_every``, #203) rather than
#: because anything went wrong. Kept apart from ``INTERRUPT_DIRNAME``: a
#: file under "interrupted" for a run that is training normally would
#: mislead anyone browsing MODELS_DIR by hand, even though both directories
#: follow the identical no-row-no-file discipline and the identical
#: path-safety rules.
PERIODIC_DIRNAME = "periodic"

#: The sub-directories of ``MODELS_DIR`` this module writes into on its own
#: initiative, and therefore owns the lifecycle of. Retention will unlink a
#: file under one of these and nowhere else -- see
#: :func:`owned_checkpoint_path`.
OWNED_DIRNAMES = (INTERRUPT_DIRNAME, PERIODIC_DIRNAME)

#: Cap on each path component built from a run/node id. Long enough to stay
#: recognisable, short enough that a deep MODELS_DIR plus two components
#: cannot push the whole path past Windows' 260-character default limit.
MAX_NAME_PART = 48

#: The filename shape :func:`interrupt_checkpoint_path` and
#: :func:`periodic_checkpoint_path` produce: two id components, then the
#: position (``-e<epoch>`` with an optional ``b<batch>``), then ``.pt``.
#: The numbers accept a leading ``-`` because both builders pass their
#: arguments through ``int()`` rather than rejecting a negative epoch.
#: ``test_generated_checkpoint_names_are_recognised_as_owned`` asserts the
#: builders and this pattern agree, so the two cannot drift apart silently.
_OWNED_NAME = re.compile(r"^.+-e-?\d+(?:b-?\d+)?\.pt$")


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Absolute, validated destination for *path*.

    A relative path is taken as relative to ``MODELS_DIR``. The result must
    stay inside the project data directory and must not name CodefyUI's own
    storage -- this is the only thing standing between a graph parameter and
    an arbitrary file write. The rule itself lives in
    :func:`core.data_paths.resolve_data_path`, shared with ``ModelSaver``
    and ``ImageWriter``; see that module for what "own storage" covers and
    why the data root alone was not enough (#224).

    WRITE scope only. Retention's delete uses :func:`owned_checkpoint_path`,
    which is deliberately much narrower -- see that function.
    """
    return resolve_data_path(path, base=settings.MODELS_DIR)


def owned_checkpoint_path(path: str | Path) -> Path | None:
    """Resolve *path* iff this module wrote it; ``None`` otherwise (#224).

    The DELETE rule, and it is not the write rule. ``resolve_checkpoint_path``
    answers "may a node write here", which is a permissive question by
    design: the whole data directory is a node's to write into. Retention
    asks a different one -- "did WE write this, so that removing the row
    obliges us to remove the file" -- and answering it with the write rule
    made an unattended, irreversible sweep as broad as an ordinary write.

    Only two call sites ever produce a ``checkpoint``-kind artifact row:
    ``loop_control.save_interrupt_checkpoint`` and
    ``save_periodic_checkpoint``, which write to
    :func:`interrupt_checkpoint_path` and :func:`periodic_checkpoint_path`
    respectively -- both under a sub-directory of ``MODELS_DIR`` named in
    :data:`OWNED_DIRNAMES`, both with a generated filename. ``kind`` is an
    open vocabulary (see ``run_store``'s ``ARTIFACT_KIND_*`` comment) and the
    plugin API can log artifacts, so anything else claiming to be a
    ``checkpoint`` is by definition not one of ours and is left alone.

    Note what this does NOT cover, on purpose: a ``CheckpointSaver`` the
    user wired into the graph. That node writes where the user told it to
    and records no artifact row at all, so retention never saw it before
    this change either -- and if something starts logging one, the file
    still belongs to the user, exactly as ``RunStore.delete_run`` already
    reasons about explicitly-named artifact files.

    ``Path.resolve()`` runs first, so a symlink planted inside ``periodic/``
    that points at the database resolves to the database, lands outside the
    owned directories and is refused. Returns the resolved path when it is
    ours, ``None`` when it is not -- callers log the rejection rather than
    raising, because one odd row must not fail a whole prune.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = settings.MODELS_DIR / resolved
    resolved = resolved.resolve()

    if not _OWNED_NAME.match(resolved.name):
        return None
    models_dir = settings.MODELS_DIR.resolve()
    if not any(resolved.is_relative_to(models_dir / d) for d in OWNED_DIRNAMES):
        return None
    return resolved


def _safe_part(value: str) -> str:
    """One filename component from an arbitrary id.

    Node ids carry preset prefixes (``preset1__inner``) and run ids are hex,
    but a plugin or an imported graph can put anything in either. Everything
    outside ``[A-Za-z0-9._-]`` becomes an underscore.
    """
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in value)
    cleaned = cleaned.strip("._") or "unnamed"
    return cleaned[:MAX_NAME_PART]


def interrupt_checkpoint_path(
    run_id: str, node_id: str, *, epoch: int, batch: int,
) -> Path:
    """Where the checkpoint for an interrupted node goes.

    Under ``MODELS_DIR/interrupted/`` rather than beside a user's own
    checkpoints, so a Stop click can never overwrite the file a
    ``CheckpointSaver`` is managing. The name carries the run, the node and
    the position, which makes it unique per interruption AND readable in a
    file listing -- the artifact row is the index, but the filename should
    not need one.
    """
    name = (f"{_safe_part(run_id)}-{_safe_part(node_id)}"
            f"-e{int(epoch)}b{int(batch)}.pt")
    return settings.MODELS_DIR / INTERRUPT_DIRNAME / name


def periodic_checkpoint_path(run_id: str, node_id: str, *, epoch: int) -> Path:
    """Where a periodic (#203) checkpoint for *epoch* goes.

    Named like :func:`interrupt_checkpoint_path` minus the batch component:
    a periodic checkpoint is only ever taken at a clean epoch boundary (see
    ``loop_control.save_periodic_checkpoint``), so there is no mid-epoch
    position to record. One file per checkpoint event rather than one
    overwritten file per run -- retention (``RunStore.prune``,
    :func:`unlink_checkpoint`) is what bounds the count once a run's row
    ages out, not this function.
    """
    name = f"{_safe_part(run_id)}-{_safe_part(node_id)}-e{int(epoch)}.pt"
    return settings.MODELS_DIR / PERIODIC_DIRNAME / name


def build_checkpoint(
    model: Any,
    optimizer: Any,
    *,
    epoch: int = 0,
    losses: Any = None,
    lr_scheduler: Any = None,
    scaler_state: Any = None,
) -> dict[str, Any]:
    """The payload dict, exactly as documented in this module's docstring.

    *scaler_state* accepts either a ``GradScaler`` or the dict its
    ``state_dict()`` returns, because both are things a caller plausibly
    holds: ``TrainingLoop`` has the live scaler, while ``CheckpointSaver``
    is handed the dict over a graph edge. Anything else is ignored with a
    warning rather than raising -- the port is ``DataType.ANY``, so an
    unrelated value wired into it must not cost the user their checkpoint.
    """
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # #149: captured explicitly, mirroring LRScheduler.__init__'s own
        # ``setdefault("initial_lr", lr)`` -- a DEFENSIVE invariant, not a
        # fix for a reachable divergence: this reads the same live
        # optimizer.param_groups that optimizer.state_dict() (above) also
        # reads, so the two can never disagree for any checkpoint this
        # function can actually produce. See the "Honest base_lrs" section
        # above for what it protects against instead. Unconditional (unlike
        # the optional keys below): every optimizer has param_groups, so
        # there is always something honest to record, even when it just
        # equals the current lr because nothing has decayed it yet.
        "initial_lrs": [
            g.get("initial_lr", g["lr"]) for g in optimizer.param_groups
        ],
    }
    if losses is not None:
        checkpoint["losses"] = losses
    if lr_scheduler is not None:
        # #118: without this the LR schedule restarts from scratch on
        # resume. The class name is stored alongside so the loader can
        # refuse to splice a StepLR state into a CosineAnnealingLR.
        #
        # CheckpointSaver's port is DataType.ANY, so anything at all can be
        # wired into it. Say what is wrong before writing the file rather
        # than crashing half way through serialization.
        if not hasattr(lr_scheduler, "state_dict"):
            raise ValueError(
                f"The lr_scheduler input is a {type(lr_scheduler).__name__}, "
                "which has no state_dict(); connect an LRScheduler node's "
                "'scheduler' output (or a CheckpointLoader's) to this port."
            )
        checkpoint["scheduler_state_dict"] = lr_scheduler.state_dict()
        checkpoint["scheduler_class"] = type(lr_scheduler).__name__
    scaler_dict = _scaler_state_dict(scaler_state)
    if scaler_dict:
        checkpoint["scaler_state_dict"] = scaler_dict
    return checkpoint


def _scaler_state_dict(scaler_state: Any) -> dict[str, Any] | None:
    """Normalise a live ``GradScaler`` or its state dict down to the dict."""
    if scaler_state is None:
        return None
    if isinstance(scaler_state, dict):
        return dict(scaler_state) or None
    reader = getattr(scaler_state, "state_dict", None)
    if callable(reader):
        try:
            state = reader()
        except Exception:  # noqa: BLE001 - never lose a checkpoint over this
            logger.warning("could not read the loss scaler's state",
                           exc_info=True)
            return None
        return dict(state) if isinstance(state, dict) and state else None
    logger.warning(
        "The grad_scaler_state input is a %s, which is neither a GradScaler "
        "state dict nor a GradScaler; the checkpoint is written without it. "
        "Wire TrainingLoop's grad_scaler_state output to this port.",
        type(scaler_state).__name__,
    )
    return None


def write_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    *,
    epoch: int = 0,
    losses: Any = None,
    lr_scheduler: Any = None,
    scaler_state: Any = None,
    resolve: bool = True,
) -> Path:
    """Build and save a checkpoint atomically; return where it landed.

    *resolve* runs *path* through :func:`resolve_checkpoint_path` (the
    relative-to-MODELS_DIR + inside-the-data-directory rules). Pass False
    only for a path this module itself produced.

    Staged through a sibling temporary file and ``os.replace``d into place;
    see the module docstring for what that does and does not guarantee. The
    temp name carries a uuid so two threads saving to the same target cannot
    stage over each other, and it is removed on ANY failure -- including
    ``KeyboardInterrupt`` and a cancellation, which is why the handler
    catches ``BaseException`` rather than ``Exception``.
    """
    import torch

    target = resolve_checkpoint_path(path) if resolve else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint(
        model, optimizer, epoch=epoch, losses=losses, lr_scheduler=lr_scheduler,
        scaler_state=scaler_state,
    )
    staged = target.with_name(f"{target.name}.{uuid4().hex[:8]}.tmp")
    try:
        torch.save(checkpoint, str(staged))
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    logger.info(
        "Saved checkpoint to %s (epoch=%d, scheduler=%s)",
        target, epoch, checkpoint.get("scheduler_class", "none"),
    )
    return target


def unlink_checkpoint(path: str | Path) -> bool:
    """Best-effort delete of a checkpoint file whose only row is gone (#156).

    Called from retention (``RunStore.prune``), which runs unattended --
    after every run finishes and at every server startup -- and must never
    fail a whole prune over one file. A locked handle, a path some earlier
    prune already removed (the SELECT-then-DELETE inside one transaction
    that ``RunStore.prune`` uses makes that the only realistic case, but a
    hand-deleted file is just as possible) or a permissions error are all
    logged and swallowed. Never raises.

    Only removes a path :func:`owned_checkpoint_path` recognises as one this
    module itself wrote -- a generated filename under ``MODELS_DIR/interrupted/``
    or ``MODELS_DIR/periodic/``. Until #224 the guard here was
    :func:`resolve_checkpoint_path`, the WRITE rule, which permits anything
    under the data root: an artifact's ``kind`` is an open vocabulary (see
    ``run_store``'s ``ARTIFACT_KIND_*`` comment) and the plugin API can log
    artifacts, so a row mislabelled ``checkpoint`` turned this automatic
    sweep into a delete of an arbitrary file -- on a default install, up to
    and including ``codefyui.db`` itself. Prove-we-wrote-it replaces
    trust-the-row.

    Returns True when a file was actually removed.
    """
    resolved = owned_checkpoint_path(path)
    if resolved is None:
        logger.warning(
            "retention: not deleting artifact path %r -- it is not a file "
            "this server wrote (expected a generated checkpoint name under "
            "MODELS_DIR/%s)", str(path), " or MODELS_DIR/".join(OWNED_DIRNAMES),
        )
        return False
    try:
        resolved.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("retention: could not delete checkpoint file %s",
                       resolved, exc_info=True)
        return False
    return True
