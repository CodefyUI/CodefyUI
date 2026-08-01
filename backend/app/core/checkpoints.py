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
``losses``                   optional tensor of per-epoch training losses
``scheduler_state_dict``     optional ``lr_scheduler.state_dict()`` (#118)
``scheduler_class``          optional class name guarding the restore (#118)
===========================  ============================================

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
the layer that owns run history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

#: Sub-directory of ``MODELS_DIR`` for checkpoints nobody asked for by name.
INTERRUPT_DIRNAME = "interrupted"

#: Cap on each path component built from a run/node id. Long enough to stay
#: recognisable, short enough that a deep MODELS_DIR plus two components
#: cannot push the whole path past Windows' 260-character default limit.
MAX_NAME_PART = 48


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Absolute, validated destination for *path*.

    A relative path is taken as relative to ``MODELS_DIR``. The result must
    stay inside the project data directory -- this is the only thing
    standing between a graph parameter and an arbitrary file write.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = settings.MODELS_DIR / resolved
    resolved = resolved.resolve()

    data_root = settings.MODELS_DIR.parent.resolve()
    if not resolved.is_relative_to(data_root):
        raise ValueError("Output path must be within the project data directory")
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


def build_checkpoint(
    model: Any,
    optimizer: Any,
    *,
    epoch: int = 0,
    losses: Any = None,
    lr_scheduler: Any = None,
) -> dict[str, Any]:
    """The payload dict, exactly as documented in this module's docstring."""
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
    return checkpoint


def write_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    *,
    epoch: int = 0,
    losses: Any = None,
    lr_scheduler: Any = None,
    resolve: bool = True,
) -> Path:
    """Build and save a checkpoint; return where it landed.

    *resolve* runs *path* through :func:`resolve_checkpoint_path` (the
    relative-to-MODELS_DIR + inside-the-data-directory rules). Pass False
    only for a path this module itself produced.
    """
    import torch

    target = resolve_checkpoint_path(path) if resolve else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint(
        model, optimizer, epoch=epoch, losses=losses, lr_scheduler=lr_scheduler,
    )
    torch.save(checkpoint, str(target))
    logger.info(
        "Saved checkpoint to %s (epoch=%d, scheduler=%s)",
        target, epoch, checkpoint.get("scheduler_class", "none"),
    )
    return target
