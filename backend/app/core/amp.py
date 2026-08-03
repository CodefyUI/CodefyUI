"""Mixed precision for the training nodes (core#135).

One 32 GB card holds roughly twice the activations in 16-bit as in 32-bit,
and on every accelerator newer than Volta the 16-bit matmul path is also the
fast one. This module is the whole of CodefyUI's AMP policy: which of the
three precisions a device can honour, what to do when it cannot, and the
four call sites a training loop needs (``autocast``, ``scale``, ``unscale_``
before clipping, ``step``/``update``).

The three precisions
--------------------
``fp32``
    Autocast off. The default, and what every graph saved before this
    existed keeps doing -- bit for bit, because :meth:`AmpPolicy.autocast`
    returns a ``nullcontext`` and every other method is the identity.

``bf16``
    ``autocast(dtype=torch.bfloat16)`` and **no gradient scaler**. bfloat16
    keeps float32's exponent range and spends the bits on mantissa instead,
    so gradients do not underflow and there is nothing for a scaler to do.
    This is the precision to reach for on Ampere and newer (RTX 30xx, 40xx,
    50xx, A100, H100) and the one this issue is written around.

``fp16``
    ``autocast(dtype=torch.float16)`` **plus** a ``GradScaler``. float16 has
    five exponent bits, so small gradients flush to zero; the scaler
    multiplies the loss by a large factor before ``backward()`` and divides
    it back out of the gradients before the step, backing the factor off
    whenever it overflows. It exists for cards without bfloat16 (Volta,
    Turing -- GTX 16xx, RTX 20xx) and for the rare model that measures
    faster in fp16.

What a device cannot do, it does not silently do
------------------------------------------------
:meth:`AmpPolicy.for_device` resolves the REQUEST against the device and
says so in the log when the two disagree. Nothing is guessed at runtime:
``policy.precision`` is what is actually in effect, ``policy.requested`` is
what was asked for, and the training node reports both in its config frame
so the difference is visible in the UI rather than only in a log file.

* CUDA without bfloat16 support (pre-Ampere) asks for ``bf16`` -> ``fp32``.
* MPS asks for anything but ``fp32`` -> ``fp32``. Apple's autocast coverage
  is narrower than CUDA's and varies by torch build; guessing wrong on
  hardware this project cannot test in CI is worse than declining.
* CPU honours both ``bf16`` and ``fp16``. Neither is fast there -- CPU
  autocast is about numerical parity, not throughput -- but a lesson about
  mixed precision has to be runnable on the machine in front of the
  student, and the tests for the scaler path run there too.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, ContextManager

logger = logging.getLogger(__name__)

#: The vocabulary of the ``precision`` param, in the order the select shows
#: it. ``fp32`` is first because it is the default and the one nobody has to
#: think about.
PRECISIONS = ("fp32", "bf16", "fp16")

#: Precisions that need a gradient scaler. Exactly one, and it is not a
#: coincidence: a scaler exists to work around float16's exponent range.
SCALED_PRECISIONS = frozenset({"fp16"})


def normalize_precision(value: Any) -> str:
    """Coerce a param value to one of :data:`PRECISIONS`.

    An unknown spelling degrades to ``fp32`` with a warning rather than
    raising: a saved graph from a newer CodefyUI (or a typo in an exported
    script's argument) must not be able to kill a training run over a
    setting whose entire purpose is to be an optimisation.
    """
    text = str(value or "fp32").strip().lower()
    if text in PRECISIONS:
        return text
    logger.warning(
        "Unknown precision %r; training in fp32. Expected one of %s.",
        value, ", ".join(PRECISIONS),
    )
    return "fp32"


def _torch_dtype(precision: str) -> Any:
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def supported_precisions(device: str) -> tuple[str, ...]:
    """Which precisions *device* can actually honour, ``fp32`` always first."""
    kind = (device or "cpu").split(":", 1)[0]
    if kind == "cpu":
        return PRECISIONS
    if kind == "cuda":
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return PRECISIONS
        except Exception:  # noqa: BLE001 - treat an unreadable capability as absent
            logger.debug("could not query bfloat16 support", exc_info=True)
        return ("fp32", "fp16")
    return ("fp32",)


@dataclass
class AmpPolicy:
    """The mixed-precision decisions for one training node's execution.

    Built once per ``execute()`` by :meth:`for_device` and then used in four
    places inside the loop::

        with policy.autocast():
            loss = loss_fn(model(data), targets)
        policy.scale(loss / accumulate_steps).backward()
        ...
        policy.unscale_(optimizer)          # before clipping, no-op in fp32
        torch.nn.utils.clip_grad_norm_(...)
        if policy.step(optimizer):          # may SKIP on an inf/nan gradient
            optimizer_steps += 1

    Under ``fp32`` every one of those is the identity (and ``step`` always
    returns True), so a loop written this way costs nothing when nobody
    asked for AMP.
    """

    #: What is actually in effect on this device.
    precision: str = "fp32"
    #: What the graph asked for. Differs from ``precision`` only when the
    #: device could not honour it; :attr:`fell_back` is the same statement.
    requested: str = "fp32"
    #: ``"cuda"``/``"cpu"``/``"mps"`` -- the device TYPE, index stripped,
    #: which is what ``autocast`` and ``GradScaler`` take.
    device_type: str = "cpu"
    scaler: Any = field(default=None, repr=False)
    #: How many optimizer steps the scaler skipped over an overflowing
    #: gradient. Reported in the node's metrics: a handful at the start of
    #: an fp16 run is the scaler finding its scale, and a steady stream is
    #: the signal to switch to bf16.
    skipped_steps: int = 0

    @classmethod
    def for_device(cls, requested: Any, device: str) -> "AmpPolicy":
        """Resolve *requested* against *device* and build the scaler if needed."""
        wanted = normalize_precision(requested)
        device_type = (device or "cpu").split(":", 1)[0]
        allowed = supported_precisions(device)
        precision = wanted
        if wanted not in allowed:
            precision = "fp32"
            logger.warning(
                "%s does not support %s here, so this run trains in fp32. "
                "Supported on this device: %s.",
                device, wanted, ", ".join(allowed),
            )

        scaler = None
        if precision in SCALED_PRECISIONS:
            try:
                import torch

                scaler = torch.amp.GradScaler(device=device_type)
            except Exception:  # noqa: BLE001 - no scaler is better than no run
                logger.warning(
                    "could not create a GradScaler for %s; training in fp32 "
                    "instead of fp16, because fp16 without a scaler "
                    "underflows small gradients to zero.",
                    device, exc_info=True,
                )
                precision = "fp32"
        return cls(precision=precision, requested=wanted,
                   device_type=device_type, scaler=scaler)

    # ── properties ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when anything at all differs from a plain fp32 run."""
        return self.precision != "fp32"

    @property
    def fell_back(self) -> bool:
        """True when the device could not honour what the graph asked for."""
        return self.precision != self.requested

    @property
    def uses_scaler(self) -> bool:
        return self.scaler is not None

    # ── the four call sites ──────────────────────────────────────────────

    def autocast(self) -> ContextManager[Any]:
        """The context to run a forward pass (and its loss) under.

        ``nullcontext`` in fp32, so an fp32 run is byte-identical to what
        the loop did before this existed.
        """
        if not self.enabled:
            return contextlib.nullcontext()
        import torch

        return torch.amp.autocast(device_type=self.device_type,
                                  dtype=_torch_dtype(self.precision))

    def scale(self, loss: Any) -> Any:
        """The value to call ``.backward()`` on. Identity without a scaler."""
        if self.scaler is None:
            return loss
        return self.scaler.scale(loss)

    def unscale_(self, optimizer: Any) -> None:
        """Divide the scale back out of the gradients, before clipping.

        Without this a gradient-norm clip would measure the SCALED norm and
        clip to a threshold that means nothing. Safe to call once per
        optimizer per step, which is exactly how the loop calls it.
        """
        if self.scaler is None:
            return
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: Any) -> bool:
        """Take the optimizer step (and the scaler update). False when SKIPPED.

        With a scaler this is ``scaler.step()`` followed by
        ``scaler.update()``: the two belong together and separating them
        buys nothing but a way to forget one.

        A skip is normal and not an error. The scaler found an inf or a nan
        among the gradients -- i.e. the loss scale was too high for this
        batch -- so it leaves the weights alone and backs the scale off.
        The next batch proceeds at the lower scale, and the scale climbs
        again after ``growth_interval`` clean steps. The verdict is read
        from the scale having been backed off, because
        ``GradScaler.step`` returns the wrapped optimizer's return value
        (``None`` for every torch optimizer) and cannot be used to tell the
        two cases apart.

        Callers use the result to count OPTIMIZER STEPS honestly: a skipped
        step advanced nothing and must not tick ``max_steps`` toward its
        budget or a scheduler toward its next milestone.
        """
        if self.scaler is None:
            optimizer.step()
            return True
        before = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        applied = self.scaler.get_scale() >= before
        if not applied:
            self.skipped_steps += 1
            logger.debug(
                "fp16 step skipped on an overflowing gradient; loss scale "
                "%s -> %s", before, self.scaler.get_scale(),
            )
        return applied

    # ── checkpoint payload ───────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any] | None:
        """The scaler's state, or None when there is no scaler.

        Plain floats and ints, so it survives ``torch.load(...,
        weights_only=True)`` -- which the checkpoint format requires.
        """
        if self.scaler is None:
            return None
        try:
            return dict(self.scaler.state_dict())
        except Exception:  # noqa: BLE001 - a checkpoint must still be written
            logger.warning("could not read GradScaler state", exc_info=True)
            return None

    def load_state_dict(self, state: Any) -> bool:
        """Restore a scaler state from a checkpoint. True when it took.

        Silently declines when there is nothing to restore into (an fp32 or
        bf16 run reading an fp16 checkpoint) or nothing to restore (a
        checkpoint written before this key existed). Both are ordinary, and
        neither is worth failing a resume over: a fresh scaler re-finds its
        loss scale within a few hundred steps.
        """
        if self.scaler is None or not isinstance(state, dict) or not state:
            return False
        try:
            self.scaler.load_state_dict(state)
        except Exception:  # noqa: BLE001 - a bad state must not kill a resume
            logger.warning(
                "could not restore the GradScaler state from the checkpoint; "
                "starting from a fresh loss scale.", exc_info=True)
            return False
        logger.info("Restored GradScaler at scale=%s", state.get("scale"))
        return True
