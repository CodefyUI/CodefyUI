"""TensorBoard event files, written without depending on TensorBoard.

Why this is hand-rolled
-----------------------
``torch.utils.tensorboard.SummaryWriter`` needs the ``tensorboard``
distribution installed, which pulls in grpcio, protobuf, werkzeug, absl-py
and markdown -- tens of megabytes on every CodefyUI install, for a feature
that is off by default and that a user only benefits from once they have
installed TensorBoard anyway to LOOK at the result. So the writer is here
and the dependency is not.

What that costs is a hand-written encoder for a very small slice of two
protobuf messages, and the risk that the slice is subtly wrong. That risk is
paid down by the test, which parses what this writes with the real
``tensorboard.backend.event_processing`` reader (``tensorboard`` is in the
``dev`` extra) rather than with a reader from this file.

The format, in full
-------------------
An event file is a sequence of TFRecords::

    uint64  length of the payload      (little endian)
    uint32  masked CRC-32C of those 8 length bytes
    bytes   payload
    uint32  masked CRC-32C of the payload

The payload is a serialised ``Event``. Only three of its fields are used:

    Event.wall_time    field 1, double
    Event.step         field 2, int64
    Event.file_version field 3, string   -- first record of every file
    Event.summary      field 5, message

and a ``Summary`` here is always one ``Summary.Value`` carrying a scalar::

    Summary.value          field 1, repeated message
    Summary.Value.tag      field 1, string
    Summary.Value.simple_value field 2, float (32-bit)

``simple_value`` is the pre-2.0 scalar encoding, and it is deliberate:
TensorBoard's reader migrates it to the modern tensor form on the way in
(``tensorboard.data_compat``), so one small encoder serves every reader
version, while emitting the tensor form directly would mean also emitting
``SummaryMetadata`` with the right plugin name and a ``TensorProto``.

The masking on the CRC is TensorFlow's, not a standard: it exists so a
record whose payload happens to be a valid CRC cannot be mistaken for a
framed one.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

logger = logging.getLogger(__name__)

#: What every event file must announce in its first record.
FILE_VERSION = "brain.Event:2"

#: Seconds between forced flushes. A training loop's metrics should show up
#: in a live TensorBoard within a few seconds, and a run that dies without
#: closing the writer should lose at most this much.
FLUSH_INTERVAL_S = 5.0

#: Largest finite value ``simple_value``'s 32-bit float can hold. A finite
#: float64 above it is clamped here rather than handed to ``struct.pack``,
#: which would raise ``OverflowError``.
FLOAT32_MAX = 3.4028234663852886e38

#: Castagnoli polynomial, bit-reversed. The one TFRecord uses; NOT the
#: CRC-32 in ``zlib``, which is a different polynomial and would make every
#: record fail its checksum.
_CRC32C_POLY = 0x82F63B78

_CRC32C_TABLE: list[int] = []


def _build_table() -> list[int]:
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table.append(crc)
    return table


def crc32c(data: bytes) -> int:
    """CRC-32C (Castagnoli) of *data*, as TFRecord frames it."""
    if not _CRC32C_TABLE:
        _CRC32C_TABLE.extend(_build_table())
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    """TensorFlow's masked CRC: rotate right 15, then add a constant."""
    crc = crc32c(data)
    return ((((crc >> 15) | (crc << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


# ── protobuf wire format, the four pieces this file needs ────────────────


def _varint(value: int) -> bytes:
    """Base-128 varint. Non-negative only; nothing here encodes a negative."""
    out = bytearray()
    value &= (1 << 64) - 1
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _length_delimited(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _scalar_summary(tag: str, value: float) -> bytes:
    """One ``Summary`` holding a single ``simple_value``."""
    entry = (
        _length_delimited(1, tag.encode("utf-8"))
        + _tag(2, 5) + struct.pack("<f", float(value))
    )
    return _length_delimited(1, entry)


def _event(
    *,
    wall_time: float,
    step: int = 0,
    summary: bytes | None = None,
    file_version: str | None = None,
) -> bytes:
    out = _tag(1, 1) + struct.pack("<d", float(wall_time))
    if step:
        out += _tag(2, 0) + _varint(int(step))
    if file_version is not None:
        out += _length_delimited(3, file_version.encode("utf-8"))
    if summary is not None:
        out += _length_delimited(5, summary)
    return out


def _record(payload: bytes) -> bytes:
    header = struct.pack("<Q", len(payload))
    return (
        header
        + struct.pack("<I", masked_crc32c(header))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def event_filename(now: float | None = None) -> str:
    """The name TensorBoard's directory scan expects.

    It globs for ``events.out.tfevents.*`` and sorts by the timestamp in the
    name, so the prefix and the position of the timestamp are load-bearing;
    the rest is provenance. The host part is sanitised because it ends up in
    a filename and a Windows machine name is not guaranteed to be safe there.

    The trailing counter is a random suffix rather than the ``0`` torch's own
    ``SummaryWriter`` starts from, and it is there for the same reason torch
    randomises it: timestamp + host + pid is NOT unique. Two writers opening
    the same directory in the same second from the same process would
    otherwise pick the same name and the second ``open(..., "wb")`` would
    truncate the first one's file, losing a whole series with no error.
    """
    stamp = int(time.time() if now is None else now)
    host = "".join(
        c if (c.isalnum() or c in "-_") else "-"
        for c in socket.gethostname()
    ) or "localhost"
    unique = uuid4().hex[:8]
    return f"events.out.tfevents.{stamp}.{host}.{os.getpid()}.{unique}"


class EventFileWriter:
    """Appends scalar events to one TensorBoard event file.

    Not thread-safe, and does not need to be: one writer belongs to one
    training node, which writes from the single thread it executes in.

    Deliberately total, like every other piece of instrumentation in the run
    path (see ``ExecutionContext.log_metric``): a full disk must not kill a
    six-hour training run. A write that fails disables the writer and logs
    once, and the caller keeps training.
    """

    #: Class-level default so :meth:`__del__` is safe on a half-built
    #: object. ``__init__`` can raise before the ``open()`` -- a read-only
    #: parent, a path too long, a full disk -- and ``open_run_writer``
    #: correctly turns that into ``None``; without this attribute the
    #: doomed instance then printed ``Exception ignored in: <function
    #: EventFileWriter.__del__>`` with an ``AttributeError`` to stderr.
    _file: BinaryIO | None = None

    def __init__(self, logdir: Path | str) -> None:
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.path = self.logdir / event_filename()
        self._last_flush = time.monotonic()
        self._file = open(self.path, "wb")
        self._write(lambda: _event(wall_time=time.time(),
                                   file_version=FILE_VERSION))

    @property
    def closed(self) -> bool:
        return self._file is None

    def _write(self, build: Callable[[], bytes]) -> None:
        """Encode and append one event. Never raises.

        *build* is a CALLABLE rather than the finished bytes, and that is
        the whole point: encoding happens inside the guard. Passing
        ``_event(...)`` as an argument evaluated it in the caller, where an
        un-encodable value (a float outside float32's range, a tag holding
        a lone surrogate) raised straight out of ``record_metric``, out of
        ``TrainingLoop.execute`` and killed the run -- with the writer left
        open, so the class docstring's "a write that fails disables the
        writer and the caller keeps training" was true only for I/O errors.
        """
        handle = self._file
        if handle is None:
            return
        try:
            handle.write(_record(build()))
            now = time.monotonic()
            if now - self._last_flush >= FLUSH_INTERVAL_S:
                handle.flush()
                self._last_flush = now
        except Exception:  # noqa: BLE001 - see the class docstring
            logger.warning(
                "TensorBoard logging to %s stopped after a write error; the "
                "run continues without it.", self.path, exc_info=True)
            self.close()

    def add_scalar(
        self,
        tag: str,
        value: Any,
        step: int,
        wall_time: float | None = None,
    ) -> None:
        """Record one point of the series *tag*.

        A value that is not a finite number is DROPPED rather than written.
        TensorBoard renders a NaN as a gap in the line, which is arguably
        the honest picture of a diverged loss -- but it also poisons the
        y-axis auto-scaling for every other series in the same chart, and
        the run's own metric store already keeps the NaN.

        A value that IS finite but outside float32's range is CLAMPED to the
        rail rather than dropped. ``simple_value`` is a 32-bit float, so
        ``struct.pack("<f", 1e39)`` raises ``OverflowError`` -- reachable
        from a float64 model, and the NaN filter above does not catch it
        because 1e39 is perfectly finite. Clamping keeps the point on the
        chart, at the largest magnitude the format can say, instead of
        either losing it or writing an ``inf`` that ruins the axis.
        """
        try:
            numeric = float(value)
            index = int(step)
        except (TypeError, ValueError):
            return
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return
        if numeric > FLOAT32_MAX:
            numeric = FLOAT32_MAX
        elif numeric < -FLOAT32_MAX:
            numeric = -FLOAT32_MAX
        stamp = time.time() if wall_time is None else wall_time
        self._write(lambda: _event(
            wall_time=stamp,
            step=index,
            summary=_scalar_summary(str(tag), numeric),
        ))

    def close(self) -> None:
        """Flush and release the file. Idempotent, and never raises.

        No logging in here, deliberately: :meth:`__del__` calls it, and a
        ``logging`` call during interpreter shutdown can find the handlers
        already torn down.
        """
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            handle.close()  # close() flushes
        except Exception:  # noqa: BLE001 - nothing useful to do about it
            pass

    def __del__(self) -> None:
        # The safety net for the path that matters: a node that raises
        # mid-training never reaches its explicit close, and the events it
        # DID write are exactly what someone is about to go looking for.
        self.close()

    def __enter__(self) -> "EventFileWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def run_logdir(run_id: str, node_id: str = "") -> Path:
    """``<data root>/runs/<run_id>/tb/<node_id>`` -- where a run's events go.

    Beside ``models/`` and ``images/`` under the same data root, so project
    mode moves it into the project's ``assets/`` with everything else and a
    ``cdui`` install keeps it out of the repo.

    The per-node leaf is what makes two training nodes in one graph readable:
    TensorBoard treats each sub-directory of ``--logdir`` as a separate run
    and colours them separately, so a graph with a pretrain loop and a
    finetune loop draws two curves instead of one zig-zag.
    """
    from ..config import settings
    from .checkpoints import _safe_part

    root = settings.MODELS_DIR.parent / "runs" / _safe_part(run_id) / "tb"
    return root / _safe_part(node_id) if node_id else root


#: ``exec_run_artifacts.kind`` for a TensorBoard log directory. Note it is
#: the only core artifact kind whose ``path`` is a DIRECTORY -- that is what
#: ``tensorboard --logdir`` takes, and pointing it at one event file inside
#: would hide the sibling nodes' curves.
ARTIFACT_KIND = "tensorboard"


def open_run_writer(context: Any) -> EventFileWriter | None:
    """A writer for the currently-executing node, or ``None``.

    ``None`` -- with an explanatory log line -- when the run has nowhere to
    put the artifact ROW. That is the "no row, no file" rule
    ``core.checkpoints`` states in full: the row is the only index of what a
    run wrote, so producing a directory nothing references would leave
    litter that no retention sweep can find. Runs with no durable signal
    consumer (the REST contract runner, an exported script, a bare
    ``execute_graph``) are the ones this turns away.
    """
    if context is None:
        return None
    can_record = getattr(context, "can_record_artifacts", None)
    if not (callable(can_record) and can_record()):
        logger.info(
            "tensorboard=true, but this run records no artifacts, so no "
            "event files were written -- the directory would have had no "
            "row referencing it.")
        return None
    node_id = getattr(context, "current_node_id", "") or ""
    run_id = getattr(context, "execution_id", "") or "run"
    try:
        return EventFileWriter(run_logdir(run_id, node_id))
    except Exception:  # noqa: BLE001 - instrumentation never fails a run
        logger.warning(
            "TensorBoard event file could not be created; the run continues "
            "without it.", exc_info=True)
        return None


def close_and_register(writer: EventFileWriter | None, context: Any) -> str | None:
    """Close *writer* and register its directory on the run. Returns the path.

    Queued LAST by every caller, per ``ArtifactSignal``'s tail-safety
    obligation: the outbox drops the OLDEST item, so an artifact row queued
    before a burst of per-batch progress frames is the one that disappears.

    Called from a ``finally``, so it runs on the exception path as well --
    which is what keeps "no row, no file" true for a run that dies
    mid-training, and is why it must be TOTAL. An exception raised in here
    would propagate out of that ``finally`` and REPLACE whatever actually
    killed the run with an error about logging.
    """
    if writer is None:
        return None
    try:
        logdir = str(writer.logdir)
        events = writer.path.name
        writer.close()
        log_artifact = getattr(context, "log_artifact", None)
        if callable(log_artifact):
            log_artifact(ARTIFACT_KIND, logdir, {"events": events})
        logger.info("TensorBoard events written to %s (open with "
                    "`tensorboard --logdir %s`)", logdir, logdir)
        return logdir
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning(
            "TensorBoard events could not be registered on the run; the "
            "directory is on disk but nothing references it.", exc_info=True)
        return None
