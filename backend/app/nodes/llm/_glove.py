"""The GloVe-50d table: from the gzip the Package Center downloads to a matrix.

The ``word-vectors`` pack fetches ``glove-wiki-gigaword-50.gz`` -- 400,000
words by 50 dimensions, as gzipped TEXT. Reading it is roughly ten seconds of
splitting strings and parsing floats, which is fine to pay once and absurd to
pay on every graph run, so the install converts it to a ``.npz`` sitting beside
the download and the node loads that instead.

Four decisions are worth knowing about.

**The npz lives beside the gz, not in a cache of our own.** ``remove_item``
deletes the download; a converted copy stashed somewhere else would outlive
the pack it came from and keep answering lookups for a table the learner
believes they uninstalled.

**The vocabulary is one UTF-8 byte blob.** 400k words in a numpy unicode array
is 400k x (longest word) x 4 bytes -- around 160 MB of mostly padding. Joined
with newlines and encoded, the same words are about 3 MB. A word cannot
contain a newline (the file format is one vector per line), so the split on
load is the exact inverse of the join.

**Conversion is atomic.** The npz is written to a scratch file and moved into
place, so a reader never opens a half-written zip, and a save that dies
half-way leaves nothing that looks converted.

**The progress dicts are a contract, not a convenience.**
``core.packs.flows._convert_glove_step`` forwards them verbatim as
``glove-50d`` progress events, so ``bytes_done`` / ``bytes_total`` /
``percent`` are the Package Center's own keys and the UI needs to learn
nothing new to draw this wait. ``bytes_done`` counts LINES here -- the unit
the converter can actually measure -- which is why the bar it draws is the
same bar and not the same number.

Imports stay cheap: this module is reached at startup through the node modules
the registry scans, so there is no torch here and ``app.core.packs`` is
imported inside the one function that needs its exception class.
"""

from __future__ import annotations

import gzip
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import _packs_bridge

if TYPE_CHECKING:  # the runtime import is inside ``_pack_missing``
    from ...core.packs import PackMissingError

logger = logging.getLogger(__name__)

#: The Package Center pack the table comes from.
GLOVE_PACK = "word-vectors"

#: What the download is saved as. Must equal the ``glove-50d`` catalog item's
#: ``filename`` -- ``test_glove_loader`` compares the two, because a copy that
#: drifted would look for a file nothing ever writes.
GLOVE_50D_ASSET = "glove-wiki-gigaword-50.gz"

#: What the conversion is saved as, beside the download.
GLOVE_50D_NPZ = "glove-50d.npz"

#: Row width. GloVe ships 50d, 100d, 200d and 300d tables; this module is
#: about the 50d one, and a file of any other width is a different download.
GLOVE_DIM = 50

#: How often the parse reports in. Two frames a second on a real table, which
#: is a bar that moves rather than a number that jumps.
PROGRESS_EVERY_LINES = 10_000

#: What the person watching the install reads while this runs. It is a
#: one-time cost and saying so is the difference between a wait and a bug.
PROGRESS_TEXT = "Converting GloVe text to npz (one-time)"

#: Nodes run on up to four worker threads, so two of them can reach a
#: not-yet-converted table at the same moment. Without this they would both
#: parse it and both write the scratch file.
_CONVERT_LOCK = threading.Lock()


def glove_source_path() -> Path | None:
    """The downloaded GloVe gzip, or None when the pack is not installed.

    Through the bridge, never through ``core.packs`` directly: the bridge is
    the seam node tests patch, and it is what keeps the packs import lazy.
    """
    return _packs_bridge.asset_path(GLOVE_PACK, GLOVE_50D_ASSET)


def npz_path_for(gz_path: Path) -> Path:
    """Where *gz_path*'s converted table belongs -- next to it."""
    return Path(gz_path).parent / GLOVE_50D_NPZ


def _pack_missing(message: str) -> PackMissingError:
    """A ``PackMissingError`` for this pack, imported where it is used.

    Lazy for the same reason ``_packs_bridge`` is lazy: a node module must
    still IMPORT in an install with no packs package, or the registry scan
    that builds the palette takes the whole palette down with it.
    """
    from ...core.packs import PackMissingError

    return PackMissingError(GLOVE_PACK, message)


def _header_count(line: str) -> int | None:
    """The word count from a word2vec header line, or None if this is not one.

    ``400000 50`` is a count and a width; ``the 0.418 0.249 ...`` is a word and
    fifty numbers. Exactly two INTEGER tokens is the whole difference, and it
    is not an ambiguous test: a vector line has fifty-one tokens and its first
    one is a word. gensim-data's release carries the header, a raw GloVe file
    does not, and both have to load.
    """
    parts = line.split()
    if len(parts) != 2:
        return None
    try:
        count, _width = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return count


def _npz_is_current(npz_path: Path, gz_path: Path) -> bool:
    """Is the converted table the one this gz would produce?

    Answered by timestamps, because the alternative -- hashing 69 MB -- costs
    more than the parse it is trying to avoid. Not-older rather than strictly
    newer: a converted file and its source can share a timestamp on a
    filesystem with a coarse clock, and re-converting on every load is a worse
    answer than trusting a tie.
    """
    try:
        return npz_path.stat().st_mtime_ns >= gz_path.stat().st_mtime_ns
    except OSError:
        # No npz yet (the usual case), or one we cannot stat. Either way the
        # answer is "convert", and the conversion will say why if it cannot.
        return False


def _report(progress: Callable[[dict], None] | None, lines: int,
            total: int | None, *, done: bool = False) -> None:
    """Send one progress frame, in the Package Center's shape.

    The keys are ``flows``' to forward, so they are spelled the way
    ``download.py`` spells them even though the unit here is lines. An unknown
    total (a file with no header) reports None rather than a made-up number --
    the same thing a download with no ``Content-Length`` reports.
    """
    if progress is None:
        return
    percent = None
    if total:
        # The final frame is 100 by definition: the parse is over, and a bar
        # that stops at 99.4% because the header over-counted is a wait the
        # learner cannot tell from a hang.
        percent = (100.0 if done
                   else min(100.0, round(100.0 * lines / total, 1)))
    progress({"bytes_done": lines, "bytes_total": total, "percent": percent,
              "text": PROGRESS_TEXT})


def _vector(values: list[str]) -> np.ndarray | None:
    """*values* as a row, or None when this line is not one.

    Both halves matter: 300d GloVe in the 50d slot has the wrong COUNT, and a
    truncated download's last line has the right count and a half-written
    number in it.
    """
    if len(values) != GLOVE_DIM:
        return None
    try:
        return np.asarray(values, dtype=np.float32)
    except ValueError:
        return None


def _parse_gz(gz_path: Path, progress: Callable[[dict], None] | None
              ) -> tuple[list[str], list[np.ndarray], int | None, int]:
    """Read *gz_path*: ``(words, rows, header total, lines read)``.

    ``errors="replace"`` rather than a crash: one byte that is not UTF-8
    should cost that word its spelling, not the other 399,999 their vectors.

    Progress counts LINES READ, not words kept, so the bar cannot go backwards
    when a line is skipped -- and it is reported at the bottom of the loop
    rather than inside the good branch, so a malformed line at exactly 10,000
    cannot eat a frame.
    """
    words: list[str] = []
    rows: list[np.ndarray] = []
    total: int | None = None
    malformed = 0
    lines = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            line = raw.rstrip("\n")
            if index == 0:
                count = _header_count(line)
                if count is not None:
                    # ``or None``: a header claiming zero words gives nothing
                    # to compute a percentage against.
                    total = count or None
                    continue
            if not line:
                # A trailing newline is not a malformed vector.
                continue

            lines += 1
            word, *values = line.split(" ")
            vector = _vector(values)
            if vector is None:
                malformed += 1
            else:
                words.append(word)
                rows.append(vector)
            if lines % PROGRESS_EVERY_LINES == 0:
                _report(progress, lines, total)

    if malformed:
        # INFO, not WARNING: the shipped table is clean, so this fires for a
        # hand-made or truncated file and the count is what says which.
        logger.info("skipped %d malformed line(s) while converting %s",
                    malformed, gz_path)
    return words, rows, total, lines


def _save_npz(npz_path: Path, words: list[str], rows: list[np.ndarray]) -> None:
    """Write the table to *npz_path*, whole or not at all.

    The scratch file carries this process's pid: the server and a
    ``cdui packs install`` in a terminal are two interpreters that can convert
    the same download at the same moment, and one shared scratch name would
    let them interleave into a corrupt zip. Each writes its own complete file
    and ``os.replace`` -- atomic on POSIX and on Windows alike -- makes
    whichever finishes last the one readers see.

    ``np.savez`` is handed an open FILE, not the path: given a name that does
    not end in ``.npz`` it helpfully appends one, and the scratch file would
    become ``glove-50d.npz.<pid>.part.npz``, left behind after a rename that
    moved a file that was never there.
    """
    part_path = npz_path.with_name(f"{npz_path.name}.{os.getpid()}.part")
    try:
        with open(part_path, "wb") as handle:
            np.savez(
                handle,
                matrix=np.stack(rows),
                vocab=np.frombuffer("\n".join(words).encode("utf-8"),
                                    dtype=np.uint8),
                dim=np.int64(GLOVE_DIM),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part_path, npz_path)
    except BaseException:
        # Including KeyboardInterrupt and a cancelled install: whatever went
        # wrong, a half-written scratch file must not survive it.
        try:
            part_path.unlink()
        except OSError:
            pass
        raise


def ensure_npz(gz_path: Path, progress: Callable[[dict], None] | None = None
               ) -> Path:
    """Convert *gz_path* to ``glove-50d.npz`` beside it, once, and return it.

    Called by the Package Center install (``flows._convert_glove_step``) right
    after the download lands, and by the node on first use for a table that
    somehow never got converted. Both pass through the same lock, so a table
    is parsed once however many callers arrive together.

    *progress* is called with the frames described in the module docstring --
    every ``PROGRESS_EVERY_LINES`` lines and once at the end.

    Raises ``ValueError`` for a file with nothing in it this module
    recognises, and lets ``gzip``'s own ``OSError`` through for one that is
    not a gzip at all. Neither is defended against beyond saying so: the
    download is checked against a sha256 in the catalog, so a file that gets
    this far and is still wrong was put there by hand.
    """
    gz_path = Path(gz_path)
    npz_path = npz_path_for(gz_path)
    if _npz_is_current(npz_path, gz_path):
        return npz_path

    with _CONVERT_LOCK:
        # Asked again inside the lock: whoever held it may have just done the
        # conversion this call was queueing up to start.
        if _npz_is_current(npz_path, gz_path):
            return npz_path

        words, rows, total, lines = _parse_gz(gz_path, progress)
        if not rows:
            # Never write this one down. An empty npz would be newer than the
            # gz -- so it would be trusted forever -- and would report every
            # word the learner types as out-of-vocabulary with nothing
            # anywhere saying why.
            raise ValueError(
                f"{gz_path} contains no {GLOVE_DIM}-dimension vectors, so it "
                f"is not the GloVe-50d table. Remove the word vectors pack in "
                f"Package Center and download it again")

        _report(progress, lines, total, done=True)
        _save_npz(npz_path, words, rows)
        logger.info("converted %s to %s (%d words x %d dims)",
                    gz_path.name, npz_path.name, len(words), GLOVE_DIM)
    return npz_path


def load_npz(npz_path: Path) -> tuple[list[str], np.ndarray]:
    """Read a converted table back: ``(words, [N, 50] float32)``.

    The two arrays are one table in two pieces, so they are checked against
    each other before either is handed out. If they disagree, every lookup
    after this point silently returns somebody else's vector -- which is worse
    than not loading at all, and unfixable from inside a graph.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        # Read INSIDE the context: an NpzFile reads a member when it is asked
        # for one, and the zip is closed on the way out.
        matrix = np.asarray(data["matrix"], dtype=np.float32)
        blob = data["vocab"]
        stored_dim = int(data["dim"]) if "dim" in data else GLOVE_DIM

    text = blob.tobytes().decode("utf-8", errors="replace")
    # ``"".split("\n")`` is ``[""]`` -- one empty word, not no words.
    words = text.split("\n") if text else []

    if stored_dim != GLOVE_DIM or matrix.shape != (len(words), GLOVE_DIM):
        raise ValueError(
            f"{npz_path} holds {matrix.shape} vectors for {len(words)} words "
            f"at {stored_dim} dimensions, which is not a {GLOVE_DIM}d table. "
            f"Remove the word vectors pack in Package Center and download it "
            f"again")
    return words, matrix


def load_glove_50d() -> tuple[list[str], np.ndarray]:
    """The real GloVe-50d table: ``(words, [400000, 50] float32)``.

    Converts on the way if the install did not (an older install, or one whose
    convert step was skipped). Raises ``PackMissingError`` -- message ending
    in ``(pack=word-vectors)``, which is what the editor reads to offer the
    download -- when the gzip is not on disk.
    """
    gz_path = glove_source_path()
    if gz_path is None:
        # Asked of the FILE, not of a cached probe: ``asset_path`` re-checks
        # the sentinel and the bytes right now, so a cache someone cleaned out
        # by hand reads as missing here rather than as a crash three lines
        # further down.
        raise _pack_missing(
            "The GloVe word vector table is not downloaded. Open Package "
            "Center > Word vectors (GloVe) and download it; graph runs never "
            "download pack contents")
    return load_npz(ensure_npz(gz_path))
