"""The GloVe-50d converter: what it parses, what it skips, and what it saves.

The pack downloads 69 MB of gzipped text and the node wants a matrix. Between
them sits one conversion that is paid ONCE, on install, and this file pins the
three things that makes it worth doing:

**It reads both spellings of the format.** The gensim-data release starts with
a word2vec header line (``400000 50``); a raw GloVe file starts with its first
word. A converter that got that wrong would either lose a word or crash on
one, so both are tested rather than assumed.

**The result is exactly what went in.** The vocabulary is stored as one UTF-8
byte blob -- 400k words as a numpy unicode array would be 160 MB of padding --
so the round trip through ``encode``/``decode`` is behaviour, and a CJK word
that came back mangled would be a table the learner cannot look anything up
in.

**The install can draw a bar.** ``core.packs.flows`` forwards this module's
progress dicts verbatim as ``glove-50d`` progress events, so their KEYS are a
contract with the Package Center UI, not an implementation detail. The last
test here runs the real flow step against a tiny gzip to prove the two halves
still meet.

Everything runs against six-word gzips built in ``tmp_path``. Nothing here
downloads anything.
"""

from __future__ import annotations

import gzip
import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest

from app.core.packs import PackMissingError, state
from app.core.packs.catalog import get_item, get_pack
from app.core.packs.paths import asset_dir, sentinel_path
from app.nodes.llm import _glove
from app.nodes.llm import _packs_bridge as bridge

#: Six "words", two of which are not ASCII on purpose -- this project is used
#: in Traditional Chinese, and GloVe's own vocabulary has accented words in it.
WORDS = ["the", ",", "king", "queen", "貓", "café"]


def _vector(row: int) -> list[str]:
    """Row *row*'s 50 numbers as text, distinct per row AND per column.

    Distinct both ways so a transposed or repeated row cannot pass: every
    value below says which cell it came from.
    """
    return [f"{row + column / 1000:.4f}" for column in range(_glove.GLOVE_DIM)]


def _expected(words: list[str] = WORDS) -> np.ndarray:
    """What the matrix has to be, built by the same float32 conversion the
    converter uses so the comparison is about placement, not rounding."""
    return np.stack([np.asarray(_vector(row), dtype=np.float32)
                     for row in range(len(words))])


def _table_lines(words: list[str] = WORDS) -> list[str]:
    return [f"{word} {' '.join(_vector(row))}"
            for row, word in enumerate(words)]


def _write_gz(path: Path, lines: list[str], *, header: bool = False) -> Path:
    """Write *lines* as a GloVe gzip, optionally with the word2vec header."""
    body = list(lines)
    if header:
        body.insert(0, f"{len(lines)} {_glove.GLOVE_DIM}")
    path.write_bytes(gzip.compress(("\n".join(body) + "\n").encode("utf-8")))
    return path


@pytest.fixture(autouse=True)
def isolated_user_data(tmp_path, monkeypatch):
    """Point the pack cache at this test's own throwaway root.

    ``load_glove_50d`` WRITES now -- it records the npz it converted on the
    item's sentinel -- and a developer running this suite may well have the
    real word-vectors pack installed. Without this, a test here would rewrite
    their sentinel to name a file inside a directory pytest is about to
    delete, and their next uninstall would leave the 83 MB table behind:
    exactly the bug the recording exists to prevent.
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "user-data"))
    state.invalidate()
    yield
    state.invalidate()


@pytest.fixture
def glove_gz(tmp_path) -> Path:
    """A six-word table in the raw GloVe spelling (no header line)."""
    return _write_gz(tmp_path / _glove.GLOVE_50D_ASSET, _table_lines())


# ── the file this module is for ───────────────────────────────────────────


def test_asset_names_match_the_catalog():
    """The filename is not a constant this module gets to choose: it is the
    name the Package Center saves the download under, and a copy that drifted
    would look for a file nothing ever writes."""
    item = get_item(get_pack(_glove.GLOVE_PACK), "glove-50d")

    assert _glove.GLOVE_PACK == "word-vectors"
    assert _glove.GLOVE_50D_ASSET == item.filename
    assert _glove.GLOVE_DIM == 50


def test_npz_lands_beside_the_gz(glove_gz):
    """Beside the download, not in a second cache of our own: uninstalling
    the item deletes that directory's file, and a converted table stashed
    somewhere else would outlive the pack it came from."""
    assert _glove.npz_path_for(glove_gz) == glove_gz.parent / "glove-50d.npz"


# ── conversion ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("header", [False, True],
                         ids=["raw-glove", "gensim-header"])
def test_converts_gz_with_and_without_header(tmp_path, header, caplog):
    """Two spellings of the same table, one answer.

    ``400000 50`` is a count and a width; ``the 0.1 0.2 ...`` is a word and a
    vector. Telling them apart is the whole of the first line's handling, and
    getting it wrong either eats the word ``400000`` or reads a header as a
    vector.

    A header mistaken for a vector would not actually reach the vocabulary --
    two tokens is not fifty -- it would be counted as a MALFORMED line
    instead, and every install of the real table would then log a complaint
    about a file that is perfectly fine. Hence the last assertion.
    """
    gz_path = _write_gz(tmp_path / "table.gz", _table_lines(), header=header)

    with caplog.at_level(logging.INFO, logger="app.nodes.llm._glove"):
        npz_path = _glove.ensure_npz(gz_path)

    assert npz_path == _glove.npz_path_for(gz_path)
    assert npz_path.is_file()
    words, matrix = _glove.load_npz(npz_path)
    assert words == WORDS
    assert matrix.shape == (len(WORDS), _glove.GLOVE_DIM)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, _expected(), atol=1e-6)
    assert not any("malformed" in record.getMessage()
                   for record in caplog.records), caplog.text


def test_second_call_does_not_reparse(glove_gz, monkeypatch):
    """The conversion is paid once. 400k lines is a ten-second wait, and a
    node that paid it on every graph run would be unusable."""
    npz_path = _glove.ensure_npz(glove_gz)
    stamp = npz_path.stat().st_mtime_ns

    def _no_reading(*args, **kwargs):
        raise AssertionError("the gz was opened a second time")

    monkeypatch.setattr(gzip, "open", _no_reading)

    assert _glove.ensure_npz(glove_gz) == npz_path
    assert npz_path.stat().st_mtime_ns == stamp
    assert _glove.load_npz(npz_path)[0] == WORDS


def test_a_future_dated_gz_is_converted_once(glove_gz):
    """A gz stamped ahead of the clock must not reconvert on every load.

    Clock skew, an archive restored with its own stamps, a file copied off a
    machine that runs fast: any of them leaves a freshly written npz older
    than its own source, so ``_npz_is_current`` says "stale" forever and
    every graph run silently re-parses 400,000 lines.
    """
    ahead = int((time.time() + 86_400) * 1_000_000_000)
    os.utime(glove_gz, ns=(ahead, ahead))

    first: list[dict] = []
    npz_path = _glove.ensure_npz(glove_gz, first.append)
    assert first, "the first call is the conversion"

    second: list[dict] = []
    assert _glove.ensure_npz(glove_gz, second.append) == npz_path

    assert second == [], "the second call re-parsed a table it already had"
    # And the table it kept is still the right one, not an empty stand-in.
    assert _glove.load_npz(npz_path)[0] == WORDS


def test_a_stale_npz_is_rebuilt(glove_gz):
    """Older than the gz means it was built from a different download -- a
    re-downloaded table whose npz never caught up would serve the old
    vectors forever."""
    npz_path = _glove.ensure_npz(glove_gz)
    npz_path.write_bytes(b"not an npz at all")
    stale = glove_gz.stat().st_mtime_ns - 10_000_000_000
    os.utime(npz_path, ns=(stale, stale))

    words, matrix = _glove.load_npz(_glove.ensure_npz(glove_gz))

    assert words == WORDS
    assert matrix.shape == (len(WORDS), _glove.GLOVE_DIM)


def test_malformed_lines_are_skipped_and_counted(tmp_path, caplog):
    """One bad line must not cost the other 399,999.

    Skipping quietly would be worse than crashing, though: a table that is
    short by a thousand words looks exactly like a table that is not, so the
    count goes in the server log where a teacher can find it.
    """
    lines = _table_lines()
    lines.insert(1, "tooshort 1.0 2.0 3.0")
    lines.insert(3, "toolong " + " ".join(_vector(0) + ["9.9"]))
    lines.append("notnumbers " + " ".join(["nan-ish"] * _glove.GLOVE_DIM))
    lines.append("")  # a blank line is not a malformed vector -- see the count
    gz_path = _write_gz(tmp_path / "ragged.gz", lines)

    with caplog.at_level(logging.INFO, logger="app.nodes.llm._glove"):
        words, matrix = _glove.load_npz(_glove.ensure_npz(gz_path))

    assert words == WORDS, "a bad line took a good one with it"
    assert np.allclose(matrix, _expected(), atol=1e-6)
    assert any("skipped 3 malformed" in record.getMessage()
               for record in caplog.records), caplog.text


def test_a_table_with_no_usable_vectors_is_an_error(tmp_path):
    """An empty npz would be cached forever and would report every word the
    learner types as out-of-vocabulary, with nothing anywhere saying why."""
    gz_path = _write_gz(tmp_path / "junk.gz", ["not a vector", "nor this one"])

    with pytest.raises(ValueError, match="no 50-dimension vectors"):
        _glove.ensure_npz(gz_path)

    assert not _glove.npz_path_for(gz_path).exists()


def test_vocab_round_trips_non_ascii_words(tmp_path):
    """The vocabulary is a UTF-8 byte blob, so decoding it is the load path's
    job and a multi-byte word is where an off-by-one in it shows up."""
    words = ["貓", "café", "naïve", "東京", "a"]
    gz_path = _write_gz(tmp_path / "unicode.gz", _table_lines(words))

    loaded, matrix = _glove.load_npz(_glove.ensure_npz(gz_path))

    assert loaded == words
    assert matrix.shape == (len(words), _glove.GLOVE_DIM)
    assert np.allclose(matrix, _expected(words), atol=1e-6)


def test_a_matrix_that_does_not_match_its_vocab_is_refused(tmp_path):
    """The two arrays are one table in two pieces. If they disagree, every
    lookup after this point returns somebody else's vector -- silently, and
    for the rest of the session."""
    npz_path = tmp_path / _glove.GLOVE_50D_NPZ
    with open(npz_path, "wb") as handle:
        np.savez(handle,
                 matrix=np.zeros((2, _glove.GLOVE_DIM), dtype=np.float32),
                 vocab=np.frombuffer("a\nb\nc".encode("utf-8"), dtype=np.uint8),
                 dim=np.int64(_glove.GLOVE_DIM))

    with pytest.raises(ValueError) as failure:
        _glove.load_npz(npz_path)

    assert "Package Center" in str(failure.value)


# ── the bar the install draws ─────────────────────────────────────────────


def test_progress_frames_have_the_event_shape(tmp_path, monkeypatch):
    """``flows._convert_glove_step`` forwards these dicts verbatim, so their
    keys belong to the Package Center's event contract.

    The cadence is patched down rather than fed 10,000 lines: what matters is
    that frames arrive DURING the parse and not only at the end, because a bar
    that moves once, at 100%, is not a bar.
    """
    assert _glove.PROGRESS_EVERY_LINES == 10_000, "the shipped cadence"
    monkeypatch.setattr(_glove, "PROGRESS_EVERY_LINES", 2)
    gz_path = _write_gz(tmp_path / "table.gz", _table_lines(), header=True)
    frames: list[dict] = []

    _glove.ensure_npz(gz_path, progress=frames.append)

    assert len(frames) > 1, frames
    for frame in frames:
        assert set(frame) == {"bytes_done", "bytes_total", "percent", "text"}
        assert frame["bytes_total"] == len(WORDS)
        assert frame["text"] == "Converting GloVe text to npz (one-time)"
    assert [frame["bytes_done"] for frame in frames] == [2, 4, 6, 6]
    assert frames[-1]["percent"] == 100.0


def test_the_bar_never_goes_backwards_over_a_ragged_file(tmp_path, monkeypatch):
    """Progress counts lines READ, not words kept.

    Counting words would let a frame report less than the one before it the
    moment a line was skipped -- and a bar that jumps backwards is how a UI
    says something has gone wrong, which here it has not.
    """
    monkeypatch.setattr(_glove, "PROGRESS_EVERY_LINES", 2)
    lines = _table_lines()
    lines.insert(1, "tooshort 1.0")
    lines.insert(2, "alsoshort 1.0")
    gz_path = _write_gz(tmp_path / "ragged.gz", lines, header=True)
    frames: list[dict] = []

    _glove.ensure_npz(gz_path, progress=frames.append)

    counted = [frame["bytes_done"] for frame in frames]
    assert counted == [2, 4, 6, 8, 8], counted
    assert counted == sorted(counted), "the bar went backwards"
    assert counted[-1] == len(lines)
    assert frames[-1]["percent"] == 100.0


def test_the_last_frame_finishes_the_bar(tmp_path):
    """A header that promises more lines than the file holds still ends at
    100: the parse IS over, and a bar stuck at 67% with nothing left to do is
    a wait the learner cannot tell from a hang."""
    lines = _table_lines()
    gz_path = tmp_path / "short.gz"
    body = [f"{len(lines) + 3} {_glove.GLOVE_DIM}", *lines]
    gz_path.write_bytes(gzip.compress(("\n".join(body) + "\n").encode("utf-8")))
    frames: list[dict] = []

    _glove.ensure_npz(gz_path, progress=frames.append)

    assert frames[-1]["bytes_done"] == len(WORDS)
    assert frames[-1]["bytes_total"] == len(WORDS) + 3
    assert frames[-1]["percent"] == 100.0


def test_the_bar_only_reaches_100_once_the_file_is_there(tmp_path):
    """100% means converted, not "finished reading".

    Writing and fsyncing 83 MB is the slowest single step of a real
    conversion. A bar that reached the end before it would leave the learner
    watching a finished progress bar do nothing for several seconds, which is
    what a hang looks like.
    """
    gz_path = _write_gz(tmp_path / "table.gz", _table_lines(), header=True)
    npz_path = _glove.npz_path_for(gz_path)
    seen: list[tuple] = []

    _glove.ensure_npz(gz_path, progress=lambda frame: seen.append(
        (frame["percent"], npz_path.is_file())))

    assert seen[-1] == (100.0, True), seen


def test_progress_without_a_header_has_no_total(glove_gz):
    """No header, no count, no percentage -- and None rather than a made-up
    number, which is what ``download.py`` reports for a server that sends no
    ``Content-Length`` too."""
    frames: list[dict] = []

    _glove.ensure_npz(glove_gz, progress=frames.append)

    assert frames, "the end of the parse is always reported"
    assert frames[-1] == {"bytes_done": len(WORDS), "bytes_total": None,
                          "percent": None,
                          "text": "Converting GloVe text to npz (one-time)"}


def test_progress_is_optional(glove_gz):
    """The node's own first-use path has no bar to draw."""
    assert _glove.ensure_npz(glove_gz, progress=None).is_file()


# ── writing it down ───────────────────────────────────────────────────────


def test_atomic_write_leaves_no_part_file(glove_gz):
    """The npz appears whole or not at all: a reader that opened a
    half-written file would see a corrupt zip, and the scratch file must not
    be left behind for the pack's uninstall to miss."""
    _glove.ensure_npz(glove_gz)

    assert list(glove_gz.parent.glob("*.part")) == []
    assert sorted(path.name for path in glove_gz.parent.iterdir()) == sorted(
        [_glove.GLOVE_50D_ASSET, _glove.GLOVE_50D_NPZ])


def test_a_failed_write_leaves_neither_a_part_file_nor_an_npz(
        glove_gz, monkeypatch):
    """A disk that filled up mid-save must not leave something that looks
    converted -- the next call has to try again, not load wreckage."""
    def _boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(np, "savez", _boom)

    with pytest.raises(OSError, match="no space"):
        _glove.ensure_npz(glove_gz)

    assert not _glove.npz_path_for(glove_gz).exists()
    assert list(glove_gz.parent.glob("*.part")) == []


# ── the node's entry point ────────────────────────────────────────────────


def test_load_glove_50d_reads_the_downloaded_asset(glove_gz, monkeypatch):
    """Source to matrix in one call, converting on the way if it has to."""
    asked: list[tuple] = []

    def _asset_path(pack_id, filename):
        asked.append((pack_id, filename))
        return glove_gz

    monkeypatch.setattr(bridge, "asset_path", _asset_path)

    words, matrix = _glove.load_glove_50d()

    assert asked == [(_glove.GLOVE_PACK, _glove.GLOVE_50D_ASSET)]
    assert words == WORDS
    assert matrix.shape == (len(WORDS), _glove.GLOVE_DIM)


def test_a_lazy_conversion_records_the_npz_so_the_uninstall_finds_it():
    """A table the INSTALL never converted still leaves a record behind.

    ``flows._convert_glove_step`` records the npz it makes, but it is not the
    only converter any more: ``WordVector`` reaches ``load_glove_50d`` on an
    older install, or one whose convert step was skipped, and writes 83 MB of
    its own. The catalog names one file per item, so nothing else on disk can
    tie that file to the download -- an unrecorded npz survives the uninstall
    that was supposed to free it, and nothing is left that names it.

    Run against the real seam rather than a patched ``asset_path``: the
    recording has to survive the sentinel, the removal checks and the path
    rules, and every one of those is a place a plausible-looking record could
    stop being one.
    """
    from app.core.packs import flows

    pack = get_pack(_glove.GLOVE_PACK)
    item = get_item(pack, "glove-50d")
    asset_dir().mkdir(parents=True, exist_ok=True)
    gz_path = _write_gz(asset_dir() / item.filename, _table_lines())
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "asset", "url": item.url, "path": str(gz_path),
        "bytes": gz_path.stat().st_size, "sha256": None,
        "at": "2026-08-28T00:00:00Z",
    })

    words, _matrix = _glove.load_glove_50d()
    npz_path = _glove.npz_path_for(gz_path)

    assert words == WORDS
    assert npz_path.is_file()

    sentinel = state.read_sentinel(sentinel_path(pack.pack_id, item.item_id))
    assert sentinel is not None
    assert sentinel["derived"] == [str(npz_path)]
    # Extended, not replaced: the download's own record is what makes the
    # item readable as present at all.
    assert sentinel["path"] == str(gz_path)

    assert flows.remove_item(pack, item.item_id) is True
    assert not npz_path.exists()
    assert not gz_path.exists()


def test_a_lazy_conversion_with_no_sentinel_still_loads(glove_gz, monkeypatch,
                                                        caplog):
    """Nothing to record on is not a reason to refuse a lookup.

    The mirror of ``test_packs_flows``' equivalent, from the node's side. A
    learner asking for a word vector gets one whatever the bookkeeping did:
    the table has already been read by the time anything can go wrong here,
    and the log line is how a maintainer finds out that this npz will not be
    removed with the pack.
    """
    from app.core.packs import flows

    monkeypatch.setattr(bridge, "asset_path", lambda *args, **kwargs: glove_gz)

    with caplog.at_level(logging.WARNING, logger=flows.__name__):
        words, _matrix = _glove.load_glove_50d()

    assert words == WORDS
    assert "will not be removed" in caplog.text


def test_load_glove_50d_without_pack_raises_pack_missing_error(monkeypatch):
    """The frontend routes on the ``(pack=<id>)`` suffix to offer the
    download, so this is the one thing about the message that is not
    wording."""
    monkeypatch.setattr(bridge, "asset_path", lambda *args, **kwargs: None)

    with pytest.raises(PackMissingError) as failure:
        _glove.load_glove_50d()

    assert failure.value.pack_id == _glove.GLOVE_PACK
    assert str(failure.value).endswith("(pack=word-vectors)")
    assert "Package Center" in str(failure.value)


# ── the seam with the installer ───────────────────────────────────────────


def test_flows_convert_step_calls_this_ensure_npz(tmp_path):
    """The other half of ``test_packs_flows``' fake converter.

    That file proves the flow forwards whatever ``ensure_npz`` reports; this
    one proves the real ``ensure_npz`` reports something the flow can forward
    -- keys and all -- and that the file the node will look for is there
    afterwards.
    """
    from app.core.packs import flows

    gz_path = _write_gz(tmp_path / _glove.GLOVE_50D_ASSET, _table_lines(),
                        header=True)
    events: list[dict] = []

    flows._convert_glove_step(gz_path, emit=events.append)

    progress = [event for event in events if event["type"] == "progress"]
    assert progress, [event["type"] for event in events]
    assert all(event["item"] == "glove-50d" for event in progress)
    assert progress[-1] == {
        "type": "progress", "item": "glove-50d",
        "bytes_done": len(WORDS), "bytes_total": len(WORDS), "percent": 100.0,
        "text": "Converting GloVe text to npz (one-time)"}
    assert _glove.npz_path_for(gz_path).is_file()
