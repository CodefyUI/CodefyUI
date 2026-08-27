"""Fetching a pack's bytes, with a progress bar somebody is watching.

A 470 MB download on a classroom connection takes minutes, so the two things
that matter are that the number on screen means something and that Stop
works. Both are pinned here:

* progress only ever goes UP, is throttled to a handful of events a second
  per item, and always ends on the true total -- including for a file that
  was already in the cache and therefore reported nothing;
* cancelling stops between files AND mid-file, from inside the progress hook.

And one property that is about trust rather than pixels: the GloVe table has
no recorded sha256 yet, so its download must LOG the digest it computed --
that is how a maintainer records it -- and must say, in the log, that nothing
was verified.

Nothing here touches the network. ``HfApi`` and ``hf_hub_download`` are
replaced with fakes that write files into ``tmp_path``, and the asset path
goes through the same in-memory ``urlopen`` fake as ``test_asset_cache``.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import types
import urllib.request
from pathlib import Path

import pytest

from app.core.asset_cache import cache_dir
from app.core.packs import download, state
from app.core.packs.catalog import ModelItem, Pack, get_item, get_pack
from app.core.packs.errors import (
    PackCancelled,
    PackInstallError,
    PackInsufficientDisk,
)
from app.core.packs.paths import asset_dir, hf_cache_dir, sentinel_path

PAYLOAD = b"g" * ((1 << 20) + 7)
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def user_data_dir(tmp_path, monkeypatch):
    """A throwaway cache root and a cold probe cache for every test."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    state.invalidate()
    yield tmp_path
    state.invalidate()


def _usage(free: int):
    """A ``shutil.disk_usage`` result carrying the only field we read."""
    return types.SimpleNamespace(total=free * 2, used=free, free=free)


class _Sibling:
    """A ``huggingface_hub`` ``RepoSibling``, reduced to what we read."""

    def __init__(self, rfilename: str, size: int | None = 0):
        self.rfilename = rfilename
        self.size = size


class _FakeApi:
    """An ``HfApi`` that answers ``model_info`` from a canned file list."""

    siblings: list[_Sibling] = []
    calls: list[dict] = []

    def model_info(self, repo_id, *, revision=None, files_metadata=False,
                   **kwargs):
        type(self).calls.append({"repo_id": repo_id, "revision": revision,
                                 "files_metadata": files_metadata})
        return types.SimpleNamespace(siblings=list(type(self).siblings))


@pytest.fixture
def fake_api(monkeypatch):
    """Install ``_FakeApi``; returns a setter for the sibling list."""
    import huggingface_hub

    _FakeApi.siblings = []
    _FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    def _set(*names_and_sizes):
        _FakeApi.siblings = [_Sibling(name, size)
                             for name, size in names_and_sizes]
        return _FakeApi.calls

    return _set


def _snapshot_file(cache_root, repo_id: str, filename: str) -> Path:
    """Where the real ``hf_hub_download`` would have put this file."""
    target = (Path(cache_root) / f"models--{repo_id.replace('/', '--')}"
              / "snapshots" / "rev0001" / filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def fake_hub_download(monkeypatch):
    """A ``hf_hub_download`` that writes real files and drives the tqdm class."""
    import huggingface_hub

    calls: list[dict] = []

    def _download(*, repo_id, filename, revision=None, cache_dir=None,
                  tqdm_class=None, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename,
                      "revision": revision, "cache_dir": cache_dir,
                      "tqdm_class": tqdm_class, **kwargs})
        target = _snapshot_file(cache_dir, repo_id, filename)
        body = b"z" * 100
        target.write_bytes(body)
        if tqdm_class is not None:
            bar = tqdm_class(total=len(body), initial=0, desc=filename,
                             unit="B", unit_scale=True)
            bar.update(60)
            bar.update(40)
            bar.close()
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    return calls


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._buffer = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def served(monkeypatch):
    """Serve bytes from memory for every ``urlopen``."""
    def _serve(payload: bytes):
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda url, timeout=None: _FakeResponse(payload))

    return _serve


def _asset_pack(sha256: str | None) -> tuple[Pack, ModelItem]:
    """A one-file asset pack whose digest the test chooses."""
    item = ModelItem(item_id="thing", kind="asset",
                     url="https://example.invalid/thing.bin",
                     filename="thing.bin", sha256=sha256,
                     approx_bytes=len(PAYLOAD), license="PDDL-1.0")
    pack = Pack(pack_id="test-assets", title="t", description="d", pip=(),
                probe_modules=(), items=(item,), depends_on=(),
                install_mode="live")
    return pack, item


def _progress(events: list[dict]) -> list[dict]:
    return [event for event in events if event["type"] == "progress"]


def _logs(events: list[dict]) -> list[str]:
    return [event["line"] for event in events if event["type"] == "log"]


# -- the file list ---------------------------------------------------------


def test_list_hf_files_filters_duplicates_and_runtimes(fake_api):
    """Keep the model and its tokenizer; skip the other runtimes' copies of it.

    A sentence-transformers repo carries the same weights three or four times
    over -- safetensors, a PyTorch ``.bin``, an ONNX export, an OpenVINO one.
    Downloading all of them costs four times the bytes for one usable model.
    """
    calls = fake_api(
        ("config.json", 500),
        ("model.safetensors", 90_000_000),
        ("pytorch_model.bin", 90_000_000),
        ("tf_model.h5", 90_000_000),
        ("vocab.txt", 200_000),
        ("merges.txt", 400_000),
        ("tokenizer.json", 700_000),
        ("sentencepiece.bpe.model", 5_000_000),
        ("cl100k_base.tiktoken", 1_000),
        ("1_Pooling/config.json", 190),
        ("onnx/model.onnx", 90_000_000),
        # These two are dropped by the PREFIX rule alone: a config and a
        # tokenizer inside an export directory look exactly like the ones at
        # the top level, and are useless without the runtime they belong to.
        ("onnx/tokenizer.json", 700_000),
        ("openvino/openvino_model.bin", 90_000_000),
        ("openvino/config.json", 500),
        ("README.md", 9_000),
    )

    files = download.list_hf_files("some/repo", "main")

    assert [name for name, _ in files] == [
        "config.json", "model.safetensors", "vocab.txt", "merges.txt",
        "tokenizer.json", "sentencepiece.bpe.model", "cl100k_base.tiktoken",
        "1_Pooling/config.json",
    ]
    assert dict(files)["model.safetensors"] == 90_000_000
    assert calls == [{"repo_id": "some/repo", "revision": "main",
                      "files_metadata": True}]


def test_list_hf_files_keeps_the_bin_when_there_is_no_safetensors(fake_api):
    """The de-duplication drops a SECOND copy of the weights, never the only
    copy: a repo that never published safetensors still has to be usable.

    The OpenVINO weights are still dropped -- with no safetensors to make
    them redundant, only the prefix rule stands between the learner and a
    second 90 MB copy of a model they cannot run.
    """
    fake_api(("config.json", 500), ("pytorch_model.bin", 90_000_000),
             ("openvino/openvino_model.bin", 90_000_000))

    files = download.list_hf_files("some/repo", "main")

    assert [name for name, _ in files] == ["config.json", "pytorch_model.bin"]


def test_list_hf_files_reports_unknown_sizes_as_zero(fake_api):
    """``files_metadata`` is best-effort; a missing size must not be a crash."""
    fake_api(("config.json", None))

    assert download.list_hf_files("some/repo", "main") == [("config.json", 0)]


# -- the meter and the tqdm hook -------------------------------------------


def test_tqdm_hook_feeds_the_meter_and_throttles():
    """tqdm's ``update`` is the only place per-file bytes exist; the meter
    turns it into events, and the throttle keeps a download that moves a
    megabyte a second from emitting a thousand of them."""
    events: list[dict] = []
    meter = download._ByteMeter(emit=events.append, item_id="m", total=1000,
                                min_interval_s=1000.0)
    bar_class = download.make_tqdm_class(meter)

    bar = bar_class(total=1000, initial=0, desc="model.safetensors", unit="B",
                    unit_scale=True)
    assert bar.disable, "a progress bar was left writing to the console"
    for _ in range(5):
        bar.update(100)
    bar.close()

    assert meter.done == 500
    # The first update always goes out, then nothing until the interval does.
    assert len(_progress(events)) == 1
    assert _progress(events)[0] == {"type": "progress", "item": "m",
                                    "bytes_done": 100, "bytes_total": 1000,
                                    "percent": 10.0}

    meter.emit_now()
    assert _progress(events)[-1]["bytes_done"] == 500


def test_meter_percent_is_none_when_the_total_is_unknown():
    """A bar with no total reports bytes, not a made-up percentage."""
    events: list[dict] = []
    meter = download._ByteMeter(emit=events.append, item_id="m", total=None,
                                min_interval_s=0.0)

    meter.add(10)

    assert _progress(events)[-1] == {"type": "progress", "item": "m",
                                     "bytes_done": 10, "bytes_total": None,
                                     "percent": None}


def test_meter_raises_when_cancelled_mid_file():
    """Stop has to work during a single 400 MB file, not only between files."""
    meter = download._ByteMeter(emit=lambda event: None, item_id="m",
                                total=1000, cancel_check=lambda: True)

    with pytest.raises(PackCancelled):
        meter.add(1)


# -- the Xet path (two bars, one meter) ------------------------------------


def _xet_build_bars(tqdm_class, *, total: int, desc: str):
    """Build bars the way ``XetDownloadProgressReporter.__init__`` does.

    Mirrors huggingface_hub 1.29's ``_xet_progress_reporting.py``: when the
    class it was handed can ``update_transfer`` it keeps ONE bar and routes
    the network figure through it; when it cannot, it builds a second bar
    from the same class and updates both. Returns
    ``(reconstruction, transfer, aggregated)``.
    """
    aggregated = callable(getattr(tqdm_class, "update_transfer", None))
    reconstruction = tqdm_class(desc=desc, total=total, unit="B",
                                unit_scale=True, position=1, leave=True,
                                bar_format="{l_bar}{bar}| {n_fmt:>5}B")
    if aggregated:
        return reconstruction, reconstruction, True
    transfer = tqdm_class(desc="Downloading bytes", total=total, unit="B",
                          unit_scale=True, position=0, leave=True,
                          bar_format="{desc}: {bar}| {n_fmt:>5}B")
    return reconstruction, transfer, False


def _xet_update(bars, *, written: int, transferred: int) -> None:
    """One ``XetDownloadProgressReporter.update_progress`` call."""
    reconstruction, transfer, aggregated = bars
    if written > 0:
        reconstruction.update(written)
        reconstruction.set_postfix_str("1MB/s", refresh=False)
    if transferred > 0 and transfer is not None:
        if aggregated:
            reconstruction.update_transfer(transferred)
            reconstruction.set_transfer_postfix_str("1MB/s", refresh=False)
        else:
            transfer.update(transferred)
            transfer.set_postfix_str("1MB/s", refresh=False)


def test_tqdm_class_flips_hf_xet_to_one_aggregated_bar():
    """huggingface_hub decides how many bars to build by asking the class we
    hand it whether it can ``update_transfer``. Without that method it builds
    TWO bars from our class -- a reconstruction bar and a transfer bar -- and
    updates both, and since both feed one meter the download reports twice
    the bytes it moved and the bar runs to 200%.

    Answering yes is what keeps it to one bar, and the network figure it then
    routes through ``update_transfer`` must NOT be counted: those are the
    same bytes the reconstruction bar already reported.
    """
    events: list[dict] = []
    meter = download._ByteMeter(emit=events.append, item_id="m", total=1000,
                                min_interval_s=0.0)
    bar_class = download.make_tqdm_class(meter)

    assert callable(getattr(bar_class, "update_transfer", None))
    assert callable(getattr(bar_class, "set_transfer_postfix_str", None))

    bars = _xet_build_bars(bar_class, total=1000, desc="model.safetensors")
    assert bars[2] is True, "hf would still build a second bar from our class"

    _xet_update(bars, written=400, transferred=400)
    _xet_update(bars, written=600, transferred=600)

    assert meter.done == 1000
    assert _progress(events)[-1]["percent"] == 100.0


def test_download_hf_item_does_not_double_count_two_xet_bars(
        fake_api, monkeypatch):
    """Belt and braces: even if a future hub builds two bars from our class
    anyway, one file cannot report more than its own size."""
    import huggingface_hub

    monkeypatch.setattr(download, "PROGRESS_MIN_INTERVAL_S", 0.0)
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    fake_api(("config.json", 400), ("model.safetensors", 600))

    def _two_bars(*, repo_id, filename, cache_dir=None, tqdm_class=None,
                  **kwargs):
        target = _snapshot_file(cache_dir, repo_id, filename)
        target.write_bytes(b"z")
        size = {"config.json": 400, "model.safetensors": 600}[filename]
        # The pre-fix shape: two independent bars, both fed.
        reconstruction = tqdm_class(desc=filename, total=size, unit="B")
        transfer = tqdm_class(desc="Downloading bytes", total=size, unit="B")
        for bar in (reconstruction, transfer):
            bar.update(size // 2)
            bar.update(size - size // 2)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _two_bars)
    events: list[dict] = []

    download.download_hf_item(pack, item, emit=events.append,
                              cancel_check=lambda: False)

    frames = _progress(events)
    assert all(frame["percent"] <= 100.0 for frame in frames), frames
    assert all(frame["bytes_done"] <= 1000 for frame in frames), frames
    assert frames[-1]["bytes_done"] == frames[-1]["bytes_total"] == 1000


def test_meter_takes_back_a_rolled_back_delta():
    """huggingface_hub rolls a bar back by ``-resume_size`` when the server
    ignores a Range request and restarts the file. A meter that treated that
    as zero would keep the bytes twice."""
    events: list[dict] = []
    meter = download._ByteMeter(emit=events.append, item_id="m", total=1000,
                                min_interval_s=0.0)

    meter.add(300)
    meter.add(-300)
    assert meter.done == 0

    # And it never goes below zero, whatever it is told.
    meter.add(-500)
    assert meter.done == 0
    assert _progress(events)[-1]["percent"] == 0.0


def test_tqdm_initial_credits_a_resumed_transfer():
    """A resumed file starts its bar at ``initial=<bytes already on disk>``
    and only ``update``s the rest. Ignoring it under-reports the whole
    download by however much had already been fetched."""
    events: list[dict] = []
    meter = download._ByteMeter(emit=events.append, item_id="m", total=1000,
                                min_interval_s=0.0)
    meter.begin_file(0, 1000)
    bar_class = download.make_tqdm_class(meter)

    bar_class(total=1000, initial=300, desc="model.safetensors", unit="B")
    assert meter.done == 300

    # A second bar for the SAME file (hf's retry path) must not re-credit it.
    bar_class(total=1000, initial=300, desc="model.safetensors", unit="B")
    assert meter.done == 300


def test_meter_clamps_one_file_to_its_own_size():
    """The ceiling is per file: whatever a bar claims, a 400-byte file cannot
    contribute more than 400 bytes to the total."""
    meter = download._ByteMeter(emit=lambda event: None, item_id="m",
                                total=1000, min_interval_s=0.0)

    meter.begin_file(0, 400)
    meter.add(4000)
    assert meter.done == 400

    meter.begin_file(400, 600)
    meter.add(6000)
    assert meter.done == 1000


def test_meter_without_a_known_size_is_not_clamped():
    """An unknown file size means no ceiling, not a ceiling of zero."""
    meter = download._ByteMeter(emit=lambda event: None, item_id="m",
                                total=None, min_interval_s=0.0)

    meter.begin_file(0, 0)
    meter.add(4000)

    assert meter.done == 4000


# -- Hugging Face items ----------------------------------------------------


def test_download_hf_item_reports_monotonic_progress_and_writes_sentinel(
        fake_api, fake_hub_download):
    """The happy path: every file fetched, progress climbs to the total, and
    the sentinel afterwards is one ``state`` accepts as "downloaded"."""
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    fake_api(("config.json", 400), ("model.safetensors", 600))
    events: list[dict] = []

    snapshot = download.download_hf_item(pack, item, emit=events.append,
                                         cancel_check=lambda: False)

    assert snapshot.name == "rev0001"
    assert snapshot.is_dir()
    assert [call["filename"] for call in fake_hub_download] == [
        "config.json", "model.safetensors"]

    reported = [event["bytes_done"] for event in _progress(events)]
    assert reported == sorted(reported), "progress went backwards"
    assert reported[0] == 0, "no starting frame, so the bar appears late"
    assert reported[-1] == 1000
    assert _progress(events)[-1]["percent"] == 100.0

    assert state.item_state(pack, item).present
    assert state.item_state(pack, item).snapshot_dir == snapshot
    recorded = state.read_sentinel(sentinel_path(pack.pack_id, item.item_id))
    assert recorded["kind"] == "hf"
    assert recorded["repo_id"] == item.repo_id
    assert recorded["revision"] == item.revision
    assert recorded["snapshot_dir"] == str(snapshot)
    assert recorded["bytes"] == 1000
    assert recorded["at"]


def test_download_hf_item_counts_a_cached_file_as_it_passes(
        fake_api, monkeypatch):
    """A file already in the cache downloads nothing and so REPORTS nothing.

    Its bytes still have to be counted as the loop passes it, or a model
    whose first file is cached and whose second takes a minute shows a bar
    frozen at zero for that minute -- and then jumps to 100%.

    The throttle is turned off here so that every frame is visible; with it
    on, this whole download happens inside one 0.25 s window.
    """
    import huggingface_hub

    monkeypatch.setattr(download, "PROGRESS_MIN_INTERVAL_S", 0.0)
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    fake_api(("config.json", 400), ("model.safetensors", 600))

    def _cached(*, repo_id, filename, cache_dir=None, **kwargs):
        target = _snapshot_file(cache_dir, repo_id, filename)
        target.write_bytes(b"")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _cached)
    events: list[dict] = []

    download.download_hf_item(pack, item, emit=events.append,
                              cancel_check=lambda: False)

    reported = [event["bytes_done"] for event in _progress(events)]
    assert reported == sorted(reported)
    assert 400 in reported, "the cached first file never moved the bar"
    assert reported[-1] == 1000
    assert _progress(events)[-1]["percent"] == 100.0


def test_download_hf_item_uses_pack_cache_dir_not_hf_home(
        fake_api, fake_hub_download, monkeypatch):
    """The download lands in CodefyUI's cache and leaves ``HF_HOME`` alone:
    that variable is the whole machine's Hugging Face cache, shared with
    every other tool its owner runs."""
    monkeypatch.setenv("HF_HOME", "D:/somebody-elses-cache")
    pack = get_pack("rag")
    item = get_item(pack, "qwen2.5-0.5b-instruct")
    fake_api(("config.json", 400))

    snapshot = download.download_hf_item(pack, item, emit=lambda event: None,
                                         cancel_check=lambda: False)

    assert fake_hub_download[0]["cache_dir"] == str(hf_cache_dir())
    assert snapshot.is_relative_to(hf_cache_dir())
    assert "local_files_only" not in fake_hub_download[0]
    assert os.environ["HF_HOME"] == "D:/somebody-elses-cache"


def test_download_hf_item_finds_the_snapshot_root_from_a_nested_file(
        fake_api, fake_hub_download):
    """The recorded directory is the SNAPSHOT, not the subfolder the last
    file happened to live in -- ``1_Pooling/config.json`` is two levels down
    and a sentence-transformers model is unloadable from either half."""
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "bge-small-zh-v1.5")
    fake_api(("model.safetensors", 600), ("1_Pooling/config.json", 190))

    snapshot = download.download_hf_item(pack, item, emit=lambda event: None,
                                         cancel_check=lambda: False)

    assert snapshot.name == "rev0001"
    assert snapshot.parent.name == "snapshots"
    assert (snapshot / "1_Pooling" / "config.json").exists()


def test_download_hf_item_cancel_between_files(fake_api, monkeypatch):
    """Stop after the first of two files: nothing is recorded as downloaded."""
    import huggingface_hub

    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    fake_api(("config.json", 400), ("model.safetensors", 600))
    fetched: list[str] = []
    stop = {"now": False}

    def _download(*, repo_id, filename, cache_dir=None, **kwargs):
        target = _snapshot_file(cache_dir, repo_id, filename)
        target.write_bytes(b"z")
        fetched.append(filename)
        stop["now"] = True   # the file finished; the user pressed Stop
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)

    with pytest.raises(PackCancelled):
        download.download_hf_item(pack, item, emit=lambda event: None,
                                  cancel_check=lambda: stop["now"])

    assert fetched == ["config.json"]
    assert not state.item_state(pack, item).present


def test_download_hf_item_without_matching_files_is_an_error(fake_api):
    """A repo whose every file was filtered out has nothing to run, and must
    not leave a sentinel claiming otherwise."""
    pack = get_pack("rag")
    item = get_item(pack, "qwen2.5-0.5b-instruct")
    fake_api(("onnx/model.onnx", 400), ("README.md", 10))

    with pytest.raises(PackInstallError):
        download.download_hf_item(pack, item, emit=lambda event: None,
                                  cancel_check=lambda: False)

    assert not state.item_state(pack, item).present


# -- single-file assets ----------------------------------------------------


def test_download_asset_item_good_and_bad_sha256(served):
    """A recorded digest is checked: matching bytes land and are recorded,
    mismatched bytes leave nothing behind at all."""
    pack, item = _asset_pack(DIGEST)
    served(PAYLOAD)
    events: list[dict] = []

    path = download.download_asset_item(pack, item, emit=events.append,
                                        cancel_check=lambda: False)

    assert path.read_bytes() == PAYLOAD
    assert path == cache_dir() / "thing.bin"
    recorded = state.read_sentinel(sentinel_path("test-assets", "thing"))
    assert recorded["kind"] == "asset"
    assert recorded["url"] == item.url
    assert recorded["path"] == str(path)
    assert recorded["bytes"] == len(PAYLOAD)
    assert recorded["sha256"] == DIGEST
    assert _progress(events)[-1]["bytes_done"] == len(PAYLOAD)
    assert state.item_state(pack, item).present

    path.unlink()
    bad_pack, bad_item = _asset_pack("0" * 64)
    served(PAYLOAD)

    with pytest.raises(PackInstallError) as failure:
        download.download_asset_item(bad_pack, bad_item,
                                     emit=lambda event: None,
                                     cancel_check=lambda: False)

    assert "sha256" in str(failure.value)
    assert not path.exists()


def test_download_asset_logs_sha256_when_unrecorded(served):
    """The GloVe table's digest is not in the catalog yet. The install must
    print the one it computed -- that is how it gets recorded -- and must say
    that nothing was verified."""
    pack = get_pack("word-vectors")
    item = get_item(pack, "glove-50d")
    assert item.sha256 is None, "this test is about the unrecorded case"
    served(PAYLOAD)
    events: list[dict] = []

    path = download.download_asset_item(pack, item, emit=events.append,
                                        cancel_check=lambda: False)

    assert path == asset_dir() / item.filename
    logs = _logs(events)
    assert any(f"sha256 {DIGEST}" in line for line in logs), logs
    assert any("not verified" in line.lower() for line in logs), logs

    recorded = state.read_sentinel(sentinel_path("word-vectors", "glove-50d"))
    assert recorded["sha256"] == DIGEST


def test_download_asset_item_cancel_mid_download(served):
    """Stop during the download of a single large file."""
    pack, item = _asset_pack(DIGEST)
    served(PAYLOAD)

    with pytest.raises(PackCancelled):
        download.download_asset_item(pack, item, emit=lambda event: None,
                                     cancel_check=lambda: True)

    assert not (cache_dir() / "thing.bin").exists()
    assert not state.item_state(pack, item).present


# -- the disk precheck -----------------------------------------------------


def test_check_disk_raises(monkeypatch):
    """Refuse before downloading, and say by how much: a 1 GB model needs
    1.5 GB of room, because the cache holds the blob and the snapshot both."""
    item = get_item(get_pack("rag"), "qwen2.5-0.5b-instruct")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: _usage(600_000_000))

    with pytest.raises(PackInsufficientDisk) as failure:
        download.check_disk([item])

    assert failure.value.needed == 1_500_000_000
    assert failure.value.free == 600_000_000


def test_check_disk_passes_with_room(monkeypatch):
    item = get_item(get_pack("rag"), "qwen2.5-0.5b-instruct")
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: _usage(99_000_000_000))

    download.check_disk([item])


def test_check_disk_measures_a_directory_that_exists(monkeypatch):
    """The cache root may not exist yet, and ``disk_usage`` raises on a path
    that is not there -- which would read as "no space" for a machine with
    plenty."""
    item = get_item(get_pack("rag"), "qwen2.5-0.5b-instruct")
    seen: list[Path] = []

    def _disk_usage(path):
        seen.append(Path(path))
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return _usage(99_000_000_000)

    monkeypatch.setattr(shutil, "disk_usage", _disk_usage)
    shutil.rmtree(cache_dir(), ignore_errors=True)

    download.check_disk([item])

    assert seen, "the disk was never measured"
    assert all(path.exists() for path in seen)
