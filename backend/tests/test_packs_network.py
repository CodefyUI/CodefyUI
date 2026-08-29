"""The real downloads, run by hand.

Every other pack test fakes the network, because CI has none. These four do
not fake anything: they fetch the actual GloVe table off a GitHub release,
the actual MiniLM snapshot off the Hugging Face hub, cancel an actual
in-flight transfer, and ask an actual index whether ``sentence-transformers``
could be installed into this interpreter. That is the only way to find out
whether the code works against the servers it will meet, and it is the reason
the whole module is OPT-IN::

    CODEFYUI_PACK_NETWORK_TESTS=1 pytest tests/test_packs_network.py -q

Without that variable every test here skips, so ``pytest`` on an offline
runner -- or on a laptop on a train -- stays green and fast.

Three properties this module guarantees about the machine it runs on:

* **nothing lands in a real cache.** ``CODEFYUI_USER_DATA_DIR`` points at
  ``tmp_path``, which is where every path helper reads its root from, and the
  fixture asserts that redirection took effect before a single byte moves.
  ``HF_XET_CACHE`` is redirected for the same reason: it is the one directory
  huggingface_hub writes to that CodefyUI's own cache root does not cover.
* **nothing is installed.** The pip test runs ``uv pip install --dry-run``
  and then re-checks that ``sentence_transformers`` is exactly as importable
  as it was before, so a maintainer running this suite cannot accidentally
  mutate the venv they are testing.
* **~250 MB is transferred.** The GloVe table, the MiniLM snapshot twice
  (once on its own, once to prove a cancelled download did not poison the
  ones after it) and a cancelled fragment. About 25 seconds on a fast
  connection; slow ones should expect minutes.

``test_real_glove_download_verifies_sha256_and_reports_progress`` doubles as
the tool that RECORDS the GloVe digest: on a catalog entry whose ``sha256``
is still None it fails with the digest it computed in the message, and that
value goes into ``catalog.py``. Afterwards the same test proves the recorded
value verifies.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import time

import pytest

from app.core import asset_cache, packs
from app.core.packs import constraints, download, runner, state
from app.core.packs.catalog import get_item, get_pack
from app.core.packs.errors import PackCancelled
from app.core.packs.paths import hf_cache_dir, sentinel_path

pytestmark = pytest.mark.skipif(
    os.environ.get("CODEFYUI_PACK_NETWORK_TESTS") != "1",
    reason="set CODEFYUI_PACK_NETWORK_TESTS=1 to run the real download tests")

#: How far into the multilingual model's weights the cancel test lets the
#: transfer get before pressing Stop. Every OTHER file in that repo together
#: is well under a megabyte, so a progress event reporting this many bytes
#: can only have come from inside the 470 MB ``model.safetensors`` -- which
#: is the whole point: cancelling BETWEEN files is already covered offline
#: (``test_download_hf_item_cancel_between_files``), and the case that needs
#: a real server is the one where Stop is pressed inside a single file, on
#: the hf_xet transport, which discards what the progress hook raises.
CANCEL_AFTER_BYTES = 4 * 1024 * 1024

#: The largest file the cancelled download may leave in the cache. The
#: weights are 470 MB, so anything under this is proof the transfer was
#: ABORTED rather than allowed to run to the end -- which is what the
#: measurement is really about: before ``download.abort_xet_transfer``
#: existed, this same test found the full 470 641 600-byte blob sitting in
#: the cache, because raising from the hook stopped nothing at all.
INCOMPLETE_UNDER_BYTES = 100_000_000


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """A throwaway cache root, asserted to be the one the code will use.

    The assertion is not ceremony. ``asset_cache.cache_dir`` and
    ``config._user_data_root`` read the environment on every call TODAY; the
    day one of them grows an ``lru_cache`` this fixture would silently start
    pointing at the developer's real cache, and a test that downloads 90 MB
    into somebody's home directory should fail loudly rather than pass.
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    # huggingface_hub's chunk cache. Not read by CodefyUI at all -- redirected
    # so this suite leaves nothing behind anywhere, not to change a code path.
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "xet"))
    state.invalidate()

    assert asset_cache.cache_dir() == tmp_path / "cache"
    assert hf_cache_dir() == tmp_path / "cache" / "hf"

    yield tmp_path
    state.invalidate()


def _progress(events: list[dict]) -> list[dict]:
    return [event for event in events if event["type"] == "progress"]


def _logs(events: list[dict]) -> list[str]:
    return [event["line"] for event in events if event["type"] == "log"]


def test_real_glove_download_verifies_sha256_and_reports_progress(
        isolated_cache):
    """66 MB off a GitHub release: it lands, the bar reaches the end, and the
    digest in the catalog is the digest of the bytes that arrived.

    The first time this runs the catalog has no digest, so the assertion that
    one is recorded fails and names the value to record -- that is how the
    hash in ``catalog.py`` was obtained, and re-running is what proves it.
    """
    pack = get_pack("word-vectors")
    item = get_item(pack, "glove-50d")
    events: list[dict] = []

    started = time.monotonic()
    path = download.download_asset_item(pack, item, emit=events.append,
                                        cancel_check=lambda: False)
    elapsed = time.monotonic() - started

    assert path == asset_cache.cache_dir() / item.filename
    assert path.is_file()
    size = path.stat().st_size
    print(f"\nglove-50d: {size} bytes in {elapsed:.1f}s -> {path}")
    # The catalog calls it 66 MB; anything an order of magnitude off is a
    # redirect page or a truncated transfer wearing the right filename.
    assert 50_000_000 < size < 90_000_000, size

    progress = _progress(events)
    assert progress, events
    assert [event["bytes_done"] for event in progress] == sorted(
        event["bytes_done"] for event in progress)
    assert progress[-1]["percent"] == 100
    assert progress[-1]["bytes_done"] == size
    assert progress[-1]["bytes_total"] == size

    recorded = state.read_sentinel(sentinel_path("word-vectors", "glove-50d"))
    assert recorded is not None
    assert recorded["kind"] == "asset"
    assert recorded["url"] == item.url
    assert recorded["path"] == str(path)
    assert recorded["bytes"] == size
    assert state.item_state(pack, item).present
    assert packs.asset_path("word-vectors", item.filename) == path

    digest = asset_cache.sha256_of(path)
    assert recorded["sha256"] == digest
    print(f"glove-50d sha256: {digest}")
    assert item.sha256 is not None, (
        f"catalog.py records no sha256 for {item.item_id}. This download "
        f"computed {digest} ({size} bytes) -- write that into the item and "
        f"run this test again to prove it verifies.")
    assert digest == item.sha256, (
        f"the bytes that arrived hash to {digest}, but catalog.py says "
        f"{item.sha256}")
    assert not any("not verified" in line.lower() for line in _logs(events)), (
        "a recorded digest was verified, so nothing may claim otherwise")

    # Second call, cached: ``asset_cache.resolve`` re-hashes what is on disk
    # and compares it to the catalog. Returning the same path is that
    # comparison passing -- a mismatch would have deleted the file and gone
    # back to the network.
    again_events: list[dict] = []
    again = download.download_asset_item(pack, item, emit=again_events.append,
                                         cancel_check=lambda: False)
    assert again == path
    assert path.stat().st_size == size
    assert not any("not verified" in line.lower()
                   for line in _logs(again_events))


def test_real_minilm_item_downloads_and_loads_offline(isolated_cache):
    """90 MB off the hub: every file a sentence-transformers model needs to
    load without a network, and nothing else.

    ``sentence_transformers`` is deliberately NOT imported -- it is not in
    this venv and never in CI -- so the claim is checked the way the
    library's own loader checks it: ``modules.json`` names the pipeline,
    ``config.json`` and the weights are the transformer, ``1_Pooling/`` is
    the pooling step.
    """
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    events: list[dict] = []

    started = time.monotonic()
    snapshot = download.download_hf_item(pack, item, emit=events.append,
                                         cancel_check=lambda: False)
    elapsed = time.monotonic() - started

    assert snapshot.is_dir()
    assert snapshot.parent.name == "snapshots"
    # Not in ~/.cache/huggingface, not under HF_HOME: in the pack cache.
    assert snapshot.is_relative_to(hf_cache_dir())

    weights = sorted(snapshot.rglob("*.safetensors"))
    files = sorted(p.relative_to(snapshot).as_posix()
                   for p in snapshot.rglob("*") if p.is_file())
    on_disk = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
    print(f"\nall-MiniLM-L6-v2: {on_disk} bytes in {elapsed:.1f}s -> {files}")

    assert (snapshot / "config.json").is_file(), files
    assert weights, files
    assert weights[0].stat().st_size > 10_000_000
    assert (snapshot / "modules.json").is_file(), files
    assert (snapshot / "1_Pooling" / "config.json").is_file(), files
    assert (snapshot / "tokenizer_config.json").is_file(), files
    # The exports ``list_hf_files`` drops are each a second copy of the
    # weights; downloading them would double the 90 MB for nothing.
    assert not [name for name in files
                if name.startswith(("onnx/", "openvino/"))], files
    assert not [name for name in files if name.endswith(".bin")], files

    progress = _progress(events)
    assert progress[-1]["percent"] == 100
    assert progress[-1]["bytes_done"] == progress[-1]["bytes_total"]
    assert progress[-1]["bytes_done"] > 10_000_000

    recorded = state.read_sentinel(
        sentinel_path("sentence-embeddings", "all-MiniLM-L6-v2"))
    assert recorded is not None
    assert recorded["kind"] == "hf"
    assert recorded["repo_id"] == item.repo_id
    assert recorded["revision"] == item.revision
    assert recorded["snapshot_dir"] == str(snapshot)
    assert state.item_state(pack, item).present

    # What a node asks for. This is the whole point of the download.
    assert packs.model_dir(item.repo_id) == snapshot
    assert packs.pack_available("sentence-embeddings", "all-MiniLM-L6-v2") is (
        state.pip_ready(pack))


def test_real_hf_cancel_surfaces_as_pack_cancelled(isolated_cache):
    """Stop, pressed in the middle of a real 470 MB transfer, stops the
    transfer and arrives as ``PackCancelled``.

    Two claims, and the first is the one that needed a real server to find.
    hf_xet drives the progress hook from its Rust download group and
    DISCARDS whatever is raised inside it, so raising ``PackCancelled``
    there stopped nothing: the file ran to completion and the cancel was
    noticed afterwards, which for this model is 470 MB of "please stop"
    being ignored. ``download.abort_xet_transfer`` is the fix; the on-disk
    assertion below is what proves it, because the progress numbers looked
    perfectly reasonable either way.

    The second claim is the type. Aborting the transport makes
    ``hf_hub_download`` raise the Rust runtime's own ``RuntimeError``, and
    ``flows`` would report that to the user as an install FAILURE -- "the
    install broke" for an install they stopped themselves.
    """
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "paraphrase-multilingual-MiniLM-L12-v2")
    events: list[dict] = []
    stop = {"now": False}

    def _emit(event: dict) -> None:
        events.append(event)
        if (event["type"] == "progress"
                and event["bytes_done"] >= CANCEL_AFTER_BYTES):
            stop["now"] = True

    started = time.monotonic()
    with pytest.raises(PackCancelled) as cancelled:
        download.download_hf_item(pack, item, emit=_emit,
                                  cancel_check=lambda: stop["now"])
    elapsed = time.monotonic() - started

    assert type(cancelled.value) is PackCancelled, cancelled.value
    assert stop["now"], "the transfer ended before the cancel was requested"

    progress = _progress(events)
    stopped_at = progress[-1]["bytes_done"]
    biggest = max((p.stat().st_size for p in hf_cache_dir().rglob("*")
                   if p.is_file()), default=0)
    print(f"\ncancelled at {stopped_at} of {progress[-1]['bytes_total']} "
          f"bytes after {elapsed:.1f}s; largest file left: {biggest}")

    assert stopped_at >= CANCEL_AFTER_BYTES, stopped_at
    assert stopped_at < progress[-1]["bytes_total"]
    # The transfer really stopped: the weights are 470 MB and nothing that
    # size is on the disk.
    assert biggest < INCOMPLETE_UNDER_BYTES, biggest

    # A cancelled download is not a download: no sentinel, so nothing later
    # reports this model as present.
    assert not sentinel_path(pack.pack_id, item.item_id).exists()
    assert not state.item_state(pack, item).present
    assert packs.model_dir(item.repo_id) is None

    # And Stop does not poison the process it was pressed in. The abort ends
    # the whole hf_xet SESSION, which is shared by everything in this
    # interpreter -- so the download after a cancelled one has to work, or
    # "Stop, then Install again" would be broken until the server restarted.
    # 90 MB to check, and worth it: nothing smaller goes down the same path.
    after = get_item(pack, "all-MiniLM-L6-v2")
    snapshot = download.download_hf_item(pack, after, emit=lambda event: None,
                                         cancel_check=lambda: False)
    assert (snapshot / "config.json").is_file()
    assert state.item_state(pack, after).present


@pytest.mark.skipif(runner.find_uv() is None, reason="uv is not on PATH")
def test_real_live_pip_install_dry_run(isolated_cache, tmp_path):
    """Would ``sentence-transformers`` actually install into THIS venv?

    A live-mode pack install is only possible if uv can resolve the pack's
    specs while every distribution already here stays exactly where it is
    (``constraints.py``). That is a question about a real index and a real
    interpreter, so it is asked for real -- with ``--dry-run``, which resolves
    and reports without writing anything.

    Either answer is a pass, because either answer is information: a zero exit
    means a maintainer can offer the in-app install on this machine, and a
    non-zero one has to NAME the package that blocked it, so the answer is
    actionable rather than "it did not work".
    """
    pack = get_pack("sentence-embeddings")
    assert pack.pip, "this test is about the pack that has pip specs"

    constraints_path = constraints.write_constraints_file(tmp_path)
    pinned = [line.split("==", 1)[0]
              for line in constraints_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    assert "torch" in pinned, "the constraints file should describe this venv"

    importlib.invalidate_caches()
    installed_before = importlib.util.find_spec("sentence_transformers") is not None

    argv = runner.pip_install_argv(pack.pip, constraints_path=constraints_path,
                                   dry_run=True)
    assert "--dry-run" in argv

    completed = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=runner.pip_env(),
        creationflags=runner.creation_flags(), timeout=900, check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    lines = output.splitlines()
    print(f"\nuv exit {completed.returncode} for {list(pack.pip)}\n"
          + "\n".join(lines[-runner.TAIL_LINES:]))

    importlib.invalidate_caches()
    assert (importlib.util.find_spec("sentence_transformers") is not None) is (
        installed_before), "--dry-run must not change the interpreter"

    assert lines, "uv said nothing at all, which is not an answer"
    if completed.returncode == 0:
        assert any(word in output.lower()
                   for word in ("resolved", "would install", "audited")), output
        return

    # A refusal has to be a resolver refusal that names a package, not a
    # network blip -- otherwise the maintainer learns nothing from it.
    assert runner.looks_like_resolver_conflict(lines), output
    assert [name for name in pinned if name and name in output.lower()], output
