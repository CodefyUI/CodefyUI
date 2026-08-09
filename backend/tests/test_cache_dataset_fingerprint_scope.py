"""Regression tests for #259 - Dataset's fingerprint covers its own files.

``DatasetNode.cache_fingerprint`` used to walk the whole of ``data_dir``
recursively. That is safe (over-invalidation costs a re-read, never a stale
answer) and wasteful: in project mode every relative ``data_dir`` collapses
to one shared ``PROJECT_DIR/assets/data``, so an unrelated dataset -- or a
model file -- changing under it busts this node's key, and a large tree
pays a full recursive stat every run whether or not this dataset moved.
That is exactly what #144 restored the caching to avoid.

The reason it could not simply be done is the interesting half, and it is
what most of this file is about. The over-invalidation had a second,
undocumented job: ``ModelSaver`` may only write under ``MODELS_DIR``,
which in the default layout sits INSIDE the tree this fingerprint walked,
so every saver write dirtied the dataset's key and the resulting miss
propagated all the way down to ``TrainingLoop``. ``TrainingLoop`` was
cacheable at the time (#253), so that accident was the only thing standing
between the shipped saver graphs and a second Run that did no training at
all. #259 was blocked on #253 for precisely that reason.

#253 is fixed at the source now -- ``TrainingLoop`` and ``SequentialModel``
are both non-cacheable -- so the accident is no longer load-bearing. That
claim is not taken on trust here: the end-to-end tests below run three
SHIPPED graphs three times each against one ``ExecutionCache`` and count
real ``TrainingLoopNode.execute()`` calls. Status is never the assertion;
a preset whose internals were all cache hits still reports ``completed``
(#260), which is why #253 went unnoticed for so long.

The graphs are used as they ship. Two params are rewritten -- the dataset's
``data_dir``, to point at a fabricated offline MNIST instead of a real
download, and the epoch/batch counts, to keep the suite fast -- and each
rewrite is asserted, so a graph that stops containing the node it rewrites
fails loudly rather than quietly testing something else.
"""

from __future__ import annotations

import functools
import json
import struct
from pathlib import Path

import pytest

from app.config import settings
from app.core.cache import ExecutionCache
from app.core.graph_engine import execute_graph, expand_presets
from app.nodes.data.dataset_node import DATASET_NAMES, DatasetNode

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The three shipped graphs that train and read MNIST. The first two have
#: no writer at all -- they are the ones #253 actually reproduced on. The
#: third has a ``ModelSaver``, which is what gave it the accidental
#: protection this change removes, so it is the one that proves the removal
#: is safe.
SHIPPED_GRAPHS = {
    "C2-5/MLP-MNIST-Training": (
        _REPO_ROOT / "plugins" / "foundations" / "examples" / "C2-5"
        / "MLP-MNIST-Training" / "graph.json", False),
    "C3-1/LeNet-MNIST-Training": (
        _REPO_ROOT / "plugins" / "deep" / "examples" / "C3-1"
        / "LeNet-MNIST-Training" / "graph.json", False),
    "CNN-MNIST/TrainCNN-MNIST": (
        _REPO_ROOT / "examples" / "Usage_Example" / "CNN-MNIST"
        / "TrainCNN-MNIST" / "graph.json", True),
}


def write_offline_mnist(root: Path, n_train: int = 64, n_test: int = 16) -> None:
    """Real IDX files torchvision's MNIST loader reads, with no network.

    ``MNIST._check_exists`` only checks that the four decompressed files
    are present (``check_integrity`` with no md5 is an ``isfile``), so a
    hand-written IDX header plus bytes is enough to make ``download=True``
    a no-op and the loader read genuine tensors. Files on real disk are the
    point: this file is about what the fingerprint sees.
    """
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for prefix, count in (("train", n_train), ("t10k", n_test)):
        pixels = bytes((i * 7 + 3) % 256 for i in range(count * 28 * 28))
        (raw / f"{prefix}-images-idx3-ubyte").write_bytes(
            struct.pack(">IIII", 2051, count, 28, 28) + pixels)
        (raw / f"{prefix}-labels-idx1-ubyte").write_bytes(
            struct.pack(">II", 2049, count) + bytes(i % 10 for i in range(count)))


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """The default layout, where ``MODELS_DIR`` is INSIDE the dataset tree.

    That nesting is the whole coupling: ``ModelSaver`` refuses to write
    outside ``MODELS_DIR.parent``, and in a stock install that parent is
    the same ``./data`` the Dataset node is pointed at.
    """
    root = tmp_path / "data"
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", root / "models")
    write_offline_mnist(root)
    return root


def _mnist_params(root: Path, split: str = "train") -> dict:
    return {"name": "MNIST", "split": split, "data_dir": str(root)}


# ─────────────────────────────────────────────────────────────────────────
# 1. The scope itself, checked against torchvision rather than a copy of it
# ─────────────────────────────────────────────────────────────────────────

def test_the_scope_matches_the_layout_torchvision_itself_uses():
    """Derive the expected directory a second way, from torchvision.

    A table of directory names copied into ``dataset_node.py`` would pass a
    test that reads the same table. These expectations come from
    torchvision's own class attributes and properties instead, so a rename
    upstream fails here rather than turning the fingerprint into something
    that silently covers nothing.
    """
    from torchvision import datasets

    root = Path("/tmp/whatever")
    for name in DATASET_NAMES:
        dataset_cls = getattr(datasets, name)
        owned = DatasetNode._owned_paths(name, "train", str(root))
        assert owned, f"{name} has no scope, so it would walk the whole tree"

        if issubclass(dataset_cls, datasets.MNIST):
            # ``raw_folder``/``processed_folder`` are instance properties
            # reading ``self.root``; an uninitialised instance is enough to
            # ask torchvision where it would put the files.
            probe = object.__new__(dataset_cls)
            probe.root = str(root)
            expected = Path(probe.raw_folder)
            assert len(owned) == 1 and expected.is_relative_to(owned[0]), (
                f"{name}: scope {owned[0]} does not contain torchvision's "
                f"own raw folder {expected}")
        elif getattr(dataset_cls, "base_folder", None):
            assert owned == [root / dataset_cls.base_folder], (
                f"{name}: scope {owned} is not root/{dataset_cls.base_folder}")
        else:
            expected = root / dataset_cls.split_list["train"][1]
            assert owned == [expected], f"{name}: scope {owned} != {expected}"


def test_an_unknown_dataset_name_falls_back_to_the_whole_tree():
    """Degrading to the old behaviour is the safe direction.

    A name this node cannot place has to be over-invalidated, not
    under-invalidated: a needless re-read costs seconds, a stale dataset
    costs a wrong experiment.
    """
    assert DatasetNode._owned_paths("NotADataset", "train", "/tmp/x") is None


def test_an_unknown_name_still_produces_a_change_sensitive_fingerprint(tmp_path):
    stray = tmp_path / "anything"
    stray.mkdir()
    (stray / "f").write_bytes(b"x")
    params = {"name": "NotADataset", "split": "train", "data_dir": str(tmp_path)}

    before = DatasetNode.cache_fingerprint(params)
    (stray / "f").write_bytes(b"xxxx")
    assert before != DatasetNode.cache_fingerprint(params)


# ─────────────────────────────────────────────────────────────────────────
# 2. What the scope buys, and what it must not lose
# ─────────────────────────────────────────────────────────────────────────

def test_a_model_saver_write_no_longer_busts_the_dataset(data_root):
    """The headline of #259, and the coupling it was blocked on.

    Before scoping, this write changed the dataset's fingerprint (measured:
    ``count`` 4 -> 5, a different ``identity_hash``) and the resulting miss
    propagated down the graph. Nothing about the dataset changed.
    """
    params = _mnist_params(data_root)
    before = DatasetNode.cache_fingerprint(params)
    (settings.MODELS_DIR / "model_weights.pt").write_bytes(b"x" * 4096)
    after = DatasetNode.cache_fingerprint(params)

    assert before == after, (
        "writing a model file under data_dir must not invalidate the "
        f"dataset's cache key.\n  before: {before}\n  after:  {after}")


def test_an_unrelated_dataset_alongside_no_longer_busts_it(data_root):
    """Two datasets in one shared ``data_dir`` stop interfering.

    The project-mode case from the issue: every relative ``data_dir``
    collapses to one directory, so downloading CIFAR-10 used to re-read
    MNIST.
    """
    params = _mnist_params(data_root)
    before = DatasetNode.cache_fingerprint(params)
    cifar = data_root / "cifar-10-batches-py"
    cifar.mkdir()
    (cifar / "data_batch_1").write_bytes(b"y" * 8192)

    assert DatasetNode.cache_fingerprint(params) == before


def test_the_dataset_s_own_files_still_bust_it(data_root):
    """Scoping must narrow the walk, not disable it."""
    params = _mnist_params(data_root)
    before = DatasetNode.cache_fingerprint(params)
    write_offline_mnist(data_root, n_train=96)

    assert DatasetNode.cache_fingerprint(params) != before, (
        "re-downloading or editing the dataset's own files must still "
        "change its cache key, or #144's staleness bug is back")


def test_a_dataset_that_is_not_there_yet_fingerprints_differently(tmp_path):
    """The first run downloads; the key has to change when it lands."""
    params = _mnist_params(tmp_path)
    missing = DatasetNode.cache_fingerprint(params)
    write_offline_mnist(tmp_path)

    assert missing != DatasetNode.cache_fingerprint(params)


def test_svhn_is_scoped_per_split_because_it_downloads_per_split():
    """The one family where split-level scoping is the honest granularity.

    MNIST and CIFAR put both splits in one directory and torchvision wants
    all of it present before it reads either, so those are scoped per
    dataset. SVHN fetches one flat ``.mat`` per split, so its two splits
    really are independent files.
    """
    train = DatasetNode._owned_paths("SVHN", "train", "/tmp/x")
    test = DatasetNode._owned_paths("SVHN", "test", "/tmp/x")
    assert train != test and train and test


# ─────────────────────────────────────────────────────────────────────────
# 3. End to end on the shipped graphs -- the check that actually matters
# ─────────────────────────────────────────────────────────────────────────

def _shipped_graph(key: str, data_dir: Path) -> tuple[list[dict], list[dict]]:
    path, _ = SHIPPED_GRAPHS[key]
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes, edges, _ = expand_presets(payload["nodes"], payload["edges"])

    rewritten = {"Dataset": 0, "TrainingLoop": 0}
    for node in nodes:
        if node["type"] == "Dataset":
            node["data"]["params"]["data_dir"] = str(data_dir)
            rewritten["Dataset"] += 1
        elif node["type"] == "TrainingLoop":
            node["data"]["params"].update(epochs=1, device="cpu")
            rewritten["TrainingLoop"] += 1
        elif node["type"] == "DataLoader":
            node["data"]["params"].update(batch_size=16, num_workers=0)

    # The shipped file IS the fixture. If it stops containing the nodes
    # rewritten above, the rewrite became a no-op and everything below
    # would still pass while measuring a different graph.
    assert rewritten == {"Dataset": 1, "TrainingLoop": 1}, (
        f"{path} no longer has exactly one Dataset and one TrainingLoop "
        f"after preset expansion: {rewritten}")
    return nodes, edges


@pytest.mark.asyncio
@pytest.mark.parametrize("key", list(SHIPPED_GRAPHS))
async def test_a_shipped_graph_trains_on_all_three_runs(key, data_root):
    """Three Runs of a shipped example, one shared cache, real call counts.

    The ``ExecutionCache`` is per WebSocket and deliberately lent to every
    run that socket starts, so this is exactly "open the editor and press
    Run three times". Before #253 the two saver-less graphs measured
    1 / 0 / 0 here. Scoping the fingerprint takes away the accident that
    was protecting the third one, so all three have to stand on #253's fix
    alone -- which is what this asserts.
    """
    from app.nodes.data.dataset_node import DatasetNode as LiveDatasetNode
    from app.nodes.training.training_loop_node import TrainingLoopNode

    nodes, edges = _shipped_graph(key, data_root)
    _, has_saver = SHIPPED_GRAPHS[key]
    dataset_id = next(n["id"] for n in nodes if n["type"] == "Dataset")

    counts = {"train": 0, "dataset": 0}
    real_train = TrainingLoopNode.execute
    real_dataset = LiveDatasetNode.execute

    # functools.wraps is required, not cosmetic: the engine decides whether
    # to pass ``context``/``progress_callback`` by inspecting execute's
    # signature, and a bare *args wrapper would silently change how the
    # node under measurement runs.
    @functools.wraps(real_train)
    def counting_train(self, *args, **kwargs):
        counts["train"] += 1
        return real_train(self, *args, **kwargs)

    @functools.wraps(real_dataset)
    def counting_dataset(self, *args, **kwargs):
        counts["dataset"] += 1
        return real_dataset(self, *args, **kwargs)

    cache = ExecutionCache()
    per_run: list[int] = []
    dataset_statuses: list[str] = []

    TrainingLoopNode.execute = counting_train
    LiveDatasetNode.execute = counting_dataset
    try:
        for _ in range(3):
            before = counts["train"]
            statuses: dict[str, str] = {}

            async def track(node_id, status, data, _s=statuses):
                if status in ("completed", "cached", "skipped", "error"):
                    _s[node_id] = status

            await execute_graph(nodes, edges, on_progress=track, cache=cache)
            per_run.append(counts["train"] - before)
            dataset_statuses.append(statuses.get(dataset_id, "?"))
    finally:
        TrainingLoopNode.execute = real_train
        LiveDatasetNode.execute = real_dataset

    assert per_run == [1, 1, 1], (
        f"{key}: TrainingLoop executed {per_run} time(s) per run. A 0 is a "
        "Run that did no training and reported success.")

    assert dataset_statuses == ["completed", "cached", "cached"], (
        f"{key}: the dataset's statuses were {dataset_statuses}. Runs 2 and "
        "3 must be cache HITS -- that is the optimisation #259 asks for, and "
        "without it this test cannot tell 'training survives a cached "
        "dataset' from 'the dataset kept invalidating and carried the loop "
        "along with it', which is the accident being removed.")

    assert counts["dataset"] == 1, (
        f"{key}: the dataset was read {counts['dataset']} time(s) across "
        "three runs; scoping the fingerprint should have made it exactly 1.")

    if has_saver:
        saved = settings.MODELS_DIR / "model_weights.pt"
        assert saved.exists(), (
            f"{key} is the graph with a ModelSaver, and the saver is the "
            "whole reason it is in this list. Nothing was written to "
            f"{saved}, so this run did not exercise the coupling at all.")
