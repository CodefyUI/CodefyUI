"""The one place a node turns a model name into a loaded sentence encoder.

``WordVector`` (on its real backends) and ``TextEmbedding`` want the same
four things, and each of them is a decision rather than a convenience:

**One registry.** ``SENTENCE_MODELS`` maps the repo id a learner picks in a
SELECT to the catalog item id the Package Center downloads under. The two
have to agree exactly -- a node that gated on an item id the catalog never
heard of would report a model as un-downloadable however many times it was
downloaded -- so ``test_sentence_models`` compares this table against
``core.packs.catalog`` rather than trusting that both were typed correctly.

**One gate, in one order.** Unknown id, then pack, then snapshot, then the
library. Each rung is a different thing gone wrong for the learner, and
each raises before the next is touched: an id that is not in the table is
an authoring bug and must not be reported as something installable, and
nothing imports ``sentence_transformers`` (which drags in torch) until the
pack has already said yes.

**A graph run never downloads pack contents.** ``local_files_only=True`` is
the mechanical half of the promise ``core.packs.require_pack`` makes in
words; the missing-snapshot branch below is the half that turns a would-be
470 MB mid-run download into a sentence naming the Package Center.

**One small, locked cache.** Nodes run on worker threads
(``MAX_PARALLEL_NODES = 4``), so two of them can ask for the same model at
the same moment; without the lock they would both miss and both load it.
Bounded to two because that is what a learner actually does -- compare an
English model against a multilingual one -- while four resident models
would hold over a gigabyte after the run had finished.

Imports here stay cheap on purpose: this module is reached at startup
through the node modules the registry scans, so ``sentence_transformers``
is imported inside the loader and ``app.core.packs`` inside the one branch
that needs its exception class.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Sequence

import numpy as np

from ...core.loop_control import EVENT_BATCH, ProgressThrottle
from . import _packs_bridge

logger = logging.getLogger(__name__)

#: The Package Center pack every model below comes from.
SENTENCE_PACK = "sentence-embeddings"

#: Hugging Face repo id -> the catalog item id it is downloaded under.
#: Insertion order is the order the SELECT lists them in, cheapest first.
SENTENCE_MODELS: dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        "paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-zh-v1.5": "bge-small-zh-v1.5",
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
}

#: The multilingual one, because this project is used in Traditional
#: Chinese as well as English and it needs no prompt prefixes to work.
DEFAULT_SENTENCE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#: How many loaded models stay resident. See the module docstring.
MAX_CACHED_MODELS = 2

_CACHE_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()


def option_packs_for_models() -> dict[str, str]:
    """The ``option_packs`` mapping for a SELECT over ``SENTENCE_MODELS``.

    Built through ``_packs_bridge.requirement`` so the ``pack:item``
    convention is written down once; the editor reads these values back to
    grey out an option and offer the download that would enable it.
    """
    return {repo_id: _packs_bridge.requirement(SENTENCE_PACK, item_id)
            for repo_id, item_id in SENTENCE_MODELS.items()}


def _pack_missing(message: str) -> Exception:
    """A ``PackMissingError`` for this pack, imported where it is used.

    Lazy for the same reason ``_packs_bridge`` is lazy: a node module must
    still IMPORT in an install with no packs package, or the registry scan
    that builds the palette takes the whole palette down with it. Safe to
    import here specifically, because every caller below has already been
    past ``require_pack`` -- which either imported the packs package or
    raised ``PacksUnavailableError`` instead of returning.
    """
    from ...core.packs import PackMissingError

    return PackMissingError(SENTENCE_PACK, message)


def load_sentence_model(repo_id: str, device: str, *,
                        max_seq_length: int = 0) -> Any:
    """The loaded encoder for *repo_id* on *device*, from the pack cache.

    *max_seq_length* is the caller's token cap, or 0 for "whatever the
    model was trained with". It is applied on EVERY call, hit or miss: the
    cap belongs to the node that asked, the model object belongs to the
    process, and two nodes sharing one cached model with different caps
    would otherwise silently inherit whichever loaded first.

    Raises ``ValueError`` for an id that is not in ``SENTENCE_MODELS``, and
    ``PackMissingError`` (message ending in ``(pack=sentence-embeddings)``)
    when the pack, the snapshot or the library is not there.
    """
    item_id = SENTENCE_MODELS.get(repo_id)
    if item_id is None:
        # Not a PackMissingError: nothing a learner can install fixes a
        # model id that is not in the catalog, and offering them a download
        # button for it would be a dead end.
        raise ValueError(
            f"Unknown embedding model {repo_id!r}; choose one of: "
            + ", ".join(SENTENCE_MODELS))

    # Through the bridge, never through ``core.packs`` directly: the bridge
    # is the seam node tests patch, and it is what keeps this import lazy.
    _packs_bridge.require_pack(SENTENCE_PACK, item_id)

    # ``require_pack`` passing is not enough. The four models are
    # ALTERNATIVES, so the pack counts as usable once any one of them is
    # downloaded -- which says nothing about the one this call wants.
    path = _packs_bridge.model_dir(repo_id)
    if path is None:
        raise _pack_missing(
            f"Model {repo_id} is not downloaded. Open Package Center > "
            "Sentence embeddings and download it; graph runs never download "
            "pack models")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        # The sentinel says installed and site-packages disagrees: a broken
        # install, not a missing download. Said as a sentence, because the
        # learner sees this in a node error and not in a server log.
        raise _pack_missing(
            "The Sentence embeddings pack reports installed but "
            f"'sentence_transformers' cannot be imported ({exc}); reinstall "
            "the pack from Package Center") from exc

    key = (repo_id, device)
    with _CACHE_LOCK:
        model = _CACHE.get(key)
        if model is None:
            # Loading holds the lock. It is seconds of disk I/O, and one
            # thread waiting for another's load is strictly better than
            # both of them loading the same half-gigabyte of weights.
            started = time.monotonic()
            model = SentenceTransformer(
                str(path),
                device=device,
                local_files_only=True,
                trust_remote_code=False,
            )
            logger.info("loaded %s on %s in %.1fs",
                        repo_id, device, time.monotonic() - started)
            _CACHE[key] = model
            while len(_CACHE) > MAX_CACHED_MODELS:
                evicted, _ = _CACHE.popitem(last=False)
                logger.info("dropped %s on %s from the model cache "
                            "(only %d stay resident)",
                            evicted[0], evicted[1], MAX_CACHED_MODELS)
        else:
            _CACHE.move_to_end(key)

        if max_seq_length > 0:
            # Mutable on the real class too. Last writer wins, which is all
            # a shared object can offer: two nodes encoding concurrently
            # with different caps are racing, and the alternative -- one
            # model per cap -- would multiply the memory this cache exists
            # to bound.
            model.max_seq_length = int(max_seq_length)

    return model


def clear_model_cache() -> None:
    """Forget every loaded model. Idempotent, and safe to call at any time.

    Tests lean on this (a model loaded from a ``tmp_path`` snapshot must not
    be handed to the next test after that directory is gone). Dropping the
    references is all it does: whatever the encoder was holding is freed by
    the ordinary refcount when the last caller lets go of it.
    """
    with _CACHE_LOCK:
        _CACHE.clear()


def encode_in_batches(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    normalize: bool = False,
    prefix: str = "",
    progress: ProgressThrottle | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, int | None]:
    """Embed *texts*, in order, reporting progress and honouring Stop.

    Returns ``(rows, stopped_at)`` -- a ``[N, D]`` float32 array and either
    None or the index of the batch the loop refused to start. A stopped run
    returns the rows it already has rather than raising: the node merges
    ``interrupted_result`` around them and the learner keeps the partial
    output, which is the whole point of a cooperative stop (see
    ``core.loop_control``).

    *prefix* is prepended to every text when non-empty (``query: `` /
    ``passage: `` for the e5 models). *progress* may be None -- the export
    runner and most unit tests have no callback to throttle.
    """
    items = list(texts)
    total = len(items)
    if total == 0:
        # Still 2-D, so a caller can stack or concatenate the result
        # without special-casing "nothing to embed". The width is unknown
        # because no forward pass ever ran.
        return np.zeros((0, 0), dtype=np.float32), None

    size = max(1, int(batch_size))
    batches = (total + size - 1) // size
    chunks: list[np.ndarray] = []
    stopped_at: int | None = None

    for index in range(batches):
        # At the TOP of the batch: a stop between two batches costs the
        # user nothing, one checked halfway through would throw away a
        # forward pass that had already been paid for.
        if should_stop is not None and should_stop():
            stopped_at = index
            break

        start = index * size
        batch = items[start:start + size]
        if prefix:
            batch = [prefix + text for text in batch]

        rows = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        chunks.append(np.asarray(rows, dtype=np.float32))

        if progress is not None:
            done = start + len(batch)
            progress.emit({
                "event": EVENT_BATCH,
                "batch": index + 1,
                "total_batches": batches,
                "text": f"Embedding {done}/{total}",
            })

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32), stopped_at
    return np.concatenate(chunks, axis=0), stopped_at
