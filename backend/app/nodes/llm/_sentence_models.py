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
would hold over a gigabyte after the run had finished. Each loaded model
also has a lock of its own, which ``encode_in_batches`` holds for one batch
at a time: every node and every run in the process shares that object, its
token cap is a setting on it, and its tokenizer raises ``Already borrowed``
when one thread changes its truncation while another is encoding.

Imports here stay cheap on purpose: this module is reached at startup
through the node modules the registry scans, so ``sentence_transformers``
is imported inside the loader and ``app.core.packs`` inside the one branch
that needs its exception class.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Sequence

import numpy as np

from ...core.loop_control import EVENT_BATCH, ProgressThrottle
from . import _packs_bridge

if TYPE_CHECKING:  # the runtime import is inside ``_pack_missing``
    from ...core.packs import PackMissingError

logger = logging.getLogger(__name__)

#: The Package Center pack every model below comes from.
SENTENCE_PACK = "sentence-embeddings"

#: Hugging Face repo id -> the catalog item id it is downloaded under.
#: Insertion order is catalog order, which is the order the SELECT lists
#: them in.
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


class _ModelState(NamedTuple):
    """What ``encode_in_batches`` keeps beside one loaded model."""

    #: Held while one batch sets its cap and encodes.
    lock: threading.Lock
    #: The cap the model came with; a caller's 0 means this.
    shipped_max_seq_length: int


#: Keyed by the model object, weakly: an entry lives exactly as long as its
#: model, including a model evicted from ``_CACHE`` that a caller is still
#: encoding with -- so ``clear_model_cache`` leaves this alone, or that
#: caller's model would get a second lock. Not keyed by ``id(model)``,
#: which Python reuses once the object is collected.
_MODEL_STATE: weakref.WeakKeyDictionary[Any, _ModelState] = (
    weakref.WeakKeyDictionary())


def _model_state_locked(model: Any) -> _ModelState:
    """*model*'s lock and shipped cap, recorded the first time it is seen.

    The caller holds ``_CACHE_LOCK``, so two threads meeting the same new
    model agree on one lock for it.
    """
    state = _MODEL_STATE.get(model)
    if state is None:
        state = _ModelState(threading.Lock(), int(model.max_seq_length))
        _MODEL_STATE[model] = state
    return state


#: How often a caller waiting for another caller's batch on the same model
#: looks at Stop. ``threading.Lock`` is not FIFO, so a waiting caller can
#: sit through several of the other caller's batches, each seconds long,
#: before it gets the lock and reaches its own next Stop check.
_STOP_POLL_S = 0.1


def _acquire_or_stop(lock: threading.Lock,
                     should_stop: Callable[[], bool] | None) -> bool:
    """Take *lock*, unless Stop is asked for while waiting for it.

    True with the lock held; False, with it not held, when *should_stop*
    said yes first. With no *should_stop* nobody can ask, so this waits.
    """
    if should_stop is None:
        lock.acquire()
        return True
    while not lock.acquire(timeout=_STOP_POLL_S):
        if should_stop():
            return False
    return True


def option_packs_for_models() -> dict[str, str]:
    """The ``option_packs`` mapping for a SELECT over ``SENTENCE_MODELS``.

    Built through ``_packs_bridge.requirement`` so the ``pack:item``
    convention is written down once; the editor reads these values back to
    grey out an option and offer the download that would enable it.
    """
    return {repo_id: _packs_bridge.requirement(SENTENCE_PACK, item_id)
            for repo_id, item_id in SENTENCE_MODELS.items()}


def _pack_missing(message: str) -> PackMissingError:
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


def load_sentence_model(repo_id: str, device: str) -> Any:
    """The loaded encoder for *repo_id* on *device*, from the pack cache.

    Every caller in the process gets the same object for the same pair, so
    encode through ``encode_in_batches``, which holds the model's own lock
    around each batch, rather than calling ``model.encode`` directly.

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

    # Not a second opinion on the same question: ``require_pack(pack, item)``
    # already asked about THIS model, and in production it is what refuses a
    # model that was never downloaded. The two read different sources.
    # ``pack_available`` reads ``state.probe_all()``, which is memoised for
    # the whole process until something calls ``invalidate()``; ``model_dir``
    # re-checks the sentinel and the bytes right now. So this rung covers the
    # window where the cached probe still says "present" and the snapshot is
    # not -- a cache someone cleaned out by hand, an uninstall, a half-
    # finished download -- and turns it into the same actionable sentence
    # rather than letting ``SentenceTransformer`` fail on a missing directory.
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
            # Before the model is in the cache, so no caller can have
            # changed its cap yet: this is the value a cap of 0 goes back to.
            _model_state_locked(model)
            _CACHE[key] = model
            while len(_CACHE) > MAX_CACHED_MODELS:
                evicted, _ = _CACHE.popitem(last=False)
                logger.info("dropped %s on %s from the model cache "
                            "(only %d stay resident)",
                            evicted[0], evicted[1], MAX_CACHED_MODELS)
        else:
            _CACHE.move_to_end(key)

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
    max_seq_length: int = 0,
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

    *max_seq_length* is the token cap for every batch of this call, or 0
    for the cap the model shipped with -- never the cap the previous caller
    left on it. Each batch sets the cap and encodes under the model's own
    lock, so every batch runs at its own caller's cap, and callers on
    different models never wait for each other's batches. The lock is
    released at every batch boundary, where any caller waiting on the same
    model can take it (``threading.Lock`` is not FIFO, so this is not strict
    alternation); a caller still waiting looks at Stop every
    ``_STOP_POLL_S`` seconds. One copy of the model per cap was rejected for
    its memory: it would multiply what ``MAX_CACHED_MODELS`` exists to bound.
    """
    if isinstance(texts, str):
        # ``list("hello")`` is five one-character texts, and every layer
        # below here would accept that: five rows come back, the shapes are
        # right, nothing raises. The learner just gets nonsense embeddings.
        raise ValueError(
            "texts must be a sequence of strings, not a single string")

    items = list(texts)
    total = len(items)
    if total == 0:
        # Still 2-D, so a caller can stack or concatenate the result
        # without special-casing "nothing to embed". The width is unknown
        # because no forward pass ever ran.
        return np.zeros((0, 0), dtype=np.float32), None

    # Read without the cache lock first, because a load holds that lock for
    # seconds and this model is usually loaded already. The read is safe:
    # an entry is written once, under the lock, is never replaced, and
    # cannot be dropped while this caller holds the model -- so a hit is
    # this model's only lock. A miss is a model that did not come through
    # the loader (a unit test's fake, built directly); it is recorded here,
    # under the lock, on first use, at the cap it has now.
    state = _MODEL_STATE.get(model)
    if state is None:
        with _CACHE_LOCK:
            state = _model_state_locked(model)
    cap = (int(max_seq_length) if max_seq_length > 0
           else state.shipped_max_seq_length)

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

        # The model's lock, for this one batch. A caller waiting behind
        # another caller's batch keeps looking at Stop and, when asked,
        # stops as it would at a batch boundary. The progress frame and the
        # next Stop check run after the release. Never ``_CACHE_LOCK``
        # here: that would queue every load, and every other model, behind
        # this forward pass.
        if not _acquire_or_stop(state.lock, should_stop):
            stopped_at = index
            break
        try:
            model.max_seq_length = cap
            rows = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        finally:
            state.lock.release()
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
