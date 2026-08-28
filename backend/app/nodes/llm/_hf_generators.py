"""The one place a node turns a model name into a loaded causal LM.

The generator half of what ``._sentence_models`` does for encoders, and
deliberately the same shape -- one registry, one gate in one order, one
locked cache -- because the two are read side by side by anybody following
the RAG chain, and a second set of conventions here would be a second set
of things to learn for no gain.

**One registry.** ``GENERATOR_MODELS`` maps the repo id a learner picks in
the SELECT to the catalog item id the Package Center downloads under. The
two have to agree exactly: a node gating on an item id the catalog never
heard of would report a model as un-downloadable however many times it was
downloaded, so ``test_hf_generators`` compares this table against
``core.packs.catalog`` rather than trusting that both were typed correctly.

**One gate, in one order.** Unknown id, unknown dtype, then pack, then
snapshot, then the library. Each rung is a different thing gone wrong for
the learner, and each raises before the next is touched: the first two are
authoring bugs and must not be offered as something installable, and
nothing imports ``transformers`` until the pack has already said yes.

**A graph run never downloads pack contents.** ``local_files_only=True`` is
the mechanical half of the promise ``core.packs.require_pack`` makes in
words; the missing-snapshot branch below is the half that turns a would-be
1 GB mid-run download into a sentence naming the Package Center.

**One locked cache, holding exactly ONE model.** Nodes run on worker
threads (``MAX_PARALLEL_NODES = 4``), so two of them can ask for the same
generator at the same moment; without the lock they would both miss and
both load a gigabyte of weights. One entry rather than the encoder cache's
two because the sizes are not comparable -- these weights are ~1 GB against
~100-500 MB, and there is only one generator in the catalog anyway, so a
second resident entry could only ever be the same model on a second device
or in a second precision. Switching either of those is a deliberate act,
and paying the reload for it is better than holding two gigabytes after the
run has finished.

**The dtype cast happens AFTER the load, never as a keyword.** The keyword
that asks ``from_pretrained`` for a precision was renamed (``torch_dtype``
-> ``dtype``) between transformers 4 and 5, so passing either one pins this
code to a library version the pack installer is free to move. ``.to(dtype=)``
on the loaded module means the same thing in every version, at the cost of
briefly holding the fp32 weights.

Imports here stay cheap on purpose: this module is reached at startup
through the node modules the registry scans, so ``transformers`` is
imported inside the loader and ``app.core.packs`` inside the one branch
that needs its exception class.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from . import _packs_bridge

if TYPE_CHECKING:  # the runtime import is inside ``_pack_missing``
    from ...core.packs import PackMissingError

logger = logging.getLogger(__name__)

#: The Package Center pack every model below comes from.
RAG_PACK = "rag"

#: Hugging Face repo id -> the catalog item id it is downloaded under.
#: Insertion order is catalog order, which is the order the SELECT lists
#: them in. Only INSTRUCTION-TUNED checkpoints -- ones whose tokenizer
#: carries a ``chat_template`` -- belong in this table: the node addresses
#: every entry through ``apply_chat_template``, and a base model has no
#: template to apply.
GENERATOR_MODELS: dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5-0.5b-instruct",
}

#: The only one so far: 0.5B parameters is about a gigabyte on disk, runs on
#: a laptop CPU at a few tokens a second, is Apache-2.0, and answers in
#: Traditional Chinese as well as English -- which is the whole shortlist of
#: requirements for a generator this project can ship to a classroom.
DEFAULT_GENERATOR = "Qwen/Qwen2.5-0.5B-Instruct"

#: What the ``dtype`` SELECT offers besides ``auto``. Kept as a table rather
#: than ``getattr(torch, name)`` so a hand-edited graph cannot reach
#: ``torch.qint8`` -- or ``torch.save`` -- through a param.
DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

#: How many loaded generators stay resident. See the module docstring.
MAX_CACHED_GENERATORS = 1

_CACHE_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple[str, str, str], tuple[Any, Any]] = OrderedDict()


def option_packs_for_generators() -> dict[str, str]:
    """The ``option_packs`` mapping for a SELECT over ``GENERATOR_MODELS``.

    Built through ``_packs_bridge.requirement`` so the ``pack:item``
    convention is written down once; the editor reads these values back to
    grey out an option and offer the download that would enable it.
    """
    return {repo_id: _packs_bridge.requirement(RAG_PACK, item_id)
            for repo_id, item_id in GENERATOR_MODELS.items()}


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

    return PackMissingError(RAG_PACK, message)


def _cuda_has_bfloat16() -> bool:
    """Whether this CUDA device can do bfloat16, treating unknown as no.

    Same guard and the same reasoning as ``core.amp.supported_precisions``:
    the capability query touches the driver, and a box whose driver cannot
    answer should fall back to float16 rather than fail the run over a
    precision nobody asked for by name.
    """
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001 - an unreadable capability is an absent one
        logger.debug("could not query bfloat16 support", exc_info=True)
        return False


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """The precision to hold the weights in, for *dtype* on *device*.

    ``auto`` is the only interesting case, and it is two decisions:

    * On CUDA, half precision -- bfloat16 where the card has it, float16
      otherwise. It halves both the memory and the time, and a 0.5B model
      generating a paragraph is not where numerical range is going to be
      the problem.
    * On CPU and MPS, float32. Half precision on a CPU is not a speedup:
      most of the kernels fall back to converting each tensor up to fp32
      and back down, so it costs time to save memory this model does not
      need. MPS is excluded for a different reason -- its float16 support
      is complete enough to run and incomplete enough to produce garbage
      on some ops, and a learner cannot tell that from a bad model.

    An explicit name maps straight through, so a user who wants to SEE
    float16 be slower on their CPU can.
    """
    name = (dtype or "auto").strip().lower() or "auto"
    if name != "auto":
        resolved = DTYPES.get(name)
        if resolved is None:
            raise ValueError(
                f"Unknown dtype {dtype!r}; choose one of: auto, "
                + ", ".join(DTYPES))
        return resolved
    # ``cuda:1`` is still cuda. Splitting rather than comparing, the same
    # way ``core.amp`` reads a device string.
    if (device or "cpu").split(":", 1)[0] != "cuda":
        return torch.float32
    return torch.bfloat16 if _cuda_has_bfloat16() else torch.float16


def load_causal_lm(repo_id: str, device: str, dtype: str) -> tuple[Any, Any]:
    """The ``(tokenizer, model)`` pair for *repo_id*, from the pack cache.

    The model comes back on *device*, cast to :func:`resolve_dtype`'s answer
    and in ``eval()`` mode -- there is no training path through this node, so
    unlike ``TextGenerate`` (which is handed a module the graph owns and must
    put back the way it found it) nothing here has a training mode to restore.

    Raises ``ValueError`` for an id that is not in ``GENERATOR_MODELS`` or a
    ``dtype`` name that is not in ``DTYPES``, and ``PackMissingError``
    (message ending in ``(pack=rag)``) when the pack, the snapshot or the
    library is not there.
    """
    item_id = GENERATOR_MODELS.get(repo_id)
    if item_id is None:
        # Not a PackMissingError: nothing a learner can install fixes a
        # model id that is not in the catalog, and offering them a download
        # button for it would be a dead end.
        raise ValueError(
            f"Unknown generator model {repo_id!r}; choose one of: "
            + ", ".join(GENERATOR_MODELS))

    # The second authoring bug, and it is checked in the same place as the
    # first: a dtype name nothing recognises is a hand-edited graph, and no
    # download fixes it. Before the pack rung so a learner is not sent to
    # the Package Center to install a gigabyte that will not help.
    resolved = resolve_dtype(dtype, device)

    # Through the bridge, never through ``core.packs`` directly: the bridge
    # is the seam node tests patch, and it is what keeps this import lazy.
    _packs_bridge.require_pack(RAG_PACK, item_id)

    # Not a second opinion on the same question -- see ``_sentence_models``
    # for the full reasoning. ``pack_available`` reads a probe memoised for
    # the whole process; ``model_dir`` re-checks the sentinel and the bytes
    # now. This rung covers the window between them: a cache cleaned out by
    # hand, an uninstall, a half-finished download.
    path = _packs_bridge.model_dir(repo_id)
    if path is None:
        raise _pack_missing(
            f"Model {repo_id} is not downloaded. Open Package Center > RAG "
            "stack and download it; graph runs never download pack models")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        # The sentinel says installed and site-packages disagrees. Named
        # precisely, because the rag pack does not ship transformers itself:
        # it depends on sentence-embeddings for it, so "reinstall the rag
        # pack" would be the wrong instruction.
        raise _pack_missing(
            "The RAG stack pack reports installed but 'transformers' cannot "
            f"be imported ({exc}); the rag pack depends on the "
            "sentence-embeddings pack for transformers; reinstall it from "
            "Package Center") from exc

    # The RESOLVED precision, not the param: ``auto`` and ``bfloat16`` name
    # the same weights on an Ampere card, and keying on the param would load
    # the model twice for two nodes that asked for the same thing.
    key = (repo_id, device, str(resolved).rsplit(".", 1)[-1])
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            # Loading holds the lock. It is seconds of disk I/O, and one
            # thread waiting for another's load is strictly better than both
            # of them reading the same gigabyte of weights.
            started = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(path), local_files_only=True)
            model = model.to(device=device, dtype=resolved).eval()
            entry = (tokenizer, model)
            logger.info("loaded %s on %s as %s in %.1fs",
                        repo_id, device, key[2], time.monotonic() - started)
            _CACHE[key] = entry
            while len(_CACHE) > MAX_CACHED_GENERATORS:
                evicted, _ = _CACHE.popitem(last=False)
                logger.info("dropped %s (%s, %s) from the generator cache "
                            "(only %d stays resident)",
                            evicted[0], evicted[1], evicted[2],
                            MAX_CACHED_GENERATORS)
        else:
            _CACHE.move_to_end(key)

    return entry


def stop_ids(tokenizer: Any, model: Any) -> set[int]:
    """Every token id that means "the answer is finished".

    Two sources, and an instruction-tuned model needs both. The TOKENIZER's
    ``eos_token_id`` is the base model's end-of-text; the chat template ends
    a turn with a different token (``<|im_end|>`` for Qwen), and the only
    place that is written down is the model's ``generation_config``, where
    it may be a single id or a list of them. Reading just one of the two
    ends a run at ``max_new_tokens`` with the model talking to itself in
    both voices.

    None-safe throughout: a tokenizer with no eos, a model with no
    generation config and a config whose ``eos_token_id`` is null are all
    ordinary things to meet, and none of them is worth an exception. An
    empty set simply means the loop stops on its token budget.
    """
    found: set[int] = set()

    def collect(raw: Any) -> None:
        if raw is None or isinstance(raw, bool):
            # ``bool`` is an ``int`` subclass, and ``eos_token_id = True``
            # would silently become "stop on token 1".
            return
        if isinstance(raw, int):
            found.add(int(raw))
            return
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                collect(item)

    collect(getattr(tokenizer, "eos_token_id", None))
    collect(getattr(getattr(model, "generation_config", None),
                    "eos_token_id", None))
    return found


def clear_generator_cache() -> None:
    """Forget the loaded generator. Idempotent, and safe to call at any time.

    Tests lean on this (a model loaded from a ``tmp_path`` snapshot must not
    be handed to the next test after that directory is gone). Dropping the
    references is all it does: the weights are freed by the ordinary
    refcount once the last caller lets go of them.
    """
    with _CACHE_LOCK:
        _CACHE.clear()
