"""LMTokenizedDataset node (#290) -- text rows into packed next-token blocks.

The node that turns a corpus into something ``TrainingLoop`` can train on. It
does the one transformation every LM training script does, and does it where a
learner can see the arithmetic:

1. **Tokenize and concatenate.** Every document's ids are appended to ONE flat
   stream, with ``tokenizer.eos_id`` after each so the model learns where
   documents end. No padding anywhere -- padding wastes compute on tokens that
   teach nothing, and a packed stream has none.
2. **Cut fixed-length blocks.** Block ``i`` is ``stream[i*seq_len : i*seq_len +
   seq_len + 1]``: ``seq_len + 1`` tokens, of which the first ``seq_len`` are
   the inputs and the last ``seq_len`` are the labels. That one-token overlap IS
   next-token prediction -- ``labels[t]`` is the token that followed
   ``inputs[t]``.
3. **Drop the remainder.** ``num_blocks = (len(stream) - 1) // seq_len``. The
   tail that cannot fill a block is discarded rather than padded, which is why
   every sample is exactly ``seq_len`` long -- ``DataLoader`` collates with
   torch's ``default_collate`` and never gets a ``collate_fn`` from the
   DataLoader node, so a ragged sample would fail to batch.

Each ``__getitem__`` returns ``(input_ids int64[seq_len], labels
int64[seq_len])`` -- the 2-tuple ``TrainingLoop`` unpacks, so the LM chain needs
no special case anywhere in the training stack.

**The disk cache.** Tokenizing a real corpus takes minutes and produces the
same stream every time, so the packed tensor is written under
``<data>/cache/lm_blocks/<key>.pt`` and reloaded on the next run. The key is a
sha256 over the corpus fingerprint, the tokenizer's identity, ``seq_len``,
``append_eos`` and ``max_tokens`` -- every input that changes the stream. This
is the node's OWN cache and has nothing to do with ``ExecutionCache``: see
``cacheable`` below for why that one cannot help here.

Nothing evicts it (#306). A distinct key means a distinct file, so sweeping
``seq_len`` over three values leaves three full copies of the packed stream --
8 bytes per token each, which is ~800 MB per entry for a 100M-token corpus.
The eviction is a person typing ``cdui cache prune``; an automatic size cap at
write time was considered and deferred, because a cap that deletes the entry
the next run was about to hit turns a documented cost into a mysterious one.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch.utils.data import Dataset

from ...core.loop_control import (
    EVENT_BATCH,
    ProgressThrottle,
    interrupted_result,
    stop_checker,
)
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

logger = logging.getLogger(__name__)

#: Where the packed streams live, relative to the data root.
CACHE_SUBDIR = ("cache", "lm_blocks")

#: At or below this many rows the corpus fingerprint hashes EVERY row; above it,
#: only the length plus the first/middle/last row (see
#: :func:`_corpus_fingerprint`). 10k rows of prose is a few megabytes to hash --
#: milliseconds, against the minutes of tokenization a hit skips -- so the whole
#: teaching-scale range gets the exact fingerprint and only corpora big enough
#: for the hash to cost real time fall back to the sample.
FULL_HASH_MAX_ROWS = 10_000

#: The members an object on the ``tokenizer`` port must have. Checked all at
#: once, before any work happens, rather than one ``hasattr`` per use site, so
#: the error names everything that is missing in one message.
TOKENIZER_MEMBERS = ("encode", "decode", "eos_id", "vocab_size")


class PackedTokenBlocks(Dataset):
    """Fixed-length next-token training blocks over one flat token stream.

    Holds the whole stream as a single 1-D int64 tensor and slices it per
    ``__getitem__``. The slices are VIEWS, so a block costs no copy and no
    allocation -- the copy happens once, in ``default_collate``, when a batch is
    stacked.

    Module scope, not a closure inside the node, so ``torch.save`` and a
    DataLoader worker process can name the class (#283).
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        # ``- 1`` because the very last token can only ever be a LABEL: a block
        # needs seq_len inputs AND the token after the last one.
        self.num_blocks = max(0, (int(tokens.numel()) - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(idx)
        if index < 0:
            index += self.num_blocks
        if not 0 <= index < self.num_blocks:
            # IndexError, not ValueError: it is what a sampler and a bare
            # ``for block in dataset`` both expect at the end.
            raise IndexError(
                f"block {idx} is out of range for {self.num_blocks} blocks")
        start = index * self.seq_len
        window = self.tokens[start : start + self.seq_len + 1]
        return window[:-1], window[1:]


def _dataset_length(dataset: Any) -> int:
    """``len(dataset)``, or a ValueError naming what to wire instead.

    A map-style dataset is a hard requirement, not a convenience: the corpus
    fingerprint indexes rows to build the cache key, so an iterable-only
    dataset could be tokenized but never cached, and it could only be consumed
    once. Refusing is clearer than half-working.
    """
    try:
        return int(len(dataset))
    except Exception as exc:  # noqa: BLE001 - any failure means "no length"
        raise ValueError(
            "LMTokenizedDataset needs a dataset with a length (a map-style "
            "dataset of text rows). Wire TextCorpusDataset into the `dataset` "
            "input; a streaming/iterable dataset cannot be packed."
        ) from exc


def _check_tokenizer(tokenizer: Any) -> None:
    """Enforce the duck-typed tokenizer contract, naming the fix."""
    if tokenizer is None:
        raise ValueError(
            "LMTokenizedDataset has no tokenizer. Wire an LMTokenizer node "
            "into the `tokenizer` input.")
    missing = [name for name in TOKENIZER_MEMBERS if not hasattr(tokenizer, name)]
    if not missing and not callable(getattr(tokenizer, "encode")):
        missing = ["encode"]
    if missing:
        raise ValueError(
            f"LMTokenizedDataset: the object on the `tokenizer` input is "
            f"missing {missing} -- a tokenizer here must provide "
            f"encode(text) -> list[int], decode(ids) -> str, eos_id and "
            f"vocab_size. Wire an LMTokenizer node into that input.")


def _corpus_fingerprint(dataset: Any, total_rows: int) -> dict[str, Any]:
    """What the cache key knows about the corpus itself.

    Exact for anything up to :data:`FULL_HASH_MAX_ROWS` rows: every row is
    hashed, so an edited corpus always gets a new key. Above that, the
    fingerprint is the row COUNT plus the first, middle and last row -- which
    can miss an edit in the middle of a large corpus. That is a real hole and
    the reason ``cache`` is a parameter: a user who edits a multi-million-row
    corpus in place turns caching off (or changes ``cache_dir``) for one run.
    The alternative -- hashing gigabytes of text on every run -- would cost
    close to what the tokenization pass this cache exists to skip costs.

    Rows are separated by a NUL byte in the hash so ["ab", "c"] and ["a", "bc"]
    fingerprint differently; without it the concatenation is identical.
    """
    digest = hashlib.sha256()
    if total_rows <= FULL_HASH_MAX_ROWS:
        indices = range(total_rows)
        sampled = False
    else:
        indices = (0, total_rows // 2, total_rows - 1)
        sampled = True
    for index in indices:
        digest.update(str(dataset[index]).encode("utf-8", "replace"))
        digest.update(b"\x00")
    return {"rows": total_rows, "sampled": sampled, "hash": digest.hexdigest()}


def _tokenizer_identity(tokenizer: Any) -> str:
    """A string that changes when the token ids would.

    ``LMTokenizer`` publishes ``name`` (the tiktoken encoding), which is
    exactly right. Anything else falls back to its class name and vocabulary
    size -- enough to keep two different tokenizer classes apart, and NOT
    enough to notice a hand-written tokenizer that changes its mapping while
    keeping both. Turn ``cache`` off while iterating on a custom tokenizer.
    """
    name = getattr(tokenizer, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"{type(tokenizer).__name__}:{int(tokenizer.vocab_size)}"


def _cache_key(
    corpus: dict[str, Any],
    tokenizer_identity: str,
    seq_len: int,
    append_eos: bool,
    max_tokens: int,
) -> str:
    """sha256 over everything that decides the packed stream.

    ``sort_keys`` so the digest does not depend on dict insertion order, which
    would silently invalidate every cached file the day a key is reordered.
    """
    payload = json.dumps(
        {
            "corpus": corpus,
            "tokenizer": tokenizer_identity,
            "seq_len": seq_len,
            "append_eos": append_eos,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_dir(params: dict[str, Any]) -> Path:
    """Where the ``.pt`` files go: ``<data>/cache/lm_blocks``, or a subdirectory.

    A ``cache_dir`` value runs through ``resolve_data_path``, the one rule
    (#224) for every node that turns a parameter into a filesystem write: a
    relative path nests under the default, and the result must stay inside the
    data directory and must not name CodefyUI's own storage.
    """
    from ...core.data_paths import data_root, resolve_data_path

    base = data_root().joinpath(*CACHE_SUBDIR)
    raw = str(params.get("cache_dir", "") or "").strip()
    if not raw:
        return base
    return resolve_data_path(raw, base=base)


def _load_cached(path: Path) -> torch.Tensor | None:
    """The cached stream, or None when there is not a usable one.

    ``weights_only=True`` because this file is data, not code: a ``.pt`` that
    someone dropped into the cache directory must not be able to execute
    anything on load. Anything unreadable, truncated or of the wrong shape is
    treated as a MISS and rebuilt -- a corrupt cache file (a killed process
    mid-write on a filesystem where the atomic rename did not hold, a partial
    copy) must cost a re-tokenization, never a failed run.
    """
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, weights_only=True)
        tokens = payload["tokens"] if isinstance(payload, dict) else payload
        if (isinstance(tokens, torch.Tensor) and tokens.dim() == 1
                and tokens.dtype == torch.int64):
            return tokens
        logger.warning(
            "LMTokenizedDataset ignoring cache file %s: not a 1-D int64 "
            "tensor", path)
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning(
            "LMTokenizedDataset could not read cache file %s; re-tokenizing",
            path, exc_info=True)
    return None


def _save_cached(path: Path, tokens: torch.Tensor) -> None:
    """Write the stream where the next run will find it. Never raises.

    Staged through a sibling temp file and ``os.replace``d into place, the same
    way ``core.checkpoints`` writes: a reader must see either the previous file
    or the whole new one, never a half-written stream that would load as a
    truncated corpus. A cache write that fails is worth a warning and nothing
    more -- the node has already produced its real output.
    """
    staged = path.with_name(f"{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokens": tokens}, str(staged))
        os.replace(staged, path)
    except Exception:  # noqa: BLE001 - see the docstring
        staged.unlink(missing_ok=True)
        logger.warning(
            "LMTokenizedDataset could not write its block cache to %s; the "
            "dataset is fine, the next run just has to tokenize again",
            path, exc_info=True)
    except BaseException:
        # A KeyboardInterrupt or a cancellation is NOT a failed cache write and
        # must keep propagating; the staged file still has to go, which is the
        # only reason this clause exists.
        staged.unlink(missing_ok=True)
        raise


def _positive_int(params: dict[str, Any], name: str, default: int) -> int:
    """A size param, floored at 1 -- never clamped to the declared range.

    Same reasoning as ``CausalLMModel._positive_int``: ``min_value`` on the
    :class:`ParamDefinition` is the editor's guard rail for a corpus somebody
    intends to train on, and enforcing it here would silently rewrite the
    deliberately tiny configurations that lessons and this file's tests are
    made of (``seq_len=4`` quietly becoming 16).
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LMTokenizedDataset: {name} must be a whole number, got {raw!r}."
        ) from exc
    return max(1, value)


class LMTokenizedDatasetNode(BaseNode):
    NODE_NAME = "LMTokenizedDataset"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Turn text rows into fixed-length training blocks: tokenize every "
        "document, join them with the end-of-text token, and cut the stream "
        "into (input_ids, labels) pairs where the labels are the inputs "
        "shifted one token to the left -- next-token prediction. Wire the "
        "output into DataLoader. The packed tokens are cached on disk, so only "
        "the first run pays for tokenization -- nothing evicts that cache, and "
        "`cdui cache prune` is what clears it."
    )

    # The inputs are LIVE handles -- a dataset object and a tokenizer object --
    # whose contents no cache key can describe: the params say seq_len, not
    # what the corpus contains, so an ExecutionCache hit would serve run 1's
    # blocks after the user pointed TextCorpusDataset at a different file. The
    # node therefore owns a disk cache with its own key, which CAN look at the
    # corpus (see _corpus_fingerprint) -- and writing that file is itself a
    # side effect the return value does not carry, reason 3 in
    # ``BaseNode.cacheable``.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description=(
                    "Rows of raw text, one string per document -- the output "
                    "of TextCorpusDataset."
                ),
            ),
            PortDefinition(
                name="tokenizer",
                data_type=DataType.ANY,
                description=(
                    "Tokenizer object providing encode(text) -> list[int], "
                    "decode(ids) -> str, eos_id and vocab_size. Wire an "
                    "LMTokenizer node here."
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description=(
                    "Training blocks: each sample is (input_ids int64"
                    "[seq_len], labels int64[seq_len]) with labels shifted one "
                    "token left. Wire it into DataLoader."
                ),
            ),
            PortDefinition(
                name="total_tokens",
                data_type=DataType.SCALAR,
                description=(
                    "Tokens in the packed stream, end-of-text tokens included. "
                    "This is the number people mean by 'trained on N tokens'."
                ),
            ),
            PortDefinition(
                name="num_blocks",
                data_type=DataType.SCALAR,
                description=(
                    "How many training blocks the stream yielded: "
                    "(total_tokens - 1) // seq_len, remainder dropped."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="seq_len",
                param_type=ParamType.INT,
                default=1024,
                min_value=16,
                max_value=8192,
                description=(
                    "Tokens per training block -- the context length the model "
                    "learns over. Must not exceed the model's max_seq_len. "
                    "Longer costs memory quadratically in attention."
                ),
            ),
            ParamDefinition(
                name="append_eos",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Put the end-of-text token after every document. Leave it "
                    "on: without it the model learns that one story flows "
                    "straight into the next, and generation never stops."
                ),
            ),
            ParamDefinition(
                name="max_tokens",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Stop after this many tokens (0 = the whole corpus). The "
                    "quickest way to make one epoch finish in a lesson."
                ),
            ),
            ParamDefinition(
                name="cache",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Save the packed tokens to disk and reuse them when the "
                    "corpus and settings are unchanged. Turn it off while "
                    "editing a corpus in place."
                ),
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Subdirectory for the cached token files (empty = the "
                    "shared cache under the data directory). One file per "
                    "distinct corpus, tokenizer, seq_len, append_eos and "
                    "max_tokens, at 8 bytes per token -- a 100M-token corpus "
                    "is about 800 MB per file, and nothing deletes them. "
                    "`cdui cache list` shows what is there, `cdui cache "
                    "prune` clears it."
                ),
                visible_when={"cache": True},
                advanced=True,
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        dataset = inputs.get("dataset")
        tokenizer = inputs.get("tokenizer")
        _check_tokenizer(tokenizer)
        total_rows = _dataset_length(dataset)

        seq_len = _positive_int(params, "seq_len", 1024)
        append_eos = bool(params.get("append_eos", True))
        # ``or 0`` is safe for both of these because 0 IS the disabled value.
        max_tokens = max(0, int(params.get("max_tokens", 0) or 0))
        use_cache = bool(params.get("cache", True))

        cache_path: Path | None = None
        if use_cache:
            cache_path = _cache_dir(params) / "{}.pt".format(_cache_key(
                _corpus_fingerprint(dataset, total_rows),
                _tokenizer_identity(tokenizer),
                seq_len, append_eos, max_tokens))
            cached = _load_cached(cache_path)
            if cached is not None:
                logger.info(
                    "LMTokenizedDataset reused %d cached tokens from %s",
                    cached.numel(), cache_path)
                return self._result(cached, seq_len, cached=True)

        tokens, rows_done, stopped = self._tokenize(
            dataset, tokenizer, total_rows,
            append_eos=append_eos, max_tokens=max_tokens,
            progress_callback=progress_callback, context=context)

        if stopped:
            # A stopped node returns what it has and says so -- never raises
            # (``core.loop_control``). And nothing is cached: the key describes
            # the WHOLE corpus, so a partial stream stored under it would serve
            # a truncated dataset to every later run of the same graph.
            result = self._result(tokens, seq_len, cached=False)
            result.update(interrupted_result(rows=rows_done))
            return result

        if (tokens.numel() - 1) // seq_len < 1:
            raise ValueError(
                f"LMTokenizedDataset: the packed stream is {tokens.numel()} "
                f"tokens, and one training block needs seq_len + 1 = "
                f"{seq_len + 1}. Lower seq_len, or point TextCorpusDataset at "
                f"more text (check max_rows and max_tokens are not capping it).")

        if cache_path is not None:
            _save_cached(cache_path, tokens)
        return self._result(tokens, seq_len, cached=False)

    @staticmethod
    def _tokenize(
        dataset: Any,
        tokenizer: Any,
        total_rows: int,
        *,
        append_eos: bool,
        max_tokens: int,
        progress_callback: Any | None,
        context: Any,
    ) -> tuple[torch.Tensor, int, bool]:
        """Encode every row into one flat stream. Returns (tokens, rows, stopped).

        The stream accumulates in an ``array('q')`` rather than a Python list:
        a list of 100M ids costs ~4 GB of boxed integers where the array costs
        the 800 MB the ids themselves are, and ``torch.frombuffer`` then adopts
        that buffer with no copy at all.
        """
        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        eos_id = int(tokenizer.eos_id)

        stream = array.array("q")
        rows_done = 0
        stopped = False
        for index in range(total_rows):
            if should_stop():
                stopped = True
                break
            text = dataset[index]
            stream.extend(tokenizer.encode(
                text if isinstance(text, str) else str(text)))
            if append_eos:
                stream.append(eos_id)
            rows_done += 1
            full = bool(max_tokens) and len(stream) >= max_tokens
            # BEFORE the break, not after it (#306). With the emit on the far
            # side, the row that tripped the cap was the one row whose work
            # never showed up -- and on a corpus whose first row fills the
            # budget that was every frame the run had.
            throttle.emit({
                # EVENT_BATCH marks this as LIVENESS, not measurement:
                # ``run_service.scalar_metrics`` mines every top-level number
                # of an unlabelled progress payload into a chart series, and a
                # "rows" series throttled by wall clock would be a chart of how
                # fast the machine is. Tokenization has no metrics worth
                # plotting -- the numbers it produces are outputs, not curves.
                "event": EVENT_BATCH,
                "rows": rows_done,
                "total_rows": total_rows,
                # Capped the same way the stream is below, so the last frame
                # counts tokens the node keeps rather than tokens it read.
                "tokens": min(len(stream), max_tokens) if max_tokens
                          else len(stream),
            })
            if full:
                # Break rather than tokenize the rest and throw it away: the
                # point of the cap is the time it saves.
                break

        if max_tokens:
            del stream[max_tokens:]
        if not len(stream):
            return torch.empty(0, dtype=torch.int64), rows_done, stopped
        # No copy: the tensor adopts the array's buffer and holds a reference
        # to it, and nothing mutates the array after this point.
        return torch.frombuffer(stream, dtype=torch.int64), rows_done, stopped

    @staticmethod
    def _result(
        tokens: torch.Tensor, seq_len: int, *, cached: bool,
    ) -> dict[str, Any]:
        blocks = PackedTokenBlocks(tokens, seq_len)
        total_tokens = int(tokens.numel())
        note = (
            f"{total_tokens:,} tokens packed into {blocks.num_blocks:,} blocks "
            f"of {seq_len}"
            f"{' (loaded from the block cache)' if cached else ''}."
        )
        # ``__log__`` is the one result key the canvas Log tab renders, and
        # dunder keys are filtered out of recorded outputs and port summaries.
        return {
            "dataset": blocks,
            "total_tokens": total_tokens,
            "num_blocks": blocks.num_blocks,
            "__log__": note,
        }
