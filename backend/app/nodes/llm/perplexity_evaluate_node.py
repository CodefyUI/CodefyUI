"""PerplexityEvaluate node (#291) -- how surprised a language model is by text.

``EvaluateModel`` answers "how often is the argmax right", which is the
question a classifier is for. A language model's question is different: over
a whole sequence it is wrong most of the time and should be, because several
continuations are legitimate. What it can be scored on is how much
probability it put on the token that actually came next -- the cross-entropy
-- and the conventional way to report that is its exponential::

    val_loss   = sum(-log p(actual token)) / number of scored tokens
    perplexity = exp(val_loss)

Perplexity has a unit a learner can hold onto: "the model was choosing
between about this many equally likely tokens at each step". A uniform guess
over a 50257-token vocabulary is a perplexity of 50257; GPT-2 small on
WikiText-103 is around 37.

**Why the mean is token-weighted and not a mean of batch means.** Every batch
here contributes ``reduction="sum"`` and its own count of scored positions,
and the division happens once at the end. Averaging per-batch means instead
would weight a ragged final batch -- or one carrying more ``-100`` labels --
exactly as heavily as a full one, which makes the reported number depend on
``batch_size``. That is not a rounding difference: 5 blocks at
``batch_size=2`` is batches of 2, 2 and 1, and the last one would get a third
of the weight instead of a fifth.

**Perplexity is not comparable across tokenizers.** It is a per-TOKEN
average, so a vocabulary that packs more text into each token is scored on
fewer, harder predictions. Two numbers are only worth comparing when they
came from the same dataset and the same tokenizer -- which is why this node
reports ``val_loss`` and ``tokens`` alongside it rather than perplexity
alone.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...core.amp import PRECISIONS
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from .lm_cross_entropy_loss_node import DEFAULT_IGNORE_INDEX

logger = logging.getLogger(__name__)


def _safe_exp(val_loss: float) -> float:
    """``exp(val_loss)``, or infinity where float64 cannot hold it.

    ``math.exp`` RAISES above about 709, and a loss that high is never a
    measurement worth failing a run over -- it means the model and the dataset
    disagree about what the token ids mean (most often a model trained against
    a different tokenizer than the blocks were packed with). Infinity is the
    honest reading, and every consumer already handles it: the port summary
    renders a non-finite scalar as no value, and the event store replaces one
    with null rather than emitting the ``Infinity`` token JSON does not have.
    """
    try:
        return math.exp(val_loss)
    except OverflowError:
        return float("inf")


def _check_batch(batch: Any) -> None:
    """Refuse a dataset that is not made of ``(input_ids, labels)`` pairs.

    The mistake this exists for is wiring ``TextCorpusDataset`` straight in:
    its batches are lists of STRINGS, and without this the failure surfaces
    three frames down as ``torch.as_tensor`` refusing a str -- naming neither
    the node nor the fix. Same 2-tuple contract ``TrainingLoop`` unpacks.
    """
    if not (isinstance(batch, (list, tuple)) and len(batch) >= 2
            and torch.is_tensor(batch[0])):
        raise ValueError(
            "PerplexityEvaluate: the dataset must yield (input_ids, labels) "
            "pairs of int64 tensors. Wire LMTokenizedDataset into `dataset` "
            "-- a raw text corpus has no labels to score against.")


def _sum_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor,
) -> tuple[float, int]:
    """Summed cross-entropy over the scored positions, and how many there were.

    Returned as Python numbers rather than tensors so the running totals
    accumulate in float64: under ``bf16`` autocast the logits are 16-bit, and
    a 16-bit running sum over hundreds of thousands of tokens loses digits
    that matter to a loss reported to four decimals. The ``.float()`` puts
    the log-softmax itself back in fp32 too, which is what ``F.cross_entropy``
    does under autocast anyway.

    Positions labelled :data:`DEFAULT_IGNORE_INDEX` contribute neither loss
    nor count, so a dataset that masks a prompt half is scored only on the
    half it meant to score.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"PerplexityEvaluate expects a language model whose logits are "
            f"(batch, seq_len, vocab_size), got {tuple(logits.shape)}. Wire a "
            f"CausalLMModel into `model`; a plain classifier belongs in "
            f"EvaluateModel.")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            f"PerplexityEvaluate: the logits are {tuple(logits.shape[:2])} "
            f"(batch, seq_len) but the labels are {tuple(labels.shape)}. The "
            f"dataset must yield labels the same length as the input ids -- "
            f"LMTokenizedDataset does.")

    vocab_size = logits.shape[-1]
    flat_labels = labels.reshape(-1).long()
    # ``reshape`` rather than ``view``: logits off an attention stack are
    # frequently non-contiguous, and ``view`` fails on exactly those.
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        flat_labels,
        ignore_index=DEFAULT_IGNORE_INDEX,
        reduction="sum",
    )
    scored = int((flat_labels != DEFAULT_IGNORE_INDEX).sum().item())
    return float(loss.item()), scored


class PerplexityEvaluateNode(BaseNode):
    NODE_NAME = "PerplexityEvaluate"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Scores a trained language model on held-out text. It runs the whole "
        "dataset, averages the cross-entropy over every token it scored, and "
        "reports exp(that) as perplexity -- roughly how many equally likely "
        "tokens the model was choosing between at each step, so a uniform "
        "guess over a 50257-token vocabulary is 50257. The average is PER "
        "TOKEN and specific to this dataset and tokenizer, so only compare "
        "numbers measured the same way."
    )

    # Same reasoning as EvaluateModel (#254), one modality over. Two halves:
    #
    # 1. This node's product is partly a side effect. It writes val_loss and
    #    perplexity through ``context.log_metric``, and a cache hit returns
    #    the recorded outputs without calling execute(), so those points are
    #    never written -- the chart is empty on a run the user was told
    #    succeeded.
    # 2. The measurement is stale on a hit in its own right. Perplexity is a
    #    function of the model's WEIGHTS, and the cache key describes how the
    #    module was built, not what TrainingLoop has since made of it. The
    #    intended graph puts this node downstream of a training loop mutating
    #    the very module it measures.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description=(
                    "Trained language model: input_ids int64 (batch, seq_len) "
                    "in, logits (batch, seq_len, vocab_size) out. Wire "
                    "TrainingLoop's model output here."
                ),
            ),
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description=(
                    "Blocks of (input_ids, labels) from LMTokenizedDataset -- "
                    "ideally packed from a validation split the model has not "
                    "trained on. Labels of -100 are skipped."
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="val_loss",
                data_type=DataType.SCALAR,
                description=(
                    "Mean cross-entropy per scored token, in nats. This is the "
                    "same quantity TrainingLoop reports as val_loss, so the "
                    "two are directly comparable."
                ),
            ),
            PortDefinition(
                name="perplexity",
                data_type=DataType.SCALAR,
                description=(
                    "exp(val_loss). Lower is better, and 'better' only means "
                    "anything against another model measured on this same "
                    "dataset with this same tokenizer."
                ),
            ),
            PortDefinition(
                name="tokens",
                data_type=DataType.SCALAR,
                description=(
                    "How many tokens the average is over (-100 labels and any "
                    "batches skipped by max_batches excluded). A perplexity "
                    "measured over a few thousand tokens is noisy."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="batch_size",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                max_value=256,
                description=(
                    "How many blocks are scored at once. It does not change "
                    "the result -- the average is weighted by tokens, not by "
                    "batches -- only speed and memory."
                ),
            ),
            ParamDefinition(
                name="max_batches",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Stop after this many batches (0 = the whole dataset). A "
                    "quick estimate during a lesson; the `tokens` output says "
                    "how much it was actually measured over."
                ),
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda"],
                description=(
                    "Device to score on ('auto' follows the global device, so "
                    "a model trained on the GPU is measured there)."
                ),
            ),
            ParamDefinition(
                name="precision",
                param_type=ParamType.SELECT,
                default="bf16",
                options=list(PRECISIONS),
                description=(
                    "Mixed precision for the forward pass. bf16 roughly halves "
                    "activation memory on Ampere and newer, which is what lets "
                    "a long context be scored at all; the loss is still summed "
                    "in fp32. A device that cannot honour the choice falls back "
                    "to fp32 and says so in the log."
                ),
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
        from ...core.amp import AmpPolicy
        from ...core.device_utils import resolve_node_device, to_device
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            loader_length,
            stop_checker,
        )

        model = inputs.get("model")
        dataset = inputs.get("dataset")
        if model is None:
            raise ValueError(
                "PerplexityEvaluate requires a `model` input. Wire a "
                "CausalLMModel -- or the model output of the TrainingLoop that "
                "trained it -- into that input.")
        if dataset is None:
            raise ValueError(
                "PerplexityEvaluate requires a `dataset` input. Wire an "
                "LMTokenizedDataset into that input, packed from the split you "
                "want to measure on.")

        batch_size = max(1, int(params.get("batch_size", 8) or 8))
        # ``or 0`` is safe here because 0 IS the disabled value.
        max_batches = max(0, int(params.get("max_batches", 0) or 0))
        # Through ``resolve_node_device`` rather than ``resolve_device``, for
        # the reason #204 spelled out on EvaluateModel: "auto" has to mean
        # "follow the run-level device", or a graph submitted with cuda trains
        # on the GPU and then silently scores on the CPU.
        device = resolve_node_device(params.get("device"), context)
        policy = AmpPolicy.for_device(params.get("precision"), device)

        # Its own loader, deliberately unshuffled: the average is over the
        # same tokens whatever the order, and an unshuffled pass makes the
        # progress frames mean something ("40% of the set").
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_batches = loader_length(loader)
        if max_batches and total_batches is not None:
            total_batches = min(total_batches, max_batches)

        model = to_device(model, device)
        # #291: restored on the way out, unlike EvaluateModel, which leaves
        # eval() latched. That is survivable there only because TrainingLoop
        # re-asserts train() at the top of every epoch; it is not a property
        # worth relying on from a node that a graph legitimately puts BETWEEN
        # two training phases, where a silently disabled dropout would change
        # what the second phase learns.
        was_training = model.training
        model.eval()

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        stopped_at_batch: int | None = None
        loss_sum = 0.0
        token_count = 0
        batches_done = 0

        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    # Checked BEFORE the stop check: a run that reached its
                    # own cap finished, and reporting it as interrupted would
                    # withhold the metric point it did earn.
                    if max_batches and batch_index >= max_batches:
                        break
                    # #122: a validation set is a long loop too, and an
                    # uninterruptible measurement is what makes Stop feel
                    # broken right after the training it follows finally
                    # stopped.
                    if should_stop():
                        stopped_at_batch = batch_index
                        break
                    _check_batch(batch)
                    input_ids = to_device(batch[0], device)
                    labels = to_device(torch.as_tensor(batch[1]), device)
                    with policy.autocast():
                        logits = model(input_ids)
                    batch_loss, batch_tokens = _sum_cross_entropy(logits, labels)
                    loss_sum += batch_loss
                    token_count += batch_tokens
                    batches_done += 1
                    running = _safe_exp(loss_sum / token_count) if token_count else 0.0
                    throttle.emit({
                        # EVENT_BATCH marks this as LIVENESS: several frames
                        # legitimately share one step, so
                        # ``run_service.scalar_metrics`` must not mine
                        # "perplexity" out of it into a chart series -- that
                        # would be a chart of how fast the machine is. The one
                        # authoritative point is logged below.
                        "event": EVENT_BATCH,
                        "batch": batches_done,
                        "total_batches": total_batches,
                        "perplexity": round(running, 4),
                    })
        finally:
            model.train(was_training)

        val_loss = loss_sum / token_count if token_count else 0.0
        perplexity = _safe_exp(val_loss)
        note = (
            f"perplexity {perplexity:.4g} (val_loss {val_loss:.4f} nats/token) "
            f"over {token_count:,} tokens in {batches_done:,} batches."
        )
        # ``__log__`` is the one result key the canvas Log tab renders, and
        # dunder keys are filtered out of recorded outputs and port summaries.
        result: dict[str, Any] = {
            "val_loss": val_loss,
            "perplexity": perplexity,
            "tokens": token_count,
            "__log__": note,
        }

        if stopped_at_batch is not None:
            # The partial average is still returned -- a perplexity over the
            # first 40% of a set beats nothing -- but the result says so, and
            # an incomplete pass is deliberately NOT filed as a measurement.
            result.update(interrupted_result(batch=stopped_at_batch))
        elif context is not None:
            # Step 0, not a configurable step like EvaluateModel's: this is
            # one measurement of a finished model, not a series. Sharing the
            # name "val_loss" with TrainingLoop is safe and deliberate --
            # metric points carry their node id and the run charts draw one
            # line per (node, series), so this appears as its own line
            # labelled with this node rather than as a stray point in the
            # training curve. Two PerplexityEvaluate nodes in one graph are
            # likewise two lines, so neither overwrites the other.
            context.log_metric("val_loss", val_loss, 0)
            context.log_metric("perplexity", perplexity, 0)

        logger.info(
            "PerplexityEvaluate: %s (device=%s, precision=%s)",
            note, device, policy.precision)
        return result
