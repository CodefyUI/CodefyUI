"""TextGenerate node (#291) -- sample text out of a trained language model.

The payoff node of the LLM epic: everything else in the chain produces a loss
curve, and this is the one that shows what the model learned to say. It is
also the smallest complete implementation of decoding, which is the point --
the loop below is the whole of it:

1. Encode the prompt into token ids.
2. Forward the ids, take the logits of the LAST position only. Those are the
   scores for "what comes next".
3. Turn the scores into one token: ``temperature`` flattens or sharpens them,
   ``top_k`` and ``top_p`` delete the tail, and one draw from what is left.
4. Append it and go back to 2, until the end-of-text token or the budget.

**The context is recomputed every step.** No KV cache: step ``t`` re-runs
attention over all ``t`` tokens, which makes generation quadratic in length
instead of linear. That is the honest teaching version -- a cache is an
optimisation with its own invariants (and a second code path to keep correct),
and at the lengths a lesson generates the recompute is a fraction of a second.

**The window slides.** A model has positions for ``max_seq_len`` tokens and
raises rather than truncating (see ``CausalLMModel``), so once the context
reaches that length the oldest tokens fall off the front. Generation past the
context length therefore works, and the model simply cannot see the beginning
of what it wrote.

**Sampling happens on the CPU, always.** The last-position logits are copied
to the CPU before the temperature/top-k/top-p arithmetic and the draw. Two
reasons: ``torch.Generator`` has no MPS flavour in every torch build this
project supports, and a single seed then produces the same text on a laptop
and on a GPU box -- which is what "seed" has to mean for it to be worth
having. The cost is one ``(1, vocab_size)`` copy per token against a full
forward pass over the whole context.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

logger = logging.getLogger(__name__)

#: The members an object on the ``tokenizer`` port must have. Same contract,
#: same check and the same error style as ``LMTokenizedDataset`` -- a partial
#: object must fail at the node the user can see.
TOKENIZER_MEMBERS = ("encode", "decode", "eos_id", "vocab_size")

#: Score assigned to a token the filters removed. ``-inf`` is what softmax
#: needs to give it exactly zero probability; a large negative number would
#: leave it a small one, and over 200 draws "small" happens.
_FILTERED = float("-inf")


def _number(params: dict[str, Any], name: str, default: float) -> float:
    """A float param, defaulted only on a missing/null/empty value.

    Same reasoning as ``CausalLMModel._float_param``: the ``float(params.get(
    name, d) or d)`` idiom reads FALSINESS as "not set", and 0.0 is a legal,
    meaningful value for both params that come through here -- ``temperature=0``
    means greedy and would have come back as 0.8.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TextGenerate: {name} must be a number, got {raw!r}.") from exc


def _integer(params: dict[str, Any], name: str, default: int) -> int:
    """An int param, defaulted only on a missing/null/empty value.

    The same trap as :func:`_number` in integer form: ``int(params.get("top_k",
    50) or 50)`` reads ``top_k=0`` -- a legal value that means "disabled" -- as
    "not set" and hands back 50, quietly re-enabling the filter the graph
    turned off.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TextGenerate: {name} must be a whole number, got {raw!r}."
        ) from exc


def _context_window(model: Any) -> int | None:
    """How many tokens of context to keep, or None for "as many as there are".

    ``CausalLMModel`` publishes ``max_seq_len`` (#289) and refuses a longer
    sequence rather than truncating it, so generation past the context length
    only works if the caller slides a window. None is returned for a module
    that does not publish a usable one -- a hand-written model, or one loaded
    from a checkpoint written before that attribute existed. Guessing a window
    there would silently truncate a model that could have handled the whole
    context; letting the model's own length check speak is the honest default.

    ``bool`` is excluded explicitly because it is an ``int`` subclass, and
    ``max_seq_len = True`` would otherwise become a one-token window.
    """
    raw = getattr(model, "max_seq_len", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(raw) if raw >= 1 else None


def _check_tokenizer(tokenizer: Any) -> None:
    """Enforce the duck-typed tokenizer contract, naming the fix."""
    if tokenizer is None:
        raise ValueError(
            "TextGenerate has no tokenizer. Wire the LMTokenizer node that "
            "tokenized the training data into the `tokenizer` input -- "
            "generating with a different vocabulary produces noise.")
    missing = [n for n in TOKENIZER_MEMBERS if not hasattr(tokenizer, n)]
    for name in ("encode", "decode"):
        if name not in missing and not callable(getattr(tokenizer, name)):
            missing.append(name)
    if missing:
        raise ValueError(
            f"TextGenerate: the object on the `tokenizer` input is missing "
            f"{sorted(missing)} -- a tokenizer here must provide "
            f"encode(text) -> list[int], decode(ids) -> str, eos_id and "
            f"vocab_size. Wire an LMTokenizer node into that input.")


def _filter_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the *top_k* highest-scoring tokens, delete the rest.

    The cheap half of the story: it bounds how far down the distribution a
    draw can reach, so the one-in-fifty-thousand nonsense token cannot be
    sampled at all. Its weakness is that ``k`` is fixed -- the same 50 tokens'
    worth of freedom applies where one continuation is obvious and where a
    hundred are plausible, which is what ``top_p`` fixes.
    """
    k = min(int(top_k), int(scores.shape[-1]))
    if k <= 0:
        return scores
    # The k-th highest score is the threshold; ``<`` rather than ``<=`` so a
    # tie at the boundary keeps every tied token rather than none of them.
    threshold = torch.topk(scores, k, dim=-1).values[..., -1, None]
    return scores.masked_fill(scores < threshold, _FILTERED)


def _filter_top_p(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probabilities reach *top_p*.

    Nucleus sampling: the cut adapts to how confident the model is. Where one
    token holds 0.98 the nucleus is that one token; where the model is unsure
    it may be hundreds. ``top_p=1`` keeps everything, which is why 1 is the
    disabled value.
    """
    ordered, indices = torch.sort(scores, descending=True, dim=-1)
    cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
    # Shifted one to the right so the token that CROSSES the threshold is
    # itself kept: without the shift a first token holding more than top_p
    # would be removed along with everything else and there would be nothing
    # left to sample. Position 0 is unconditionally kept for the same reason.
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    ordered = ordered.masked_fill(remove, _FILTERED)
    # Back to vocabulary order, so the sampled index is a token id.
    return torch.full_like(scores, _FILTERED).scatter_(-1, indices, ordered)


def _next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> int:
    """One token id out of one position's logits.

    *logits* is ``(1, vocab_size)`` on the CPU in fp32. The order of the three
    knobs is the conventional one and it matters: temperature rescales BEFORE
    the filters, so raising it widens the nucleus rather than only flattening
    what survived it.
    """
    if temperature <= 0.0:
        # Greedy. Not "temperature = a very small number": that is a division
        # by something near zero, which overflows to inf and makes the softmax
        # a nan. 0 means argmax, and argmax is a separate branch.
        return int(logits.argmax(dim=-1).item())

    scores = logits / temperature
    if top_k > 0:
        scores = _filter_top_k(scores, top_k)
    if top_p < 1.0:
        scores = _filter_top_p(scores, top_p)
    probabilities = scores.softmax(dim=-1)
    return int(torch.multinomial(
        probabilities, num_samples=1, generator=generator).item())


class TextGenerateNode(BaseNode):
    NODE_NAME = "TextGenerate"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Continues a prompt with a trained language model, one token at a "
        "time, and streams the text as it appears. temperature, top_k and "
        "top_p are the three knobs that decide how adventurous the writing "
        "is: temperature 0 always takes the most likely token (repetitive but "
        "safe), and raising it trades coherence for variety. Stops at the "
        "end-of-text token or after max_new_tokens."
    )

    # Sampling is not a function of the params: it advances a seeded generator
    # per token, and every run of a graph is entitled to a fresh draw of text.
    # Streaming makes it worse -- the streamed tokens are the point of running
    # the node, and a cache hit returns the final string having emitted
    # nothing, so a run the user was told succeeded shows an empty panel. Same
    # reasoning as LLMChat.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description=(
                    "Trained language model: input_ids int64 (batch, seq_len) "
                    "in, logits (batch, seq_len, vocab_size) out. Its "
                    "max_seq_len attribute sizes the sliding context window."
                ),
            ),
            PortDefinition(
                name="tokenizer",
                data_type=DataType.ANY,
                description=(
                    "Tokenizer object providing encode(text) -> list[int], "
                    "decode(ids) -> str, eos_id and vocab_size. It must be the "
                    "one the model was trained with -- wire the same "
                    "LMTokenizer node."
                ),
            ),
            PortDefinition(
                name="prompt",
                data_type=DataType.STRING,
                description=(
                    "Text to continue. Optional -- falls back to the `prompt` "
                    "param when not connected."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description=(
                    "The prompt followed by the generated continuation. The "
                    "end-of-text token is not written out, so a completion "
                    "that stopped on its own simply ends."
                ),
            ),
            PortDefinition(
                name="token_count",
                data_type=DataType.SCALAR,
                description=(
                    "How many NEW tokens were generated, not counting the "
                    "prompt or a closing end-of-text token. Below "
                    "max_new_tokens means the model chose to stop."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="prompt",
                param_type=ParamType.STRING,
                default="Once upon a time",
                description=(
                    "Text to continue. Used when no `prompt` input is "
                    "connected. A prompt in the style of the training data "
                    "gets the most out of a small model."
                ),
            ),
            ParamDefinition(
                name="max_new_tokens",
                param_type=ParamType.INT,
                default=200,
                min_value=1,
                max_value=4096,
                description=(
                    "How many tokens to generate at most. Each one costs a "
                    "forward pass over everything written so far, so this is "
                    "the knob that decides how long the node takes."
                ),
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=0.8,
                min_value=0.0,
                max_value=2.0,
                description=(
                    "Divides the scores before sampling: below 1 sharpens the "
                    "distribution, above 1 flattens it. 0 = greedy, always the "
                    "single most likely token, which is reproducible and tends "
                    "to fall into loops."
                ),
            ),
            ParamDefinition(
                name="top_k",
                param_type=ParamType.INT,
                default=50,
                min_value=0,
                max_value=1000,
                description=(
                    "Only ever sample from the k highest-scoring tokens "
                    "(0 = disabled). This is what stops a one-in-fifty-thousand "
                    "token from derailing a sentence."
                ),
            ),
            ParamDefinition(
                name="top_p",
                param_type=ParamType.FLOAT,
                default=0.95,
                min_value=0.0,
                max_value=1.0,
                description=(
                    "Nucleus sampling: keep the most likely tokens until their "
                    "probabilities add up to p, and sample from those "
                    "(1 = disabled). Unlike top_k the cut adapts -- narrow "
                    "where the model is sure, wide where it is not."
                ),
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Seed for the sampling draws. The same seed and the same "
                    "model give the same text, on any device, so a comparison "
                    "of two temperatures differs only by the temperature."
                ),
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda"],
                description=(
                    "Device to generate on ('auto' follows the global device, "
                    "so a model trained on the GPU generates there)."
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
        from ...core.device_utils import resolve_node_device, to_device
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            stop_checker,
        )

        model = inputs.get("model")
        tokenizer = inputs.get("tokenizer")
        if model is None:
            raise ValueError(
                "TextGenerate requires a `model` input. Wire a CausalLMModel "
                "-- or the model output of the TrainingLoop that trained it -- "
                "into that input.")
        _check_tokenizer(tokenizer)

        # Same fallback contract as Tokenizer's `text` port: a connected input
        # wins, and the param is what the node uses on its own. An input
        # carrying "" is a wired-up upstream node producing nothing, not an
        # unset param, so it does NOT fall back -- generating from text the
        # user cannot see in the graph is worse than saying the prompt is empty.
        raw_prompt = inputs.get("prompt")
        if raw_prompt is None:
            prompt = str(params.get("prompt") or "")
        else:
            prompt = str(raw_prompt)

        # All four numeric params go through the helpers rather than the
        # ``int(params.get(name, d) or d)`` idiom, because 0 is a legal and
        # meaningful value for three of them (greedy, disabled, disabled) and
        # that idiom reads falsiness as "not set".
        max_new_tokens = max(1, _integer(params, "max_new_tokens", 200))
        temperature = max(0.0, _number(params, "temperature", 0.8))
        top_k = max(0, _integer(params, "top_k", 50))
        top_p = min(1.0, max(0.0, _number(params, "top_p", 0.95)))
        seed = _integer(params, "seed", 0)
        device = resolve_node_device(params.get("device"), context)

        ids = list(tokenizer.encode(prompt))
        if not ids:
            raise ValueError(
                "TextGenerate: the prompt is empty, so the model has nothing "
                "to continue from -- it needs at least one token to predict "
                "the next one. Type something into the `prompt` param, or wire "
                "a non-empty string into the `prompt` input.")

        eos_id = int(tokenizer.eos_id)
        window = _context_window(model)

        model = to_device(model, device)
        # Restored on the way out: the same module flows on to whatever is
        # wired after this node, and a second training phase downstream must
        # not silently run with dropout disabled because a generation happened
        # in between.
        was_training = model.training
        model.eval()

        # Deliberately a local generator, not ``torch.manual_seed``: the global
        # RNG belongs to the run (the DataLoader's shuffle, dropout masks), and
        # a node that reseeded it would change every other node's randomness.
        # CPU regardless of ``device`` -- see this module's docstring.
        generator = torch.Generator()
        generator.manual_seed(seed)

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        new_ids: list[int] = []
        stopped_after: int | None = None
        finished_on_eos = False

        try:
            with torch.no_grad():
                for _step in range(max_new_tokens):
                    # #155's placement, one modality over: checked at the TOP
                    # of the step, so a stop that landed while this node sat in
                    # the queue does not pay for a whole forward pass, and the
                    # partial text always holds every token that was actually
                    # produced.
                    if should_stop():
                        stopped_after = len(new_ids)
                        break
                    context_ids = ids[-window:] if window else ids
                    batch = torch.tensor(
                        [context_ids], dtype=torch.int64, device=device)
                    logits = model(batch)
                    if logits.dim() != 3:
                        raise ValueError(
                            f"TextGenerate expects a language model whose "
                            f"logits are (batch, seq_len, vocab_size), got "
                            f"{tuple(logits.shape)}. Wire a CausalLMModel into "
                            f"`model`; a classifier has nothing to continue a "
                            f"prompt with.")
                    # The last position only: rows 0..T-2 are predictions for
                    # tokens we already have.
                    next_id = _next_token(
                        logits[:, -1, :].float().cpu(),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                    )
                    if next_id == eos_id:
                        # The model said it was done. The token itself is not
                        # written out -- it is punctuation for the machine, and
                        # decoding it would append a literal <|endoftext|>.
                        finished_on_eos = True
                        break
                    ids.append(next_id)
                    new_ids.append(next_id)
                    throttle.emit({
                        # EVENT_BATCH marks this as LIVENESS: the frames are
                        # throttled by wall clock, so several share one step
                        # and ``run_service.scalar_metrics`` must not mine
                        # "tokens" into a chart series -- that would be a chart
                        # of how fast the machine is.
                        "event": EVENT_BATCH,
                        # The whole text so far, re-decoded each frame rather
                        # than concatenated per-token pieces: a BPE token is
                        # not a character boundary, so decoding ids one at a
                        # time mangles any multi-byte character that spans two
                        # of them. Decoding a few hundred ids is microseconds
                        # against the forward pass that produced them.
                        "text": prompt + tokenizer.decode(new_ids),
                        "tokens": len(new_ids),
                        "total_tokens": max_new_tokens,
                    })
        finally:
            model.train(was_training)

        text = prompt + tokenizer.decode(new_ids)
        reason = (
            "the model emitted end-of-text" if finished_on_eos
            else "stopped" if stopped_after is not None
            else "reached max_new_tokens"
        )
        # ``__log__`` is the one result key the canvas Log tab renders, and
        # dunder keys are filtered out of recorded outputs and port summaries.
        note = f"generated {len(new_ids)} tokens ({reason})."
        result: dict[str, Any] = {
            "text": text,
            "token_count": len(new_ids),
            "__log__": note,
        }
        if stopped_after is not None:
            # The partial text is still returned -- half a story beats nothing
            # -- and the result says it is partial. Never raises (see
            # ``core.loop_control``).
            result.update(interrupted_result(tokens=stopped_after))
        logger.info(
            "TextGenerate: %s (device=%s, context window=%s)",
            note, device, window if window else "unbounded")
        return result
