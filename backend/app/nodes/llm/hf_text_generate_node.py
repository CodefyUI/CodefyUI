"""HFTextGenerate -- the answer at the end of the RAG chain.

    ... -> Retriever -> PromptBuilder -> HFTextGenerate

The node that finally says something. Everything before it decides WHICH
paragraphs the model gets to read; this one loads a small instruction-tuned
model off disk, hands it the prompt ``PromptBuilder`` assembled, and streams
the reply back a token at a time.

**Why a second generator node.** ``TextGenerate`` continues text with a model
the learner trained on the canvas -- a few thousand parameters that learned
one corpus -- and that is the right node for watching a language model be
built. It is the wrong node for watching RAG work: a model that cannot follow
"answer using only the context below" makes retrieval unfalsifiable, because
a bad answer is equally well explained by the model being tiny. So this node
loads pre-trained weights (Qwen2.5-0.5B-Instruct, Apache-2.0, about a
gigabyte) and the two nodes stay separate rather than growing a mode switch:
their inputs are different (a MODEL port versus a pack download), their
tokenizers are different, and the lesson each one carries is different.

**Why the chat template is applied here.** An instruction-tuned model was
fine-tuned on a specific transcript format -- role markers, turn boundaries,
a token that ends the assistant's turn. Handing it the bare prompt string
produces the same weights doing a visibly worse job, and it is exactly the
kind of failure a learner cannot diagnose. ``apply_chat_template`` is the
tokenizer's own answer to "how does THIS model want to be addressed", so the
node asks it rather than hard-coding Qwen's markers.

**Why the KV cache is here and not in TextGenerate.** ``TextGenerate``
deliberately recomputes the whole context every step: that is the honest
teaching version, and at its lengths the recompute is a fraction of a
second. Here the model is three orders of magnitude larger and the target is
a laptop CPU, where a quadratic decode turns a 200-token answer into
minutes. So each step after the first feeds ONE token and the cache carries
the rest. The cache object is treated as opaque -- taken off one forward
pass and handed to the next -- because its internal type has changed more
than once across transformers releases and nothing here needs to look
inside.

**Sampling happens on the CPU, always**, and through ``TextGenerate``'s own
``_next_token``: one implementation of temperature/top-k/top-p means a
learner who has understood those three knobs on one node has understood them
on both, and a seed produces the same answer on a laptop and on a GPU box.
See that module's docstring for the full reasoning.

**The run never downloads.** ``load_causal_lm`` raises ``PackMissingError``
naming the pack, and nothing here catches it: the editor reads the
``(pack=<id>)`` suffix back to offer the download, and a wrapped error would
cost the learner that button.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

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
from ._hf_generators import (
    DEFAULT_GENERATOR,
    GENERATOR_MODELS,
    RAG_PACK,
    load_causal_lm,
    option_packs_for_generators,
    stop_ids,
)
from .text_generate_node import _next_token

logger = logging.getLogger(__name__)


def _number(params: dict[str, Any], name: str, default: float) -> float:
    """A float param, defaulted only on a missing/null/empty value.

    Same trap and the same fix as ``TextGenerate._number``: the
    ``float(params.get(name, d) or d)`` idiom reads FALSINESS as "not set",
    and 0.0 is a legal, meaningful value for both params that come through
    here -- ``temperature=0`` means greedy and would have come back as 0.2.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HFTextGenerate: {name} must be a number, got {raw!r}.") from exc


def _integer(params: dict[str, Any], name: str, default: int) -> int:
    """An int param, defaulted only on a missing/null/empty value.

    The same trap in integer form: ``int(params.get("top_k", 50) or 50)``
    reads ``top_k=0`` -- a legal value meaning "disabled" -- as "not set"
    and quietly re-enables the filter the graph turned off.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HFTextGenerate: {name} must be a whole number, got {raw!r}."
        ) from exc


class HFTextGenerateNode(BaseNode):
    NODE_NAME = "HFTextGenerate"
    CATEGORY = "LLM"
    # The whole node, not just some options: its only model comes out of this
    # pack, so an install without it cannot run the node at all.
    REQUIRES_PACK = RAG_PACK
    DESCRIPTION = (
        "Answer a prompt with a small instruction-tuned open model that runs "
        "locally: Qwen2.5-0.5B-Instruct (Apache-2.0, about 1 GB, from the rag "
        "pack). The chat template is applied for you, and tokens stream to "
        "the canvas as they are produced. Expect a few tokens per second on a "
        "laptop CPU; a GPU is much faster. Unlike TextGenerate, which "
        "continues text with a model you trained on the canvas, this one "
        "loads pre-trained weights and follows instructions."
    )

    # Sampling is not a function of the params: it advances a seeded generator
    # per token, and every run of a graph is entitled to a fresh draw. Streaming
    # makes it worse -- the streamed tokens are the point of running the node,
    # and a cache hit returns the final string having emitted nothing, so a run
    # the user was told succeeded shows an empty panel. Same reasoning as
    # TextGenerate and LLMChat.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="prompt",
                data_type=DataType.STRING,
                description=(
                    "The question to answer, normally PromptBuilder.prompt; "
                    "falls back to the prompt param when not connected."
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
                    "The generated answer. The prompt is not echoed, and the "
                    "end-of-turn token is not written out."
                ),
            ),
            PortDefinition(
                name="token_count",
                data_type=DataType.SCALAR,
                description=(
                    "How many tokens were generated. Below max_new_tokens "
                    "means the model chose to stop."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="model",
                param_type=ParamType.SELECT,
                default=DEFAULT_GENERATOR,
                # Built through ``option_packs_for_generators`` so the
                # ``pack:item`` spelling is written down once instead of
                # retyped here, where ``packs.parse_requirement`` would
                # reject the typo only once a learner picked the option.
                options=list(GENERATOR_MODELS),
                option_packs=option_packs_for_generators(),
                description=(
                    "The model to load. Greyed options need the rag pack "
                    "from Package Center."
                ),
            ),
            ParamDefinition(
                name="prompt",
                param_type=ParamType.STRING,
                default=(
                    "In one sentence, what is retrieval-augmented generation?"
                ),
                description=(
                    "Prompt when no `prompt` input is connected."
                ),
            ),
            ParamDefinition(
                name="system_prompt",
                param_type=ParamType.STRING,
                default=(
                    "You are a helpful assistant. Answer concisely and in the "
                    "language of the question."
                ),
                description=(
                    "System instruction placed before the user message. Leave "
                    "empty to send none."
                ),
            ),
            ParamDefinition(
                name="max_new_tokens",
                param_type=ParamType.INT,
                default=200,
                min_value=1,
                max_value=2048,
                description=(
                    "Upper bound on generated tokens; this decides how long "
                    "the node runs."
                ),
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                max_value=2.0,
                description=(
                    "Scores are divided by this before sampling: 0 = greedy "
                    "(always the most likely token, reproducible); higher = "
                    "more varied."
                ),
            ),
            ParamDefinition(
                name="top_p",
                param_type=ParamType.FLOAT,
                default=0.9,
                min_value=0.0,
                max_value=1.0,
                description=(
                    "Nucleus sampling: only the most likely tokens whose "
                    "cumulative probability reaches p (1 = off)."
                ),
            ),
            ParamDefinition(
                name="top_k",
                param_type=ParamType.INT,
                default=50,
                min_value=0,
                description=(
                    "Only the k highest-scoring tokens are sampled (0 = off)."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Sampling seed. Same seed and model give the same answer."
                ),
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description=(
                    "Where to generate (auto follows the global device)."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="dtype",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "float32", "float16", "bfloat16"],
                description=(
                    "Weight precision. auto uses bfloat16/float16 on CUDA and "
                    "float32 on CPU and MPS."
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
        # Function-level, like every device-aware node in this package: the
        # registry imports this module at startup to build the palette, and
        # the palette must not pay for anything a run pays for.
        from ...core.device_utils import resolve_node_device

        # Same fallback contract as TextGenerate's `prompt` port: a connected
        # input wins, and the param is what the node uses on its own. An input
        # carrying "" is a wired-up upstream node producing nothing, not an
        # unset param, so it does NOT fall back -- answering a question the
        # user cannot see in the graph is worse than saying it is empty.
        raw_prompt = inputs.get("prompt")
        if raw_prompt is None:
            prompt = str(params.get("prompt") or "")
        else:
            prompt = str(raw_prompt)
        if not prompt.strip():
            # Before anything is loaded: a wiring mistake must read as a
            # wiring mistake, not as a gigabyte of weights and then silence.
            raise ValueError(
                "HFTextGenerate: the prompt is empty, so there is nothing to "
                "answer. Type something into the `prompt` param, or wire a "
                "non-empty string -- normally PromptBuilder.prompt -- into "
                "the `prompt` input.")

        repo = str(params.get("model") or DEFAULT_GENERATOR)
        system_prompt = str(params.get("system_prompt") or "")
        # Through the helpers rather than the ``int(params.get(name, d) or d)``
        # idiom, because 0 is a legal and meaningful value for three of these
        # (greedy, disabled, a seed) and that idiom reads falsiness as unset.
        max_new_tokens = max(1, _integer(params, "max_new_tokens", 200))
        temperature = max(0.0, _number(params, "temperature", 0.2))
        top_k = max(0, _integer(params, "top_k", 50))
        top_p = min(1.0, max(0.0, _number(params, "top_p", 0.9)))
        seed = _integer(params, "seed", 0)
        device = resolve_node_device(params.get("device"), context)
        dtype = str(params.get("dtype") or "auto")

        tokenizer, model = load_causal_lm(repo, device, dtype)

        # An empty system prompt sends NO system message rather than an empty
        # one: some chat templates render the role marker regardless, and a
        # blank system turn is a real (if small) instruction to the model.
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # What the FIRST forward pass gets: the whole prompt. Every pass
        # after it gets a single token, and the cache carries the rest.
        step_ids = tokenizer(chat, return_tensors="pt")["input_ids"].to(device)

        stops = stop_ids(tokenizer, model)
        # Deliberately a local generator, not ``torch.manual_seed``: the global
        # RNG belongs to the run (a DataLoader's shuffle, dropout masks), and a
        # node that reseeded it would change every other node's randomness. CPU
        # regardless of ``device`` -- see ``text_generate_node``'s docstring.
        generator = torch.Generator()
        generator.manual_seed(seed)

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        new_ids: list[int] = []
        past: Any = None
        stopped_after: int | None = None
        finished_on_eos = False

        # Started AFTER the load: the first run in a session pays seconds to
        # read a gigabyte off disk and every run after it pays none, so a clock
        # started above would report one wildly different tok/s figure and then
        # honest ones for identical work.
        started = time.monotonic()
        with torch.no_grad():
            for _step in range(max_new_tokens):
                # At the TOP of the step, #155's placement: a stop that landed
                # while this node sat in the queue does not pay for a whole
                # forward pass, and the partial text always holds every token
                # that was actually produced.
                if should_stop():
                    stopped_after = len(new_ids)
                    break
                out = model(
                    input_ids=step_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                # Opaque, both ways: whatever the model handed back goes
                # straight into the next call. See the module docstring.
                past = out.past_key_values
                next_id = _next_token(
                    out.logits[:, -1, :].float().cpu(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )
                if next_id in stops:
                    # The model said the turn was over. The token itself is not
                    # written out -- it is punctuation for the machine, and
                    # decoding it would append a literal <|im_end|>.
                    finished_on_eos = True
                    break
                new_ids.append(next_id)
                # Only the new token from here on; the cache carries the rest.
                step_ids = torch.tensor(
                    [[next_id]], dtype=torch.int64, device=device)
                throttle.emit({
                    # EVENT_BATCH marks this as LIVENESS: the frames are
                    # throttled by wall clock, so several share one step and
                    # ``run_service.scalar_metrics`` must not mine "tokens"
                    # into a chart series -- that would be a chart of how fast
                    # the machine is.
                    "event": EVENT_BATCH,
                    # The whole answer so far, re-decoded each frame rather
                    # than concatenated per-token pieces: a BPE token is not a
                    # character boundary, so decoding ids one at a time mangles
                    # any multi-byte character spanning two of them -- which in
                    # Chinese is most of them.
                    "text": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "tokens": len(new_ids),
                    "total_tokens": max_new_tokens,
                })
        elapsed = time.monotonic() - started

        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        reason = (
            "model emitted end-of-turn" if finished_on_eos
            else "stopped" if stopped_after is not None
            else "hit max_new_tokens"
        )
        rate = len(new_ids) / elapsed if elapsed > 0 else 0.0
        # ``__log__`` is the one result key the canvas Log tab renders, and
        # dunder keys are filtered out of recorded outputs and port summaries.
        # The tok/s is in it because it is the number that tells a learner
        # whether their next experiment should be a shorter max_new_tokens or
        # a different device.
        note = (f"generated {len(new_ids)} tokens ({reason}) in "
                f"{elapsed:.1f}s ({rate:.1f} tok/s) on {device}")
        result: dict[str, Any] = {
            "text": text,
            "token_count": len(new_ids),
            "__log__": note,
        }
        if stopped_after is not None:
            # The partial answer is still returned -- half a sentence beats
            # nothing -- and the result says it is partial. Never raises (see
            # ``core.loop_control``).
            result.update(interrupted_result(tokens=stopped_after))
        logger.info("HFTextGenerate: %s (model=%s)", note, repo)
        return result
