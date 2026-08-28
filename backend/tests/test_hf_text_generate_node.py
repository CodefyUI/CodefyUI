"""Tests for HFTextGenerateNode -- the answer at the end of the RAG chain.

Nothing here downloads anything and nothing here is a transformer. The
``fake_transformers`` fixture from ``conftest`` installs a stand-in library
AND tells the packs bridge the rag pack is present; both halves are needed,
or a test stops at the gate before it reaches the node's own logic.

The fake model's rule is ``next = (last id + 1) % 128`` and its ids are code
points, so every expected string can be written down rather than hoped for.
That is what makes the assertions below about the node's OWN loop -- the
chat template, the KV cache, where the stop check sits, what the progress
frames carry -- rather than about a language model's taste.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from app.core.execution_context import INTERRUPTED_KEY
from app.core.loop_control import EVENT_BATCH
from app.core.node_base import DataType
from app.core.packs import parse_requirement
from app.core.packs.catalog import get_item, get_pack
from app.nodes.llm._hf_generators import (
    DEFAULT_GENERATOR,
    GENERATOR_MODELS,
    RAG_PACK,
    option_packs_for_generators,
)
from app.nodes.llm.hf_text_generate_node import HFTextGenerateNode

#: Every default the node ships, read off the node itself so a test names
#: only the params it actually changes -- and so a changed default cannot
#: quietly stop being the thing under test.
_DEFAULTS = {p.name: p.default for p in HFTextGenerateNode.define_params()}


def _never_stop() -> bool:
    return False


@dataclass
class FakeContext:
    """The two attributes this node reads off an ExecutionContext."""

    device: str = "cpu"
    should_stop: Callable[[], bool] = _never_stop


def _run(*, inputs=None, context=None, progress_callback=None, **params):
    p = dict(_DEFAULTS)
    p.update(params)
    return HFTextGenerateNode().execute(
        dict(inputs or {}), p, progress_callback, context=context)


def _continuation(chat: str, count: int) -> str:
    """What the fake model writes after *chat*: one code point higher each
    step, starting from the prompt's last character."""
    start = ord(chat[-1]) % 128
    return "".join(chr((start + step) % 128) for step in range(1, count + 1))


def _chat(module, index: int = 0) -> str:
    """The templated string the node handed the tokenizer on run *index*."""
    return module.tokenizers[0].chat_calls[index]["rendered"]


# ── what the palette and the editor see ──────────────────────────────────


def test_node_metadata_requires_rag_pack_and_is_not_cacheable():
    """The two class-level promises, and the SELECT's download offer.

    ``REQUIRES_PACK`` greys the node out in the palette before anybody
    presses Run; ``cacheable = False`` is what keeps a re-run from returning
    the previous answer having streamed nothing, which would show the
    learner an empty panel under a run they were told succeeded.
    """
    assert HFTextGenerateNode.REQUIRES_PACK == RAG_PACK
    assert HFTextGenerateNode.cacheable is False
    assert parse_requirement(HFTextGenerateNode.REQUIRES_PACK) == (
        RAG_PACK, None)

    prompt_port = HFTextGenerateNode.define_inputs()[0]
    assert prompt_port.name == "prompt"
    assert prompt_port.data_type == DataType.STRING
    assert prompt_port.optional is True
    assert [p.name for p in HFTextGenerateNode.define_outputs()] == [
        "text", "token_count"]

    model_param = {p.name: p for p in HFTextGenerateNode.define_params()}["model"]
    assert model_param.options == list(GENERATOR_MODELS)
    assert model_param.default == DEFAULT_GENERATOR
    assert model_param.option_packs == option_packs_for_generators()
    for requirement in model_param.option_packs.values():
        pack_id, item_id = parse_requirement(requirement)
        # The editor offers this download, so the catalog has to have it --
        # and it has to be the SAME repo the SELECT option names, or the
        # download button would fetch a model this node cannot load.
        assert get_item(get_pack(pack_id), item_id).repo_id in {
            repo for repo, value in model_param.option_packs.items()
            if value == requirement}


# ── the prompt ───────────────────────────────────────────────────────────


def test_chat_template_is_applied_with_system_and_user(fake_transformers):
    """An instruction-tuned model is addressed the way it was fine-tuned.

    Handing it the bare prompt gives the same weights doing a visibly worse
    job, which is the kind of failure a learner cannot diagnose -- so what
    reaches the tokenizer must be the TEMPLATED string, and the roles must
    be the ones the template expects.
    """
    _run(max_new_tokens=1, temperature=0.0, prompt="What is RAG?",
         system_prompt="Be brief.")

    tokenizer = fake_transformers.tokenizers[0]
    call = tokenizer.chat_calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is RAG?"},
    ]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert tokenizer.encoded == [call["rendered"]], (
        "the raw prompt reached the model instead of the templated string")

    # Whitespace is not an instruction: a blank system prompt sends none,
    # rather than a system turn saying nothing.
    _run(max_new_tokens=1, temperature=0.0, prompt="What is RAG?",
         system_prompt="   ")
    assert tokenizer.chat_calls[1]["messages"] == [
        {"role": "user", "content": "What is RAG?"}]

    # A connected input wins over the param -- the PromptBuilder case.
    _run(max_new_tokens=1, temperature=0.0, inputs={"prompt": "from upstream"},
         prompt="the fallback nobody wired")
    assert tokenizer.chat_calls[2]["messages"][-1] == {
        "role": "user", "content": "from upstream"}


def test_empty_prompt_is_an_error(fake_transformers):
    """And it is an error BEFORE a gigabyte of weights is read off disk."""
    with pytest.raises(ValueError) as caught:
        _run(prompt="")
    assert "prompt is empty" in str(caught.value)

    # A connected input carrying nothing is an upstream node that produced
    # nothing, not an unset param, so it does NOT fall back to the param.
    with pytest.raises(ValueError):
        _run(inputs={"prompt": "   "}, prompt="a perfectly good fallback")

    assert fake_transformers.model_loads == [], (
        "the model was loaded before the prompt was checked")


# ── the decode loop ──────────────────────────────────────────────────────


def test_greedy_is_deterministic_and_stops_on_eos(fake_transformers):
    """Five tokens and then end-of-turn: the run stops itself, and neither
    the prompt nor the stop token is written out."""
    fake_transformers.eos_at_step = 5

    first = _run(temperature=0.0, prompt="ask me")
    chat = _chat(fake_transformers)
    model = fake_transformers.models[0]

    assert first["token_count"] == 5
    assert first["text"] == _continuation(chat, 5)
    assert "ask me" not in first["text"], "the prompt was echoed"
    assert "\x00" not in first["text"], "the end-of-turn token was decoded"
    assert "end-of-turn" in first["__log__"]
    # Six forward passes: five that produced a token, and the one that
    # returned end-of-turn.
    assert len(model.calls) == 6

    # The same seed and the same model give the same answer -- including
    # across a cache HIT, where the second run reuses this very object.
    second = _run(temperature=0.0, prompt="ask me")
    assert second["text"] == first["text"]
    assert second["token_count"] == 5
    assert fake_transformers.model_loads == [
        fake_transformers.model_loads[0]], "the cached model was reloaded"


def test_max_new_tokens_bounds_the_loop(fake_transformers):
    """A model that never stops is stopped by the budget, and the log says
    which of the two happened."""
    res = _run(max_new_tokens=4, temperature=0.0)

    assert res["token_count"] == 4
    assert res["text"] == _continuation(_chat(fake_transformers), 4)
    assert len(fake_transformers.models[0].calls) == 4
    assert "hit max_new_tokens" in res["__log__"]
    assert "tok/s" in res["__log__"]
    assert INTERRUPTED_KEY not in res


def test_kv_cache_is_passed_back(fake_transformers):
    """The first step feeds the whole prompt; every step after it feeds ONE
    token and the cache carries the rest.

    Without this the decode is quadratic in length, which on the CPU this
    node targets turns a 200-token answer into minutes.
    """
    _run(max_new_tokens=3, temperature=0.0)

    model = fake_transformers.models[0]
    chat = _chat(fake_transformers)
    prompt_len = len(chat)

    assert [len(call["input_ids"][0]) for call in model.calls] == [
        prompt_len, 1, 1]
    assert model.calls[0]["input_ids"] == [[ord(c) % 128 for c in chat]]
    assert model.calls[0]["past_key_values"] is None
    assert all(call["use_cache"] is True for call in model.calls)

    # What came off one forward pass went into the next, untouched.
    assert model.calls[1]["past_key_values"].fed_back is True
    assert model.calls[1]["past_key_values"].length == prompt_len
    assert model.calls[2]["past_key_values"].length == prompt_len + 1


def test_progress_frames_carry_the_running_text(fake_transformers, monkeypatch):
    """Streamed tokens are the point of running this node, so every frame
    carries the whole answer so far rather than a per-token fragment."""
    monkeypatch.setattr("app.core.loop_control.PROGRESS_MIN_INTERVAL_S", 0.0)
    frames: list[dict] = []

    res = _run(max_new_tokens=3, temperature=0.0,
               progress_callback=frames.append)
    chat = _chat(fake_transformers)

    assert frames == [
        {"event": EVENT_BATCH, "text": _continuation(chat, 1),
         "tokens": 1, "total_tokens": 3},
        {"event": EVENT_BATCH, "text": _continuation(chat, 2),
         "tokens": 2, "total_tokens": 3},
        {"event": EVENT_BATCH, "text": _continuation(chat, 3),
         "tokens": 3, "total_tokens": 3},
    ]
    assert res["text"] == frames[-1]["text"]


def test_stop_returns_partial_text_and_interrupted_marker(fake_transformers):
    """Stop is checked at the TOP of a step, so a click never pays for one
    more forward pass -- and what was already generated comes back rather
    than being thrown away."""
    answers = iter([False, False, True])

    res = _run(max_new_tokens=10, temperature=0.0,
               context=FakeContext(should_stop=lambda: next(answers)))

    assert res["token_count"] == 2
    assert res["text"] == _continuation(_chat(fake_transformers), 2)
    assert len(fake_transformers.models[0].calls) == 2, (
        "a stopped loop paid for another forward pass")
    assert res[INTERRUPTED_KEY] == {"tokens": 2}
    assert "stopped" in res["__log__"]


def test_seed_does_not_touch_global_rng(fake_transformers):
    """A node that called ``torch.manual_seed`` would silently change every
    other node's randomness -- a DataLoader's shuffle, a dropout mask."""
    import torch

    torch.manual_seed(4321)
    before = torch.rand(3)

    torch.manual_seed(4321)
    # temperature > 0, so the multinomial draw really happens.
    res = _run(max_new_tokens=5, temperature=1.0, seed=7)

    assert res["token_count"] == 5
    assert torch.equal(torch.rand(3), before)
