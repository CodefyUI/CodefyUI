"""Tests for CausalLMModelNode (#289).

Three groups: the node's contract (metadata, params, the parameter count the
UI reports), the module's forward behaviour under every option combination,
and one end-to-end run through the REAL ``TrainingLoop``/``Optimizer`` nodes
that proves the whole thing can actually learn.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.llm.causal_lm_model_node import (
    ACTIVATIONS,
    NORM_TYPES,
    POSITIONAL_MODES,
    CausalLMModelNode,
    _resolve_config,
)
from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode
from app.nodes.training.optimizer_node import OptimizerNode
from app.nodes.training.training_loop_node import TrainingLoopNode

#: The teaching-sized configuration every test below starts from. Well under
#: the ``min_value`` the editor declares, deliberately -- see
#: ``test_a_tiny_config_is_not_clamped_up_to_the_declared_minimum``.
TINY = {
    "vocab_size": 100,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 64,
    "max_seq_len": 16,
}


def _build(**overrides):
    """The MODEL output for ``TINY`` plus *overrides*."""
    params = {**TINY, "seed": 1, **overrides}
    return CausalLMModelNode().execute({}, params)


def _zh_tw_block(node_name: str) -> str:
    """This node's entry in the zh-TW catalog, as raw text.

    Same file and same block shape ``test_api_nodes.py`` reads for the
    translation ratchet. Read here as well because one assertion below is
    about what the CHINESE copy claims, and a wrong number there is invisible
    to a reviewer reading the Python.
    """
    catalog = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n"
        / "nodeLocales" / "zh-TW.ts"
    ).read_text(encoding="utf-8")
    block = re.search(rf"\n  {node_name}: \{{(.*?)\n  \}},", catalog, re.DOTALL)
    assert block, f"{node_name} has no zh-TW entry"
    return block.group(1)


def _expected_param_count(
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    d_ff: int,
    max_seq_len: int,
    positional: str = "learned",
    norm: str = "layernorm",
    tie_embeddings: bool = True,
) -> int:
    """The parameter count derived from the architecture, term by term.

    Written out rather than read off a built model, because "the number the
    node reports equals the number the node computed" would assert nothing.
    n_heads is absent on purpose: splitting a fixed width into more heads
    reshapes the projections without adding a single weight, which is one of
    the things this count is here to keep true.
    """
    # LayerNorm has a gain and a bias; RMSNorm has only the gain.
    norm_params = 2 * d_model if norm == "layernorm" else d_model
    per_block = (
        norm_params                                  # pre-attention norm
        + (3 * d_model * d_model + 3 * d_model)      # fused q/k/v projection
        + (d_model * d_model + d_model)              # attention out-projection
        + norm_params                                # pre-MLP norm
        + (d_model * d_ff + d_ff)                    # MLP up-projection
        + (d_ff * d_model + d_model)                 # MLP down-projection
    )
    total = (
        vocab_size * d_model            # token embedding
        + n_layers * per_block
        + norm_params                   # final norm
    )
    if positional == "learned":
        total += max_seq_len * d_model  # one trained vector per position
    if not tie_embeddings:
        total += vocab_size * d_model   # a separate LM head matrix
    # The LM head never has a bias, tied or not, so nothing is added for it.
    return total


# ── the node's contract ───────────────────────────────────────────────────

def test_node_metadata():
    assert CausalLMModelNode.NODE_NAME == "CausalLMModel"
    assert CausalLMModelNode.CATEGORY == "LLM"
    outputs = {p.name: p.data_type.value
               for p in CausalLMModelNode.define_outputs()}
    assert outputs == {"model": "MODEL", "param_count": "SCALAR"}
    assert CausalLMModelNode.define_inputs() == []


def test_the_node_is_not_cacheable_because_it_hands_out_a_live_model():
    """The #253/#254 invariant, pinned on this node by name.

    ``test_cache_live_handle_nodes.py`` asserts it registry-wide; this says
    it here too, because a cache hit on THIS node is a run that trains the
    module the previous run already trained.
    """
    assert CausalLMModelNode.cacheable is False


def test_every_construction_param_is_in_the_structure_hash():
    """A param outside ``structural_params`` is a param the user can edit
    while a persisted module quietly ignores it."""
    declared = {p.name for p in CausalLMModelNode.define_params()}
    assert set(CausalLMModelNode.structural_params) == declared


def test_param_definitions_match_the_issue_spec():
    """Names, defaults and ranges are the wire contract Graph Copilot writes
    against (#289), so they are pinned rather than described."""
    params = {p.name: p for p in CausalLMModelNode.define_params()}
    expected = {
        "vocab_size": (50257, 256, 300000),
        "d_model": (1024, 64, 8192),
        "n_layers": (12, 1, 64),
        "n_heads": (16, 1, 64),
        "d_ff": (4096, 64, 32768),
        "max_seq_len": (1024, 16, 8192),
        "dropout": (0.0, 0.0, 0.9),
    }
    for name, (default, minimum, maximum) in expected.items():
        assert params[name].default == default, name
        assert params[name].min_value == minimum, name
        assert params[name].max_value == maximum, name

    assert params["positional"].default == "learned"
    assert params["positional"].options == ["learned", "sinusoidal", "rope"]
    assert params["norm"].default == "layernorm"
    assert params["norm"].options == ["layernorm", "rmsnorm"]
    assert params["activation"].default == "gelu"
    assert params["activation"].options == ["gelu", "relu", "silu"]
    assert params["tie_embeddings"].default is True
    assert params["gradient_checkpointing"].default is False
    assert params["init_std"].default == 0.02
    assert params["seed"].default == 0


def test_the_execute_side_fallbacks_agree_with_the_declared_defaults():
    """Each default is written twice -- once for the editor, once as
    ``execute``'s fallback for a graph missing the key -- which is this
    repo's prevailing pattern and therefore a drift risk. A node whose
    empty-params behaviour differs from what the form shows is a node whose
    exported Python and canvas run disagree.
    """
    declared = {p.name: p.default for p in CausalLMModelNode.define_params()}
    resolved = _resolve_config({})
    assert resolved.keys() == declared.keys()
    for name, default in declared.items():
        assert resolved[name] == default, name


def test_the_beginner_facing_params_are_not_hidden_behind_advanced():
    """The two-tier UI (#134): the seven knobs that decide the model's size
    stay on the default view, the tuning knobs collapse."""
    advanced = {p.name: p.advanced for p in CausalLMModelNode.define_params()}
    basic = {name for name, flag in advanced.items() if not flag}
    assert basic == {
        "vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
        "max_seq_len", "tie_embeddings",
    }


# ── the parameter count the node reports ──────────────────────────────────

@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"tie_embeddings": False},
        {"positional": "sinusoidal"},
        {"positional": "rope"},
        {"norm": "rmsnorm"},
        {"positional": "rope", "norm": "rmsnorm", "tie_embeddings": False},
        {"n_heads": 8},
        {"n_layers": 5, "d_ff": 128},
    ],
)
def test_param_count_matches_the_analytic_formula(overrides):
    result = _build(**overrides)
    config = {**TINY, **overrides}
    expected = _expected_param_count(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        positional=config.get("positional", "learned"),
        norm=config.get("norm", "layernorm"),
        tie_embeddings=config.get("tie_embeddings", True),
    )
    assert result["param_count"] == expected, overrides


def test_the_advertised_default_size_matches_the_declared_defaults():
    """The node's copy quotes a model size, and copy drifts silently.

    Shipped as "~350M" in the DESCRIPTION, the ``param_count`` port
    description and the zh-TW block -- roughly 70% above the 204M the declared
    defaults actually build. Nothing caught it, because no test builds the
    DEFAULT model: at 204M parameters that is ~800MB of float32, which is not
    something a unit suite should allocate. So the claim is checked against
    the same term-by-term formula the counts above are checked against, and
    both places it appears are checked against that one number.

    Epic #292's target is a ~200M-parameter model with exactly these
    defaults, so a figure that stops matching means either the copy or the
    defaults moved -- both worth failing on.
    """
    declared = {p.name: p.default for p in CausalLMModelNode.define_params()}
    total = _expected_param_count(
        vocab_size=declared["vocab_size"],
        d_model=declared["d_model"],
        n_layers=declared["n_layers"],
        d_ff=declared["d_ff"],
        max_seq_len=declared["max_seq_len"],
        positional=declared["positional"],
        norm=declared["norm"],
        tie_embeddings=declared["tie_embeddings"],
    )
    assert total == 203_668_480, (
        f"the declared defaults now build {total:,} parameters. That is the "
        f"reference shape #292 sizes its run against, so check the defaults "
        f"before updating this number -- and update the copy either way.")

    advertised = f"{round(total / 1e6)}M"      # "204M"
    assert advertised in CausalLMModelNode.DESCRIPTION, (
        f"the DESCRIPTION does not advertise {advertised}: "
        f"{CausalLMModelNode.DESCRIPTION!r}")
    # The block itself is deliberately NOT in the message: dumping a few
    # hundred characters of CJK into a Windows cp950 console is not a useful
    # failure report.
    assert advertised in _zh_tw_block("CausalLMModel"), (
        f"the zh-TW description of CausalLMModel does not advertise "
        f"{advertised}; a learner reading Chinese would be told a different "
        f"model size. See frontend/src/i18n/nodeLocales/zh-TW.ts.")
    # Only the DESCRIPTION states the figure. Repeating it on the port is how
    # the two came to disagree in the first place.
    port = next(p for p in CausalLMModelNode.define_outputs()
                if p.name == "param_count")
    assert not re.search(r"\d+\s*M\b", port.description), (
        f"the param_count port states a model size again: "
        f"{port.description!r}. One fact in two places is one fact that can "
        f"disagree with itself.")


def test_param_count_counts_only_trainable_parameters():
    result = _build()
    model = result["model"]
    assert result["param_count"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad)


def test_tying_the_embeddings_shares_one_matrix_and_counts_it_once():
    tied, untied = _build()["model"], _build(tie_embeddings=False)["model"]
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.lm_head.weight is not untied.tok_emb.weight
    saved = TINY["vocab_size"] * TINY["d_model"]
    assert (_build(tie_embeddings=False)["param_count"]
            - _build()["param_count"]) == saved


# ── forward behaviour ─────────────────────────────────────────────────────

def test_forward_maps_token_ids_to_next_token_logits():
    model = _build()["model"]
    ids = torch.randint(0, TINY["vocab_size"], (2, 8), dtype=torch.int64)
    logits = model(ids)
    assert logits.shape == (2, 8, TINY["vocab_size"])
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_the_module_exposes_max_seq_len_and_vocab_size():
    """A contract, not a convenience: ``TextGenerate`` (#291) sizes its
    sliding context window from ``model.max_seq_len``, so removing the
    attribute breaks generation rather than this node."""
    model = _build(max_seq_len=12)["model"]
    assert model.max_seq_len == 12
    assert model.vocab_size == TINY["vocab_size"]


def test_a_sequence_longer_than_max_seq_len_is_refused_by_name():
    model = _build(max_seq_len=16)["model"]
    too_long = torch.zeros(1, 17, dtype=torch.int64)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(too_long)
    # The boundary itself is allowed -- an off-by-one here would cap every
    # model one token below what the user asked for.
    assert model(torch.zeros(1, 16, dtype=torch.int64)).shape[1] == 16


def test_a_single_sequence_without_a_batch_dimension_says_so():
    model = _build()["model"]
    with pytest.raises(ValueError, match="unsqueeze"):
        model(torch.zeros(8, dtype=torch.int64))


def test_float_input_ids_are_refused_rather_than_rounded():
    model = _build()["model"]
    with pytest.raises(ValueError, match="integer token ids"):
        model(torch.zeros(1, 8, dtype=torch.float32))


def test_narrower_integer_ids_are_accepted():
    """int32 ids come out of numpy and out of hand-built datasets; widening
    an id is never lossy, so it is done rather than reported."""
    model = _build()["model"]
    ids = torch.randint(0, TINY["vocab_size"], (1, 4), dtype=torch.int32)
    assert model(ids).shape == (1, 4, TINY["vocab_size"])


@pytest.mark.parametrize("positional", POSITIONAL_MODES)
@pytest.mark.parametrize("norm", NORM_TYPES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_every_option_combination_runs_a_forward_pass(
    positional, norm, activation,
):
    model = _build(
        positional=positional, norm=norm, activation=activation)["model"]
    ids = torch.randint(0, TINY["vocab_size"], (2, 8), dtype=torch.int64)
    logits = model(ids)
    assert logits.shape == (2, 8, TINY["vocab_size"])
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("positional", POSITIONAL_MODES)
def test_a_position_cannot_see_the_tokens_that_come_after_it(positional):
    """The defining property of a decoder-only LM, asserted directly.

    Written after a mutation run: flipping ``is_causal=True`` to ``False``
    left every other test in this file green, the overfit run included -- a
    model allowed to read ahead memorises a repeated batch FASTER, because it
    can copy the answer out of the next position's embedding. Nothing but this
    test stands between that mutation and a graph that reports a beautiful
    training loss and generates nonsense.
    """
    model = _build(positional=positional, max_seq_len=8)["model"].eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    edited = ids.clone()
    edited[0, 4] = 9  # change one token late in the sequence

    with torch.no_grad():
        before, after = model(ids), model(edited)

    # Positions 0..3 precede the edit, so their logits must not move at all.
    assert torch.allclose(before[0, :4], after[0, :4], atol=1e-6)
    # The edited position itself must move, or the assertion above would also
    # hold for a model that ignores its input.
    assert not torch.allclose(before[0, 4], after[0, 4], atol=1e-4)


@pytest.mark.parametrize("positional", POSITIONAL_MODES)
def test_position_information_actually_reaches_the_logits(positional):
    """Reordering the tokens BEFORE the last one has to change its prediction.

    Without any position information, one causal block's output at the final
    position is a function of the SET of tokens it can see plus its own -- so
    permuting the prefix leaves it bit-identical. That invariance is the
    measurement: it holds for exactly the bug this test is looking for (a
    forgotten table add, an un-rotated query) and for nothing else. Verified
    against a mutant with all three position paths removed: the difference
    below collapses from ~1 to ~3e-8.

    Two details are load-bearing. ONE block, because with two the position-2
    hidden states already differ for an unrelated reason (their own prefixes
    differ), and the invariance stops being exact. And a large ``init_std``,
    because at 0.02 the attention logits are near zero, attention is nearly
    uniform, and a uniform average over a permuted prefix is invariant by
    accident -- the real signal sits only ~100x above float noise, which is
    not a margin worth building a test on.
    """
    model = _build(
        positional=positional, n_layers=1, init_std=0.5, max_seq_len=8,
    )["model"].eval()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    shuffled_prefix = torch.tensor([[3, 2, 1, 4]], dtype=torch.int64)

    with torch.no_grad():
        moved = (model(ids)[0, -1]
                 - model(shuffled_prefix)[0, -1]).abs().max().item()
    assert moved > 0.1, (
        f"the last position's logits moved by only {moved:.2e} when the "
        f"tokens before it were reordered, so {positional!r} position "
        f"information is not reaching them.")


def test_d_model_must_divide_evenly_by_n_heads():
    with pytest.raises(ValueError) as excinfo:
        _build(d_model=32, n_heads=5)
    message = str(excinfo.value)
    assert "d_model" in message and "n_heads" in message


def test_rope_needs_an_even_head_width():
    """RoPE rotates each head's dimensions in pairs, so d_model // n_heads
    has to be even even when it divides evenly."""
    with pytest.raises(ValueError, match="rope"):
        _build(d_model=12, n_heads=4, d_ff=32, positional="rope")


@pytest.mark.parametrize(
    "param, value",
    [("positional", "alibi"), ("norm", "batchnorm"), ("activation", "swiglu")],
)
def test_an_unknown_select_value_is_refused(param, value):
    with pytest.raises(ValueError, match=value):
        _build(**{param: value})


# ── determinism ───────────────────────────────────────────────────────────

def test_the_same_seed_gives_bit_identical_weights():
    first = _build(seed=7)["model"].state_dict()
    second = _build(seed=7)["model"].state_dict()
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_a_different_seed_gives_different_weights():
    first = _build(seed=7)["model"]
    other = _build(seed=8)["model"]
    assert not torch.equal(first.tok_emb.weight, other.tok_emb.weight)


def test_the_weights_do_not_depend_on_the_global_rng():
    """The init draws from a LOCAL ``torch.Generator``, so ``seed`` is the
    only thing that decides the starting model.

    Drawing from the global RNG instead would still pass the two tests above
    -- two builds in a row with the same global state agree -- while making
    the model depend on how many random numbers the rest of the run happened
    to draw first. That is the failure that looks like "my reproducible run
    is not reproducible".

    Note the narrower claim: CONSTRUCTING the modules does advance the global
    RNG, because ``nn.Linear.reset_parameters`` draws from it before this node
    overwrites every weight it produced. What must not happen is the reverse
    direction -- the global stream leaking INTO the weights.
    """
    torch.manual_seed(1234)
    first = _build(seed=99)["model"].state_dict()
    torch.manual_seed(4321)
    torch.randn(17)  # advance the global stream by an odd amount
    second = _build(seed=99)["model"].state_dict()
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_the_residual_projections_start_smaller_than_the_rest():
    """GPT-2's scaled residual init: without it, the activations arriving at
    the last block grow with depth and a deep stack diverges in its first
    steps. Measured as a ratio so it pins the sqrt(2 * n_layers) rule rather
    than one particular seed's draw.
    """
    n_layers, init_std = 8, 0.02
    model = _build(n_layers=n_layers, init_std=init_std)["model"]
    expected_ratio = 1.0 / (2 * n_layers) ** 0.5

    scaled = torch.cat([
        model.blocks[i].attn.out_proj.weight.flatten() for i in range(n_layers)
    ] + [
        model.blocks[i].mlp.fc2.weight.flatten() for i in range(n_layers)
    ])
    plain = torch.cat([
        model.blocks[i].attn.qkv.weight.flatten() for i in range(n_layers)
    ])
    assert scaled.std().item() == pytest.approx(
        init_std * expected_ratio, rel=0.1)
    assert plain.std().item() == pytest.approx(init_std, rel=0.1)


def test_the_norm_gains_start_at_one_and_the_biases_at_zero():
    model = _build()["model"]
    assert torch.equal(model.norm_f.weight, torch.ones_like(model.norm_f.weight))
    assert torch.equal(
        model.blocks[0].attn.qkv.bias,
        torch.zeros_like(model.blocks[0].attn.qkv.bias),
    )


# ── mixed precision, gradients, memory ────────────────────────────────────

@pytest.mark.parametrize("positional", POSITIONAL_MODES)
def test_the_forward_survives_bf16_autocast(positional):
    """``TrainingLoop`` runs the forward inside ``policy.autocast()``, so a
    hard-coded float32 in the position tables or the RoPE rotation would
    raise a dtype mismatch on every bf16 run. Exercised on CPU because that
    is where CI is; the CUDA path is the test at the end of this file.
    """
    model = _build(positional=positional)["model"]
    ids = torch.randint(0, TINY["vocab_size"], (2, 8), dtype=torch.int64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(ids)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all()


def test_gradient_checkpointing_still_produces_gradients():
    """The point of checkpointing is that the backward pass recomputes what
    the forward threw away, so a missing gradient is the failure mode."""
    model = _build(gradient_checkpointing=True, dropout=0.1)["model"]
    model.train()
    ids = torch.randint(0, TINY["vocab_size"], (2, 8), dtype=torch.int64)
    model(ids).sum().backward()
    missing = [name for name, param in model.named_parameters()
               if param.grad is None]
    assert missing == []


def test_gradient_checkpointing_computes_the_same_forward():
    """It is a memory/time trade, not a different model: with dropout off
    and eval mode the two paths must agree numerically."""
    plain = _build(seed=5)["model"].eval()
    checkpointed = _build(seed=5, gradient_checkpointing=True)["model"].eval()
    ids = torch.randint(0, TINY["vocab_size"], (1, 8), dtype=torch.int64)
    with torch.no_grad():
        assert torch.allclose(plain(ids), checkpointed(ids), atol=1e-6)


# ── defensive coercion of hand-built graph JSON ───────────────────────────

def test_a_tiny_config_is_not_clamped_up_to_the_declared_minimum():
    """The ``min_value`` on each param is the editor's guard rail for a model
    somebody means to train. Enforcing it in ``execute`` would silently
    rewrite every teaching-sized configuration -- including the ones in this
    file -- so the node builds exactly what it was asked for.
    """
    model = _build()["model"]
    assert model.vocab_size == 100          # below the declared 256
    assert model.tok_emb.weight.shape[1] == 32   # below the declared 64


@pytest.mark.parametrize(
    "raw, expected",
    [("8", 8), (8.0, 8), (0, 1), (-4, 1), (None, 12), ("", 12)])
def test_size_params_survive_hand_written_json(raw, expected):
    """Only reachable through hand-edited or generated graph JSON; the INT
    widget cannot produce any of these. Floored at 1 rather than rejected,
    because a zero-layer model has an obvious intended reading; a null or an
    empty string means "not set" and falls back to the declared default (12
    layers), not to whatever the caller passed for the other params."""
    model = _build(n_layers=raw)["model"]
    assert len(model.blocks) == expected


def test_a_size_param_that_is_not_a_number_at_all_is_reported():
    with pytest.raises(ValueError, match="whole number"):
        _build(d_model="wide")


def test_an_explicit_init_std_of_zero_is_honoured_not_defaulted():
    """The ``float(params.get(name, d) or d)`` idiom this repo uses elsewhere
    reads 0.0 as "not set", which is harmless wherever the default is itself 0
    and wrong here: ``init_std=0`` is the declared minimum and a lesson of its
    own (every weight starts identical, so every head learns the same thing).
    Caught by review; pinned so the idiom cannot come back.
    """
    def total_magnitude(weight: torch.Tensor) -> float:
        return float(weight.detach().abs().sum())

    model = _build(init_std=0.0)["model"]
    assert total_magnitude(model.tok_emb.weight) == 0.0
    assert total_magnitude(model.blocks[0].attn.qkv.weight) == 0.0
    # ...and the default really is 0.02, i.e. the assertion above is not just
    # measuring a node that never initialises anything.
    assert total_magnitude(_build()["model"].tok_emb.weight) > 0.0


def test_a_float_param_that_is_not_a_number_is_reported():
    with pytest.raises(ValueError, match="init_std must be a number"):
        _build(init_std="small")


def test_dropout_is_clamped_into_a_probability():
    """torch raises for p > 1, and p = 1 zeroes every activation, so an
    out-of-range dropout has no reading worth preserving."""
    assert _build(dropout=5.0)["model"].drop.p == 0.9
    assert _build(dropout=-1.0)["model"].drop.p == 0.0


# ── weight persistence ────────────────────────────────────────────────────

def _persisting_context(**kwargs) -> ExecutionContext:
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="lm",
        **kwargs,
    )


def test_persisted_weights_survive_a_second_run():
    """"Persist weights between runs" has to mean something here: a second
    execute with the same params continues from what training left behind."""
    node, context = CausalLMModelNode(), _persisting_context()
    params = {**TINY, "seed": 1}
    first = node.execute({}, params, context=context)["model"]
    with torch.no_grad():
        first.tok_emb.weight.fill_(0.5)
    second = node.execute({}, params, context=context)["model"]
    assert second is first
    assert torch.equal(second.tok_emb.weight, first.tok_emb.weight)


def test_changing_the_architecture_drops_the_persisted_weights():
    node, context = CausalLMModelNode(), _persisting_context()
    first = node.execute({}, {**TINY, "seed": 1}, context=context)["model"]
    wider = node.execute(
        {}, {**TINY, "seed": 1, "d_ff": 128}, context=context)["model"]
    assert wider is not first
    assert wider.blocks[0].mlp.fc1.out_features == 128


# ── end to end: it can learn ──────────────────────────────────────────────

def _repeated_batch_dataset(vocab_size: int, seq_len: int, repeats: int):
    """One fixed (input_ids, labels) pair, repeated.

    The labels are the inputs shifted one position left, which is the whole
    of next-token prediction. Fixed-length by construction: the DataLoader
    node passes no ``collate_fn``, so ``default_collate`` needs every sample
    to be the same length.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, vocab_size, (seq_len + 1,), dtype=torch.int64)
    inputs = ids[:-1].unsqueeze(0).repeat(repeats, 1).contiguous()
    labels = ids[1:].unsqueeze(0).repeat(repeats, 1).contiguous()
    return torch.utils.data.TensorDataset(inputs, labels)


def _overfit_run(**loop_params):
    """One real ``TrainingLoop`` run over 20 identical batches.

    Deliberately assembled from the production nodes rather than from bare
    torch: what is on test is that ``CausalLMModel`` + ``LMCrossEntropyLoss``
    satisfy ``TrainingLoop``'s batch contract -- a 2-tuple unpacked into
    ``loss_fn(model(data), targets)`` with no reshaping anywhere.
    """
    vocab_size, seq_len = 32, 8
    params = {
        "vocab_size": vocab_size, "d_model": 32, "n_layers": 2, "n_heads": 4,
        "d_ff": 64, "max_seq_len": seq_len, "seed": 3,
    }
    model = CausalLMModelNode().execute({}, params)["model"]
    optimizer = OptimizerNode().execute(
        {"model": model}, {"type": "AdamW", "lr": 1e-2})["optimizer"]
    loss_fn = LMCrossEntropyLossNode().execute({}, {})["loss_fn"]
    loader = torch.utils.data.DataLoader(
        _repeated_batch_dataset(vocab_size, seq_len, repeats=20),
        batch_size=4,
    )
    return TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer,
         "loss_fn": loss_fn},
        {"epochs": 20, "device": "cpu", **loop_params},
    )


def test_the_model_overfits_a_single_repeated_batch():
    """The only test here that would notice a subtly wrong gradient path.

    Every shape assertion above passes for a model whose attention is
    non-causal, whose residual stream is disconnected, or whose LM head is
    reading the wrong axis. Memorising eight tokens does not.
    """
    started = time.monotonic()
    result = _overfit_run()
    elapsed = time.monotonic() - started

    losses = result["losses"]
    assert losses.shape == (20,)
    assert torch.isfinite(losses).all()
    assert float(losses[-1]) < 0.1, (
        f"the model did not memorise one repeated batch: loss went "
        f"{float(losses[0]):.3f} -> {float(losses[-1]):.3f}. A loss stuck "
        f"near ln(vocab_size) = {torch.tensor(32.0).log().item():.3f} means "
        f"no gradient is reaching the weights.")
    assert float(losses[-1]) < float(losses[0])
    # A guard rail on the fixture, not a benchmark: this run takes well under
    # a second, and anything near the limit means the config drifted into
    # something a test suite should not be training.
    assert elapsed < 30.0, f"the overfit fixture took {elapsed:.1f}s"


def test_the_loss_is_not_treated_as_a_classification_loss():
    """``TrainingLoop`` computes ``val_accuracy`` only for
    ``nn.CrossEntropyLoss``/``nn.NLLLoss``, via ``argmax(dim=1)`` -- which on
    [B, T, V] logits would argmax over TIME and report a meaningless number.
    Running a validation pass is what proves the gate stays shut.
    """
    vocab_size, seq_len = 32, 8
    params = {
        "vocab_size": vocab_size, "d_model": 32, "n_layers": 1, "n_heads": 4,
        "d_ff": 64, "max_seq_len": seq_len, "seed": 3,
    }
    model = CausalLMModelNode().execute({}, params)["model"]
    optimizer = OptimizerNode().execute(
        {"model": model}, {"type": "AdamW", "lr": 1e-3})["optimizer"]
    loss_fn = LMCrossEntropyLossNode().execute({}, {})["loss_fn"]
    dataset = _repeated_batch_dataset(vocab_size, seq_len, repeats=8)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    result = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer,
         "loss_fn": loss_fn, "val_dataloader": loader},
        {"epochs": 1, "device": "cpu"},
    )
    assert result["metrics"]["final_val_loss"] is not None
    assert result["metrics"].get("final_val_accuracy") is None
    assert not isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="needs a CUDA device")
def test_bf16_accumulation_and_clipping_train_on_cuda():
    """The configuration #292's ~200M-parameter run actually uses.

    bf16 autocast, gradient accumulation and gradient clipping each touch a
    different part of the loop, and all three together are what an RTX 4080
    needs to fit the real model -- so they are exercised as one setting
    rather than three.
    """
    result = _overfit_run(
        device="cuda", precision="bf16", accumulate_steps=2,
        grad_clip_norm=1.0,
    )
    losses = result["losses"]
    assert torch.isfinite(losses).all()
    assert float(losses[-1]) < float(losses[0])


# ── #299: GQA / qk-norm / bias research knobs ───────────────────────────────


def test_default_n_kv_heads_keeps_the_pre_299_layout_and_count():
    fused = _build()["model"]
    explicit = _build(n_kv_heads=0)["model"]
    assert _build()["param_count"] == _build(n_kv_heads=0)["param_count"]
    # The MHA path keeps the fused qkv projection byte-for-byte.
    assert fused.blocks[0].attn.qkv is not None
    for a, b in zip(fused.parameters(), explicit.parameters()):
        assert torch.equal(a, b)


def test_gqa_param_count_matches_analytic_delta():
    mha = _build()["param_count"]
    gqa = _build(n_kv_heads=2)["param_count"]
    d = TINY["d_model"]
    head_dim = d // TINY["n_heads"]
    # Fused qkv (3d*d + 3d) becomes q (d*d + d) + kv (2*kvh*hd*d + 2*kvh*hd).
    per_block_delta = (3 * d * d + 3 * d) - (
        (d * d + d) + (2 * 2 * head_dim * d + 2 * 2 * head_dim))
    assert mha - gqa == TINY["n_layers"] * per_block_delta


def test_gqa_and_mqa_forward_backward_and_causality():
    for kv_heads in (2, 1):
        model = _build(n_kv_heads=kv_heads, positional="rope")["model"]
        ids = torch.randint(0, TINY["vocab_size"], (2, 12))
        logits = model(ids)
        assert logits.shape == (2, 12, TINY["vocab_size"])
        logits.sum().backward()
        # Causality: perturbing the future must not change the past.
        model.eval()
        base = torch.randint(0, TINY["vocab_size"], (1, 10))
        changed = base.clone()
        changed[0, 6:] = (changed[0, 6:] + 1) % TINY["vocab_size"]
        with torch.no_grad():
            assert torch.allclose(
                model(base)[0, :6], model(changed)[0, :6], atol=1e-5)


def test_invalid_n_kv_heads_names_both_params():
    with pytest.raises(ValueError, match=r"n_kv_heads=3.*n_heads=4"):
        _build(n_kv_heads=3)


def test_qk_norm_adds_per_head_gains_and_trains():
    plain = _build()["param_count"]
    normed_result = _build(qk_norm=True)
    head_dim = TINY["d_model"] // TINY["n_heads"]
    assert normed_result["param_count"] - plain == (
        TINY["n_layers"] * 2 * head_dim)
    model = normed_result["model"]
    ids = torch.randint(0, TINY["vocab_size"], (2, 8))
    model(ids).sum().backward()
    assert model.blocks[0].attn.q_norm.weight.grad is not None


def test_bias_false_removes_attention_and_mlp_biases():
    model = _build(bias=False)["model"]
    for name, _ in model.named_parameters():
        assert not (
            name.endswith(".bias")
            and (".attn." in name or ".mlp." in name)
        ), f"unexpected bias parameter: {name}"
    # Analytic: per block the fused qkv (3d) + out_proj (d) + fc1 (d_ff)
    # + fc2 (d) biases disappear.
    d, ff = TINY["d_model"], TINY["d_ff"]
    expected_delta = TINY["n_layers"] * (3 * d + d + ff + d)
    assert _build()["param_count"] - _build(bias=False)["param_count"] == expected_delta


def test_knobs_are_seed_deterministic():
    kwargs = {"n_kv_heads": 2, "qk_norm": True, "bias": False}
    first = _build(**kwargs)["model"]
    second = _build(**kwargs)["model"]
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_new_knobs_are_structural_params():
    for name in ("n_kv_heads", "qk_norm", "bias"):
        assert name in CausalLMModelNode.structural_params
