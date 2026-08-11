# Train a Causal LM on TinyStories

Pretrain a decoder-only language model from scratch, entirely from GUI nodes. No
Python is written anywhere in this example and none executes as part of it — the
architecture is `CausalLMModel`'s parameters and everything else is wiring.

This is the graph epic [#292] asks for: the seven LLM nodes ([#289], [#290],
[#291]) assembled into a run that starts at raw text and ends with a model
writing sentences.

The training spine is one straight line:

```
TextCorpusDataset ──> LMTokenizedDataset ──> DataLoader ──> TrainingLoop
   (train split)         (1024-token blocks)
```

Four things join it from the side:

- `LMTokenizer` supplies the tokenizer object to **both** `LMTokenizedDataset`
  nodes and to `TextGenerate`.
- `CausalLMModel` feeds `TrainingLoop.model` and `Optimizer.model`; `Optimizer`
  then feeds both `TrainingLoop.optimizer` and `CheckpointSaver.optimizer` —
  the saver needs the optimizer state as well as the weights to write a
  checkpoint a later run can resume from.
- `LMCrossEntropyLoss` feeds `TrainingLoop.loss_fn`.
- A second `TextCorpusDataset` (`validation` split) goes through its own
  `LMTokenizedDataset` into `PerplexityEvaluate.dataset`.

`TrainingLoop`'s two outputs then fan out five ways: `model` to
`CheckpointSaver`, `PerplexityEvaluate` and `TextGenerate`; `losses` to
`CheckpointSaver` and `Visualize`. `PerplexityEvaluate.perplexity` and
`TextGenerate.text` each land in a `GraphOutput`, so both are readable from a
headless run.

## Before you run it

| | |
|---|---|
| GPU | 16 GB is the target (an RTX 4080 was the reference card). Less than that: turn on `gradient_checkpointing` in `CausalLMModel`, or drop `DataLoader.batch_size` to 4 and raise `TrainingLoop.accumulate_steps` to 8 to keep the same effective batch. |
| Network | Needed **once**. The first run downloads the TinyStories corpus from the Hugging Face Hub and tiktoken's gpt2 BPE ranks. Both are cached on disk, so every later run is offline. |
| Disk | The corpus cache, plus this example's own packed-block cache under `<data>/cache/lm_blocks/`. `LMTokenizedDataset` writes the packed tensor once and reloads it, so only the first run pays the tokenisation. |
| Time | One epoch, on the capped corpus, is 2,441 micro-batches / about 610 optimizer steps — well under an hour on the reference card. Uncapped it is days; see [Token budgets](#token-budgets). |

## Recipe

| | |
|---|---|
| Model | `CausalLMModel` at its declared defaults: pre-LN decoder, 12 blocks, `d_model` 1024, 16 heads, `d_ff` 4096, learned positions, LayerNorm, GELU, tied embeddings, context 1024 — **203,668,480 parameters** |
| Tokenizer | `LMTokenizer`, `gpt2` encoding (50,257 ids). One node feeds both packers **and** the generator |
| Corpus | `roneneldan/TinyStories`, `text` column, `train` split for training and `validation` for scoring |
| Packing | `seq_len` 1024, `append_eos` on. Documents are concatenated into one stream and cut into blocks whose labels are the inputs shifted by one token |
| Loss | `LMCrossEntropyLoss`, `ignore_index` -100, no label smoothing |
| Optimizer | AdamW, lr 3e-4, betas 0.9, 0.95, weight decay 0.1 |
| Batch | 8 sequences x 4 accumulation steps = **effective batch 32**, i.e. 32,768 tokens per optimizer step |
| Precision | bf16 autocast, gradients clipped at global norm 1.0 |
| Epochs | 1 |
| Scoring | `PerplexityEvaluate`, batch 8, bf16, over the whole capped validation split (1,953 blocks) |
| Generation | `TextGenerate`, prompt `Once upon a time`, 200 new tokens, temperature 0.8, top-k 50, top-p 0.95, seed 1234 |
| Checkpoint | `tinystories-lm/causal_lm_204m.pt` under the models directory |

`betas 0.9, 0.95` is the one value here that is not the `Optimizer` node's own
default. 0.999 is the vision-training beta2; LM pretraining has used 0.95 since
GPT-2, and a shorter beta2 memory is what keeps Adam stable when the gradient
scale moves quickly early in a run.

### Token budgets

`LMTokenizedDataset.max_tokens` caps the packed stream: **20,000,000** tokens
for training, **2,000,000** for validation. Both are deliberate, and they are
the first thing to change for a real run — `0` means "all of it".

Uncapped, TinyStories `train` is a few hundred million tokens, and one epoch of
it through a 204M-parameter model is days of consumer GPU. That is not an
example anybody runs. Capped, the arithmetic is:

| | |
|---|---|
| Stream | 20,000,000 tokens |
| Blocks | `(20,000,000 - 1) // 1024` = **19,531** — the remainder that cannot fill a block is dropped, not padded, because `DataLoader` collates with torch's `default_collate` and a ragged sample would fail to batch |
| Micro-batches | `19,531 // 8` = **2,441** (`drop_last` is on, so every accumulation window is full) |
| Optimizer steps | `2,441 // 4` = **610** |
| Tokens seen | 610 x 32,768 = about 20M, i.e. one pass |

610 steps is a *very* short pretraining run. Expect the loss to fall steeply and
the samples to be recognisably English with shaky plots — TinyStories is chosen
precisely because a small model gets somewhere on it fast. Raise `max_tokens`
(and `epochs`) to go further.

## Reading the result

Three surfaces, all wired:

- **`Visualize`** plots the training loss. `batch_metrics` is on, so it updates
  per batch rather than once per epoch — with a single epoch that is the only
  way to see a curve at all.
- **`PerplexityEvaluate`** reports `val_loss`, `perplexity` and the number of
  tokens scored, into a `GraphOutput` named `perplexity`. Perplexity is "the
  model was choosing between about this many equally likely tokens each step",
  so 50,257 is a uniform guess and anything well below it is learning. The
  number is only comparable against another run on **this** dataset with **this**
  tokenizer.
- **`TextGenerate`** continues `Once upon a time` into a `GraphOutput` named
  `sample`. This is the one that tells you whether the run was worth it.

Scoring happens on the `validation` split, packed by its own
`LMTokenizedDataset` node from its own `TextCorpusDataset`, so the perplexity is
a held-out number. Nothing selects on it — `early_stopping_patience` is 0 and
the checkpoint is the final model.

## Notes for anyone editing this graph

- **Every root node needs its own trigger edge from `Start`.** Execution walks
  forward from the entry points along *data* edges, so a node with no incoming
  data edge is pruned without one. This graph has exactly five such nodes and
  wires all five: `tok` (`LMTokenizer`), `ds-train` and `ds-val`
  (`TextCorpusDataset`), `model` (`CausalLMModel`) and `loss`
  (`LMCrossEntropyLoss`). Leaving a trigger out produces a missing-input error
  naming a node much further downstream.
- **`CausalLMModel.vocab_size` must match the tokenizer.** 50,257 is gpt2's. A
  model narrower than the tokenizer crashes in `nn.Embedding` on the first
  batch that contains a high id; a wider one silently wastes parameters.
- **`LMTokenizedDataset.seq_len` must not exceed `CausalLMModel.max_seq_len`.**
  A longer block is *rejected* rather than truncated, so the run dies on batch
  one. Both are 1024 here.
- **`TextCorpusDataset` cannot feed `DataLoader` directly.** It yields plain
  strings, not `(data, target)` pairs — a corpus has no targets until it is
  tokenised. `LMTokenizedDataset` is what produces the
  `(input_ids, labels)` 2-tuple `TrainingLoop` unpacks.
- **`LMCrossEntropyLoss`, not `Loss`.** The generic node's `CrossEntropyLoss`
  expects `[B, C]` logits against `[B]` labels; a language model produces
  `[B, T, V]`. It is also deliberately not a *subclass* of
  `nn.CrossEntropyLoss`, which is what keeps `TrainingLoop` from reporting a
  meaningless token-level accuracy off `argmax(dim=1)` over the time axis.
- **One tokenizer node, three consumers.** Both packers and `TextGenerate` read
  the same `LMTokenizer`. A second tokenizer with a different encoding would
  give the same integer two different meanings, and no port would disagree.
- **`num_workers` is 0.** The packed blocks are already a tensor in memory;
  indexing it is trivial, so workers would only add per-process copies of it.

## Continuous integration

Three tests in `backend/tests/test_builtin_examples.py` guard this example, plus
the two suite-wide sweeps that pick it up by glob
(`test_builtin_graph_executes` validates it structurally and skips execution
because it downloads and trains; `test_codegen.py` checks it still exports to
compilable Python).

`test_tinystories_lm_example_warns_inside_the_card_truncation` asserts the
requirements at the top of this file still fit inside the **80 characters** the
canvas gallery card shows. It truncates there and offers no tooltip, so
anything past it is invisible on the surface a user reads before pressing Run —
which makes *where* the warning sits in the description an invariant, not a
style preference. The same test caps the description's length and requires this
README to exist, since the card's last sentence points at it.

`test_tinystories_lm_example_still_describes_itself` asserts that every number
*this file* quotes is still derivable from the graph's params: the reference
shape, the effective batch, the tokens per optimizer step, both token budgets,
and the block / micro-batch / optimizer-step chain above. Those are computed
from `max_tokens`, `seq_len`, `batch_size`, `accumulate_steps` and `drop_last`
rather than transcribed, so editing any of the five fails the test until this
file is updated with it.

`test_tinystories_lm_example_scores_a_split_it_did_not_train_on` asserts the
perplexity node reads blocks packed from the `validation` split while the
dataloader trains on `train`, from different nodes, and that exactly one
`LMTokenizer` feeds exactly the three consumers named above. Both properties
type-check perfectly when broken, so neither is catchable any other way.

## Provenance

Unlike the [ResNet-18 / CIFAR-10
baseline](../../Usage_Example/ResNet18-CIFAR10-Baseline/README.md), this example
ships **no measured result**: the timings above are estimates from the step
count, not from a run, and there is no evidence directory. If you run it, the
loss curve, perplexity and wall clock belong here.

TinyStories is Eldan and Li, *TinyStories: How Small Can Language Models Be and
Still Speak Coherent English?* (2023). It is fetched at run time and not
redistributed here.

[#289]: https://github.com/CodefyUI/CodefyUI/issues/289
[#290]: https://github.com/CodefyUI/CodefyUI/issues/290
[#291]: https://github.com/CodefyUI/CodefyUI/issues/291
[#292]: https://github.com/CodefyUI/CodefyUI/issues/292
