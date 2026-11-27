# Training a causal language model

The shipped example is `examples/LLM/TrainCausalLM-TinyStories`. Its wiring:

```
Start --trigger--> TextCorpusDataset(train), TextCorpusDataset(validation),
                   LMTokenizer, CausalLMModel, LMCrossEntropyLoss

TextCorpusDataset(train).dataset --> LMTokenizedDataset(train).dataset
LMTokenizer.tokenizer            --> LMTokenizedDataset(train).tokenizer
TextCorpusDataset(val).dataset   --> LMTokenizedDataset(val).dataset
LMTokenizer.tokenizer            --> LMTokenizedDataset(val).tokenizer
LMTokenizedDataset(train).dataset --> DataLoader --> TrainingLoop.dataloader
CausalLMModel.model --> Optimizer.model, TrainingLoop.model
LMCrossEntropyLoss.loss_fn --> TrainingLoop.loss_fn
TrainingLoop.model --> PerplexityEvaluate.model
LMTokenizedDataset(val).dataset --> PerplexityEvaluate.dataset
TrainingLoop.model, LMTokenizer.tokenizer --> TextGenerate
```

Rules that are easy to get wrong:

- `CausalLMModel.vocab_size` must equal the tokenizer's vocabulary: 50257 for the `gpt2`
  encoding (the default). `LMTokenizer` also outputs `vocab_size` to read it from.
- `LMTokenizedDataset.seq_len` must not exceed `CausalLMModel.max_seq_len`.
- `LMTokenizedDataset.max_tokens` caps how much text is tokenized (0 = all). Cap the
  validation set too, so every comparison scores the same fixed held-out tokens.
- The headline metric is `PerplexityEvaluate.val_loss` (lower is better; `perplexity`
  is its exponential). Training loss on the training stream is not evidence of quality.

## Sizing for research loops

The example is a 200M-parameter model (`d_model` 1024, `n_layers` 12) meant for a GPU and
hours. For a quick comparison of training choices, scale down: `d_model` 128-384,
`n_layers` 2-6, `n_heads` dividing `d_model`, `d_ff` about 4x `d_model`, and a
`max_tokens` of a few million with `TrainingLoop.max_steps` set to match. Vary one knob
per variant: the optimizer's `lr`, an architecture switch (`positional`: learned,
sinusoidal or rope; `norm`: layernorm or rmsnorm; `activation`; `n_kv_heads`;
`qk_norm`), the data mix (`DataMixDataset`), or `CausalLMModel.seed` for seed replicates.

Useful `TrainingLoop` settings for LMs: `precision` `bf16`, `grad_clip_norm` 1.0,
`accumulate_steps` for a larger effective batch, `batch_metrics` on to record
`train_loss_batch`, `scheduler_step` `optimizer_step` with a `warmup_cosine`
`LRScheduler`, and `val_every_steps` with a validation loader for `val_loss_step`.
