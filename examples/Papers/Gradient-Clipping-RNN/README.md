# Gradient clipping rescues an RNN

Razvan Pascanu, Tomas Mikolov and Yoshua Bengio. *On the difficulty of training recurrent neural networks.* ICML 2013.

Claim: a recurrent network's gradient can explode, a single huge update throws away the training so far, and rescaling the gradient whenever its norm passes a threshold fixes it.

## Recipe

| | |
|---|---|
| Data | `SyntheticSequence` `recall_first`, seq_len 30, 10 classes + 10 distractor tokens; 6000 training sequences (seed 0), 500 test sequences (seed 1) |
| Model | Embedding (20 -> 16), RNN (tanh, hidden 256), SelectIndex (last step), Linear (256 -> 10); two `SequentialModel` nodes with the same layers |
| Optimizer | SGD, lr 0.2, batch size 64 |
| Epochs | 25 per run |
| Difference | `TrainingLoop.grad_clip_norm` 0 against 1.0; `log_grad_norm` on for both |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11.

Run seed 0 (the graph's `settings.seed`):

| | No clipping | Clip at 1.0 |
|---|---|---|
| Test accuracy (chance 0.10) | 0.150 | 1.000 |
| Final training loss (guessing: 2.30) | 6.06 | 0.000 |
| Largest pre-clip gradient norm | 134 | 40 |

Both runs creep along the ln 10 plateau for about 9 epochs. From epoch 10 a burst of large gradients (the largest, of norm 134, in epoch 11) throws the unclipped run's loss to 5.4, and it stays above the level of guessing to the end. The clipped run leaves the plateau around epoch 11 and reaches a loss near 0. The `grad_norm` and `grad_norm_clipped` series are in the Runs panel under Metrics. Whole graph: about 15 s.

Run seeds 0 to 7, same graph:

| Seed | No clipping: accuracy | No clipping: final loss | Clip at 1.0: accuracy |
|---|---|---|---|
| 0 | 0.150 | 6.06 | 1.000 |
| 1 | 0.334 | 7.46 | 1.000 |
| 2 | 0.318 | 4.33 | 1.000 |
| 3 | 0.324 | 3.98 | 1.000 |
| 4 | 0.172 | 6.24 | 1.000 |
| 5 | 0.354 | 3.62 | 1.000 |
| 6 | 0.246 | 5.31 | 0.912 |
| 7 | 0.100 | 11.65 | 1.000 |

The unclipped net's loss ends far above guessing on every seed, while its accuracy varies (0.10 to 0.35): a diverged network is wildly overconfident, which the loss punishes and the argmax does not. The run sits close to divergence by design, so float rounding on another CPU moves the individual numbers; the test pins the bands (clipped at least 0.85, unclipped loss above 3.0 and accuracy at most 0.45).

## Scale

The paper's experiments ran sequences of 50 to 200 steps for many thousands of updates on several synthetic problems; this is one problem at 30 steps and 25 epochs.
