# Gradient clipping rescues an RNN

Razvan Pascanu, Tomas Mikolov and Yoshua Bengio. *On the difficulty of training recurrent neural networks.* ICML 2013.

Claim: a recurrent network's gradient can explode, a single huge update throws away the training so far, and rescaling the gradient whenever its norm passes a threshold fixes it.

## Recipe

| | |
|---|---|
| Data | `SyntheticSequence` `recall_first`, seq_len 30, 10 classes + 10 distractor tokens; 6000 training sequences (seed 0), 500 test sequences (seed 1) |
| Model | Embedding (20 -> 16), RNN (tanh, hidden 64), SelectIndex (last step), Linear (64 -> 10); two `SequentialModel` nodes with the same layers |
| Optimizer | SGD, lr 0.1, batch size 32 |
| Epochs | 25 per run |
| Difference | `TrainingLoop.grad_clip_norm` 0 against 1.0; `log_grad_norm` on for both |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11, run seed 0 (the graph's `settings.seed`).

| | No clipping | Clip at 1.0 |
|---|---|---|
| Test accuracy (chance 0.10) | 0.106 | 0.886 (1.0 on the Linux CI runner) |
| Final training loss | 2.305 | 0.147 |
| Largest pre-clip gradient norm | 117 | 459 |

The unclipped run's loss was creeping down from ln 10 (2.30) when a gradient of norm 117 late in epoch 13 knocked it back up to 2.35 in epoch 14; it never recovered. The clipped run saw even larger raw gradients, but every applied step was capped at norm 1, and its loss left the plateau at epoch 16. The `grad_norm` and `grad_norm_clipped` series are in the Runs panel under Metrics. Whole graph: 11.0 s.

This run sits near the edge of divergence, so float rounding on another CPU moves its numbers: on the Linux CI runner the clipped net reaches 1.0, and the unclipped net stays at chance on both. Over run seeds 0 to 4 on the M3 the clipped run reached 0.71 to 0.91 and the unclipped one 0.08 to 0.50, so the gap holds beyond seed 0.

## Scale

The paper's experiments ran sequences of 50 to 200 steps for many thousands of updates on several synthetic problems; this is one problem at 30 steps and 25 epochs.
