# Weight decay smooths the fit

Anders Krogh and John A. Hertz. *A simple weight decay can improve generalization.* NeurIPS 4, 1992.

Claim: penalising large weights stops a network from fitting the noise in a small training set, which lowers its error on new inputs.

## Recipe

| | |
|---|---|
| Data | 20 points of sin(3x) + Gaussian noise (std 0.3), x uniform on [-1, 1]; test on the clean sin(3x) at 200 grid points |
| Model | MLP 1-128-128-1 with Tanh; two `SequentialModel` nodes with the same layers |
| Loss / optimizer | MSE, AdamW lr 0.01, full batch |
| Epochs | 1000 per run |
| Difference | `weight_decay` 0 against 0.5 |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11, run seed 0 (the graph's `settings.seed`).

| | weight_decay 0 | weight_decay 0.5 |
|---|---|---|
| Final training MSE | 0.0012 | 0.056 |
| Test MSE against the clean curve | 0.269 | 0.033 |

Whole graph: 1.4 s. Over run seeds 0 to 2 the decayed net scored 0.033 to 0.042 and the undecayed one 0.14 to 0.27.

## Scale

The paper analysed linear networks and small nonlinear ones with plain L2-penalised gradient descent; this uses AdamW's decoupled decay on a 17K-parameter MLP.
