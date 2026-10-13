# Early stopping on validation loss

Lutz Prechelt. *Early stopping, but when?* In G. B. Orr and K.-R. Müller (eds.), Neural Networks: Tricks of the Trade, LNCS 1524, Springer, 1998.

Claim: stopping when the validation error stops improving, and keeping the weights from the best epoch, gives a model that generalises better than training to the end.

## Recipe

| | |
|---|---|
| Data | 20 training and 20 validation points of sin(3x) + Gaussian noise (std 0.3); test on the clean sin(3x) at 200 grid points |
| Model | MLP 1-128-128-1 with Tanh; two `SequentialModel` nodes with the same layers |
| Loss / optimizer | MSE, Adam lr 0.01, full batch |
| Epochs | up to 1000 |
| Difference | `early_stopping_patience` 0 against 100, `monitor` val_loss |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11, run seed 0 (the graph's `settings.seed`).

| | Trained to epoch 1000 | Early stopping |
|---|---|---|
| Epochs run | 1000 | 221 |
| Weights kept | epoch 1000 | epoch 121 (best val_loss) |
| Test MSE against the clean curve | 0.095 | 0.033 |

Whole graph: 1.2 s. Over run seeds 0 to 2 the early-stopped model scored 0.03 to 0.04 and the fully trained one 0.07 to 0.20.

## Scale

Prechelt compared 14 stopping criteria over 1296 runs on 12 PROBEN1 problems; this is one criterion (patience) on one toy regression.
