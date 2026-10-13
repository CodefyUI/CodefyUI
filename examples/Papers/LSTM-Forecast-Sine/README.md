# LSTM forecasts a noisy wave

Sepp Hochreiter and Jürgen Schmidhuber. *Long short-term memory.* Neural Computation 9(8), 1997.

Claim: gated memory cells let a recurrent network carry information across many time steps; here that shows as forecasting a noisy signal from its recent past.

## Recipe

| | |
|---|---|
| Data | sin(2πt/50) + 0.5 sin(2πt/13) + Gaussian noise (std 0.15), 2000 steps, windows of 30 values predicting the next; first 70% of time to train, last 30% to test |
| Model | LSTM (input 1, hidden 32), SelectIndex (last step), Linear (32 -> 1) |
| Loss / optimizer | MSE, Adam lr 0.005, batch size 32 |
| Epochs | 20 |
| Baseline | repeat the last value of the window |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11, run seed 0 (the graph's `settings.seed`).

| | R² on 591 held-out windows |
|---|---|
| LSTM | 0.949 |
| Repeat the last value | 0.873 |
| Ceiling set by the noise | 0.966 |

Whole graph: 1.9 s.

## Scale

The 1997 paper's experiments were synthetic long-lag memory tasks over hundreds to a thousand steps; this is a 30-step forecasting task, the setting LSTMs are most often used in today.
