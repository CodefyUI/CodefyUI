# Transformer learns to sort

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin. *Attention is all you need.* NeurIPS 2017.

Claim: self-attention alone, with no recurrence, lets every position read every other one in one step. Sorting needs exactly that: each output position depends on the whole input.

## Recipe

| | |
|---|---|
| Data | 5000 random sequences of 10 digits (0-9) and their sorted versions; 1000 fresh sequences to test |
| Model | Embedding (10 -> 64), sinusoidal PositionalEncoding, TransformerEncoder (2 layers, 4 heads, feed-forward 128), Linear (64 -> 10) |
| Loss / optimizer | `LMCrossEntropyLoss` over all 10 positions, Adam lr 0.001, batch size 64 |
| Epochs | 20 |
| Seed | 0 |

## Measured

Measured with `backend/run_graph.py` on an Apple M3 CPU, torch 2.11, run seed 0 (the graph's `settings.seed`).

| | 1000 fresh sequences |
|---|---|
| Sorted exactly right | 1.000 |
| Positions right | 1.000 |

Whole graph: 12.2 s. With 15 epochs it is 0.982 exact.

`TensorDataset` cannot carry `[N, 10]` integer targets (it stores number targets as float32, which `LMCrossEntropyLoss` refuses, and flattens class targets to one label per row), so the generating `PythonScript` returns a small dataset class directly.

## Scale

The paper trained 65M-parameter encoder-decoders on WMT translation; this is a 68K-parameter encoder on a toy task, the same sort demo minGPT uses.
