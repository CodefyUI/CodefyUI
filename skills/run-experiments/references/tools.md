# Sweeps and studies

Both run on the server's run queue and keep going after the browser tab closes. Full
reference: the Run Queue and Studies pages of the CodefyUI docs.

## Sweep: `POST /api/sweeps`

```json
{
  "base_graph": {"nodes": [], "edges": []},
  "name": "lr x batch",
  "sweep_spec": {
    "method": "grid",
    "params": [
      {"node_id": "opt", "param": "lr",
       "range": {"min": 1e-4, "max": 1e-1, "count": 4, "scale": "log", "type": "float"}},
      {"node_id": "loader", "param": "batch_size", "values": [32, 64, 128]}
    ]
  },
  "objective": {"metric": "val_loss", "direction": "minimize"},
  "options": {"device": "cuda:0"}
}
```

- Each param names a node by id and a param, with `values` or a `range` (`count` points,
  `linear` or `log` scale, `type` `int` or `float`). Values are checked against the node's
  schema before anything runs.
- `method: grid` runs every combination; `method: random` needs `samples` and a `seed`
  (the same seed draws the same combinations). An oversized plan is refused, never
  truncated.
- The objective `metric` is a series name (`train_loss`, `val_loss`, `val_accuracy`,
  `eval_accuracy`, `accuracy`, `mse`, `r2`, `success_rate`, ...).

## Study: `POST /api/studies`

```json
{
  "name": "lr study",
  "objective": {"node_id": "train", "metric": "val_loss", "direction": "minimize"},
  "base_graph": {"nodes": [], "edges": []},
  "variants": [
    {"label": "baseline", "ops": []},
    {"label": "lr 1e-3", "ops": [{"node_id": "opt", "param": "lr", "value": 0.001}]}
  ],
  "repetitions": 3,
  "seed": 7,
  "budget": {"max_runs": 12, "max_parallel_runs": 1}
}
```

- A variant is `ops` applied to `base_graph` (`[]` is the baseline) or a whole `graph`.
- Every repetition runs with a different seed derived from `seed`, so repetitions are
  independent samples of the variant.
- `budget` stops the study before it uses more than allowed; the Studies panel (Runs tab
  -> Studies) follows it, ranks variants, and shows ties.
- A study survives a server restart and can be resumed without re-running completed
  repetitions.

## Agent experiment tools

An agent in the editor may have its own comparison tools that run candidate graphs
off-canvas within a fixed per-turn run budget and ask the user to confirm first. Follow the
same design rules. If such a tool re-runs the identical graph for each repetition, vary a
`seed` param (for example `CausalLMModel.seed`) in separate variants when seed variation is
what you want to measure, or submit a study instead.

## Concurrency

Run one at a time (`max_parallel_runs` 1) when runtime is part of the comparison or when
runs share a GPU: contention changes both speed and, on some kernels, the numbers.
