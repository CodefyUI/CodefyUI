---
name: run-experiments
description: Design and read CodefyUI experiments - parameter searches, ablations and multi-seed comparisons with sweeps and studies. Covers choosing one metric and a baseline, varying one factor at a time, how many repetitions to run, reading means and intervals, and which conclusions the evidence cannot support. Use when the user asks to compare settings, tune hyperparameters, run an ablation, or decide whether a change helped.
license: AGPL-3.0-only
metadata:
  codefyui-skill-version: "1"
---

# Run experiments and read them honestly

An experiment here is the same graph run several ways and ranked on one number. The
machinery is easy; the design and the reading are where results go wrong.

## 1. Design before running

1. **Write the hypothesis** so it can fail: "AdamW with `weight_decay` 0.1 lowers
   `val_loss` against `weight_decay` 0 on this graph."
2. **Pick ONE metric and its direction.** Use a series the graph records: `val_loss`
   (minimize), `val_accuracy` or `eval_accuracy` (maximize), `PerplexityEvaluate`'s
   `val_loss`, `success_rate`, `mse`, `r2`, ... A held-out metric answers questions about
   generalisation; training loss does not. If two nodes record the same name, name the
   node too (`objective.node_id`).
3. **Keep an explicit baseline**: the graph unchanged, as one of the variants.
4. **Change one factor per variant.** Two changes in one variant cannot say which one
   mattered. Interactions are a second experiment, after the single factors.
5. **Fix the data.** Same split, same `seed` on `DatasetSplit`, same `max_tokens`, and a
   test set nobody tunes against.
6. **Count the cost first.** variants x repetitions runs, each as long as one training run.
   Tell the user the total before starting, and start only when they agree.

## 2. Choose the tool

| Need | Use |
| --- | --- |
| A grid or random search over node params | A **sweep**: `POST /api/sweeps` with `sweep_spec` (`grid` or `random` with a `seed`) and an `objective` |
| Named variants (param changes or whole graphs), repeated with a new seed each time, that survive closing the tab | A **study**: `POST /api/studies` with `variants`, `repetitions`, `seed`, `budget` |
| A quick comparison from an agent in the editor | The agent's own experiment tools, within their run budget |

Studies give every repetition its own seed, so repetitions are independent samples. A
tool that re-runs the identical graph without changing a seed measures only the
nondeterminism of the hardware, not the variance across seeds: do not call those
repetitions independent seeds. Details: [tools](references/tools.md).

## 3. Repetitions

- **1 run per variant** is a pilot: it can rule out a broken idea, not rank close ones.
- **3 repetitions** show obvious instability; **5** is still a small sample.
- Decide how many from the spread you see and the size of difference that matters. A
  difference smaller than the run-to-run spread of the baseline is not a finding.

## 4. Reading the numbers

- Report **mean, spread and n** for each variant, with the metric's exact name and
  direction. Quote what the runs recorded; never round toward the conclusion.
- A **95% interval** (mean +/- t x sd / sqrt(n)) describes where the mean plausibly is.
  Overlapping intervals mean the ranking is not settled; non-overlapping ones from three
  runs are still screening evidence, not a test.
- An **effect size** against the baseline (difference over pooled spread, such as
  Hedges' g) says how large the change is relative to the noise.
- **Ties are ties.** Exactly equal means are reported as a tie, not broken by runtime.
- **Failed runs are results.** Report how many failed and why; a variant whose runs did
  not all finish and record the metric cannot be ranked.

## 5. What the evidence cannot support

Do not claim, from a sweep or study alone:

- "significantly better" - no hypothesis test or multiple-comparison correction was run;
- "proven" or "always better" - one dataset, one graph, a handful of seeds;
- "novel" - nothing here searched the literature;
- that the winner generalises - unless it was re-checked on held-out data or a second
  dataset.

Say instead what was measured, on what, how many times, with what spread, and what the
next experiment should be (more seeds, a second dataset, the interaction of two winners).
More in [interpreting results](references/interpreting.md).
