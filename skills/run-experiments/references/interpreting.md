# Interpreting a comparison

## The arithmetic

For a variant with n successful runs x1..xn of the metric:

- mean m = (x1 + ... + xn) / n
- sample standard deviation s = sqrt( sum (xi - m)^2 / (n - 1) )   (needs n >= 2)
- 95% interval: m +/- t(0.975, n-1) * s / sqrt(n), with t = 12.71 (n=2), 4.30 (n=3),
  3.18 (n=4), 2.78 (n=5)
- difference against the baseline, direction-adjusted: (m_variant - m_baseline) for a
  metric to maximize, (m_baseline - m_variant) for one to minimize, so positive is better
- Hedges' g: the difference over the pooled standard deviation, times the small-sample
  correction 1 - 3 / (4 (n1 + n2) - 9)

With n = 3 the t value is 4.30: intervals are wide on purpose. That is the honest width of
three runs.

## Reading levels

| Level | Question | What it takes |
| --- | --- | --- |
| Observation | Did one variant rank higher in this pilot? | The metric, the baseline, every value, the failures |
| Mechanism | Which component caused it? | One-factor ablations, checks for confounders |
| Robustness | Does it hold up? | More seeds, data regimes, model sizes, a second dataset |
| Contribution | Is it new and meaningful? | Literature, strong baselines, proper statistics, reproducible artifacts |

A sweep or study reaches the first level, sometimes the second. Say which level a
conclusion is at.

## Common traps

- **Tuning on the test set.** Selecting the best of many variants on the same split you
  report inflates the winner. Select on validation, report once on test.
- **Many comparisons.** With 16 variants, one will look best by chance. The more variants,
  the more repetitions the winner needs before it is believed.
- **Confounded variants.** Changing `lr` and `batch_size` together, or a schedule whose
  length follows `epochs`, changes two things.
- **Shared weights.** Two training nodes fed by one model output train the same weights
  one after another (`shared_model_training` warning). Each arm needs its own model node.
- **Different budgets.** A variant that trains for more steps is not comparable on
  quality alone; fix `epochs` or `max_steps`, or compare at equal compute.
