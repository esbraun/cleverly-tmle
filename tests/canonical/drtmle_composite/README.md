# Observational missing-data DR-TMLE evidence artifacts

This directory holds the registered study of the composite-indicator construction. Three
scenarios use laws of `tests/studies/mar_arm_indexed_laws.py`:

| scenario | arms | missing |
| --- | --- | --- |
| `binary_observational_mar` | 2 | the outcome |
| `binary_mar_outcome_and_treatment` | 2 | the outcome and the treatment |
| `three_arm_mar_outcome_and_treatment` | 3 | the outcome and the treatment |

The treatment is observational in each scenario. The fit estimates its mechanism and receives
no known probabilities.

The comparator is pinned R `drtmle` 1.1.2. It implements the same construction: an `NA`
treatment gives `DeltaA = 0`, and the fluctuation indicator is `A == a & DeltaA == 1 & DeltaY
== 1`. The runner passes the oracle outcome regressions and the oracle composite mechanism,
so `drtmle` skips `estimateG`. `out$drtmle` pairs with the composite DR-TMLE. `out$tmle` pairs
with the composite TMLE under `tmle_ate` and `tmle_ate_mid`. On a binary outcome both
conventions for `Y` agree: the runner passes `NA` where `DeltaY = 0`, and `fluctuateQ1` scales
by the observed values.

Run a disposable smoke study into a scratch directory outside the repository:

```console
python -m tests.canonical.drtmle_composite.regenerate --replicates 8 --output <scratch> --skip-properties --allow-failures
```

A declared run passes `--output <scratch>` and optionally `--jobs`, and nothing else.
`tests/canonical/declared_run.py` describes the guard, the run log and the copy into this
directory.
