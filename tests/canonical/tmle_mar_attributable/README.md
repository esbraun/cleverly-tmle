# Missing-outcome attributable-effect TMLE evidence

This directory holds the repeated-sampling study for PAR and PAF when outcomes are missing at
random. The subject is the joint natural-course fit of ordinary TMLE. It solves the shipped
natural-course fluctuation and the shipped arm-mean fluctuation from one initial fit. It
reports both means and their contrasts from one stack.
`tests/studies/canonical_mar_attributable.py` declares the study.

| scenario | law | fit |
| --- | --- | --- |
| `binary_mar_attributable` | L1 | in sample, the law's own nuisances |
| `continuous_mar_attributable` | L1 tables, Beta outcome | in sample, `q_bounds=(0, 1)` |
| `three_arm_mar_attributable` | L3 | in sample, `reference="low"` |
| `binary_mar_attributable_cvtmle` | L1 | stacked, ten folds, depth-five trees |
| `three_arm_mar_attributable_cvtmle` | L3 | stacked, ten folds, depth-five trees |
| `binary_mar_attributable_weighted` | L1 | in sample, fixed weights 0.6, 1.0 and 1.8 by `W` |
| `binary_mar_attributable_clustered` | L1, 100 clusters of 20 rows | in sample, `id=` |

R `tmle` 2.1.1 reports no PAR or PAF. `run_study.R` composes them from two shipped paths. Both
paths use the predictions that the Python fit supplied, so the comparison conditions on them.

| path | settings |
| --- | --- |
| natural course | `A = rep(1, n)`, `W = data.frame(A_original, W)`, `Q = cbind(qn, qn)`, `g1W = rep(1, n)`, `pDelta1 = cbind(pin, pin)` |
| arm `a` | `A = rep(1, n)`, `Delta = 1{A = a} Delta`, `Q = cbind(q_a, q_a)`, `g1W = g_a`, `pDelta1 = cbind(pi_a, pi_a)` |
| both | `fluctuation = "logistic"`, `Qbounds = c(0, 1)`, `gbound = 0.01`, `alpha = 0.9995`, `id =`, `obsWeights =` |

The arm path's clever covariate is `1{A = a} Delta / (g_a pi_a)`, which is the arm-mean
covariate. The runner reads each point from `fit$estimates$EY1$psi` and each curve from
`fit$estimates$IC$IC.EY1`. It forms PAR as the difference of the two curves and PAF by the
delta method. R averages a curve within each `id` and divides `var(IC)` by the number of ids,
so the composed variance uses the same cluster means.

R takes the continuous outcome scale from every non-`NA` `Y`. On the continuous law the runner
sets `Y` to 0 and 1 on two rows that the path's response indicator excludes.
`probe_scale_workaround.R` checks this workaround on both paths of every continuous replication
and writes `scale-probe.csv`. A row passes when these five conditions hold:

| check | condition |
| --- | --- |
| planted scale | the scale R takes is `c(0, 1)` exactly |
| unplanted scale | the scale without the planted rows differs |
| moved rows | two other planted rows give a bitwise equal point and curve |
| rebuild | an independent rebuild matches the point and curve to `1e-12` |
| witness | the point without the workaround differs |

The container is the `tests/canonical/tmle_mar` image. Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_attributable.regenerate --jobs 16 --reference-jobs 16
```

For a disposable primary smoke run with both R runners:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_attributable.regenerate --replicates 4 --skip-properties --allow-failures --output build/tmle-mar-attributable-smoke --jobs 1 --reference-jobs 1
```
