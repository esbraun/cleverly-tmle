# Missing-outcome natural-course TMLE evidence

This directory freezes the independent repeated-sampling study for `ey_obs` under MAR. It covers
binary and bounded-continuous outcomes under the same observational binary-treatment law. The
response probabilities and binary outcome means are exact. The continuous law uses a correctly
specified affine outcome learner on the scaled outcome. A treatment learner that raises if called
verifies that the natural-course score does not use a treatment mechanism.

The property study uses the binary finite law to test the two-nuisance union model, three-size
root-n contraction and exact-EIF efficiency, interval calibration and two derived controls,
targeting necessity, and the consequence of silently switching to a complete-case analysis.

Díaz, Carone and van der Laan (2016), Section 2 and Equations (1)–(5) specify the estimator.

The comparator is the R `tmle` 2.1.1 population-mean path. `run_study.R` runs it on both
primary laws with the same supplied predictions that the Python fit used. The comparison
therefore conditions on those predictions. The runner uses these settings:

| setting | value |
| --- | --- |
| synthetic treatment | `A = rep(1, n)`, which selects the population-mean result |
| covariates | `W = data.frame(A_original, W)` |
| outcome predictions | `Q = cbind(qn, qn)`, the realized-arm outcome prediction twice |
| treatment and response | `g1W = rep(1, n)`, `pDelta1 = cbind(pin, pin)` |
| family | `binomial` for the binary law, `gaussian` for the continuous law |
| bounds | `Qbounds = c(0, 1)`, `gbound = 0.01`, `alpha = 0.9995` |

R takes the continuous outcome scale from every non-`NA` `Y`. The runner therefore sets `Y` to
0 and 1 on the first two rows whose response indicator is zero. The response indicator removes
both rows from the fluctuation and the curve. `probe_scale_workaround.R` checks this workaround
on every continuous replication and writes `scale-probe.csv`. A row passes when these five
conditions hold:

| check | condition |
| --- | --- |
| planted scale | the scale R takes is `c(0, 1)` exactly |
| unplanted scale | the scale without the planted rows differs |
| moved rows | two other planted rows give a bitwise equal point and curve |
| rebuild | an independent rebuild matches the point and curve to `1e-12` |
| witness | `unplanted_point_difference`, the point shift without the workaround, is above zero |

The driver refuses publication when a probe row fails or a continuous replication has no row.
`tests/unit/test_mar_natural_course_method_study.py` checks the per-replication parity.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course.regenerate --jobs 16 --reference-jobs 16
```

For a disposable primary smoke run with both R runners:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-natural-course-smoke --jobs 16 --reference-jobs 16
```
