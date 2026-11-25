# Sampled refutation

This directory holds a repeated-sampling study of `refute(result, sample_limit=m)`, which draws
`m` of a fit's `n` independent units without replacement, refits the stored estimator on them,
and runs each test on that sampled baseline. Under i.i.d. units its verdict has the distribution
of the full-data verdict at `n = m`, whatever `m/n`. The study reads that claim by `m` and `m/n`,
i.i.d. and in whole clusters. The comparison plan of X106 part H (section 6) declares the design.

The law, the fit and the three rules are those of `refuter-calibration-point` cell T1: a
Gaussian outcome with 20 covariates, `TMLE` of the ATE with linear and logistic nuisances at five
folds, and the placebo `t`, subset `rho` and negative-control `z` rules over 20 refits, each
two-sided at 0.05. The clustered cells group rows in clusters of five with a cluster effect on
the treatment, the outcome and the controls, and a cluster effect modifier on the treatment
effect, and the fit declares the clusters.

| cell | units | `m` | `n` | trials | role |
| --- | --- | --- | --- | --- | --- |
| I2-F | subjects | 2,000 | 2,000 | 1,000 | full rule at `n = m` (registered T1) |
| I2-H | subjects | 2,000 | 4,000 | 1,000 | sampled |
| I2-T | subjects | 2,000 | 20,000 | 1,000 | sampled |
| I1-F | subjects | 1,000 | 1,000 | 1,000 | full rule below the registry's `n_min` |
| I1-T | subjects | 1,000 | 10,000 | 1,000 | sampled |
| C-F | clusters of 5 | 400 | 400 | 1,000 | full rule, clustered |
| C-H | clusters of 5 | 400 | 800 | 1,000 | sampled |
| C-T | clusters of 5 | 400 | 4,000 | 1,000 | sampled |
| P-4k | subjects | full | 4,000 | 50 | full rule at moderate `n` |
| P-20k | subjects | full | 20,000 | 50 | full rule at moderate `n` |

Trials 0 to 49 of every cell also run each defect at a strong and a weak level. The gates are
an absolute size count on the I2 cells (at most 73 rejections of 1,000), a relative size test of
each sampled cell against its full cell (the 90% interval of the difference of rejection rates
inside ±0.05), a power count on the I2 cells' strong defects (at least 38 of 50), and a mutation
control that recomputes the subset `rho` from the full fit's SE and unit count and must fail both
size gates. `tests/unit/test_refuter_sampled_design.py` checks the law, the defect sizes and the
gates' operating characteristics.

`trials.csv` keeps every trial's readings and printed verdicts, the sampled baseline beside the
full fit, and the trial's seeds. `reported.csv` holds the per-cell printed-verdict shares and the
reported rates. `cells.csv` records each cell's fit configuration.
`tests/unit/test_refuter_sampled_registry.py` recomputes every gated verdict and the verdict scope
from `trials.csv`.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

The declared run took about 52 minutes of wall time on a cloud runner.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.refuter_sampled.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/refuter_sampled.py` and
`tests/studies/refuter_sampled_properties.py`.
