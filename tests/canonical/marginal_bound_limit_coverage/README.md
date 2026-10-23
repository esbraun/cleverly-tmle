# Marginal sensitivity bound limit coverage (X82 SH)

This directory holds a repeated-sampling study of the confidence limits of
`cleverly.sensitivity.marginal_sensitivity_bounds`. Each row reads one end of the ATE bound at
`Gamma = 1.5` as an estimate of the sharp end. `X82-H-plan.md` rev 2.4, section 8, in the plans
repository declares the design. A passing cell licenses its `msm_bound_limit` rule. A red cell
leaves the rule descriptive.

| cell | `msm_bound_limit` rule | law (`tests.studies.marginal_bound_law`) | rows |
| --- | --- | --- | --- |
| SH1 | `continuous` | Gaussian linear outcome, 20 covariates | `ate_lower`, `ate_upper` |
| SH2 | `binary` | logistic outcome, prevalence near 0.33, 20 covariates | `ate_lower`, `ate_upper` |

Each fit is a 5-fold cross-fitted `TMLE` with correctly specified GLM nuisances. SH1 calls the
bounds with the default quantile and tail learners. Each truth is the sharp set of the law, by
quadrature. Each replication draws `n = 8000` rows. Each cell has 1,600 primary replications.

| property family | cells | rule |
| --- | --- | --- |
| `interval_calibration` | `correctly_specified` for each end | SE ratio and coverage inside the calibration bands |
| `interval_calibration` | `inflated_se_control` for each end (SE x 1.5) | SE ratio above the band |
| `interval_calibration` | `wrong_gamma_control` for each end (the `Gamma = 1.25` end) | coverage below the band |

The property family reads 6,800 draws per cell. `cells.csv` records the fit configuration of
each row, read off one fit per cell.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.marginal_bound_limit_coverage.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/marginal_bound_coverage.py` and
`tests/studies/marginal_bound_coverage_properties.py`. The design test is
`tests/unit/test_marginal_bound_coverage_design.py`.
