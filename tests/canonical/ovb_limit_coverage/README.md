# Omitted-variable bound limit coverage (X82 SG)

This directory holds a repeated-sampling study of the one-sided confidence limits that X82 part G
added to `omitted_variable_bounds`. `X82-SG-plan.md` rev 2 in the plans repository declares the
design. A passing row licenses its `ovb_limit` rule. A red row leaves the rule descriptive.

| cell | `ovb_limit` rule | law | parameters |
| --- | --- | --- | --- |
| O1 | `response` | the part G law, `Y` missing at random given `A, W` | `ate`, `ey1` |
| O2 | `intermediate` | the part G law with a binary `Z` that depends on `A` and `W` | `ate` at `Z = 0` and at `Z = 1` |
| O3 | `regime` | the part G law, complete data | `ey_regime[coin]`, `ate_regime[coin vs always]` |
| O4 | `msm` | the part G law, complete data | `msm[W2]`, `msm[a:W2]` of `MSM.linear(modifiers=("W2",))` |
| O5 | `shift` | `tests.discrete_law_shift` on its four dose atoms | `ey_policy[+1]`, `ate_policy[+1 vs natural course]` |

`tests/studies/sensitivity_limit_laws.py` states the part G law and the closed form of `psi`,
`sigma^2` and `nu^2` for each parameter. Each cell fixes `cf_y = 0.3`, `cf_d = 0.2` and `rho = 1`.
Each row reads one end of a bound as an estimate of its population value. The standard error
comes from the reported one-sided limit: `(lower - ci_lower) / z_0.95`, and the mirror for the
upper end.

Each replication draws `n = 2000` rows. Every nuisance is a logistic regression on all pairwise
products of its inputs, which is saturated on these laws. Each fit is in sample. Each cell has
1,600 primary replications.

| property family | cells | rule |
| --- | --- | --- |
| `interval_calibration` | `correctly_specified` for each row | SE ratio and coverage inside the calibration bands |
| `interval_calibration` | `inflated_se_control` for each row (SE x 1.5) | SE ratio above the band |
| `response_mask_necessity` | O1 rows, `masked` against `unmasked_sigma2` | displacement of at least 0.25 positive-arm SDs |

The property families read 4,000 draws per cell. `cells.csv` records the fit configuration of
each row, read off one fit per cell. The MSM Gram term has no control: removing it moves the limit
SE by at most 4% up to `n = 40,000`. Its witness is the Gateaux unit test in
`tests/unit/test_omitted_variable_axes.py`.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.ovb_limit_coverage.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/ovb_limit_coverage.py` and
`tests/studies/ovb_limit_coverage_properties.py`. The design test is
`tests/unit/test_sensitivity_limit_coverage_design.py`.
