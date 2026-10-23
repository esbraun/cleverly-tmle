# E-value limit coverage (X82 SG)

This directory holds a repeated-sampling study of the risk-ratio intervals that two part G
E-value limits map. An E-value limit is a monotone map of its risk-ratio limit, so the coverage
of that interval is the claim. `X82-SG-plan.md` rev 2 in the plans repository declares the
design. A passing row licenses its `evalue_limit` rule. A red row leaves the rule descriptive.

| cell | `evalue_limit` rule | law and fit | rows |
| --- | --- | --- | --- |
| E3 | `stratum` | `tests.studies.stratified_law`, fit as `canonical-stratified-tmle` fits, estimands `rr` | `rr[V=0]`, `rr[V=1]`, `rr[V=2]` |
| E4 | `time_to_event` | the `survival` law of `point-treatment-survival`, fit as `point-treatment-survival-hazard` fits | `risk_regimen[arm1 @ t=5] / risk_regimen[arm0 @ t=5]`, from `result.ratio(..., kind="rr")` |

Each row reads the reported interval on the log scale. Each replication draws `n = 2000` rows.
Each cell has 1,600 primary replications.

| property family | cells | rule |
| --- | --- | --- |
| `interval_calibration` | `correctly_specified` for each row | SE ratio and coverage inside the calibration bands |
| `interval_calibration` | `inflated_se_control` for each row (SE x 1.5) | SE ratio above the band |

The property family reads 4,000 draws per cell. `cells.csv` records the fit configuration of
each row, read off one fit per cell.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.evalue_limit_coverage.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/evalue_limit_coverage.py` and
`tests/studies/evalue_limit_coverage_properties.py`. The design test is
`tests/unit/test_sensitivity_limit_coverage_design.py`.
