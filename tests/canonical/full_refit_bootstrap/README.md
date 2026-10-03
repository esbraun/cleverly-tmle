# Full-refit bootstrap and derived contrasts

This directory freezes an independent repeated-sampling study of four post-fit outputs: the
log-scale ratio of two regimen levels, the RMST and its contrast, the `LTMLE` full-refit
bootstrap percentile interval, and the point-treatment `TMLE` bootstrap percentile interval. No
pinned comparator ships these outputs, so the study has no reference and `equivalence.csv` is
empty and schema-valid.

The primary table is the log risk ratio and log odds ratio of always versus never on the two-node
end-of-study law. Every other cell is an `interval_calibration` property cell. The cells, their
sizes, the 200 bootstrap replicates, the 4,000 property replicates, the 1% failed-replicate cap, the red-cell policy and the
budget are declared in `tests/studies/canonical_full_refit_bootstrap.py` and pinned by
`tests/unit/test_full_refit_bootstrap_cell_design.py` before the run.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.full_refit_bootstrap.regenerate
```
