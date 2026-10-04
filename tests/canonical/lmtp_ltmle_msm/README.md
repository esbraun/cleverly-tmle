# Cross-fitted longitudinal MSM projection evidence

This fixture holds the `cross-fitted-longitudinal-msm` study. It compares the cross-fitted
`cleverly` longitudinal MSM projection with a fixed projection of four pinned R `lmtp` 1.5.4
regimen fits. Both sides use the same panels, the same exact per-node density ratios and the
same realized folds. The law, the regimens and the projection are those of `longitudinal-msm`.

`lmtp` has no MSM. The runner `run_study.R` fits each regimen with `lmtp_tmle_with_folds` and
projects the four estimates and their joint per-unit influence curves by the declared weighted
least-squares operator. On every replication it checks that the operator equals an independent
`lm.wfit` of the four estimates to 1e-10. The two constructions differ twice: `lmtp` fluctuates
on each training fold, and it targets each regimen with its own scalar fluctuation.

Run a disposable smoke before the full study.

```console
python -m tests.canonical.lmtp_ltmle_msm.regenerate --replicates 20 --primary-only --jobs 1 --output .tmp/x27-smoke
```

Run the declared study once, after every source change is final.

```console
python -m tests.canonical.lmtp_ltmle_msm.regenerate
python -m tests.studies.evidence.document --slug cross-fitted-longitudinal-msm
```

The full run fits 800 primary replications and the property families that
`tests/studies/crossfit_longitudinal_msm_properties.py` declares. The policy is gated. The study
module states the declared red-cell route and the failure rule.
