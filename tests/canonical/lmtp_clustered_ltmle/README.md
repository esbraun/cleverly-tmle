# Clustered cross-fitted end-of-study LTMLE evidence

This fixture holds the `clustered-cross-fitted-ltmle` study. It compares `cleverly` with pinned
R `lmtp` 1.5.4 and `ife` 0.2.3 on the same clustered panels, the same exact per-node density
ratios and the same realized grouped folds. The law is declared in
`tests/studies/clustered_longitudinal_laws.py`. Each draw holds 100 clusters of 40 rows.

The runner `run_study.R` checks two facts on every fit. `lmtp` keeps the supplied fold
assignment and the cluster identifier. The `ife` standard error equals the cluster-sum
standard error of the `lmtp` row curve to 1e-10, which holds at equal cluster sizes.

Run a disposable smoke before the full study.

```console
python -m tests.canonical.lmtp_clustered_ltmle.regenerate --replicates 20 --primary-only --jobs 1 --output .tmp/clustered-ltmle-smoke
```

Run the declared study once, after every source change is final.

```console
python -m tests.canonical.lmtp_clustered_ltmle.regenerate
python -m tests.studies.evidence.document --slug clustered-cross-fitted-ltmle
```

The full run fits 1,600 primary replications, 6,000 replications for each of the three
`clustered_inference` fit sets and 2,400 for the band cell. The policy is gated. The study
module states the declared red-cell route.
