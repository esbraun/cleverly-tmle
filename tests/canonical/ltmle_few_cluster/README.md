# Few-cluster t-reference evidence

This fixture holds the `clustered-few-cluster-tmle` study. The primary rows pair in-sample
`LTMLE` with `id=` against R `ltmle` 1.3-0 with `id=` at 20 clusters. The reference image is
the one `tests/canonical/ltmle/Dockerfile` pins. The runner `run_study.R` checks that its
household standard error equals ltmle's own to 1e-10.

Run a disposable smoke before the full study.

```console
python -m tests.canonical.ltmle_few_cluster.regenerate --replicates 20 --skip-properties --jobs 1 --output .tmp/few-cluster-smoke
```

Run the declared study once, after the unequal-size study.

```console
python -m tests.canonical.ltmle_few_cluster.regenerate
python -m tests.studies.evidence.document --slug clustered-few-cluster-tmle
```

The policy is `reporting`. Every red cell is published, and F28 owns it.
