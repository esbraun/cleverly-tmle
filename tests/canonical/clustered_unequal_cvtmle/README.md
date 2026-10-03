# Clustered point-treatment CV-TMLE evidence at unequal cluster sizes

This fixture holds the `clustered-unequal-cvtmle` study. It has no comparator. Each draw
holds 200 clusters whose sizes are uniform on 2 to 18. The law is declared in
`tests/studies/clustered_unequal_laws.py`, whose docstring records the pilot that set it.

Run a disposable smoke before the full study.

```console
python -m tests.canonical.clustered_unequal_cvtmle.regenerate --replicates 20 --skip-properties --jobs 1 --output .tmp/unequal-smoke
```

Run the declared study once, after every source change is final.

```console
python -m tests.canonical.clustered_unequal_cvtmle.regenerate
python -m tests.studies.evidence.document --slug clustered-unequal-cvtmle
```

The full run fits 800 primary replications per scenario and 2,400 replications per property
pair. The policy is gated. A red cell routes to F28 by the declaration in the study module.
