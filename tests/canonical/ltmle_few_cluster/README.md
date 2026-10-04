# Few-cluster t-reference evidence

This fixture holds the `clustered-few-cluster-tmle` study. The primary rows pair in-sample
`LTMLE` with `id=` against R `ltmle` 1.3-0 with `id=` at 20 clusters. The reference image is
the one `tests/canonical/ltmle/Dockerfile` pins. The runner `run_study.R` checks that its
household standard error equals ltmle's own to 1e-10.

Run a disposable smoke into a scratch directory outside the repository.

```console
python -m tests.canonical.ltmle_few_cluster.regenerate --replicates 20 --output <scratch>
```

Run the declared study once, from a clean pushed commit, with every thread variable set to 1.

```console
python -m tests.canonical.ltmle_few_cluster.regenerate --output <empty scratch> --jobs 16
python -m tests.studies.evidence.document --slug clustered-few-cluster-tmle
```

The declared run form of `tests/canonical/declared_run.py` writes the artifacts, the manifest and
`run.log` to the scratch directory, and it copies them here.

The policy is `reporting`. Every red cell is published, and F28 owns it.
