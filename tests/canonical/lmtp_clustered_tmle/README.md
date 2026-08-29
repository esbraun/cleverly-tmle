# Clustered point-treatment CV-TMLE evidence

This fixture compares `cleverly` with pinned R `lmtp` 1.5.4 and `ife` 0.2.3. Both
implementations use the same clustered samples, exact treatment mechanism, and five grouped
folds. The R runner keeps the cluster identifier in each `ife` estimate. It forms the ATE by
subtracting the two arm objects, so the joint influence curve supplies cluster-robust inference.

Run a disposable comparison before the full study.

```console
python -m tests.canonical.lmtp_clustered_tmle.regenerate --replicates 3 --n 200 --primary-only --output .tmp/clustered-smoke
```

Run the declared study only once the smoke output passes review.

```console
python -m tests.canonical.lmtp_clustered_tmle.regenerate
python -m tests.studies.evidence.document --slug clustered-tmle
```

The full run fits 800 primary replications and 2,400 paired property replications. It must not
use `--allow-failures`. The manifest records the pinned container, runner, adapter, harness,
configuration, and all result-determining Python modules.
