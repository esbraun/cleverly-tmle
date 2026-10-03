# Cross-fitted clustered LTMLE on a t reference at few clusters

This fixture holds the `few-cluster-cross-fitted-ltmle` study. It has no comparator. The law is
the end-of-study law of `tests/studies/clustered_longitudinal_laws.py` with the scaled latent.
The property grid crosses 20 and 30 clusters with equal and unequal cluster sizes.

Run a disposable smoke before the full study.

```console
python -m tests.canonical.few_cluster_crossfit_ltmle.regenerate --replicates 20 --skip-properties --jobs 1 --output .tmp/few-cluster-crossfit-smoke
```

Run the declared study once, after `clustered-cross-fitted-ltmle`.

```console
python -m tests.canonical.few_cluster_crossfit_ltmle.regenerate
python -m tests.studies.evidence.document --slug few-cluster-cross-fitted-ltmle
```

The policy is `reporting`. Every red cell is published, and F28 owns it.
