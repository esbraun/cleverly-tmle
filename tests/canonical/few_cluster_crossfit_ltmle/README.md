# Cross-fitted clustered LTMLE on a t reference at few clusters

This fixture holds the `few-cluster-cross-fitted-ltmle` study. It has no comparator. The law is
the end-of-study law of `tests/studies/clustered_longitudinal_laws.py` with the scaled latent.
The property grid crosses 20 and 30 clusters with equal and unequal cluster sizes.

Run a disposable smoke into a scratch directory outside the repository.

```console
python -m tests.canonical.few_cluster_crossfit_ltmle.regenerate --replicates 20 --jobs 1 --output <scratch>
```

Run the declared study once, after `clustered-cross-fitted-ltmle`, from a clean pushed commit,
with every thread variable set to 1.

```console
python -m tests.canonical.few_cluster_crossfit_ltmle.regenerate --output <empty scratch> --jobs 16
python -m tests.studies.evidence.document --slug few-cluster-cross-fitted-ltmle
```

The declared run form of `tests/canonical/declared_run.py` writes the artifacts, the manifest and
`run.log` to the scratch directory, and it copies them here.

Each `iid_t_control` cell passes only when two clauses hold, as in `clustered-few-cluster-tmle`:
the 99% upper endpoint of the paired ratio, the mean IID SE over the mean cluster-robust SE of the
same fits, below 0.80, and the paired `t_reference` arm's SE-ratio upper endpoint at most 1.20.

The policy is `reporting`. Every red cell is published, and F28 owns it.
