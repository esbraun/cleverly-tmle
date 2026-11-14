# Clustered point-treatment CV-TMLE evidence at unequal cluster sizes

This fixture holds the `clustered-unequal-cvtmle` study. It has no comparator. Each draw
holds 200 clusters whose sizes are uniform on 2 to 18. The law is declared in
`tests/studies/clustered_unequal_laws.py`, whose docstring records the pilot that set it.

Run a disposable smoke into a scratch directory outside the repository.

```console
python -m tests.canonical.clustered_unequal_cvtmle.regenerate --replicates 20 --output <scratch>
```

Run the declared study once, from a clean pushed commit, with every thread variable set to 1.

```console
python -m tests.canonical.clustered_unequal_cvtmle.regenerate --output <empty scratch> --jobs 16
python -m tests.studies.evidence.document --slug clustered-unequal-cvtmle
```

The declared run form of `tests/canonical/declared_run.py` writes the artifacts, the manifest and
`run.log` to the scratch directory, and it copies them here.

The full run fits 800 primary replications per scenario and 2,400 replications per property
pair, except the fold-evaluated covariate pair, which fits 4,000 on its own stream
(`covariate_fold_evaluated_t_reference`) and reads $t_{30}$, $J - V$ at 40 clusters in 10 folds.
Its fits also publish a reported arm with the normal quantile.
The fold-evaluated pair on the informative law left the study, because `cv_evaluation=True`
refuses clusters of unequal size. The policy is `reporting`, and F28 owns any red cell.
