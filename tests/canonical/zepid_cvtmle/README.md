# Fold-targeted point-treatment CV-TMLE versus Python `zEpid`

This directory freezes a paired study against `zEpid` 0.9.1 at commit `16a0f96`. Both
implementations use the same binary samples, equal two-fold assignments, main-effects logistic
nuisance learners, treatment bounds, fold-specific targeting, and fold-evaluated ATE inference.
`cleverly` averages folds equally at $1/V$; zEpid takes the mean over stacked targeted rows and
therefore size-weights folds. Those weights coincide here only because the folds are equal. Their
finite-sample variance formulas are not identical: zEpid uses a within-fold
sample variance with `ddof=1`, while `cleverly` uses the raw influence-curve second moment.

The Python runner calls `SingleCrossfitTMLE` with one partition. It checks the row identities in
each native split before it fits a nuisance model. The two-split design makes each nuisance
training split the complete complement of its validation split.

The double-robustness properties use a bounded nonlinear confounded law with exact ATE 1.75.
Its treatment mechanism stays between 0.182 and 0.742, so the configured bounds do not clip it.
The wrong main-effects outcome regression has a constant contrast, while the true contrast varies.
The treatment-correct cell uses n = 2,000 and 1,200 replications. The other three cells use
n = 700 and 1,200 replications. All cells use their predeclared seeds and the existing margins.

Run a disposable primary smoke from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.zepid_cvtmle.regenerate --replicates 10 --primary-only --output $env:TEMP\zepid-cvtmle-probe
```

Regenerate the declared study after the smoke passes:

```powershell
uv run --extra dev python -m tests.canonical.zepid_cvtmle.regenerate
```

The declaration is `tests/studies/fold_targeted_cvtmle.py`. The manifest pins the base image,
package commit, Python dependencies, settings, seeds, source hashes, and artifact hashes.
