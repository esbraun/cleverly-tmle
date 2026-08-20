# Canonical point-treatment TMLE versus R `tmle3`

This directory freezes a paired repeated-sampling comparison with R `tmle3` 0.2.0 at
commit `ed72f8a`. The R package and cleverly receive the same realized datasets, every
covariate in each data-generating law, ordinary TMLE settings, pointwise 95% intervals,
and corresponding GLM nuisance learners. The comparison is secondary implementation
evidence, not the derivation or the scientific oracle.

The R runner uses the public `tmle3` specifications directly except for ATT and ATC. Those
two estimands use the constrained, one-dimensional updater exercised by the pinned
package's own ATT/ATC tests. The public ATT convenience path reaches a non-finite
convergence state on 5 of the 400 bounded-continuous samples; the package-tested path
completes every fixed replication without dropping or replacing data.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.tmle3.regenerate
```

Use `--replicates 10 --allow-failures` for a smoke run in a disposable checkout; coverage
and equivalence gates are intentionally underpowered at that size. The published artifact
uses 400 replications; do not replace it with a smoke run. Raw simulated datasets are
temporary. The committed `replicates.csv.gz` retains the estimates, standard errors,
intervals, truth, coverage indicator, and pairing key needed to reproduce every summary.

The `paf` row needs a qualification: `tmle3` constructs its interval by transforming a
Wald interval for the log risk ratio, while cleverly uses the PAF influence curve on the
fraction scale. Both are first-order delta-method intervals for the same parameter, but
their standard errors and finite-sample endpoints are not numerically interchangeable.

`performance-tests.csv` independently tests each implementation's bias, coverage, and SE
calibration with 99% intervals. `equivalence.csv` contains paired 99% similarity tests and
one-sided 99% non-inferiority bounds: `cleverly` may be better, but cannot be materially
worse than R under the declared RMSE, coverage, and calibration margins. The margins,
confidence level, bootstrap count, package pins, and artifact hashes are machine-readable
in `manifest.json`.
