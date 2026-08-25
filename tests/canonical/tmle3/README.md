# Canonical point-treatment TMLE versus R `tmle3`

This directory freezes a paired repeated-sampling comparison with R `tmle3` 0.2.0 at
commit `ed72f8a`. The R package and cleverly receive the same realized datasets, every
covariate in each data-generating law, ordinary TMLE settings, pointwise 95% intervals,
and corresponding GLM nuisance learners. The comparison is secondary implementation
evidence, not the derivation or the scientific oracle.

The study is declared in `tests/studies/canonical_tmle.py`; everything that turns its rows into
summaries and verdicts is the shared machinery in `tests/studies/evidence/`. The reader-facing
account is [`docs/technical-reference/method-evidence/canonical-point-treatment-tmle.md`](../../../docs/technical-reference/method-evidence/canonical-point-treatment-tmle.md).

The R runner uses the public `tmle3` specifications directly except for ATT and ATC. Those
two estimands use the constrained, one-dimensional updater exercised by the pinned
package's own ATT/ATC tests. The public ATT convenience path reaches a non-finite
convergence state on a small fraction of the bounded-continuous samples; the package-tested
path completes every fixed replication without dropping or replacing data. The runner aborts on
any failed replication rather than continuing with fewer.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.tmle3.regenerate
```

The run takes the whole core budget: the Python side fits every replication across it, and the R
container is given the same count afterwards rather than beside it, so the two phases never
contend for the same machine. `--jobs` overrides the count.

For a smoke run, write somewhere disposable rather than over the published results:

```powershell
uv run --extra dev python -m tests.canonical.tmle3.regenerate --replicates 10 --allow-failures --output $env:TEMP\probe
```

Coverage and equivalence gates are intentionally underpowered at that size and will refuse to
conclude anything, which is the correct behaviour of an equivalence-shaped gate rather than a
failure. Seeds are derived per scenario and replication, so a short run redraws exactly the first
replications of the published one. The published artifact uses the replication count and sample
size declared on the study record and recorded in `manifest.json`; do not replace it with a smoke
run.

Raw simulated datasets are temporary. The committed `replicates.csv.gz` retains the estimates,
standard errors, intervals, truth, coverage indicator, and pairing key needed to reproduce every
summary.

The `paf` row needs a qualification: `tmle3` constructs its interval by transforming a
Wald interval for the log risk ratio, while cleverly uses the PAF influence curve on the
fraction scale. Both are first-order delta-method intervals for the same parameter, but
their standard errors and finite-sample endpoints are not numerically interchangeable. The study
record declares that exemption by name and a test requires the two implementations to really
report different scales for it.

`performance-tests.csv` independently tests each implementation's bias, coverage, and SE
calibration against known truth with 99% intervals, all three bounded by margins declared before
the run. `equivalence.csv` contains paired 99% similarity tests and one-sided 99% non-inferiority
bounds: `cleverly` -- the `subject` -- may be better, but cannot be materially worse than R under
the declared RMSE, coverage, and calibration margins. Its `passed` column is exactly those two
claims; `subject_valid` and `reference_valid` carry each implementation's own verdict separately,
so a reference that degrades is reported against the reference. The margins, confidence level,
bootstrap count, package pins, the cleverly version and commit the run came from, and artifact
hashes are machine-readable in `manifest.json`.
