# Selector-based C-TMLE versus R `ctmle`

This directory freezes a paired repeated-sampling comparison of Cleverly's greedy,
ordered, and discrete selector strategies with R
[`ctmle`](https://github.com/jucheng1992/ctmle) at commit `18de559`. The base R image is
pinned by digest, and the full package commit, source hashes, configuration, result hashes,
and package versions are recorded in `manifest.json`.

Python generates every binary-outcome sample, its exact ATE truth, and its
treatment-stratified five-fold selector assignment. The R runner receives those exact rows
and folds. Both implementations fit the same three-covariate GLM nuisances, use the
0.025--0.975 propensity bounds, disable the selector penalty, and report pointwise 95% ATE
intervals. The unpenalized configuration is intentional: Cleverly's default penalty follows
the published equation and is not claimed to be numerically identical to R's adjustment.

The independent property study uses Cleverly's public nested-cross-fit configuration. It
checks double robustness with a both-wrong control, forced collaborative selection against
an empty-path control, two root-n rates, efficiency, interval calibration, type-I error, and
power. It is independent evidence, not part of the R parity claim.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_selector.regenerate
```

For a deliberately underpowered primary-only probe, write outside this artifact directory:

```powershell
uv run --extra dev python -m tests.canonical.ctmle_selector.regenerate --replicates 10 --n 300 --primary-only --output $env:TEMP\ctmle-selector-probe
```

The comparison is limited to binary, two-arm, complete-outcome ATE estimation with GLMs and
no cross-fitting in the parity fit. Continuous outcomes, Cleverly's penalty, default nested
cross-fitting, and other estimands are not inherited from this R comparison; cross-fitted
selector behavior is assessed only by the independent property cells. Missing outcomes,
weights, clusters, strata, multi-valued treatment, simultaneous or bootstrap intervals,
flexible learner libraries, and severe practical-positivity behavior are outside scope.

The reader-facing measurements and limitations are in
[`docs/technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md`](../../../docs/technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md).
