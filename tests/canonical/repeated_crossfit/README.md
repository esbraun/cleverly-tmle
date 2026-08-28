# Repeated point-treatment cross-fitted TMLE

This directory freezes an independent repeated-sampling study of `cleverly`'s repeated,
stacked point-treatment CV-TMLE. The primary study uses five treatment-stratified folds per
draw, three fold draws, pooled targeting, the median of the draw-specific point estimates,
and the median of within-draw variance plus squared split displacement. Risk ratios and odds
ratios are aggregated on their log inference scale.

The study tests arm means, ATE, ATT, ATC, observed mean, PAR, and, where defined, PAF, RR, and OR
against exact truth under binary and bounded continuous outcome laws. It reuses the shared
CV-TMLE property machinery for double robustness, empirical and reported root-n rates, interval
calibration, type-I error, and power.

No full estimator is compared. Chernozhukov et al. establish the median aggregation rule and
fixed-repeat first-order validity, and zEpid implements the same aggregation formula. zEpid's
nuisance training and foldwise targeting differ from `cleverly`'s complement-trained stacked
pooled update, so it is corroboration for the reporting layer rather than numerical parity for
the complete estimator. `equivalence.csv` is intentionally empty and schema-valid.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.repeated_crossfit.regenerate
```

The declarations are in `tests/studies/repeated_crossfit.py` and
`tests/studies/repeated_crossfit_properties.py`. The reader-facing scope, measurements, and
limitations are in
[`docs/technical-reference/method-evidence/repeated-cross-fitting.md`](../../../docs/technical-reference/method-evidence/repeated-cross-fitting.md).
The manifest records all margins, seeds, configuration, source hashes, and artifact hashes.
