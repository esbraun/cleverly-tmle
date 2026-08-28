# Repeated point-treatment cross-fitted TMLE

This directory freezes an independent repeated-sampling study of `cleverly`'s repeated,
stacked point-treatment CV-TMLE. The primary study uses five treatment-stratified folds per
draw, three fold draws, pooled targeting, a row-aligned average influence curve, and the
arithmetic mean of the draw-specific point estimates. Risk ratios and odds ratios are
aggregated on their log inference scale.

The study tests arm means, ATE, ATT, ATC, observed mean, PAR, and, where defined, PAF, RR, and OR
against exact truth under binary and bounded continuous outcome laws. It reuses the shared CV-TMLE laws and property
machinery for double robustness, empirical and reported root-n rates, interval calibration,
type-I error and power, and flexible-learner cross-fitting against an in-sample control.

Three independent repeat-specific experiments supplement those shared properties. A fixed
nonlinear sample varies only the fold seed to compare three-draw rowwise averaging with a
single split and equal-fold evaluation. A naturally generated rare tail stratum then makes
some fold draws materially less stable without altering fitted estimates after the fact. On
identical samples and fold assignments, that experiment compares mean and median point
aggregation and compares the shipped row-averaged influence-curve interval with the
Chernozhukov et al. repeated-split mean variance adjustment. An oracle-outcome arm is the
specificity control: robust aggregation should not buy a material improvement when nuisance
fits are stable.

No canonical implementation is compared. Chernozhukov et al. establish mean and median
aggregation and fixed-repeat first-order validity, but the maintained implementations surveyed
for this study do not expose the same stacked targeting and row-aligned averaged influence
curve as `cleverly`. The exact-truth, calibration, paired-control, and mutation evidence is
therefore more direct than a surrogate parity comparison. `equivalence.csv` is intentionally
empty and schema-valid.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.repeated_crossfit.regenerate
```

The declarations are in `tests/studies/repeated_crossfit.py` and
`tests/studies/repeated_crossfit_properties.py`. The reader-facing scope, measurements, red
results, and limitations are in
[`docs/technical-reference/method-evidence/repeated-cross-fitting.md`](../../../docs/technical-reference/method-evidence/repeated-cross-fitting.md).
The manifest records all margins, seeds, configuration, source hashes, and artifact hashes.
