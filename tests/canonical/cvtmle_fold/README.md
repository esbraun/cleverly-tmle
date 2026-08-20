# Fold-evaluated point-treatment CV-TMLE

This directory freezes the independent repeated-sampling study for cleverly's original
fold-evaluated CV-TMLE report.  It uses treatment-stratified ten-fold nuisance fitting and one
pooled targeting update, then averages the fold-specific plug-in reports equally and uses the
cross-validated influence-curve variance.

The study tests `ey1`, `ey0`, `ate`, `att`, and `atc` against exact truth under binary and
bounded continuous outcome laws.  It also measures double robustness with a both-wrong
negative control, empirical and reported root-n rates, efficiency and interval calibration at
three sample sizes, type-I error with a power control, and a flexible-learner overfitting case
against a deliberately in-sample control.

There is no external comparison in this row.  The pinned R `tmle3` comparison is a separate
stacked CV-TMLE construction; treating it as parity evidence for equal-fold plug-in evaluation
would erase a real method choice.  This artifact therefore commits an empty, schema-valid
`equivalence.csv` and rests on its exact-truth and statistical-property evidence.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.cvtmle_fold.regenerate
```

The declaration is `tests/studies/fold_evaluated_cvtmle.py`; the reader-facing scope,
measurements, and limitations are in
[`docs/technical-reference/method-evidence.md`](../../../docs/technical-reference/method-evidence.md).
The manifest records all margins, seeds, configuration, source hashes, and artifact hashes.
