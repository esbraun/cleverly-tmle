# Baseline-strata point-treatment TMLE versus R `tmle3`

This registered study fits ordinary point-treatment TMLE with `strata=["V"]` on a finite-support
law with three unequal baseline strata. The law is `tests/studies/stratified_law.py`, which gives
every stratum truth and efficient influence function exactly.

The primary scenario pairs `cleverly` with pinned R `tmle3` 0.2.0 at `ed72f8a`. The reference
runs `tmle_stratified(..., base_estimate = FALSE)` in two stages: `tmle_TSM_all()` for the six
stratum arm means and `tmle_ATE(1, 0)` for the three stratum contrasts. Both sides fit main-terms
logistic regressions. The runner checks on every replication that each row's nonzero stratum
weight is its own stratum's `n / n_s`.

Regenerate from the repository root with Docker running:

```powershell
.venv/Scripts/python.exe -m tests.canonical.tmle3_stratified.regenerate --jobs 16 --reference-jobs 8
```

The reader-facing results are in
[`stratified-point-treatment-tmle.md`](../../../docs/technical-reference/method-evidence/stratified-point-treatment-tmle.md).
