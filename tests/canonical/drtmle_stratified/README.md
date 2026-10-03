# DR-TMLE with baseline strata versus R `drtmle`

This registered study fits cross-fitted `DRTMLE(guard=("Q", "g"))` with `strata=["V"]` on the
stratified paper law of `tests/studies/stratified_alternating_law.py`: the complete-data binary
law of Benkeser et al. (2017), Section 5.1, with a stratum that shifts both intercepts and the
treatment effect. Every truth is a one-dimensional quadrature.

The primary scenario pairs the subject with pinned R `drtmle` 1.1.2 at `538a3a2`. `drtmle` has
no stratified form, so the runner calls it once per stratum subset. Both
sides read the same rows, the same ten-fold assignment and the subject's initial nuisance
arrays. Inside a stratum, `drtmle` fits its reduced regressions on the stratum's rows, which is
the construction the subject uses.

Run a disposable smoke study before the declared run:

```powershell
.venv/Scripts/python.exe -m tests.canonical.drtmle_stratified.regenerate --replicates 4 --n 2000 --skip-properties --allow-failures --output build/x8-s2-smoke
```

Regenerate from the repository root with Docker running:

```powershell
.venv/Scripts/python.exe -m tests.canonical.drtmle_stratified.regenerate --jobs 16 --reference-jobs 8
```

The reader-facing results are in
[`stratified-dr-tmle.md`](../../../docs/technical-reference/method-evidence/stratified-dr-tmle.md).
