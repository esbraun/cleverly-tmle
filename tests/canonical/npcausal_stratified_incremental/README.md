# Stratified incremental and MSM targeting versus R `npcausal`

This registered study fits stratified incremental interventions, stratified linked and
continuous-dose MSMs, and the cross-fitted natural-course mean with strata. It uses the laws of
`tests/studies/stratified_alternating_law.py`, which give every truth exactly.

The primary scenario pairs the in-sample stratified incremental fit with pinned R `npcausal` at
`56a5ac1`. `npcausal` has no stratified estimator, so the runner calls `ipsi` once per stratum
subset and multiplier, and once per multiplier on every row for the five marginal names.
`cleverly` fits saturated nuisances. `npcausal` fits `SL.glm.interaction`, which is saturated
inside a stratum and correct, but not saturated, on every row. The reference cross-fits over two
splits, because its single-split path selects no training rows at this commit.

Run a disposable smoke study before the declared run:

```powershell
.venv/Scripts/python.exe -m tests.canonical.npcausal_stratified_incremental.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/x8-s1-smoke
```

Regenerate from the repository root with Docker running:

```powershell
.venv/Scripts/python.exe -m tests.canonical.npcausal_stratified_incremental.regenerate --jobs 16 --reference-jobs 8
```

The reader-facing results are in
[`stratified-incremental-msm.md`](../../../docs/technical-reference/method-evidence/stratified-incremental-msm.md).
