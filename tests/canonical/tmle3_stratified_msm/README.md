# The identity-link MSM with baseline strata versus R `tmle3`

This registered study fits `MSM.linear(modifiers=("W",), interaction=False)` with
`strata=["V"]` on L1 of `tests/studies/stratified_alternating_law.py`, with the bounded outcome
`Y ~ Beta(24 Q, 24 (1 - Q))`. Each stratum's coefficients are exact truths of the law.

The primary scenario pairs the nine stratum coefficients with pinned R `tmle3` 0.2.0 at
`ed72f8a`. `tmle3` has no stratified MSM, so the runner calls `Param_MSM` once per stratum
subset, as the marginal `tmle3_msm` runner builds it, and transforms the arm-indicator
coefficients to the `(1, a, W)` basis.

Run a disposable smoke study before the declared run:

```powershell
.venv/Scripts/python.exe -m tests.canonical.tmle3_stratified_msm.regenerate --replicates 4 --n 2000 --skip-properties --allow-failures --output build/x8-s1b-smoke
```

Regenerate from the repository root with Docker running:

```powershell
.venv/Scripts/python.exe -m tests.canonical.tmle3_stratified_msm.regenerate --jobs 16 --reference-jobs 8
```

The reader-facing results are in
[`stratified-msm-identity.md`](../../../docs/technical-reference/method-evidence/stratified-msm-identity.md).
