# The identity-link MSM with baseline strata versus R `tmle3`

This registered study fits `MSM.linear(modifiers=("W",), interaction=False)` with
`strata=["V"]` on L1 of `tests/studies/stratified_alternating_law.py`, with the bounded outcome
`Y ~ Beta(24 Q, 24 (1 - Q))`. Each stratum's coefficients are exact truths of the law.

The primary scenario pairs the nine stratum coefficients with pinned R `tmle3` 0.2.0 at
`ed72f8a`. `tmle3` has no stratified MSM, so the runner calls `Param_MSM` once per stratum
subset, as the marginal `tmle3_msm` runner builds it, and transforms the arm-indicator
coefficients to the `(1, a, W)` basis.

Run a disposable smoke study into a scratch directory outside the repository:

```console
python -m tests.canonical.tmle3_stratified_msm.regenerate --replicates 4 --n 2000 --skip-properties --allow-failures --output <scratch>
```

A declared run passes `--output <scratch>` and optionally `--jobs`, and nothing else.
`tests/canonical/declared_run.py` describes the guard, the run log and the copy into this
directory. Docker must be running.

The reader-facing results are in
[`stratified-msm-identity.md`](../../../docs/technical-reference/method-evidence/stratified-msm-identity.md).
