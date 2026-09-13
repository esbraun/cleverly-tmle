# Missing-outcome natural-course TMLE evidence

This directory freezes the independent repeated-sampling study for `ey_obs` under MAR. It covers
binary and bounded-continuous outcomes under the same observational binary-treatment law. The
response probabilities and binary outcome means are exact. The continuous law uses a correctly
specified affine outcome learner on the scaled outcome. A treatment learner that raises if called
verifies that the natural-course score does not use a treatment mechanism.

The property study uses the binary finite law to test the two-nuisance union model, three-size
root-n contraction and exact-EIF efficiency, interval calibration and two derived controls,
targeting necessity, and the consequence of silently switching to a complete-case analysis.

There is no external comparison in this historical row. A later stacked study reaches the R
`tmle` 2.1.1 population-mean path with supplied stitched predictions. This ordinary study ran no
such adapter and inherits no parity claim. Its bounded-continuous target also remains uncompared.
Díaz, Carone and van der Laan (2016), Section 2 and Equations (1)–(5) specify the estimator. The
study therefore commits a schema-valid empty `equivalence.csv` and rests on exact-law and
repeated-sampling evidence.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course.regenerate --jobs 1
```

For a disposable primary smoke run:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-natural-course-smoke --jobs 1
```
