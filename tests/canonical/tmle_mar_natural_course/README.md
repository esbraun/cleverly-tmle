# Missing-outcome natural-course TMLE evidence

This directory freezes the independent repeated-sampling study for `ey_obs` under MAR. It covers
binary and bounded-continuous outcomes under the same observational binary-treatment law. The
response probabilities and binary outcome means are exact. The continuous law uses a correctly
specified affine outcome learner on the scaled outcome. A treatment learner that raises if called
verifies that the natural-course score does not use a treatment mechanism.

The property study uses the binary finite law to test the two-nuisance union model, three-size
root-n contraction and exact-EIF efficiency, interval calibration and two derived controls,
targeting necessity, and the consequence of silently switching to a complete-case analysis.

Díaz, Carone and van der Laan (2016), Section 2 and Equations (1)–(5) specify the estimator.

This study runs no external comparison. It therefore commits a schema-valid empty
`equivalence.csv` and rests on exact-law and repeated-sampling evidence. The R `tmle` 2.1.1
population-mean path accepts supplied predictions, and the stacked study in
`tests/canonical/tmle_mar_natural_course_cvtmle/` compares with it. These artifacts predate that
adapter and were not regenerated, so they make no parity claim for the binary or
bounded-continuous law. Roadmap item RM10 in `docs/roadmap.md` tracks both comparisons.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course.regenerate --jobs 1
```

For a disposable primary smoke run:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-natural-course-smoke --jobs 1
```
