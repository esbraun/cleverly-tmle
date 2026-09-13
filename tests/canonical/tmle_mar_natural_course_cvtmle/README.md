# Stacked missing-outcome natural-course CV-TMLE evidence

This directory freezes the registered repeated-sampling study for the one-repeat, pooled,
whole-sample-evaluated stacked CV-TMLE estimator of `ey_obs` under MAR. The primary study fits
separate depth-five classification trees, with minimum leaf size 25, for the outcome regression and
the response mechanism.

The law has six `(A, W)` covariate cells, and each expected training-complement cell holds more
than 25 rows. Those trees therefore fit the saturated model, which is correct for this law. The
primary rows and the flexible-learning calibration cell do not test calibration under a
misspecified data-adaptive learner.

The property study has four parts:

| part | what it checks |
| --- | --- |
| flexible-learning calibration | coverage and SE calibration with the saturated depth-five trees |
| shrunken-SE control | the same fits with each standard error multiplied by 0.70 |
| union model | each nuisance correct in turn, and a both-wrong control that must show detectable bias |
| overfitting pair | a fully grown outcome tree with eight noise covariates and the exact response mechanism, fitted out of fold and in sample on the same draws |

The overfitting pair is the data-adaptive evidence.

The study compares the R `tmle` 2.1.1 population-mean path. The adapter supplies the same stitched
out-of-fold outcome and response predictions. It uses a constant synthetic treatment so R targets
the natural-course population mean without fitting a treatment mechanism, while retaining the
original treatment beside `W` to preserve the conditioning set represented by the predictions.

R keeps its native centered sample variance and bounded binary-outcome interval. `cleverly` uses
the raw influence-curve second moment and an unbounded Wald interval. The comparison therefore
tests point-estimator equivalence and native-inference non-inferiority, not bitwise interval parity.

`docs/references.md` records the source locator for each candidate:

| candidate | disposition |
| --- | --- |
| R `tmle` 2.1.1 | compared through its population-mean path with supplied stitched predictions |
| `tmle3` at commit `ed72f8a` | its generic treatment-specific outcome fit can use the `Delta = 0` pseudo-outcomes when it predicts under `Delta = 1`, while `cleverly` fits that regression on respondents only |
| zEpid 0.9.1 | its cross-fit TMLE targets inside each fold rather than with one pooled fluctuation |
| Newey and Robins (2018) | the construction fits the outcome and inverse response regressions on distinct subsamples, which is a different estimator |

The `comparator_search` string in `manifest.json` records the compared and rejected candidates.

Run a disposable primary smoke study from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course_cvtmle.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-natural-course-cvtmle-smoke --jobs 1
```

That smoke probes the Python and R primary fitting paths. Exercise one fit from every property arm,
including the response-oracle adapter used with noise covariates, before the full run:

```powershell
uv run --extra dev python -c "from tests.studies.mar_natural_course_cvtmle_properties import generate_smoke_property_rows; rows = generate_smoke_property_rows(); assert len(rows) == 7; print(rows[['property', 'cell']].to_string(index=False))"
```

Regenerate the complete declared study:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course_cvtmle.regenerate
```
