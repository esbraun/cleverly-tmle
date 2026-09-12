# Stacked missing-outcome natural-course CV-TMLE evidence

This directory freezes the registered repeated-sampling study for the one-repeat, pooled,
whole-sample-evaluated stacked CV-TMLE estimator of `ey_obs` under MAR. The primary study fits
separate data-adaptive classification trees for the outcome regression and response mechanism.

The property study checks learned-nuisance interval calibration, a derived shrunken-standard-error
control, both directions of the two-nuisance union model and a both-wrong control, and the coverage
gain from fitting the same fully grown outcome tree out of fold rather than in sample.

No maintained external package was found with the exact target and construction. R `tmle` 2.1.1
was rejected because its MAR API reports intervention-arm means; `tmle3` was rejected because no
maintained callable task for this missing-outcome pooled stacked natural-course construction was
identified; and zEpid was rejected because its cross-fit TMLE targets treatment contrasts. The
study therefore commits a schema-valid, zero-row `equivalence.csv`.

Run a disposable primary smoke study from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course_cvtmle.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-natural-course-cvtmle-smoke --jobs 1
```

That smoke probes primary fitting only. Exercise one fit from every declared property arm,
including the response-oracle adapter used with noise covariates, before the full run:

```powershell
uv run --extra dev python -c "from tests.studies.mar_natural_course_cvtmle_properties import generate_smoke_property_rows; rows = generate_smoke_property_rows(); assert len(rows) == 7; print(rows[['property', 'cell']].to_string(index=False))"
```

Regenerate the complete declared study:

```powershell
uv run --extra dev python -m tests.canonical.tmle_mar_natural_course_cvtmle.regenerate
```
