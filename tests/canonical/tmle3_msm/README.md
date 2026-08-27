# Canonical R evidence for the point-treatment MSM projection

This directory compares the ordinary identity-link MSM projection in `cleverly` with the
Gaussian projection in pinned R `tmle3` `Param_MSM`. Both implementations receive the same
bounded continuous-outcome samples, the same unsaturated working model, the same fixed
nonuniform projection measure, and correctly specified nuisance regressions.

The working model has an intercept, a treatment term, and a baseline term against six
counterfactual cells, so it stays a genuine projection rather than a saturated outcome model.
The R arm-indicator coefficients are transformed to the `cleverly` parameterization. The same
transformation applies to the joint influence curve, so the treatment coefficient's standard
error retains the covariance between both R arm indicators.

The pinned `tmle3` release compares a documented custom weight function with two string
sentinels before it checks `is.function()`. The runner gives that function a narrow equality
method so the package reaches its documented custom-weight branch without patching the pinned
source.

The comparison is secondary evidence. The parameter, its efficient influence curve, and the
efficiency bound come from the finite-support law in `tests/discrete_law.py`, differentiated
by complex step in `tests/studies/point_msm_properties.py`. That property study independently
checks double robustness, root-n behavior, calibration, type-I error, power, targeting, and
the declared projection measure.

Run a disposable smoke study from the repository root:

```powershell
uv run python -m tests.canonical.tmle3_msm.regenerate `
  --replicates 12 --n 400 --skip-properties --allow-failures `
  --output .tmp/point-msm-smoke
```

Run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.tmle3_msm.regenerate
uv run python -m tests.studies.evidence.document --slug point-msm
```

The replication count and sample size come from the study record and are recorded in
`manifest.json`. Do not publish a smoke run over them. The Dockerfile pins R 4.5.2 by digest
and both `tmle3` and `sl3` by commit. The manifest records every result-determining module and
reference file.
