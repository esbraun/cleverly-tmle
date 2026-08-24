# Canonical R evidence for ordinary survival-curve LTMLE

This directory compares ordinary, non-cross-fitted survival-curve LTMLE with R
`ltmle` 1.3-0. Both implementations receive the same censored two-time-point
samples, three regimens, known mechanisms, sequential regression families, and
pointwise influence-curve inference.

R fits each horizon prefix with `survivalOutcome=TRUE`. The study reports five
unique regimen risks and three correlated contrasts. The dynamic regimen equals
always treatment at the first horizon, so a fast identity test covers that duplicate.

The comparison is secondary evidence. The parameter and efficient influence curve
come from `tests/discrete_law_survival.py`. The property study independently checks
double robustness, root-n behavior, calibration, type-I error, power, targeting, and
the event-dependent risk set.

Run a disposable smoke study from the repository root:

```powershell
uv run python -m tests.canonical.ltmle_survival.regenerate `
  --replicates 12 --n 400 --skip-properties --allow-failures `
  --output .tmp/ltmle-survival-smoke
```

Run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.ltmle_survival.regenerate
uv run python -m tests.studies.evidence.document --slug canonical-ltmle-survival
```

The Dockerfile pins R 4.5.2 by digest and checks the `ltmle` 1.3-0 tarball hash.
The manifest records every result-determining module and reference file.
