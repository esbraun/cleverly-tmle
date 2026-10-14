# Canonical R evidence for repeated-outcome LTMLE

This directory compares ordinary, non-cross-fitted repeated-outcome LTMLE with R `ltmle`
1.3-0. Both implementations receive the same censored three-time-point samples of
`cleverly.datasets.make_longitudinal_repeated`. They also receive three plans, the generating
mechanisms, the same sequential regression families, and pointwise influence-curve inference.

`cleverly` fits each panel once with `outcome=Repeated(["Y1", "Y2", "Y3"])`. R fits the
prefix through `Y_h` once for each horizon `h` with `survivalOutcome=FALSE`. The study reports
eight plan means and five correlated contrasts. The rule equals `always` at the first node,
so a fast test covers that duplicate.

The comparison is secondary evidence. The truths come from an exact enumeration of the
binary law in `cleverly.datasets.repeated_truth`. `tests/studies/ltmle_repeated_law.py`
repeats it on the observed-data law. The property study checks interval calibration,
double robustness, cross-fitting, a joint band, a flexible learner, a continuous outcome,
the rule, and two negative controls.

The study is declared and not yet run. The directory holds no artifacts until the declared
run.

Run a disposable smoke study from the repository root:

```powershell
uv run python -m tests.canonical.ltmle_repeated.regenerate `
  --replicates 12 --n 400 --skip-properties --allow-failures `
  --output .tmp/ltmle-repeated-smoke
```

Run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.ltmle_repeated.regenerate
uv run python -m tests.studies.evidence.document --slug longitudinal-repeated-outcome
```

The runner reuses the `cleverly-ltmle-reference:1.3-0` image. The manifest records every
result-determining module and reference file, including `ltmle_repeated_adapter.R`.
