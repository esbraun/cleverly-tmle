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
the rule, targeting, the outcome history, and the controls that must fail.

The R runs of the fourth declaration are pending (TRACKER Study queue item 14(c)). The record
declares `pending_reference="ltmle"`, so the committed artifacts hold the declared `cleverly`
rows and the property study, and no R row. The change that registers the R rows moves the name
to `reference` and reruns the study with `--skip-properties`.

Run a disposable smoke study from the repository root. It uses the declared sample size. At
`n = 400`, a stratified regression at the rule's third node can be rank deficient, and the R
side then stops on purpose. The smoke study draws 700 replications. Seven of them have a separated
outcome fit on the primary stream: 157, 242, 309, 444, 506, 528 and 693.

```powershell
uv run python -m tests.canonical.ltmle_repeated.regenerate `
  --replicates 700 --n 3000 --skip-properties --allow-failures `
  --output .tmp/ltmle-repeated-smoke
```

Check the parity rule on the smoke rows. The rule compares the two implementations with the
size of R's own targeting move. The fast tier applies the same rule to the committed rows. The
command prints one line for each breach and exits with code 1 when there is a breach.

```powershell
uv run python -m tests.unit.test_ltmle_repeated_method_study `
  .tmp/ltmle-repeated-smoke/replicates.parquet
```

Run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.ltmle_repeated.regenerate
uv run python -m tests.studies.evidence.document --slug longitudinal-repeated-outcome
```

The runner reuses the `cleverly-ltmle-reference:1.3-0` image. The manifest records every
result-determining module and reference file, including `ltmle_repeated_adapter.R`.
