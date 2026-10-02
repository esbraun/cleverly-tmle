# R `calc_varIC` witness for `logistic_plugin`

This directory holds one construction-matched comparison. It is not a registered study.

`regenerate.py` fits the selector study's first `binary_discrete` sample
(`tests/studies/canonical_ctmle_selector.py`) and writes the inputs of R `ctmle`'s
`calc_varIC` to `inputs.csv`: the outcome, the treatment, the targeted outcome regression,
the bounded propensity of the selected candidate, and that candidate's covariates. The
pinned image `cleverly-ctmle-reference:18de559` runs `run_calc_varic.R`, which calls
`ctmle:::calc_varIC(..., ICg = TRUE)` on those rows and writes `output.csv`.
`manifest.json` records the image id and the SHA-256 of each file.

`tests/unit/test_ctmle_logistic_plugin.py` refits the sample, checks that the fit
reproduces `inputs.csv`, and compares `logistic_plugin(result)["ate"]` with R's `var_ic`
at a relative tolerance of `1e-6`. The fit selects `W1` and `W2`, so the correction term
is applied.

The paired selector study could not supply this witness. The study's paired rows record
no selected candidate, and the rows whose estimates agree to `1e-6` all select the
intercept-only candidate, where the term is absent.

Regenerate from the repository root with Docker running:

```powershell
python -m tests.canonical.ctmle_logistic_plugin.regenerate
```
