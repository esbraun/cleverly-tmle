# Omitted-variable bound plug-in standard error

This directory holds a repeated-sampling study of the second-order-corrected plug-in
omitted-variable bound. Under `nu2_estimator="plugin"`, the package reports
$\hat\nu^2 = E_n[\hat\alpha^2] - \hat B$ and stacks the curve of $\hat\nu^2$ with the score of
the treatment mechanism. The study judges the lower and the upper bound of each parameter: the
point estimate against the known population limit, and the standard error read back off the
reported limit. The corrected $\nu^2$ is measured and not judged.

| scenario | law and fit |
| --- | --- |
| `binary_logistic` | `make_linear_ate(n=1000)`, in-sample TMLE, `LogisticRegression(C=numpy.inf)` |
| `binary_logistic_crossfit` | the same law, 5-fold cross-fit, outcome support `(-40, 45)` |
| `binary_logistic_ridge` | the same as `binary_logistic` with `LogisticRegression(max_iter=1000)`, so `C = 1` |
| `binary_logistic_ridge_crossfit` | the same as `binary_logistic_crossfit` with the ridge learner |
| `multinomial_logistic` | the three-arm law of the study module, in-sample, `g_bounds=(0.005, 0.995)` |
| `multinomial_logistic_crossfit` | the three-arm law, 5-fold cross-fit, `g_bounds=(0.005, 0.995)` |
| `binary_known` | `make_linear_ate(n=1000)` with its own mechanism declared through `treatment_probabilities=` |
| `binary_unbalanced` | the binary law with intercept -1.5 (treated share 0.189), `g_bounds=(0.005, 0.995)` |

The strength is `cf_y = 0.5`, `cf_d = 0.5` and `rho = 1`. Each scenario fills the ATE, the ATT
and the ATC (the three-arm law reads `ate_2v0`, `att_2v0` and `atc_1v0`). The property study
holds one `interval_calibration` family with 48 positive cells (a lower and an upper bound for
each of 24 scenario and parameter pairs), 28 `inflated_se_control` cells and 4
`shrunken_se_control` cells. Each parameter is its own primary scenario, so a refusal of one
parameter fails the cells of that parameter alone. The corrected $\nu^2$ rows are published as
`nu2-replicates.csv.gz` and `nu2-reading.csv`, with no verdict. The property phase publishes two
further readings, also with no verdict. `correction-reading.csv` gives each positive cell with and
without the package variance correction, and `refusal-reading.csv` gives the refusals of each
parameter by key. No canonical implementation is compared, so `equivalence.csv` is empty and
schema-valid.

Regenerate from the repository root:

```text
python -m tests.canonical.omitted_variable_bound_plugin.regenerate
```

The declarations are in `tests/studies/omitted_variable_bound_plugin.py` and
`tests/studies/omitted_variable_bound_plugin_properties.py`. The reader-facing scope and
limitations are in
[`docs/technical-reference/method-evidence/omitted-variable-bound-plugin-standard-error.md`](../../../docs/technical-reference/method-evidence/omitted-variable-bound-plugin-standard-error.md).
