# Omitted-variable bound plug-in standard error

This directory holds a repeated-sampling study of the plug-in omitted-variable bound. Under
`nu2_estimator="plugin"`, X26 stacks the curve of $E_n[\hat\alpha^2]$ with the score of a
declared treatment mechanism. The study reads three quantities of each parameter as estimates
of known population values: $\nu^2$ with the standard error of its own curve, and the lower and
upper bound with the standard error read back off the reported limit.

| scenario | law and fit |
| --- | --- |
| `binary_logistic` | `make_linear_ate(n=1000)`, in-sample TMLE, `LogisticRegression(C=numpy.inf)` |
| `binary_logistic_ridge` | the same as `binary_logistic` with `LogisticRegression(max_iter=1000)`, so `C = 1` |
| `multinomial_logistic` | the three-arm law of the study module, in-sample, `LogisticRegression(C=numpy.inf)` |
| `binary_known` | `make_linear_ate(n=1000)` with its own mechanism declared through `treatment_probabilities=` |

The strength is `cf_y = 0.5`, `cf_d = 0.5` and `rho = 1`. The property study holds one
`interval_calibration` family with 36 positive cells, 18 `inflated_se_control` cells and 6
`shrunken_se_control` cells. No canonical implementation is compared, so `equivalence.csv` is
empty and schema-valid.

Regenerate from the repository root:

```text
python -m tests.canonical.omitted_variable_bound_plugin.regenerate
```

The declarations are in `tests/studies/omitted_variable_bound_plugin.py` and
`tests/studies/omitted_variable_bound_plugin_properties.py`. The reader-facing scope and
limitations are in
[`docs/technical-reference/method-evidence/omitted-variable-bound-plugin-standard-error.md`](../../../docs/technical-reference/method-evidence/omitted-variable-bound-plugin-standard-error.md).
