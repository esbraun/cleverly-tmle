# Implementation validation studies

Each page here validates one method that `cleverly` implements. A page exists only for a study
registered in `tests/studies/evidence/registry.py`. The shared machinery in
`tests/studies/evidence/` computes every verdict, and `tests/unit/test_method_evidence.py` checks
these pages against the committed results. Every table is generated from those results, so a stale
number is a test failure and not a reading error.

Read the pages in this order.

| page | what it gives you |
| --- | --- |
| [Implementation validation grid](validation-grid.md) | all thirty-three studies in one table, with the counts and the declared limits |
| [How to read these studies](how-to-read.md) | the three questions, the verdict rules, and the terms every study below applies |
| the thirty-three study pages | one row per committed test, with what it checked and the verdict its own endpoints produced |

To register a new study, follow
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row).

```{toctree}
:maxdepth: 2

validation-grid
how-to-read
canonical-point-treatment-tmle
weighted-point-treatment-tmle
learned-weighted-point-treatment-tmle
ordinary-multi-arm-tmle
stacked-point-treatment-cv-tmle
clustered-point-treatment-cv-tmle
fold-evaluated-point-treatment-cv-tmle
repeated-cross-fitting
selector-based-point-treatment-c-tmle
selector-based-multi-arm-c-tmle
outcome-adaptive-point-treatment-c-tmle
outcome-adaptive-multi-arm-c-tmle
canonical-dr-tmle
multi-arm-dr-tmle
ordinary-missing-outcome-tmle
controlled-direct-effect-tmle
randomized-missing-outcome-dr-tmle
point-treatment-msm-projection
deterministic-point-treatment-regimes
stochastic-point-treatment-regimes
continuous-modified-treatment-policies
incremental-propensity-interventions
ordinary-end-of-study-longitudinal-tmle
ordinary-weighted-end-of-study-longitudinal-tmle
ordinary-categorical-longitudinal-tmle
ordinary-longitudinal-msm-projection
cross-fitted-end-of-study-longitudinal-tmle
cross-fitted-weighted-end-of-study-longitudinal-tmle
cross-fitted-categorical-longitudinal-tmle
ordinary-survival-curve-longitudinal-tmle
cross-fitted-survival-curve-longitudinal-tmle
ordinary-competing-risk-longitudinal-tmle
cross-fitted-competing-risk-longitudinal-tmle
```
