# Implementation validation studies

Each page here validates one method that `cleverly` implements. A page exists only for a study
registered in `tests/studies/evidence/registry.py`. The shared machinery in
`tests/studies/evidence/` computes every verdict, and `tests/unit/test_method_evidence.py` checks
these pages against the committed results. Every table is generated from those results, so a stale
number is a test failure and not a reading error.

Read the pages in this order.

| page | what it gives you |
| --- | --- |
| [Implementation validation grid](validation-grid.md) | all twelve studies in one table, with the counts and the declared limits |
| [How to read these studies](how-to-read.md) | the three questions, the verdict rules, and the terms every study below applies |
| the twelve study pages | one row per committed test, with what it checked and the verdict its own endpoints produced |

To register a new study, follow
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row).

```{toctree}
:maxdepth: 2

validation-grid
how-to-read
canonical-point-treatment-tmle
stacked-point-treatment-cv-tmle
fold-evaluated-point-treatment-cv-tmle
selector-based-point-treatment-c-tmle
outcome-adaptive-point-treatment-c-tmle
canonical-dr-tmle
ordinary-end-of-study-longitudinal-tmle
cross-fitted-end-of-study-longitudinal-tmle
ordinary-survival-curve-longitudinal-tmle
cross-fitted-survival-curve-longitudinal-tmle
ordinary-competing-risk-longitudinal-tmle
cross-fitted-competing-risk-longitudinal-tmle
```
