# Treatment-mechanism covariates on point TMLE (`option-treatment-adjustment`)

This study measures point TMLE fitted with `treatment_covariates=` (the low-level form of
`PointTreatment(treatment_adjustment=...)`): the interval under the declared restriction, and
the growing bias of the term the restriction hides when treatment also depends on a left-out
covariate (Fix-8 plan sections 2.4 and 5.4). It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_treatment_adjustment.py` | The record and the primary phase. |
| `tests/studies/option_treatment_adjustment_properties.py` | The property families and their rules. |
| `tests/studies/_option_adjustment_laws.py` | The laws, learners, fits and exact-law limits. |
| `tests/studies/option_treatment_adjustment_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_option_treatment_adjustment_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_option_adjustment_witness.py` | The fast-tier exact-law witnesses and mutation controls. |

## State of this directory

Registered. The declared run is committed here and the record is in `registered()`. The
support rows `T54.in_sample` and `T54.stacked` cover a declared `treatment_covariates=` fit
that reads the ATE, the means, the risk ratio or the odds ratio, and the `treatment_adjustment=`
condition is gone from `src/cleverly/estimators/_option_support.py`. The evidence page is
`docs/technical-reference/method-evidence/option-treatment-adjustment.md`.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_treatment_adjustment/regenerate.py --replicates 2 --output /tmp/option-treatment-adjustment-smoke
```

Run the declared study:

```bash
python tests/canonical/option_treatment_adjustment/regenerate.py --output <empty scratch directory>
```
