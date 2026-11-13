# Per-arm outcome regressions on point TMLE (`option-outcome-by-arm`)

This study measures point TMLE fitted with `outcome_by_arm=True`, one outcome regression per
treatment arm, on laws where a pooled main-terms regression misses an arm-by-covariate
interaction (Fix-8 plan sections 2.2 and 5.3). It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_outcome_by_arm.py` | The record and the primary phase. |
| `tests/studies/option_outcome_by_arm_properties.py` | The property families and their rules. |
| `tests/studies/_option_by_arm_law.py` | The laws, learners, fits and exact-law limits. |
| `tests/studies/option_outcome_by_arm_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_option_outcome_by_arm_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_option_outcome_by_arm_witness.py` | The fast-tier exact-law witnesses and mutation controls. |

## State of this directory

Registered. The declared run is committed here and the record is in `registered()`. The
support rows `T53.in_sample.{binary,continuous}` and `T53.stacked.{binary,bounded}` cover a
per-arm fit that reads the ATE, the means, the risk ratio or the odds ratio, and the two
`outcome_by_arm` conditions are gone from `src/cleverly/estimators/_option_support.py`. The
`T53` limit says that the evidence that the option is honoured is the fast-tier witness and
the spy on the study's fits (`tests/unit/test_option_outcome_by_arm_witness.py`): no study cell
would read differently if `outcome_by_arm=True` were ignored, because the pooled fit is also
unbiased under the correct mechanism. The evidence page is
`docs/technical-reference/method-evidence/option-outcome-by-arm.md`.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_outcome_by_arm/regenerate.py --replicates 2 --output /tmp/option-outcome-by-arm-smoke
```

Run the declared study:

```bash
python tests/canonical/option_outcome_by_arm/regenerate.py --output <empty scratch directory>
```
