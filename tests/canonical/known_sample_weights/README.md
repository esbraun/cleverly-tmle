# A declared treatment mechanism beside sample-reading weights (`known-mechanism-sample-weights`)

This study measures `mechanism_law="sample"`: a weighted fit that declares the treatment
mechanism of the rows as drawn estimates the mechanism of the weight-tilted law from the
declaration and a weighted classification, and counts it as estimated (Fix-16 plan sections 2.2,
3 and 7). It reads point TMLE on law SW at n = 1,000, 4,000 and 16,000, LTMLE on the two-node
law LW at n = 4,000, and the old `"target"` reading as a control. It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/canonical_known_sample_weights.py` | The record and the primary phase. |
| `tests/studies/known_sample_weights_properties.py` | The property families, their rules and the smoke gate. |
| `tests/studies/known_sample_weights_law.py` | The laws, learners, fits and exact-law quantities. |
| `tests/studies/known_sample_weights_pilot.py` | The pilot, its committed summary and the sizing functions. |
| `tests/unit/test_known_sample_weights_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_known_mechanism_weights.py`, `tests/unit/test_known_mechanism_weights_longitudinal.py` | The fast-tier exact-law tests and mutation controls. |

## State of this directory

Declared, not run. No artifact is committed, so the record is not in `registered()`. The
support rows and the evidence page land with the registration, after the run.

## Run

The smoke gate, which must print `smoke gate: passed` and exit 0:

```bash
python -m tests.studies.known_sample_weights_properties smoke --draws 2
```

A disposable smoke of the primary phase:

```bash
python tests/canonical/known_sample_weights/regenerate.py --replicates 2 --output /tmp/known-sample-weights-smoke
```

The declared study:

```bash
python tests/canonical/known_sample_weights/regenerate.py --output <empty scratch directory>
```
