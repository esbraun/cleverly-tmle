# Stopping rules on point TMLE (`option-stopping-point`)

This study measures the interval of a fit under `stopping="scaled_var"` or
`stopping="sample_size"`, which adds the remaining score to its half-width (Fix-8 plan
sections 2.1 and 5). It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_stopping_point.py` | The record, the streams, the cells and the primary phase. |
| `tests/studies/option_stopping_point_properties.py` | The property families K, W, N and the reported families, and their rules. |
| `tests/studies/_option_stopping_laws.py` | The laws, learners, fits and the threshold scale. |
| `tests/studies/_option_stopping_cells.py` | The cell grid both stopping-rule studies share. |
| `tests/studies/option_stopping_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_option_stopping_point_design.py` | The design test: every declared number, recomputed. |

## State of this directory

Declared, not run. No artifact is committed, so the record is not in `registered()`. The
support rows `T52` and the evidence page land with the registration, after the run.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_stopping_point/regenerate.py --replicates 2 --output /tmp/option-stopping-point-smoke
```

Run the declared study:

```bash
python tests/canonical/option_stopping_point/regenerate.py --output <empty scratch directory>
```
