# Stopping rules on LTMLE (`option-stopping-ltmle`)

This study measures the interval of a fit under `stopping="scaled_var"` or
`stopping="sample_size"`, which adds the remaining score to its half-width (Fix-8 plan
sections 2.1 and 5). It has no comparator.

On L3 the per-node Newton walk lands at the default fit: its pilot residual is about 7.6e-11 SE
under either rule, and its coverage and SE ratio equal the relative fit's. The L3 cells therefore
show no rule effect, and they license only the Newton walk under a rule, not a one-step
end-of-study walk.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_stopping_ltmle.py` | The record, the streams, the cells and the primary phase. |
| `tests/studies/option_stopping_ltmle_properties.py` | The property families K, W, N and the reported families, and their rules. |
| `tests/studies/_option_stopping_laws.py` | The laws, learners, fits and the threshold scale. |
| `tests/studies/_option_stopping_cells.py` | The cell grid both stopping-rule studies share. |
| `tests/studies/option_stopping_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_option_stopping_ltmle_design.py` | The design test: every declared number, recomputed. |

## State of this directory

Declared, not run. No artifact is committed, so the record is not in `registered()`. The
support rows `G55` and the evidence page land with the registration, after the run.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_stopping_ltmle/regenerate.py --replicates 2 --output /tmp/option-stopping-ltmle-smoke
```

Run the declared study:

```bash
python tests/canonical/option_stopping_ltmle/regenerate.py --output <empty scratch directory>
```
