# Node bounds on survival fits (`survival-node-bounds`)

This study fits `LTMLE(node_bounds=..., hazard_pooling="pooled")` on a point-treatment survival law
with five visits. It is Python only for now. The `survtmle` 1.1.1 comparator (P1 with
`method="mean"`, P2 with `method="hazard"`) is declared and pending (`STUDY.pending_reference`).
A later change runs the R runner and registers the comparator runs.

| file | role |
| --- | --- |
| `tests/studies/canonical_survival_node_bounds.py` | The record, the fits, the exceedance readers and the primary phase. |
| `tests/studies/survival_node_bounds_properties.py` | The property families K, W, C3 and N, and their rules. |
| `tests/studies/survival_node_bounds_law.py` | The law, the exact truths, the bounds and the sampler. |
| `tests/unit/test_survival_node_bounds_design.py` | The design test: every declared number and the pinned pilots. |

## State of this directory

No run artifact exists yet. The design is complete and the study is not in `registered()`, because
a registered study needs its committed artifacts. The property module is listed in
`SYNTHETIC_MODULES` of `tests/canonical/property_differential.py`. The change that commits the run
artifacts also adds the registry line, the evidence page and the red-cell owners.

The driver hashes the declared reference sources into the manifest. The R runner is
`tests/canonical/survival_node_bounds/run_study.R`. It is written by inspection and has not run,
because the cloud container has no R.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/survival_node_bounds/regenerate.py --primary-only --replicates 2 --n 2000 --output .tmp/survival-node-bounds-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/survival_node_bounds/regenerate.py
python -m tests.studies.evidence.document --slug survival-node-bounds
```
