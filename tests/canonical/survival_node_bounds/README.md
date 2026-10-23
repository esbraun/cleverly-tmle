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

The Python run finished with 7,200 replicates for each model. Its artifacts are committed here, and
the study is in `registered()`. The evidence page is
`docs/technical-reference/method-evidence/survival-node-bounds.md`, and the validation grid has its
row. The support rows `G45.mean_model` and `G45.hazard_model` cite this study.

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
