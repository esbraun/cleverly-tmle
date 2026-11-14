# The boundary-node supplement of LTMLE (`boundary-node-ltmle`)

This study measures the interval of an `LTMLE` estimate that carries the boundary-node
supplement: a node regression whose fitted rows hold one value adds its variance at the exact
one-sided binomial bound of its hazard. It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/boundary_node_ltmle.py` | The record, the cells, the fit, the three arms and the primary phase. |
| `tests/studies/boundary_node_properties.py` | The `boundary_node` family, its readings and its declared rules. |
| `tests/studies/boundary_node_laws.py` | Laws A and B, their declared grids and their exact figures. |
| `tests/studies/boundary_node_pilot.py` | The pilot and its readings; its rows are `pilot.csv.gz`. |
| `tests/unit/test_boundary_node_study_design.py` | The design test: every declared number, recomputed. |

## State of this directory

Declared, not run. No artifact but the pilot is committed, so the record is not in
`registered()`. The evidence page and the grid rows land with the registration, after the run.
`regenerate.py` holds the declaration: the gates, the red-cell readings and the budget.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/boundary_node_ltmle/regenerate.py --replicates 2 --output /tmp/boundary-node-ltmle-smoke
```

Run the declared study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python tests/canonical/boundary_node_ltmle/regenerate.py --output <empty scratch directory> --jobs "$(nproc --all)"
```
