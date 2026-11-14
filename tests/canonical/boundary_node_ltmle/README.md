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

Redeclared, not yet rerun. The first declared run (declaration `91cb39ae`, artifacts `cbbd634c`)
is committed here and stays in history; the rerun under this declaration overwrites its
artifacts. Its paired clause (ii) compared the plug-in's coverage on firing draws (0.845) with
its coverage on the non-firing draws of the same cell (0.861). Those draws hold one or two events
at node 4 and under-cover for the rare-event reason, so the clause compared the wrong groups, and
it went red although the supplement gained 304 firing draws and lost none.

This redeclaration is informed by that run. Clause (ii') replaces clause (ii): on the same firing
draws, the coverage of the interval with the flagged node's true variance (`oracle_node`) minus
the plug-in's, with its exact 99% lower end at least 0.05. A withheld target now reads as a miss
with a NaN standard error instead of losing its draw, and the rerun uses fresh seeds and stream
labels. Every other cell, gate, margin and owner is unchanged. `regenerate.py` holds the
declaration, its reasons and the budget.

The record is not in `registered()`, and the evidence page and grid rows land with the
registration, after the rerun. The two-replicate smoke below runs the primary phase and writes
the artefacts and the manifest; its primary rows are red by size alone and are not evidence.

## Run

Run each command from the repository root, as a module, so that `tests` imports. The smoke
skips the property study, so it does not exercise the property fits.

Run a disposable smoke of the primary phase:

```bash
python -m tests.canonical.boundary_node_ltmle.regenerate --replicates 2 --output /tmp/boundary-node-ltmle-smoke
```

Run the declared study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.boundary_node_ltmle.regenerate --output <empty scratch directory> --jobs "$(nproc --all)"
```
