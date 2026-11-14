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

Registered from the rerun under declaration `27cd70c3` (review revisions `6c94eeab`, artifacts
`8169c609`). The first declared run (declaration `91cb39ae`, artifacts `cbbd634c`) stays in
history. Its paired clause (ii) compared the plug-in's coverage on firing draws with its coverage
on the non-firing draws of the same cell, which under-cover for the rare-event reason, so it went
red although the supplement gained 304 firing draws and lost none. Clause (ii') replaced it: on
the same firing draws, the coverage of the interval with the flagged node's true variance
(`oracle_node`) minus the plug-in's, with its exact 99% lower end at least 0.05. A withheld
target reads as a miss with a NaN standard error. `regenerate.py` holds the declaration, its
reasons and the budget.

After the run, the statistic label of clause (ii') moved from S7 to S2: it reads a coverage rate
on the control's firing draws against a diagnostic arm, not the positive-control pair, and a lost
draw counts as a firing miss of both arms. The control cell lost no draws, so the verdict and
every committed number are unchanged.

The ungated reading `width_ratio_to_oracle` in `properties.csv` has a defect: its median is not
NaN-aware, so it reads NaN in a cell with a withheld draw (interior_n100, `always` and the
contrast). The fixed formula is the median, over firing draws where both standard errors are
finite, of the supplemented standard error over the `oracle_node` one. The column is published as
run, because `properties.csv` is hashed in `manifest.json` and the harness has no re-analysis path
that leaves that hash describing bytes that ran. The fix lands with the next rerun.

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
