# The clustered interval floor of cross-fitted LTMLE (`few-cluster-floor-cross-fitted-ltmle`)

This study measures the clustered interval of the cross-fitted `LTMLE` at every cluster count
from 4 to 20, plus 30 and 39, on the end-of-study law with equal (40 rows) and unequal (10 to
70 rows) clusters. It sets the family's floor, and it compares the shipped $t_{J-2}$ reference
with Wang et al.'s inflated variance and the delete-one-cluster jackknife
(`cluster_variance="jackknife"`, from 10 clusters) on the same draws. It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/few_cluster_floor_ltmle.py` | The record, the fit, the arms and the primary phase. |
| `tests/studies/few_cluster_floor_ltmle_properties.py` | The property loop, the summary, the floor and the part (c) readings. |
| `tests/studies/few_cluster_floor.py` | What both floor studies share: rows, the floor rule, the part (c) rule and decision, the sizing. |
| `tests/studies/few_cluster_floor_pilot.py` | The pilot, on its own seed stream. |
| `tests/unit/test_few_cluster_floor_design.py` | The design test: every declared number recomputed, a mutation control per rule. |

## State of this directory

Declared for design review. The pilot has not run, and no artifact is committed, so the record
is not in `registered()`. `regenerate.py` holds the declaration: the gates, the floor rule, the
part (c) decision, the measured budget and the red-cell readings.

## Run

Run the pilot (after the design review), then commit its `pilot.csv.gz` here:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tests.studies.few_cluster_floor_pilot ltmle --output <scratch dir> --jobs "$(nproc --all)"
```

Run a disposable smoke of the primary phase:

```bash
python -m tests.canonical.few_cluster_floor_ltmle.regenerate --replicates 2 --output /tmp/few-cluster-floor-ltmle-smoke
```

Run the declared study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.few_cluster_floor_ltmle.regenerate --output <empty scratch directory> --jobs "$(nproc --all)"
```
