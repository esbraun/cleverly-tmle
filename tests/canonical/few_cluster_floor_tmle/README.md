# The clustered interval floor of point TMLE and in-sample LTMLE (`few-cluster-floor-tmle`)

This study measures the clustered interval of point `TMLE`, `DRTMLE` and in-sample `LTMLE` at
every cluster count of a ladder below each family's current floor (4 to 10 clusters for point
fits, 4 to 20 for `LTMLE`, with 15, 20, 30 and 39 above), on an equal-size and an
informative unequal-size law. It sets each family's floor, and it compares the shipped
$t_{J-2}$ reference with Wang et al.'s inflated variance and the delete-one-cluster jackknife
(`cluster_variance="jackknife"`) on the same draws. It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/few_cluster_floor_tmle.py` | The record, the fits, the arms and the primary phase. |
| `tests/studies/few_cluster_floor_tmle_properties.py` | The property loop, the summary, the floors and the part (c) readings. |
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
  python -m tests.studies.few_cluster_floor_pilot tmle --output <scratch dir> --jobs "$(nproc --all)"
```

Run a disposable smoke of the primary phase:

```bash
python -m tests.canonical.few_cluster_floor_tmle.regenerate --replicates 2 --output /tmp/few-cluster-floor-tmle-smoke
```

Run the declared study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.few_cluster_floor_tmle.regenerate --output <empty scratch directory> --jobs "$(nproc --all)"
```
