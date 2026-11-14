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
is not in `registered()`. Until its run commits, the property differential reads synthetic rows
of the subset the property module declares. `regenerate.py` holds the declaration: the gates,
the floor rule, the part (c) decision, the measured budget and the red-cell readings, owned by
`X54-cluster-floor` in `docs/roadmap.md`.

## Run

Run the pilot after the design review. It writes `pilot.csv.gz` and six `pilot-*.csv` tables
here; `tests/studies/few_cluster_floor_pilot.py` lists them. Commit them, and copy R, the
predicted reds and (C)'s predicted shortfalls into the study module before the run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tests.studies.few_cluster_floor_pilot tmle --jobs "$(nproc --all)"
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
