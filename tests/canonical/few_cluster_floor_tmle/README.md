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

Registered. The declared run (artifacts `d4fbd52e`) took 15.3 hours on 4 cores. The floors it
reads ship in `CLUSTER_INTERVAL_FLOORS` for the ATE: in-sample TMLE 6, fold-evaluated TMLE 8,
cross-fitted DR-TMLE with a declared mechanism 8 and in-sample LTMLE 6 (for the always-treat
versus never-treat contrast). Every other estimate keeps the default floor of its kind. The
stacked family keeps 10: the d5 IID control is red at 10 clusters, and
`X54-stacked-d5-redeclaration` holds it. Every other red cell lies below its estimate's floor,
and `X54-cluster-floor` holds it. The property differential reads the committed rows.
`regenerate.py` holds the declaration: the gates,
the floor rule, the part (c) decision, the measured budget and the red-cell readings, owned by
`X54-cluster-floor` in `docs/roadmap.md`.

## Run

The pilot has run. To refit it (it writes `pilot.csv.gz` and six `pilot-*.csv` tables here,
listed in `tests/studies/few_cluster_floor_pilot.py`), or, with `--resize`, to rewrite only
`pilot-sizing.csv` from the committed `pilot-summary.csv`:

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
