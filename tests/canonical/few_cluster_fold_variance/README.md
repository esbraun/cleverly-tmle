# The centred fold cluster variance at few clusters per fold (`few-cluster-fold-variance`)

This study measures the variance that a fold-evaluated `TMLE` (`cv_evaluation=True`) with `id=`
reports, and its interval on $t_{J-V}$, at 2, 4 and 6 clusters per validation fold with
$V = 2$ and $V = 5$, at equal cluster sizes. Cells are read by the clusters each training
complement holds, 2 to 24. For an estimated and a known propensity, it gates the coverage of the
interval, its coverage over every draw, and the lower end of the variance ratio (the mean squared
standard error over the variance of the estimates), and it sets a training-cluster floor for each.
It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/few_cluster_fold_variance.py` | The record, the cells, the fit, the eight arms and the primary phase. |
| `tests/studies/few_cluster_fold_variance_properties.py` | The `fold_variance_calibration` family, its readings and its declared rules. |
| `tests/studies/few_cluster_fold_variance_pilot.py` | The pilot, its readings and the sizing; its rows are `pilot.csv.gz`. |
| `tests/unit/test_few_cluster_fold_variance_design.py` | The design test: every declared number, recomputed. |

## State of this directory

Declared, not run. No artifact but the pilot is committed, so the record is not in
`registered()`. The evidence page and the grid rows land with the registration, after the run.
`regenerate.py` holds the declaration: the design changes from the pilot, the gates, the
training floor, the pilot readings, the predicted red cells, the red-cell readings and the budget.

## Run

Run a disposable smoke of the primary phase:

```bash
python -m tests.canonical.few_cluster_fold_variance.regenerate --replicates 2 --output /tmp/few-cluster-fold-variance-smoke
```

Run the declared study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.few_cluster_fold_variance.regenerate --output <empty scratch directory> --jobs "$(nproc --all)"
```
