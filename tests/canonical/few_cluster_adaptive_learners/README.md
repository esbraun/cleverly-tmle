# Few-cluster intervals with the default learner library (`few-cluster-adaptive-learners`)

This study measures the Student $t_{J-2}$ interval of a clustered `TMLE` and a cross-fitted
`LTMLE` whose nuisances are fitted by the default learner library, at 10, 20 and 40 clusters. A
configuration that fails the declared caveat rule takes the `adaptive_nuisance` shortfall of its
support row (`cleverly._support_rows.ADAPTIVE_NUISANCE_SHORTFALLS`). It has no comparator.

| file | role |
| --- | --- |
| `tests/studies/few_cluster_adaptive_learners.py` | The record, the laws, the fits, the arms and the primary phase. |
| `tests/studies/few_cluster_adaptive_properties.py` | The `adaptive_learners` family, its readings and its declared rules. |
| `tests/studies/clustered_nonlinear_law.py` | Law L2 and its exact truth. |
| `tests/studies/few_cluster_adaptive_pilot.py` | The 400-draw pilot, with checkpoints, and its predictive pass probabilities. |
| `tests/unit/test_few_cluster_adaptive_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_adaptive_nuisance_support.py` | The qualifier, the shortfall table and the label it moves. |

## State of this directory

Designed, not declared. The 400-draw pilot that sizes the study has not run, so no artifact is
committed, the record is not in `registered()` and the shortfall table is empty. After the
pilot, its `pilot.csv.gz` is committed here, the design test recomputes the pass probabilities
from it, and the redeclaration in `regenerate.py` states the predicted red cells. The evidence
page, the grid rows and the shortfall entries land with the registration, after the run.

## Run

Run the pilot (about 20 CPU-hours, about 1.5 hours on 16 cores; it checkpoints each chunk under
`<directory>/chunks/`, and the same command resumes a stopped run):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.studies.few_cluster_adaptive_pilot --output <directory> --jobs "$(nproc --all)"
```

Run a disposable smoke of the primary phase:

```bash
python -m tests.canonical.few_cluster_adaptive_learners.regenerate --replicates 2 --output /tmp/few-cluster-adaptive-learners-smoke
```

Run the declared study, only after the redeclaration:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.few_cluster_adaptive_learners.regenerate --output <empty scratch directory> --jobs "$(nproc --all)"
```
