# C-TMLE nested cross-fitting cost

## Question

Selection-fold nuisances and candidate propensities must not be scored on predictions from models
that saw the scored row. The `tmle3`/`sl3` design uses fold fits for held-out training predictions
and a full-training fit for new rows. What does applying that convention inside each C-TMLE
selection fold cost, and how many inner folds are a usable default?

## Method

Measured 2026-08-09 on Windows 11, Python 3.13.7, Intel Family 6 Model 186. Each cell is a complete
`CTMLE.fit(...).single()` on `make_instrument(n=2000, seed=seed)` for seeds 0, 1, and 2, using
`estimands=("ate",)`, direct linear/logistic sklearn nuisance estimators, default selection folds,
and otherwise default settings. An unrecorded `n=200` ordered fit warmed imports and caches before
each implementation run.

The baseline was `origin/main` at `411c575`. The fully nested exploratory version reused the outer
`n_folds=10`, requiring ten inner fold fits plus one full-selection-training fit. The proposed
default uses `selection_inner_folds=2`, requiring two fold fits plus the full fit.

## Results

| search | implementation | seconds, seeds 0/1/2 | median | vs baseline |
| --- | --- | --- | ---: | ---: |
| ordered | `origin/main` (in-sample selection training) | 14.872, 11.743, 11.835 | 11.835 | 1.00× |
| ordered | 10 inner folds + full fit | 48.788, 44.707, 46.127 | 46.127 | 3.90× |
| ordered | 2 inner folds + full fit | 18.479, 18.475, 19.965 | 18.479 | 1.56× |
| greedy | `origin/main` (in-sample selection training) | 11.866, 11.626, 11.939 | 11.866 | 1.00× |
| greedy | 10 inner folds + full fit | 45.765, 45.166, 48.768 | 45.765 | 3.86× |
| greedy | 2 inner folds + full fit | 20.602, 19.213, 18.874 | 19.213 | 1.62× |

## Decision

Nested cross-fitting stays: row-ID spy tests establish that neither an outer validation row nor an
inner training row is predicted by a model that included it. Reusing the outer ten-fold setting is
not a viable hidden default—it makes even this three-covariate GLM example take about 46 seconds,
before substituting a Super Learner. `selection_inner_folds=2` is therefore public and defaults to
two. It preserves the fold-fit/full-fit construction at a measured 56–62% overhead here, and lets
callers pay linearly for more inner folds when their learner and sample size justify it.

These are wall-clock measurements for this machine and dependency environment, not portable
performance promises. Their durable conclusion is the relative cost shape and the need for an
explicit inner-fold control.
