# Fold-evaluated CV-TMLE with baseline strata

This directory holds the gated repeated-sampling study of roadmap X28 (X11 PR E1). The subject is
`TMLE(cross_fit=True, n_folds=5, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto", estimands=("ate", "ey"), simultaneous=False)`
fitted with `strata=["S"]`, rows as the unit, complete data and `repeats=1`, with
`q_bounds=(0, 1)` on a proportion outcome and no `q_bounds` on a binary one. Each stratum
estimate is the fold average of the stratum plug-ins, each fold's stratum curve divides by the
fold's own stratum share, and the marginal estimate is the share mixture of the stratum
estimates. Each fit reports 12 parameters: `ate`, `ey[0]` and `ey[1]` in strata 0, 1 and 2 and
marginally.

| part | declaration |
| --- | --- |
| law | the RM30 family: `W1 ~ U(-1, 1)`, `W2 ~ Bernoulli(0.5)`, `logit g0 = 0.3 W1 - 0.2 W2`, mean `m = Qbar0(A, W)` with `logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + a b(W1)`, and two outcome laws: `Y ~ Beta(12 m, 12 (1 - m))` with `b = 0.1 + 1.5 W1` (bounded; stratum ATEs -0.220, -0.064, 0.184) and `Y ~ Bernoulli(m)` with `b = 0.1 - 0.3 W1` (binary; stratum ATEs 0.084, 0.043, -0.016). `S` cuts `W1` at -0.6 and 0.1 (shares 0.2, 0.35, 0.45) |
| why two outcome laws | the bounded outcome exists for the share-dropped control: its share term carries about 27% of the marginal `ate` variance, so a fit without it reads an SE ratio of 0.854 and fails the band. Under a binary outcome the residual variance swamps the share term (ratios without it 0.999, 0.995 and 0.999, above the floor 0.93), so no binary label carries that control. The binary settings license the binary outcome, which the support key compares exactly. Their blip keeps every stratum-arm's mean between 0.38 and 0.62, so the smallest stratum-arm (treated rows of stratum 0) holds about 38 rows of the rarer outcome value and the positives test the construction, not sparse events. The first binary pilot used the bounded blip (about 16 events there) and covered `binary_weighted` `ey[1]` in stratum 0 at 0.932, stacked fit alike; that regime is the sparse-event diagnostic's (design record, D1) |
| scenarios | `unweighted` and `weighted` (bounded outcome; pairwise-interaction quasibinomial outcome learner on `(A, W1, W2, S)`), `binary_unweighted` and `binary_weighted` (binary outcome; RM30's logistic outcome learner on the same products). The weight is `w = 1 + 1.5 W2`, which varies inside every stratum. The treatment learner is logistic on `(W1, W2, S)`; every learner is correctly specified. n = 1,000 and 2,000 primary replications per scenario |
| truth | fixed: each stratum mean by the trapezoid rule on 4,001 points of `W1` per stratum for each `W2`, weighted by `w(W2)`; the marginal truth is the share mixture (checked against an adaptive integral to 1e-8) |
| `interval_calibration` | one positive cell per setting and parameter (48), 8,800 draws per setting, each with the shrunken-SE and noise controls; `unnormalised_curve_control` on every stratum parameter (36); `share_dropped_control` on the marginal `ate` and `ey[1]` of the two bounded settings (4) |
| `targeting_necessity` | the bounded rows-only fit with RM30's `drop_w1_learner()` on the quasibinomial solver, against its untargeted fold plug-in, on the nine stratum parameters, 2,800 draws |
| `sparse_event_diagnostic` | role `diagnostic`, no verdict: the binary rows-only law with the steeper blip `0.1 + 3.0 W1` (mean of `ey[1]` in stratum 0 is 0.071), 2,000 draws at n = 1,000. Each draw fits the declared fit and the stacked stratified fit (`cv_evaluation=False`, same learners, same folds). The summary gives both fits' coverage and SE ratio with intervals, and the mean and 5th percentile of the stratum-arm event count |
| verdicts | the framework defaults; the two construction controls are below-band SE controls |
| comparator | none: no pinned comparator reports a fold-averaged stratified estimate |

## What each control shows

The two construction controls are reported apart, because they show different things.
`unnormalised_curve_control` fails by construction: its SE ratio is the stratum share, 0.2 to
0.45. It shows that the instrument resolves a gross normalisation error, not that the fold-local
share is the right one; the fold-local against the global share is witnessed by unit tests only.
`share_dropped_control` tests the marginal derivation, and its power comes from the bounded
outcome. The targeting control's learner sees neither `W1` nor `S`, so its plug-in is nearly the
same in every stratum and its untargeted biases mostly measure that blindness. The cell shows
that targeting repairs a stratum-blind initial fit, not a subtle misspecification within a
stratum.

## What registration must state

These quantities are not on the support key, so the registered rows state them as limits:

- Support rows over the outcome kind, as `fold-evaluated-cvtmle` (T03) does: `binary` from the
  binary cells and `continuous` (bounded) from the beta cells, each with `weighting="fixed"`
  rows only where that outcome has a weighted cell (both do).
- Measured at n = 1,000 with 5 folds and three strata whose smallest share is 0.2, about 200
  rows and about 17 treated rows per stratum-fold. `n_folds` is not on the key. With 10 folds at
  n = 500 the fit refuses about 14% of samples for an empty stratum-fold arm; the refusal is an
  error, never a silent estimate. More folds or smaller strata were not measured.
- On the binary outcome the smallest stratum-arm had about 38 rows of the rarer outcome value on
  average (5th percentile 30), with the sparse-event sentence from `properties.csv` for fewer.
- The share term of the marginal curve is tested on the bounded outcome, where it carries 27% of
  the marginal `ate` variance. It is not separately resolvable on the binary outcome, and the
  formula does not read the outcome type.
- Intervals are pointwise (`simultaneous=False`); no `band` kind is listed.
- `repeats=1` only: with `repeats > 1` the marginal estimate is the median of the draws'
  mixtures, outside the row.
- The two construction controls are counted separately, and the targeting control is described
  as stratum-blind.

The declarations are in `tests/studies/fold_evaluated_strata.py`,
`tests/studies/fold_evaluated_strata_properties.py` and
`tests/studies/_fold_evaluated_strata_law.py`. The run form is the X11 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library. It
fits about 50,000 times (8,000 primary, 35,200 calibration, 2,800 necessity and 4,000 for the
sparse-event pairs) in about 20 minutes on 4 cores; allow two hours with the summaries:

```bash
python -u -m tests.canonical.fold_evaluated_strata.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository. `--property-harness-check 3` also fits three draws of every property cell and prints the solver warnings:

```bash
python -u -m tests.canonical.fold_evaluated_strata.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 3
```
