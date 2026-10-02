# Treatment-learner experiment on `navigation_data`

Date: 2026-10-01. Worktree branch `agent/notebook-review` (= origin/main, 5f33902b), `cleverly`
imported from the worktree `src` (asserted). Scripts, per-seed rows, and logs:
`.tmp/notebook-review/learner-exp/` (`run.py`, `run2.py`, `rows.csv`, `rows_strong.csv`,
`summary.csv`, `summary_strong.csv`, `truth_mc.py`, `strong_truth.py`, `timing.py`).

## Question

Which treatment (propensity) learner gives an honest 95% interval for the ATE on the law behind
`navigation_data` (`nonlinear_bounded_dgp`), in the headline fits of
`docs/examples/point-treatment-tmle.ipynb` and `docs/examples/cross-fitting.ipynb`?

## Truth

| source | ATE |
| --- | --- |
| package truth (`navigation_data(...)[1]["ate"]`, Sobol quadrature) | 0.1628580 |
| independent 10^7-draw Monte Carlo from the structural equations (`truth_mc.py`) | 0.162868 (MC SE 2.1e-5) |
| notebook print | 0.163 |

The two agree within 0.5 MC SE. The package value is used as the truth. Under the true g,
P(g < 0.0114 or g > 0.9886) = 0.32%; E[1/g] diverges (the `-0.4 W2^2` term), so the
untruncated influence-curve variance is infinite.

## Configuration (copied from the notebooks; only the treatment learner varies)

Both headline fits: `CausalStudy(frame, PointTreatment(outcome="transition_score",
treatment="transition_navigation", adjustment=("discharge_risk", "prior_utilization",
"medication_burden", "age")))`, `.identify(ATE(reference=0))`, then

```python
# point-treatment-tmle.ipynb cell 18 (config "point", rs = 21)
TMLEMethod(
    models=ModelSpec(outcome_learner=HistGradientBoostingRegressor(random_state=rs),
                     treatment_learner=<candidate>),
    cross_fitting=CrossFitting(n_folds=5),
    targeting=Targeting(q_bounds=(0.0, 1.0)),
    inference=Inference(alpha=0.05),
    runtime=Runtime(random_state=rs, n_jobs=1),
)
# cross-fitting.ipynb cell 15 (config "cross", rs = 34): identical but without `inference=`
```

`g_bounds` is the default `"auto"`: 5 / (sqrt(n) log n) = 0.0114 at n = 3000 and 0.00486 at
n = 12000 (confirmed from `PositivityReport.bounds`). Sensitivity call (both notebooks):
`result.sensitivity.robustness_value()` with the default `nu2_estimator` (auto -> doubly_robust).

| id | treatment learner (`rs` = config random state) |
| --- | --- |
| a | `HistGradientBoostingClassifier(random_state=rs)` (current notebooks) |
| b | `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=rs)` |
| c | `HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=rs)` |
| d | `LogisticRegression(max_iter=1000, random_state=rs)` (main terms, misspecified) |
| e | `make_pipeline(FunctionTransformer(true_terms), LogisticRegression(C=1e6, max_iter=1000))`, `true_terms` = W1, W2^2, W2*W3, 1(W4>0). On n = 200000 it recovers (0.606, -0.399, 0.495, 0.320) vs truth (0.6, -0.4, 0.5, 0.3). |

Data seeds 5000..5299 (300 per cell, nothing dropped). The `point` and `cross` cells use the
**same** data seeds; they differ only in learner random state and fold split, so they are not
independent replicates (psi correlation 0.86 to 0.96; covers-truth agreement 96 to 99.7%).
Execution: 12-process pool, one thread per process. 3200 fits in 18 min.

## Results, n = 3000

Coverage MC SE = sqrt(p(1-p)/300) (about 0.013 at 95%). Bias MC SE is in parentheses. "trunc any"
is the share of fits where at least one unit hit the g bound. "sens ok" is the share where the
default `robustness_value()` returned instead of raising `CapabilityError` (negative doubly robust
nu^2).

| config | learner | bias (MCSE) | emp SD | mean SE | SE/SD | coverage (MCSE) | trunc any | mean trunc frac | median cal slope | sens ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | a default HGB | -0.0012 (0.0004) | 0.0066 | 0.0113 | 1.72 | 0.993 (0.005) | 1.00 | 0.0072 | 0.47 | 0.06 |
| point | b regularised HGB | -0.0005 (0.0003) | 0.0057 | 0.0056 | 0.99 | 0.940 (0.014) | 0.03 | 0.0000 | 1.00 | 1.00 |
| point | c early-stopping HGB | -0.0005 (0.0003) | 0.0058 | 0.0062 | 1.07 | 0.953 (0.012) | 0.01 | 0.0000 | 0.76 | 1.00 |
| point | d main-terms logit | -0.0016 (0.0003) | 0.0055 | 0.0054 | 0.98 | 0.923 (0.015) | 0.00 | 0.0000 | 0.98 | 1.00 |
| point | e correct logit | 0.0001 (0.0004) | 0.0063 | 0.0060 | 0.95 | 0.930 (0.015) | 1.00 | 0.0035 | 0.99 | 1.00 |
| cross | a default HGB | -0.0012 (0.0004) | 0.0064 | 0.0113 | 1.77 | 0.997 (0.003) | 1.00 | 0.0072 | 0.47 | 0.08 |
| cross | b regularised HGB | -0.0003 (0.0003) | 0.0058 | 0.0056 | 0.97 | 0.933 (0.014) | 0.03 | 0.0000 | 1.01 | 1.00 |
| cross | c early-stopping HGB | -0.0004 (0.0004) | 0.0061 | 0.0062 | 1.02 | 0.950 (0.013) | 0.02 | 0.0000 | 0.76 | 1.00 |
| cross | d main-terms logit | -0.0016 (0.0003) | 0.0056 | 0.0054 | 0.97 | 0.937 (0.014) | 0.00 | 0.0000 | 0.98 | 1.00 |
| cross | e correct logit | 0.0001 (0.0004) | 0.0063 | 0.0060 | 0.95 | 0.933 (0.014) | 1.00 | 0.0035 | 0.99 | 1.00 |

Readings:

- (a) is not honest in the other direction: its SE is 1.7 to 1.8 times the empirical SD, so the
  interval is about 75% too wide and covers 99.3 to 99.7%. Its out-of-fold g is over-extreme
  (calibration slope 0.47, every fit truncates about 0.7% of units), which inflates the estimated
  influence-curve variance but not the estimator's variance. The default `robustness_value()`
  refuses on 92 to 94% of seeds, as the notebooks currently show.
- (b), (c), (d), (e) all have SE/SD within 0.95 to 1.07. (c) is the only one at nominal
  coverage in both configs (95.3%, 95.0%). (b) and (e) sit 1 to 1.7 MC SE below 95%.
- (d) carries a bias of -0.0016, 6 MC SE from zero and about 0.3 SD, from the misspecified g;
  this costs about 2 points of coverage in the point config.

## Results, n = 12000 (point config, seeds 5000..5099)

| learner | g bound | bias (MCSE) | emp SD | mean SE | SE/SD | coverage (MCSE) | trunc any | sens ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b regularised HGB | 0.00486 | -0.0002 (0.0003) | 0.0029 | 0.0026 | 0.89 | 0.930 (0.026) | 0.00 | 1.00 |
| e correct logit | 0.00486 | 0.0000 (0.0003) | 0.0031 | 0.0029 | 0.95 | 0.920 (0.027) | 1.00 | 1.00 |

The shortfall does not shrink with n (93 -> 93% for b, 93 -> 92% for e), but 100 seeds cannot
separate 92% from 95% (MC SE 2.7 points). Bias is zero within 0.1 SD at both n, so truncation bias
does not explain the shortfall. The SE/SD ratio stays below 1 (0.89 and 0.95). That pattern fits
SE instability from a heavy-tailed influence curve with infinite untruncated variance: the
estimated SE is small in the seeds that happen to contain no extreme-weight unit. This is a
diagnosis consistent with the data, not a proof.

## Strong-positivity navigation law

Orchestrator addendum. Law: `dataclasses.replace(nonlinear_bounded_dgp(), propensity=g_nav)` with
`g_nav(w) = 0.05 + 0.90 * expit(0.6 w1 - 0.4 w2^2 + 0.5 w2 w3 + 0.3 (w4 > 0))`, drawn through
`cleverly.datasets.synthetic._make` and renamed with `navigation._PROGRAM_COLUMNS`
(`run2.strong_navigation_data`). Column names and order match `navigation_data`.

| quantity | value |
| --- | --- |
| ATE truth (package quadrature) | 0.1628580, identical to the current law (difference 0.0) |
| ATT / ATC | 0.177342 / 0.150529 (current law: 0.179109 / 0.149271) |
| true nu^2 = E[1/g + 1/(1-g)] (10^7 draws) | 4.830 (MC SE 0.0005) |
| min g over 10^7 draws | 0.0500 |
| P(A=1), seed 21, n = 3000 | 0.465 |

Point-treatment config, n = 3000, seeds 5000..5199 (200 per cell). Candidate (e2) replaces the
exact-form logit with `make_pipeline(StandardScaler(), PolynomialFeatures(2),
LogisticRegression(C=1e6, max_iter=5000, random_state=21))`, which has W2^2 and W2*W3 but not
1(W4>0).

| learner | bias (MCSE) | emp SD | mean SE | SE/SD | coverage (MCSE) | trunc any | mean trunc frac | median cal slope | sens ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a default HGB | -0.0016 (0.0004) | 0.0056 | 0.0103 | 1.84 | 1.000 (0.000) | 0.97 | 0.0026 | 0.43 | 0.075 |
| b regularised HGB | -0.0007 (0.0004) | 0.0051 | 0.0056 | 1.09 | 0.960 (0.014) | 0.005 | 0.0000 | 0.96 | 1.00 |
| c early-stopping HGB | -0.0010 (0.0004) | 0.0053 | 0.0060 | 1.13 | 0.990 (0.007) | 0.00 | 0.0000 | 0.72 | 1.00 |
| e2 degree-2 logit | -0.0008 (0.0004) | 0.0057 | 0.0063 | 1.11 | 0.960 (0.014) | 0.975 | 0.0016 | 0.94 | 0.965 |

Readings:

- The law change does not rescue (a). Its SE/SD is 1.84, it covers in 200 of 200 seeds, and the
  default sensitivity call still refuses on 92.5% of seeds. The over-extreme g (calibration slope
  0.43) is a property of the default HGB, not of the law's positivity.
- On this law every flexible learner's SE overstates the SD by 9 to 13%. (b) and (e2) cover 96%;
  (c) covers 99% (calibration slope 0.72, so it still overfits g a little).
- (e2) truncates at least one unit in 97.5% of fits: the polynomial logit extrapolates g below
  0.0114 for a few units even though the true g never goes below 0.05. The default sensitivity
  call refuses on 3.5% of its seeds.

## Runtime per fit (n = 3000, one process, machine idle, 3 seeds)

| learner | seconds per fit |
| --- | --- |
| a default HGB | 2.25 to 2.38 |
| b regularised HGB | 2.08 to 2.21 |
| c early-stopping HGB | 1.95 to 2.11 |
| d main-terms logit | 1.16 to 1.24 |
| e correct logit | 1.21 to 1.27 |
| e2 degree-2 logit (strong law) | 1.17 to 1.31 |

Under the 12-worker pool the medians were about twice these (4.0 to 4.7 s for HGB, 2.4 s for
logits). At n = 12000 (pool): b 6.2 s, e 3.4 s. The time includes the outcome HGB; the
`robustness_value()` call is negligible. All candidates fit a 60 s notebook.

## Recommendation

On the current `navigation_data` law, use candidate (c):
`HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.2, n_iter_no_change=10,
random_state=<seed>)` in both notebooks' headline fits.

| criterion | (a) current | (b) regularised | (c) early stopping | (d) main logit | (e) correct logit |
| --- | --- | --- | --- | --- | --- |
| coverage, current law (point / cross) | 99.3 / 99.7 (too wide, SE/SD 1.7) | 94.0 / 93.3 | 95.3 / 95.0 | 92.3 / 93.7 (biased) | 93.0 / 93.3 |
| default sensitivity succeeds | 6 to 8% | 100% | 100% | 100% | 100% |
| realism (no knowledge of truth) | yes | hand-picked hyperparameters | yes, data-driven stopping | yes, but misspecified | no, oracle form |
| fit time, idle | 2.3 s | 2.1 s | 2.0 s | 1.2 s | 1.2 s |

(c) is the only candidate at nominal coverage in both configurations, never triggers the default
sensitivity refusal, uses a tuning rule an analyst would choose without the truth, and is the
fastest HGB variant. Any change from (a) removes the notebooks' "the default estimator refused"
branch: both notebooks assert that the refusal happens (`raise AssertionError("the doubly robust
nu^2 was positive on this fit")`), so that cell must be rewritten with the learner.

If the orchestrator switches to the strong-positivity law, (b) or (e2) gives the closest-to-nominal
interval (96%, SE/SD about 1.1), and (c) is honest but conservative (99%, SE/SD 1.13). The law
change gives a finite nu^2 (4.83), but it does not by itself fix the notebooks' problem, which is
the default learner (a). On the evidence here, the learner change is necessary and sufficient on
the current law; the law change is optional. On the current law the remaining shortfall of
(b) and (e) is 1 to 2 coverage points, and the n = 12000 runs do not show it closing; (c) does
not show it at n = 3000.

Caveats: point and cross cells share data seeds, so they are two looks at the same 300 data sets,
not 600 independent replicates. Single-seed notebook outputs can still miss the truth in about
5% of seeds with any honest learner. On the notebooks' own seeds, (c) covers and the default
sensitivity call succeeds:

| notebook seed | psi | SE | 95% CI | covers 0.1629 | rv (default) |
| --- | --- | --- | --- | --- | --- |
| point, seed 21 | 0.1704 | 0.0056 | (0.1594, 0.1814) | yes | 0.450 |
| cross, seed 34 | 0.1545 | 0.0062 | (0.1424, 0.1666) | yes | 0.409 |
