# Step 9 coverage of `survey-nonresponse.ipynb`: root cause

Question: the Step 9 five-fold stacked arm-indexed missing-outcome TMLE covered the binary-law truth
on 461 of 500 draws (0.922) in `probes/survey-nonresponse-final/sweep.py`, and `boxcal.py` found
0.928 on 600 draws. Is the standard error too small because of a library defect?

**Answer: no library defect. The two low counts are Monte Carlo noise of the specific seed blocks.**
Oracle constructions on the *same draws* reproduce both low counts. Over 3,000 draws, the library fit
covers 0.946, and over 20,000 draws an oracle covers 0.947.

Probes: `reviews/notebook-review/probes/sn-step9/` (`probe.py`, `oracle.py`, `summarize.py`;
outputs `probe-n2000.csv`, `probe-n4000.csv`, `probe-n4000b.csv`, `probe-n16000.csv`,
`oracle-4000.csv`, `summary.log`, `summary-n4000b.log`, `oracle-4000.log`, `boxcal-check.log`).
Every probe asserts `cleverly.__file__` under this worktree's `src`. Workers are single-threaded.
Truth: ATE 0.186224 (`probes/survey-nonresponse-final/truth.log`, Gauss-Hermite quadrature).

## Constructions compared on each draw

| label | nuisances | targeting |
| --- | --- | --- |
| `lib_cv` | library, 5 folds, sklearn `LogisticRegression` (Step 9 code) | library |
| `lib_in` | library, in sample (`CrossFitting(enabled=False)`) | library |
| `own_cv` | independent unpenalized IRLS logistic, 5 folds | independent pooled two-coefficient logistic fluctuation on respondents |
| `own_in` | independent unpenalized IRLS logistic, in sample | the same |
| `oracle` | the true g, pi, Q of `missing_outcome_binary_dgp` | the same |
| `eif` | none: `truth + mean(EIF at the truth)`, SE `sd(EIF)/sqrt(n)` | none (pure CLT) |

## Findings

1. **[executed] The reported IC is the efficient influence function.** On every draw at every n,
   the library's `ParameterEstimate.influence_curve` equals
   `D = A R/(g pi1)(Y - Q1*) - (1-A) R/((1-g) pi0)(Y - Q0*) + Q1* - Q0* - psi`, rebuilt from the
   fit's own `nuisance.propensity.values`, `nuisance.missingness`, and
   `fluctuations["mean"].targeted.arms`, to a maximum absolute difference of 1.4e-14.
   The reported SE over the `ddof=0` recomputed SE is `sqrt(n/(n-1))` (1.000125 at n = 4000),
   the centered `n - 1` rule. The construction is `counterfactual_means` in
   `src/cleverly/inference/influence.py:845-899` [read]. No term is missing: with correctly specified
   nuisances the EIF is the whole first-order expansion. Pooled targeting across arms and the
   response-model fit add no first-order term.

2. **[executed] Every construction tracks the oracle draw by draw.** Correlation of `lib_cv` with
   `oracle` estimates is 0.993 to 0.999. Coverage, empirical SD, and mean SE agree across all five
   constructions to within 3 draws per 1,000 (`summary.log`).

   | n | draws (seeds) | lib_cv | lib_in | own_cv | oracle | lib_cv SE/SD |
   | --- | --- | --- | --- | --- | --- | --- |
   | 2,000 | 1,000 (1001-2000) | 0.957 | 0.961 | 0.961 | 0.956 | 1.035 |
   | 4,000 | 1,000 (1001-2000) | 0.936 | 0.937 | 0.939 | 0.937 | 0.946 |
   | 4,000 | 2,000 (2001-4000) | 0.951 | 0.950 | 0.951 | 0.951 | 0.996 |
   | 16,000 | 500 (1001-1500) | 0.936 | 0.936 | 0.938 | 0.936 | 0.988 |

   Bias is within one Monte Carlo SE of zero in every cell. Pooled at n = 4000, `lib_cv` covers
   2,837 of 3,000 (0.946). Cross-fitted and in-sample fits do not differ (probe 4).

3. **[executed] The sweep's draws are an unlucky block, and the oracle sees the same block.**
   The sweep fitted the binary law at seeds 1001-1500 (`seed + 1`). On those draws:

   | seeds | lib_cv covers | oracle covers | lib_cv emp SD | mean SE |
   | --- | --- | --- | --- | --- |
   | 1001-1500 (the sweep's draws) | 461 / 500 | 463 / 500 | 0.0202 | 0.0183 |
   | 1501-2000 | 475 / 500 | 474 / 500 | 0.0186 | 0.0184 |

   This probe's `lib_cv` reproduces the sweep's 461 although its fold seed differs (estimate
   correlation 0.998). `boxcal.py` used seeds 5000-5599 (`boxcal_5000.csv`). On those draws the
   oracle TMLE covers 557 / 600 (0.928) and the pure-CLT `eif` construction covers 555 / 600,
   the same as boxcal's 557 (`boxcal-check.log`). The empirical SD exceeds the true SD
   (0.01829, below) on these blocks because of the draws, not the estimator.

4. **[executed] With true nuisances and no estimation, coverage is nominal over many draws.**
   `oracle.py`, 20,000 draws at n = 4000 (seeds 1001-21000): the oracle TMLE covers 0.9472
   (0.9440-0.9503) and the pure-CLT EIF mean covers 0.9483 (0.9452-0.9513). The true SD of the
   EIF from 4,000,000 rows is 1.15689, so the true SE at n = 4000 is 0.018292. The library's mean
   SE is 0.0184. Per consecutive 500-seed block, the pure-CLT construction covers between 459 and
   488. One block of 40 is at or below 461 (`oracle-4000.log`). The sweep drew that block.
   The EIF has skewness 0.88 and kurtosis 20.6; min g pi is about 0.01 per draw. This heavy tail
   widens block-to-block spread and costs at most about 0.003 of coverage at n = 4000.

5. **[executed] n does not reveal a trend (probe 3).** Coverage is 0.957, 0.946 (pooled), and 0.936
   at n = 2000, 4000, 16000. The 16,000 cell is 500 draws on the sweep's seeds 1001-1500; its
   Wald interval 0.915-0.957 includes 0.95.

6. **[read] Registered study `tmle_mar_arm_indexed_cvtmle`.** `properties.csv`,
   `interval_calibration/l1_ate__learned_nuisances`: law L1 (two arms, binary outcome, one
   three-level covariate), n = 2,000, 2,000 replicates, ten folds, depth-five trees that fit the
   saturated nuisances. Coverage 0.950 (0.936-0.962), SE ratio 0.979 (0.941-1.020). The `rr`,
   `or`, `ey0`, `ey1` rows are 0.946 to 0.950. Its 2,000 replicates resolve a coverage gap of
   about 0.014, so a true coverage of 0.922 would fall outside its interval. It would detect a
   variance defect in the stacked construction. It does not exercise continuous covariates or
   small g pi (its cells are tables), so it does not exercise this page's heavy-tailed weights.
   The finding above shows those do not matter here at n = 4000.

7. **[executed] ATE, RR, and OR counts are one result, not three.** On the sweep, the ATE and OR
   covering events agree on all 500 draws and RR on 492. Each is a function of the same two arm
   means, so the identical 461 is not independent confirmation.

## Root cause

Monte Carlo sampling error of the draws. The seeds 1001-1500 (sweep) and 5000-5599 (boxcal) are
low-coverage blocks for any estimator of this law, including the true-EIF sample mean. The
library's IC, SE, and point are correct to 1e-14 against an independent recomputation, and match an
independent TMLE written from scratch.

## Recommendation

Not a library bug. No fix, no failing-first test, and no registered study moves.

For the page (`docs/examples/survey-nonresponse.ipynb`, "How far to trust this"):

- Remove "The three Step 9 intervals each covered on 461 of 500 draws" from the under-coverage
  paragraph. It reads as a defect. The 500-draw count is consistent with nominal coverage once the
  oracle comparison is known.
- If the page keeps a Step 9 number, cite a larger or oracle-anchored run. A suggested statement:
  "Over 3,000 draws, the Step 9 interval covered the ATE on 0.946 of draws. A fit with the true
  nuisance functions covered on 0.947 of 20,000 draws. The 500-draw sweep covered on 0.922, and the
  true nuisance functions covered on 0.926 of the same draws." Cite
  `reviews/notebook-review/probes/sn-step9/` or regenerate the page probe with more seeds.
- Keep the ATE, RR, and OR as one coverage statement, per finding 7.
- The Step 6 under-coverage (0.920, SE/SD 0.935) is a different law with a misspecified linear
  outcome regression at strength 2. This investigation does not cover it, and the same paragraph
  must not cite Step 9 as corroboration.

No technical-reference change is needed. The stacked contract in
`docs/technical-reference/point-treatment-tmle.md:245-305` states the curve this probe verified.
