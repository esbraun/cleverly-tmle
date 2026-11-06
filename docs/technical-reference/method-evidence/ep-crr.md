# EP learner of the log CRR

This study validates the EP learner of the log conditional risk ratio,
`effect.estimate("ep", sieve=Sieve.default(modifiers, size=k))` on a
`ConditionalContrast(..., scale="log_ratio")` with `model=None`, at the declared schedule
$k(n) = 3$ and 5 cosines per modifier at n = 1,000 and 4,000. The learner has no interval, so
the study measures the integrated squared error of the log CRR against that of the oracle
learner $\theta_{n0}$ of EPL Theorems 3 to 5: the minimiser over the same sieve of the one-step
log-CRR risk at the true nuisances, solved by Newton.
[Conditional effects](../conditional-effects.md#the-ep-learner) gives the estimator and what
its theory claims.

Every gate reads the paired mean over replicates of
$\log \mathrm{ISE}(\text{estimator}) - \log \mathrm{ISE}(\theta_{n0})$. The one-step log-CRR risk
is not convex in general (EPL Section 3.2), so each replicate records whether the oracle's
Hessian is positive definite. A replicate where it is not leaves its group's comparisons, and a
gated row is red when its group excluded more than 1% of its replicates.

**No canonical implementation is compared.** hte3 `Lrnr_crr_EP` at `eac64673` is declared as
the comparator in `tests/studies/ep_crr.py`, and its R runner `hte3_crr_runner.R` has not run.
A zero-row equivalence artifact records the absence of a comparison.

## What was tested

| setting | declaration |
| --- | --- |
| laws | hte3's `generate_data_RR.R` at `eac64673`: $W \sim U(-1, 1)^3$, a binary outcome with $\operatorname{logit}\mu_0(0, w) = -1 + 0.3\sum_k (w_k + \sin 4w_k)$, the log CRR $0.1(-1 + \sum_k w_k)$ (simple) or $0.1(-1 + \sum_k \{w_k + \sin 4w_k\})$ (complex), and moderate ($\operatorname{logit}\pi = \sum_k w_k / 3$) or limited ($\operatorname{logit}\pi = \sum_k w_k$) overlap. $\mu(1, w)$ stays below 0.8, so nothing is clipped |
| modifiers | every covariate, on $[-1, 1]$; the sieve is `Sieve.default` (additive cosines, at most 6 per modifier) |
| fit | `study.identify(ConditionalContrast(modifiers, scale="log_ratio")).estimate("ep", sieve=..., outcome_by_arm=True, n_folds=5)`, every other setting at its default; the offset learner models both arms in one fit (`outcome_by_arm=False`) |
| learners | the classification ensemble per arm for $\mu$ and for $\pi$. On the complex law at limited overlap, two wrong outcome learners with the same $\pi$: an intercept-only learner per arm, which ignores $W$, and the true regression with the treated arm's logit off by $2\{(\sum_k W_k)^2 - 1\}$, whose cross terms lie outside the additive sieve |
| estimators | EP at $k(n)$, the T-learner on each learner and the same sieve, the oracle $\theta_{n0}$, and EP with the size chosen by cross-validation (one group at 1,000). No DR-learner exists for this risk |
| primary rows | the projection TMLE of conditional-projection's `obs_4k` law on this study's own seed, 2,400 replications at n = 4,000: the EP curve has no interval for the harness's primary rows to read |
| budgets | 300 replicates per group, 100 in the cross-validation group |
| seeds | `seed=20261123` and `resampling_seed=20261124`; every group draws from its own labelled stream |
| policy | `gated` |

The laws, groups, learners, margins, controls and seeds were fixed before the run in the P1
plan (section 5.3 and amendments rev 2.4 and 2.5) and `tests/studies/ep_crr.py`. The replicate
counts come from pilots on their own streams (`tests/studies/ep_crr_pilot.csv`,
`ep_crr_growth_pilot.csv`, `ep_crr_offset_pilot.csv` and `ep_crr_hessian_pilot.csv`), and
`tests/unit/test_ep_crr_design.py` pins them and the laws.

## Accuracy against known truth

The primary rows are the projection TMLE of the shared observational law, as in
[ep-cate](ep-cate.md).

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[(intercept)]` | the intercept of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-crr study: the EP curve has no interval for the primary rows to read | -0.000732 to 0.000895 | 0.9513 | 1.0038 | pass |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[W1]` | the W1 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-crr study: the EP curve has no interval for the primary rows to read | -0.0017 to 0.0010 | 0.9558 | 1.0150 | pass |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[W3]` | the W3 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-crr study: the EP curve has no interval for the primary rows to read | -0.0019 to 0.000907 | 0.9571 | 1.0262 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `oracle_comparison` | at n = 4,000, is the one-sided 99% upper bound of EP's mean log ISE ratio below $\log 1.5$ on every law with the ensemble (EPL Theorem 5), and is the bound of its change from 1,000 to 4,000 below $\log 1.25$; with each wrong outcome learner on the complex law at limited overlap, is the ratio's bound below $\log 1.5$, with no growth for the intercept-only learner | each wrong learner's T-learner must read a lower bound above $\log 1.5$ |
| `oracle_hessian`, `nuisance_error` | the share of positive-definite oracle Hessians, and the L2 errors of $\mu$ and $\pi$ and their product | reported with no verdict |

The two rescue cells see different things. The log-ratio debiasing features span a per-arm
logistic model on the sieve, and this law's $\operatorname{logit}\mu_0$ is nearly additive, so
with the intercept-only learner even a debiasing without the $1/\pi$ weight recovers the curve.
That cell shows that the debiasing corrects an intercept-only $\mu$. The offset learner's error
lies outside the sieve, so only the weighted correction removes it: that cell is the study's
witness of the $1/\pi$ weight. Its growth row is reported, not gated, because with $\mu$ wrong
outside the sieve the remainder is first order in the error of $\pi$.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `nuisance_error` | `d3_complex_limited_1k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0715 | reported |
| `nuisance_error` | `d3_complex_limited_1k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0359 | reported |
| `nuisance_error` | `d3_complex_limited_1k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0026 | reported |
| `nuisance_error` | `d3_complex_limited_1k__misspecified_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the L2 error of the outcome regression, the intercept-only outcome learner | reported with no verdict | mean 0.1340 | reported |
| `nuisance_error` | `d3_complex_limited_1k__misspecified_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the L2 error of the propensity, the intercept-only outcome learner | reported with no verdict | mean 0.0359 | reported |
| `nuisance_error` | `d3_complex_limited_1k__misspecified_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the product of the two L2 errors, the intercept-only outcome learner | reported with no verdict | mean 0.0048 | reported |
| `nuisance_error` | `d3_complex_limited_1k__offset_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the L2 error of the outcome regression, the offset outcome learner | reported with no verdict | mean 0.2141 | reported |
| `nuisance_error` | `d3_complex_limited_1k__offset_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the L2 error of the propensity, the offset outcome learner | reported with no verdict | mean 0.0359 | reported |
| `nuisance_error` | `d3_complex_limited_1k__offset_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the product of the two L2 errors, the offset outcome learner | reported with no verdict | mean 0.0077 | reported |
| `nuisance_error` | `d3_complex_limited_4k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0409 | reported |
| `nuisance_error` | `d3_complex_limited_4k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0177 | reported |
| `nuisance_error` | `d3_complex_limited_4k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.000726 | reported |
| `nuisance_error` | `d3_complex_limited_4k__misspecified_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the L2 error of the outcome regression, the intercept-only outcome learner | reported with no verdict | mean 0.1322 | reported |
| `nuisance_error` | `d3_complex_limited_4k__misspecified_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the L2 error of the propensity, the intercept-only outcome learner | reported with no verdict | mean 0.0177 | reported |
| `nuisance_error` | `d3_complex_limited_4k__misspecified_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the product of the two L2 errors, the intercept-only outcome learner | reported with no verdict | mean 0.0023 | reported |
| `nuisance_error` | `d3_complex_limited_4k__offset_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the L2 error of the outcome regression, the offset outcome learner | reported with no verdict | mean 0.2137 | reported |
| `nuisance_error` | `d3_complex_limited_4k__offset_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the L2 error of the propensity, the offset outcome learner | reported with no verdict | mean 0.0177 | reported |
| `nuisance_error` | `d3_complex_limited_4k__offset_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the product of the two L2 errors, the offset outcome learner | reported with no verdict | mean 0.0038 | reported |
| `nuisance_error` | `d3_complex_moderate_1k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0706 | reported |
| `nuisance_error` | `d3_complex_moderate_1k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0404 | reported |
| `nuisance_error` | `d3_complex_moderate_1k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0029 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0400 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0193 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.000775 | reported |
| `nuisance_error` | `d3_simple_limited_1k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0692 | reported |
| `nuisance_error` | `d3_simple_limited_1k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0351 | reported |
| `nuisance_error` | `d3_simple_limited_1k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0024 | reported |
| `nuisance_error` | `d3_simple_limited_4k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0392 | reported |
| `nuisance_error` | `d3_simple_limited_4k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0178 | reported |
| `nuisance_error` | `d3_simple_limited_4k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.000699 | reported |
| `nuisance_error` | `d3_simple_moderate_1k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0670 | reported |
| `nuisance_error` | `d3_simple_moderate_1k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0405 | reported |
| `nuisance_error` | `d3_simple_moderate_1k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0027 | reported |
| `nuisance_error` | `d3_simple_moderate_4k__ensemble_mu_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.0386 | reported |
| `nuisance_error` | `d3_simple_moderate_4k__ensemble_pi_l2` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0193 | reported |
| `nuisance_error` | `d3_simple_moderate_4k__ensemble_product` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.000744 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__ep` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0919, SD 0.2722 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__ep_misspecified` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: EP with the intercept-only outcome learner, which ignores W: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.1153, SD 0.2661 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__ep_offset` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: EP with the true outcome regression whose treated-arm logit is off by 2 {(sum W)^2 - 1}, outside the additive sieve: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.2091, SD 0.4463 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.7784, SD 0.4821 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__t_misspecified` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the T-learner on the intercept-only outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0362, SD 0.7328 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__t_offset` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: the T-learner on that offset outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.3181, SD 0.5681 | reported |
| `oracle_comparison` | `d3_complex_limited_4k__ep` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio -0.0803, 99% upper -0.0625, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_limited_4k__ep_misspecified` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: EP with the intercept-only outcome learner, which ignores W: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio -0.0882, 99% upper -0.0706, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_limited_4k__ep_offset` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: EP with the true outcome regression whose treated-arm logit is off by 2 {(sum W)^2 - 1}, outside the additive sieve: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.2547, 99% upper 0.3010, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_limited_4k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.7691, SD 0.3414 | reported |
| `oracle_comparison` | `d3_complex_limited_4k__t_misspecified` | control | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the T-learner on the intercept-only outcome learner: log ISE against the oracle's, paired by replicate | one-sided 99% lower bound above log 1.5: the control must fail | mean log ratio 0.9187, 99% lower 0.8537, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_limited_4k__t_offset` | control | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: the T-learner on that offset outcome learner: log ISE against the oracle's, paired by replicate | one-sided 99% lower bound above log 1.5: the control must fail | mean log ratio 1.2216, 99% lower 1.1659, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_limited__ep_growth` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio 0.0116, 99% upper 0.0523, margin 0.2231 | pass |
| `oracle_comparison` | `d3_complex_limited__ep_misspecified_growth` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap: the change in the mean log ISE ratio of EP with the intercept-only outcome learner, which ignores W from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio 0.0271, 99% upper 0.0670, margin 0.2231 | pass |
| `oracle_comparison` | `d3_complex_limited__ep_offset_growth` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap: the change in the mean log ISE ratio of EP with the true outcome regression whose treated-arm logit is off by 2 {(sum W)^2 - 1}, outside the additive sieve from n = 1,000 to n = 4,000, from an independent-samples interval | reported with no verdict | mean log ratio 0.0456, SD 0.0325 | reported |
| `oracle_comparison` | `d3_complex_moderate_1k__ep` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0455, SD 0.1810 | reported |
| `oracle_comparison` | `d3_complex_moderate_1k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.5561, SD 0.2934 | reported |
| `oracle_comparison` | `d3_complex_moderate_1k_cv__ep` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000, sieve size chosen by cross-validation: EP with the default sieve size chosen by cross-validation, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0245, SD 0.1561 | reported |
| `oracle_comparison` | `d3_complex_moderate_4k__ep` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 4,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio -0.0296, 99% upper -0.0179, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_moderate_4k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.5283, SD 0.2582 | reported |
| `oracle_comparison` | `d3_complex_moderate__ep_growth` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio 0.0158, 99% upper 0.0429, margin 0.2231 | pass |
| `oracle_comparison` | `d3_simple_limited_1k__ep` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 1,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0696, SD 0.2720 | reported |
| `oracle_comparison` | `d3_simple_limited_1k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.9122, SD 0.5097 | reported |
| `oracle_comparison` | `d3_simple_limited_4k__ep` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 4,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio -0.0556, 99% upper -0.0368, margin 0.4055 | pass |
| `oracle_comparison` | `d3_simple_limited_4k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.8523, SD 0.3965 | reported |
| `oracle_comparison` | `d3_simple_limited__ep_growth` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio 0.0140, 99% upper 0.0552, margin 0.2231 | pass |
| `oracle_comparison` | `d3_simple_moderate_1k__ep` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 1,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0414, SD 0.1787 | reported |
| `oracle_comparison` | `d3_simple_moderate_1k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.6566, SD 0.3348 | reported |
| `oracle_comparison` | `d3_simple_moderate_4k__ep` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 4,000: EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio -0.0282, 99% upper -0.0170, margin 0.4055 | pass |
| `oracle_comparison` | `d3_simple_moderate_4k__t_learner` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.6505, SD 0.3118 | reported |
| `oracle_comparison` | `d3_simple_moderate__ep_growth` | positive | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 5 at n = 1,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio 0.0132, 99% upper 0.0397, margin 0.2231 | pass |
| `oracle_hessian` | `d3_complex_limited_1k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 1,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_complex_limited_4k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and limited overlap, n = 4,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_complex_moderate_1k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_complex_moderate_1k_cv__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 1,000, sieve size chosen by cross-validation: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_complex_moderate_4k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W + sum sin(4 W)) and moderate overlap, n = 4,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_simple_limited_1k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 1,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_simple_limited_4k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and limited overlap, logit pi = sum W, n = 4,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_simple_moderate_1k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 1,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
| `oracle_hessian` | `d3_simple_moderate_4k__oracle_pd` | diagnostic | hte3's log-CRR law with the log CRR 0.1 (-1 + sum W) and moderate overlap, logit pi = sum W / 3, n = 4,000: whether the oracle fit's Hessian is positive definite; a replicate where it is not leaves every comparison of its group | reported with no verdict; a gated row of a group that excluded more than 1% of its replicates is red | mean 1 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, no fit failed, and every oracle
Hessian was positive definite. At n = 4,000 EP's mean log ISE ratio is below 0 on every law
(upper bounds from $-0.063$ to $-0.017$, against $\log 1.5 = 0.405$), and no growth bound
exceeds 0.067, against $\log 1.25 = 0.223$.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 3 of 3 | `performance-tests.csv` |
| gated property cells passed | 13 of 13 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

With the intercept-only learner EP reads $-0.09$ at 4,000 and its T-learner 0.92 (lower bound
0.85). With the offset learner EP reads 0.25 (upper bound 0.30) and its T-learner 1.22 (lower
bound 1.17). The T-learner on the ensemble reads between $-0.91$ and $-0.53$: with a weak
log-CRR signal the oracle carries the pseudo-outcome's noise, so a smooth plug-in can beat it.
On these laws the ensemble cells say little about the debiasing, and the rescue cells carry
that claim.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 3 | parameter tests against the truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 13 | gated repeated-sampling property cells |
| `property_cells_passed` | 13 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0182 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9513 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9388 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9688 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0642 | highest bootstrap primary SE-ratio endpoint |
| `properties[oracle_comparison/d3_simple_moderate_4k__ep]:ratio_upper` | -0.0170 | largest Theorem 5 upper bound |
| `properties[oracle_comparison/d3_complex_limited_4k__ep]:ratio_upper` | -0.0625 | smallest Theorem 5 upper bound |
| `properties[oracle_comparison/d3_complex_limited__ep_misspecified_growth]:ratio_upper` | 0.0670 | largest gated growth bound |
| `properties[oracle_comparison/d3_complex_limited_4k__ep_misspecified]:ratio_upper` | -0.0706 | intercept-only rescue upper bound |
| `properties[oracle_comparison/d3_complex_limited_4k__t_misspecified]:ratio_lower` | 0.8537 | intercept-only T-learner control lower bound |
| `properties[oracle_comparison/d3_complex_limited_4k__ep_offset]:ratio_upper` | 0.3010 | offset rescue upper bound |
| `properties[oracle_comparison/d3_complex_limited_4k__t_offset]:ratio_lower` | 1.1659 | offset T-learner control lower bound |
| `properties[oracle_comparison/d3_complex_limited__ep_offset_growth]:ratio_mean` | 0.0456 | offset growth, reported |
| `properties[oracle_hessian/d3_complex_limited_4k__oracle_pd]:mean_estimate` | 1 | share of positive-definite oracle Hessians |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failures_allowed` | 24 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| No interval | The study measures the curve's ISE against the oracle's, not any interval: the EP curve has none. The intervals of the log-CRR projection come from the projection TMLE |
| The comparator has not run | hte3 `Lrnr_crr_EP` is declared and pending; the row rests on the oracle comparisons and the truth tests |
| Weak signal on the ensemble cells | The log CRR varies by about 0.1 per unit of $W$, so the oracle carries much of the pseudo-outcome's noise and the T-learner beats it. The ensemble cells show EP at the oracle's level, and only the rescue cells show what the debiasing adds |
| One schedule, one sieve, one learner | Every gated cell uses $k(n) = 3, 5$ additive cosines with the ensemble and $V = W$. Another schedule, sieve or modifier set reads supported and rests on EPL's conditions, which the user must judge |
| Laws, sizes and compositions | n is 1,000 and 4,000 on hte3's laws with binary treatment, a binary outcome and complete outcomes. Weights, clusters, missing outcomes, three arms and a known treatment mechanism were not measured and read experimental |
| Fixed size and cross-validation | A fixed size on the log-ratio scale was not measured, and its debiasing does not solve the projection's efficient score, so it reads experimental. The size chosen by cross-validation is reported and reads experimental |

## Reproduction

The declared run is `python -m tests.canonical.ep_crr.regenerate`, whose docstring gives the
run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_crr/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_crr/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_crr/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_crr/properties.csv)
carry every published row.
