# EP learner of the CATE

This study validates the EP learner of the conditional average treatment effect,
`effect.estimate("ep", sieve=Sieve.default(modifiers, size=k))` on a
`ConditionalContrast` with `model=None`, at the declared schedule
$k(n) = 3 + \lfloor \log_2(n / 1000) \rfloor$ cosines per modifier. The learner has no
interval, so the study measures its integrated squared error
$\mathrm{ISE} = \int (\hat\theta - \theta_0)^2\, dP_V$ against that of the oracle learner
$\theta_{n0}$ of EPL Theorems 3 to 5: least squares, on the same sieve, of the doubly robust
pseudo-outcome at the true nuisances. [Conditional effects](../conditional-effects.md#the-ep-learner)
gives the estimator and what its theory claims.

Every gate reads the paired mean over replicates of
$\log \mathrm{ISE}(\text{estimator}) - \log \mathrm{ISE}(\theta_{n0})$, with the ISE computed by
Monte Carlo on 20,000 fixed draws of $W$ per law. A fixed-size cell checks the EP coefficients of
a declared linear model against the projection's truth.

**No canonical implementation is compared.** hte3 `Lrnr_cate_EP` at `eac64673`, with
`targeting_style="dr"`, is declared as the comparator in `tests/studies/ep_cate.py`, and its R
runner `hte3_ep_runner.R` has not run. Its debiasing is EPL's linear Method 2 and this package's
is the logistic Method 1, so the two agree only to $o_p(n^{-1/2})$. A zero-row equivalence
artifact records the absence of a comparison.

## What was tested

| setting | declaration |
| --- | --- |
| laws | hte3's code at `eac64673`, which produced the paper's figures. d = 3: $W \sim U(-1, 1)^3$, baseline $0.5\sum_k W_k + \sum_k \sin 5W_k + \sum_k 1/(W_k + 1.2)$, the simple CATE $1 + W_1$ (moderate overlap, $\operatorname{logit}\pi = \sum_k W_k / 3$) or $1 + \sum_k W_k$ (limited overlap, $\operatorname{logit}\pi = \sum_k W_k$), the complex CATE $1 + \sum_k W_k + \sum_k \sin 5W_k$ under both, and normal noise of SD 1 truncated at $\pm 4$. d = 20: `sim.CATEHighDim(positivity=FALSE)`, with the same truncation. The two rescue laws are the complex d = 3 laws with EPL App B.2's noise, SD 2 truncated at $\pm 8$ |
| modifiers | every covariate, on $[-1, 1]$; the sieve is `Sieve.default` (additive cosines, at most 6 per modifier) |
| fit | `study.identify(ConditionalContrast(modifiers)).estimate("ep", sieve=..., outcome_by_arm=True, n_folds=5, q_bounds=...)`, every other setting at its default |
| learners | the ensemble of a spline GLM, shallow gradient boosting and the lasso per arm for $\mu$, and the classification ensemble for $\pi$; on the rescue laws, and at 4,000 on the complex d = 3 laws, a per-arm main-terms linear outcome learner with the same $\pi$ |
| estimators | EP at $k(n)$, the T-learner and the DR-learner on the same sieve and nuisances, the oracle $\theta_{n0}$, EP with the size chosen by cross-validation (one group at 2,000), and EP with a gradient-boosting final learner (100 replicates of the complex moderate law) |
| fixed size | `ep_fixed_obs_4k`: the observational law of [conditional-projection](conditional-projection.md) at n = 4,000 with its spline GLM learners and `model=Linear(["W1", "W3"])`, 2,400 replicates |
| primary rows | the projection TMLE of `obs_4k` on this study's own seed, 2,400 replications at n = 4,000: the EP curve has no interval for the harness's primary rows to read |
| budgets | 300 replicates per gated d = 3 and rescue group, 200 per d = 20 group, and 100 at the ungated d = 3 size of 2,000 |
| seeds | `seed=20261121` and `resampling_seed=20261122`; every group draws from its own labelled stream |
| policy | `gated` |

The laws, groups, learners, margins, controls and seeds were fixed before the run in the P1
plan (section 5.2 and amendments rev 2.4 and 2.5) and `tests/studies/ep_cate.py`. The replicate
counts come from pilots on their own streams (`tests/studies/ep_cate_pilot.csv`) by the plan's
sizing rule, and `tests/unit/test_ep_cate_design.py` pins them and the laws.

## Accuracy against known truth

The primary rows are the projection TMLE of the shared observational law, the estimator the
fixed-size EP cell is compared with.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[(intercept)]` | the intercept of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-cate study: the EP curve has no interval for the primary rows to read | -0.0012 to 0.000407 | 0.9537 | 1.0168 | pass |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[W1]` | the W1 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-cate study: the EP curve has no interval for the primary rows to read | -0.0025 to 0.000293 | 0.9487 | 1.0022 | pass |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[W3]` | the W3 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-cate study: the EP curve has no interval for the primary rows to read | -0.0013 to 0.0016 | 0.9504 | 0.9890 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `oracle_comparison` | at n = 4,000, is the one-sided 99% upper bound of EP's mean log ISE ratio below $\log 1.5$ on every law with the ensemble (EPL Theorem 5), and is the bound of its change from 1,000 to 4,000 below $\log 1.25$; on the two rescue laws, with the main-terms outcome learner, is the ratio's bound below $\log 3$ with no growth | the T-learner on the main-terms learner must read a lower bound above $\log 3$ on both rescue laws: a debiasing that did nothing would read like it |
| `bias` | at a fixed declared basis, is each EP coefficient's standardized-bias interval inside $\pm 0.25$ | none of its own; the paired EP-minus-TMLE difference is published beside it |
| `nuisance_error`, `diagnostic` | the L2 errors of $\mu$ and $\pi$ and their product, and the paired EP-minus-TMLE difference | reported with no verdict |

A growth row red while its 1,000-row mean is below 0 and the same estimator's 4,000-row gate is
green was declared before the run to read as EP approaching the oracle from below, not as a
defect. No growth row is red.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `ep_fixed_obs_4k_W1` | positive | conditional-projection's observational law at n = 4,000 with its spline GLM learners: the W1 slope of the EP fit with model=Linear(['W1', 'W3']) | standardized-bias interval inside +-0.25 | bias -0.000786 to 0.0020, margin 0.0065 | pass |
| `bias` | `ep_fixed_obs_4k_W3` | positive | conditional-projection's observational law at n = 4,000 with its spline GLM learners: the W3 slope of the EP fit with model=Linear(['W1', 'W3']) | standardized-bias interval inside +-0.25 | bias -0.000606 to 0.0023, margin 0.0068 | pass |
| `bias` | `ep_fixed_obs_4k_intercept` | positive | conditional-projection's observational law at n = 4,000 with its spline GLM learners: the intercept of the EP fit with model=Linear(['W1', 'W3']) | standardized-bias interval inside +-0.25 | bias -0.000445 to 0.0012, margin 0.0039 | pass |
| `diagnostic` | `ep_fixed_obs_4k_W1__ep_minus_tmle` | diagnostic | conditional-projection's observational law, n = 4,000: the W1 slope of the EP fit with model=Linear(['W1', 'W3']): the paired difference of the EP and projection-TMLE W1 slopes on the same draw | reported with no verdict | mean difference 0.000003 to 0.000006 | reported |
| `diagnostic` | `ep_fixed_obs_4k_W3__ep_minus_tmle` | diagnostic | conditional-projection's observational law, n = 4,000: the W3 slope of the EP fit with model=Linear(['W1', 'W3']): the paired difference of the EP and projection-TMLE W3 slopes on the same draw | reported with no verdict | mean difference -0.000001 to 0.000002 | reported |
| `diagnostic` | `ep_fixed_obs_4k_intercept__ep_minus_tmle` | diagnostic | conditional-projection's observational law, n = 4,000: the intercept of the EP fit with model=Linear(['W1', 'W3']): the paired difference of the EP and projection-TMLE intercepts on the same draw | reported with no verdict | mean difference -0.000003 to -7.091e-07 | reported |
| `nuisance_error` | `d20_complex_1k__ensemble_mu_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.4700 | reported |
| `nuisance_error` | `d20_complex_1k__ensemble_pi_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0621 | reported |
| `nuisance_error` | `d20_complex_1k__ensemble_product` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0292 | reported |
| `nuisance_error` | `d20_complex_4k__ensemble_mu_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2573 | reported |
| `nuisance_error` | `d20_complex_4k__ensemble_pi_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0326 | reported |
| `nuisance_error` | `d20_complex_4k__ensemble_product` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0084 | reported |
| `nuisance_error` | `d20_simple_1k__ensemble_mu_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.4549 | reported |
| `nuisance_error` | `d20_simple_1k__ensemble_pi_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0609 | reported |
| `nuisance_error` | `d20_simple_1k__ensemble_product` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0277 | reported |
| `nuisance_error` | `d20_simple_4k__ensemble_mu_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2534 | reported |
| `nuisance_error` | `d20_simple_4k__ensemble_pi_l2` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0334 | reported |
| `nuisance_error` | `d20_simple_4k__ensemble_product` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0085 | reported |
| `nuisance_error` | `d3_complex_limited_1k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.3531 | reported |
| `nuisance_error` | `d3_complex_limited_1k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0361 | reported |
| `nuisance_error` | `d3_complex_limited_1k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0127 | reported |
| `nuisance_error` | `d3_complex_limited_2k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 2,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2999 | reported |
| `nuisance_error` | `d3_complex_limited_2k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 2,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0252 | reported |
| `nuisance_error` | `d3_complex_limited_2k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 2,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0076 | reported |
| `nuisance_error` | `d3_complex_limited_4k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2612 | reported |
| `nuisance_error` | `d3_complex_limited_4k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0178 | reported |
| `nuisance_error` | `d3_complex_limited_4k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0046 | reported |
| `nuisance_error` | `d3_complex_limited_4k__misspecified_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the L2 error of the outcome regression, the main-terms linear outcome learner | reported with no verdict | mean 2.3749 | reported |
| `nuisance_error` | `d3_complex_limited_4k__misspecified_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the L2 error of the propensity, the main-terms linear outcome learner | reported with no verdict | mean 0.0178 | reported |
| `nuisance_error` | `d3_complex_limited_4k__misspecified_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the product of the two L2 errors, the main-terms linear outcome learner | reported with no verdict | mean 0.0422 | reported |
| `nuisance_error` | `d3_complex_limited_noise2_1k__misspecified_mu_l2` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the L2 error of the outcome regression, the main-terms linear outcome learner | reported with no verdict | mean 2.3931 | reported |
| `nuisance_error` | `d3_complex_limited_noise2_1k__misspecified_pi_l2` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the L2 error of the propensity, the main-terms linear outcome learner | reported with no verdict | mean 0.0355 | reported |
| `nuisance_error` | `d3_complex_limited_noise2_1k__misspecified_product` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the product of the two L2 errors, the main-terms linear outcome learner | reported with no verdict | mean 0.0851 | reported |
| `nuisance_error` | `d3_complex_limited_noise2_4k__misspecified_mu_l2` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the L2 error of the outcome regression, the main-terms linear outcome learner | reported with no verdict | mean 2.3787 | reported |
| `nuisance_error` | `d3_complex_limited_noise2_4k__misspecified_pi_l2` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the L2 error of the propensity, the main-terms linear outcome learner | reported with no verdict | mean 0.0172 | reported |
| `nuisance_error` | `d3_complex_limited_noise2_4k__misspecified_product` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the product of the two L2 errors, the main-terms linear outcome learner | reported with no verdict | mean 0.0408 | reported |
| `nuisance_error` | `d3_complex_moderate_1k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.3438 | reported |
| `nuisance_error` | `d3_complex_moderate_1k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0398 | reported |
| `nuisance_error` | `d3_complex_moderate_1k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0137 | reported |
| `nuisance_error` | `d3_complex_moderate_2k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2901 | reported |
| `nuisance_error` | `d3_complex_moderate_2k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0283 | reported |
| `nuisance_error` | `d3_complex_moderate_2k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0082 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2566 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0192 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0049 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__misspecified_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the L2 error of the outcome regression, the main-terms linear outcome learner | reported with no verdict | mean 2.3676 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__misspecified_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the L2 error of the propensity, the main-terms linear outcome learner | reported with no verdict | mean 0.0192 | reported |
| `nuisance_error` | `d3_complex_moderate_4k__misspecified_product` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the product of the two L2 errors, the main-terms linear outcome learner | reported with no verdict | mean 0.0455 | reported |
| `nuisance_error` | `d3_complex_moderate_noise2_1k__misspecified_mu_l2` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the L2 error of the outcome regression, the main-terms linear outcome learner | reported with no verdict | mean 2.3823 | reported |
| `nuisance_error` | `d3_complex_moderate_noise2_1k__misspecified_pi_l2` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the L2 error of the propensity, the main-terms linear outcome learner | reported with no verdict | mean 0.0406 | reported |
| `nuisance_error` | `d3_complex_moderate_noise2_1k__misspecified_product` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the product of the two L2 errors, the main-terms linear outcome learner | reported with no verdict | mean 0.0968 | reported |
| `nuisance_error` | `d3_complex_moderate_noise2_4k__misspecified_mu_l2` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the L2 error of the outcome regression, the main-terms linear outcome learner | reported with no verdict | mean 2.3728 | reported |
| `nuisance_error` | `d3_complex_moderate_noise2_4k__misspecified_pi_l2` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the L2 error of the propensity, the main-terms linear outcome learner | reported with no verdict | mean 0.0194 | reported |
| `nuisance_error` | `d3_complex_moderate_noise2_4k__misspecified_product` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the product of the two L2 errors, the main-terms linear outcome learner | reported with no verdict | mean 0.0459 | reported |
| `nuisance_error` | `d3_simple_limited_1k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2999 | reported |
| `nuisance_error` | `d3_simple_limited_1k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0365 | reported |
| `nuisance_error` | `d3_simple_limited_1k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0110 | reported |
| `nuisance_error` | `d3_simple_limited_2k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 2,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2417 | reported |
| `nuisance_error` | `d3_simple_limited_2k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 2,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0258 | reported |
| `nuisance_error` | `d3_simple_limited_2k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 2,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0062 | reported |
| `nuisance_error` | `d3_simple_limited_4k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2042 | reported |
| `nuisance_error` | `d3_simple_limited_4k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0181 | reported |
| `nuisance_error` | `d3_simple_limited_4k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0037 | reported |
| `nuisance_error` | `d3_simple_moderate_1k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 1,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2911 | reported |
| `nuisance_error` | `d3_simple_moderate_1k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 1,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0391 | reported |
| `nuisance_error` | `d3_simple_moderate_1k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 1,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0114 | reported |
| `nuisance_error` | `d3_simple_moderate_2k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 2,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2366 | reported |
| `nuisance_error` | `d3_simple_moderate_2k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 2,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0264 | reported |
| `nuisance_error` | `d3_simple_moderate_2k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 2,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0062 | reported |
| `nuisance_error` | `d3_simple_moderate_4k__ensemble_mu_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 4,000: the L2 error of the outcome regression, the ensemble | reported with no verdict | mean 0.2023 | reported |
| `nuisance_error` | `d3_simple_moderate_4k__ensemble_pi_l2` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 4,000: the L2 error of the propensity, the ensemble | reported with no verdict | mean 0.0194 | reported |
| `nuisance_error` | `d3_simple_moderate_4k__ensemble_product` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 4,000: the product of the two L2 errors, the ensemble | reported with no verdict | mean 0.0039 | reported |
| `oracle_comparison` | `d20_complex_1k__dr_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 1,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1754, SD 0.1728 | reported |
| `oracle_comparison` | `d20_complex_1k__ep` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 1,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1303, SD 0.1640 | reported |
| `oracle_comparison` | `d20_complex_1k__t_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.9675, SD 0.2060 | reported |
| `oracle_comparison` | `d20_complex_4k__dr_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 4,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0339, SD 0.0897 | reported |
| `oracle_comparison` | `d20_complex_4k__ep` | positive | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 4,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.0230, 99% upper 0.0375, margin 0.4055 | pass |
| `oracle_comparison` | `d20_complex_4k__t_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.8328, SD 0.1399 | reported |
| `oracle_comparison` | `d20_complex__ep_growth` | positive | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + {sin(4 W1) + sin(4 W5) + cos(4 W9) + 1.5 (W15^2 - W10^2)} / 5: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.1073, 99% upper -0.0766, margin 0.2231 | pass |
| `oracle_comparison` | `d20_simple_1k__dr_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 1,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1682, SD 0.1917 | reported |
| `oracle_comparison` | `d20_simple_1k__ep` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 1,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1225, SD 0.1721 | reported |
| `oracle_comparison` | `d20_simple_1k__t_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -1.0604, SD 0.2648 | reported |
| `oracle_comparison` | `d20_simple_4k__dr_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 4,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0308, SD 0.0858 | reported |
| `oracle_comparison` | `d20_simple_4k__ep` | positive | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 4,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.0197, 99% upper 0.0332, margin 0.4055 | pass |
| `oracle_comparison` | `d20_simple_4k__t_learner` | diagnostic | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.9087, SD 0.1293 | reported |
| `oracle_comparison` | `d20_simple__ep_growth` | positive | hte3's sim.CATEHighDim(positivity=FALSE), d = 20, with the CATE 1 + (W1 + W5 + W9 + W10 + W15) / 5: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.1028, 99% upper -0.0714, margin 0.2231 | pass |
| `oracle_comparison` | `d3_complex_limited_1k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 1,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0830, SD 0.1858 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 1,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0764, SD 0.1826 | reported |
| `oracle_comparison` | `d3_complex_limited_1k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0561, SD 0.2514 | reported |
| `oracle_comparison` | `d3_complex_limited_2k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 2,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0543, SD 0.1201 | reported |
| `oracle_comparison` | `d3_complex_limited_2k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 2,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0531, SD 0.1194 | reported |
| `oracle_comparison` | `d3_complex_limited_2k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 2,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0021, SD 0.1758 | reported |
| `oracle_comparison` | `d3_complex_limited_4k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0634, SD 0.1158 | reported |
| `oracle_comparison` | `d3_complex_limited_4k__ep` | positive | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.0603, 99% upper 0.0757, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_limited_4k__ep_misspecified` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: EP with the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 1.4990, SD 0.4370 | reported |
| `oracle_comparison` | `d3_complex_limited_4k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1182, SD 0.2513 | reported |
| `oracle_comparison` | `d3_complex_limited_4k__t_misspecified` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap, n = 4,000: the T-learner on the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 4.2000, SD 0.2846 | reported |
| `oracle_comparison` | `d3_complex_limited__ep_growth` | positive | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and limited overlap: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.0161, 99% upper 0.0129, margin 0.2231 | pass |
| `oracle_comparison` | `d3_complex_limited_noise2_1k__ep_misspecified` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: EP with the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.7335, SD 0.4797 | reported |
| `oracle_comparison` | `d3_complex_limited_noise2_1k__t_misspecified` | diagnostic | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the T-learner on the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 2.1309, SD 0.4231 | reported |
| `oracle_comparison` | `d3_complex_limited_noise2_4k__ep_misspecified` | positive | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: EP with the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 3 | mean log ratio 0.7167, 99% upper 0.7662, margin 1.0986 | pass |
| `oracle_comparison` | `d3_complex_limited_noise2_4k__t_misspecified` | control | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the T-learner on the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | one-sided 99% lower bound above log 3: the control must fail | mean log ratio 2.9385, 99% lower 2.8945, margin 1.0986 | pass |
| `oracle_comparison` | `d3_complex_limited_noise2__ep_misspecified_growth` | positive | the complex d = 3 law at limited overlap with EPL App B.2's noise, SD 2 truncated at +-8: the change in the mean log ISE ratio of EP with the per-arm main-terms linear outcome learner from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.0168, 99% upper 0.0646, margin 0.2231 | pass |
| `oracle_comparison` | `d3_complex_moderate_1k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0689, SD 0.1292 | reported |
| `oracle_comparison` | `d3_complex_moderate_1k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0681, SD 0.1287 | reported |
| `oracle_comparison` | `d3_complex_moderate_1k__ep_greedy_hgb` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: EP with a gradient-boosting final learner on the same nuisances: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.5926, SD 0.2102 | reported |
| `oracle_comparison` | `d3_complex_moderate_1k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0496, SD 0.1511 | reported |
| `oracle_comparison` | `d3_complex_moderate_2k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0433, SD 0.0748 | reported |
| `oracle_comparison` | `d3_complex_moderate_2k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0430, SD 0.0747 | reported |
| `oracle_comparison` | `d3_complex_moderate_2k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0789, SD 0.1184 | reported |
| `oracle_comparison` | `d3_complex_moderate_2k_cv__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 2,000, sieve size chosen by cross-validation: EP with the default sieve size chosen by cross-validation, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0556, SD 0.0944 | reported |
| `oracle_comparison` | `d3_complex_moderate_4k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0328, SD 0.1004 | reported |
| `oracle_comparison` | `d3_complex_moderate_4k__ep` | positive | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.0325, 99% upper 0.0461, margin 0.4055 | pass |
| `oracle_comparison` | `d3_complex_moderate_4k__ep_greedy_hgb` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: EP with a gradient-boosting final learner on the same nuisances: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.7200, SD 0.2235 | reported |
| `oracle_comparison` | `d3_complex_moderate_4k__ep_misspecified` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: EP with the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 1.4648, SD 0.4324 | reported |
| `oracle_comparison` | `d3_complex_moderate_4k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.2253, SD 0.2367 | reported |
| `oracle_comparison` | `d3_complex_moderate_4k__t_misspecified` | diagnostic | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap, n = 4,000: the T-learner on the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 4.3030, SD 0.2764 | reported |
| `oracle_comparison` | `d3_complex_moderate__ep_growth` | positive | hte3's d = 3 law with the CATE 1 + sum W + sum sin(5 W) and moderate overlap: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.0355, 99% upper -0.0136, margin 0.2231 | pass |
| `oracle_comparison` | `d3_complex_moderate_noise2_1k__ep_misspecified` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: EP with the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.6783, SD 0.4385 | reported |
| `oracle_comparison` | `d3_complex_moderate_noise2_1k__t_misspecified` | diagnostic | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 1,000: the T-learner on the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 2.2010, SD 0.3967 | reported |
| `oracle_comparison` | `d3_complex_moderate_noise2_4k__ep_misspecified` | positive | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: EP with the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 3 | mean log ratio 0.6661, 99% upper 0.7153, margin 1.0986 | pass |
| `oracle_comparison` | `d3_complex_moderate_noise2_4k__t_misspecified` | control | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8, n = 4,000: the T-learner on the per-arm main-terms linear outcome learner: log ISE against the oracle's, paired by replicate | one-sided 99% lower bound above log 3: the control must fail | mean log ratio 3.0842, 99% lower 3.0385, margin 1.0986 | pass |
| `oracle_comparison` | `d3_complex_moderate_noise2__ep_misspecified_growth` | positive | the complex d = 3 law at moderate overlap with EPL App B.2's noise, SD 2 truncated at +-8: the change in the mean log ISE ratio of EP with the per-arm main-terms linear outcome learner from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.0123, 99% upper 0.0645, margin 0.2231 | pass |
| `oracle_comparison` | `d3_simple_limited_1k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 1,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1152, SD 0.2559 | reported |
| `oracle_comparison` | `d3_simple_limited_1k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 1,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.1064, SD 0.2525 | reported |
| `oracle_comparison` | `d3_simple_limited_1k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.1405, SD 0.3165 | reported |
| `oracle_comparison` | `d3_simple_limited_2k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 2,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0523, SD 0.1702 | reported |
| `oracle_comparison` | `d3_simple_limited_2k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 2,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0479, SD 0.1716 | reported |
| `oracle_comparison` | `d3_simple_limited_2k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 2,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.1489, SD 0.2961 | reported |
| `oracle_comparison` | `d3_simple_limited_4k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 4,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0494, SD 0.1266 | reported |
| `oracle_comparison` | `d3_simple_limited_4k__ep` | positive | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 4,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.0466, 99% upper 0.0637, margin 0.4055 | pass |
| `oracle_comparison` | `d3_simple_limited_4k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.1945, SD 0.2756 | reported |
| `oracle_comparison` | `d3_simple_limited__ep_growth` | positive | hte3's d = 3 law with the CATE 1 + W1 + W2 + W3 and limited overlap, logit pi = W1 + W2 + W3: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.0598, 99% upper -0.0218, margin 0.2231 | pass |
| `oracle_comparison` | `d3_simple_moderate_1k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 1,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0614, SD 0.1854 | reported |
| `oracle_comparison` | `d3_simple_moderate_1k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 1,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0603, SD 0.1849 | reported |
| `oracle_comparison` | `d3_simple_moderate_1k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 1,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0277, SD 0.1930 | reported |
| `oracle_comparison` | `d3_simple_moderate_2k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 2,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0562, SD 0.1357 | reported |
| `oracle_comparison` | `d3_simple_moderate_2k__ep` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 2,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0558, SD 0.1357 | reported |
| `oracle_comparison` | `d3_simple_moderate_2k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 2,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0245, SD 0.1548 | reported |
| `oracle_comparison` | `d3_simple_moderate_4k__dr_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 4,000: the DR-learner on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio 0.0301, SD 0.1022 | reported |
| `oracle_comparison` | `d3_simple_moderate_4k__ep` | positive | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 4,000: EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners: log ISE against the oracle's, paired by replicate | one-sided 99% upper bound below log 1.5 | mean log ratio 0.0298, 99% upper 0.0436, margin 0.4055 | pass |
| `oracle_comparison` | `d3_simple_moderate_4k__t_learner` | diagnostic | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3, n = 4,000: the T-learner (no debiasing) on the same sieve and ensemble learners: log ISE against the oracle's, paired by replicate | reported with no verdict | mean log ratio -0.0678, SD 0.1814 | reported |
| `oracle_comparison` | `d3_simple_moderate__ep_growth` | positive | hte3's d = 3 law with the CATE 1 + W1 and moderate overlap, logit pi = (W1 + W2 + W3) / 3: the change in the mean log ISE ratio of EP at the declared schedule k(n) = 3, 4, 5 at n = 1,000, 2,000, 4,000, ensemble learners from n = 1,000 to n = 4,000, from an independent-samples interval | one-sided 99% upper bound below log 1.25 | mean log ratio -0.0305, 99% upper -0.0021, margin 0.2231 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. At n = 4,000 the
upper bounds of EP's mean log ISE ratio run from 0.033 (d = 20, simple) to 0.076 (d = 3,
complex, limited overlap), against $\log 1.5 = 0.405$. No law's ratio grows from 1,000 to
4,000: every growth bound is at most 0.065, against $\log 1.25 = 0.223$.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 3 of 3 | `performance-tests.csv` |
| gated property cells passed | 21 of 21 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

On the rescue laws EP with the main-terms outcome learner reads 0.67 (moderate overlap) and
0.72 (limited) at 4,000, upper bounds 0.72 and 0.77 against $\log 3 = 1.10$, while its
T-learner reads 3.08 and 2.94. At hte3's noise SD of 1 the same learner reads 1.46 and 1.50,
above $\log 3$, and its T-learner 4.30 and 4.20. Those rows are diagnostic, as declared: with a
wrong outcome regression the residual $Y - \mu_n$ enters EP's variance and not the oracle's,
so the rescue margin holds at App B.2's noise and not at hte3's.

The fixed-size EP coefficients have standardized biases of at most 0.030, and they differ from
the projection TMLE's on the same draw by about $10^{-6}$ on average. The size chosen by
cross-validation reads 0.056 at 2,000, beside 0.043 for the schedule. The gradient-boosting
final learner reads 0.59 at 1,000 and 0.72 at 4,000, outside the theorems and far above the
sieve. The T-learner on the ensemble is close to the oracle or below it on most laws (down to
$-0.91$ at d = 20), because the spline candidate is near the right family; the rescue cells,
not these, show that the debiasing works.

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
| `property_cells_total` | 21 | gated repeated-sampling property cells |
| `property_cells_passed` | 21 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0415 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9487 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9360 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9544 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0547 | highest bootstrap primary SE-ratio endpoint |
| `properties[oracle_comparison/d3_complex_limited_4k__ep]:ratio_upper` | 0.0757 | largest Theorem 5 upper bound |
| `properties[oracle_comparison/d20_simple_4k__ep]:ratio_upper` | 0.0332 | smallest Theorem 5 upper bound |
| `properties[oracle_comparison/d3_complex_limited__ep_growth]:ratio_upper` | 0.0129 | largest ensemble growth bound |
| `properties[oracle_comparison/d3_complex_limited_noise2_4k__ep_misspecified]:ratio_upper` | 0.7662 | rescue upper bound, limited overlap |
| `properties[oracle_comparison/d3_complex_moderate_noise2_4k__ep_misspecified]:ratio_upper` | 0.7153 | rescue upper bound, moderate overlap |
| `properties[oracle_comparison/d3_complex_limited_noise2_4k__t_misspecified]:ratio_lower` | 2.8945 | T-learner control lower bound, limited overlap |
| `properties[oracle_comparison/d3_complex_moderate_noise2_4k__t_misspecified]:ratio_lower` | 3.0385 | T-learner control lower bound, moderate overlap |
| `properties[oracle_comparison/d3_complex_moderate_4k__ep_misspecified]:ratio_mean` | 1.4648 | main-terms EP at hte3's noise, moderate overlap |
| `properties[oracle_comparison/d3_complex_moderate_2k_cv__ep]:ratio_mean` | 0.0556 | EP with the size chosen by cross-validation |
| `properties[bias/ep_fixed_obs_4k_W3]:standardized_bias` | 0.0304 | largest fixed-size EP bias |
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
| No interval | The study measures the curve's ISE against the oracle's, not any interval: the EP curve has none. The intervals on this axis come from the projection TMLE and `assess_score` |
| The comparator has not run | hte3 `Lrnr_cate_EP` is declared and pending; the row rests on the oracle comparisons and the truth tests |
| Every covariate is a modifier | Each schedule law takes $V = W$. A modifier set smaller than the adjustment set rests on the theory alone (EPL Examples 1 and 2) |
| The rescue depends on the noise | The rescue gates hold at App B.2's noise (SD 2). At hte3's SD of 1 the misspecified EP reads above $\log 3$, so with a wrong outcome regression and little noise the curve can be far from oracle |
| One schedule, one sieve, one learner | Every gated cell uses $k(n) = 3, 4, 5$ additive cosines and the ensemble. Another schedule or sieve reads supported and rests on EPL's conditions, which the user must judge |
| Laws, sizes and outcomes | n runs from 1,000 to 4,000 on hte3's laws with a bounded continuous outcome, binary treatment and complete outcomes. The fixed-size cell uses a binary outcome. Weights, clusters, missing outcomes, three arms and a known treatment mechanism were not measured and read experimental |
| Compositions outside the theorems | The size chosen by cross-validation, the default, and a declared final learner are reported and read experimental; a fixed size with `model=None` was not measured and reads experimental |

## Reproduction

The declared run is `python -m tests.canonical.ep_cate.regenerate`, whose docstring gives the
run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_cate/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_cate/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_cate/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_cate/properties.csv)
carry every published row.
