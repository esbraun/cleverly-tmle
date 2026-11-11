# Targeted missingness tilt intervals

This study measures the 95% Wald intervals that `missingness_tilt` reports off $\gamma = 0$. At
each declared $\gamma$ the function re-targets the tilted functional
$\psi_\gamma(a) = E_W[\pi_a \bar{Q}_a + (1 - \pi_a)\operatorname{expit}(\operatorname{logit}\bar{Q}_a + \gamma)]$
from the fit's own nuisances. It reads the interval from the tilted influence curve, which
[targeted missingness tilt](../validation-methods.md#targeted-missingness-tilt) derives. Each
interval resolves through the evidence registry with the instrument `missingness_tilt_interval`
and the rule that `_tilt_reading` computes for the fit. A passing cell adds the registry row of
the rule it measured, so the 17 rules of `_tilt_rows()` in
[`validation/_interpretation.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/_interpretation.py)
cite this study.

**No canonical implementation is compared.** No published implementation reports this targeted
tilt. The zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| scenarios | `in_sample` (`TMLE(cross_fit=False)`), `in_sample_4k` (the same fit at $n = 4000$), `cross_fit` (stacked CV-TMLE at the default 10 folds), `cv_evaluation` (`cv_evaluation=True`, 10 folds), `weighted` (in sample, `weights="wt"` with $w = \exp(0.5 W_1)$) and `natural_course` (in sample) |
| targets | `ey1`, `ey0`, `ate`, `att` and `rr` in every scenario except two. `cv_evaluation` omits `rr`, because the fold-evaluated report forms no ratio. `natural_course` reports `ey_obs`, `ey0` and `par` |
| law | `tests/studies/missingness_tilt_law.py`: $W_1 \sim N(0, 1)$ and $W_2 \sim \text{Bernoulli}(0.5)$, a main-terms logistic propensity and response mechanism, and a logistic outcome regression with one $A W_1$ interaction. The binary outcome is missing at random given $(A, W)$, and is rare among the observed rows ($E\,\bar{Q}_0 \approx 0.23$) |
| learners | outcome: `PolynomialFeatures(2, interaction_only=True)`, `StandardScaler` and `LogisticRegression(C=1e4)`, which holds the $A W_1$ term. Treatment and response: main-terms `LogisticRegression(C=1e4)`. Every nuisance is correctly specified |
| primary estimands | each target at $\gamma \in \{-1, 1, 2\}$, named `<target>[gamma=<g>]`, 81 rows. `rr` is read on the log scale |
| primary interval | the targeted estimate plus or minus 1.959964 reported standard errors from the tilted curve |
| truth | the tilted functional of the observed-data law, by an 80-node Gauss-Hermite rule in $W_1$ and a two-point sum in $W_2$. `weighted` reads the truths of the weight-tilted covariate law, which differ from the unweighted ones by more than half an SD at $n = 1000$ for `ey1` and `ate` |
| size | 8,400 replications per scenario at $n = 1000$ ($n = 4000$ for `in_sample_4k`). The calibration family reads 4,000 draws and the targeting-necessity family 1,200 |
| Monte Carlo inference | 99% intervals around every declared endpoint |

The truths are values of the tilted functional of the observed-data law, not causal claims about
an unobserved outcome. At $\gamma = 2$ every target's truth moves from its missing-at-random value
by at least half its SD at $n = 1000$, so those cells test that the tilt is applied and targeted.
At $\gamma = -1$ the contrasts move little (`ate` moves 0.07 SD), so those cells test the
validity of the interval off $\gamma = 0$, not that the tilt was applied. The study module
[`tests/studies/missingness_tilt_inference.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/studies/missingness_tilt_inference.py)
fixed the cells, the learners, the size and the seeds from a pilot on separate seeds before any
run, and `regenerate.py` declares the red-cell readings.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=-1]` | tilted ATE at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000894 to 0.000556 | 0.9530 | 1.0158 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=1]` | tilted ATE at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0014 to 0.000428 | 0.9500 | 1.0138 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=2]` | tilted ATE at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0023 to -0.000357 | 0.9508 | 1.0141 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=-1]` | tilted ATT at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0011 to 0.000548 | 0.9518 | 1.0144 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=1]` | tilted ATT at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0017 to 0.000212 | 0.9512 | 1.0132 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=2]` | tilted ATT at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0026 to -0.000549 | 0.9494 | 1.0171 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=-1]` | tilted arm-0 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000714 to 0.000254 | 0.9476 | 1.0084 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=1]` | tilted arm-0 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.000106 to 0.0014 | 0.9502 | 1.0061 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=2]` | tilted arm-0 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0014 to 0.0029 | 0.9542 | 1.0134 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=-1]` | tilted arm-1 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000962 to 0.000165 | 0.9476 | 1.0124 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=1]` | tilted arm-1 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000378 to 0.000933 | 0.9477 | 1.0100 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=2]` | tilted arm-1 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.000092 to 0.0015 | 0.9463 | 1.0063 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=-1]` | tilted risk ratio, on the log scale at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0023 to 0.0106 | 0.9531 | 1.0107 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=1]` | tilted risk ratio, on the log scale at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0022 to 0.0054 | 0.9510 | 1.0091 | pass |
| stacked CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=2]` | tilted risk ratio, on the log scale at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0068 to -0.000068 | 0.9510 | 1.0124 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=-1]` | tilted ATE at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000989 to 0.000485 | 0.9482 | 0.9985 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=1]` | tilted ATE at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0014 to 0.000423 | 0.9494 | 0.9975 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=2]` | tilted ATE at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0024 to -0.000364 | 0.9518 | 1.0032 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=-1]` | tilted ATT at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0011 to 0.000657 | 0.9501 | 1.0057 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=1]` | tilted ATT at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0016 to 0.000447 | 0.9517 | 1.0080 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=2]` | tilted ATT at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0024 to -0.000280 | 0.9531 | 1.0166 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=-1]` | tilted arm-0 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000527 to 0.000447 | 0.9471 | 1.0028 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=1]` | tilted arm-0 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.000236 to 0.0015 | 0.9490 | 1.0026 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=2]` | tilted arm-0 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0015 to 0.0030 | 0.9500 | 1.0100 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=-1]` | tilted arm-1 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000859 to 0.000275 | 0.9510 | 1.0061 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=1]` | tilted arm-1 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000269 to 0.0010 | 0.9496 | 1.0049 | pass |
| fold-evaluated CV-TMLE, 10 folds, at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=2]` | tilted arm-1 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.000193 to 0.0016 | 0.9489 | 1.0053 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=-1]` | tilted ATE at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0010 to 0.000451 | 0.9485 | 0.9935 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=1]` | tilted ATE at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000928 to 0.000868 | 0.9479 | 0.9951 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=2]` | tilted ATE at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000479 to 0.0015 | 0.9463 | 0.9914 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=-1]` | tilted ATT at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0012 to 0.000481 | 0.9467 | 0.9913 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=1]` | tilted ATT at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000964 to 0.000982 | 0.9479 | 0.9944 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=2]` | tilted ATT at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000340 to 0.0017 | 0.9483 | 0.9941 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=-1]` | tilted arm-0 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000310 to 0.000665 | 0.9451 | 0.9850 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=1]` | tilted arm-0 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0013 to 0.000007 | 0.9440 | 0.9914 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=2]` | tilted arm-0 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0026 to -0.0012 | 0.9448 | 0.9936 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=-1]` | tilted arm-1 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000677 to 0.000472 | 0.9440 | 0.9851 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=1]` | tilted arm-1 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0013 to 0.000002 | 0.9445 | 0.9836 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=2]` | tilted arm-1 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0021 to -0.000664 | 0.9412 | 0.9799 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=-1]` | tilted risk ratio, on the log scale at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.000202 to 0.0086 | 0.9504 | 0.9910 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=1]` | tilted risk ratio, on the log scale at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0020 to 0.0096 | 0.9523 | 0.9944 | pass |
| in-sample TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=2]` | tilted risk ratio, on the log scale at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0032 to 0.0100 | 0.9487 | 0.9910 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=-1]` | tilted ATE at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000251 to 0.000469 | 0.9533 | 1.0105 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=1]` | tilted ATE at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000284 to 0.000605 | 0.9518 | 1.0086 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ate[gamma=2]` | tilted ATE at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000214 to 0.000762 | 0.9523 | 1.0051 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=-1]` | tilted ATT at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000350 to 0.000479 | 0.9508 | 1.0057 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=1]` | tilted ATT at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000362 to 0.000603 | 0.9502 | 1.0046 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `att[gamma=2]` | tilted ATT at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000244 to 0.000763 | 0.9505 | 1.0026 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=-1]` | tilted arm-0 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000138 to 0.000343 | 0.9512 | 1.0014 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=1]` | tilted arm-0 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000388 to 0.000248 | 0.9485 | 0.9984 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=2]` | tilted arm-0 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000716 to -0.000005 | 0.9473 | 0.9960 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=-1]` | tilted arm-1 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000073 to 0.000495 | 0.9501 | 0.9972 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=1]` | tilted arm-1 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000238 to 0.000418 | 0.9482 | 0.9974 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey1[gamma=2]` | tilted arm-1 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000435 to 0.000261 | 0.9475 | 0.9973 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=-1]` | tilted risk ratio, on the log scale at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000604 to 0.0035 | 0.9552 | 1.0101 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=1]` | tilted risk ratio, on the log scale at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000143 to 0.0036 | 0.9507 | 1.0072 | pass |
| in-sample TMLE at n = 4,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `rr[gamma=2]` | tilted risk ratio, on the log scale at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.000231 to 0.0036 | 0.9511 | 1.0037 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=-1]` | tilted arm-0 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000417 to 0.000549 | 0.9444 | 0.9937 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=1]` | tilted arm-0 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0012 to 0.000046 | 0.9455 | 0.9894 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey0[gamma=2]` | tilted arm-0 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0024 to -0.0010 | 0.9450 | 0.9878 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey_obs[gamma=-1]` | tilted observed-outcome mean under the natural course at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000467 to 0.000300 | 0.9481 | 1.0040 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey_obs[gamma=1]` | tilted observed-outcome mean under the natural course at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0011 to -0.000179 | 0.9477 | 0.9980 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `ey_obs[gamma=2]` | tilted observed-outcome mean under the natural course at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0020 to -0.000987 | 0.9468 | 0.9946 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `par[gamma=-1]` | tilted PAR, natural course versus arm 0 at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000557 to 0.000259 | 0.9479 | 0.9915 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `par[gamma=1]` | tilted PAR, natural course versus arm 0 at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000521 to 0.000426 | 0.9480 | 0.9880 | pass |
| in-sample natural-course TMLE at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs | `par[gamma=2]` | tilted PAR, natural course versus arm 0 at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000260 to 0.000726 | 0.9458 | 0.9871 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ate[gamma=-1]` | tilted ATE at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0014 to 0.000566 | 0.9417 | 0.9715 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ate[gamma=1]` | tilted ATE at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000701 to 0.0015 | 0.9421 | 0.9725 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ate[gamma=2]` | tilted ATE at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000055 to 0.0022 | 0.9401 | 0.9745 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `att[gamma=-1]` | tilted ATT at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0020 to 0.000353 | 0.9350 | 0.9543 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `att[gamma=1]` | tilted ATT at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0011 to 0.0015 | 0.9349 | 0.9558 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `att[gamma=2]` | tilted ATT at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000320 to 0.0023 | 0.9364 | 0.9585 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ey0[gamma=-1]` | tilted arm-0 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000076 to 0.0014 | 0.9346 | 0.9663 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ey0[gamma=1]` | tilted arm-0 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0015 to 0.000247 | 0.9350 | 0.9712 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ey0[gamma=2]` | tilted arm-0 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.0023 to -0.000459 | 0.9369 | 0.9737 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ey1[gamma=-1]` | tilted arm-1 mean at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000431 to 0.000925 | 0.9512 | 0.9979 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ey1[gamma=1]` | tilted arm-1 mean at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000926 to 0.000481 | 0.9482 | 0.9965 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `ey1[gamma=2]` | tilted arm-1 mean at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | -0.000986 to 0.000411 | 0.9482 | 0.9961 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `rr[gamma=-1]` | tilted risk ratio, on the log scale at gamma -1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0023 to 0.0118 | 0.9368 | 0.9576 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `rr[gamma=1]` | tilted risk ratio, on the log scale at gamma 1 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0052 to 0.0133 | 0.9430 | 0.9641 | pass |
| in-sample TMLE with weights exp(0.5 W1) at n = 1,000; binary outcome missing at random given A, W1, W2; correctly specified GLMs; truths of the weight-tilted covariate law | `rr[gamma=2]` | tilted risk ratio, on the log scale at gamma 2 | `cleverly` targeted missingness tilt (`missingness_tilt`) | 0.0053 to 0.0122 | 0.9449 | 0.9691 | pass |
<!-- /generated -->

## Theory properties

Both families fit the in-sample construction at $n = 1000$ and read the tilted `ate` at
$\gamma = 2$, the declared tilt farthest from missing at random. The calibration positive must put
its SE-ratio interval in $[0.93, 1.07]$, its coverage interval in $[0.92, 0.98]$, and both
efficiency ratios against the law's exact tilted efficient SD in $[0.90, 1.10]$. The
`shrunken_se_control` multiplies the reported standard error by 0.70, and the `noise_control`
adds one efficiency unit of independent noise to each estimate. The `mar_variance_control` is the
interval the tilt published before the targeted tilt: the untargeted tilted plug-in with the
fit's missing-at-random standard error. On the law the missing-at-random efficient SD of `ate` is
0.81 of the tilted one at $\gamma = 2$, so that control's SE-ratio interval must fall below the
band.

The targeting-necessity family drops the $A W_1$ term from the outcome regression and keeps both
mechanisms correctly specified. The targeted arm's remainder is second order in the regression
error, and its large-$n$ limit is $-0.019$ targeted-arm SDs at $n = 1000$, inside the 0.25
margin. The untargeted tilted plug-in carries a first-order bias, with a limit of $+0.379$ SDs.
The targeted arm's bias must be equivalent to zero, the untargeted arm's must be discriminated
from it, and the displacement must be at least 0.25 targeted-arm SDs.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `ate_gamma2__correctly_specified` | positive | the tilted ATE at gamma 2, in-sample fit at n = 1,000: the targeted tilt's Wald interval, every nuisance correctly specified, read with an independently computed efficiency bound | SE ratio and coverage intervals inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9383 to 0.9566, SE ratio 0.9647 to 1.0200, empirical efficiency ratio 0.9739 to 1.0300, reported efficiency ratio 0.9921 to 0.9953 | pass |
| `interval_calibration` | `ate_gamma2__mar_variance_control` | control | the tilted ATE at gamma 2, in-sample fit at n = 1,000: the untargeted tilted plug-in with the fit's missing-at-random standard error, the interval the tilt published before targeting | the SE-ratio interval must fall below the calibration band | coverage 0.8775 to 0.9031, SE ratio 0.8011 to 0.8468, empirical efficiency ratio 0.9579 to 1.0126, reported efficiency ratio 0.8101 to 0.8123 | pass |
| `interval_calibration` | `ate_gamma2__noise_control` | control | the tilted ATE at gamma 2, in-sample fit at n = 1,000: one efficiency unit of independent noise is added to each estimate | the empirical efficiency-ratio interval must rise above its band | coverage 0.8072 to 0.8385, SE ratio 0.6768 to 0.7163, empirical efficiency ratio 1.3875 to 1.4684, reported efficiency ratio 0.9921 to 0.9954 | pass |
| `interval_calibration` | `ate_gamma2__shrunken_se_control` | control | the tilted ATE at gamma 2, in-sample fit at n = 1,000: the reported standard error is multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8031 to 0.8347, SE ratio 0.6757 to 0.7149, empirical efficiency ratio 0.9730 to 1.0295, reported efficiency ratio 0.6945 to 0.6967 | pass |
| `targeting_necessity` | `ate_gamma2__targeted` | positive | the tilted ATE at gamma 2, in-sample fit at n = 1,000: the outcome regression omits the A W1 term and the tilt is targeted | bias equivalent to zero within the margin | bias -0.0025 to 0.0027, margin 0.0088 | pass |
| `targeting_necessity` | `ate_gamma2__untargeted` | control | the tilted ATE at gamma 2, in-sample fit at n = 1,000: the same regression and the untargeted tilted plug-in | bias discriminated from the targeted arm, displaced by at least 0.25 SDs | bias 0.0117 to 0.0167, margin 0.0085 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The registry gains one
`missingness_tilt_interval` row for each rule below. Each row cites the cells of its rule and the
calibration positive.

| rule | covered configuration |
| --- | --- |
| `level` | `ey1` and `ey0` of an in-sample fit (`in_sample`, `in_sample_4k`), and `ey0` of `natural_course` |
| `contrast` | `ate` of an in-sample fit |
| `conditional` | `att` of an in-sample fit |
| `ratio` | `rr` of an in-sample fit, on the log scale |
| `natural_course` | `ey_obs` of an in-sample fit |
| `par` | `par` of an in-sample fit |
| `level+cross_fit`, `contrast+cross_fit`, `conditional+cross_fit`, `ratio+cross_fit` | the same kinds on a stacked cross-fitted fit at 10 folds |
| `level+cv_evaluation`, `contrast+cv_evaluation`, `conditional+cv_evaluation` | the same kinds on a fold-evaluated fit at 10 folds |
| `level+weighted`, `contrast+weighted`, `conditional+weighted`, `ratio+weighted` | the same kinds on an in-sample fit with fixed observation weights |

Every row covers binary outcomes only: its scope names the outcome family `binomial`, so the
tilt interval of a continuous outcome reads descriptive. Every row also requires `TMLE`, two
arms, influence-curve inference, no declared treatment mechanism, $n \ge 1000$, and the declared
GLM learners above, matched by their fingerprints. No row covers `atc`, `or` or `paf`, so their intervals read descriptive.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 8400 | primary replications per scenario |
| `n` | 1000 | observations per primary replication, except `in_sample_4k` |
| `independent_tests_total` | 81 | tilted-interval tests against truth |
| `independent_tests_passed` | 81 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 6 | repeated-sampling property cells |
| `property_cells_passed` | 6 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0881 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9346 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9274 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9356 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0372 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 84 | the same bound, in primary replications |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control applies to each SE |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement, in targeted-arm SDs |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No published implementation reports the targeted tilt. The row rests on the truth tests and the property cells |
| Correctly specified GLM learners only | The tilted parameter is not doubly robust in $\bar{Q}$, so a flexible or misspecified outcome regression moves the target even with a correct $g$ and $\pi$. A fit with other learners reads descriptive |
| One law, $n \ge 1000$ | A smaller fit reads descriptive. The law has two covariates and a rare binary outcome |
| Binary outcomes only | Every row's scope names the outcome family `binomial`. The tilt interval of a continuous outcome (`+continuous`) reads descriptive |
| Binary treatment only | A multi-arm fit (`+multi_arm`) is covered by no row |
| One cross-fitting draw at the default 10 folds | A fit with `repeats` above 1 reads `+repeats`, which no row covers. The cross-fitted rows were measured at 10 folds only |
| No clusters, strata, estimated weights, `arm_gamma` or targeting variant | `+clustered`, `+stratified`, `+estimated_weights`, `+arm_direction` and `+targeting_variant` are covered by no row |
| `atc`, `or` and `paf` | The study measured `att` and `rr` only, and each row's scope names the estimand it measured. These three read descriptive |
| Regime means and MSM coefficients | They keep the plug-in diagnostic under the `plugin_` names and have no interval off $\gamma = 0$ |
| `DRTMLE` and `CTMLE` | The targeted tilt excludes them, so they keep the diagnostic |
| The weighted cells cover the least | The eight lowest primary coverages are weighted cells, and `min_coverage` is one of them. Each still passed the declared floor |

## Reproduction

The [regeneration script](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/missingness_tilt_inference/regenerate.py)
gives the declared run form and the red-cell readings:

```bash
python -m tests.canonical.missingness_tilt_inference.regenerate --output <empty scratch dir> --jobs <cores>
```

The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/missingness_tilt_inference/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes. The
design test is `tests/unit/test_missingness_tilt_inference_design.py`.
