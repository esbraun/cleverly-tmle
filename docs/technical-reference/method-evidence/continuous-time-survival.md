# Continuous-time survival TMLE

This study validates `ContinuousSurvivalTMLE` in `cleverly.survival`, the one-step TMLE of
Rytgaard and van der Laan (2024) for the absolute risks of competing causes in continuous time.
The parameters are the risk $F_j(t)$ of each cause $j \in \{1, 2\}$ under each arm of a binary
baseline treatment, and the difference between the arms, at the median and upper quartile of the
observed event times. [Continuous-time survival and competing risks](../continuous-time-survival.md) states the
estimator, its path and its stopping rule.

No maintained package implements this construction, so no canonical implementation is compared.
R `concrete` 1.0.5 implements an earlier form and is archived; the X6 roadmap item records why it
is not used. The study rests on exact truths and on repeated-sampling properties.

## What was fitted

| scenario | law, nuisances and estimands | folds |
| --- | --- | --- |
| `both_cf_1k` | the base law below. The default libraries: three Cox models for each cause hazard and for the censoring hazard (on the treatment alone, on main terms, and with treatment interactions), chosen by cross-validated partial likelihood, and a main-terms and L1 logistic selection for the treatment. The risk of each cause under each arm and the two risk differences at $t = 1.12$ and $t = 2.32$, n = 1,000 | ten |

The base law is van Hage et al. (2025), scenario 1, with one stated departure. $X_1, X_2$ indicate
the levels of a three-level factor, $X_3 \sim N(0, 1)$, and $X_4 = 1\{X_3 < 1\}$. The treatment is
$Z \sim \operatorname{Bern}\{\operatorname{expit}(X_1 - X_2 + X_3)\}$. Given $(Z, X)$ the two causes
have constant hazards
$h_1 = \exp(-1.5 + X_1 - X_2 + 0.5 X_3 - Z + Z X_4)$ and
$h_2 = \exp(-2 - X_1 + X_2 - 0.5 X_3 - 0.5 Z)$, and censoring is exponential with rate 0.15.

The departure is the arm-specific term $Z X_4$, which only the interaction member of the default
library contains, so the library holds the truth through that member alone. The target times
$0.46, 1.12, 2.32$ are the quartiles of the observed event times. The law and its truths are in
`tests/studies/continuous_survival_law.py`; the truths come from quadrature over $X_3$ for each
factor level, and the efficient spreads from the same rule over the efficient influence curve.
`tests/unit/test_continuous_survival_design.py` checks both against Monte Carlo on the law.

The record gates the two later times. The first time's six targets come from the same fits and
are published without a verdict in [the early time](#the-early-time), because the pilot found
their standard error low before the run was declared.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.0015 to 0.000374 | 0.9449 | 1.0038 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.0020 to -0.000004 | 0.9486 | 1.0132 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000358 to 0.0013 | 0.9330 | 0.9878 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | 0.000191 to 0.0021 | 0.9446 | 1.0022 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000585 to 0.0010 | 0.9396 | 0.9919 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000356 to 0.0012 | 0.9416 | 0.9887 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000523 to 0.000459 | 0.9479 | 1.0033 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000921 to 0.000279 | 0.9485 | 1.0065 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000946 to 0.000215 | 0.9466 | 0.9999 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.0013 to 0.000147 | 0.9482 | 1.0013 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | -0.000244 to 0.0011 | 0.9231 | 0.9782 | pass |
| a three-level factor and a normal covariate, a binary treatment, two competing causes with constant hazards and independent censoring (van Hage et al. 2025, scenario 1, with an arm-specific term); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted (Rytgaard and van der Laan 2024) | 0.000045 to 0.0016 | 0.9420 | 1.0048 | pass |
<!-- /generated -->

Every primary row passes. Coverage is 0.923 to 0.949 and the SE ratio 0.978 to 1.013 over
7,586 fits; 14 of the 7,600 draws failed with a recordable data failure, inside the declared bound
of 76.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `treated_c1__both_wrong` | control | the continuous-time competing-risks law with covariate-dependent censoring, the treated arm's risk of cause 1 by t = 1.12, ten outer folds, n = 1,000: the event hazards, the treatment model and the censoring hazard all wrong: no sufficient condition holds | bias interval must fall entirely outside the margin | bias 0.0158 to 0.0173, margin 0.0040, SE ratio 1.0311 | pass |
| `double_robustness` | `treated_c1__mechanism_correct` | positive | the continuous-time competing-risks law with covariate-dependent censoring, the treated arm's risk of cause 1 by t = 1.12, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library | bias interval inside the equivalence margin | bias 0.000077 to 0.0016, margin 0.0041, SE ratio 1.0343 | pass |
| `double_robustness` | `treated_c1__outcome_correct` | positive | the continuous-time competing-risks law with covariate-dependent censoring, the treated arm's risk of cause 1 by t = 1.12, ten outer folds, n = 1,000: the default event-hazard library, with the treatment model wrong (it omits the normal covariate) and the censoring hazard wrong (a Cox model on the treatment alone) | bias interval inside the equivalence margin | bias -0.0014 to -0.000016, margin 0.0038, SE ratio 1.0399 | pass |
| `interval_calibration` | `both_rd__correctly_specified` | positive | the continuous-time competing-risks law, the difference in the risk of cause 1 by t = 1.12 between treatment and no treatment, ten outer folds, n = 1,000: the default libraries, each of which holds the true model, read against the exact efficient influence-curve spread with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands, with both efficiency-ratio intervals inside their bands | coverage 0.9330 to 0.9542, SE ratio 0.9584 to 1.0251, empirical efficiency ratio 0.9884 to 1.0568, reported efficiency ratio 1.0086 to 1.0177 | pass |
| `interval_calibration` | `both_rd__noise_control` | control | the continuous-time competing-risks law, the difference in the risk of cause 1 by t = 1.12 between treatment and no treatment, ten outer folds, n = 1,000: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8142 to 0.8486, SE ratio 0.6870 to 0.7358, empirical efficiency ratio 1.3773 to 1.4744, reported efficiency ratio 1.0086 to 1.0176 | pass |
| `interval_calibration` | `both_rd__shrunken_se_control` | control | the continuous-time competing-risks law, the difference in the risk of cause 1 by t = 1.12 between treatment and no treatment, ten outer folds, n = 1,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8041 to 0.8393, SE ratio 0.6721 to 0.7180, empirical efficiency ratio 0.9882 to 1.0542, reported efficiency ratio 0.7061 to 0.7124 | pass |
<!-- /generated -->

The robustness cells read the treated arm's risk of cause 1 at $t = 1.12$, the risk that the
arm-specific term moves, on the base law with censoring hazard $0.15 \exp(0.5 X_3 + 0.5 Z)$, so
that a wrong censoring model is wrong. `mechanism_correct` replaces each cause hazard by the Cox
model on the treatment alone. `outcome_correct` keeps the default hazard library and fits the
treatment on $X_1, X_2$ only and the censoring hazard on the treatment alone. With all three wrong
the standardized bias is 0.82, outside the margin, so the cells can fail.

The calibration cell reads cause 1's risk difference at $t = 1.12$ against the exact efficient
spread and the efficiency band 0.90 to 1.10. The two shared controls are derived from its rows.
`tests/studies/continuous_survival_properties.py` declares every cell.

## Reported readings

These readings carry no verdict.

### The early time

`early-time.csv` holds the six targets at $t = 0.46$ from the primary fits. Every estimate is
unbiased (standardized bias within 0.035), and its spread matches the efficient one, but the
influence-curve standard error runs below the sampling spread. The treated arm's risk of cause 2
reads SE ratio 0.924 and coverage 0.881, the control arm's risk of cause 1 reads 0.962 and 0.919,
and the other four read SE ratios 0.965 to 1.005 and coverage 0.924 to 0.945. The fit's support
label reads experimental at a time this early (see [limits](#limits)), and the
[roadmap](../../roadmap.md#x6-continuous-time-survival-and-competing-risks) carries the finding.

### Reported cells

`reported-cells.csv` holds four cells fitted outside the gated configuration. On the two later
times:

| cell | fits | coverage | SE ratio |
| --- | --- | --- | --- |
| `both_cf_4k`, cross-fitted at n = 4,000 | 400 | 0.922 to 0.960 | 0.930 to 1.026 |
| `both_fs_1k`, in sample at n = 1,000 | 1,000 | 0.900 to 0.950 | 0.920 to 1.001 |
| `both_fs_4k`, in sample at n = 4,000 | 400 | 0.928 to 0.978 | 0.923 to 1.080 |
| `positivity_4k`, Hage's treatment law S2 at n = 4,000 | 351 | 0.459 to 0.966 | 0.307 to 1.025 |

At the early time the in-sample fit at n = 1,000 covers down to 0.855. `positivity_4k` draws the
treatment with coefficients $(6, -1, 1)$, which puts a third of the units below 0.025 or above
0.975; 49 of its 400 draws failed and the rest under-cover badly, as expected.

### The walk

`targeting-readings.csv` gives, per scenario, the steps, the share of fits that met the stopping
rule, and the floor shares. Every primary fit met the rule, in 3.2 steps on average and 14 at
most. The share of (unit, arm) pairs held still for a step because the step would have taken
their survival below the floor averaged 0.14%.

## Sizing

Every size was fixed before the run from the pilot in `tests/studies/continuous_survival_pilot.py`,
on seed streams no declared cell reads. The primary count, 7,600, follows the shared rule of
`tests/studies/repeated_sizing.py` with one implementation. The calibration count, 3,200, is the
smallest multiple of 400 whose family-wise pass probability reaches the shared share, read from
8,000 pilot draws. The robustness count, 4,800, is the smallest multiple of 400 at which each
positive cell passes with probability above 0.98 at its pilot's upper bias bound; `mechanism_correct`
sets it. The design test recomputes all three.

## Red-cell route

The study is gated, and no cell is red. The declared readings are in
`tests/canonical/continuous_survival/regenerate.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 7600 | primary replications |
| `n` | 1000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 12 | truth tests passing |
| `independent_tests_total` | 12 | truth tests reported |
| `property_cells_passed` | 6 | property cells passing |
| `property_cells_total` | 6 | property cells reported |
| `max_standardized_bias` | 0.0355 | largest primary standardized bias |
| `min_coverage` | 0.9231 | lowest primary coverage |
| `properties[double_robustness/treated_c1__both_wrong]:standardized_bias` | 0.8222 | standardized bias of the all-wrong control |
| `properties[interval_calibration/both_rd__correctly_specified]:se_ratio` | 0.9914 | SE ratio of the calibration cell |
| `properties[interval_calibration/both_rd__correctly_specified]:coverage` | 0.9443 | coverage of the calibration cell |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:standardized_bias` | 0.2000 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 76 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |

## Limits

| limit | what it means for use |
| --- | --- |
| one law, the default libraries, n = 1,000 | the evidence covers the default Cox libraries and treatment learner, cross-fitted, on one law; a replaced library or learner reads experimental |
| times by which 40% to 85% of the observed events have happened | the gated times sit at 50% and 75%. At 25% the standard error ran up to 8% low and coverage fell to 0.88, so an estimate at an earlier time reads experimental, as does one beyond 85% |
| two causes | a single event type, or more than two, reads experimental |
| risks and risk differences | risk ratios and the simultaneous band are not measured and read experimental |
| in-sample fits | coverage fell to 0.900 at the gated times and 0.855 at the early time; an in-sample fit reads experimental |
| near-violations of positivity | `positivity_4k` covered as little as 0.37; check the treatment probabilities before trusting an interval |
| a binary baseline treatment and baseline covariates | other treatments and time-varying covariates are refused |
| no comparator | no maintained implementation of this construction was located |

## Reproduction

`tests/canonical/continuous_survival/regenerate.py` regenerates the study and states the run
form. It runs no reference process, because the study has no comparator. The manifest records the
seeds, the margins, the estimator configuration and the source and result hashes, and the run log
records the commit, the clean tree and the wall time.
