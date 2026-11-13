---
orphan: true
---

# Working model over the hazard curve

This study validates longitudinal TMLE with `msm=`, `survival_model="hazard"` and
`targeting="one_step"` on a held design. The fit projects the risk curve of each static arm onto a
working model and targets the coefficient scores of every cause together. The subsection "A
working model over the hazard curve" of
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting)
states the estimand, the influence curve and the exact-law evidence.

The study is declared and has not run. The page is not linked from the index until the run
commits its artifacts and the record is registered, so the three result blocks below are empty.

## The estimand and the laws

A cell is a static arm $r \in \{a_1, a_0\}$ and a visit $t = 1, \dots, T$, with weight
$h = 1$. The working model is $m(c, V; \beta) = \operatorname{expit}(\beta_0 + \beta_1 a_r +
\beta_2 t + \beta_3 a_r W_1)$ with $V = W_1$, and each cause $j$ has its own coefficients, the
solution of

$$
E\Bigl[\sum_c \varphi(c, V)\,\{F_j(t \mid a_r, W) - m(c, V; \beta)\}\Bigr] = 0,
\qquad \varphi(c, V) = (1, a_r, t, a_r W_1).
$$

The parameter is defined whether or not the working model holds, and it does not hold on these
laws. One calibration cell fits the same design under the identity link.

Every law is finite. $W_1 \in \{0, 1\}$ and $W_2 \in \{0, \dots, 3\}$ are independent and
uniform. The arm is logistic in $W$ and the visit dropout is logistic in the arm and $W$. At each
visit each cause occurs with a logistic pseudo-hazard in the visit, the arm, $W_1$, $W_2$ and
$a W_1$.

| law | visits | causes | $P(A = 1 \mid W)$ | what it carries |
| --- | --- | --- | --- | --- |
| `survival` | 5 | one | 0.15 to 0.76 when $W_1 = 0$, 0.45 when $W_1 = 1$ | the primary rows and every gated family |
| `competing` | 4 | relapse and death | as `survival` | the primary rows and calibration |
| `overlap` | 5 | one, the `survival` hazard | 0.055, 0.223, 0.587 and 0.875 at $W_2 = 0, \dots, 3$ when $W_1 = 0$, 0.206 when $W_1 = 1$ | the reported `iterated_mean_difference` family only |

The primary rows report `msm_regimen[(intercept)]`, `msm_regimen[a]`, `msm_regimen[t]` and
`msm_regimen[a_W1]` on `survival`, and the same four terms for each cause on `competing`, as
`msm_regimen[<term>, <cause>]`.

The gated laws have good overlap, and on them the hazard fit and the iterated-mean fit of the
same working model agree. With correctly specified parametric learners both fits are efficient,
so their difference shrinks faster than $n^{-1/2}$. The `overlap` law has the same hazard and a
smallest treatment probability of 0.055. On it the two fits differ by up to about 8% of an
efficiency standard error, and both are efficient, so they cannot differ by more than
$o_P(n^{-1/2})$ on a law where both nuisance fits are correct.

The laws are defined in `tests/studies/hazard_msm_law.py`. They are not the shared laws of
[point-treatment survival](point-treatment-survival.md). On the shared `survival` law the wrong
hazard learner's plug-in coefficient of $a W_1$ lies 0.07 efficiency standard errors from the
truth at n = 2,000, so a necessity cell there could not fail. The study's laws add the $a W_1$
term to each hazard and raise the $W_2$ coefficient.

Every truth is the exact projection of the law's cell risks, computed by a Newton solve that does
not call the estimator. Every efficiency bound is the exact standard deviation of
$D^*_\beta = M^{-1}\sum_c \varphi\,[D^{(1)}_c + F_j(t \mid a_r, W) - m(c, V; \beta)]$ over the
law's support. `tests/unit/test_hazard_msm_study_design.py` checks the truths against a root
finder and against a fit on each law's exact sample. It checks the bounds against the
cause-specific hazard form of the curve at every support point and against the influence curve
of a fit at n = 40,000.

## The fits

| nuisance | correct learner | wrong learner |
| --- | --- | --- |
| hazard | unpenalized logistic regression on `W1`, `W2`, the arm and the arm times `W1`, at each visit and cause | logistic regression on `W1` and the arm (`NoW2Logistic`) |
| treatment | logistic regression on `W1`, `W2` and `W1 W2` | the prior share |
| censoring | logistic regression on `W1`, `W2` and the arm, at each visit | the prior share |

The correct learners are correctly specified and are not saturated. With saturated treatment and
retention fits on a finite $W$, the targeted coefficients of the right and the wrong hazard
learner coincide, so no cell could see the hazard fit. The design test keeps that witness.

Every fit uses one fold, `tol=1e-10` and no simultaneous band, except the `survival` calibration
fits, which also report the simultaneous band.

No maintained package targets the coefficient scores of a working model over discrete-time
hazards, so the study names no comparator and its agreement block stays empty. The exact truths
and bounds, the property families below and `tests/unit/test_hazard_msm.py` are the independent
evidence.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

The primary rows draw 1,600 samples of 2,000 rows for each scenario and report all twelve
coefficients. The count is provisional: the shared primary rule
(`repeated_sizing.primary_replicates`) sets it from the `survival_logit` and `competing_logit`
calibration pilots, which have not run.

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | each coefficient of the three fits, with the shrunken-SE and noise controls on the four `survival` coefficients | 4,000 at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside 0.90 to 1.10 |
| `simultaneous_coverage` | the band over the four `survival` coefficients, and its pointwise control | 4,000 at n = 2,000 | joint coverage outside the calibration band |
| `double_robustness` | the `survival` coefficient of $a W_1$ with both, one or neither nuisance correct, and the `competing` death coefficient of $a W_1$ with the wrong hazard learner and the correct mechanism | 1,200 at n = 2,000 | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the same coefficient from the wrong hazard learner, targeted and plug-in | 1,200 at n = 2,000 | the targeted bias outside the margin, or a displacement below 0.25 |
| `root_n_and_efficiency`, `root_n_rate` | the four `survival` coefficients at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the $a W_1$ coefficient against zero | 800 at n = 2,000 | a rejection lower bound below the minimum power |
| `iterated_mean_difference` | on the `overlap` law: the hazard coefficient minus the iterated-mean coefficient of the same draw, each fit against the truth, and the largest decrease of the iterated-mean risk curve. On the `double_robustness` `mechanism_correct` draws: the same paired difference with both regressions misspecified | 800 at n = 2,000, and the 1,200 robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.smallest_replicates`:
each cell passes with probability at least 0.90, and all thirteen arms pass together with
probability at least 0.95, at a pilot of 8,000 draws that reads a ratio of one. The committed
calibration pilot replaces that reading once it runs. At 4,000 replicates the shrunken-SE control
leaves the SE-ratio band and the noise control leaves the efficiency band with probability above
0.999.

The wrong hazard learner's plug-in coefficient of $a W_1$ has an exact limit 5.9 efficiency
standard errors from the truth at n = 2,000, so the plug-in arm of `targeting_necessity` can fail.
With both nuisances wrong the exact limit lies 2.1 efficiency standard errors from the truth, so
the `both_wrong` control can fail; if it does not clear the margin, the control has failed and
the study is a defect suspect. On `competing` the wrong hazard learner's plug-in death
coefficient of $a W_1$ lies 4.6 efficiency standard errors from the truth, while its targeted
limit is the truth, so that cell reads the clever covariate that carries the relapse hazard into
the death update. The power cell's exact-law power at n = 2,000 is above 0.99.

At bias zero every positive bias arm passes with probability above 0.99. That probability stays
above 0.99 while the true bias is below 0.108 standard deviations of the arm at 1,200 replicates
and below 0.077 at 800. The 200-draw wrong-learner pilot reads a targeted bias of 0.005 standard
deviations for $a W_1$. The calibration pilot reads the other arms when it runs.

The `iterated_mean_difference` rows read the `overlap` law. They measure how far the hazard fit
and the iterated-mean fit of the same working model sit apart, how each fit's interval
calibrates, and how often the iterated-mean risk curve, fitted without a working model and
without isotonic repair, decreases between visits. The summary reports, for each coefficient,
the SD of the paired difference over the efficiency-bound standard error and each fit's coverage
and SE ratio side by side. It also reports the share of decreasing curves with its exact
interval. These rows describe the two constructions; they are not a claim about either.

Before the run, the design test requires the committed 200-draw pilot to reach a
paired-difference ratio of at least 0.05 on the largest coefficient and on $a W_1$. The pilot
reads these ratios:

| coefficient | paired ratio, `overlap` law | paired ratio, `mechanism_correct` draws |
| --- | --- | --- |
| intercept | 0.028 | 0.038 |
| $a$ | 0.077 | 0.070 |
| $t$ | 0.035 | 0.038 |
| $a W_1$ | 0.054 | 0.043 |

So the two fits separate by 3 to 8% of an efficiency standard error on the `overlap` law, and by
4 to 7% when both regressions omit $W_2$ and the mechanism is correct. Neither reading separates
them at first order. A Wald standard-error shortfall on the `overlap` law is shared by the two
fits and small: the 200-draw pilot reads an SE ratio of 0.92 for the intercept in both fits, and
an 800-draw planning run read 0.94 (hazard) and 0.95 (iterated mean) for the arm coefficient. It
is reported and not gated. The share of decreasing curves carries no floor: the pilot found none
in 200 draws, with a 99% interval of 0 to 0.026.

## Limitations

| limit | what it means for use |
| --- | --- |
| one-step targeting with a working model only | the per-horizon algorithm and `isotonic=True` are refused beside `msm=`, and the cell fits without `msm=` are covered by [point survival hazard](point-survival-hazard.md) |
| finite laws with discrete covariates | the evidence covers correct or misspecified low-dimensional logistic learners. A flexible learner is outside it |
| uniform cell weights and $V = W_1$ | a projection weight that varies with $V$ or the cell is covered by the exact-law tests only |
| no comparator | no external implementation targets these coefficient scores |
| no cross-fitting, observation weights or clusters | each is a separate composition and needs its own row |

## Reproduction

Run `python tests/canonical/hazard_msm/regenerate.py --output DIR` with an empty scratch
directory outside the repository to regenerate the artifacts. The pilot runs separately, as
`python -m tests.studies.hazard_msm_pilot --jobs N`, and writes
`tests/canonical/hazard_msm/pilot.csv.gz`. The red-cell readings are declared in
`tests/canonical/hazard_msm/regenerate.py` before any run.
