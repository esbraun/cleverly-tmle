# Small-sample Wald intervals

This study measures a small-sample rule for the 95% Wald interval of a TMLE estimate in a small
stratum or a sparse arm. A bounded binary-outcome mean or difference takes the normal interval on
the range-logit scale, `Transform.range_logit(*bounds)` with `cleverly.inference.wald_ci`. Any
other binary-outcome target keeps the normal interval on its own scale. A continuous-outcome
target takes a Student t interval at `max(small_sample_df, 5)`. No interval is shown when an arm
the estimate reads holds one observed outcome value in its stratum
(`cleverly.inference.constant_outcome_cells`), or when a bounded estimate lies on or outside its
bounds.

The study computed the rule from the public functions (`tests/studies/small_sample_wald.py`,
`rule_interval`), so it measured the rule before any default read it. **Only bounded means are
supported**, and the default now reads the rule's logit interval for them. The run refutes the
rule for bounded differences with one arm near its bound and for skewed continuous outcomes, and
the logistic working-model coefficients of a small stratum stay red on their SE. [Result](#result)
names each refuted class and its owner. For the two refuted classes the default reads the
asymmetric intervals that [asymmetric-wald](asymmetric-wald-intervals.md) measured on new held-out
laws.

**No canonical implementation is compared.** No published implementation applies a range-logit
or moment-df interval to a TMLE influence curve with these withholding triggers, so no comparison
was run. The absence does not block the row, and a zero-row equivalence artifact records it.

## What was tested

| setting | declaration |
| --- | --- |
| primary | 1,000 fits of the binary-outcome law with three unequal strata of `canonical-stratified-tmle` at n = 2,000, its `ey[1][V=2]` (truth 0.9358) and `ate[V=2]`. The primary rows publish the rule's interval, a withheld one as the point, so the primary measures the correction. The shipped interval of the same fits is in `primary-shipped.csv`, which no verdict reads |
| owner and diagnosis cells | `boundary_mean` at n = 2,000 and 8,000 on the law of the primary; the logit-MSM treatment coefficient and the identity-MSM W coefficient of stratum 2 on the alternating laws of the stratified MSM studies; and four diagnosis laws that chose the rule: a rare binary outcome (`rare_ey0`), a Student t error on 3 df (`student_ey1`, `student_ate`) and a lognormal error of skewness 2.0 (`lognormal06_ey1`) |
| held-out cells | three laws declared before any fit on them, each with exact truths. H1: a sparse treated arm with no strata, treated mean near 1. H2: a continuous outcome with a point mass at 0 and a gamma part, in stratum V = 2 of three, fitted with a saturated cell-mean outcome model. H3: a logistic working MSM $(1, a, W)$ in stratum 2 of about 300 rows. `tests/studies/small_sample_laws.py` defines every law |
| non-H1 cells | law L of `drtmle-one-inconsistent-nuisance` at n = 4,000 under a wrong outcome regression: two arm means with a cross-fitted logistic treatment model, and a control whose treatment model converges slower than root n |
| reported rows | the same laws at n = 500 and 1,000, the in-sample and cross-fitted random-forest pair, the no-harm cells of three owner studies, the DR-TMLE stratum, the lognormal law of skewness 6.2, and every control re-reading of the gated cells |
| verdicts | a positive coverage cell passes when its lower leg's 99% Clopper-Pearson lower end is at least 0.92, counting lost and withheld fits as misses, and its shown fits' 99% upper end is at most 0.98. `boundary_mean` and `student_ey1` also need the exact conditional 99.5% lower bound of the paired gain over the normal interval above 1/2. A non-H1 positive cell needs its lower leg's lower end at least 0.92 and the control its coverage's upper end below 0.90 |
| property budget | 2,000 replications per group (1,000 for the forest pair and the slow-mechanism control) |
| seeds | `seed=2026111201` and `resampling_seed=2026111202`; every group draws from its own stream, keyed by its label |
| policy | `reporting` |

The groups, laws, sizes, fits, controls, falsifiers and design numbers were fixed before the run,
at commit `34ed366`. At perfect calibration the gated cells, both paired gains and the non-H1 pair
pass together with probability 0.952. At the exploratory readings the designed chance of at least
one red gated cell was 0.094, most of it `lognormal06_ey1` (0.057).
`tests/unit/test_small_sample_wald_design.py` recomputes every design number from the law module.
`tests/canonical/small_sample_wald/regenerate.py` states the declaration and the red-cell reading.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the binary-outcome law with three unequal baseline strata of `canonical-stratified-tmle`, whose stratum V = 2 has a treated mean near 1, n = 2,000 | `ate[V=2]` | average treatment effect in stratum V = 2, with the rule's interval (a withheld one counts as a miss) | `cleverly` stratified TMLE with the small-sample Wald rule: a range-logit normal interval for a bounded binary mean or difference, the normal interval on its own scale for any other binary-outcome target, a Student t interval at max(df, 5) for a continuous outcome, and no interval when an arm's stratum holds one outcome value or the estimate sits on a bound | -0.0019 to 0.0030 | 0.9570 | 1.0368 | pass |
| the binary-outcome law with three unequal baseline strata of `canonical-stratified-tmle`, whose stratum V = 2 has a treated mean near 1, n = 2,000 | `ey[1][V=2]` | mean outcome under treatment in stratum V = 2, with the rule's interval (a withheld one counts as a miss) | `cleverly` stratified TMLE with the small-sample Wald rule: a range-logit normal interval for a bounded binary mean or difference, the normal interval on its own scale for any other binary-outcome target, a Student t interval at max(df, 5) for a continuous outcome, and no interval when an arm's stratum holds one outcome value or the estimate sits on a bound | -0.0018 to 0.0017 | 0.9540 | 0.9783 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `corrected_coverage` | the rule repairs the owner cells it targets, at the owner's law | either coverage leg leaves (0.92, 0.98), or the paired gain does not exceed 1/2 |
| `diagnosis_bounded`, `diagnosis_continuous` | the rule covers on the laws that chose it | either coverage leg leaves the band |
| `heldout_coverage` | the rule covers on laws no estimator was fitted on before the declaration | either coverage leg leaves the band |
| `non_h1` | the rule does not hide a wrong nuisance: it covers with one inconsistent nuisance at root-n, and fails when the mechanism is slower | the positive cells fall below 0.92, or the control's upper end reaches 0.90 |
| `control` | each gated cell's fits read under the normal interval, a Kish-count t, a range-logit t and a moment-df t | reported; the falsifiers of `falsifiers.csv` read them on the held-out cells |
| `reported`, `witness` | the rule at other sizes and fits, and the plug-in interval on the constant-cell fits it withholds | reported |

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `control` | `boundary_mean__capped_t_binary_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9322 to 0.9586, shown 0.9322 to 0.9586 | reported |
| `control` | `boundary_mean__kish_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9468 to 0.9700, shown 0.9468 to 0.9700 | reported |
| `control` | `boundary_mean__logit_t_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9587 to 0.9789, shown 0.9587 to 0.9789 | reported |
| `control` | `boundary_mean__normal_reference_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9069 to 0.9380, shown 0.9069 to 0.9380 | reported |
| `control` | `boundary_mean_n8000__capped_t_binary_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 8,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9322 to 0.9586, shown 0.9322 to 0.9586 | reported |
| `control` | `boundary_mean_n8000__kish_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 8,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9372 to 0.9626, shown 0.9372 to 0.9626 | reported |
| `control` | `boundary_mean_n8000__logit_t_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 8,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9428 to 0.9670, shown 0.9428 to 0.9670 | reported |
| `control` | `boundary_mean_n8000__normal_reference_control` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 8,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9233 to 0.9515, shown 0.9233 to 0.9515 | reported |
| `control` | `ho_hurdle_ate__kish_control` | diagnostic | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, effect in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9422 to 0.9665, shown 0.9433 to 0.9674 | reported |
| `control` | `ho_hurdle_ate__normal_reference_control` | diagnostic | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, effect in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9211 to 0.9497, shown 0.9222 to 0.9506 | reported |
| `control` | `ho_hurdle_ey1__kish_control` | diagnostic | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9305 to 0.9573, shown 0.9316 to 0.9582 | reported |
| `control` | `ho_hurdle_ey1__normal_reference_control` | diagnostic | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.8993 to 0.9317, shown 0.9003 to 0.9325 | reported |
| `control` | `ho_logit_msm_a__capped_t_binary_control` | diagnostic | held-out law H3, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9278 to 0.9551, shown 0.9278 to 0.9551 | reported |
| `control` | `ho_logit_msm_a__kish_control` | diagnostic | held-out law H3, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9344 to 0.9604, shown 0.9344 to 0.9604 | reported |
| `control` | `ho_logit_msm_a__normal_reference_control` | diagnostic | held-out law H3, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9195 to 0.9484, shown 0.9195 to 0.9484 | reported |
| `control` | `ho_logit_msm_w__capped_t_binary_control` | diagnostic | held-out law H3, logistic working MSM (1, a, W), W coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.8729 to 0.9092, shown 0.8729 to 0.9092 | reported |
| `control` | `ho_logit_msm_w__kish_control` | diagnostic | held-out law H3, logistic working MSM (1, a, W), W coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.8842 to 0.9189, shown 0.8842 to 0.9189 | reported |
| `control` | `ho_logit_msm_w__normal_reference_control` | diagnostic | held-out law H3, logistic working MSM (1, a, W), W coefficient in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.8638 to 0.9013, shown 0.8638 to 0.9013 | reported |
| `control` | `ho_sparse_treat_ate__capped_t_binary_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, effect, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9278 to 0.9551, shown 0.9278 to 0.9551 | reported |
| `control` | `ho_sparse_treat_ate__kish_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, effect, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9434 to 0.9674, shown 0.9434 to 0.9674 | reported |
| `control` | `ho_sparse_treat_ate__logit_t_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, effect, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9311 to 0.9578, shown 0.9311 to 0.9578 | reported |
| `control` | `ho_sparse_treat_ate__normal_reference_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, effect, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9096 to 0.9403, shown 0.9096 to 0.9403 | reported |
| `control` | `ho_sparse_treat_ey1__capped_t_binary_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, treated mean, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9322 to 0.9586, shown 0.9322 to 0.9586 | reported |
| `control` | `ho_sparse_treat_ey1__kish_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, treated mean, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9479 to 0.9709, shown 0.9479 to 0.9709 | reported |
| `control` | `ho_sparse_treat_ey1__logit_t_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, treated mean, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9611 to 0.9806, shown 0.9611 to 0.9806 | reported |
| `control` | `ho_sparse_treat_ey1__normal_reference_control` | diagnostic | held-out law H1, a sparse treated arm with no strata, treated mean, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9080 to 0.9389, shown 0.9080 to 0.9389 | reported |
| `control` | `identity_msm_v2_w__kish_control` | diagnostic | the alternating Beta-outcome law, identity working MSM, W coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9536 to 0.9751, shown 0.9536 to 0.9751 | reported |
| `control` | `identity_msm_v2_w__normal_reference_control` | diagnostic | the alternating Beta-outcome law, identity working MSM, W coefficient in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9456 to 0.9691, shown 0.9456 to 0.9691 | reported |
| `control` | `logit_msm_v2_a__capped_t_binary_control` | diagnostic | the alternating binary law, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9541 to 0.9756, shown 0.9541 to 0.9756 | reported |
| `control` | `logit_msm_v2_a__kish_control` | diagnostic | the alternating binary law, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9599 to 0.9798, shown 0.9599 to 0.9798 | reported |
| `control` | `logit_msm_v2_a__normal_reference_control` | diagnostic | the alternating binary law, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9339 to 0.9600, shown 0.9339 to 0.9600 | reported |
| `control` | `lognormal06_ey1__kish_control` | diagnostic | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9289 to 0.9560, shown 0.9289 to 0.9560 | reported |
| `control` | `lognormal06_ey1__normal_reference_control` | diagnostic | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9064 to 0.9376, shown 0.9064 to 0.9376 | reported |
| `control` | `rare_ey0__capped_t_binary_control` | diagnostic | a rare binary outcome, three strata, untreated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9355 to 0.9613, shown 0.9355 to 0.9613 | reported |
| `control` | `rare_ey0__kish_control` | diagnostic | a rare binary outcome, three strata, untreated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9513 to 0.9734, shown 0.9513 to 0.9734 | reported |
| `control` | `rare_ey0__logit_t_control` | diagnostic | a rare binary outcome, three strata, untreated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the range-logit scale at max(df, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9570 to 0.9777, shown 0.9570 to 0.9777 | reported |
| `control` | `rare_ey0__normal_reference_control` | diagnostic | a rare binary outcome, three strata, untreated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9053 to 0.9367, shown 0.9053 to 0.9367 | reported |
| `control` | `student_ate__kish_control` | diagnostic | a continuous outcome with a Student t error on 3 df, effect in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9456 to 0.9691, shown 0.9456 to 0.9691 | reported |
| `control` | `student_ate__normal_reference_control` | diagnostic | a continuous outcome with a Student t error on 3 df, effect in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9278 to 0.9551, shown 0.9278 to 0.9551 | reported |
| `control` | `student_ey1__kish_control` | diagnostic | a continuous outcome with a Student t error on 3 df, treated mean in V = 2, n = 2,000: the same fits read under a Student t interval on the identity scale at max(Kish count, 5) | reported with no verdict; a falsifier may read it | lower leg 0.9490 to 0.9717, shown 0.9490 to 0.9717 | reported |
| `control` | `student_ey1__normal_reference_control` | diagnostic | a continuous outcome with a Student t error on 3 df, treated mean in V = 2, n = 2,000: the same fits read under the normal interval on the identity scale | reported with no verdict; a falsifier may read it | lower leg 0.9135 to 0.9435, shown 0.9135 to 0.9435 | reported |
| `corrected_coverage` | `boundary_mean` | positive | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98, and the exact 99.5% lower bound of the paired gain over the normal interval above 1/2 | lower leg 0.9372 to 0.9626, shown 0.9372 to 0.9626, paired gain 110 against 55, lower bound 0.5653 | pass |
| `corrected_coverage` | `boundary_mean_n8000` | positive | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 8,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9383 to 0.9635, shown 0.9383 to 0.9635 | pass |
| `corrected_coverage` | `identity_msm_v2_w` | positive | the alternating Beta-outcome law, identity working MSM, W coefficient in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9507 to 0.9730, shown 0.9507 to 0.9730 | pass |
| `corrected_coverage` | `logit_msm_v2_a` | positive | the alternating binary law, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9339 to 0.9600, shown 0.9339 to 0.9600 | pass |
| `diagnosis_bounded` | `rare_ey0` | positive | a rare binary outcome, three strata, untreated mean in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9389 to 0.9639, shown 0.9389 to 0.9639 | pass |
| `diagnosis_continuous` | `lognormal06_ey1` | positive | a continuous outcome with a lognormal error of skewness 2.0, treated mean in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9178 to 0.9471, shown 0.9178 to 0.9471 | **fail** |
| `diagnosis_continuous` | `student_ate` | positive | a continuous outcome with a Student t error on 3 df, effect in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9378 to 0.9630, shown 0.9378 to 0.9630 | pass |
| `diagnosis_continuous` | `student_ey1` | positive | a continuous outcome with a Student t error on 3 df, treated mean in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98, and the exact 99.5% lower bound of the paired gain over the normal interval above 1/2 | lower leg 0.9339 to 0.9600, shown 0.9339 to 0.9600, paired gain 37 against 0, lower bound 0.8666 | pass |
| `heldout_coverage` | `ho_hurdle_ate` | positive | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, effect in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9339 to 0.9600, shown 0.9349 to 0.9608 | pass |
| `heldout_coverage` | `ho_hurdle_ey1` | positive | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, treated mean in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9135 to 0.9435, shown 0.9145 to 0.9443 | **fail** |
| `heldout_coverage` | `ho_logit_msm_a` | positive | held-out law H3, logistic working MSM (1, a, W), treatment coefficient in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9195 to 0.9484, shown 0.9195 to 0.9484 | **fail** |
| `heldout_coverage` | `ho_logit_msm_w` | positive | held-out law H3, logistic working MSM (1, a, W), W coefficient in V = 2, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.8638 to 0.9013, shown 0.8638 to 0.9013 | **fail** |
| `heldout_coverage` | `ho_sparse_treat_ate` | positive | held-out law H1, a sparse treated arm with no strata, effect, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9124 to 0.9426, shown 0.9124 to 0.9426 | **fail** |
| `heldout_coverage` | `ho_sparse_treat_ey1` | positive | held-out law H1, a sparse treated arm with no strata, treated mean, n = 2,000: the rule's interval | the lower leg's 99% lower end at least 0.92, lost and withheld fits as misses, and the shown fits' 99% upper end at most 0.98 | lower leg 0.9468 to 0.9700, shown 0.9468 to 0.9700 | pass |
| `non_h1` | `nonh1_glm_g_cf_ey0` | positive | law L of `drtmle-one-inconsistent-nuisance`, a wrong outcome regression and a cross-fitted logistic treatment model, mean under control, n = 4,000: the rule's interval | the lower leg's 99% lower end at least 0.92 | coverage 0.9462 to 0.9696 | pass |
| `non_h1` | `nonh1_glm_g_cf_ey1` | positive | law L of `drtmle-one-inconsistent-nuisance`, a wrong outcome regression and a cross-fitted logistic treatment model, mean under treatment, n = 4,000: the rule's interval | the lower leg's 99% lower end at least 0.92 | coverage 0.9490 to 0.9717 | pass |
| `non_h1` | `nonh1_slow_g_control` | control | law L of `drtmle-one-inconsistent-nuisance`, a wrong outcome regression and a treatment model that converges slower than root n, effect, n = 4,000: the rule's interval | the coverage's 99% upper end below 0.90 | coverage 0.7602 to 0.8269 | pass |
| `reported` | `boundary_mean_n1000_ate` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, effect in V = 2, n = 1,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9233 to 0.9515, shown 0.9322 to 0.9587 | reported |
| `reported` | `boundary_mean_n1000_ey1` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 1,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9439 to 0.9678, shown 0.9532 to 0.9749 | reported |
| `reported` | `boundary_mean_n500_ate` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, effect in V = 2, n = 500: the rule's interval | reported with no verdict: both legs | lower leg 0.8532 to 0.8920, shown 0.9531 to 0.9758 | reported |
| `reported` | `boundary_mean_n500_ey1` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE, treated mean in V = 2, n = 500: the rule's interval | reported with no verdict: both legs | lower leg 0.8384 to 0.8789, shown 0.9356 to 0.9625 | reported |
| `reported` | `composite_high` | diagnostic | the three-arm law of `canonical-composite-drtmle`, mean under the high arm, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9400 to 0.9648, shown 0.9400 to 0.9648 | reported |
| `reported` | `drtmle_v2_ate` | diagnostic | the stratified law of `canonical-stratified-drtmle`, DR-TMLE, effect in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9217 to 0.9502, shown 0.9217 to 0.9502 | reported |
| `reported` | `drtmle_v2_ey1` | diagnostic | the stratified law of `canonical-stratified-drtmle`, DR-TMLE, treated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9167 to 0.9462, shown 0.9167 to 0.9462 | reported |
| `reported` | `drtmle_v2_ey1_foldwise` | diagnostic | the stratified law of `canonical-stratified-drtmle`, DR-TMLE with fold-wise reduced regressions, treated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9151 to 0.9448, shown 0.9151 to 0.9448 | reported |
| `reported` | `ho_hurdle_ey0` | diagnostic | held-out law H2, a continuous outcome with a point mass at 0 and a gamma part, untreated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9233 to 0.9515, shown 0.9244 to 0.9524 | reported |
| `reported` | `ho_logit_msm_intercept` | diagnostic | held-out law H3, logistic working MSM (1, a, W), intercept in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.8858 to 0.9202, shown 0.8858 to 0.9202 | reported |
| `reported` | `ho_sparse_treat_ey0` | diagnostic | held-out law H1, a sparse treated arm with no strata, untreated mean, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9372 to 0.9626, shown 0.9372 to 0.9626 | reported |
| `reported` | `identity_msm_v2_a` | diagnostic | the alternating Beta-outcome law, identity working MSM, treatment coefficient in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9366 to 0.9622, shown 0.9366 to 0.9622 | reported |
| `reported` | `leverage_crossfit_ate` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE with a random-forest outcome model, cross-fitted, effect in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs, and the SE-ratio interval containing 1 | lower leg 0.9249 to 0.9627, shown 0.9249 to 0.9627 | reported |
| `reported` | `leverage_crossfit_ey1` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE with a random-forest outcome model, cross-fitted, treated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs, and the SE-ratio interval containing 1 | lower leg 0.9342 to 0.9694, shown 0.9342 to 0.9694 | reported |
| `reported` | `leverage_in_sample_ate` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE with a random-forest outcome model, in sample, effect in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs, and the SE-ratio interval containing 1 | lower leg 0.9237 to 0.9619, shown 0.9237 to 0.9619 | reported |
| `reported` | `leverage_in_sample_ey1` | diagnostic | the stratified law of `canonical-stratified-tmle`, stratified TMLE with a random-forest outcome model, in sample, treated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs, and the SE-ratio interval containing 1 | lower leg 0.9295 to 0.9661, shown 0.9295 to 0.9661 | reported |
| `reported` | `lognormal_ey1` | diagnostic | a continuous outcome with a lognormal error of skewness 6.2, treated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.8804 to 0.9156, shown 0.8804 to 0.9156 | reported |
| `reported` | `mar_multi_arm_ate` | diagnostic | the three-arm law with missing outcomes of `canonical-multi-arm-mar-drtmle`, low against high, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9451 to 0.9687, shown 0.9451 to 0.9687 | reported |
| `reported` | `rare_ate` | diagnostic | a rare binary outcome, three strata, effect in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9162 to 0.9457, shown 0.9355 to 0.9614 | reported |
| `reported` | `rare_ey1` | diagnostic | a rare binary outcome, three strata, treated mean in V = 2, n = 2,000: the rule's interval | reported with no verdict: both legs | lower leg 0.9322 to 0.9586, shown 0.9521 to 0.9742 | reported |
| `witness` | `constant_cell_witness` | diagnostic | the fits of boundary_mean_n500_ey1 whose treated V = 2 cell holds one outcome value: the plug-in interval on the range-logit scale, the point itself on a bound | reported: its coverage's 99% upper end below 0.50 shows why the interval is withheld | coverage 0 to 0.0274 | reported |
<!-- /generated -->

## Result

Both primary rows pass. The rule's interval covers `ey[1][V=2]` in 0.954 of fits against 0.917
for the shipped interval of the same fits, and `ate[V=2]` in 0.957 against 0.955. Twelve of the
17 gated cells pass and five are red. The study publishes under `reporting`, so each red cell has
an owner in the [roadmap](../../roadmap.md#red-cell-owners) and a row in the
[red-cell ledger](red-cells.md). A post-run review diagnosed each red cell on fresh draws
per law, committed in
[`tests/diagnostics/fix3_postrun/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/fix3_postrun)
and rebuilt by `tests/unit/test_fix3_postrun_reading.py`. It found no defect in the estimator or
the design, so no re-run follows.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 2 of 2 | `performance-tests.csv` |
| gated property cells passed | 12 of 17 | `properties.csv` |
| failed fits | 2 of 2,000 in group `ho_hurdle` (an empty design cell) | `property-failures.csv` |

**Bounded means (supported).** `boundary_mean` covers 0.951 at n = 2,000 and 0.952 at 8,000, and
its paired gain over the normal interval is 110 against 55. The rare-outcome mean covers 0.9525,
and the held-out treated mean of H1 covers 0.9595. Of the 191 fits of `boundary_mean_n500_ey1`
whose treated stratum holds one outcome value, the plug-in interval covers none, which is why the
rule withholds the interval there.

| owner | cells | reading |
| --- | --- | --- |
| `X8-logit-small-stratum` | `heldout_coverage/ho_logit_msm_a`, `heldout_coverage/ho_logit_msm_w` | the logistic working-model coefficients of H3's stratum 2 cover 0.935 and 0.8835 under the normal reference, with SE ratios of 0.941 and 0.889 and coverage of 0.949 and 0.9495 at the empirical SD. A post-run diagnosis on fresh draws ([`tests/diagnostics/fix3_postrun/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/fix3_postrun)) found the fitted curve tracking H3's exact efficient influence function at n = 400,000, with root-n SE within 1% of the efficiency SD. The SE ratios of 0.945 and 0.891 at n = 2,000 rise to 1.025 and 0.989 at n = 8,000. The reading is the O(1/n) SE shortfall this owner already holds on the alternating law |
| `F3-normal-reference` (formerly `F3-difference-near-bound`) | `heldout_coverage/ho_sparse_treat_ate` | H1's difference covers 0.9285 with an SE ratio of 0.985 and coverage of 0.956 at the empirical SD. Its interval lies above the truth in 0.0555 of fits and below it in 0.016, and the shipped normal interval on the same fits covers 0.926. On fresh draws ([`tests/diagnostics/fix3_postrun/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/fix3_postrun)) the treated arm, whose mean sits near 1, carries a median 0.907 of the difference's variance. The rule's premise that a difference escapes the coupling of estimate and SE near a bound is refuted |
| `F3-normal-reference` (formerly `F3-skewed-continuous`) | `heldout_coverage/ho_hurdle_ey1`, `diagnosis_continuous/lognormal06_ey1` | the moment-df t covers 0.9304 on H2, over the completed fits, and 0.9335 on the lognormal law of skewness 2.0, with coverage of 0.947 and 0.9545 at the empirical SD. The misses are one-sided (0.055 against 0.015 and 0.055 against 0.0115), which no symmetric df corrects. On H2 a second part adds to the skewness: the saturated outcome model's smallest stratum-2 cell holds about 6.5 rows, and its leverage shrinks the residuals. On fresh draws ([`tests/diagnostics/fix3_postrun/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/fix3_postrun)) a hand-built curve matches the fit's, and dividing each residual by the cell's leverage moves the SE ratio of `ey[1][V=2]` from 0.934 to 0.966 |

Three declared falsifiers fired, each on a red cell: the composition falsifier on
`ho_sparse_treat_ate`, the df falsifier on `ho_hurdle_ey1` and the binary-coefficient falsifier on
`ho_logit_msm_a`. A falsifier fires only when its control passes on the same fits, so a passing
control there is selected and not out-of-sample evidence for it. No falsifier fired on a bounded
mean. Every reported row keeps its `passed` reading in `properties.csv`, and the diagnostic rows at
n = 500, the DR-TMLE stratum, the lognormal law of skewness 6.2 and `rare_ate` read as the
declaration expected.

**Cells this study does not close.** The owner `strata-boundary-mean` keeps the band cell of
`canonical-stratified-tmle`. Its two primary truth rows and its pointwise cell moved to
`F3-normal-reference` when the shipped default took the rule for bounded means. They stay as the
record of the identity interval, beside an audit file that recomputes them under the shipped rule.
This study's three refuted cells moved to `F3-normal-reference` in the same way when the shipped
default took correlated MOVER for bounded differences and the skew-widened t for continuous
targets, which [asymmetric-wald](asymmetric-wald-intervals.md) measured on fresh seeds of the
same laws. They stay as the record of the rule this study refuted.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 2 | parameter tests against the truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 17 | gated repeated-sampling property cells |
| `property_cells_passed` | 12 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0192 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9540 | lowest measured primary coverage, the rule's interval |
| `min_coverage_ci_lower` | 0.9342 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `properties[corrected_coverage/boundary_mean]:coverage` | 0.9510 | owner cell, n = 2,000 |
| `properties[corrected_coverage/boundary_mean]:gain_lower` | 0.5653 | its paired gain's lower bound, against 1/2 |
| `properties[corrected_coverage/boundary_mean_n8000]:coverage` | 0.9520 | owner cell, n = 8,000 |
| `properties[diagnosis_bounded/rare_ey0]:coverage` | 0.9525 | rare-outcome mean |
| `properties[heldout_coverage/ho_sparse_treat_ey1]:coverage` | 0.9595 | held-out treated mean |
| `properties[heldout_coverage/ho_sparse_treat_ate]:coverage` | 0.9285 | held-out difference near a bound |
| `properties[heldout_coverage/ho_sparse_treat_ate]:se_ratio` | 0.9846 | its SE ratio |
| `properties[heldout_coverage/ho_hurdle_ey1]:coverage` | 0.9304 | held-out skewed continuous mean, over the completed fits |
| `properties[diagnosis_continuous/lognormal06_ey1]:coverage` | 0.9335 | lognormal mean, skewness 2.0 |
| `properties[heldout_coverage/ho_logit_msm_a]:coverage` | 0.9350 | held-out logit-MSM treatment coefficient |
| `properties[heldout_coverage/ho_logit_msm_w]:coverage` | 0.8835 | held-out logit-MSM W coefficient |
| `properties[heldout_coverage/ho_logit_msm_w]:se_ratio` | 0.8893 | its SE ratio |
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
| `margin:subject_failures_allowed` | 10 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The rule measured before the default read it | The study measured the rule computed from public functions. The default reads it for bounded means only |
| Bounded differences | Refuted when one arm sits near its bound and carries most of the difference's variance. A bounded difference now takes correlated MOVER, measured by [asymmetric-wald](asymmetric-wald-intervals.md), and this study's committed row stays as the record of the refuted rule (`F3-normal-reference`) |
| Skewed continuous outcomes | Refuted at moderate skewness in a small stratum, and on H2 also by a saturated learner's leverage on cells of about 6.5 rows. A continuous target now takes the skew-widened t, measured by [asymmetric-wald](asymmetric-wald-intervals.md), and this study's committed rows stay as the record of the refuted rule (`F3-normal-reference`) |
| Binary working-model coefficients | Keep the normal reference. In a stratum of about 300 rows their SE falls short at O(1/n) (`X8-logit-small-stratum`), and no t reference repairs it |
| Withheld intervals | At n = 500 the rule withholds about a tenth of the intervals of the boundary mean, each counted as a miss. The interval shown on the other fits covers, and a withheld interval is a claim limit, not a coverage |
| Held-out laws now seen | H1 to H3 have been looked at. A revised rule chosen because it wins on them is in-sample, so it needs its own declaration on new held-out laws |

## Reproduction

The declared run is `python tests/canonical/small_sample_wald/regenerate.py --output <empty
directory>`, whose docstring gives the run form, the declaration and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/small_sample_wald/manifest.json)
records the seeds, the margins, the configuration, the source hashes and the result hashes. The
[run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/small_sample_wald/run.log)
records the commit, the clean tree and the wall time. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/small_sample_wald/properties.csv),
the [falsifiers](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/small_sample_wald/falsifiers.csv)
and the [shipped interval of the primary fits](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/small_sample_wald/primary-shipped.csv)
carry every published row.
